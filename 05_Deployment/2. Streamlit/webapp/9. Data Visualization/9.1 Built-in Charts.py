import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Built-in Charts Tutorial",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Streamlit Built-in Charts — Complete Guide")
st.markdown(
    """
Streamlit provides high-performance, native chart primitives (`st.line_chart`, `st.area_chart`, 
`st.bar_chart`, `st.scatter_chart`, and `st.map`). Built atop **Altair** and **Vega-Lite**, 
these native methods require zero boilerplate, handle automatic axis inference, 
support pan/zoom interactivity, and adapt seamlessly to light and dark themes.
"""
)

st.divider()

# ==============================================================================
# 1. st.line_chart()
# ==============================================================================
st.header("1. Line Charts (`st.line_chart`)", divider="rainbow")
st.markdown(
    r"""
`st.line_chart()` displays continuous trends over epochs, steps, or time intervals.

```python
st.line_chart(
    data=None, *, x=None, y=None, x_label=None, y_label=None,
    color=None, width="stretch", height=None, use_container_width=None
)
```

- **`data`**: Pandas DataFrame, Series, NumPy array, or iterable.
- **`x`**: Column name for the horizontal X-axis. Defaults to the DataFrame index.
- **`y`**: Column name (or list of column names) for the vertical Y-axis values.
- **`color`**: Column name to group lines by color, or a single color hex code.
- **`x_label` / `y_label`**: Custom axis titles displayed directly on the chart.
"""
)

col_ln1, col_ln2 = st.columns(2)

with col_ln1:
    st.markdown("**💻 Python Source Code (Multi-Series Loss Curves)**")
    code_line = '''import streamlit as st
import pandas as pd
import numpy as np

# Generate training & validation loss history
epochs = np.arange(1, 26)
train_loss = 2.5 * np.exp(-epochs / 6.0) + np.random.normal(0, 0.02, len(epochs))
val_loss = 2.6 * np.exp(-epochs / 7.0) + 0.15 + np.random.normal(0, 0.03, len(epochs))

loss_df = pd.DataFrame({
    "Epoch": epochs,
    "Training Loss": np.clip(train_loss, 0.05, None),
    "Validation Loss": np.clip(val_loss, 0.18, None),
})

st.line_chart(
    loss_df,
    x="Epoch",
    y=["Training Loss", "Validation Loss"],
    color=["#3b82f6", "#ef4444"],
    x_label="Training Epoch",
    y_label="Cross-Entropy Loss",
    height=320,
)'''
    st.code(code_line, language="python")

with col_ln2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(42)
    epochs = np.arange(1, 26)
    train_loss = 2.5 * np.exp(-epochs / 6.0) + np.random.normal(0, 0.02, len(epochs))
    val_loss = 2.6 * np.exp(-epochs / 7.0) + 0.15 + np.random.normal(0, 0.03, len(epochs))

    loss_df = pd.DataFrame({
        "Epoch": epochs,
        "Training Loss": np.clip(train_loss, 0.05, None),
        "Validation Loss": np.clip(val_loss, 0.18, None),
    })

    st.line_chart(
        loss_df,
        x="Epoch",
        y=["Training Loss", "Validation Loss"],
        color=["#3b82f6", "#ef4444"],
        x_label="Training Epoch",
        y_label="Cross-Entropy Loss",
        height=320,
    )
    st.caption("Interactive dual-series line plot with custom epoch axis and hex color mapping.")

st.divider()

# ==============================================================================
# 2. st.area_chart()
# ==============================================================================
st.header("2. Area Charts (`st.area_chart`)", divider="rainbow")
st.markdown(
    r"""
`st.area_chart()` renders stacked area plots ideal for showing **cumulative proportions** 
or volume distributions over time, such as token consumption across multiple LLM models.

```python
st.area_chart(
    data=None, *, x=None, y=None, x_label=None, y_label=None,
    color=None, stack=True, width="stretch", height=None
)
```
"""
)

col_ar1, col_ar2 = st.columns(2)

with col_ar1:
    st.markdown("**💻 Python Source Code (Token Throughput Volume)**")
    code_area = '''# 7-day token consumption (millions of tokens)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
token_df = pd.DataFrame({
    "Day": days,
    "Claude 3.7": [12.4, 15.8, 18.2, 19.5, 22.1, 14.2, 11.0],
    "GPT-4.5":    [10.1, 12.3, 14.0, 15.2, 16.5, 11.8,  9.5],
    "DeepSeek R1": [8.5, 11.2, 13.6, 14.1, 18.0, 15.3, 13.8],
})

st.area_chart(
    token_df,
    x="Day",
    y=["Claude 3.7", "GPT-4.5", "DeepSeek R1"],
    x_label="Day of Week",
    y_label="Tokens Generated (Millions)",
    color=["#8b5cf6", "#10b981", "#f59e0b"],
    height=320,
)'''
    st.code(code_area, language="python")

with col_ar2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    token_df = pd.DataFrame({
        "Day": days,
        "Claude 3.7": [12.4, 15.8, 18.2, 19.5, 22.1, 14.2, 11.0],
        "GPT-4.5":    [10.1, 12.3, 14.0, 15.2, 16.5, 11.8,  9.5],
        "DeepSeek R1": [8.5, 11.2, 13.6, 14.1, 18.0, 15.3, 13.8],
    })

    st.area_chart(
        token_df,
        x="Day",
        y=["Claude 3.7", "GPT-4.5", "DeepSeek R1"],
        x_label="Day of Week",
        y_label="Tokens Generated (Millions)",
        color=["#8b5cf6", "#10b981", "#f59e0b"],
        height=320,
    )
    st.caption("Stacked area chart displaying daily cumulative inference token usage.")

st.divider()

# ==============================================================================
# 3. st.bar_chart()
# ==============================================================================
st.header("3. Bar Charts (`st.bar_chart`)", divider="rainbow")
st.markdown(
    r"""
`st.bar_chart()` displays discrete category comparisons (benchmarks, hardware utilization, latency).
- Supports **multi-series grouped bars** or **stacked bars** via the `horizontal=False|True` and `stack` options.
"""
)

col_br1, col_br2 = st.columns(2)

with col_br1:
    st.markdown("**💻 Python Source Code (LLM Benchmark Comparison)**")
    code_bar = '''bench_df = pd.DataFrame({
    "Benchmark": ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench"],
    "Claude 3.7": [89.4, 92.0, 96.2, 78.4, 70.3],
    "GPT-4.5":    [88.7, 90.2, 95.1, 75.8, 65.2],
    "DeepSeek R1": [90.8, 93.6, 97.0, 84.2, 68.9],
})

st.bar_chart(
    bench_df,
    x="Benchmark",
    y=["Claude 3.7", "GPT-4.5", "DeepSeek R1"],
    color=["#3b82f6", "#10b981", "#f97316"],
    x_label="Standard AI Benchmark",
    y_label="Accuracy Pass Rate (%)",
    height=320,
)'''
    st.code(code_bar, language="python")

with col_br2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    bench_df = pd.DataFrame({
        "Benchmark": ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench"],
        "Claude 3.7": [89.4, 92.0, 96.2, 78.4, 70.3],
        "GPT-4.5":    [88.7, 90.2, 95.1, 75.8, 65.2],
        "DeepSeek R1": [90.8, 93.6, 97.0, 84.2, 68.9],
    })

    st.bar_chart(
        bench_df,
        x="Benchmark",
        y=["Claude 3.7", "GPT-4.5", "DeepSeek R1"],
        color=["#3b82f6", "#10b981", "#f97316"],
        x_label="Standard AI Benchmark",
        y_label="Accuracy Pass Rate (%)",
        height=320,
    )
    st.caption("Clustered categorical bar chart comparing top frontier LLM reasoning scores.")

st.divider()

# ==============================================================================
# 4. st.scatter_chart()
# ==============================================================================
st.header("4. Scatter Charts (`st.scatter_chart`)", divider="rainbow")
st.markdown(
    r"""
`st.scatter_chart()` visualizes relationships between continuous features:
- **`size`**: Map a numeric column to circle diameter (e.g. Model Parameters in Billions).
- **`color`**: Categorical or continuous variable to color individual scatter points.
"""
)

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**💻 Python Source Code (Latency vs. Accuracy Pareto Frontier)**")
    code_scatter = '''scatter_df = pd.DataFrame({
    "Model": ["Claude 3.7", "GPT-4.5", "Gemini 2.5", "Llama 3.3 70B", "Qwen 2.5 72B", "Mistral Large", "Gemma 2 27B"],
    "Latency_ms": [380, 420, 310, 240, 260, 290, 180],
    "Score":      [92.0, 90.2, 89.5, 82.1, 84.3, 83.0, 75.2],
    "Params_B":   [180, 220, 150, 70, 72, 123, 27],
    "Provider":   ["Anthropic", "OpenAI", "Google", "Meta", "Alibaba", "Mistral", "Google"]
})

st.scatter_chart(
    scatter_df,
    x="Latency_ms",
    y="Score",
    size="Params_B",
    color="Provider",
    x_label="Time-to-First-Token Latency (ms)",
    y_label="Coding Pass@1 Score (%)",
    height=340,
)'''
    st.code(code_scatter, language="python")

with col_sc2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    scatter_df = pd.DataFrame({
        "Model": ["Claude 3.7", "GPT-4.5", "Gemini 2.5", "Llama 3.3 70B", "Qwen 2.5 72B", "Mistral Large", "Gemma 2 27B"],
        "Latency_ms": [380, 420, 310, 240, 260, 290, 180],
        "Score":      [92.0, 90.2, 89.5, 82.1, 84.3, 83.0, 75.2],
        "Params_B":   [180, 220, 150, 70, 72, 123, 27],
        "Provider":   ["Anthropic", "OpenAI", "Google", "Meta", "Alibaba", "Mistral", "Google"]
    })

    st.scatter_chart(
        scatter_df,
        x="Latency_ms",
        y="Score",
        size="Params_B",
        color="Provider",
        x_label="Time-to-First-Token Latency (ms)",
        y_label="Coding Pass@1 Score (%)",
        height=340,
    )
    st.caption("Point size indicates model parameter count; colors distinguish AI providers.")

st.divider()

# ==============================================================================
# 5. st.map()
# ==============================================================================
st.header("5. Geographic Maps (`st.map`)", divider="rainbow")
st.markdown(
    r"""
`st.map()` renders an interactive 2D Mapbox/OpenStreetMap visualization.
- **Latitude / Longitude**: Expects columns named `lat`/`latitude` and `lon`/`longitude`.
- **`size`**: Controls geographic point circle radius.
- **`color`**: Hex color code or column name for color encoding.
- **`zoom`**: Initial map camera zoom level.
"""
)

col_mp1, col_mp2 = st.columns(2)

with col_mp1:
    st.markdown("**💻 Python Source Code (Global AI GPU Data Centers)**")
    code_map = '''gpu_clusters = pd.DataFrame({
    "lat": [37.7749, 47.6062, 52.5200, 35.6762, 1.3521, 51.5074, 39.0437],
    "lon": [-122.4194, -122.3321, 13.4050, 139.6503, 103.8198, -0.1278, -77.4875],
    "H100_GPUs": [4096, 8192, 2048, 4096, 2048, 3072, 16384],
})

st.map(
    gpu_clusters,
    latitude="lat",
    longitude="lon",
    size="H100_GPUs",
    color="#06b6d4",
    zoom=1,
    height=340,
)'''
    st.code(code_map, language="python")

with col_mp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    gpu_clusters = pd.DataFrame({
        "lat": [37.7749, 47.6062, 52.5200, 35.6762, 1.3521, 51.5074, 39.0437],
        "lon": [-122.4194, -122.3321, 13.4050, 139.6503, 103.8198, -0.1278, -77.4875],
        "H100_GPUs": [4096, 8192, 2048, 4096, 2048, 3072, 16384],
    })

    st.map(
        gpu_clusters,
        latitude="lat",
        longitude="lon",
        size="H100_GPUs",
        color="#06b6d4",
        zoom=1,
        height=340,
    )
    st.caption("Global distribution of H100 inference clusters with point radii scaled to GPU count.")

st.divider()

# ==============================================================================
# 6. Plotting Pandas DataFrames (Wide vs. Long Formats)
# ==============================================================================
st.header("6. Plotting Pandas DataFrames (Wide vs. Long Data)", divider="rainbow")
st.markdown(
    r"""
Streamlit handles both DataFrame shapes natively:
- **Wide Format**: Separate columns for each metric (e.g. `['Model_A', 'Model_B']`). Ideal for quick multi-line comparison.
- **Long / Tidy Format**: Categorical column (e.g. `'Experiment'`) with a `'Value'` column. Uses `color='Experiment'`.
"""
)

col_pd1, col_pd2 = st.columns(2)

with col_pd1:
    st.markdown("**💻 Python Source Code (Long vs. Wide Ingestion)**")
    code_pd = '''# Wide Format with DateTime Index:
date_range = pd.date_range("2026-10-01", periods=14, freq="D")
wide_df = pd.DataFrame({
    "API_Calls_US": np.random.randint(1000, 5000, size=14),
    "API_Calls_EU": np.random.randint(800, 3500, size=14),
}, index=date_range)

# Index is automatically used as the X-axis:
st.line_chart(wide_df, height=300)'''
    st.code(code_pd, language="python")

with col_pd2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(101)
    date_range = pd.date_range("2026-10-01", periods=14, freq="D")
    wide_df = pd.DataFrame({
        "API_Calls_US": np.random.randint(1000, 5000, size=14),
        "API_Calls_EU": np.random.randint(800, 3500, size=14),
    }, index=date_range)

    st.line_chart(wide_df, height=300)
    st.caption("Wide DataFrame: Streamlit automatically maps DateTimeIndex to X and each series column to distinct lines.")

st.divider()

# ==============================================================================
# 7. Plotting NumPy Arrays
# ==============================================================================
st.header("7. Plotting NumPy Arrays Directly", divider="rainbow")
st.markdown(
    r"""
You do not need to convert arrays to DataFrames first! You can pass raw 1D or 2D NumPy arrays directly:
- **1D Array `(N,)`**: Single curve with auto-generated integer index `0, 1, ..., N-1`.
- **2D Array `(N, M)`**: `M` distinct series curves across `N` steps.
"""
)

col_np1, col_np2 = st.columns(2)

with col_np1:
    st.markdown("**💻 Python Source Code (Raw Array Plotting)**")
    code_numpy = '''# 2D Array: Shape (50, 3) representing 3 random walk walks:
steps = 50
walks = np.cumsum(np.random.randn(steps, 3), axis=0)

# Pass raw NumPy array directly:
st.line_chart(walks, height=280)'''
    st.code(code_numpy, language="python")

with col_np2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(77)
    steps = 50
    walks = np.cumsum(np.random.randn(steps, 3), axis=0)
    st.line_chart(walks, height=280)
    st.caption("Raw NumPy 2D array `(50, 3)` rendered directly as 3 random walk trajectory paths.")

st.divider()

# ==============================================================================
# 8. Configuring Chart Dimensions & Dynamic Layouts
# ==============================================================================
st.header("8. Configuring Chart Dimensions & Responsiveness", divider="rainbow")
st.markdown(
    r"""
Control chart geometry using dimension parameters:
- **`width="stretch"`**: Chart automatically spans 100% of the parent column or container.
- **`width=int`**: Explicit pixel width (e.g. `width=500`).
- **`height=int`**: Explicit pixel height (e.g. `height=250` or `height=450`).
"""
)

col_dm1, col_dm2 = st.columns(2)

with col_dm1:
    st.markdown("**💻 Python Source Code (Dynamic Dimensions)**")
    code_dim = '''chart_height = st.slider("Select Chart Height (px)", 150, 500, 260, 20)

sample_data = pd.DataFrame({
    "Step": np.arange(1, 21),
    "Gradient Norm": np.abs(np.random.normal(1.0, 0.4, 20))
})

st.bar_chart(
    sample_data,
    x="Step",
    y="Gradient Norm",
    color="#ec4899",
    height=chart_height,
    width="stretch"
)'''
    st.code(code_dim, language="python")

with col_dm2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    chart_height = st.slider("Select Chart Height (px)", 150, 500, 260, 20, key="slider_chart_height")

    np.random.seed(99)
    sample_data = pd.DataFrame({
        "Step": np.arange(1, 21),
        "Gradient Norm": np.abs(np.random.normal(1.0, 0.4, 20)),
    })

    st.bar_chart(
        sample_data,
        x="Step",
        y="Gradient Norm",
        color="#ec4899",
        height=chart_height,
        width="stretch",
    )
    st.caption(f"Chart rendered with dynamic height = {chart_height}px and `width='stretch'`.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Native Streamlit Charts Quick Reference", divider="rainbow")
st.markdown(
    r"""
| Method | Primary Use Case | Key Parameters | Data Input Types |
| :--- | :--- | :--- | :--- |
| **`st.line_chart`** | Continuous metrics, loss curves, time series | `data, x, y, color, x_label, y_label, height` | `DataFrame`, `Series`, `np.ndarray` |
| **`st.area_chart`** | Cumulative volumes, resource shares, tokens | `data, x, y, color, stack, height` | `DataFrame`, `Series`, `np.ndarray` |
| **`st.bar_chart`** | Categorical comparisons, benchmarks, throughput | `data, x, y, color, horizontal, stack, height` | `DataFrame`, `Series`, `np.ndarray` |
| **`st.scatter_chart`** | Multidimensional correlation, Pareto frontiers | `data, x, y, size, color, height` | `DataFrame` with continuous columns |
| **`st.map`** | Geographic coordinate scatter & heat distribution | `data, latitude, longitude, size, color, zoom` | `DataFrame` with lat/lon columns |
"""
)
