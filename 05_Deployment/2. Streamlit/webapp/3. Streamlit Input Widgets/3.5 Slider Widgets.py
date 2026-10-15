import streamlit as st
import pandas as pd
from datetime import date, time, timedelta

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Slider Widgets Tutorial",
    page_icon="🎚️",
    layout="wide",
)

st.title("🎚️ Slider Widgets (`st.slider` & `st.select_slider`) — Complete Guide")
st.markdown(
    """
Sliders provide intuitive, continuous tactile controls for selecting numeric values, 
two-sided range intervals, dates, times, and ordered categorical scales. 
This master guide covers syntax, integer/float formatting, dual-thumb range sliders, temporal datetime intervals, 
and dynamic dataset filtering.
"""
)

st.divider()

# ==============================================================================
# 1. st.slider() Core Architecture
# ==============================================================================
st.header("1. Core API & Slider Architecture", divider="rainbow")
st.markdown(
    """
`st.slider()` renders an interactive track with draggable thumb handles. 
Streamlit dynamically determines the slider type (integer, float, date, time, or range) 
based on the data types passed to `value`, `min_value`, and `max_value`.
"""
)

st.markdown("#### 📐 Formal Signature: `st.slider()`")
st.code(
    """st.slider(
    label: str,
    min_value: int | float | date | time | None = None,
    max_value: int | float | date | time | None = None,
    value: T | tuple[T, T] | None = None,
    step: int | float | timedelta | None = None,
    format: str | None = None,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    width: "stretch" | "content" = "stretch"
) -> T | tuple[T, T]""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.slider()`")
st.markdown(
    """
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`label`** | `str` | *Required* | Label displayed above the slider track. |
| **`min_value`** | `T` | Default lower bound | Minimum allowable position on the track. |
| **`max_value`** | `T` | Default upper bound | Maximum allowable position on the track. |
| **`value`** | `T` or `(T, T)` | `min_value` | Initial thumb position. Passing a 2-tuple activates a **dual-thumb range slider**. |
| **`step`** | `T` or `timedelta` | Inferred | Step increment when dragging or using keyboard arrows. |
| **`format`** | `str` | `None` | Printf format (numbers) or strftime format (dates/times). |
| **`width`** | `str` | `"stretch"` | Controls container width (`"stretch"` or `"content"`). |
"""
)

st.divider()

# ==============================================================================
# 2. Integer Sliders
# ==============================================================================
st.header("2. Discrete Integer Sliders", divider="rainbow")
st.markdown(
    """
When all bounds are integers, the slider enforces discrete whole-number steps:
- Ideal for epoch counts, batch sizes, worker counts, and layer depths.
- Supports printf format strings like `format="%d epochs"`.
"""
)

col_int1, col_int2 = st.columns(2)

with col_int1:
    st.markdown("**💻 Python Source Code**")
    code_int = '''# Discrete integer slider for model training epochs:
epochs = st.slider(
    label="Maximum Training Epochs",
    min_value=1,
    max_value=100,
    value=25,
    step=1,
    format="%d epochs",
    help="Total training passes over the dataset"
)

st.write(f"Configured Epochs: **{epochs}**")'''
    st.code(code_int, language="python")

with col_int2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    epochs_val = st.slider(
        label="Maximum Training Epochs",
        min_value=1,
        max_value=100,
        value=25,
        step=1,
        format="%d epochs",
        help="Total training passes over the dataset",
        key="slider_demo_epochs"
    )
    st.caption(f"Selected Value: **{epochs_val}** (Type: `{type(epochs_val).__name__}`)")

st.divider()

# ==============================================================================
# 3. Float Sliders & Precision Formatting
# ==============================================================================
st.header("3. Floating-Point Sliders & Precision Formatting", divider="rainbow")
st.markdown(
    """
Passing floating-point numbers enables fractional continuum adjustments:
- Essential for hyperparameters like **Sampling Temperature**, **Top-P Probability**, and **Dropout Rate**.
- Use **`format="%.2f"`** or **`format="%.3f"`** to control decimal display on the track.
"""
)

col_fl1, col_fl2 = st.columns(2)

with col_fl1:
    st.markdown("**💻 Python Source Code**")
    code_float = '''# Floating-point hyperparameter sliders:
temperature = st.slider(
    label="LLM Temperature (Creativity vs Determinism)",
    min_value=0.0,
    max_value=2.0,
    value=0.70,
    step=0.05,
    format="%.2f"
)

top_p = st.slider(
    label="Top-P (Nucleus Sampling Probability)",
    min_value=0.01,
    max_value=1.00,
    value=0.95,
    step=0.01,
    format="%.2f"
)'''
    st.code(code_float, language="python")

with col_fl2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    temp_val = st.slider(
        label="LLM Temperature (Creativity vs Determinism)",
        min_value=0.0,
        max_value=2.0,
        value=0.70,
        step=0.05,
        format="%.2f",
        key="slider_demo_temp"
    )
    top_p_val = st.slider(
        label="Top-P (Nucleus Sampling Probability)",
        min_value=0.01,
        max_value=1.00,
        value=0.95,
        step=0.01,
        format="%.2f",
        key="slider_demo_top_p"
    )
    st.caption(f"Sampling Spec: Temperature = **{temp_val:.2f}** | Top-P = **{top_p_val:.2f}**")

st.divider()

# ==============================================================================
# 4. Range Sliders (Dual-Thumb Interval Selection)
# ==============================================================================
st.header("4. Dual-Thumb Range Sliders", divider="rainbow")
st.markdown(
    """
Passing a **2-tuple `(low, high)`** to the `value` parameter activates a **dual-thumb range slider**:
- Renders two independently draggable handles on the same track.
- Returns a Python `tuple[T, T]` representing `(start_bound, end_bound)`.
- Perfect for filtering confidence intervals, price bands, and parameter size brackets.
"""
)

col_rng1, col_rng2 = st.columns(2)

with col_rng1:
    st.markdown("**💻 Python Source Code**")
    code_range = '''# Dual-thumb range slider:
price_range = st.slider(
    label="Compute Hardware Hourly Budget ($/hr)",
    min_value=0.0,
    max_value=50.0,
    value=(5.0, 25.0),  # 2-tuple activates range slider!
    step=1.0,
    format="$%.2f"
)

st.write(f"Min Budget: ${price_range[0]:.2f} | Max Budget: ${price_range[1]:.2f}")'''
    st.code(code_range, language="python")

with col_rng2:
    st.markdown("**🖥️ Rendered Range Slider**")
    price_range = st.slider(
        label="Compute Hardware Hourly Budget ($/hr)",
        min_value=0.0,
        max_value=50.0,
        value=(5.0, 25.0),
        step=1.0,
        format="$%.2f",
        key="slider_demo_price_rng"
    )
    min_price, max_price = price_range
    st.caption(f"Selected Window: **${min_price:.2f}** to **${max_price:.2f}** (Delta: **${max_price - min_price:.2f}**)")

st.divider()

# ==============================================================================
# 5. Date Sliders & 6. Time Sliders
# ==============================================================================
st.header("5. Date Sliders & 6. Time Sliders", divider="rainbow")
st.markdown(
    """
Passing `datetime.date` or `datetime.time` objects transforms the widget into a temporal slider:
- **Date Slider**: Bound by `date(YYYY, M, D)`. Accepts a tuple `(start_date, end_date)` for date intervals.
- **Time Slider**: Bound by `time(HH, MM)`. Control time steps with `datetime.timedelta(minutes=15)`.
- **Date Formats**: Custom formatting using standard tokens like `format="YYYY/MM/DD"` or `format="MM/DD/YY"`.
"""
)

col_dt1, col_dt2 = st.columns(2)

with col_dt1:
    st.markdown("**💻 Python Source Code (Date & Time Sliders)**")
    code_temporal = '''from datetime import date, time, timedelta

# 1. Date Interval Range Slider:
date_window = st.slider(
    label="Telemetry Log Audit Window",
    min_value=date(2026, 1, 1),
    max_value=date(2026, 12, 31),
    value=(date(2026, 3, 1), date(2026, 9, 30)),
    format="YYYY/MM/DD"
)

# 2. Time of Day Slider with 30-minute steps:
meeting_time = st.slider(
    label="Automated Pipeline Run Time",
    min_value=time(0, 0),
    max_value=time(23, 45),
    value=time(9, 30),
    step=timedelta(minutes=30),
    format="HH:mm"
)'''
    st.code(code_temporal, language="python")

with col_dt2:
    st.markdown("**🖥️ Rendered Temporal Sliders**")
    date_window = st.slider(
        label="Telemetry Log Audit Window",
        min_value=date(2026, 1, 1),
        max_value=date(2026, 12, 31),
        value=(date(2026, 3, 1), date(2026, 9, 30)),
        format="YYYY/MM/DD",
        key="slider_demo_dates"
    )
    meeting_time = st.slider(
        label="Automated Pipeline Run Time",
        min_value=time(0, 0),
        max_value=time(23, 45),
        value=time(9, 30),
        step=timedelta(minutes=30),
        format="HH:mm",
        key="slider_demo_time"
    )
    st.caption(f"Dates: **{date_window[0]}** to **{date_window[1]}** | Daily Run: **{meeting_time.strftime('%H:%M UTC')}**")

st.divider()

# ==============================================================================
# 7. Step Configuration & st.select_slider()
# ==============================================================================
st.header("7. Step Configuration & Ordered `st.select_slider`", divider="rainbow")
st.markdown(
    """
For non-numeric or non-linear ordered scales, use **`st.select_slider()`**:
- Maps slider positions to an arbitrary ordered sequence of categories.
- Works with categorical labels (e.g. `["XS", "S", "M", "L", "XL"]` or `["Nano", "Small", "Medium", "Large"]`).
- Also supports dual-thumb range selection by passing a 2-tuple default!
"""
)

col_ss1, col_ss2 = st.columns(2)

with col_ss1:
    st.markdown("**💻 Python Source Code (`st.select_slider`)**")
    code_select_slider = '''model_sizes = ["7B (Edge)", "13B (Workstation)", "34B (Server)", "70B (Cluster)", "405B (Supercomputer)"]

selected_size = st.select_slider(
    label="Target Model Parameter Footprint",
    options=model_sizes,
    value="70B (Cluster)",
    help="Select hardware tier capacity"
)

st.write("Selected Footprint:", selected_size)'''
    st.code(code_select_slider, language="python")

with col_ss2:
    st.markdown("**🖥️ Rendered Categorical Select Slider**")
    model_sizes = ["7B (Edge)", "13B (Workstation)", "34B (Server)", "70B (Cluster)", "405B (Supercomputer)"]
    selected_size = st.select_slider(
        label="Target Model Parameter Footprint",
        options=model_sizes,
        value="70B (Cluster)",
        help="Select hardware tier capacity",
        key="select_slider_demo"
    )
    st.caption(f"Hardware Allocation: **{selected_size}**")

st.divider()

# ==============================================================================
# 8. Real-World Use Case: Using Sliders for Interactive Data Filtering
# ==============================================================================
st.header("8. Real-World Application: Live Dataset Filtering with Sliders", divider="rainbow")
st.markdown(
    """
Sliders shine when used as **interactive analytical filters** over datasets. 
Below, dual-range and threshold sliders dynamically slice an AI Model Benchmark leaderboard in real time:
"""
)

# Dataset
raw_models_data = pd.DataFrame({
    "Model Name": [
        "Llama-3.1-8B", "Gemma-2-9B", "Mistral-7B", "Qwen-2.5-14B", 
        "Llama-3.3-70B", "DeepSeek-V3", "Claude-3.5-Haiku", "GPT-4o-Mini", 
        "Claude-3.5-Sonnet", "GPT-4o"
    ],
    "Parameter Size (B)": [8, 9, 7, 14, 70, 671, 20, 8, 175, 200],
    "MMLU Score (%)": [68.4, 71.3, 64.2, 79.5, 88.5, 88.5, 81.0, 82.0, 94.0, 92.0],
    "Inference Latency (ms)": [45, 52, 38, 74, 185, 290, 85, 60, 210, 180],
    "Cost / 1M Tokens ($)": [0.20, 0.25, 0.15, 0.40, 0.70, 0.27, 1.00, 0.15, 3.00, 5.00]
})

col_filt_code, col_filt_live = st.columns([1, 1])

with col_filt_code:
    st.markdown("**💻 Python Source Code (Multi-Slider Filtering)**")
    code_filter_pipeline = '''# 1. Sliders for multi-dimensional criteria:
param_filter = st.slider("Parameter Range (B)", 1, 700, (5, 100), step=5)
min_mmlu = st.slider("Minimum MMLU Score (%)", 50.0, 95.0, 70.0, step=1.0)
max_latency = st.slider("Max Latency Ceiling (ms)", 30, 350, 200, step=10)

# 2. Vectorized DataFrame slicing:
mask = (
    (df["Parameter Size (B)"] >= param_filter[0]) &
    (df["Parameter Size (B)"] <= param_filter[1]) &
    (df["MMLU Score (%)"] >= min_mmlu) &
    (df["Inference Latency (ms)"] <= max_latency)
)

filtered_leaderboard = df[mask]
st.dataframe(filtered_leaderboard, width="stretch", hide_index=True)'''
    st.code(code_filter_pipeline, language="python")

with col_filt_live:
    st.markdown("**🖥️ Live Interactive Multi-Slider Leaderboard**")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        param_filt = st.slider("Parameter Range (B)", 1, 700, (5, 100), step=5, key="filter_param_range")
        min_mmlu_val = st.slider("Minimum MMLU Score (%)", 50.0, 95.0, 70.0, step=1.0, format="%.1f%%", key="filter_min_mmlu")
    with col_s2:
        max_lat_val = st.slider("Max Latency Ceiling (ms)", 30, 350, 200, step=10, key="filter_max_latency")

    # Vectorized filter condition
    filter_mask = (
        (raw_models_data["Parameter Size (B)"] >= param_filt[0]) &
        (raw_models_data["Parameter Size (B)"] <= param_filt[1]) &
        (raw_models_data["MMLU Score (%)"] >= min_mmlu_val) &
        (raw_models_data["Inference Latency (ms)"] <= max_lat_val)
    )
    matched_models = raw_models_data[filter_mask]

    st.markdown(f"##### Filtered Matches: **{len(matched_models)} / {len(raw_models_data)}** models")
    st.dataframe(matched_models, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Slider Widgets Cheat Sheet", divider="rainbow")
st.markdown(
    """
| Scenario | Recommended Function & Configuration | Example Code |
| :--- | :--- | :--- |
| **Integer Count** | Integer bounds, `step=1`, `format="%d"` | `st.slider("Epochs", 1, 50, 10, step=1)` |
| **Float Precision** | Float bounds, `step=0.01`, `format="%.2f"` | `st.slider("Temp", 0.0, 1.0, 0.7, step=0.05)` |
| **Two-Sided Interval** | Pass 2-tuple `value=(low, high)` | `st.slider("Price", 0, 100, (10, 50))` |
| **Date Interval** | `date()` objects, `format="YYYY/MM/DD"` | `st.slider("Period", date(2026,1,1), date(2026,12,31))` |
| **Time of Day** | `time()` objects, `step=timedelta(minutes=15)` | `st.slider("Time", time(8,0), time(18,0))` |
| **Non-Numeric Scale** | `st.select_slider()` with list of strings | `st.select_slider("Size", options=["S", "M", "L"])` |
"""
)
