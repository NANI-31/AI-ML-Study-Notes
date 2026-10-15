import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Numeric Input Widgets Tutorial",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 Numeric Input Widgets (`st.number_input`) — Complete Guide")
st.markdown(
    """
`st.number_input()` provides a type-safe input field for numeric values with built-in increment/decrement 
spinners, range clamping, precision formatting, and scientific notation.
This master reference covers bounds, step inference, integer vs. floating-point type promotion, 
and real-time computational pipeline integration.
"""
)

st.divider()

# ==============================================================================
# 1. st.number_input() Core API & Mechanics
# ==============================================================================
st.header("1. Core API & Spinner Controls", divider="rainbow")
st.markdown(
    """
`st.number_input()` renders an HTML `<input type="number">` accompanied by increment/decrement step buttons (`+` and `-`).
Users can adjust values using:
1. Direct keyboard text entry.
2. Clicking the `+` / `-` buttons.
3. Using the **Up / Down arrow keys** on their keyboard.
"""
)

st.markdown("#### 📐 Formal Signature: `st.number_input()`")
st.code(
    """st.number_input(
    label: str,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    value: int | float | "min" | None = "min",
    step: int | float | None = None,
    format: str | None = None,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    placeholder: str | None = None,
    disabled: bool = False,
    required: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    icon: str | None = None,
    width: "stretch" | "content" = "stretch"
) -> int | float | None""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.number_input()`")
st.markdown(
    """
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`label`** | `str` | *Required* | Label displayed above the input field. |
| **`min_value`** | `int`, `float`, `None` | `None` | Minimum allowed bound. Clamps input on commit. |
| **`max_value`** | `int`, `float`, `None` | `None` | Maximum allowed bound. Clamps input on commit. |
| **`value`** | `int`, `float`, `"min"`, `None` | `"min"` | Initial pre-populated value. Defaults to `min_value` or `0`/`0.0`. Set `None` for a blank input. |
| **`step`** | `int`, `float`, `None` | `None` | Increment/decrement delta. Defaults to `1` for ints and `0.01` for floats. |
| **`format`** | `str` | `None` | Printf-style string format (e.g. `"%d"`, `"%.2f"`, `"%.4f"`, `"%.1e"`). |
| **`help`** | `str` | `None` | Tooltip icon explaining expected bounds. |
| **`disabled`** | `bool` | `False` | Disables user interaction. |
"""
)

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("**💻 Python Source Code**")
    code_core = '''# Basic integer input:
worker_threads = st.number_input(
    label="Worker Threads",
    min_value=1,
    max_value=64,
    value=8,
    step=1,
    help="Number of parallel execution workers"
)

st.write(f"Allocated Threads: **{worker_threads}**")'''
    st.code(code_core, language="python")

with col_c2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    worker_threads = st.number_input(
        label="Worker Threads",
        min_value=1,
        max_value=64,
        value=8,
        step=1,
        help="Number of parallel execution workers",
        key="demo_worker_threads"
    )
    st.caption(f"Allocated Threads: **{worker_threads}** (Type: `{type(worker_threads).__name__}`)")

st.divider()

# ==============================================================================
# 2. Minimum and Maximum Bounds (Clamping & Validation)
# ==============================================================================
st.header("2. Minimum and Maximum Values (`min_value`, `max_value`)", divider="rainbow")
st.markdown(
    """
Setting explicit bounds restricts input within a safe operational range:
- **Two-Sided Interval**: `min_value=0`, `max_value=100` (e.g. percentage scales).
- **One-Sided Lower Bound**: `min_value=1`, `max_value=None` (e.g. batch size must be positive).
- **Out-of-Bounds Behavior**: If a user types `150` when `max_value=100`, Streamlit displays an alert 
  and clamps the input upon form commit.
"""
)

col_mm1, col_mm2 = st.columns(2)

with col_mm1:
    st.markdown("**💻 Python Source Code**")
    code_minmax = '''# Bounded percentage interval:
split_ratio = st.number_input(
    label="Train Partition Split (%)",
    min_value=50.0,
    max_value=95.0,
    value=80.0,
    step=2.5,
    help="Bounded between 50.0% and 95.0%"
)

test_ratio = 100.0 - split_ratio
st.write(f"Train: {split_ratio:.1f}% | Test: {test_ratio:.1f}%")'''
    st.code(code_minmax, language="python")

with col_mm2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    split_ratio = st.number_input(
        label="Train Partition Split (%)",
        min_value=50.0,
        max_value=95.0,
        value=80.0,
        step=2.5,
        help="Bounded between 50.0% and 95.0%",
        key="demo_split_ratio"
    )
    test_ratio = 100.0 - split_ratio
    st.caption(f"Partition: **{split_ratio:.1f}% Train** / **{test_ratio:.1f}% Test**")

st.divider()

# ==============================================================================
# 3. Step Sizes & Automatic Step Inference
# ==============================================================================
st.header("3. Step Sizes (`step`)", divider="rainbow")
st.markdown(
    """
The `step` parameter controls the increment/decrement magnitude when clicking the `+` or `-` buttons:
- **Tensor Dimensions & GPU Batch Sizes**: Step in powers of 2 (e.g. `step=8` or `step=16`).
- **Precision Learning Rates**: Step by small fractions (e.g. `step=0.0005`).
- **Step Inference Rule**:
  - If all inputs (`value`, `min_value`, `max_value`) are integers $\\rightarrow$ default step is `1`.
  - If any input is a float $\\rightarrow$ default step is `0.01`.
"""
)

col_step1, col_step2 = st.columns(2)

with col_step1:
    st.markdown("**💻 Python Source Code (Custom GPU Batch Step)**")
    code_step = '''# Custom step of 8 for GPU alignment:
batch_size = st.number_input(
    label="DataLoader Batch Size (Multiples of 8)",
    min_value=8,
    max_value=256,
    value=32,
    step=8,
    help="Hardware aligned tensor batch size"
)

st.write(f"Configured Batch Size: {batch_size}")'''
    st.code(code_step, language="python")

with col_step2:
    st.markdown("**🖥️ Rendered Step Size Demo**")
    batch_size = st.number_input(
        label="DataLoader Batch Size (Multiples of 8)",
        min_value=8,
        max_value=256,
        value=32,
        step=8,
        help="Hardware aligned tensor batch size",
        key="demo_batch_step"
    )
    st.caption(f"Clicking `+` or `-` jumps by **8**: Active = **{batch_size}**")

st.divider()

# ==============================================================================
# 4. Default Numeric Values (value="min", value=None)
# ==============================================================================
st.header("4. Default Numeric Values & Optional Empty Inputs", divider="rainbow")
st.markdown(
    """
Streamlit provides flexible initialization behaviors for the `value` parameter:
- **Explicit Default**: `value=42` pre-fills 42.
- **Default to Bound**: `value="min"` automatically sets the default to `min_value`.
- **Blank Optional Input**: `value=None` allows the field to render completely empty! 
  The widget returns `None` until the user provides a number.
"""
)

col_def1, col_def2 = st.columns(2)

with col_def1:
    st.markdown("**💻 Python Source Code**")
    code_defaults = '''# A. Default to min_value automatically:
port_number = st.number_input("Server Port", min_value=1024, max_value=65535, value="min")

# B. Optional blank input (Returns None if empty):
optional_budget = st.number_input("Max GPU Budget ($)", min_value=0.0, value=None, placeholder="Optional budget limit...")

st.write("Port:", port_number)
st.write("Budget Return:", repr(optional_budget))'''
    st.code(code_defaults, language="python")

with col_def2:
    st.markdown("**🖥️ Rendered Output**")
    port_number = st.number_input("Server Port", min_value=1024, max_value=65535, value="min", key="demo_port_min")
    optional_budget = st.number_input("Max GPU Budget ($)", min_value=0.0, value=None, placeholder="Optional budget limit...", key="demo_budget_none")

    st.caption(f"Port: **{port_number}** | Budget: **{repr(optional_budget)}** (Is None: `{optional_budget is None}`)")

st.divider()

# ==============================================================================
# 5. Integer Inputs (int & format="%d")
# ==============================================================================
st.header("5. Strict Integer Inputs", divider="rainbow")
st.markdown(
    """
When all parameters (`min_value`, `max_value`, `value`, `step`) are integers, Streamlit guarantees 
that the return value is of type **`int`**:
- Format string `%d` renders standard whole numbers.
- Ideal for counts, random seeds, epoch counts, and layer depths.
"""
)

col_int1, col_int2 = st.columns(2)

with col_int1:
    st.markdown("**💻 Python Source Code**")
    code_int = '''# Strict integer random seed:
random_seed = st.number_input(
    label="Reproducibility Random Seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
    format="%d"
)

st.write(f"Type: `{type(random_seed).__name__}` | Value: {random_seed}")'''
    st.code(code_int, language="python")

with col_int2:
    st.markdown("**🖥️ Rendered Integer Input**")
    random_seed = st.number_input(
        label="Reproducibility Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        format="%d",
        key="demo_seed_int"
    )
    st.caption(f"Python Type: `{type(random_seed).__name__}` | Seed Value: **{random_seed}**")

st.divider()

# ==============================================================================
# 6. Floating-Point Inputs & Precision Formatting
# ==============================================================================
st.header("6. Floating-Point Inputs & Precision Formatting", divider="rainbow")
st.markdown(
    """
If **any** parameter passed to `st.number_input` is a float, Streamlit promotes the return value to **`float`**.
Control decimal precision and notation using standard format strings:
- **`%.2f`**: Two decimal places (e.g. Currency `$12.50`, Confidence `0.95`).
- **`%.4f`**: High precision (e.g. Temperature `0.7000`, Weight decay `0.0001`).
- **`%.1e` or `%.2e`**: Scientific exponential notation (e.g. Learning rates `1.0e-04`).
"""
)

col_fl1, col_fl2 = st.columns(2)

with col_fl1:
    st.markdown("**💻 Python Source Code**")
    code_float = '''# 1. Decimal currency format (%.2f)
hourly_cost = st.number_input("GPU Rental Cost ($/hr)", min_value=0.0, value=3.25, step=0.25, format="%.2f")

# 2. High precision decimal (%.4f)
temp_param = st.number_input("LLM Sampling Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05, format="%.4f")

# 3. Scientific notation (%.1e)
learning_rate = st.number_input("AdamW Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.1e")'''
    st.code(code_float, language="python")

with col_fl2:
    st.markdown("**🖥️ Rendered Precision Formats**")
    hourly_cost = st.number_input("GPU Rental Cost ($/hr)", min_value=0.0, value=3.25, step=0.25, format="%.2f", key="demo_cost_fl")
    temp_param = st.number_input("LLM Sampling Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05, format="%.4f", key="demo_temp_fl")
    learning_rate = st.number_input("AdamW Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.1e", key="demo_lr_sci")

    st.caption(f"Cost: **${hourly_cost:.2f}** | Temp: **{temp_param:.4f}** | LR: **{learning_rate:.1e}**")

st.divider()

# ==============================================================================
# 7. Validating Numeric Ranges & Edge Cases
# ==============================================================================
st.header("7. Validating Numeric Ranges & Consistency", divider="rainbow")
st.markdown(
    """
> [!CAUTION]
> **Type Consistency Rule**  
> In Streamlit, all bounds (`min_value`, `max_value`, `value`, `step`) must share the **same data type**.  
> Mixing integer `min_value=0` with floating-point `value=0.5` without matching types or format strings will raise a `StreamlitAPIException`!
"""
)

col_ed1, col_ed2 = st.columns(2)

with col_ed1:
    st.markdown("**💻 Python Source Code (Range Validation)**")
    code_edge = '''# Dynamic sanity check on input values:
num_clusters = st.number_input("K-Means Cluster Count (k)", min_value=1, max_value=50, value=3, step=1)
sample_count = 100

if num_clusters > sample_count:
    st.error("❌ Cluster count (k) cannot exceed the dataset sample size!")
elif num_clusters == 1:
    st.warning("⚠️ A single cluster will not partition your data.")
else:
    st.success(f"✅ Valid configuration: {num_clusters} clusters across {sample_count} samples.")'''
    st.code(code_edge, language="python")

with col_ed2:
    st.markdown("**🖥️ Rendered Range Validation**")
    num_clusters = st.number_input("K-Means Cluster Count (k)", min_value=1, max_value=50, value=3, step=1, key="demo_k_means")
    sample_count = 100

    if num_clusters > sample_count:
        st.error("❌ Cluster count (k) cannot exceed the dataset sample size!")
    elif num_clusters == 1:
        st.warning("⚠️ A single cluster will not partition your data.")
    else:
        st.success(f"✅ Valid configuration: **{num_clusters}** clusters across {sample_count} samples.")

st.divider()

# ==============================================================================
# 8. Real-World Calculation: LLM Training Cost & VRAM Estimator
# ==============================================================================
st.header("8. Real-World Application: LLM Cluster Training Cost Estimator", divider="rainbow")
st.markdown(
    """
This production calculation demonstrates combining multiple integer and floating-point inputs 
to compute **GPU VRAM requirements, training duration, and total dollar cost** reactively in real time.
"""
)

col_calc_code, col_calc_live = st.columns([1, 1])

with col_calc_code:
    st.markdown("**💻 Python Source Code**")
    code_estimator = '''# Model Architecture Inputs:
param_billions = st.number_input("Model Size (Billions of Params)", min_value=0.5, max_value=175.0, value=7.0, step=0.5, format="%.1f")
tokens_billions = st.number_input("Training Tokens (Billions)", min_value=10.0, max_value=5000.0, value=200.0, step=50.0, format="%.1f")
num_gpus = st.number_input("Cluster GPU Node Count", min_value=1, max_value=512, value=8, step=8)
gpu_rate_hourly = st.number_input("GPU Rental Cost ($/hr per GPU)", min_value=0.5, value=2.40, step=0.20, format="%.2f")

# Calculations:
# FLOPs = 6 * Parameters * Tokens
flops = 6 * (param_billions * 1e9) * (tokens_billions * 1e9)
h100_tflops = 989e12  # BF16 TFLOPs per H100 GPU
gpu_hours = flops / (h100_tflops * 0.40 * 3600)  # 40% MFU efficiency
wall_clock_days = (gpu_hours / num_gpus) / 24
total_cost = gpu_hours * gpu_rate_hourly

# Render Output Metrics:
st.metric("Total Projected Cost", f"${total_cost:,.2f}")
st.metric("Wall-Clock Time", f"{wall_clock_days:.1f} Days")'''
    st.code(code_estimator, language="python")

with col_calc_live:
    st.markdown("**🖥️ Live Interactive Calculator**")
    
    calc_c1, calc_c2 = st.columns(2)
    with calc_c1:
        p_billions = st.number_input("Model Size (Billion Params)", min_value=0.5, max_value=175.0, value=7.0, step=0.5, format="%.1f", key="p_billions")
        tok_billions = st.number_input("Training Tokens (Billions)", min_value=10.0, max_value=5000.0, value=200.0, step=50.0, format="%.1f", key="tok_billions")
    with calc_c2:
        gpus = st.number_input("Cluster GPU Node Count", min_value=1, max_value=512, value=8, step=8, key="gpus_count")
        gpu_rate = st.number_input("GPU Rental Cost ($/hr/GPU)", min_value=0.5, value=2.40, step=0.20, format="%.2f", key="gpu_rate")

    # Formula calculations
    total_flops = 6 * (p_billions * 1e9) * (tok_billions * 1e9)
    gpu_perf_eff = 989e12 * 0.40 * 3600  # Effective throughput per GPU-hour
    total_gpu_hours = total_flops / gpu_perf_eff
    wall_clock_days = (total_gpu_hours / gpus) / 24
    total_dollar_cost = total_gpu_hours * gpu_rate

    st.markdown("##### 📊 Real-Time Financial & Operational Projection")
    m1, m2, m3 = st.columns(3)
    m1.metric("Projected Total Cost", f"${total_dollar_cost:,.2f}", border=True)
    m2.metric("Wall-Clock Training Time", f"{wall_clock_days:.1f} Days", border=True)
    m3.metric("Total GPU Hours", f"{total_gpu_hours:,.0f} hrs", border=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Numeric Input Cheat Sheet", divider="rainbow")
st.markdown(
    """
| Scenario | Recommended Configuration | Example Code |
| :--- | :--- | :--- |
| **Integer Count** | Integer bounds, `step=1`, `format="%d"` | `st.number_input("Epochs", min_value=1, value=10, step=1, format="%d")` |
| **Currency** | Float bounds, `step=0.25`, `format="%.2f"` | `st.number_input("Price", min_value=0.0, value=9.99, format="%.2f")` |
| **Scientific Notation** | Float bounds, `format="%.1e"` | `st.number_input("LR", min_value=1e-5, value=1e-3, format="%.1e")` |
| **Batch Powers-of-2** | Step size aligned to tensor dimension | `st.number_input("Batch", min_value=8, value=32, step=8)` |
| **Optional Empty** | `value=None` | `st.number_input("Age", min_value=0, value=None)` |
"""
)
