import streamlit as st
import time

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Metrics and Status Displays Tutorial",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Metrics & Status Displays — Complete Master Guide")
st.markdown(
    """
Effective dashboards communicate high-level system health and quantitative performance at a glance. 
Streamlit provides powerful KPI widgets like **`st.metric()`** (with trend arrows, sparklines, and card borders), 
tag badges with **`st.badge()`**, expandable multi-stage pipeline workflows with **`st.status()`**, 
progress tracking with **`st.progress()`**, and full-spectrum callout alert banners.
"""
)

st.divider()

# ==============================================================================
# 1. st.metric() Core Architecture & 2. Labels and Values
# ==============================================================================
st.header("1. Core KPI Metrics & 2. Labels and Values (`st.metric`)", divider="rainbow")
st.markdown(
    """
`st.metric()` renders a large, prominent scorecard card designed for key performance indicators (KPIs):
- **`label`**: The title or metric name (supports markdown and icons).
- **`value`**: The primary quantitative measurement (string, integer, or float).
- **`border=True`**: Surrounds the metric in a sleek, bordered card container.
- **`help`**: Hover tooltip explaining how the metric is calculated.
"""
)

st.markdown("#### 📐 Formal Signature: `st.metric()`")
st.code(
    """st.metric(
    label: str,
    value: int | float | str | None,
    delta: int | float | str | None = None,
    delta_color: "normal" | "inverse" | "off" = "normal",
    *,
    help: str | None = None,
    border: bool = False,
    chart_data: list | np.ndarray | None = None,  # Inline sparkline
    chart_type: "line" | "bar" | "area" = "line"
) -> DeltaGenerator""",
    language="python",
)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("**💻 Python Source Code**")
    code_metric_basic = '''# KPI metric cards with border=True:
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric(
        label="⚡ Token Throughput",
        value="1,420 tok/s",
        border=True,
        help="Average tokens generated across the inference fleet"
    )

with col_kpi2:
    st.metric(
        label="🎯 Model Accuracy (MMLU)",
        value="88.7%",
        border=True
    )

with col_kpi3:
    st.metric(
        label="💰 Monthly Token Spend",
        value="$12,450",
        border=True
    )'''
    st.code(code_metric_basic, language="python")

with col_m2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(
            label="⚡ Token Throughput",
            value="1,420 tok/s",
            border=True,
            help="Average tokens generated across the inference fleet"
        )
    with col_kpi2:
        st.metric(
            label="🎯 Model Accuracy (MMLU)",
            value="88.7%",
            border=True
        )
    with col_kpi3:
        st.metric(
            label="💰 Monthly Token Spend",
            value="$12,450",
            border=True
        )

st.divider()

# ==============================================================================
# 3. Delta Values & 4. Positive and Negative Trends
# ==============================================================================
st.header("3. Delta Indicators & 4. Positive vs. Negative Trends", divider="rainbow")
st.markdown(
    """
The `delta` parameter displays comparative change against a historical baseline:
- **`delta_color="normal"` (Default)**: Positive delta is **Green ⬆** (growth is good); Negative is **Red ⬇** (decline is bad).
- **`delta_color="inverse"`**: Positive delta is **Red ⬆** (cost/latency increase is bad); Negative is **Green ⬇** (cost/latency reduction is good!).
- **`delta_color="off"`**: Renders delta in neutral gray without directional colored semantics.
"""
)

col_del1, col_del2 = st.columns(2)

with col_del1:
    st.markdown("**💻 Python Source Code (Delta Modes)**")
    code_deltas = '''c1, c2, c3 = st.columns(3)

# 1. Normal mode: Higher throughput is good (Green +)
with c1:
    st.metric(
        label="Requests / Second",
        value="4,850 rps",
        delta="+350 rps (7.8%)",
        delta_color="normal",
        border=True
    )

# 2. Inverse mode: Lower p99 latency is good (Green -)
with c2:
    st.metric(
        label="p99 Latency (TTFT)",
        value="128 ms",
        delta="-42 ms (-24.7%)",
        delta_color="inverse",
        border=True
    )

# 3. Inverse mode: Higher compute cost is bad (Red +)
with c3:
    st.metric(
        label="Cluster Cloud Cost",
        value="$3,820",
        delta="+$420 (+12.3%)",
        delta_color="inverse",
        border=True
    )'''
    st.code(code_deltas, language="python")

with col_del2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Requests / Second",
            value="4,850 rps",
            delta="+350 rps (7.8%)",
            delta_color="normal",
            border=True
        )
    with c2:
        st.metric(
            label="p99 Latency (TTFT)",
            value="128 ms",
            delta="-42 ms (-24.7%)",
            delta_color="inverse",
            border=True
        )
    with c3:
        st.metric(
            label="Cluster Cloud Cost",
            value="$3,820",
            delta="+$420 (+12.3%)",
            delta_color="inverse",
            border=True
        )

st.divider()

# ==============================================================================
# 5. Colored Badges (st.badge)
# ==============================================================================
st.header("5. Colored Badges (`st.badge`)", divider="rainbow")
st.markdown(
    """
`st.badge()` renders compact pill badges with custom labels, icons, and colors:
- **Supported Colors**: `'red'`, `'orange'`, `'yellow'`, `'blue'`, `'green'`, `'violet'`, `'gray'`, `'primary'`.
- **Markdown Directive**: Badges can also be inserted inline into any Markdown text using `:color-badge[Label]`.
- **Applications**: Service status indicators, deployment environment tags, release channels (`Beta`, `Prod`, `Staging`).
"""
)

col_bdg1, col_bdg2 = st.columns(2)

with col_bdg1:
    st.markdown("**💻 Python Source Code**")
    code_badge = '''# 1. Native st.badge widget calls:
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.badge("Production", icon="🚀", color="green")
with col_b2:
    st.badge("High Latency", icon="⚠️", color="orange")
with col_b3:
    st.badge("Service Outage", icon="🛑", color="red")
with col_b4:
    st.badge("Experimental", icon="🧪", color="violet")

# 2. Markdown inline color badges:
st.markdown("Cluster **A100-Fleet-01** is :green-badge[● Healthy] running version :blue-badge[v2.4.1].")'''
    st.code(code_badge, language="python")

with col_bdg2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.badge("Production", icon="🚀", color="green")
    with col_b2:
        st.badge("High Latency", icon="⚠️", color="orange")
    with col_b3:
        st.badge("Service Outage", icon="🛑", color="red")
    with col_b4:
        st.badge("Experimental", icon="🧪", color="violet")

    st.markdown("Cluster **A100-Fleet-01** is :green-badge[● Healthy] running version :blue-badge[v2.4.1].")

st.divider()

# ==============================================================================
# 6. st.status() — Multi-Stage Process Containers
# ==============================================================================
st.header("6. Multi-Stage Process Container (`st.status`)", divider="rainbow")
st.markdown(
    """
`st.status()` creates an expandable task container for multi-step AI or ETL pipelines:
- **Lifecycle States**: Begins in `state="running"`, then transitions via `.update(state="complete")` or `.update(state="error")`.
- **Live Output**: Inside the `with st.status(...)` context, print intermediate logs or step messages.
- **Collapsible**: Automatically collapses upon completion to keep the dashboard clean.
"""
)

col_st1, col_st2 = st.columns(2)

with col_st1:
    st.markdown("**💻 Python Source Code**")
    code_status = '''# Multi-stage model fine-tuning workflow:
with st.status("Initializing Distributed Training...", expanded=True) as status:
    st.write("1. Allocating GPU cluster nodes...")
    # time.sleep(1)
    st.write("2. Ingesting tokenized training dataset...")
    # time.sleep(1)
    st.write("3. Compiling model graph with PyTorch Inductor...")
    # time.sleep(1)
    status.update(
        label="Distributed Fine-Tuning Pipeline Complete!",
        state="complete",
        expanded=False
    )'''
    st.code(code_status, language="python")

with col_st2:
    st.markdown("**🖥️ Live Interactive Multi-Stage Status Container**")
    
    if st.button("▶️ Trigger Training Pipeline Demo", key="btn_run_status"):
        with st.status("Executing LLM Fine-Tuning Pipeline...", expanded=True) as status_demo:
            st.write("🔹 Stage 1: Initializing CUDA 12.4 memory pool...")
            st.write("🔹 Stage 2: Ingesting 50,000 instruction-tuning pairs...")
            st.write("🔹 Stage 3: Running AdamW backpropagation across 8x H100s...")
            status_demo.update(
                label="✅ Fine-Tuning Pipeline Completed Successfully!",
                state="complete",
                expanded=True
            )
    else:
        st.caption("Click the button above to simulate a multi-stage status workflow.")

st.divider()

# ==============================================================================
# 7. st.progress() & 8. st.spinner()
# ==============================================================================
st.header("7. Progress Bars (`st.progress`) & 8. Spinners (`st.spinner`)", divider="rainbow")
st.markdown(
    """
- **`st.progress(value, text)`**: Renders a deterministic progress bar (from `0` to `100` or `0.0` to `1.0`) with animated filling.
- **`st.spinner(text)`**: A context manager that displays an indeterminate circular loading spinner while executing blocking I/O or model inference.
"""
)

col_pr_sp1, col_pr_sp2 = st.columns(2)

with col_pr_sp1:
    st.markdown("**💻 Python Source Code**")
    code_progress_spinner = '''# 1. Deterministic Progress Bar:
bar = st.progress(0, text="Embedding knowledge base chunks...")
for percent in range(0, 101, 25):
    bar.progress(percent, text=f"Embedding chunks: {percent}%")

# 2. Indeterminate Loading Spinner:
with st.spinner("Generating LLM Response with Claude-3.5..."):
    # blocking_call()
    pass'''
    st.code(code_progress_spinner, language="python")

with col_pr_sp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    
    st.progress(75, text="📊 Evaluation Benchmark Progress: 75% Complete")
    
    if st.button("Run 1-Second Spinner Demo", key="btn_run_spinner"):
        with st.spinner("Generating Synthetic Benchmark Prompts..."):
            time.sleep(1)
        st.success("Prompt generation finished!")
    else:
        st.caption("Click above to view the live spinner animation.")

st.divider()

# ==============================================================================
# 9. st.success(), 10. st.info(), st.warning(), and st.error()
# ==============================================================================
st.header("9. Success & 10. Alert Banners (`st.info`, `st.warning`, `st.error`)", divider="rainbow")
st.markdown(
    """
Streamlit provides four distinct callout alert containers styled according to severity:
- **`st.success()`**: Green banner indicating successful completion or passing validations.
- **`st.info()`**: Blue banner for informational notices, guidance, or metadata.
- **`st.warning()`**: Amber/yellow banner warning about deprecations, high costs, or resource limits.
- **`st.error()`**: Red banner displaying exceptions, validation failures, or critical errors.
All banners accept custom emojis or Material icons via the `icon` parameter.
"""
)

col_alt1, col_alt2 = st.columns(2)

with col_alt1:
    st.markdown("**💻 Python Source Code (Callout Banners)**")
    code_banners = '''st.success("Model weights checkpoint saved to S3 bucket!", icon="💾")
st.info("New model version v3.2 is available for deployment.", icon="ℹ️")
st.warning("GPU memory usage exceeded 90% threshold!", icon="⚠️")
st.error("Connection timeout: Inference endpoint unreachable.", icon="🚨")'''
    st.code(code_banners, language="python")

with col_alt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.success("Model weights checkpoint saved to S3 bucket!", icon="💾")
    st.info("New model version v3.2 is available for deployment.", icon="ℹ️")
    st.warning("GPU memory usage exceeded 90% threshold!", icon="⚠️")
    st.error("Connection timeout: Inference endpoint unreachable.", icon="🚨")

st.divider()

# ==============================================================================
# Master AI Operations Executive Dashboard
# ==============================================================================
st.header("🚀 Comprehensive Real-World AI Operations Dashboard", divider="rainbow")
st.markdown(
    """
Combining metrics with `border=True`, badges, progress indicators, and alert banners 
into a cohesive production monitoring layout:
"""
)

# Top KPI row with bordered metrics
row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)

with row1_c1:
    st.metric(
        label="Fleet Availability",
        value="99.98%",
        delta="+0.04%",
        delta_color="normal",
        border=True
    )

with row1_c2:
    st.metric(
        label="Inference Latency (p95)",
        value="84 ms",
        delta="-18 ms",
        delta_color="inverse",
        border=True
    )

with row1_c3:
    st.metric(
        label="Active Serving Nodes",
        value="64 / 64",
        delta="0 node churn",
        delta_color="off",
        border=True
    )

with row1_c4:
    st.metric(
        label="Cost Per 1K Queries",
        value="$0.42",
        delta="-$0.08",
        delta_color="inverse",
        border=True
    )

# Middle operational status row
mid_c1, mid_c2 = st.columns([2, 1])

with mid_c1:
    st.markdown("##### 🖥️ Distributed Cluster Health Status")
    st.markdown(
        """
        - Region `us-east-1` (Virginia): :green-badge[● Operational] (32 GPUs active)
        - Region `eu-west-1` (Dublin): :green-badge[● Operational] (16 GPUs active)
        - Region `ap-northeast-1` (Tokyo): :orange-badge[● Maintenance] (16 GPUs draining)
        """
    )
    st.progress(92, text="Fleet Resource Utilization: 92% of Peak VRAM")

with mid_c2:
    st.markdown("##### 🛡️ Production Policy Guardrails")
    st.info("Guardrail Filter v2.1 Active: Toxic prompt rejection rate at 0.04%.", icon="🛡️")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Metrics & Status Displays Reference", divider="rainbow")
st.markdown(
    """
| Primitive | Primary Syntax | Behavior & Use Case |
| :--- | :--- | :--- |
| **`st.metric`** | `st.metric(label, value, delta, border=True)` | Prominent KPI tiles with directional delta arrows and card borders |
| **`delta_color="normal"`** | `delta_color="normal"` | Green for positive growth, Red for decline (Accuracy, Throughput) |
| **`delta_color="inverse"`** | `delta_color="inverse"` | Green for decreases, Red for increases (Latency, Error Rate, Cost) |
| **`st.badge`** | `st.badge("Text", icon="🚀", color="green")` | Pill tags for deployment states, environments, and categories |
| **`st.status`** | `with st.status("Task...", expanded=False) as s:` | Collapsible multi-step pipeline status container |
| **`st.progress`** | `st.progress(pct, text="Label")` | Deterministic progress bar for quantitative steps |
| **`st.spinner`** | `with st.spinner("Loading..."):` | Indeterminate circular loading animation for background tasks |
| **`st.success`** | `st.success("Message", icon="✅")` | Green confirmation callout banner |
| **`st.info / warning / error`**| `st.info()`, `st.warning()`, `st.error()` | Severity-graded communication and alert banners |
"""
)
