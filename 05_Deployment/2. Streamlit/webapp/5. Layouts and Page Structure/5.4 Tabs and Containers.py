import streamlit as st
import pandas as pd
import numpy as np
import time

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Tabs and Containers Tutorial",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Tabs & Dynamic Containers (`st.tabs` & `st.empty`) — Complete Guide")
st.markdown(
    """
Modern enterprise dashboards must present vast amounts of data without overwhelming users. 
**`st.tabs()`** divides dense content into categorized, client-side switchable views with zero rerun latency. 
**`st.container()`** bundles related elements into bordered cards and scrollable panes. 
**`st.empty()`** provides single-slot placeholders for in-place live updates and dynamic animation streams.
"""
)

st.divider()

# ==============================================================================
# 1. st.tabs() Core Architecture & 2. Creating Multiple Tabs
# ==============================================================================
st.header("1. Core Tab Architecture & 2. Creating Multiple Tabs (`st.tabs`)", divider="rainbow")
st.markdown(
    """
`st.tabs(titles)` creates a top navigation bar with tabbed panes:
- **Client-Side Switching**: Switching between tabs occurs instantly in the browser without rerunning the Python script!
- **Context Managers**: Each tab acts as a container (`with tab1:`, `with tab2:`).
- **Tab Titles**: Accepts emoji prefixes, markdown, and concise descriptive labels.
"""
)

st.markdown("#### 📐 Formal Signature: `st.tabs()`")
st.code(
    """st.tabs(
    tabs: Sequence[str],
    *,
    default: str | None = None,
    key: str | int | None = None,
    width: "stretch" = "stretch"
) -> Sequence[TabContainer]""",
    language="python",
)

col_tb1, col_tb2 = st.columns(2)

with col_tb1:
    st.markdown("**💻 Python Source Code (3 Tabs Structure)**")
    code_tabs = '''# Create and unpack tabs:
tab_overview, tab_latency, tab_pricing = st.tabs([
    "📊 Fleet Overview", 
    "⚡ Latency Telemetry", 
    "💰 Cost Breakdown"
])

with tab_overview:
    st.write("Fleet of 64 active H100 GPU nodes serving LLM traffic.")
    st.metric("Total Tokens / Sec", "1,840 tok/s")

with tab_latency:
    st.write("p99 Time-to-First-Token (TTFT) metrics.")
    st.metric("p99 TTFT", "42 ms", "-6 ms", delta_color="inverse")

with tab_pricing:
    st.write("Hourly compute hardware costs.")
    st.metric("Hourly Fleet Burn", "$185.50 / hr")'''
    st.code(code_tabs, language="python")

with col_tb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    tab_overview, tab_latency, tab_pricing = st.tabs([
        "📊 Fleet Overview", 
        "⚡ Latency Telemetry", 
        "💰 Cost Breakdown"
    ])
    with tab_overview:
        st.write("Fleet of 64 active H100 GPU nodes serving LLM traffic.")
        st.metric("Total Tokens / Sec", "1,840 tok/s")
    with tab_latency:
        st.write("p99 Time-to-First-Token (TTFT) metrics.")
        st.metric("p99 TTFT", "42 ms", "-6 ms", delta_color="inverse")
    with tab_pricing:
        st.write("Hourly compute hardware costs.")
        st.metric("Hourly Fleet Burn", "$185.50 / hr")

st.divider()

# ==============================================================================
# 3. Displaying Different Views of the Same Dataset
# ==============================================================================
st.header("3. Displaying Different Views of Data via Tabs", divider="rainbow")
st.markdown(
    """
Tabs are ideal for offering multiple angles on the same underlying data:
- **Tab 1: Summary Table**: Filtered DataFrame with custom column configurations.
- **Tab 2: Metric Scorecard**: High-level KPIs and distributions.
- **Tab 3: Raw Export**: Full JSON records and instant CSV download buttons.
"""
)

eval_df = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Llama-3.3-70B"],
    "MMLU (%)": [88.7, 88.6, 88.5, 86.4],
    "HumanEval (%)": [92.0, 90.2, 89.1, 81.7],
    "Price / 1M Input ($)": [3.00, 2.50, 0.27, 0.70]
})

col_vw1, col_vw2 = st.columns(2)

with col_vw1:
    st.markdown("**💻 Python Source Code (Multi-View Dataset Tabs)**")
    code_multi_view = '''tab_grid, tab_kpi, tab_export = st.tabs([
    "📋 Tabular Grid", 
    "🎯 Metric Scorecards", 
    "💾 Data Export"
])

with tab_grid:
    st.dataframe(eval_df, width="stretch", hide_index=True)

with tab_kpi:
    c1, c2 = st.columns(2)
    c1.metric("Average MMLU", f"{eval_df['MMLU (%)'].mean():.1f}%")
    c2.metric("Median Cost", f"${eval_df['Price / 1M Input ($)'].median():.2f}")

with tab_export:
    csv_bytes = eval_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Leaderboard CSV", csv_bytes, "leaderboard.csv")'''
    st.code(code_multi_view, language="python")

with col_vw2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    tab_grid, tab_kpi, tab_export = st.tabs([
        "📋 Tabular Grid", 
        "🎯 Metric Scorecards", 
        "💾 Data Export"
    ])
    with tab_grid:
        st.dataframe(eval_df, width="stretch", hide_index=True, key="tabs_demo_grid")
    with tab_kpi:
        c1, c2 = st.columns(2)
        c1.metric("Average MMLU", f"{eval_df['MMLU (%)'].mean():.1f}%")
        c2.metric("Median Cost", f"${eval_df['Price / 1M Input ($)'].median():.2f}")
    with tab_export:
        csv_bytes = eval_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Leaderboard CSV", csv_bytes, "leaderboard.csv", key="tabs_demo_dl")

st.divider()

# ==============================================================================
# 4. st.container() — Bordered Cards & Fixed-Height Windows
# ==============================================================================
st.header("4. Container Primitives (`st.container`)", divider="rainbow")
st.markdown(
    """
`st.container()` bundles multiple widgets together into unified layout blocks:
- **`border=True`**: Surrounds elements in an elevated card boundary.
- **`height=<pixels>`**: Enforces vertical constraints with smooth internal scrolling.
- **Out-of-Order Execution**: Assign a container to a variable `card = st.container()` to append elements later in the script execution flow.
"""
)

col_cnt1, col_cnt2 = st.columns(2)

with col_cnt1:
    st.markdown("**💻 Python Source Code (Bordered Cards & Scrollable Containers)**")
    code_container_demo = '''# 1. Bordered Action Card:
with st.container(border=True):
    st.markdown("##### 🛡️ Production Rate Limiting Policy")
    st.write("Current Token Bucket Capacity: 10,000 requests/minute")
    st.button("Modify Quotas", key="btn_quota")

# 2. Fixed-Height Scrollable Log Stream:
with st.container(height=130, border=True):
    for i in range(1, 9):
        st.caption(f"Log event #{i}: Cluster health check OK")'''
    st.code(code_container_demo, language="python")

with col_cnt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        st.markdown("##### 🛡️ Production Rate Limiting Policy")
        st.write("Current Token Bucket Capacity: 10,000 requests/minute")
        st.button("Modify Quotas", key="demo_btn_quota")

    with st.container(height=130, border=True):
        for i in range(1, 9):
            st.caption(f"Log event #{i}: Cluster health check OK (Node worker #{i})")

st.divider()

# ==============================================================================
# 5. st.empty() & 6. Updating Container Contents Dynamically
# ==============================================================================
st.header("5. Dynamic Placeholders (`st.empty`) & 6. Live In-Place Updates", divider="rainbow")
st.markdown(
    """
`st.empty()` creates a **single-element placeholder** that can be repeatedly overwritten in-place:
- **`placeholder.text(...)` / `placeholder.metric(...)`**: Replaces the contents of the slot instantly.
- **`placeholder.empty()`**: Clears the container, leaving zero whitespace on screen.
- **Contrast with `st.container()`**: `st.container()` appends multiple elements downward; `st.empty()` updates exactly one element in-place!
"""
)

col_emp1, col_emp2 = st.columns(2)

with col_emp1:
    st.markdown("**💻 Python Source Code (In-Place Live Training Loop)**")
    code_empty = '''# Create dynamic single-slot placeholder:
status_placeholder = st.empty()

# Button triggers in-place updates:
if st.button("Run Model Training Simulation"):
    for epoch in range(1, 6):
        loss = round(2.5 / epoch, 4)
        # Overwrite the exact same slot in-place:
        status_placeholder.info(f"Epoch {epoch}/5: Cross-Entropy Loss = {loss}")
        time.sleep(0.3)
    status_placeholder.success("Training Complete! Best Loss: 0.5000")'''
    st.code(code_empty, language="python")

with col_emp2:
    st.markdown("**🖥️ Live Interactive Simulation**")
    
    status_slot = st.empty()
    status_slot.caption("Status: Idle. Click button below to start live simulation.")
    
    if st.button("▶️ Run Training Simulation", key="btn_run_empty_sim"):
        for epoch in range(1, 6):
            loss = round(2.5 / epoch, 4)
            status_slot.info(f"🔄 Training Epoch **{epoch}/5**: Loss = `{loss}`")
            time.sleep(0.3)
        status_slot.success("✅ Training Complete! Model converged at Loss = `0.5000`.")

st.divider()

# ==============================================================================
# 7. Building Multi-Section Dashboards & 8. Organizing Complex Layouts
# ==============================================================================
st.header("7. Multi-Section Dashboards & 8. Complex Layout Hierarchy", divider="rainbow")
st.markdown(
    """
A production dashboard requires a rigorous structural hierarchy to avoid visual noise:
1. **Level 1 — Page Header**: Title, status badges, global breadcrumbs.
2. **Level 2 — Executive KPIs**: 3–4 high-level `st.metric()` cards with `border=True`.
3. **Level 3 — Categorical Tabs**: `st.tabs()` grouping domain pillars (e.g. Models, Costs, Hardware).
4. **Level 4 — Multi-Column Layouts**: `st.columns()` distributing data tables alongside controls inside each tab.
"""
)

# Executive KPI Scorecard
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric("Total Models in Fleet", "18", "+2 Deployments", border=True)
with kpi_c2:
    st.metric("Daily Active Queries", "1.25M", "+14.2%", border=True)
with kpi_c3:
    st.metric("p95 TTFT Latency", "62 ms", "-14 ms", delta_color="inverse", border=True)
with kpi_c4:
    st.metric("Fleet Budget Burn", "$4,850 / mo", "-$210", delta_color="inverse", border=True)

# Categorical Deep-Dive Tabs
dash_tab1, dash_tab2, dash_tab3 = st.tabs([
    "🤖 Active Model Registry", 
    "📈 Performance Analytics", 
    "⚙️ Node Topology"
])

with dash_tab1:
    t1_left, t1_right = st.columns([3, 1], gap="medium")
    with t1_left:
        st.markdown("##### Production Deployment Registry")
        st.dataframe(eval_df, width="stretch", hide_index=True, key="dash_reg_df")
    with t1_right:
        with st.container(border=True):
            st.markdown("##### Quick Filter")
            st.selectbox("Filter Provider", ["All", "Anthropic", "OpenAI", "DeepSeek"], key="dash_filt_prov")
            st.toggle("Show Canary Builds", value=False, key="dash_canary_tog")

with dash_tab2:
    st.markdown("##### Real-Time GPU Utilization Streams")
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        with st.container(border=True):
            st.metric("GPU VRAM Utilization", "84.2%", "+2.1%")
            st.progress(84, text="Allocation: 67.4 GB / 80 GB per H100")
    with perf_col2:
        with st.container(border=True):
            st.metric("InfiniBand Fabric Bandwidth", "342 Gb/s", "-12 Gb/s")
            st.progress(85, text="Fabric Capacity: 342 / 400 Gb/s")

with dash_tab3:
    with st.container(border=True):
        st.markdown("##### Cluster Node Interconnect Topology")
        st.info("NVIDIA Quantum-2 InfiniBand Leaf-Spine architecture active across 8 racks.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Container Primitives Reference Guide", divider="rainbow")
st.markdown(
    r"""
| Layout Primitive | Primary Role | In-Place Update Behavior |
| :--- | :--- | :--- |
| **`st.tabs()`** | Categorizes dense views horizontally | Client-side tab switching; zero script reruns |
| **`st.container()`** | Groups multiple elements with optional borders | Appends multiple elements; supports `border=True` & `height` |
| **`st.empty()`** | Single-element dynamic slot | Overwrites single element in-place; supports `.empty()` clearance |
| **`st.columns()`** | Slices horizontal canvas into side-by-side blocks | Responsive grid; stacks vertically on mobile devices |
"""
)
