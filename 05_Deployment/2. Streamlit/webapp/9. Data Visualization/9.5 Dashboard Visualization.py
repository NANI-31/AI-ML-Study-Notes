import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Dashboard Visualization Tutorial",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Production Dashboard Visualization — Complete Guide")
st.markdown(
    """
Dashboards synthesize complex, multi-dimensional business and operational metrics into 
cohesive, decision-ready visual interfaces. A world-class Streamlit dashboard combines 
**KPI summary cards**, **time-series forecasting**, **statistical distributions**, 
**correlation matrices**, **geographic maps**, and **bidirectional cross-filtering**.
"""
)

st.divider()

# ==============================================================================
# 1. KPI Cards & Metric Ribbons
# ==============================================================================
st.header("1. KPI Cards & Metric Ribbons (`st.metric` in Containers)", divider="rainbow")
st.markdown(
    r"""
The top of every executive dashboard starts with a **KPI summary ribbon**. 
Using `st.container(border=True)` combined with `st.columns` gives KPI cards a modern card appearance.

```python
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ARR (Annual Run Rate)", "$1.42M", "+18.4%")
    col2.metric("Total Tokens Ingested", "482.1M", "+24.2%")
    col3.metric("P99 Inference Latency", "340 ms", "-12.5%", delta_color="inverse")
    col4.metric("Uptime SLA", "99.98%", "+0.02%")
```
"""
)

col_kpi1, col_kpi2 = st.columns(2)

with col_kpi1:
    st.markdown("**💻 Python Source Code (Executive KPI Ribbon)**")
    code_kpi = '''with st.container(border=True):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monthly Recurring Revenue", "$124,500", "+14.2%")
    k2.metric("Active LLM API Clients", "1,840", "+85 new")
    k3.metric("Avg P99 Latency", "380 ms", "-45 ms", delta_color="inverse")
    k4.metric("Cost per 1M Tokens", "$1.12", "-$0.18", delta_color="inverse")'''
    st.code(code_kpi, language="python")

with col_kpi2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Monthly Revenue", "$124,500", "+14.2%")
        k2.metric("Active Clients", "1,840", "+85 new")
        k3.metric("P99 Latency", "380 ms", "-45 ms", delta_color="inverse")
        k4.metric("Cost / 1M Tokens", "$1.12", "-$0.18", delta_color="inverse")
    st.caption("Structured KPI ribbon with inverted delta styling for latency and cost reductions.")

st.divider()

# ==============================================================================
# 2. Sales & Revenue Funnel Dashboards
# ==============================================================================
st.header("2. Revenue & Sales Conversion Funnels", divider="rainbow")
st.markdown(
    """
Track multi-tier enterprise subscription revenue and customer conversion stages 
using grouped bar charts and funnel visualizations (`go.Funnel`).
"""
)

col_sl1, col_sl2 = st.columns(2)

with col_sl1:
    st.markdown("**💻 Python Source Code (Conversion Funnel)**")
    code_funnel = '''import plotly.graph_objects as go

funnel_stages = ["API Signups", "Free Tier Active", "Credit Card Added", "Enterprise Tier"]
funnel_users = [24000, 14200, 4800, 1120]

fig = go.Figure(go.Funnel(
    y=funnel_stages,
    x=funnel_users,
    textinfo="value+percent previous",
    marker=dict(color=["#3b82f6", "#06b6d4", "#10b981", "#8b5cf6"])
))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=20))
st.plotly_chart(fig, width="stretch")'''
    st.code(code_funnel, language="python")

with col_sl2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    funnel_stages = ["API Signups", "Free Tier Active", "Credit Card Added", "Enterprise Tier"]
    funnel_users = [24000, 14200, 4800, 1120]

    fig_fn = go.Figure(go.Funnel(
        y=funnel_stages,
        x=funnel_users,
        textinfo="value+percent previous",
        marker=dict(color=["#3b82f6", "#06b6d4", "#10b981", "#8b5cf6"])
    ))
    fig_fn.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fn, width="stretch", key="funnel_chart_demo")

st.divider()

# ==============================================================================
# 3. Time-Series Telemetry & Forecasting
# ==============================================================================
st.header("3. Time-Series Dashboards (Moving Averages & Confidence Bands)", divider="rainbow")
st.markdown(
    """
Display long-term trends alongside smoothed 7-day rolling averages and anomaly bounds 
to prevent knee-jerk reactions to single-day noise.
"""
)

col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    st.markdown("**💻 Python Source Code (Trend + 7-Day Moving Average)**")
    code_ts = '''dates = pd.date_range("2026-09-01", periods=40, freq="D")
raw_vals = 1000 + np.cumsum(np.random.normal(50, 80, 40))
ts_df = pd.DataFrame({"Date": dates, "Daily Calls": raw_vals})
ts_df["7-Day MA"] = ts_df["Daily Calls"].rolling(7, min_periods=1).mean()

fig = px.line(ts_df, x="Date", y=["Daily Calls", "7-Day MA"],
              color_discrete_map={"Daily Calls": "#94a3b8", "7-Day MA": "#2563eb"},
              title="API Volume Trend & Moving Average")
st.plotly_chart(fig, width="stretch")'''
    st.code(code_ts, language="python")

with col_ts2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(42)
    dates = pd.date_range("2026-09-01", periods=40, freq="D")
    raw_vals = 1000 + np.cumsum(np.random.normal(50, 80, 40))
    ts_df = pd.DataFrame({"Date": dates, "Daily Calls": raw_vals})
    ts_df["7-Day MA"] = ts_df["Daily Calls"].rolling(7, min_periods=1).mean()

    fig_ts = px.line(
        ts_df, x="Date", y=["Daily Calls", "7-Day MA"],
        color_discrete_map={"Daily Calls": "#94a3b8", "7-Day MA": "#2563eb"},
        title="API Request Volume & 7-Day Rolling Trend"
    )
    fig_ts.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ts, width="stretch", key="ts_chart_demo")

st.divider()

# ==============================================================================
# 4. Distribution Analysis (Percentiles & Outliers)
# ==============================================================================
st.header("4. Distribution Analysis (Box Plots & Histograms)", divider="rainbow")
st.markdown(
    """
Averages obscure variance. Use **Box Plots** and **Violin Plots** to diagnose latency tails (P50 vs P99) 
and detect rogue edge cases across distinct model families.
"""
)

col_dst1, col_dst2 = st.columns(2)

with col_dst1:
    st.markdown("**💻 Python Source Code (Latency Box Distribution)**")
    code_box = '''box_df = pd.DataFrame({
    "Latency_ms": np.concatenate([
        np.random.normal(250, 30, 200),
        np.random.normal(380, 45, 200),
        np.random.normal(580, 90, 200)
    ]),
    "Model": ["Claude 3.7"] * 200 + ["GPT-4.5"] * 200 + ["DeepSeek R1"] * 200
})

fig = px.box(box_df, x="Model", y="Latency_ms", color="Model",
             points="outliers", title="Latency Distribution & Outlier Spread")
st.plotly_chart(fig, width="stretch")'''
    st.code(code_box, language="python")

with col_dst2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(77)
    box_df = pd.DataFrame({
        "Latency_ms": np.concatenate([
            np.random.normal(250, 30, 200),
            np.random.normal(380, 45, 200),
            np.random.normal(580, 90, 200)
        ]),
        "Model": ["Claude 3.7"] * 200 + ["GPT-4.5"] * 200 + ["DeepSeek R1"] * 200
    })

    fig_bx = px.box(
        box_df, x="Model", y="Latency_ms", color="Model",
        points="outliers", title="Latency Tail Profiles by Model",
        color_discrete_map={"Claude 3.7": "#3b82f6", "GPT-4.5": "#10b981", "DeepSeek R1": "#f97316"}
    )
    fig_bx.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_bx, width="stretch", key="box_dist_demo")

st.divider()

# ==============================================================================
# 5. Correlation Heatmaps
# ==============================================================================
st.header("5. Feature & Metric Correlation Heatmaps", divider="rainbow")
st.markdown(
    """
Detect collinearity and trade-offs (e.g. prompt length vs token cost vs response quality) 
using normalized correlation heatmaps (`df.corr()`).
"""
)

col_cr1, col_cr2 = st.columns(2)

with col_cr1:
    st.markdown("**💻 Python Source Code (Correlation Matrix)**")
    code_corr = '''metrics_df = pd.DataFrame({
    "Prompt_Tokens": np.random.randint(100, 2000, 100),
    "Completion_Tokens": np.random.randint(50, 800, 100),
    "Latency_ms": np.random.randint(200, 1200, 100),
    "Quality_Score": np.random.uniform(0.7, 1.0, 100),
    "Cost_USD": np.random.uniform(0.01, 0.15, 100)
})

corr = metrics_df.corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, title="Metrics Correlation Matrix")
st.plotly_chart(fig, width="stretch")'''
    st.code(code_corr, language="python")

with col_cr2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(99)
    p_tok = np.random.randint(100, 2000, 80)
    c_tok = np.random.randint(50, 800, 80)
    lat = p_tok * 0.4 + c_tok * 0.8 + np.random.normal(50, 20, 80)
    cost = (p_tok * 0.000003 + c_tok * 0.000015)
    qual = 0.85 + (p_tok / 5000) - np.random.uniform(0, 0.1, 80)

    metrics_df = pd.DataFrame({
        "Prompt_Tok": p_tok,
        "Compl_Tok": c_tok,
        "Latency": lat,
        "Cost": cost,
        "Quality": qual
    })

    corr_mat = metrics_df.corr()
    fig_cr = px.imshow(
        corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Inference Metrics Correlation Matrix"
    )
    fig_cr.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_cr, width="stretch", key="corr_demo_chart")

st.divider()

# ==============================================================================
# 6. Geographic Dashboards
# ==============================================================================
st.header("6. Geographic Dashboards (Regional Cluster Telemetry)", divider="rainbow")
st.markdown(
    """
Display global datacenter distributions, active user concentrations, or latency zones 
using `st.map()` or Plotly's `scatter_geo`.
"""
)

col_geo1, col_geo2 = st.columns(2)

with col_geo1:
    st.markdown("**💻 Python Source Code (Regional Scatter Map)**")
    code_geo = '''geo_df = pd.DataFrame({
    "lat": [37.7749, 40.7128, 51.5074, 35.6762, 1.3521, -33.8688],
    "lon": [-122.4194, -74.0060, -0.1278, 139.6503, 103.8198, 151.2093],
    "Region": ["US-West", "US-East", "EU-West", "AP-North", "AP-South", "AU-East"],
    "Requests_K": [420, 580, 310, 280, 190, 120]
})

st.map(geo_df, latitude="lat", longitude="lon", size="Requests_K", color="#0284c7")'''
    st.code(code_geo, language="python")

with col_geo2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    geo_df = pd.DataFrame({
        "lat": [37.7749, 40.7128, 51.5074, 35.6762, 1.3521, -33.8688],
        "lon": [-122.4194, -74.0060, -0.1278, 139.6503, 103.8198, 151.2093],
        "Region": ["US-West", "US-East", "EU-West", "AP-North", "AP-South", "AU-East"],
        "Requests_K": [420, 580, 310, 280, 190, 120]
    })
    st.map(geo_df, latitude="lat", longitude="lon", size="Requests_K", color="#0284c7", zoom=0.5)

st.divider()

# ==============================================================================
# 7. Interactive Filters & 8. Cross-Filtering Concepts
# ==============================================================================
st.header("7. Interactive Filters & 8. Cross-Filtering Architecture", divider="rainbow")
st.markdown(
    r"""
### What is Cross-Filtering?
In a traditional dashboard, filtering is one-way (e.g. sidebar controls the page). 
In modern dashboards, **interaction with any chart element immediately filters the rest of the dashboard**:
1. User filters by Region or Model family $\to$ KPI cards update instantly.
2. Time-series chart zooms to a specific week $\to$ Distribution box plot recalculates for that exact week.
3. User lasso-selects outliers on the scatter plot $\to$ Summary table lists those specific records.
"""
)

# ==============================================================================
# 9. Capstone: Complete Multi-Chart Enterprise AI Observability Dashboard
# ==============================================================================
st.header("9. Capstone: Live Multi-Chart Enterprise AI Observability Dashboard", divider="rainbow")

# Synthetic Multi-Dimensional Telemetry Dataset
np.random.seed(123)
n_records = 300
all_models = ["Claude 3.7", "GPT-4.5", "DeepSeek R1", "Llama 3.3 70B", "Qwen 2.5 72B"]
all_regions = ["North America", "Europe", "Asia-Pacific"]

dash_df = pd.DataFrame({
    "Date": np.random.choice(pd.date_range("2026-10-01", "2026-10-12"), size=n_records),
    "Model": np.random.choice(all_models, size=n_records),
    "Region": np.random.choice(all_regions, size=n_records, p=[0.5, 0.3, 0.2]),
    "Tokens_Prompt": np.random.randint(100, 3000, size=n_records),
    "Tokens_Completion": np.random.randint(50, 1500, size=n_records),
    "Latency_ms": np.random.normal(350, 80, size=n_records).clip(120, 800),
    "Cost_USD": np.random.uniform(0.005, 0.08, size=n_records),
    "Status": np.random.choice(["200 OK", "200 OK", "200 OK", "429 RateLimit", "504 Timeout"], size=n_records)
})

# Top Filter Ribbon
with st.container(border=True):
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        sel_models = st.multiselect("Filter Models", options=all_models, default=["Claude 3.7", "GPT-4.5", "DeepSeek R1"], key="dash_sel_models")
    with fc2:
        sel_regions = st.multiselect("Filter Region", options=all_regions, default=all_regions, key="dash_sel_regions")
    with fc3:
        max_lat_cutoff = st.slider("Max Allowed Latency (ms)", 200, 800, 750, 25, key="dash_lat_cut")

# Apply Dynamic Cross-Filtering
filtered_dash = dash_df[
    (dash_df["Model"].isin(sel_models)) &
    (dash_df["Region"].isin(sel_regions)) &
    (dash_df["Latency_ms"] <= max_lat_cutoff)
]

# Section A: Live KPI Cards
with st.container(border=True):
    kp1, kp2, kp3, kp4 = st.columns(4)
    total_tokens = (filtered_dash["Tokens_Prompt"] + filtered_dash["Tokens_Completion"]).sum()
    total_cost = filtered_dash["Cost_USD"].sum()
    avg_latency = filtered_dash["Latency_ms"].mean() if len(filtered_dash) > 0 else 0
    error_rate = ((filtered_dash["Status"] != "200 OK").mean() * 100) if len(filtered_dash) > 0 else 0

    kp1.metric("Processed Tokens", f"{total_tokens / 1e3:,.1f}k")
    kp2.metric("Cumulative Cost", f"${total_cost:.2f}")
    kp3.metric("Avg Latency", f"{avg_latency:.1f} ms")
    kp4.metric("Error Rate", f"{error_rate:.1f}%", delta_color="inverse")

# Section B: Multi-Chart Layout (Time Series & Regional Breakdown)
ch_col1, ch_col2 = st.columns([3, 2])

with ch_col1:
    with st.container(border=True):
        st.subheader("Daily Token Throughput by Model")
        daily_tokens = filtered_dash.groupby(["Date", "Model"])[["Tokens_Prompt", "Tokens_Completion"]].sum().sum(axis=1).reset_index(name="Tokens")
        fig_d_ts = px.line(
            daily_tokens, x="Date", y="Tokens", color="Model",
            markers=True
        )
        fig_d_ts.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_d_ts, width="stretch", key="dash_daily_ts")

with ch_col2:
    with st.container(border=True):
        st.subheader("Regional Traffic Share")
        reg_counts = filtered_dash["Region"].value_counts().reset_index()
        reg_counts.columns = ["Region", "Requests"]
        fig_reg = px.pie(reg_counts, names="Region", values="Requests", hole=0.45)
        fig_reg.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_reg, width="stretch", key="dash_reg_pie")

# Section C: Distribution & Outlier Drift
d_col1, d_col2 = st.columns(2)

with d_col1:
    with st.container(border=True):
        st.subheader("Latency Distribution by Model")
        fig_d_lat = px.box(filtered_dash, x="Model", y="Latency_ms", color="Model")
        fig_d_lat.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_d_lat, width="stretch", key="dash_lat_box")

with d_col2:
    with st.container(border=True):
        st.subheader("HTTP Status Code Breakdown")
        status_counts = filtered_dash.groupby(["Model", "Status"]).size().reset_index(name="Count")
        fig_stat = px.bar(status_counts, x="Model", y="Count", color="Status", barmode="stack")
        fig_stat.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_stat, width="stretch", key="dash_status_bar")

# Section D: Raw Data Table Drilldown
with st.expander(f"🔍 Inspect Filtered Telemetry Records ({len(filtered_dash)} Rows)", expanded=False):
    st.dataframe(filtered_dash, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Production Dashboard Design Principles", divider="rainbow")
st.markdown(
    r"""
| Dashboard Tier | Component Pattern | Best Practice |
| :--- | :--- | :--- |
| **Header Ribbon** | `st.container(border=True)` + `st.metric()` | 3–5 high-level metrics with delta indicators |
| **Filter Bar** | Top-of-page columns with multiselects | Keep global filters prominent; preserve URL query params |
| **Primary Trends**| Line / Stacked Area charts (`px.line`) | Use 7-day rolling averages to smooth out day-to-day noise |
| **Proportions** | Donut charts (`px.pie(hole=0.45)`) | Limit to $\le 6$ slices; sort descending |
| **Distributions** | Box / Violin plots (`px.box`) | Highlight tail risk (P90, P99) rather than relying only on mean |
| **Drilldown** | `st.expander` with `st.dataframe` | Keep dense raw records collapsed by default to avoid visual clutter |
"""
)
