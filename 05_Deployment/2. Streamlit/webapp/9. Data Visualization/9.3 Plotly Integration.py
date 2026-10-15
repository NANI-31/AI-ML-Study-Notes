import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Plotly Integration Tutorial",
    page_icon="✨",
    layout="wide",
)

st.title("✨ Plotly Interactive Visualizations — Complete Guide")
st.markdown(
    """
**Plotly** delivers rich, client-side JavaScript interactivity directly within Streamlit apps. 
Unlike static raster graphics, Plotly charts offer fluid pan, zoom, hover tooltips, trace toggling, 
3D rotation, and native **bidirectional selection events** (`on_select="rerun"`). 
With Streamlit's integrated `theme="streamlit"`, Plotly visuals automatically match your 
app's typography, dark/light theme, and color palette.
"""
)

st.divider()

# ==============================================================================
# 1. Installing Plotly & 2. Creating Interactive Charts (px vs. go)
# ==============================================================================
st.header("1. Architecture: Plotly Express (`px`) vs. Graph Objects (`go`)", divider="rainbow")
st.markdown(
    r"""
```bash
pip install plotly
```

Plotly provides two complementary APIs:
- **Plotly Express (`px`)**: High-level, concise declarative syntax that accepts tidy DataFrames.
- **Graph Objects (`go`)**: Lower-level, granular object hierarchy (`go.Figure()`, `go.Scatter()`) for complex multi-axis and custom subplots.

| Feature | Plotly Express (`px`) | Graph Objects (`go`) |
| :--- | :--- | :--- |
| **Primary Use** | Rapid exploratory data analysis | Bespoke, complex custom layouts |
| **Data Format** | Tidy (Long) Pandas DataFrames | Dicts, lists, NumPy arrays, Series |
| **Syntax** | Single-line function calls | Explicit trace and layout composition |
| **Complexity** | Minimal boilerplate | Full control over every canvas element |
"""
)

# ==============================================================================
# 3. st.plotly_chart() Parameters & Theming
# ==============================================================================
st.header("2. `st.plotly_chart()` Anatomy & Theme Synchronization", divider="rainbow")
st.markdown(
    r"""
```python
st.plotly_chart(
    figure_or_data: Figure | dict,
    width: "stretch" | "content" = "stretch",
    theme: "streamlit" | None = "streamlit",
    on_select: "ignore" | "rerun" = "ignore",
    selection_mode: Iterable["points" | "box" | "lasso"] = ("points", "box", "lasso"),
    key: str | int | None = None,
    **kwargs
)
```

- **`theme="streamlit"`**: Dynamically injects Streamlit theme colors, fonts, and dark mode styling. Set to `None` to preserve native Plotly aesthetics.
- **`width="stretch"`**: Automatically resizes chart to 100% width of parent container or column.
"""
)

col_th1, col_th2 = st.columns(2)

with col_th1:
    st.markdown("**💻 Python Source Code (Theme Comparison)**")
    code_theme = '''import plotly.express as px

fig = px.scatter(df, x="Epoch", y="Loss", color="Model", title="Loss Curves")

# Streamlit Theme (Default, adapts to Dark/Light mode)
st.plotly_chart(fig, width="stretch", theme="streamlit")

# Native Plotly Theme (Default Plotly styling)
st.plotly_chart(fig, width="stretch", theme=None)'''
    st.code(code_theme, language="python")

with col_th2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    theme_choice = st.radio("Active Theme Mode", ["Streamlit Theme (Adaptive)", "Native Plotly Theme (Raw)"], horizontal=True, key="rad_theme_sel")
    chosen_theme = "streamlit" if "Streamlit" in theme_choice else None

    df_demo = pd.DataFrame({
        "Step": [1, 2, 3, 4, 5, 6, 7],
        "Loss": [2.4, 1.8, 1.2, 0.9, 0.6, 0.45, 0.38],
        "Model": ["Qwen-2.5", "Qwen-2.5", "Qwen-2.5", "Qwen-2.5", "Qwen-2.5", "Qwen-2.5", "Qwen-2.5"]
    })
    fig_th = px.line(df_demo, x="Step", y="Loss", markers=True, title="Training Loss Descent")
    st.plotly_chart(fig_th, width="stretch", theme=chosen_theme, key="fig_th_demo")

st.divider()

# ==============================================================================
# 4. Interactive Line Charts (Multi-Trace, Sliders, Hover)
# ==============================================================================
st.header("3. Interactive Line Charts with Range Selectors", divider="rainbow")
st.markdown(
    """
Line charts in Plotly include interactive hover tooltips, click-to-hide legend traces, 
and optional X-axis range sliders for inspecting dense time-series telemetry.
"""
)

col_ln1, col_ln2 = st.columns(2)

with col_ln1:
    st.markdown("**💻 Python Source Code (Multi-Model Convergence)**")
    code_line = '''epochs = np.arange(1, 31)
line_df = pd.DataFrame({
    "Epoch": np.tile(epochs, 3),
    "Loss": np.concatenate([
        2.5 * np.exp(-epochs / 8.0) + np.random.normal(0, 0.02, 30),
        2.8 * np.exp(-epochs / 6.0) + np.random.normal(0, 0.02, 30),
        3.0 * np.exp(-epochs / 10.0) + np.random.normal(0, 0.02, 30)
    ]),
    "Model": ["Claude 3.7"] * 30 + ["GPT-4.5"] * 30 + ["DeepSeek R1"] * 30
})

fig = px.line(
    line_df, x="Epoch", y="Loss", color="Model",
    title="Fine-Tuning Convergence Comparison",
    markers=True
)
fig.update_layout(hovermode="x unified")
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, width="stretch")'''
    st.code(code_line, language="python")

with col_ln2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(42)
    ep = np.arange(1, 31)
    line_df = pd.DataFrame({
        "Epoch": np.tile(ep, 3),
        "Loss": np.concatenate([
            2.5 * np.exp(-ep / 8.0) + np.random.normal(0, 0.02, 30),
            2.8 * np.exp(-ep / 6.0) + np.random.normal(0, 0.02, 30),
            3.0 * np.exp(-ep / 10.0) + np.random.normal(0, 0.02, 30)
        ]),
        "Model": ["Claude 3.7"] * 30 + ["GPT-4.5"] * 30 + ["DeepSeek R1"] * 30
    })

    fig_line = px.line(
        line_df, x="Epoch", y="Loss", color="Model",
        title="Fine-Tuning Convergence Comparison",
        markers=True,
        color_discrete_map={"Claude 3.7": "#3b82f6", "GPT-4.5": "#10b981", "DeepSeek R1": "#f97316"}
    )
    fig_line.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_line, width="stretch", key="interactive_line_demo")
    st.caption("Hover over the chart to see unified tooltips across all models. Click legend items to toggle traces.")

st.divider()

# ==============================================================================
# 5. Interactive Bar Charts (Grouped, Stacked, Horizontal)
# ==============================================================================
st.header("4. Interactive Bar Charts (`barmode='group'` and `'stack'`)", divider="rainbow")
st.markdown(
    """
Bar charts support grouped and stacked visual comparisons across categorical dimensions. 
Add `text_auto=True` to automatically display values directly on top of each bar.
"""
)

col_bar1, col_bar2 = st.columns(2)

with col_bar1:
    st.markdown("**💻 Python Source Code (Benchmark Leaderboard)**")
    code_bar = '''bar_data = pd.DataFrame({
    "Benchmark": ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench"] * 2,
    "Score": [89.4, 92.0, 96.2, 78.4, 70.3, 88.7, 90.2, 95.1, 75.8, 65.2],
    "Model": ["Claude 3.7"] * 5 + ["GPT-4.5"] * 5
})

fig = px.bar(
    bar_data,
    x="Benchmark",
    y="Score",
    color="Model",
    barmode="group",
    text_auto=".1f",
    title="Frontier Model Reasoning Benchmarks"
)
st.plotly_chart(fig, width="stretch")'''
    st.code(code_bar, language="python")

with col_bar2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    barmode_sel = st.selectbox("Bar Layout Mode", ["group", "stack", "relative"], key="bar_mode_choice")
    bar_data = pd.DataFrame({
        "Benchmark": ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench"] * 2,
        "Score": [89.4, 92.0, 96.2, 78.4, 70.3, 88.7, 90.2, 95.1, 75.8, 65.2],
        "Model": ["Claude 3.7"] * 5 + ["GPT-4.5"] * 5
    })

    fig_bar = px.bar(
        bar_data,
        x="Benchmark",
        y="Score",
        color="Model",
        barmode=barmode_sel,
        text_auto=".1f",
        title=f"Reasoning Benchmarks (Mode: {barmode_sel})",
        color_discrete_map={"Claude 3.7": "#3b82f6", "GPT-4.5": "#10b981"}
    )
    st.plotly_chart(fig_bar, width="stretch", key="interactive_bar_demo")

st.divider()

# ==============================================================================
# 6. Interactive Scatter Plots & 9. Hover Tooltips
# ==============================================================================
st.header("5. Scatter Plots with Custom Hover Tooltips (`hovertemplate`)", divider="rainbow")
st.markdown(
    r"""
Scatter plots display multi-dimensional metrics by simultaneously mapping variables 
to **X-position**, **Y-position**, **Marker Size**, **Color**, and **Custom Hover Tooltips**.

```python
fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>Latency: %{x} ms<br>Pass@1: %{y:.1f}%<br>Params: %{marker.size}B"
)
```
"""
)

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**💻 Python Source Code (Pareto Frontier + Custom Hover)**")
    code_sc = '''scatter_df = pd.DataFrame({
    "Model": ["Claude 3.7", "GPT-4.5", "Gemini 2.5", "Llama 3.3 70B", "Qwen 2.5 72B", "DeepSeek R1"],
    "Latency_ms": [380, 420, 310, 240, 260, 920],
    "Accuracy":   [92.0, 90.2, 89.5, 82.1, 84.3, 93.6],
    "Params_B":   [180, 220, 150, 70, 72, 671],
    "Provider":   ["Anthropic", "OpenAI", "Google", "Meta", "Alibaba", "DeepSeek"]
})

fig = px.scatter(
    scatter_df,
    x="Latency_ms",
    y="Accuracy",
    size="Params_B",
    color="Provider",
    hover_name="Model",
    hover_data={"Latency_ms": True, "Accuracy": True, "Params_B": True, "Provider": False},
    title="Latency vs. Accuracy Frontier"
)
st.plotly_chart(fig, width="stretch")'''
    st.code(code_sc, language="python")

with col_sc2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    scatter_df = pd.DataFrame({
        "Model": ["Claude 3.7", "GPT-4.5", "Gemini 2.5", "Llama 3.3 70B", "Qwen 2.5 72B", "DeepSeek R1"],
        "Latency_ms": [380, 420, 310, 240, 260, 920],
        "Accuracy":   [92.0, 90.2, 89.5, 82.1, 84.3, 93.6],
        "Params_B":   [180, 220, 150, 70, 72, 671],
        "Provider":   ["Anthropic", "OpenAI", "Google", "Meta", "Alibaba", "DeepSeek"]
    })

    fig_sc = px.scatter(
        scatter_df,
        x="Latency_ms",
        y="Accuracy",
        size="Params_B",
        color="Provider",
        hover_name="Model",
        hover_data={"Latency_ms": ":.0f ms", "Accuracy": ":.1f%", "Params_B": ":.0f B", "Provider": False},
        title="Latency vs. Accuracy Frontier",
        size_max=35
    )
    st.plotly_chart(fig_sc, width="stretch", key="interactive_scatter_demo")
    st.caption("Hover over any marker to inspect model parameters, latency, and accuracy.")

st.divider()

# ==============================================================================
# 7. Pie & Donut Charts
# ==============================================================================
st.header("6. Pie and Donut Charts (`hole=0.4`)", divider="rainbow")
st.markdown(
    """
Visualize proportional budget, GPU allocation, or token expense breakdowns using `px.pie`. 
Adding `hole=0.4` converts a standard pie into an executive **Donut Chart**.
"""
)

col_pie1, col_pie2 = st.columns(2)

with col_pie1:
    st.markdown("**💻 Python Source Code (Cost Allocation Donut)**")
    code_pie = '''cost_df = pd.DataFrame({
    "Category": ["Fine-Tuning (H100)", "Inference Hosting", "Vector DB Storage", "Data Annotation"],
    "Cost_USD": [48500, 32100, 8400, 15200]
})

fig = px.pie(
    cost_df,
    names="Category",
    values="Cost_USD",
    hole=0.45,  # Creates the donut center hole
    title="Monthly AI Infrastructure Budget ($104,200)",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig, width="stretch")'''
    st.code(code_pie, language="python")

with col_pie2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    cost_df = pd.DataFrame({
        "Category": ["Fine-Tuning (H100)", "Inference Hosting", "Vector DB Storage", "Data Annotation"],
        "Cost_USD": [48500, 32100, 8400, 15200]
    })

    donut_hole = st.slider("Donut Hole Size", 0.0, 0.8, 0.45, 0.05, key="pie_hole_slider")
    fig_pie = px.pie(
        cost_df,
        names="Category",
        values="Cost_USD",
        hole=donut_hole,
        title=f"Monthly AI Budget Breakdown (Hole = {donut_hole:.2f})",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, width="stretch", key="interactive_pie_demo")

st.divider()

# ==============================================================================
# 8. Heatmaps (Correlation & Attention Matrices)
# ==============================================================================
st.header("7. Interactive Heatmaps (`px.imshow` & `go.Heatmap`)", divider="rainbow")
st.markdown(
    """
Heatmaps excel at showing dense 2D matrices, such as feature correlation grids 
or token-to-token attention weight matrices.
"""
)

col_hm1, col_hm2 = st.columns(2)

with col_hm1:
    st.markdown("**💻 Python Source Code (Attention Weight Heatmap)**")
    code_hm = '''tokens = ["The", "quick", "brown", "fox", "jumps"]
attention_matrix = np.array([
    [0.65, 0.15, 0.10, 0.05, 0.05],
    [0.10, 0.70, 0.12, 0.05, 0.03],
    [0.05, 0.12, 0.68, 0.10, 0.05],
    [0.08, 0.10, 0.22, 0.52, 0.08],
    [0.05, 0.08, 0.12, 0.25, 0.50]
])

fig = px.imshow(
    attention_matrix,
    x=tokens,
    y=tokens,
    color_continuous_scale="Blues",
    text_auto=".2f",
    title="Self-Attention Matrix (Head #4)"
)
st.plotly_chart(fig, width="stretch")'''
    st.code(code_hm, language="python")

with col_hm2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    attention_matrix = np.array([
        [0.65, 0.15, 0.10, 0.05, 0.05],
        [0.10, 0.70, 0.12, 0.05, 0.03],
        [0.05, 0.12, 0.68, 0.10, 0.05],
        [0.08, 0.10, 0.22, 0.52, 0.08],
        [0.05, 0.08, 0.12, 0.25, 0.50]
    ])

    color_scheme = st.selectbox("Color Scale", ["Blues", "Viridis", "Magma", "Teal"], key="hm_color_sel")
    fig_hm = px.imshow(
        attention_matrix,
        x=tokens,
        y=tokens,
        color_continuous_scale=color_scheme,
        text_auto=".2f",
        title="Transformer Self-Attention Weights",
        labels=dict(x="Key Token", y="Query Token", color="Weight")
    )
    st.plotly_chart(fig_hm, width="stretch", key="interactive_hm_demo")

st.divider()

# ==============================================================================
# 10. Chart Selection Events (Bidirectional Filtering)
# ==============================================================================
st.header("8. Bidirectional Chart Selection Events (`on_select='rerun'`)", divider="rainbow")
st.markdown(
    r"""
Streamlit features native **bidirectional chart selections**! 
By setting `on_select="rerun"`, Streamlit listens for user click, box-select, or lasso-select events 
and returns the selected data points as a stateful dictionary.

```python
selection = st.plotly_chart(
    fig,
    on_select="rerun",
    selection_mode=["points", "box"],
    key="interactive_selector"
)

# Extract selected points dynamically:
selected_points = selection["selection"]["points"]
```
"""
)

col_ev1, col_ev2 = st.columns(2)

with col_ev1:
    st.markdown("**💻 Python Source Code (Cross-Filtering Architecture)**")
    code_events = '''# 1. Render chart with interactive selection listener:
event = st.plotly_chart(
    fig,
    on_select="rerun",
    selection_mode=["points", "box", "lasso"],
    key="model_selector"
)

# 2. Inspect selection state:
selected_indices = [p["point_index"] for p in event["selection"]["points"]]

if selected_indices:
    filtered_data = df.iloc[selected_indices]
    st.dataframe(filtered_data)
else:
    st.info("Click or drag-select points on the chart above to filter data.")'''
    st.code(code_events, language="python")

with col_ev2:
    st.markdown("**🖥️ Live Interactive Selection Drilldown Demo**")
    st.caption("👇 **Try it:** Click any point or use the Box/Lasso tool in the chart toolbar above the plot.")

    eval_data = pd.DataFrame({
        "Model": ["Claude 3.7", "GPT-4.5", "Gemini 2.5", "Llama 3.3 70B", "Qwen 2.5 72B", "DeepSeek R1"],
        "Latency_ms": [380, 420, 310, 240, 260, 920],
        "Accuracy":   [92.0, 90.2, 89.5, 82.1, 84.3, 93.6],
        "Params_B":   [180, 220, 150, 70, 72, 671],
        "Cost_per_1M": [3.0, 2.5, 1.8, 0.6, 0.7, 0.55]
    })

    fig_sel = px.scatter(
        eval_data,
        x="Latency_ms",
        y="Accuracy",
        size="Params_B",
        color="Model",
        title="Interactive Model Selection Canvas",
        hover_name="Model"
    )

    chart_selection = st.plotly_chart(
        fig_sel,
        width="stretch",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        key="bidirectional_plotly_chart"
    )

    # Process selections
    points = chart_selection["selection"]["points"]
    if points:
        selected_pt_indices = [p["point_index"] for p in points if "point_index" in p]
        if selected_pt_indices:
            filtered_subset = eval_data.iloc[selected_pt_indices]
            st.success(f"🎯 **{len(filtered_subset)}** Model(s) Selected from Chart:")
            st.dataframe(filtered_subset, width="stretch", hide_index=True)
        else:
            st.info("Point indices not captured. Click a marker to select.")
    else:
        st.info("💡 Select one or more points above to reveal detailed model pricing and specs.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Plotly Integration Best Practices Reference", divider="rainbow")
st.markdown(
    r"""
| Capability | Best Practice | Code Pattern |
| :--- | :--- | :--- |
| **Theme Sync** | Inherit dark/light app theme | `st.plotly_chart(fig, theme="streamlit")` |
| **Full Width** | Fill column or container | `width="stretch"` |
| **Hover Tooltips** | Unified X-axis comparison | `fig.update_layout(hovermode="x unified")` |
| **Bar Labels** | Display values directly on bars | `px.bar(..., text_auto=".1f")` |
| **Donut Charts** | Center hole proportion | `px.pie(..., hole=0.45)` |
| **Heatmaps** | Matrix visualization | `px.imshow(matrix, color_continuous_scale="Blues")` |
| **Bidirectional Events**| React to chart clicks/lasso | `event = st.plotly_chart(fig, on_select="rerun")` |
"""
)
