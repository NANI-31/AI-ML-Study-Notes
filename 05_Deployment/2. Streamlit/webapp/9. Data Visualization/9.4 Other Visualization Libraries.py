import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pydeck as pdk

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Other Visualization Libraries",
    page_icon="🌌",
    layout="wide",
)

st.title("🌌 Specialized & Advanced Visualization Libraries")
st.markdown(
    """
Beyond native charts and Plotly, Streamlit provides deep, first-class integrations with 
**Altair** (declarative Grammar of Graphics), **Seaborn** (statistical modeling), 
**PyDeck** (hardware-accelerated 3D WebGL geospatial rendering), **Graphviz** (workflow DAGs & neural architectures), 
and **Vega-Lite** (pure JSON declarative specs).
"""
)

st.divider()

# ==============================================================================
# 1. Seaborn Integration (Advanced Statistical Visualizations)
# ==============================================================================
st.header("1. Seaborn Integration (Statistical Distributions & Multi-KDE)", divider="rainbow")
st.markdown(
    r"""
Seaborn excels at high-level statistical summaries (KDE distributions, violin plots, regression fits). 
Because Seaborn draws directly on Matplotlib canvases, render it with `st.pyplot(fig)` and **always call `plt.close(fig)`**.

```python
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=df, x="Latency", hue="Model", fill=True, common_norm=False, alpha=0.4, ax=ax)
st.pyplot(fig)
plt.close(fig)
```
"""
)

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**💻 Python Source Code (Multi-Model Latency Density)**")
    code_seaborn = '''fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)

sns.kdeplot(
    data=latency_df,
    x="Latency_ms",
    hue="Model",
    fill=True,
    common_norm=False,
    palette={"Claude 3.7": "#3b82f6", "GPT-4.5": "#10b981", "DeepSeek R1": "#f97316"},
    alpha=0.35,
    linewidth=2,
    ax=ax
)
ax.set_title("Inference Latency Density by Frontier Model", fontsize=11, fontweight="bold")
ax.set_xlabel("Time-to-First-Token (ms)")
ax.set_ylabel("Probability Density")
ax.grid(True, linestyle=":", alpha=0.6)

st.pyplot(fig, width="stretch")
plt.close(fig)'''
    st.code(code_seaborn, language="python")

with col_sb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    np.random.seed(42)
    latency_df = pd.DataFrame({
        "Latency_ms": np.concatenate([
            np.random.normal(320, 35, 300),
            np.random.normal(390, 45, 300),
            np.random.normal(510, 60, 300)
        ]),
        "Model": ["Claude 3.7"] * 300 + ["GPT-4.5"] * 300 + ["DeepSeek R1"] * 300
    })

    fig_sb, ax_sb = plt.subplots(figsize=(6.5, 3.6), dpi=100)
    sns.kdeplot(
        data=latency_df,
        x="Latency_ms",
        hue="Model",
        fill=True,
        common_norm=False,
        palette={"Claude 3.7": "#3b82f6", "GPT-4.5": "#10b981", "DeepSeek R1": "#f97316"},
        alpha=0.35,
        linewidth=2,
        ax=ax_sb
    )
    ax_sb.set_title("Inference Latency Density by Frontier Model", fontsize=10, fontweight="bold")
    ax_sb.set_xlabel("Time-to-First-Token (ms)")
    ax_sb.set_ylabel("Probability Density")
    ax_sb.grid(True, linestyle=":", alpha=0.5)

    st.pyplot(fig_sb, width="stretch")
    plt.close(fig_sb)

st.divider()

# ==============================================================================
# 2. Altair Integration (Declarative Grammar of Graphics)
# ==============================================================================
st.header("2. Altair Integration (`st.altair_chart`)", divider="rainbow")
st.markdown(
    r"""
Streamlit's native charts are internally compiled into **Altair** and **Vega-Lite**. 
Using Altair directly unlocks expressive capabilities like interactive brush intervals, 
multi-layer compositions (`+`), and custom encodings.

```python
st.altair_chart(
    altair_chart: Chart | LayerChart | VConcatChart,
    width: "stretch" | "content" = "stretch",
    theme: "streamlit" | None = "streamlit",
    on_select: "ignore" | "rerun" = "ignore",
    key: str | int | None = None
)
```
"""
)

col_alt1, col_alt2 = st.columns(2)

with col_alt1:
    st.markdown("**💻 Python Source Code (Interactive Interval Brush)**")
    code_alt = '''import altair as alt

# Define interactive selection interval:
brush = alt.selection_interval(encodings=["x"])

# Base scatter points:
base = alt.Chart(perf_df).mark_circle(size=80).encode(
    x=alt.X("Tokens_Per_Sec:Q", title="Throughput (Tokens/s)"),
    y=alt.Y("Accuracy:Q", title="HumanEval Pass@1 (%)"),
    color=alt.condition(brush, "Provider:N", alt.value("#cbd5e1")),
    tooltip=["Model", "Tokens_Per_Sec", "Accuracy", "Provider"]
).add_params(brush)

# Combined bar summary filtered by brush:
bars = alt.Chart(perf_df).mark_bar().encode(
    y=alt.Y("Provider:N", title="Provider"),
    x=alt.X("count():Q", title="Count"),
    color="Provider:N"
).transform_filter(brush)

chart = alt.vconcat(base, bars).properties(spacing=15)
st.altair_chart(chart, width="stretch", theme="streamlit")'''
    st.code(code_alt, language="python")

with col_alt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    perf_df = pd.DataFrame({
        "Model": ["Claude 3.7", "Claude 3.5", "GPT-4.5", "GPT-4o", "DeepSeek R1", "Llama 3.3 70B", "Qwen 2.5 72B", "Gemini 2.5 Flash"],
        "Tokens_Per_Sec": [68, 82, 55, 105, 42, 125, 95, 160],
        "Accuracy": [92.0, 90.5, 90.2, 88.4, 93.6, 82.1, 84.3, 85.0],
        "Provider": ["Anthropic", "Anthropic", "OpenAI", "OpenAI", "DeepSeek", "Meta", "Alibaba", "Google"]
    })

    brush = alt.selection_interval(encodings=["x"])
    points = alt.Chart(perf_df).mark_circle(size=90).encode(
        x=alt.X("Tokens_Per_Sec:Q", title="Throughput (Tokens/s)"),
        y=alt.Y("Accuracy:Q", title="HumanEval Pass@1 (%)", scale=alt.Scale(domain=[75, 100])),
        color=alt.condition(brush, "Provider:N", alt.value("#cbd5e1")),
        tooltip=["Model", "Tokens_Per_Sec", "Accuracy", "Provider"]
    ).add_params(brush).properties(height=200)

    bars = alt.Chart(perf_df).mark_bar().encode(
        y=alt.Y("Provider:N", title=None),
        x=alt.X("count():Q", title="Models Selected in Interval"),
        color="Provider:N"
    ).transform_filter(brush).properties(height=100)

    composed_alt = alt.vconcat(points, bars)
    st.altair_chart(composed_alt, width="stretch", theme="streamlit", key="alt_composed_demo")
    st.caption("Drag an interval brush horizontally across the top scatter plot to dynamically filter the bar chart below.")

st.divider()

# ==============================================================================
# 3. Bokeh Integration
# ==============================================================================
st.header("3. Bokeh Integration (`st.bokeh_chart`)", divider="rainbow")
st.markdown(
    r"""
Bokeh is a specialized library for rendering interactive graphics in modern web browsers:
- Supported in Streamlit via **`st.bokeh_chart(figure, width="stretch")`**.
- Ideal for streaming sensor telemetry, custom HTML tooltips, and specialized glyph arrangements.

```python
# Installation:
# pip install bokeh

import bokeh.plotting as bkp

p = bkp.figure(title="Real-Time Telemetry", x_axis_label="Step", y_axis_label="Norm")
p.line(x, y, line_width=2, color="#3b82f6")
st.bokeh_chart(p, width="stretch")
```
"""
)

col_bk1, col_bk2 = st.columns(2)

with col_bk1:
    st.markdown("**💻 Python Source Code (Bokeh Architecture Pattern)**")
    code_bokeh = '''try:
    from bokeh.plotting import figure
    
    p = figure(
        title="GPU Memory Utilization (GB)",
        x_axis_label="Timeline (Minutes)",
        y_axis_label="VRAM Used",
        width=550, height=280
    )
    p.line(x_mins, vram_gb, line_width=2.5, color="#10b981", legend_label="Allocated")
    p.circle(x_mins, vram_gb, size=6, color="#059669")
    
    st.bokeh_chart(p, width="stretch")
except ImportError:
    st.info("Install Bokeh to enable: pip install bokeh")'''
    st.code(code_bokeh, language="python")

with col_bk2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    try:
        from bokeh.plotting import figure  # type: ignore
        x_steps = np.arange(1, 15)
        y_vram = [24, 28, 45, 62, 70, 78, 79, 80, 80, 78, 65, 40, 26, 24]
        p_bk = figure(
            title="GPU Cluster VRAM Allocation",
            x_axis_label="Minutes",
            y_axis_label="Allocated VRAM (GB)",
            height=260
        )
        p_bk.line(x_steps, y_vram, line_width=2.5, color="#10b981", legend_label="VRAM")
        p_bk.circle(x_steps, y_vram, size=7, color="#059669")
        st.bokeh_chart(p_bk, width="stretch", key="bokeh_demo_chart")
    except ImportError:
        st.warning("⚠️ `bokeh` is not installed in the active environment. Run `pip install bokeh` to view native Bokeh canvas widgets.")
        st.caption("Streamlit's `st.bokeh_chart(p, width='stretch')` cleanly binds Bokeh glyph canvases when installed.")

st.divider()

# ==============================================================================
# 4. PyDeck Integration (3D WebGL Geospatial Overlays)
# ==============================================================================
st.header("4. PyDeck Integration (`st.pydeck_chart`)", divider="rainbow")
st.markdown(
    r"""
**PyDeck** (built on Uber's **deck.gl**) delivers hardware-accelerated 3D WebGL rendering 
for millions of points, 3D extruded hexagons, and inter-datacenter geodesic routing arcs.

```python
st.pydeck_chart(
    pydeck_obj: pdk.Deck,
    width: "stretch" | "content" = "stretch",
    on_select: "ignore" | "rerun" = "ignore",
    key: str | int | None = None
)
```
"""
)

col_pd1, col_pd2 = st.columns(2)

with col_pd1:
    st.markdown("**💻 Python Source Code (3D Column & Inter-DC Arcs)**")
    code_pydeck = '''import pydeck as pdk

# Global AI Data Centers
h100_clusters = pd.DataFrame({
    "name": ["US-East", "US-West", "EU-Central", "AP-East"],
    "lat": [39.0437, 37.7749, 52.5200, 35.6762],
    "lon": [-77.4875, -122.4194, 13.4050, 139.6503],
    "gpus": [16384, 8192, 4096, 6144]
})

# 3D Extruded Hexagon/Column Layer:
column_layer = pdk.Layer(
    "ColumnLayer",
    data=h100_clusters,
    get_position=["lon", "lat"],
    get_elevation="gpus",
    elevation_scale=50,
    radius=120000,
    get_fill_color=[56, 189, 248, 200],
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(latitude=35, longitude=10, zoom=1, pitch=45)
deck = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
st.pydeck_chart(deck, width="stretch")'''
    st.code(code_pydeck, language="python")

with col_pd2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    h100_clusters = pd.DataFrame({
        "name": ["US-East (Virginia)", "US-West (California)", "EU-Central (Frankfurt)", "AP-East (Tokyo)"],
        "lat": [39.0437, 37.7749, 50.1109, 35.6762],
        "lon": [-77.4875, -122.4194, 8.6821, 139.6503],
        "gpus": [16384, 8192, 4096, 6144]
    })

    column_layer = pdk.Layer(
        "ColumnLayer",
        data=h100_clusters,
        get_position=["lon", "lat"],
        get_elevation="gpus",
        elevation_scale=70,
        radius=180000,
        get_fill_color=[14, 165, 233, 220],
        pickable=True,
        auto_highlight=True
    )

    deck_view = pdk.ViewState(latitude=35, longitude=10, zoom=0.8, pitch=45, bearing=15)
    deck_obj = pdk.Deck(
        layers=[column_layer],
        initial_view_state=deck_view,
        tooltip={"text": "{name}\nH100 GPU Cluster: {gpus:,} units"}
    )
    st.pydeck_chart(deck_obj, width="stretch", key="pydeck_demo_chart")
    st.caption("Interactive 3D WebGL globe with GPU cluster columns extruded in proportion to cluster capacity.")

st.divider()

# ==============================================================================
# 5. Graphviz Diagrams (Agentic DAGs & Pipeline Architectures)
# ==============================================================================
st.header("5. Graphviz Diagrams (`st.graphviz_chart`)", divider="rainbow")
st.markdown(
    r"""
`st.graphviz_chart()` renders **Directed Acyclic Graphs (DAGs)**, workflow pipelines, 
and agent multi-step reasoning architectures using pure **DOT language**:
- **Zero Binary Dependency**: Streamlit parses and renders DOT strings on the client side using WebAssembly!
"""
)

col_gv1, col_gv2 = st.columns(2)

with col_gv1:
    st.markdown("**💻 Python Source Code (Agentic RAG Pipeline DAG)**")
    code_gv = '''rag_dag = """
digraph {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#f8fafc", color="#94a3b8"];
    edge [color="#64748b", fontname="Helvetica", fontsize=9];

    Query [label="User Prompt", fillcolor="#e0f2fe", color="#0284c7"];
    Router [label="Semantic Router", shape=diamond, fillcolor="#fef3c7", color="#d97706"];
    VecDB [label="Milvus / Pinecone\\nVector DB", fillcolor="#ecfdf5", color="#059669"];
    Reranker [label="Cross-Encoder\\nReranker", fillcolor="#f5f3ff", color="#7c3aed"];
    LLM [label="DeepSeek R1 / Claude\\nSynthesis", fillcolor="#fae8ff", color="#c026d3"];
    Response [label="Validated Output", fillcolor="#dcfce7", color="#16a34a"];

    Query -> Router;
    Router -> VecDB [label="Semantic Search"];
    Router -> LLM [label="Direct Answer"];
    VecDB -> Reranker [label="Top-K Chunks"];
    Reranker -> LLM [label="Top-N Context"];
    LLM -> Response;
}
"""
st.graphviz_chart(rag_dag, width="stretch")'''
    st.code(code_gv, language="python")

with col_gv2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    rag_dag = """
    digraph {
        rankdir=LR;
        node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#f8fafc", color="#94a3b8"];
        edge [color="#64748b", fontname="Helvetica", fontsize=9];

        Query [label="User Prompt", fillcolor="#e0f2fe", color="#0284c7"];
        Router [label="Semantic Router", shape=diamond, fillcolor="#fef3c7", color="#d97706"];
        VecDB [label="Milvus / Pinecone\\nVector DB", fillcolor="#ecfdf5", color="#059669"];
        Reranker [label="Cross-Encoder\\nReranker", fillcolor="#f5f3ff", color="#7c3aed"];
        LLM [label="DeepSeek R1 / Claude\\nSynthesis", fillcolor="#fae8ff", color="#c026d3"];
        Response [label="Validated Output", fillcolor="#dcfce7", color="#16a34a"];

        Query -> Router;
        Router -> VecDB [label="Semantic Search"];
        Router -> LLM [label="Direct Answer"];
        VecDB -> Reranker [label="Top-K Chunks"];
        Reranker -> LLM [label="Top-N Context"];
        LLM -> Response;
    }
    """
    st.graphviz_chart(rag_dag, width="stretch")
    st.caption("Agentic Retrieval-Augmented Generation (RAG) architectural DAG rendered natively via DOT grammar.")

st.divider()

# ==============================================================================
# 6. Vega-Lite Visualizations (`st.vega_lite_chart`)
# ==============================================================================
st.header("6. Vega-Lite Visualizations (`st.vega_lite_chart`)", divider="rainbow")
st.markdown(
    r"""
If your application stores chart configurations as **JSON schemas** (e.g., dynamically generated by LLM function calling), 
`st.vega_lite_chart()` renders them directly without constructing Python object trees.

```python
st.vega_lite_chart(
    data: DataFrame | None,
    spec: dict,
    width: "stretch" | "content" = "stretch",
    theme: "streamlit" | None = "streamlit"
)
```
"""
)

col_vl1, col_vl2 = st.columns(2)

with col_vl1:
    st.markdown("**💻 Python Source Code (Pure JSON Spec)**")
    code_vl = '''vega_spec = {
    "mark": {"type": "bar", "cornerRadiusEnd": 4},
    "encoding": {
        "x": {"field": "Model", "type": "nominal", "axis": {"labelAngle": 0}},
        "y": {"field": "PassRate", "type": "quantitative", "title": "Pass@1 (%)"},
        "color": {
            "field": "PassRate",
            "type": "quantitative",
            "scale": {"scheme": "tealblues"}
        }
    }
}

st.vega_lite_chart(benchmark_df, vega_spec, width="stretch")'''
    st.code(code_vl, language="python")

with col_vl2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    benchmark_df = pd.DataFrame({
        "Model": ["Claude 3.7", "GPT-4.5", "DeepSeek R1", "Qwen 2.5", "Gemma 2"],
        "PassRate": [92.0, 90.2, 93.6, 84.3, 75.2]
    })

    vega_spec = {
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "x": {"field": "Model", "type": "nominal", "axis": {"labelAngle": -20}},
            "y": {"field": "PassRate", "type": "quantitative", "title": "Pass@1 (%)", "scale": {"domain": [70, 100]}},
            "color": {
                "field": "PassRate",
                "type": "quantitative",
                "scale": {"scheme": "tealblues"}
            }
        }
    }
    st.vega_lite_chart(benchmark_df, vega_spec, width="stretch")
    st.caption("Rendered from pure JSON dictionary specification using `st.vega_lite_chart`.")

st.divider()

# ==============================================================================
# 7. Comparing Visualization Libraries & 8. Choosing the Right Tool
# ==============================================================================
st.header("7. Comparative Matrix & 8. Decision Guide", divider="rainbow")
st.markdown(
    r"""
### Visualization Ecosystem Comparison

| Library | Engine / Rendering | Interactivity | 3D Support | Primary AI / Data Science Fit |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit Native** | Altair / Vega-Lite | Pan / Zoom / Hover | ❌ No | Rapid prototyping, zero-boilerplate telemetry |
| **Matplotlib** | Raster / Vector (Agg) | Static (PNG) | Limited | Academic papers, complex multi-panel subplots |
| **Seaborn** | Matplotlib wrapper | Static (PNG) | ❌ No | Statistical modeling, KDEs, regression residuals |
| **Plotly** | WebGL / D3.js | Bidirectional (`on_select`) | ✅ Yes | Production dashboards, Pareto frontiers, drill-downs |
| **Altair** | Vega-Lite | Brush & Interval Filters | ❌ No | Linked multi-view exploratory visualizations |
| **PyDeck** | deck.gl (WebGL) | 3D Camera Controls | ✅ Yes | Geospatial maps, million-point point clouds |
| **Graphviz** | WebAssembly (DOT) | Static DAG | ❌ No | Pipeline workflows, LangGraph agent execution DAGs |

---

### Decision Framework: Which Library Should You Choose?
"""
)

st.graphviz_chart(
    r"""
    digraph VizDecisionTree {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        root [label="Visualization Need?", fillcolor="#DBEAFE", color="#1D4ED8"];

        dag [label="st.graphviz_chart()\nPipeline architecture, DAGs, LLM agent steps", fillcolor="#FEF3C7", color="#D97706"];
        geo [label="st.pydeck_chart()\n3D Geospatial, point clouds", fillcolor="#FEF3C7", color="#D97706"];
        sns [label="Seaborn + st.pyplot()\nKDE, violin, regression distributions", fillcolor="#D1FAE5", color="#059669"];
        mpl [label="Matplotlib + st.pyplot()\nMulti-axis academic publication figures", fillcolor="#D1FAE5", color="#059669"];
        alt [label="Altair + st.altair_chart()\nLinked multi-view interval brushing", fillcolor="#EDE9FE", color="#7C3AED"];
        ply [label="Plotly + st.plotly_chart()\nInteractive dashboards, click/lasso selections", fillcolor="#FCE7F3", color="#DB2777"];
        nat [label="Native st.line_chart() / st.bar_chart()\nFast responsive charts, zero setup", fillcolor="#F1F5F9", color="#475569"];

        root -> dag [label="Pipelines / DAGs"];
        root -> geo [label="3D Geospatial"];
        root -> sns [label="Statistical KDE"];
        root -> mpl [label="Publication Custom"];
        root -> alt [label="Linked Brushing"];
        root -> ply [label="Interactive Drilldowns"];
        root -> nat [label="Fast Zero-Setup"];
    }
    """
)
