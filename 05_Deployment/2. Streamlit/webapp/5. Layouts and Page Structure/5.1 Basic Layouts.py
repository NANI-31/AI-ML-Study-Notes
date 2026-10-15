import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Basic Layouts Tutorial",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Basic Layouts (`st.columns` & `st.container`) — Complete Guide")
st.markdown(
    """
Layout architecture dictates how users navigate and consume insights in an analytical web app. 
By default, Streamlit streams elements in a single top-to-bottom vertical stack. 
With **`st.columns()`** and **`st.container()`**, you can structure responsive multi-column grids, 
card wrappers with borders, scrollable sub-panels, vertical alignments, and complex dashboard hierarchies.
"""
)

st.divider()

# ==============================================================================
# 1. Understanding the Streamlit Page Layout
# ==============================================================================
st.header("1. Understanding the Streamlit Page Layout", divider="rainbow")
st.markdown(
    """
Streamlit apps have a foundational page structure configured via `st.set_page_config()`:
- **`layout="centered"` (Default)**: Restricts main content to a fixed max-width (~730px). Ideal for text-heavy articles, linear questionnaires, and simple chatbots.
- **`layout="wide"`**: Expands content to fill 100% of the browser viewport width. Essential for data grids, multi-column analytics dashboards, and AI evaluation consoles.
- **Execution Flow**: Elements render sequentially top-to-bottom, unless captured inside containers or column blocks.
"""
)

col_pg1, col_pg2 = st.columns(2)

with col_pg1:
    st.markdown("**💻 Python Source Code**")
    code_page_cfg = '''import streamlit as st

# Must be the very first Streamlit command in your script:
st.set_page_config(
    page_title="AI Operations Dashboard",
    page_icon="📊",
    layout="wide",  # "wide" (full width) or "centered" (narrow column)
    initial_sidebar_state="expanded"
)'''
    st.code(code_page_cfg, language="python")

with col_pg2:
    st.markdown("**🖥️ Layout Specifications Reference**")
    st.info(
        """
        - **Current Active Mode**: `layout="wide"` (Full browser width enabled).
        - **Header Bar**: Top system bar holding app status, rerun menu, and settings.
        - **Sidebar**: Left collapsible drawer for global filters and navigation.
        - **Main Area**: Fluid grid canvas supporting multi-column layouts and cards.
        """
    )

st.divider()

# ==============================================================================
# 2. st.container() — Grouping, Cards & Scrollable Windows
# ==============================================================================
st.header("2. Container Primitives (`st.container`)", divider="rainbow")
st.markdown(
    """
`st.container()` organizes elements into logical visual units:
- **`border=True`**: Surrounds contained elements with a clean card outline.
- **`height=<pixels>`**: Enforces a fixed vertical height with an internal scrollbar.
- **Out-of-Order Execution**: Assign a container to a variable `c = st.container()` and write into it later in your script!
"""
)

st.markdown("#### 📐 Formal Signature: `st.container()`")
st.code(
    """st.container(
    *,
    border: bool | None = None,
    height: int | None = None,  # Fixed pixel height creates a scrollable container
    width: "stretch" | "content" = "stretch",
    gap: "small" | "medium" | "large" | None = "small",
    key: str | int | None = None
) -> DeltaGenerator""",
    language="python",
)

col_cnt1, col_cnt2 = st.columns(2)

with col_cnt1:
    st.markdown("**💻 Python Source Code (Bordered Cards & Scrollable Panels)**")
    code_container = '''# 1. Bordered Card Container:
with st.container(border=True):
    st.markdown("##### 🛡️ Production Model Security Guardrail")
    st.write("Real-time prompt injection & PII redaction layer active.")
    st.button("Audit Guardrail Rules", key="btn_guardrail")

# 2. Fixed-Height Scrollable Log Container:
with st.container(height=140, border=True):
    st.caption("Live Inference Event Stream (Scrollable):")
    for i in range(1, 11):
        st.write(f"[{i:02d}] 2026-10-12 12:00:{i:02d} — Request handled (Status 200)")'''
    st.code(code_container, language="python")

with col_cnt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    # 1. Bordered Card
    with st.container(border=True):
        st.markdown("##### 🛡️ Production Model Security Guardrail")
        st.write("Real-time prompt injection & PII redaction layer active.")
        st.button("Audit Guardrail Rules", key="demo_btn_guardrail")

    # 2. Scrollable Container
    with st.container(height=140, border=True):
        st.caption("Live Inference Event Stream (Scrollable):")
        for i in range(1, 11):
            st.write(f"[{i:02d}] 2026-10-12 12:00:{i:02d} — Request handled (Status 200)")

st.divider()

# ==============================================================================
# 3. st.columns() & 4. Creating Multiple Equal Columns
# ==============================================================================
st.header("3. Multi-Column Grids & 4. Equal Columns (`st.columns`)", divider="rainbow")
st.markdown(
    """
`st.columns(spec)` slices the horizontal space into side-by-side columns:
- Pass an integer `n` to create `n` equal-width columns (`st.columns(3)`).
- Returns a list of column containers `[col1, col2, ...]`.
- Use context managers `with col1:` or direct method calls `col1.write(...)`.
- **`border=True`**: Wraps each column in a card border.
- **`gap="small" | "medium" | "large"`**: Controls gutter spacing between columns.
"""
)

st.markdown("#### 📐 Formal Signature: `st.columns()`")
st.code(
    """st.columns(
    spec: int | Sequence[int | float],
    *,
    gap: "small" | "medium" | "large" = "small",
    vertical_alignment: "top" | "center" | "bottom" = "top",
    border: bool = False,
    width: "stretch" = "stretch"
) -> list[DeltaGenerator]""",
    language="python",
)

col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown("**💻 Python Source Code (3 Equal Columns with Borders)**")
    code_equal_cols = '''# Create 3 equal columns with borders:
col_a, col_b, col_c = st.columns(3, border=True)

with col_a:
    st.markdown("##### 🚀 Fast Inference")
    st.write("Model: `Llama-3.1-8B`")
    st.metric("Latency", "24 ms")

with col_b:
    st.markdown("##### 🧠 Deep Reasoning")
    st.write("Model: `Claude-3.5-Sonnet`")
    st.metric("MMLU", "88.7%")

with col_c:
    st.markdown("##### 💰 Low Cost")
    st.write("Model: `DeepSeek-V3`")
    st.metric("Price / 1M", "$0.27")'''
    st.code(code_equal_cols, language="python")

with col_eq2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    col_a, col_b, col_c = st.columns(3, border=True)
    with col_a:
        st.markdown("##### 🚀 Fast Inference")
        st.write("Model: `Llama-3.1-8B`")
        st.metric("Latency", "24 ms")
    with col_b:
        st.markdown("##### 🧠 Deep Reasoning")
        st.write("Model: `Claude-3.5-Sonnet`")
        st.metric("MMLU", "88.7%")
    with col_c:
        st.markdown("##### 💰 Low Cost")
        st.write("Model: `DeepSeek-V3`")
        st.metric("Price / 1M", "$0.27")

st.divider()

# ==============================================================================
# 5. Column Width Ratios
# ==============================================================================
st.header("5. Custom Column Width Ratios", divider="rainbow")
st.markdown(
    """
Passing a list or tuple of numbers to `spec` distributes width proportionally:
- **`[3, 1]`**: 75% primary content area and 25% sidebar control panel.
- **`[1, 2, 1]`**: Centered dominant card with flanking margins.
- **`[0.6, 0.4]`**: Fractional ratios representing 60% and 40%.
"""
)

col_rt1, col_rt2 = st.columns(2)

with col_rt1:
    st.markdown("**💻 Python Source Code (3:1 Ratio Layout)**")
    code_ratios = '''# 3:1 width ratio (Main Dashboard : Sidebar Controls)
main_col, control_col = st.columns([3, 1], gap="medium")

with main_col:
    st.markdown("##### 📊 Primary Telemetry Table (75% Width)")
    sample_df = pd.DataFrame({
        "Model": ["Claude-3.5", "GPT-4o", "DeepSeek-V3"],
        "Throughput (tok/s)": [95.0, 112.5, 140.2],
        "Health": ["Healthy", "Healthy", "Healthy"]
    })
    st.dataframe(sample_df, width="stretch", hide_index=True)

with control_col:
    st.markdown("##### 🎛️ Filters (25%)")
    st.selectbox("Region", ["Global", "US-East", "EU-West"], key="ratio_demo_region")
    st.slider("Min Tok/s", 50, 150, 80, key="ratio_demo_slider")'''
    st.code(code_ratios, language="python")

with col_rt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    main_col, control_col = st.columns([3, 1], gap="medium")
    with main_col:
        st.markdown("##### 📊 Primary Telemetry Table (75% Width)")
        sample_df = pd.DataFrame({
            "Model": ["Claude-3.5", "GPT-4o", "DeepSeek-V3"],
            "Throughput (tok/s)": [95.0, 112.5, 140.2],
            "Health": ["Healthy", "Healthy", "Healthy"]
        })
        st.dataframe(sample_df, width="stretch", hide_index=True, key="demo_ratio_df")
    with control_col:
        st.markdown("##### 🎛️ Filters (25%)")
        st.selectbox("Region", ["Global", "US-East", "EU-West"], key="ratio_demo_region")
        st.slider("Min Tok/s", 50, 150, 80, key="ratio_demo_slider")

st.divider()

# ==============================================================================
# 6. Nested Columns
# ==============================================================================
st.header("6. Nested Columns Architecture", divider="rainbow")
st.markdown(
    """
Columns can be cleanly nested inside other columns or container blocks:
- **Use Case**: Breaking a parent card into sub-metrics or dual input buttons.
- **Design Rule**: Limit nesting depth to 1–2 levels to prevent cramped elements on smaller screens.
"""
)

col_nest1, col_nest2 = st.columns(2)

with col_nest1:
    st.markdown("**💻 Python Source Code (Nested Columns)**")
    code_nested = '''# Outer column split:
left_card, right_card = st.columns(2, border=True)

with left_card:
    st.markdown("##### Cluster Alpha (H100 Node)")
    # Sub-columns nested inside left_card:
    sub1, sub2 = st.columns(2)
    with sub1:
        st.metric("Active Jobs", "14")
    with sub2:
        st.metric("VRAM Usage", "84%")

with right_card:
    st.markdown("##### Cluster Beta (A100 Node)")
    sub3, sub4 = st.columns(2)
    with sub3:
        st.metric("Active Jobs", "8")
    with sub4:
        st.metric("VRAM Usage", "52%")'''
    st.code(code_nested, language="python")

with col_nest2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    left_card, right_card = st.columns(2, border=True)
    with left_card:
        st.markdown("##### Cluster Alpha (H100 Node)")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.metric("Active Jobs", "14")
        with sub2:
            st.metric("VRAM Usage", "84%")
    with right_card:
        st.markdown("##### Cluster Beta (A100 Node)")
        sub3, sub4 = st.columns(2)
        with sub3:
            st.metric("Active Jobs", "8")
        with sub4:
            st.metric("VRAM Usage", "52%")

st.divider()

# ==============================================================================
# 7. Vertical Layouts & Alignment
# ==============================================================================
st.header("7. Vertical Layouts & Alignment (`vertical_alignment`)", divider="rainbow")
st.markdown(
    """
When placing inputs alongside buttons or text, mismatched widget heights can cause misaligned elements.
Streamlit provides `vertical_alignment` on `st.columns()`:
- **`"top"` (Default)**: Aligns element tops.
- **`"center"`**: Vertically centers elements across columns.
- **`"bottom"`**: Perfect for placing an **Action Button** next to a **Text Input** or **Selectbox**!
"""
)

col_va1, col_va2 = st.columns(2)

with col_va1:
    st.markdown("**💻 Python Source Code (`vertical_alignment='bottom'`)**")
    code_vert_align = '''# Vertically align action button with text input:
search_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")

with search_col:
    query = st.text_input("Search Model Repository", placeholder="e.g. meta-llama/Llama-3.1")

with btn_col:
    st.button("🔍 Search", width="stretch")'''
    st.code(code_vert_align, language="python")

with col_va2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    search_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
    with search_col:
        st.text_input("Search Model Repository", placeholder="e.g. meta-llama/Llama-3.1", key="va_demo_search")
    with btn_col:
        st.button("🔍 Search", width="stretch", key="va_demo_btn")
    st.caption("💡 *Notice how the 'Search' button perfectly aligns with the input field bottom baseline.*")

st.divider()

# ==============================================================================
# 8. Responsive Layout Concepts
# ==============================================================================
st.header("8. Responsive Layout Concepts & Best Practices", divider="rainbow")
st.markdown(
    """
Streamlit automatically adapts to desktop, tablet, and mobile screens:
- **Desktop Screens (> 768px)**: `st.columns()` render horizontally side-by-side.
- **Mobile Screens (≤ 768px)**: Columns **automatically collapse and stack vertically** to prevent cramped, unreadable text.
- **Best Practices**:
  1. Use `width="stretch"` on dataframes and buttons so elements fit fluidly across all screen sizes.
  2. Avoid high column counts (e.g., > 4 columns) unless displaying compact KPI metrics.
  3. Keep column ratio differentials reasonable (e.g. `[3, 1]` or `[2, 1]` rather than `[10, 1]`).
"""
)

# Responsive comparison showcase
st.markdown("##### 📱 Responsive Grid Prototype (3 Metrics + Filter Row):")
res_c1, res_c2, res_c3 = st.columns(3)
with res_c1:
    st.metric("Total API Requests", "2.4M", "+18%", border=True)
with res_c2:
    st.metric("Average Latency", "94 ms", "-12 ms", delta_color="inverse", border=True)
with res_c3:
    st.metric("Error Rate", "0.02%", "-0.01%", delta_color="inverse", border=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Layout Primitives Reference Guide", divider="rainbow")
st.markdown(
    r"""
| Primitive | Core Parameters | Primary Behavior & Use Case |
| :--- | :--- | :--- |
| **`st.columns(n)`** | `n: int`, `gap="small"`, `border=False` | Creates `n` equal-width horizontal columns |
| **`st.columns(spec)`**| `spec: [3, 1]`, `vertical_alignment="bottom"` | Weighted proportional columns with custom vertical alignment |
| **`st.container()`** | `border=True` | Visual card outline grouping related controls and metrics |
| **`st.container()`** | `height=200` | Creates a scrollable viewport container for logs or tables |
| **`layout="wide"`** | In `st.set_page_config()` | Expands page canvas to 100% browser viewport width |
| **`vertical_alignment`** | `"top" \| "center" \| "bottom"` | Eliminates widget vertical height misalignment in columns |
"""
)
