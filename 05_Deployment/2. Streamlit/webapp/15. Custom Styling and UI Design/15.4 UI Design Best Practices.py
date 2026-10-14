import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="15.4 UI Design Best Practices",
    page_icon="✨",
    layout="wide"
)

st.title("15.4 UI Design Best Practices in Streamlit")
st.caption("Human-Computer Interaction (HCI) standards, visual hierarchy, 8pt spatial grid, accessible contrast, progressive disclosure, and executive dashboard design.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Principles of High-Performance Streamlit UI/UX

Designing production-grade Streamlit applications requires balancing simplicity with visual sophistication.
Enterprise user interfaces adhere to four core HCI pillars:
1. **Perceptual Hierarchy**: Directing user attention through size, weight, contrast, and layout position.
2. **Predictable Mental Models**: Standardizing navigation, error states, and feedback loops across pages.
3. **Cognitive Load Reduction**: Applying Miller's Law ($7 \pm 2$ chunks) and progressive disclosure (`st.expander`, `@st.dialog`).
4. **Spatial Cadence**: Aligning margins, padding, and container boundaries to a consistent 8pt design grid.

""")

st.graphviz_chart(
    r'''
    digraph UIHierarchy {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        kpi [label="1. Executive KPI Summary\nst.columns([1, 1, 1]) -> st.metric() cards", fillcolor="#DBEAFE", color="#1D4ED8"];
        charts [label="2. Interactive Visual Analytics\nst.tabs() -> [Model Charts, Feature Drift, Loss]", fillcolor="#FEF3C7", color="#D97706"];
        audit [label="3. Auditing & Raw Inspection\nst.dataframe() with column config & CSV export", fillcolor="#D1FAE5", color="#059669"];

        kpi -> charts [label="High-level health"];
        charts -> audit [label="Drilldown details"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Spacing & 8pt Grid",
    "2. Visual Hierarchy",
    "3. Readable Typography",
    "4. Color & Contrast",
    "5. Responsive Layouts",
    "6. Clear Error Feedback",
    "7. Consistent Navigation",
    "8. Loading States",
    "9. Progressive Disclosure",
    "10. Capstone Dashboard"
])

# =============================================================================
# 1. SPACING & 8PT GRID
# =============================================================================
with tabs[0]:
    st.header("1. Consistent Spacing & The 8pt Spatial Cadence")
    st.markdown(r"""
    A foundational design rule is the **8pt Grid System**: all padding, margins, gaps, and heights 
    are multiples of 8 (8px, 16px, 24px, 32px, 48px). This creates visual rhythm and eliminates haphazard gaps.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Recommended 8pt Spatial Grid:
# 8px  -> Micro gap (between label and input)
# 16px -> Standard container padding
# 24px -> Gap between major columns
# 32px -> Section break

with st.container(border=True):
    st.subheader("Model Cluster Alpha")
    st.caption("Active in us-east-1 region")
    
    # Clean 16px column layout
    c1, c2 = st.columns(2)
    c1.metric("Memory In-Use", "48.2 GB", "68% of 64GB")
    c2.metric("GPU Temp", "62 °C", "Nominal")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.subheader("Model Cluster Alpha")
            st.caption("Active in us-east-1 region")

            c1, c2 = st.columns(2)
            c1.metric("Memory In-Use", "48.2 GB", "68% of 64GB")
            c2.metric("GPU Temp", "62 °C", "Nominal")
            st.caption("✅ Containerized grouping with structured 16px internal padding.")

# =============================================================================
# 2. VISUAL HIERARCHY
# =============================================================================
with tabs[1]:
    st.header("2. Establishing Visual Hierarchy (F-Pattern Scanning)")
    st.markdown(r"""
    Users scan analytical screens in an **F-pattern**: top horizontal bar first, followed by a secondary horizontal sweep, 
    and then a vertical glance down the left side. Structure headers and metrics accordingly.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Hierarchy Level 1: Page Title (Application Scope)
st.title("Model Evaluation Leaderboard")

# Hierarchy Level 2: Section Header (Actionable Area)
st.header("Benchmark Performance")

# Hierarchy Level 3: Card Subheader (Component Scope)
st.subheader("Top Performing Checkpoint")

# Hierarchy Level 4: Supporting Metadata
st.caption("Evaluated across 1,200 unseen holdout reasoning tasks")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown("### Visual Typographic Scale")
            st.markdown("# Heading 1: Top-Level Context (`st.title`)")
            st.markdown("## Heading 2: Section Divider (`st.header`)")
            st.markdown("### Heading 3: Widget / Card Title (`st.subheader`)")
            st.markdown("Paragraph body copy: Explaining key operational context.")
            st.caption("Caption copy: Timestamps, sample sizes, and source citations (`st.caption`).")

# =============================================================================
# 3. READABLE TYPOGRAPHY
# =============================================================================
with tabs[2]:
    st.header("3. Readable Typography & Line-Length Ergonomics")
    st.markdown(r"""
    - **Line Length**: Keep paragraphs between 50 to 75 characters per line for optimal readability.
    - **Tabular Numbers**: Use monospace fonts or formatted numbers for metrics and dollar amounts to align digits vertically.
    - **Avoid ALL CAPS**: Limit uppercase text to small status pills and acronyms.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Aligning numerical data in tabular layout
df_metrics = pd.DataFrame({
    "Endpoint": ["/v1/chat/completions", "/v1/embeddings", "/v1/rerank"],
    "Requests": [1450200, 389400, 112000],
    "P95_Latency": ["142.5 ms", "18.2 ms", "34.1 ms"],
    "Error_Rate": ["0.012%", "0.000%", "0.004%"]
})

st.dataframe(df_metrics, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_metrics = pd.DataFrame({
                "Endpoint": ["/v1/chat/completions", "/v1/embeddings", "/v1/rerank"],
                "Requests": [1450200, 389400, 112000],
                "P95_Latency": ["142.5 ms", "18.2 ms", "34.1 ms"],
                "Error_Rate": ["0.012%", "0.000%", "0.004%"]
            })
            st.caption("Clean Tabular Typography:")
            st.dataframe(df_metrics, width="stretch", hide_index=True)

# =============================================================================
# 4. COLOR & CONTRAST
# =============================================================================
with tabs[3]:
    st.header("4. Accessible Color Contrast & Colorblind Safety")
    st.markdown(r"""
    Never rely on color alone to convey critical state. 
    Always pair colors with explicit icons, labels, or badges to maintain accessibility for colorblind users.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# ❌ BAD: Solely relying on red / green without text or icon
# st.write("<span style='color:red;'>Failed</span>")

# ✅ GOOD: Multi-channel indicators (Color + Icon + Text)
col_a, col_b, col_c = st.columns(3)
col_a.success("✅ Operational (Healthy)")
col_b.warning("⚠️ High Load (84% CPU)")
col_c.error("🚨 Outage (Cluster Down)")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Multi-Channel Accessible State Indicators:")
            col_a, col_b, col_c = st.columns(3)
            col_a.success("✅ Operational")
            col_b.warning("⚠️ High Latency")
            col_c.error("🚨 Outage")
            st.info("💡 Combining color hues with recognizable emojis ensures accessibility across red-green color deficiencies.")

# =============================================================================
# 5. RESPONSIVE LAYOUTS
# =============================================================================
with tabs[4]:
    st.header("5. Responsive Layouts & Fluid Column Wrapping")
    st.markdown(r"""
    Streamlit columns automatically stack vertically on screens smaller than 768px.
    Avoid designing rows with more than 4 or 5 narrow columns, as they compress severely on tablets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Recommended: 2-column or 3-column balanced grids
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("Inference Throughput")
    st.line_chart([10, 15, 12, 18, 24, 22, 30], height=200)

with col_side:
    st.subheader("Summary")
    st.metric("Peak Load", "30 req/s")
    st.metric("P95", "180 ms")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            col_main, col_side = st.columns([3, 1])
            with col_main:
                st.caption("Inference Throughput (3/4 width)")
                st.line_chart([10, 15, 12, 18, 24, 22, 30], height=180)
            with col_side:
                st.caption("Summary (1/4 width)")
                st.metric("Peak Load", "30 req/s")
                st.metric("P95", "180 ms")

# =============================================================================
# 6. CLEAR ERROR FEEDBACK
# =============================================================================
with tabs[5]:
    st.header("6. Actionable Error Messages vs Raw Tracebacks")
    st.markdown(r"""
    A professional UI never exposes unformatted Python tracebacks to end users.
    Actionable error messages explain **What Happened**, **Why**, and **How to Fix It**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Anti-pattern: Letting unhandled exceptions crash the page
# raise ConnectionError("Connection refused by peer")

# Best Practice: Structured, actionable remediation
try:
    # Simulated failing database query
    raise TimeoutError("Database connection timed out after 5.0 seconds")
except Exception as e:
    st.error("🚨 Unable to load Model Registry records.")
    with st.expander("Troubleshooting Guidance & Actions"):
        st.markdown("""
        - **Cause**: Upstream PostgreSQL primary instance is currently restarting.
        - **Action**: Check status page at `status.internal.ai` or verify VPN connectivity.
        """)
        if st.button("🔄 Retry Connection"):
            st.rerun()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.error("🚨 Unable to load Model Registry records.")
            with st.expander("Troubleshooting Guidance & Actions"):
                st.markdown("""
                - **Cause**: Upstream PostgreSQL primary instance is currently restarting.
                - **Action**: Check status page at `status.internal.ai` or verify VPN connectivity.
                """)
                if st.button("🔄 Retry Connection", key="err_retry_btn"):
                    st.rerun()

# =============================================================================
# 7. CONSISTENT NAVIGATION
# =============================================================================
with tabs[6]:
    st.header("7. Consistent Navigation & Breadcrumb Anchors")
    st.markdown(r"""
    Maintain a consistent orientation model across pages.
    Users should always know **where they are**, **how they got there**, and **how to go back**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Breadcrumb navigation pattern
st.markdown("`Platform` / `Governance` / **Audit Logs**")

# Section Navigation Pills
selected_subview = st.pills(
    "Select Subview",
    ["Overview", "Access Logs", "Anomalies", "Policies"],
    default="Overview"
)

st.caption(f"Currently viewing: {selected_subview}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown("`Platform` / `Governance` / **Audit Logs**")
            subview = st.pills(
                "Select Subview",
                ["Overview", "Access Logs", "Anomalies", "Policies"],
                default="Overview",
                key="pills_subview"
            )
            st.caption(f"Currently viewing: **{subview}**")

# =============================================================================
# 8. LOADING STATES
# =============================================================================
with tabs[7]:
    st.header("8. Loading States (`st.spinner` & `st.status`)")
    st.markdown(r"""
    Keep users informed during asynchronous or long-running computational steps.
    Use `st.spinner` for fast operations (<3s) and `st.status` for multi-stage pipelines (>3s).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import time

if st.button("Execute Pipeline Benchmark"):
    with st.status("Executing Multi-Stage Benchmark...", expanded=True) as status:
        st.write("1. Pulling weights from Hugging Face cache...")
        time.sleep(0.5)
        st.write("2. Running 50-shot MMLU reasoning suite...")
        time.sleep(0.5)
        st.write("3. Computing perplexity quantiles...")
        time.sleep(0.5)
        status.update(label="Benchmark Complete!", state="complete", expanded=False)
    st.success("All evaluations passed with score: 0.884")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if st.button("Execute Pipeline Benchmark", key="run_bench_demo"):
                with st.status("Executing Multi-Stage Benchmark...", expanded=True) as status_box:
                    st.write("1. Pulling weights from Hugging Face cache...")
                    time.sleep(0.2)
                    st.write("2. Running 50-shot MMLU reasoning suite...")
                    time.sleep(0.2)
                    st.write("3. Computing perplexity quantiles...")
                    time.sleep(0.2)
                    status_box.update(label="Benchmark Complete!", state="complete", expanded=False)
                st.success("All evaluations passed with score: 0.884")
            else:
                st.info("Click the button above to observe multi-stage `st.status` progressive loading.")

# =============================================================================
# 9. PROGRESSIVE DISCLOSURE
# =============================================================================
with tabs[8]:
    st.header("9. Progressive Disclosure (Avoiding Cognitive Clutter)")
    st.markdown(r"""
    **Progressive Disclosure** defers advanced features and raw traces until the user explicitly requests them.
    Use `st.expander` for optional technical logs and `@st.dialog` for focused sub-tasks.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Primary executive summary (Visible by default)
st.metric("Model Health", "99.8%", "+0.2% this week")

# Secondary details tucked inside expander
with st.expander("🔍 View Raw Telemetry Logs & Worker Traces"):
    st.caption("Container worker logs for pod `m-worker-08b`:")
    st.code("""
    [2024-10-12 14:02:11] INFO: Worker initialized on port 8080
    [2024-10-12 14:02:12] DEBUG: Loaded weights into VRAM in 1.4s
    [2024-10-12 14:02:15] INFO: Serving warm requests
    """)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.metric("Model Health", "99.8%", "+0.2% this week")
            with st.expander("🔍 View Raw Telemetry Logs & Worker Traces"):
                st.caption("Container worker logs for pod `m-worker-08b`:")
                st.code(
                    "[2024-10-12 14:02:11] INFO: Worker initialized\n"
                    "[2024-10-12 14:02:12] DEBUG: VRAM ready in 1.4s\n"
                    "[2024-10-12 14:02:15] INFO: Serving warm requests"
                )

# =============================================================================
# 10. CAPSTONE DASHBOARD
# =============================================================================
with tabs[9]:
    st.header("10. Designing Professional Enterprise Dashboards")
    st.markdown(r"""
    A unified capstone assembling all design principles:
    - **Header**: High-level title and live environment badge.
    - **KPI Ribbon**: Top-line summary metrics with context deltas.
    - **Main Canvas**: Two-column balanced visual comparison.
    - **Footer**: Progressive disclosure expander for raw telemetry.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

# 1. Header with Status Pill
head_l, head_r = st.columns([4, 1])
head_l.title("Enterprise AI Operations Console")
head_r.success("🟢 All Systems Normal")

# 2. KPI Summary Ribbon
k1, k2, k3, k4 = st.columns(4)
k1.metric("Daily Active Queries", "1.42 M", "+12.4%")
k2.metric("Median Response Time", "118 ms", "-8 ms")
k3.metric("Cache Hit Ratio", "84.2%", "+2.1%")
k4.metric("Cost per 1k Tokens", "$0.0014", "-$0.0002")

# 3. Main Analytical Canvas
with st.container(border=True):
    st.subheader("Hourly Request Volume by Region")
    chart_df = pd.DataFrame(
        np.random.randint(100, 500, (12, 3)),
        columns=["Americas", "EMEA", "APAC"]
    )
    st.area_chart(chart_df, height=200)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            head_l, head_r = st.columns([3, 1])
            head_l.subheader("AI Operations Console")
            head_r.success("🟢 Systems Normal")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Daily Queries", "1.42 M", "+12%")
            k2.metric("Latency", "118 ms", "-8 ms")
            k3.metric("Cache Hit", "84.2%", "+2.1%")
            k4.metric("Cost / 1k", "$0.0014", "-$0.0002")

            with st.container(border=True):
                st.caption("Hourly Request Volume by Region:")
                np.random.seed(42)
                chart_df = pd.DataFrame(
                    np.random.randint(100, 500, (12, 3)),
                    columns=["Americas", "EMEA", "APAC"]
                )
                st.area_chart(chart_df, height=180)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### UI Design Best Practices Cheatsheet

| Principle | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Grid Alignment** | Arbitrary heights and margins | Standardize on 8pt increments (8px, 16px, 24px, 32px). |
| **Color Semantics** | Using red for non-critical alerts | Reserve red exclusively for blocking errors; use yellow/orange for warnings and blue/indigo for informational notices. |
| **Error Handling** | Raw Python exception stack traces | Provide clear, actionable remediation messages with retry buttons. |
| **Long Operations** | Freezing the browser without feedback | Use `st.spinner()` for short waits and `st.status()` for multi-step pipelines. |
| **Cognitive Load** | Displaying 50 widgets on a single screen | Defer secondary controls to `st.sidebar`, `st.expander`, or `@st.dialog`. |
""")
