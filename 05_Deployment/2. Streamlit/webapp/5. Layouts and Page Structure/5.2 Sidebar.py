import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Sidebar Tutorial",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧭 Sidebar Architecture (`st.sidebar`) — Complete Guide")
st.markdown(
    """
The **Sidebar** (`st.sidebar`) is Streamlit's designated control center. 
It resides as a collapsible drawer on the left side of the screen, separating global filters, 
system navigation, authentication credentials, and user preferences from the main analytical canvas. 
This master guide covers API mechanics, widget placement, multi-view navigation, 
structured filter panels, and real-world dashboard control centers.
"""
)

st.divider()

# ==============================================================================
# 1. st.sidebar Core Architecture
# ==============================================================================
st.header("1. Core Architecture & `st.sidebar` Syntax", divider="rainbow")
st.markdown(
    """
`st.sidebar` is a pre-initialized `DeltaGenerator` container. You can add elements to it using two identical syntaxes:
1. **Dot Notation**: `st.sidebar.markdown(...)`, `st.sidebar.selectbox(...)`
2. **Context Manager (Recommended)**: `with st.sidebar:` block grouping all sidebar UI elements cleanly.
- **Collapsible State**: Users can collapse the sidebar using the chevron arrow `«` at the top left. 
  Control default state in `st.set_page_config(initial_sidebar_state="expanded" | "collapsed" | "auto")`.
"""
)

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**💻 Python Source Code (Dot Notation vs. Context Manager)**")
    code_syntax = '''# Approach A: Context Manager (Cleaner, Recommended)
with st.sidebar:
    st.title("🛡️ Enterprise AI Console")
    st.divider()
    env = st.selectbox("Environment", ["Production", "Staging", "Dev"])

# Approach B: Dot Notation (Equivalent)
st.sidebar.caption("Console Version: v2.5.0")
st.sidebar.button("Purge KV-Cache", key="btn_purge")'''
    st.code(code_syntax, language="python")

with col_sb2:
    st.markdown("**🖥️ Architecture Reference Table**")
    st.markdown(
        """
        | Property | Description |
        | :--- | :--- |
        | **Container Type** | Persistent left-aligned collapsible drawer |
        | **Initial State** | Configured in `st.set_page_config(initial_sidebar_state=...)` |
        | **Scope** | Global across all page reruns |
        | **Responsive Behavior** | Closes automatically on mobile screens to maximize canvas space |
        """
    )
    st.info("💡 *Look to the left of your browser window to see the live sidebar in action!*")

st.divider()

# ==============================================================================
# 2. Sidebar Titles and Text & 3. Sidebar Buttons
# ==============================================================================
st.header("2. Titles, Brand Text & 3. Action Buttons", divider="rainbow")
st.markdown(
    """
Brand your sidebar with titles, emojis, status badges, and action buttons:
- Use `st.sidebar.title()` and `st.sidebar.caption()` for hierarchy.
- Use `st.sidebar.button()` for global maintenance actions (e.g. cache reset, sync models, trigger jobs).
"""
)

col_tb1, col_tb2 = st.columns(2)

with col_tb1:
    st.markdown("**💻 Python Source Code**")
    code_text_btn = '''with st.sidebar:
    st.title("🤖 Fleet Controller")
    st.caption("Active Cluster: `us-east-1-h100-pool`")
    
    # Action buttons with callbacks or if conditions:
    if st.button("🔄 Sync Model Checkpoints", width="stretch", key="sb_sync_btn"):
        st.toast("Checkpoint registry synchronized with S3 bucket!")
        
    st.divider()'''
    st.code(code_text_btn, language="python")

with col_tb2:
    st.markdown("**🖥️ Rendered Interactive Simulation (In-Page Preview)**")
    with st.container(border=True):
        st.markdown("### 🤖 Fleet Controller *(Sidebar Mock)*")
        st.caption("Active Cluster: `us-east-1-h100-pool`")
        if st.button("🔄 Sync Model Checkpoints", width="stretch", key="mock_sync_btn"):
            st.toast("Checkpoint registry synchronized with S3 bucket!")
            st.success("Synchronized!")

st.divider()

# ==============================================================================
# 4. Sidebar Input Widgets & 5. Sidebar Filters
# ==============================================================================
st.header("4. Sidebar Input Widgets & 5. Reactive Global Data Filters", divider="rainbow")
st.markdown(
    """
Placing inputs in the sidebar isolates query controls from data visualization:
- When any sidebar widget changes, Streamlit automatically reruns the whole script top-to-bottom.
- Filter criteria collected in `st.sidebar` are immediately applied to slice DataFrames and charts on the main page.
"""
)

# Sample evaluation dataset
model_eval_data = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Llama-3.3-70B", "Mistral-Large-2", "Gemma-2-9B"],
    "Provider": ["Anthropic", "OpenAI", "DeepSeek", "Meta", "Mistral", "Google"],
    "Context (k)": [200, 128, 128, 128, 128, 8],
    "MMLU (%)": [88.7, 88.6, 88.5, 86.4, 84.0, 71.3],
    "Price / 1M ($)": [3.00, 2.50, 0.27, 0.70, 2.00, 0.25]
})

col_flt1, col_flt2 = st.columns(2)

with col_flt1:
    st.markdown("**💻 Python Source Code (Sidebar-Driven Data Slicing)**")
    code_filter_demo = '''# 1. Widgets declared inside sidebar:
with st.sidebar:
    st.header("🔍 Leaderboard Slicer")
    selected_providers = st.multiselect(
        "Filter by Provider",
        options=df["Provider"].unique(),
        default=df["Provider"].unique()
    )
    max_price = st.slider("Max Budget ($/1M)", 0.10, 5.00, 3.00, step=0.10)

# 2. Main page reads sidebar variables:
mask = (df["Provider"].isin(selected_providers)) & (df["Price / 1M ($)"] <= max_price)
filtered_df = df[mask]
st.dataframe(filtered_df, width="stretch", hide_index=True)'''
    st.code(code_filter_demo, language="python")

with col_flt2:
    st.markdown("**🖥️ Live Interactive Filter Simulation**")
    
    filt_col_a, filt_col_b = st.columns(2)
    with filt_col_a:
        demo_prov = st.multiselect(
            "Filter Provider",
            options=model_eval_data["Provider"].tolist(),
            default=["Anthropic", "OpenAI", "DeepSeek"],
            key="demo_sb_prov"
        )
    with filt_col_b:
        demo_cost = st.slider("Max Budget ($/1M)", 0.10, 5.00, 3.00, step=0.10, key="demo_sb_cost")

    filtered_models = model_eval_data[
        (model_eval_data["Provider"].isin(demo_prov)) & 
        (model_eval_data["Price / 1M ($)"] <= demo_cost)
    ]
    st.dataframe(filtered_models, width="stretch", hide_index=True, key="demo_sb_filtered_df")
    st.caption(f"Showing **{len(filtered_models)} / {len(model_eval_data)}** models based on filter settings.")

st.divider()

# ==============================================================================
# 6. Sidebar Navigation (Multi-View Routing Pattern)
# ==============================================================================
st.header("6. Sidebar Navigation & Page Routing", divider="rainbow")
st.markdown(
    """
Before creating multi-file apps with `st.navigation`, the canonical single-file routing pattern 
uses `st.sidebar.radio()` or `st.sidebar.selectbox()` as a **view router**:
- The user selects a section in the sidebar.
- Conditional `if/elif/else` branches on the main canvas render different UI views.
"""
)

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    st.markdown("**💻 Python Source Code (Sidebar Router Pattern)**")
    code_nav = '''# Define navigation in sidebar:
with st.sidebar:
    selected_view = st.radio(
        "System Navigation",
        ["📊 Cluster Overview", "⚡ Inference Latency", "⚙️ Hardware Config"],
        key="app_nav_radio"
    )

# Main page conditionally renders based on router selection:
if selected_view == "📊 Cluster Overview":
    st.markdown("### Cluster Health Overview")
    st.metric("Total Nodes", "128 Active")
elif selected_view == "⚡ Inference Latency":
    st.markdown("### Latency Telemetry")
    st.metric("p99 Latency", "48 ms")
else:
    st.markdown("### Cluster Hardware Settings")
    st.write("Configuring NVIDIA H100 interconnects.")'''
    st.code(code_nav, language="python")

with col_nav2:
    st.markdown("**🖥️ Live Interactive Router Demo**")
    
    mock_nav = st.radio(
        "Simulate Navigation Choice:",
        ["📊 Cluster Overview", "⚡ Inference Latency", "⚙️ Hardware Config"],
        horizontal=True,
        key="demo_sim_nav"
    )
    
    with st.container(border=True):
        if mock_nav == "📊 Cluster Overview":
            st.markdown("##### 📊 View: Cluster Overview")
            st.metric("Active Nodes", "128 H100s", "+4 Added")
        elif mock_nav == "⚡ Inference Latency":
            st.markdown("##### ⚡ View: Latency Telemetry")
            st.metric("p99 TTFT Latency", "48 ms", "-12 ms", delta_color="inverse")
        else:
            st.markdown("##### ⚙️ View: Hardware Settings")
            st.info("NVIDIA InfiniBand Quantum-2 fabric configured at 400 Gb/s.")

st.divider()

# ==============================================================================
# 7. Building Structured Filter Panels (Expanders & Resets)
# ==============================================================================
st.header("7. Structured Filter Panels with Collapsible Expanders & Reset Buttons", divider="rainbow")
st.markdown(
    """
Complex dashboards often have dozens of filter parameters. Keep the sidebar clutter-free by:
1. **Collapsible Sections**: Grouping advanced filters inside `st.sidebar.expander()`.
2. **Session State Reset Pattern**: Providing a "Reset All Filters" button that clears `st.session_state` keys and reruns the app.
"""
)

col_fp1, col_fp2 = st.columns(2)

with col_fp1:
    st.markdown("**💻 Python Source Code (Filter Panel Architecture)**")
    code_filter_panel = '''def reset_filters():
    st.session_state["filter_model"] = "All"
    st.session_state["filter_temp"] = 0.70
    st.session_state["filter_stream"] = True

with st.sidebar:
    st.subheader("🎛️ Filter Console")
    
    # 1. Primary Filter
    st.selectbox("Model Family", ["All", "Claude", "GPT", "Llama"], key="filter_model")
    
    # 2. Collapsible Advanced Hyperparameters
    with st.expander("🛠️ Advanced Parameters", expanded=False):
        st.slider("Sampling Temperature", 0.0, 1.0, key="filter_temp")
        st.toggle("Stream Tokens", key="filter_stream")
        
    # 3. Reset Button
    st.button("↺ Reset All Filters", on_click=reset_filters)'''
    st.code(code_filter_panel, language="python")

with col_fp2:
    st.markdown("**🖥️ Live Interactive Filter Panel Simulation**")
    
    if "fp_model" not in st.session_state:
        st.session_state["fp_model"] = "All"
    if "fp_temp" not in st.session_state:
        st.session_state["fp_temp"] = 0.70

    def do_reset_fp():
        st.session_state["fp_model"] = "All"
        st.session_state["fp_temp"] = 0.70

    with st.container(border=True):
        st.markdown("##### 🎛️ Control Panel")
        st.selectbox("Model Family", ["All", "Claude", "GPT", "Llama", "DeepSeek"], key="fp_model")
        with st.expander("🛠️ Advanced Hyperparameters", expanded=False):
            st.slider("Sampling Temperature", 0.0, 1.0, key="fp_temp")
            st.toggle("Streaming Output", value=True, key="fp_stream")
        st.button("↺ Reset All Filters", on_click=do_reset_fp, key="btn_fp_reset")
        st.caption(f"Active Settings: Model = **{st.session_state['fp_model']}** | Temp = **{st.session_state['fp_temp']}**")

st.divider()

# ==============================================================================
# 8. Creating a Production Dashboard Sidebar
# ==============================================================================
st.header("8. Real-World Production AI Operations Sidebar", divider="rainbow")
st.markdown(
    """
Notice the **live sidebar** on the left of this application! It has been populated with a production-grade 
control panel implementing all the concepts above:
- Brand identity header and environment badge
- Primary routing navigation
- Multi-dimensional filter controls
- Live cache purging and sync action buttons
"""
)

# Actually populate the real left sidebar of the application:
with st.sidebar:
    st.title("🧭 System Control Center")
    st.badge("Production Cluster", icon="🟢", color="green")
    st.divider()

    st.markdown("##### 🧭 View Navigation")
    live_nav = st.radio(
        "Go to View:",
        ["Cluster Fleet", "Cost Analytics", "Audit Logs"],
        key="real_sb_nav",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("##### 🎛️ Global Fleet Filters")
    st.selectbox("Hardware Type", ["All GPUs", "NVIDIA H100", "NVIDIA A100", "L40S"], key="real_sb_hw")
    st.slider("Min Utilization (%)", 0, 100, 20, key="real_sb_util")

    with st.expander("⚙️ Advanced Maintenance", expanded=False):
        st.checkbox("Enable Canary Routing", value=True, key="real_sb_canary")
        st.checkbox("Verbose Telemetry", value=False, key="real_sb_verbose")

    st.divider()
    if st.button("🧹 Flush Redis Cache", width="stretch", key="real_sb_flush"):
        st.toast("Redis inference cache purged successfully!")

st.success("✅ The live sidebar on the left has been populated with interactive controls. Try opening and adjusting it!")

# Display current selection from real sidebar
st.markdown(f"#### Active Main View: **{live_nav}**")
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Serving Endpoints", "24 Active", border=True)
with kpi2:
    st.metric("Total P95 Latency", "52 ms", "-8 ms", delta_color="inverse", border=True)
with kpi3:
    st.metric("Cluster Load", f"{st.session_state.get('real_sb_util', 20)}% Min Threshold", border=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Sidebar API Reference Guide", divider="rainbow")
st.markdown(
    """
| Operation | Recommended Syntax | Description |
| :--- | :--- | :--- |
| **Context Block** | `with st.sidebar:` | Recommended pattern for grouping all sidebar elements |
| **Dot Notation** | `st.sidebar.selectbox(...)` | Direct placement of individual elements |
| **Initial State** | `st.set_page_config(initial_sidebar_state="expanded")` | `"expanded"`, `"collapsed"`, or `"auto"` |
| **Brand Logo** | `st.logo(image_url)` | Places official organization logo in top sidebar header |
| **Page Routing** | `view = st.sidebar.radio("Nav", options)` | Classical single-file router pattern |
| **Collapsible Sections** | `with st.sidebar.expander("Advanced"):` | Organizes complex filter sets without clutter |
| **Filter Reset** | `st.sidebar.button("Reset", on_click=reset_fn)` | Resets session state filter keys and reruns app |
"""
)
