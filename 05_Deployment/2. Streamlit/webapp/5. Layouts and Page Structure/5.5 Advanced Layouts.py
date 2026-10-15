import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Advanced Layouts Tutorial",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ Advanced Layouts & Modal Containers — Complete Guide")
st.markdown(
    """
Advanced Streamlit layouts elevate standard analytical scripts into modern web applications. 
This master guide covers advanced multi-column nesting, bordered card containers (`st.container(border=True)`), 
horizontal flexbox layouts (`st.container(horizontal=True)`), true modal dialogs with **`@st.dialog`**, 
floating overlays with **`st.popover`**, fixed-position UI concepts, and best practices for avoiding deep layout nesting.
"""
)

st.divider()

# ==============================================================================
# 1. st.columns() with Nested Components
# ==============================================================================
st.header("1. Multi-Column Grids with Nested Components", divider="rainbow")
st.markdown(
    """
By nesting containers, metrics, and progress bars inside `st.columns()`, you can construct rich composite cards:
- Slices horizontal real estate into modular panels.
- Each column hosts its own independent layout hierarchy.
"""
)

col_nest1, col_nest2 = st.columns(2)

with col_nest1:
    st.markdown("**💻 Python Source Code**")
    code_nested_grid = '''# Composite card grid with nested components:
col_card1, col_card2 = st.columns(2)

with col_card1:
    with st.container(border=True):
        st.markdown("##### 🚀 Fast Inference Tier")
        st.caption("Active Model: `Llama-3.1-8B-Instruct`")
        sub_c1, sub_c2 = st.columns(2)
        sub_c1.metric("Latency", "24 ms", "-4 ms", delta_color="inverse")
        sub_c2.metric("Throughput", "1,850 t/s", "+120")
        st.progress(65, text="GPU Memory: 65% (A100)")

with col_card2:
    with st.container(border=True):
        st.markdown("##### 🧠 Deep Reasoning Tier")
        st.caption("Active Model: `Claude-3.5-Sonnet`")
        sub_c3, sub_c4 = st.columns(2)
        sub_c3.metric("Latency", "180 ms", "+12 ms", delta_color="inverse")
        sub_c4.metric("MMLU", "88.7%", "+1.2%")
        st.progress(92, text="Fleet Allocation: 92%")'''
    st.code(code_nested_grid, language="python")

with col_nest2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    col_card1, col_card2 = st.columns(2)
    with col_card1:
        with st.container(border=True):
            st.markdown("##### 🚀 Fast Inference Tier")
            st.caption("Active Model: `Llama-3.1-8B-Instruct`")
            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("Latency", "24 ms", "-4 ms", delta_color="inverse")
            sub_c2.metric("Throughput", "1,850 t/s", "+120")
            st.progress(65, text="GPU Memory: 65% (A100)")
    with col_card2:
        with st.container(border=True):
            st.markdown("##### 🧠 Deep Reasoning Tier")
            st.caption("Active Model: `Claude-3.5-Sonnet`")
            sub_c3, sub_c4 = st.columns(2)
            sub_c3.metric("Latency", "180 ms", "+12 ms", delta_color="inverse")
            sub_c4.metric("MMLU", "88.7%", "+1.2%")
            st.progress(92, text="Fleet Allocation: 92%")

st.divider()

# ==============================================================================
# 2. st.container(border=True) & 3. st.container(horizontal=True)
# ==============================================================================
st.header("2. Bordered Cards (`border=True`) & 3. Horizontal Flex (`horizontal=True`)", divider="rainbow")
st.markdown(
    """
- **`st.container(border=True)`**: Wraps children in an elevated card boundary. Essential for grouping related forms and scorecards.
- **`st.container(horizontal=True)`**: Streamlit 1.38+ introduces horizontal flexbox containers! Elements placed inside flow **horizontally from left to right**, wrapping naturally without manual column math.
"""
)

col_cnt1, col_cnt2 = st.columns(2)

with col_cnt1:
    st.markdown("**💻 Python Source Code**")
    code_containers = '''# 1. Bordered Card Container
with st.container(border=True):
    st.markdown("##### 🛡️ Production Firewall & Safety Guardrails")
    st.write("Real-time toxic prompt filtering and PII masking active.")

# 2. Horizontal Flexbox Container (Button toolbar)
with st.container(horizontal=True):
    st.button("🟢 Start Node", key="btn_start")
    st.button("🟡 Pause Ingestion", key="btn_pause")
    st.button("🔴 Terminate Fleet", key="btn_stop")
    st.button("🔄 Sync Checkpoints", key="btn_sync")'''
    st.code(code_containers, language="python")

with col_cnt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        st.markdown("##### 🛡️ Production Firewall & Safety Guardrails")
        st.write("Real-time toxic prompt filtering and PII masking active across all gateway routes.")
    
    st.caption("Horizontal Flexbox Container (`horizontal=True`):")
    with st.container(horizontal=True):
        st.button("🟢 Start Node", key="demo_btn_start")
        st.button("🟡 Pause Ingestion", key="demo_btn_pause")
        st.button("🔴 Terminate Fleet", key="demo_btn_stop")
        st.button("🔄 Sync Checkpoints", key="demo_btn_sync")

st.divider()

# ==============================================================================
# 4. Modal Dialogs (@st.dialog)
# ==============================================================================
st.header("4. True Modal Dialogs (`@st.dialog`)", divider="rainbow")
st.markdown(
    """
The **`@st.dialog`** decorator transforms a Python function into an interactive modal window:
- **Modal Focus**: Greys out the background canvas and focuses user attention on the dialog.
- **Fragment Reruns**: Interactions inside the modal rerun **only the dialog function**, leaving the background page untouched!
- **Closing the Modal**: Calling `st.rerun()` inside the dialog dismisses the modal and updates the parent application.
- **Width Options**: `width="small"` or `width="large"`.
"""
)

# Define modal dialog function
@st.dialog("🚀 Deploy New Foundation Model Checkpoint", width="small")
def deploy_model_dialog():
    st.write("Configure deployment spec for the new cluster endpoint:")
    model_id = st.text_input("Hugging Face Model ID", value="meta-llama/Llama-3.3-70B-Instruct")
    replica_count = st.number_input("GPU Replicas (H100)", min_value=1, max_value=32, value=4)
    confirm = st.button("Confirm & Launch Deployment", type="primary", key="btn_confirm_deploy")
    if confirm:
        st.session_state["deployed_model_notice"] = f"Successfully queued `{model_id}` on {replica_count}x H100 replicas!"
        st.rerun()

col_dlg1, col_dlg2 = st.columns(2)

with col_dlg1:
    st.markdown("**💻 Python Source Code (@st.dialog Decorator)**")
    code_dialog = '''# 1. Define dialog function with @st.dialog:
@st.dialog("🚀 Deploy New Model", width="small")
def deploy_dialog():
    st.write("Configure deployment spec:")
    model_id = st.text_input("Model ID", value="meta-llama/Llama-3.3-70B")
    replicas = st.number_input("Replicas", 1, 32, 4)
    
    if st.button("Confirm Deployment", type="primary"):
        st.session_state["notice"] = f"Deployed {model_id} on {replicas} GPUs!"
        st.rerun()  # Closes modal and updates main app!

# 2. Trigger dialog from button click:
if st.button("Open Deployment Dialog"):
    deploy_dialog()'''
    st.code(code_dialog, language="python")

with col_dlg2:
    st.markdown("**🖥️ Live Interactive Dialog Demo**")
    
    if "deployed_model_notice" in st.session_state:
        st.success(st.session_state["deployed_model_notice"])
        if st.button("Dismiss Notice", key="btn_clear_notice"):
            del st.session_state["deployed_model_notice"]
            st.rerun()

    if st.button("🚀 Open Model Deployment Modal Dialog", type="primary", key="btn_open_dialog"):
        deploy_model_dialog()
    st.caption("💡 *Click the button above to test the true modal dialog with background dimming.*")

st.divider()

# ==============================================================================
# 5. st.popover() & 6. Floating & Fixed-Position UI Concepts
# ==============================================================================
st.header("5. Floating Popovers (`st.popover`) & 6. Fixed-Position UI", divider="rainbow")
st.markdown(
    """
Streamlit offers multiple floating and fixed-position primitives:
1. **`st.popover()`**: Lightweight floating dropdown attached directly to a button.
2. **`st.toast()`**: Ephemeral floating notification at the bottom-right corner of the browser viewport.
3. **`st.chat_input()`**: Floating input container pinned to the bottom of the viewport.
"""
)

col_flt1, col_flt2 = st.columns(2)

with col_flt1:
    st.markdown("**💻 Python Source Code**")
    code_floating = '''# 1. Floating Popover settings modal:
with st.popover("⚙️ Instant Cluster Settings", icon="🎛️"):
    st.write("Live Sampling Knobs:")
    st.slider("Max Tokens", 256, 8192, 4096)
    st.toggle("Streaming Output", value=True)

# 2. Fixed-position bottom-right Toast notification:
if st.button("Trigger Floating Toast"):
    st.toast("Checkpoint saved to AWS S3 bucket!", icon="💾")'''
    st.code(code_floating, language="python")

with col_flt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        with st.popover("⚙️ Cluster Settings", icon="🎛️"):
            st.markdown("##### Live Sampling Knobs")
            st.slider("Max Tokens", 256, 8192, 4096, key="pop_demo_tokens")
            st.toggle("Streaming Output", value=True, key="pop_demo_stream")
            st.button("Save Config", type="primary", key="pop_demo_save")
    with col_p2:
        if st.button("📢 Trigger Floating Toast", key="btn_demo_toast"):
            st.toast("Checkpoint saved to AWS S3 bucket!", icon="💾")
    st.caption("💡 *Popover opens a floating window; Toast shows a notification in the bottom right corner.*")

st.divider()

# ==============================================================================
# 7. Dashboard Grid Layouts & 8. Avoiding Deeply Nested Layouts
# ==============================================================================
st.header("7. Production Dashboard Grid & 8. Avoiding Deep Nesting Anti-Patterns", divider="rainbow")
st.markdown(
    """
### ⚠️ The Deep Nesting Anti-Pattern
Nesting columns 3+ levels deep (`columns -> columns -> columns`) results in:
- Horizontally compressed text on tablet and laptop screens.
- Completely unreadable layouts on mobile screens (where columns collapse unpredictably).

### ✅ Best Practice Architectural Hierarchy (Max 2 Levels)
1. **Row 1**: High-level KPI Scorecard (`st.columns(4)` with `st.metric(border=True)`).
2. **Row 2**: Main Content split via 3:1 ratio (`st.columns([3, 1])`).
3. **Internal Organization**: Use **`st.tabs()`** and **`st.container(border=True)`** rather than nested columns!
"""
)

# Master Production Grid Demonstration
st.markdown("##### 🏛️ Master Production AI Operations Console")

# Row 1: KPI Scorecards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Active Model Instances", "42 Nodes", "+4 Nodes", border=True)
k2.metric("Mean Latency", "38 ms", "-6 ms", delta_color="inverse", border=True)
k3.metric("Fleet GPU VRAM Load", "78.4%", "+3.2%", border=True)
k4.metric("Daily Cloud Spend", "$1,420", "-$110", delta_color="inverse", border=True)

# Row 2: 3:1 Grid Layout
main_view, side_panel = st.columns([3, 1], gap="medium")

with main_view:
    with st.container(border=True):
        st.markdown("##### 📊 Model Performance Telemetry")
        demo_telemetry = pd.DataFrame({
            "Endpoint": ["ep-llama3-70b", "ep-claude35-sonnet", "ep-deepseek-v3"],
            "Region": ["us-east-1", "us-east-1", "eu-west-1"],
            "Throughput (tok/s)": [112.5, 95.0, 140.2],
            "Health": ["Healthy", "Healthy", "Healthy"]
        })
        st.dataframe(demo_telemetry, width="stretch", hide_index=True, key="master_dash_df")

with side_panel:
    with st.container(border=True):
        st.markdown("##### 🎛️ Rapid Fleet Actions")
        st.button("🔄 Reload Model Weights", width="stretch", key="act_reload")
        st.button("🧹 Clear In-Memory Cache", width="stretch", key="act_clear")
        st.button("📋 Download Telemetry (CSV)", width="stretch", key="act_export")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Advanced Layout Primitives Reference Guide", divider="rainbow")
st.markdown(
    r"""
| Layout Primitive | Primary Role | Key Options & Properties |
| :--- | :--- | :--- |
| **`st.columns(spec)`** | Horizontal partitioning | `spec=[3, 1]`, `vertical_alignment="bottom"`, `border=True` |
| **`st.container(border=True)`** | Visual card boundary | Groups related metrics, tables, and actions cleanly |
| **`st.container(horizontal=True)`**| Horizontal flexbox row | Left-to-right element flow with automatic wrapping |
| **`@st.dialog(title)`** | True modal dialog window | Background overlay, fragment-scoped reruns, `width="small" \| "large"` |
| **`st.popover(label)`** | Floating contextual dropdown | Attached to button, dismisses on outside click |
| **`st.toast(message)`** | Ephemeral status alert | Floating temporary banner in bottom-right viewport corner |
| **`Nesting Rule of Thumb`** | Maintain readability | Cap column nesting at **≤ 2 levels**; use tabs for deep complexity |
"""
)
