import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Expanders & Popovers Tutorial",
    page_icon="📂",
    layout="wide",
)

st.title("📂 Expanders & Popovers (`st.expander` & `st.popover`) — Complete Guide")
st.markdown(
    """
Clean UI design relies on **Progressive Disclosure** — revealing complex features and information only when the user needs them. 
Streamlit provides two powerful primitives for progressive disclosure:
1. **`st.expander()`**: An inline collapsible accordion drawer that expands vertically into the page layout.
2. **`st.popover()`**: A floating overlay modal attached to a trigger button that pops up over the page without disrupting layout flow.
"""
)

st.divider()

# ==============================================================================
# 1. st.expander() Core Architecture & 2. Collapsible Sections
# ==============================================================================
st.header("1. Core Accordion Architecture & 2. Collapsible Sections (`st.expander`)", divider="rainbow")
st.markdown(
    """
`st.expander()` creates an inline accordion container:
- **`label`**: The title displayed on the clickable toggle bar.
- **`expanded=False` (Default)**: Renders closed by default to save vertical screen real estate.
- **`expanded=True`**: Renders open upon page load.
- **`icon`**: An optional emoji or Material icon displayed next to the label.
- **Use Cases**: System documentation, mathematical derivations, FAQs, and intermediate telemetry logs.
"""
)

st.markdown("#### 📐 Formal Signature: `st.expander()`")
st.code(
    """st.expander(
    label: str,
    expanded: bool = False,
    *,
    icon: str | None = None,
    width: "stretch" = "stretch",
    key: str | int | None = None
) -> ExpanderContainer""",
    language="python",
)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("**💻 Python Source Code**")
    code_expander = '''# 1. Collapsed by default (Documentation / Reference)
with st.expander("📖 Transformer Self-Attention Mathematical Formulation", expanded=False, icon="📐"):
    st.latex(r"\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V")
    st.caption("Where Q, K, V are Query, Key, and Value matrices scaled by the square root of key dimension d_k.")

# 2. Expanded by default (Important Alert or Overview)
with st.expander("🚨 Production Deployment Notice", expanded=True, icon="⚠️"):
    st.warning("Cluster maintenance scheduled for Sunday at 02:00 UTC. Canary routing active.")'''
    st.code(code_expander, language="python")

with col_exp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.expander("📖 Transformer Self-Attention Mathematical Formulation", expanded=False, icon="📐"):
        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
        st.caption("Where Q, K, V are Query, Key, and Value matrices scaled by the square root of key dimension d_k.")

    with st.expander("🚨 Production Deployment Notice", expanded=True, icon="⚠️"):
        st.warning("Cluster maintenance scheduled for Sunday at 02:00 UTC. Canary routing active.")

st.divider()

# ==============================================================================
# 3. st.popover() — Floating Contextual Overlays
# ==============================================================================
st.header("3. Floating Overlay Modals (`st.popover`)", divider="rainbow")
st.markdown(
    """
`st.popover()` renders a button that, when clicked, opens a **floating overlay container**:
- **Zero Layout Shift**: Unlike expanders which push downward elements further down the page, popovers float above the canvas.
- **Click-Outside Dismissal**: Automatically closes when the user clicks anywhere outside the popover window.
- **Custom Button Styling**: Supports `type="primary" | "secondary" | "tertiary"`, `icon`, and `help`.
"""
)

st.markdown("#### 📐 Formal Signature: `st.popover()`")
st.code(
    """st.popover(
    label: str,
    *,
    type: "primary" | "secondary" | "tertiary" = "secondary",
    icon: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    width: "content" | "stretch" = "content",
    key: str | int | None = None
) -> PopoverContainer""",
    language="python",
)

col_pop1, col_pop2 = st.columns(2)

with col_pop1:
    st.markdown("**💻 Python Source Code**")
    code_popover = '''# Floating popover button with inputs inside:
with st.popover("⚙️ Filter Pipeline", icon="🎛️"):
    st.markdown("##### Quick In-Memory Filters")
    selected_tier = st.selectbox("Model Tier", ["All", "Frontier LLM", "Edge SLM"])
    min_mmlu = st.slider("Min MMLU Score", 60.0, 95.0, 80.0)
    apply_btn = st.button("Apply Filter", type="primary")

st.caption("Popover floats over the page when clicked without pushing content down.")'''
    st.code(code_popover, language="python")

with col_pop2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.popover("⚙️ Filter Pipeline", icon="🎛️"):
        st.markdown("##### Quick In-Memory Filters")
        selected_tier = st.selectbox("Model Tier", ["All", "Frontier LLM", "Edge SLM"], key="pop_demo_tier")
        min_mmlu = st.slider("Min MMLU Score", 60.0, 95.0, 80.0, key="pop_demo_mmlu")
        if st.button("Apply Filter", type="primary", key="pop_demo_apply"):
            st.success(f"Filter updated: {selected_tier} (≥ {min_mmlu}%)")
    st.caption("💡 *Click the 'Filter Pipeline' button above to open the floating modal!*")

st.divider()

# ==============================================================================
# 4. Contextual Controls & Quick Action Modals
# ==============================================================================
st.header("4. Contextual Controls & Quick Action Modals", divider="rainbow")
st.markdown(
    """
Popovers shine when used as **compact action triggers** embedded alongside metrics, data tables, or card headers:
- Quick configuration panels attached to specific charts.
- Export and download dialogues.
- User profile or API key credentials input without leaving the active workflow.
"""
)

col_ctx1, col_ctx2 = st.columns(2)

with col_ctx1:
    st.markdown("**💻 Python Source Code (Contextual Card with Popover Action)**")
    code_contextual = '''# Metric Card with Contextual Configuration Popover:
card_header, action_col = st.columns([4, 1], vertical_alignment="center")

with card_header:
    st.markdown("##### 🚀 Inference Throughput: **1,420 tok/s**")

with action_col:
    with st.popover("⚙️ Configure"):
        st.write("Target SLA Throughput:")
        st.number_input("Threshold (tok/s)", value=1000, step=100)
        st.toggle("Trigger Alert on Drop", value=True)'''
    st.code(code_contextual, language="python")

with col_ctx2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        card_header, action_col = st.columns([4, 1], vertical_alignment="center")
        with card_header:
            st.markdown("##### 🚀 Inference Throughput: **1,420 tok/s**")
            st.caption("Fleet Health: :green-badge[● Nominal] (99.98% SLA)")
        with action_col:
            with st.popover("⚙️ Alerts"):
                st.write("Configure Alerts:")
                st.number_input("Min Threshold (tok/s)", value=1000, step=100, key="ctx_demo_thresh")
                st.toggle("PagerDuty Webhook", value=True, key="ctx_demo_pager")

st.divider()

# ==============================================================================
# 5. Nested Expanders Architecture
# ==============================================================================
st.header("5. Nested Expanders Architecture", divider="rainbow")
st.markdown(
    """
Modern Streamlit supports **nesting expanders inside expanders**:
- Organize complex, multi-tiered configuration systems cleanly.
- Outer expander groups the broader architectural category (e.g. `Distributed Training Configuration`).
- Inner expanders detail specialized sub-systems (e.g. `LoRA Parameter Offloading`, `FlashAttention Kernels`).
"""
)

col_nst1, col_nst2 = st.columns(2)

with col_nst1:
    st.markdown("**💻 Python Source Code (Multi-Tier Nested Expanders)**")
    code_nested_exp = '''# Outer Category Expander:
with st.expander("🛠️ Distributed Training & Fine-Tuning Setup", expanded=True):
    st.markdown("Primary Distributed Backend: **PyTorch FSDP (Fully Sharded Data Parallel)**")
    
    # Nested Sub-Expander 1:
    with st.expander("⚡ Parameter-Efficient Fine-Tuning (LoRA)", expanded=False):
        st.slider("LoRA Rank (r)", 4, 128, 16)
        st.slider("LoRA Alpha (α)", 8, 256, 32)
        st.multiselect("Target Modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        
    # Nested Sub-Expander 2:
    with st.expander("🧠 Memory & Kernel Acceleration", expanded=False):
        st.checkbox("Enable FlashAttention-2", value=True)
        st.checkbox("Enable Activation Checkpointing", value=True)'''
    st.code(code_nested_exp, language="python")

with col_nst2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.expander("🛠️ Distributed Training & Fine-Tuning Setup", expanded=True):
        st.markdown("Primary Distributed Backend: **PyTorch FSDP (Fully Sharded Data Parallel)**")
        
        with st.expander("⚡ Parameter-Efficient Fine-Tuning (LoRA)", expanded=False):
            st.slider("LoRA Rank (r)", 4, 128, 16, key="lora_r_demo")
            st.slider("LoRA Alpha (α)", 8, 256, 32, key="lora_alpha_demo")
            st.multiselect("Target Modules", ["q_proj", "k_proj", "v_proj", "o_proj"], default=["q_proj", "v_proj"], key="lora_mod_demo")
            
        with st.expander("🧠 Memory & Kernel Acceleration", expanded=False):
            st.checkbox("Enable FlashAttention-2", value=True, key="fa2_demo")
            st.checkbox("Enable Activation Checkpointing", value=True, key="act_chk_demo")

st.divider()

# ==============================================================================
# 6. Showing Advanced Settings & 7. Hiding Optional Information
# ==============================================================================
st.header("6. Advanced Settings Panels & 7. Hiding Optional Info", divider="rainbow")
st.markdown(
    """
Two critical UX patterns in AI & data science dashboards:
1. **Advanced Hyperparameters Drawer**: Keep primary inputs intuitive (e.g. prompt and model) while collapsing esoteric knobs (temperature, frequency penalty, top_k, logit bias).
2. **Raw Telemetry & Debug Payloads**: Hide huge raw JSON response payloads and stack traces inside closed drawers so non-technical stakeholders see a clean interface while engineers can inspect full raw data.
"""
)

col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    st.markdown("**💻 Python Source Code (Advanced Knobs & Debug Payloads)**")
    code_adv = '''# Primary user interface:
prompt = st.text_input("User Prompt", value="Summarize quantum computing principles.")

# 1. Advanced Settings Expander:
with st.expander("⚙️ Advanced Sampling Hyperparameters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Temperature", 0.0, 2.0, 0.70)
        st.slider("Top-P (Nucleus)", 0.0, 1.0, 0.95)
    with col2:
        st.slider("Presence Penalty", -2.0, 2.0, 0.0)
        st.slider("Frequency Penalty", -2.0, 2.0, 0.0)

# 2. Raw JSON Debug Payload (Hidden by default):
with st.expander("🔍 View Raw API Response JSON (Debug)", expanded=False):
    st.json({
        "id": "chatcmpl-98a7bc",
        "model": "claude-3-5-sonnet",
        "usage": {"prompt_tokens": 14, "completion_tokens": 128, "total_tokens": 142},
        "latency_ms": 182
    })'''
    st.code(code_adv, language="python")

with col_adv2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.text_input("User Prompt", value="Summarize quantum computing principles.", key="demo_adv_prompt")

    with st.expander("⚙️ Advanced Sampling Hyperparameters", expanded=False):
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.slider("Temperature", 0.0, 2.0, 0.70, key="demo_adv_temp")
            st.slider("Top-P (Nucleus)", 0.0, 1.0, 0.95, key="demo_adv_topp")
        with c_adv2:
            st.slider("Presence Penalty", -2.0, 2.0, 0.0, key="demo_adv_pres")
            st.slider("Frequency Penalty", -2.0, 2.0, 0.0, key="demo_adv_freq")

    with st.expander("🔍 View Raw API Response JSON (Debug)", expanded=False):
        st.json({
            "id": "chatcmpl-98a7bc",
            "model": "claude-3-5-sonnet",
            "usage": {"prompt_tokens": 14, "completion_tokens": 128, "total_tokens": 142},
            "finish_reason": "stop",
            "latency_ms": 182
        })

st.divider()

# ==============================================================================
# Summary Comparison Reference Table
# ==============================================================================
st.header("📋 Comparison: `st.expander` vs. `st.popover`", divider="rainbow")
st.markdown(
    r"""
| Architectural Property | `st.expander()` | `st.popover()` |
| :--- | :--- | :--- |
| **Presentation Type** | Inline vertical accordion drawer | Floating modal overlay attached to a trigger button |
| **Canvas Impact** | Pushes subsequent page elements down when opened | Zero layout shift; floats above the canvas |
| **Dismissal Mechanism** | Explicitly clicked toggle bar | Clicking outside the floating box closes it automatically |
| **Button Customization** | Accordion header bar with optional icon | Full button styling (`primary`, `secondary`, `tertiary`, `icon`) |
| **Best Used For** | FAQs, math proofs, advanced settings panels, debug logs | Quick action filters, row-level options, inline forms, modals |
"""
)
