import streamlit as st
import time

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Button Widgets Tutorial",
    page_icon="🔘",
    layout="wide",
)

st.title("🔘 Button Widgets (`st.button`) — Complete Guide")
st.markdown(
    """
Buttons are the primary interactive triggers in Streamlit applications. 
Unlike input fields that retain their value across sessions, buttons operate on a **transient single-run lifecycle**.
This master reference covers formal syntax, key hashing, callback execution order, calculation triggers, 
and state persistence patterns.
"""
)

st.divider()

# ==============================================================================
# 1. st.button() & Visual Variants
# ==============================================================================
st.header("1. Core Function & Visual Button Variants", divider="rainbow")
st.markdown(
    """
`st.button()` renders a clickable button element. Streamlit supports three visual button types:
- **`primary`**: High-emphasis accent color (used for primary calls-to-action like Submit, Train, Deploy).
- **`secondary`**: Outlined neutral button (default standard button).
- **`tertiary`**: Borderless text-link button (used for low-emphasis actions like Cancel, Learn More).
"""
)

st.markdown("#### 📐 Formal Signature: `st.button()`")
st.code(
    """st.button(
    label: str,
    key: str | int | None = None,
    help: str | None = None,
    on_click: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    type: "primary" | "secondary" | "tertiary" = "secondary",
    icon: str | None = None,
    icon_position: "left" | "right" = "left",
    disabled: bool = False,
    width: "stretch" | "content" = "content",
    shortcut: str | None = None
) -> bool""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.button()`")
st.markdown(
    """
| Parameter | Type | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`label`** | `str` | *Required* | Any string / Markdown | Text displayed inside the button body. |
| **`key`** | `str or int` | `None` | Unique identifier | Explicit unique key to identify the widget and prevent collisions. |
| **`type`** | `str` | `"secondary"` | `"primary"`, `"secondary"`, `"tertiary"` | Visual weight and hierarchy styling. |
| **`icon`** | `str` | `None` | Emoji (e.g. `"🚀"`) or Material icon (e.g. `":material/send:"`) | Icon rendered alongside the button label. |
| **`icon_position`** | `str` | `"left"` | `"left"`, `"right"` | Renders the icon before or after the label. |
| **`disabled`** | `bool` | `False` | `True`, `False` | Whether the button is grayed out and unclickable. |
| **`width`** | `str` | `"content"` | `"stretch"`, `"content"` | `"stretch"` fills container width; `"content"` hugs label text. |
| **`help`** | `str` | `None` | Tooltip string | Hover tooltip describing the action. |
"""
)

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("**💻 Python Source Code**")
    code_variants = '''# Visual button variants:
c1, c2, c3 = st.columns(3)
with c1:
    st.button("Primary Action", type="primary", icon="🚀", width="stretch")
with c2:
    st.button("Secondary Action", type="secondary", icon="⚙️", width="stretch")
with c3:
    st.button("Tertiary Action", type="tertiary", icon="ℹ️", width="stretch")

# Disabled state:
st.button("Unavailable (Disabled)", disabled=True, width="stretch")'''
    st.code(code_variants, language="python")

with col_v2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Primary Action", type="primary", icon="🚀", width="stretch", key="demo_btn_prim"):
            st.toast("Primary clicked!")
    with c2:
        if st.button("Secondary Action", type="secondary", icon="⚙️", width="stretch", key="demo_btn_sec"):
            st.toast("Secondary clicked!")
    with c3:
        if st.button("Tertiary Action", type="tertiary", icon="ℹ️", width="stretch", key="demo_btn_tert"):
            st.toast("Tertiary clicked!")

    st.button("Unavailable (Disabled)", disabled=True, width="stretch", key="demo_btn_dis")

st.divider()

# ==============================================================================
# 2. Button Labels (Emojis, Icons, Positions & Dynamic Labels)
# ==============================================================================
st.header("2. Button Labels & Icon Positioning", divider="rainbow")
st.markdown(
    """
Button labels can incorporate standard emojis, Material design icons, and dynamic text that reacts 
to user state (e.g. switching from "Start Training" to "Pause Training"):
"""
)

col_lbl1, col_lbl2 = st.columns(2)

with col_lbl1:
    st.markdown("**💻 Python Source Code**")
    code_labels = '''# 1. Material Icons with left vs right placement
st.button("Deploy to Production", icon="🚢", icon_position="left")
st.button("Export Results", icon="📥", icon_position="right")

# 2. Dynamic state-driven label
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False

label_text = "⏸️ Pause Pipeline" if st.session_state["is_running"] else "▶️ Start Pipeline"
btn_type = "primary" if not st.session_state["is_running"] else "secondary"

if st.button(label_text, type=btn_type):
    st.session_state["is_running"] = not st.session_state["is_running"]
    st.rerun()'''
    st.code(code_labels, language="python")

with col_lbl2:
    st.markdown("**🖥️ Rendered Interactive Output**")
    st.button("Deploy to Production", icon="🚢", icon_position="left", key="btn_dep_left")
    st.button("Export Results", icon="📥", icon_position="right", key="btn_exp_right")

    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False

    dyn_label = "⏸️ Pause Pipeline" if st.session_state["pipeline_running"] else "▶️ Start Pipeline"
    dyn_type = "secondary" if st.session_state["pipeline_running"] else "primary"

    if st.button(dyn_label, type=dyn_type, key="btn_dyn_toggle"):
        st.session_state["pipeline_running"] = not st.session_state["pipeline_running"]
        st.rerun()

    st.caption(f"Current State: Pipeline is **{'ACTIVE (RUNNING)' if st.session_state['pipeline_running'] else 'IDLE (STOPPED)'}**")

st.divider()

# ==============================================================================
# 3. Button Keys & DuplicateWidgetID Prevention
# ==============================================================================
st.header("3. Button Keys & Disambiguation", divider="rainbow")
st.markdown(
    """
### 🔑 The Anatomy of Widget Keys
Streamlit automatically generates internal keys by hashing the widget's label and location.
However, if two buttons share the same label or appear inside loops, Streamlit cannot distinguish them:
- **Omitted Key**: Streamlit hashes `hash("Delete")` $\\rightarrow$ triggers `DuplicateWidgetID` if repeated.
- **Explicit Key**: Providing `key=f"delete_user_{user_id}"` guarantees clean uniqueness.
- **Session State Access**: The boolean button click status can be inspected directly via `st.session_state[key]`.
"""
)

col_key1, col_key2 = st.columns(2)

with col_key1:
    st.markdown("**💻 Python Source Code (Identical Labels with Distinct Keys)**")
    code_keys = '''# Multiple buttons with identical labels require explicit keys:
col_a, col_b = st.columns(2)

with col_a:
    if st.button("Delete Item", key="delete_cart_item", type="secondary"):
        st.error("Deleted from Shopping Cart!")

with col_b:
    if st.button("Delete Item", key="delete_wishlist_item", type="secondary"):
        st.warning("Deleted from Wishlist!")'''
    st.code(code_keys, language="python")

with col_key2:
    st.markdown("**🖥️ Rendered Output**")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("Delete Item", key="delete_cart_item", type="secondary"):
            st.error("Deleted from Shopping Cart!")
    with col_k2:
        if st.button("Delete Item", key="delete_wishlist_item", type="secondary"):
            st.warning("Deleted from Wishlist!")

st.divider()

# ==============================================================================
# 4. Button Return Values (The Transient Lifecycle)
# ==============================================================================
st.header("4. Button Return Values & The Transient Lifecycle", divider="rainbow")
st.markdown(
    """
> [!IMPORTANT]
> **The Transient Lifecycle Rule**  
> `st.button()` returns `True` for **EXACTLY ONE execution pass** (the rerun triggered by the physical mouse click).  
> On any subsequent rerun (such as adjusting an adjacent slider or typing in a text field), `st.button()` immediately reverts to `False`!
"""
)

col_ret1, col_ret2 = st.columns(2)

with col_ret1:
    st.markdown("**💻 Python Source Code (Transient Pitfall)**")
    code_ret = '''# The Transient Pitfall:
btn_clicked = st.button("Show Secret Analysis", key="btn_secret")

if btn_clicked:
    st.success("🎉 Secret revealed!")
    # BUT if the user adjusts this slider, the script reruns,
    # btn_clicked becomes False, and this entire block vanishes!
    st.slider("Adjust Sensitivity:", 1, 10)'''
    st.code(code_ret, language="python")

with col_ret2:
    st.markdown("**🖥️ Live Demonstration of the Transient Lifecycle**")
    raw_button_val = st.button("Check Return Value", key="btn_check_return")
    
    st.metric("`st.button()` Current Return Value", f"{raw_button_val}", delta="True for 1 rerun only" if raw_button_val else "False on subsequent runs")
    
    # Interaction slider that causes a rerun
    st.slider("Drag this slider to trigger a rerun without clicking the button:", 0, 100, 50, key="transient_slider")
    st.caption("Notice how dragging the slider reruns the app and instantly resets the button back to `False`.")

st.divider()

# ==============================================================================
# 5. Handling Button Clicks (Inline vs. Callbacks)
# ==============================================================================
st.header("5. Handling Button Clicks: Inline vs. `on_click` Callbacks", divider="rainbow")
st.markdown(
    """
There are two distinct architectural paradigms for handling button clicks in Streamlit:
1. **Inline Condition (`if st.button():`)**: Executes within the normal top-to-bottom script flow.
2. **Callback Handler (`on_click=fn, args=(...)`)**: Executes **BEFORE the script begins rerunning** from line 1. 
   Ideal for updating `st.session_state` or mutating backend models before widgets render.
"""
)

# Initialize callback log
if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = []

def record_audit_event(action_name: str, actor: str):
    """Callback function executed prior to script rerun."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state["audit_log"].append(f"[{timestamp}] Action: {action_name} by {actor}")

col_cb1, col_cb2 = st.columns(2)

with col_cb1:
    st.markdown("**💻 Python Source Code (Callback with Arguments)**")
    code_callbacks = '''def record_audit_event(action_name: str, actor: str):
    """Executes BEFORE main script executes line 1!"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state["audit_log"].append(f"[{timestamp}] {action_name} by {actor}")

# Button with on_click callback and arguments:
st.button(
    "Authorize Deployment",
    on_click=record_audit_event,
    args=("PROD_DEPLOY_V2", "lead-engineer@mlops.io"),
    type="primary",
    key="btn_cb_auth"
)'''
    st.code(code_callbacks, language="python")

with col_cb2:
    st.markdown("**🖥️ Rendered Callback Execution**")
    st.button(
        "Authorize Deployment",
        on_click=record_audit_event,
        args=("PROD_DEPLOY_V2", "lead-engineer@mlops.io"),
        type="primary",
        key="btn_cb_auth"
    )

    if st.session_state["audit_log"]:
        st.markdown("**Audit Event Trail (Recorded in Callback):**")
        for log_entry in st.session_state["audit_log"][-3:]:
            st.code(log_entry, language="text")

st.divider()

# ==============================================================================
# 6. Triggering Calculations (Persistent State Pattern)
# ==============================================================================
st.header("6. Triggering Heavy Calculations (Persistent State Pattern)", divider="rainbow")
st.markdown(
    """
To prevent calculation results from disappearing when adjacent inputs change, store the results 
in **`st.session_state`**. Combine with `st.spinner()` for a responsive user experience.
"""
)

if "calc_completed" not in st.session_state:
    st.session_state["calc_completed"] = False
    st.session_state["calc_payload"] = None

col_calc1, col_calc2 = st.columns(2)

with col_calc1:
    st.markdown("**💻 Python Source Code (State-Latching Calculation)**")
    code_calc = '''# State-Latching Pattern:
if "calc_completed" not in st.session_state:
    st.session_state["calc_completed"] = False
    st.session_state["calc_payload"] = None

num_samples = st.slider("Dataset Size (Rows):", 1000, 50000, 10000)

if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner("Simulating stochastic distribution..."):
        time.sleep(0.5)  # Simulate compute workload
        st.session_state["calc_payload"] = {
            "mean": 42.15,
            "variance": 1.84,
            "samples": num_samples
        }
        st.session_state["calc_completed"] = True

# Results stay visible even when other widgets update!
if st.session_state["calc_completed"]:
    res = st.session_state["calc_payload"]
    st.success(f"Simulation Complete on {res['samples']:,} samples!")'''
    st.code(code_calc, language="python")

with col_calc2:
    st.markdown("**🖥️ Rendered Interactive Calculation**")
    num_samples_input = st.slider("Dataset Size (Rows):", 1000, 50000, 10000, step=1000, key="sim_samples_slider")

    if st.button("Run Monte Carlo Simulation", type="primary", key="btn_run_sim"):
        with st.spinner("Simulating stochastic distribution..."):
            time.sleep(0.4)
            st.session_state["calc_payload"] = {
                "mean": 42.15,
                "variance": 1.84,
                "samples": num_samples_input
            }
            st.session_state["calc_completed"] = True
            st.toast("Simulation finished successfully!")

    if st.session_state["calc_completed"]:
        res_data = st.session_state["calc_payload"]
        st.success(f"Simulation Complete on **{res_data['samples']:,}** samples!")
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Empirical Mean", f"{res_data['mean']:.2f}")
        res_c2.metric("Variance", f"{res_data['variance']:.2f}")

st.divider()

# ==============================================================================
# 7. Reset Buttons (Reverting State to Defaults)
# ==============================================================================
st.header("7. Reset Buttons (Reverting Application State)", divider="rainbow")
st.markdown(
    """
Reset buttons restore application state back to clean initial defaults.
Best practice is to clear target keys in `st.session_state` and invoke **`st.rerun()`** to force an instant clean slate.
"""
)

col_rst1, col_rst2 = st.columns(2)

with col_rst1:
    st.markdown("**💻 Python Source Code (Clean Reset Pattern)**")
    code_rst = '''def reset_simulation_state():
    """Clears cached results and resets flags."""
    st.session_state["calc_completed"] = False
    st.session_state["calc_payload"] = None
    st.session_state["audit_log"] = []
    st.rerun()

st.button(
    "Reset All Simulation Results",
    on_click=reset_simulation_state,
    type="secondary",
    icon="🔄"
)'''
    st.code(code_rst, language="python")

with col_rst2:
    st.markdown("**🖥️ Rendered Reset Action**")
    def reset_simulation_state_demo():
        st.session_state["calc_completed"] = False
        st.session_state["calc_payload"] = None
        st.session_state["audit_log"] = []
        st.rerun()

    if st.button("Reset All Simulation Results", icon="🔄", key="btn_reset_all_sim"):
        reset_simulation_state_demo()

    st.caption("Clicking reset clears the Monte Carlo calculation and audit log, then instantly reloads the clean canvas.")

st.divider()

# ==============================================================================
# 8. Multiple Buttons & Action Bars
# ==============================================================================
st.header("8. Multiple Buttons & Action Bars", divider="rainbow")
st.markdown(
    """
Applications frequently require multiple buttons arranged horizontally to form **Action Bars** 
(e.g. Save / Discard, Previous / Next, Train / Validate / Deploy):
- Organize buttons side-by-side using `st.columns()`.
- Use visual hierarchy: Reserve `type="primary"` for the default expected action, and `type="secondary"` or `type="tertiary"` for alternatives.
"""
)

col_mb1, col_mb2 = st.columns(2)

with col_mb1:
    st.markdown("**💻 Python Source Code (Hierarchical Action Bar)**")
    code_action_bar = '''# 3-Tier Workflow Action Bar:
col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

with col_act1:
    if st.button("Deploy Model to Staging", type="primary", icon="🚀", width="stretch"):
        st.success("Deployment triggered!")

with col_act2:
    if st.button("Dry Run Test", type="secondary", icon="🧪", width="stretch"):
        st.info("Validation tests passed.")

with col_act3:
    if st.button("Cancel", type="tertiary", width="stretch"):
        st.warning("Action cancelled.")'''
    st.code(code_action_bar, language="python")

with col_mb2:
    st.markdown("**🖥️ Rendered Action Bar**")
    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

    with col_act1:
        if st.button("Deploy Model to Staging", type="primary", icon="🚀", width="stretch", key="act_deploy"):
            st.success("Deployment triggered!")

    with col_act2:
        if st.button("Dry Run Test", type="secondary", icon="🧪", width="stretch", key="act_dryrun"):
            st.info("Validation tests passed.")

    with col_act3:
        if st.button("Cancel", type="tertiary", width="stretch", key="act_cancel"):
            st.warning("Action cancelled.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Button Development Cheat Sheet", divider="rainbow")
st.markdown(
    """
| Scenario | Recommended Approach | Code Example |
| :--- | :--- | :--- |
| **Primary Action** | Use `type="primary"` | `st.button("Submit", type="primary")` |
| **Cancel / Dismiss** | Use `type="tertiary"` | `st.button("Cancel", type="tertiary")` |
| **Repeated Buttons** | Always specify unique `key` | `st.button("Delete", key=f"del_{item_id}")` |
| **Pre-Rerun Mutating** | Use `on_click` callback | `st.button("Save", on_click=save_db, args=(payload,))` |
| **Persistent Output** | Store in `st.session_state` | `if st.button("Calc"): st.session_state["val"] = res` |
| **Resetting Canvas** | Delete keys and call `st.rerun()` | `st.session_state.clear(); st.rerun()` |
| **Full-Width Button** | Set `width="stretch"` | `st.button("Full Width", width="stretch")` |
"""
)
