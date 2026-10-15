import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Forms Tutorial",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Creating Forms (`st.form` & `st.form_submit_button`) — Complete Guide")
st.markdown(
    """
By default, Streamlit reruns your script every time a user modifies an input widget. 
While ideal for rapid live reactivity, this behavior is inefficient and disruptive when filling out 
multi-field registration forms or complex ML configuration panels. 
**`st.form()`** batches inputs into an atomic unit, **freezing reruns** until the user clicks 
**`st.form_submit_button()`**.
"""
)

st.divider()

# ==============================================================================
# 1. st.form() Core Architecture & 2. Form Keys
# ==============================================================================
st.header("1. Core Architecture & 2. Mandatory Form Keys", divider="rainbow")
st.markdown(
    """
`st.form(key)` is a specialized container that changes the execution semantics of all enclosed widgets:
- **Batching Execution**: Enclosed widgets update internally in the browser without triggering a Python script rerun.
- **Mandatory `key`**: Every form requires a globally unique string identifier `st.form("unique_form_key")`.
- **Mandatory Submit Button**: Every form **must contain at least one `st.form_submit_button()`**.
- **Rule on Buttons**: Standard `st.button()` **cannot** be placed inside a form (will raise a `StreamlitAPIException`).
"""
)

st.markdown("#### 📐 Formal Signature: `st.form()`")
st.code(
    """st.form(
    key: str,  # Required unique identifier
    clear_on_submit: bool = False,
    *,
    enter_to_submit: bool = True,
    border: bool = True,
    width: "stretch" = "stretch"
) -> DeltaGenerator""",
    language="python",
)

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.markdown("**💻 Python Source Code (Minimal Form Architecture)**")
    code_min_form = '''with st.form("minimal_auth_form"):
    st.subheader("Login Credentials")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Must have a submit button inside:
    submit_btn = st.form_submit_button("Log In", type="primary")

if submit_btn:
    st.success(f"Authenticated as `{username}`!")'''
    st.code(code_min_form, language="python")

with col_f2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.form("demo_minimal_auth_form"):
        st.markdown("##### 🔐 Login Credentials")
        u_name = st.text_input("Username", placeholder="e.g. adanani", key="demo_auth_user")
        p_word = st.text_input("Password", type="password", key="demo_auth_pass")
        submit_auth = st.form_submit_button("Log In", type="primary")

    if submit_auth:
        if u_name:
            st.success(f"✅ Successfully authenticated user: `{u_name}`")
        else:
            st.error("❌ Username is required!")

st.divider()

# ==============================================================================
# 3. st.form_submit_button() Options
# ==============================================================================
st.header("3. Submit Button Options (`st.form_submit_button`)", divider="rainbow")
st.markdown(
    """
`st.form_submit_button` triggers the atomic rerun and returns `True` only on the frame of submission:
- **`type="primary"`**: Renders with the primary brand accent color.
- **`icon`**: Adds an icon prefix to the button label.
- **`on_click`**: Attaches a pre-rerun validation callback function.
"""
)

st.markdown("#### 📐 Formal Signature: `st.form_submit_button()`")
st.code(
    """st.form_submit_button(
    label: str = "Submit",
    help: str | None = None,
    on_click: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    type: "primary" | "secondary" | "tertiary" = "secondary",
    icon: str | None = None,
    disabled: bool = False,
    width: "content" | "stretch" = "content"
) -> bool""",
    language="python",
)

st.divider()

# ==============================================================================
# 4. Grouping Heterogeneous Input Fields Inside Forms
# ==============================================================================
st.header("4. Grouping Complex Inputs into Multi-Column Forms", divider="rainbow")
st.markdown(
    """
Inside `st.form()`, you can freely use **`st.columns()`** to arrange diverse inputs into structured, professional cards:
- Text inputs, sliders, selectboxes, number inputs, and toggles coexist seamlessly.
- None of these inputs trigger a rerun until the submit button is clicked.
"""
)

col_grp1, col_grp2 = st.columns(2)

with col_grp1:
    st.markdown("**💻 Python Source Code (Multi-Column Form)**")
    code_group_form = '''with st.form("model_config_form"):
    st.markdown("##### ⚙️ Foundation Model Deployment Spec")
    
    c1, c2 = st.columns(2)
    with c1:
        model = st.selectbox("Model", ["Llama-3.1-8B", "DeepSeek-V3", "Claude-3.5"])
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    with c2:
        max_tokens = st.number_input("Max Output Tokens", 256, 8192, 2048)
        streaming = st.toggle("Token Streaming", value=True)
        
    submitted = st.form_submit_button("Deploy Configuration", type="primary")

if submitted:
    st.write(f"Deployed {model} with temp={temperature}")'''
    st.code(code_group_form, language="python")

with col_grp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.form("demo_model_config_form"):
        st.markdown("##### ⚙️ Foundation Model Deployment Spec")
        c1, c2 = st.columns(2)
        with c1:
            cfg_model = st.selectbox("Model", ["Llama-3.1-8B", "DeepSeek-V3", "Claude-3.5-Sonnet"], key="demo_form_model")
            cfg_temp = st.slider("Temperature", 0.0, 1.5, 0.70, step=0.05, key="demo_form_temp")
        with c2:
            cfg_tokens = st.number_input("Max Output Tokens", 256, 8192, 2048, step=256, key="demo_form_tokens")
            cfg_stream = st.toggle("Token Streaming", value=True, key="demo_form_stream")
            
        cfg_submitted = st.form_submit_button("Deploy Configuration", type="primary")

    if cfg_submitted:
        st.info(f"🚀 Active Profile: **{cfg_model}** | Temp: **{cfg_temp}** | Tokens: **{cfg_tokens:,}** | Streaming: **{'ON' if cfg_stream else 'OFF'}**")

st.divider()

# ==============================================================================
# 5. Form Submission Behavior & 6. Preventing Unnecessary Reruns
# ==============================================================================
st.header("5. Submission Behavior & 6. Preventing Unnecessary Reruns", divider="rainbow")
st.markdown(
    """
### ⚡ The Problem with Unbatched Inputs
Consider an AI parameter panel with 6 fields. Without `st.form()`:
1. User types model name → **Rerun #1** (triggers API request / model re-eval).
2. User adjusts slider → **Rerun #2**.
3. User toggles boolean → **Rerun #3**.
4. User selects dropdown → **Rerun #4**.
*Total: 4 heavy computational passes while the user was still entering inputs!*

### 🛡️ The Form Solution
With `st.form()`:
- Reruns #1, #2, #3, and #4 are **completely prevented**.
- Exactly **one single atomic rerun** occurs when the user clicks **Submit**.
"""
)

# Counter comparison demonstrating rerun prevention
if "without_form_reruns" not in st.session_state:
    st.session_state["without_form_reruns"] = 0
if "with_form_reruns" not in st.session_state:
    st.session_state["with_form_reruns"] = 0

col_cmp1, col_cmp2 = st.columns(2)

with col_cmp1:
    with st.container(border=True):
        st.markdown("##### ❌ Without Form: Live Inputs Trigger Reruns")
        st.caption("Moving any input below immediately reruns the application.")
        st.text_input("Live Text Input", placeholder="Type here...", key="live_input_demo")
        st.slider("Live Slider", 0, 100, 50, key="live_slider_demo")
        st.caption("Notice how your browser re-executes on every keystroke.")

with col_cmp2:
    with st.container(border=True):
        st.markdown("##### ✅ With Form: Batched Until Submit")
        with st.form("rerun_proof_form"):
            st.caption("Change inputs below — zero reruns occur until submit!")
            st.text_input("Batched Text Input", placeholder="Type here...", key="batch_input_demo")
            st.slider("Batched Slider", 0, 100, 50, key="batch_slider_demo")
            form_done = st.form_submit_button("Submit Batched Inputs")
        if form_done:
            st.success("Form submitted in a single atomic network request!")

st.divider()

# ==============================================================================
# 7. Handling Submitted Values & 8. Resetting Forms (clear_on_submit)
# ==============================================================================
st.header("7. Handling Submitted Values & 8. Resetting Forms (`clear_on_submit`)", divider="rainbow")
st.markdown(
    """
Streamlit provides two powerful patterns for managing post-submission state:
1. **`clear_on_submit=True`**: Wipes all form fields back to their blank default values immediately upon submission. Ideal for messaging inputs, feedback forms, and transactional log entries.
2. **`clear_on_submit=False` (Default)**: Keeps the submitted values visible in the input boxes for easy iteration or editing.
"""
)

# In-memory ticket submission log
if "submitted_support_tickets" not in st.session_state:
    st.session_state["submitted_support_tickets"] = []

col_rst1, col_rst2 = st.columns(2)

with col_rst1:
    st.markdown("**💻 Python Source Code (`clear_on_submit=True`)**")
    code_clear_form = '''# Form resets its inputs immediately upon submission:
with st.form("support_ticket_form", clear_on_submit=True):
    st.subheader("Submit AI Model Issue Ticket")
    title = st.text_input("Issue Summary")
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
    description = st.text_area("Detailed Error Log")
    
    submit_ticket = st.form_submit_button("File Ticket", type="primary")

if submit_ticket:
    if title:
        # Append to database / session state
        st.session_state["tickets"].append({"title": title, "priority": priority})
        st.success(f"Ticket '{title}' filed! Form inputs cleared.")'''
    st.code(code_clear_form, language="python")

with col_rst2:
    st.markdown("**🖥️ Live Interactive `clear_on_submit` Demo**")
    with st.form("demo_support_ticket_form", clear_on_submit=True):
        st.markdown("##### 🎫 File Support Ticket (`clear_on_submit=True`)")
        t_title = st.text_input("Issue Summary", placeholder="e.g. Out of Memory on H100 Node 4", key="ticket_title_inp")
        t_pri = st.selectbox("Severity Priority", ["Low", "Medium", "High", "Critical (P0)"], key="ticket_pri_inp")
        t_desc = st.text_area("Stack Trace / Repro Details", placeholder="Enter error reproduction steps...", key="ticket_desc_inp", height=70)
        submit_t = st.form_submit_button("File Ticket", type="primary")

    if submit_t:
        if t_title:
            st.session_state["submitted_support_tickets"].append({
                "Summary": t_title,
                "Priority": t_pri,
                "Status": "Open"
            })
            st.success(f"✅ Ticket filed: **{t_title}**! Form fields have been cleanly reset.")
        else:
            st.error("Please provide an issue summary before submitting.")

    if st.session_state["submitted_support_tickets"]:
        st.caption("Logged Tickets in Session:")
        st.dataframe(pd.DataFrame(st.session_state["submitted_support_tickets"]), width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Forms Architecture Cheat Sheet", divider="rainbow")
st.markdown(
    r"""
| Parameter / Concept | Type / Syntax | Behavior & Rule |
| :--- | :--- | :--- |
| **`st.form(key)`** | `key: str` (Required) | Container that freezes reruns on all internal widgets |
| **`clear_on_submit`** | `bool` (Default `False`) | Automatically resets all form inputs to default values on submit |
| **`enter_to_submit`** | `bool` (Default `True`) | Submits the form when user presses Enter inside single-line inputs |
| **`st.form_submit_button`**| `st.form_submit_button()` | Mandatory submit button; returns `True` only on submit rerun pass |
| **Enclosed Buttons** | Prohibited | Normal `st.button()` raises an exception inside `st.form()` |
| **Form Layouts** | Fully supported | `st.columns()` and containers work seamlessly inside forms |
"""
)
