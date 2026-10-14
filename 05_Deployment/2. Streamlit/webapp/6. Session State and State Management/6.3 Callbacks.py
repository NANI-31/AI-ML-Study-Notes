import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Callbacks Tutorial",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Streamlit Callbacks (`on_change` & `on_click`) — Complete Guide")
st.markdown(
    """
Callbacks are Python functions executed **prior** to the main script rerun when a user interacts with a widget. 
They provide the cleanest, most robust mechanism to synchronize state, process form submissions, 
pass parametric arguments via `args` and `kwargs`, and prevent UI rendering synchronization bugs.
"""
)

st.divider()

# ==============================================================================
# 1. Understanding Callbacks & The Execution Lifecycle
# ==============================================================================
st.header("1. Understanding Callbacks & The Execution Lifecycle", divider="rainbow")
st.markdown(
    """
### 🔄 The Three-Stage Callback Lifecycle
When a user interacts with a widget with a callback attached:
1. **Stage 1 (State Update)**: Streamlit updates the widget's value in `st.session_state`.
2. **Stage 2 (Callback Execution)**: Streamlit executes the callback function. Any mutations made to other `st.session_state` keys occur **here**.
3. **Stage 3 (Main Script Rerun)**: Streamlit reruns the script from top to bottom. The entire app sees the newly updated state immediately!
"""
)

st.info(
    "💡 **Key Advantage**: Because callbacks execute *before* the script reruns, downstream widgets and metrics render the latest updated state on the exact same frame without requiring an extra rerun!"
)

st.divider()

# ==============================================================================
# 2. on_change (Input Widgets) & 3. on_click (Action Widgets)
# ==============================================================================
st.header("2. `on_change` (Inputs) & 3. `on_click` (Buttons)", divider="rainbow")
st.markdown(
    """
- **`on_change`**: Available on value-holding input widgets (`st.text_input`, `st.slider`, `st.selectbox`, `st.checkbox`, etc.). Fires whenever the user alters the value.
- **`on_click`**: Available on action trigger widgets (`st.button`, `st.form_submit_button`, `st.download_button`). Fires when the button is clicked.
"""
)

# State initialization
if "prompt_char_count" not in st.session_state:
    st.session_state["prompt_char_count"] = 0
if "audit_click_counter" not in st.session_state:
    st.session_state["audit_click_counter"] = 0

def on_prompt_change():
    text = st.session_state.get("user_prompt_input", "")
    st.session_state["prompt_char_count"] = len(text)

def on_audit_button_click():
    st.session_state["audit_click_counter"] += 1

col_cb1, col_cb2 = st.columns(2)

with col_cb1:
    st.markdown("**💻 Python Source Code**")
    code_basic_cb = '''# 1. on_change callback on text input:
def track_chars():
    st.session_state["char_count"] = len(st.session_state["prompt_box"])

st.text_input("Enter Prompt:", key="prompt_box", on_change=track_chars)
st.caption(f"Live Length: {st.session_state['char_count']} chars")

# 2. on_click callback on button:
def count_clicks():
    st.session_state["total_clicks"] += 1

st.button("Run Audit Event", on_click=count_clicks)'''
    st.code(code_basic_cb, language="python")

with col_cb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.text_input(
        "Enter AI Prompt:", 
        placeholder="Type something to trigger on_change...",
        key="user_prompt_input", 
        on_change=on_prompt_change
    )
    st.caption(f"📏 Live Character Count: **{st.session_state['prompt_char_count']}** characters")

    st.button("Run Audit Event (`on_click`)", on_click=on_audit_button_click, key="btn_audit_demo")
    st.metric("Total Audit Runs", st.session_state["audit_click_counter"])

st.divider()

# ==============================================================================
# 4. Callback Functions & 5. Using args and kwargs
# ==============================================================================
st.header("4. Callback Functions & 5. Passing `args` and `kwargs`", divider="rainbow")
st.markdown(
    """
You can pass dynamic arguments into your callback functions using `args` (positional tuple) and `kwargs` (keyword dictionary):
- **Tuple Requirement**: `args` must be an iterable tuple. For a single argument, remember the trailing comma: `args=("model_id",)`.
- **Keyword Dictionary**: `kwargs={"increment": 5}` passes named parameters.
"""
)

if "active_preset_model" not in st.session_state:
    st.session_state["active_preset_model"] = "Llama-3.1-8B"
if "configured_tokens" not in st.session_state:
    st.session_state["configured_tokens"] = 2048

def apply_hyperparameter_preset(model_name: str, max_tokens: int, provider: str = "HuggingFace"):
    st.session_state["active_preset_model"] = model_name
    st.session_state["configured_tokens"] = max_tokens
    st.session_state["preset_provider"] = provider

col_arg1, col_arg2 = st.columns(2)

with col_arg1:
    st.markdown("**💻 Python Source Code (`args` & `kwargs`)**")
    code_args = '''def apply_preset(model_name: str, max_tokens: int, provider: str = "Meta"):
    st.session_state["active_model"] = model_name
    st.session_state["tokens"] = max_tokens
    st.session_state["provider"] = provider

# Pass positional tuple (args) and keyword dict (kwargs):
st.button(
    "Set Llama Preset",
    on_click=apply_preset,
    args=("Llama-3.3-70B", 8192),
    kwargs={"provider": "Meta"}
)

st.button(
    "Set Claude Preset",
    on_click=apply_preset,
    args=("Claude-3.5-Sonnet", 4096),
    kwargs={"provider": "Anthropic"}
)'''
    st.code(code_args, language="python")

with col_arg2:
    st.markdown("**🖥️ Live Interactive Preset Buttons**")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        st.button(
            "⚡ Apply Llama 70B Preset",
            on_click=apply_hyperparameter_preset,
            args=("Llama-3.3-70B", 8192),
            kwargs={"provider": "Meta AI"},
            key="btn_preset_llama"
        )
    with c_btn2:
        st.button(
            "🧠 Apply Claude 3.5 Preset",
            on_click=apply_hyperparameter_preset,
            args=("Claude-3.5-Sonnet", 4096),
            kwargs={"provider": "Anthropic"},
            key="btn_preset_claude"
        )

    with st.container(border=True):
        st.markdown(f"##### Selected Model: **{st.session_state['active_preset_model']}**")
        st.caption(f"Provider: **{st.session_state.get('preset_provider', 'Meta AI')}** | Context Window: **{st.session_state['configured_tokens']:,}** tokens")

st.divider()

# ==============================================================================
# 6. Updating State Through Callbacks (Experiment History Buffer)
# ==============================================================================
st.header("6. Updating State Through Callbacks (Experiment Logger)", divider="rainbow")
st.markdown(
    """
Callbacks are the ideal place to accumulate audit logs, compute derived summaries, 
or push new records to an in-memory database table before the canvas renders.
"""
)

if "experiment_log_records" not in st.session_state:
    st.session_state["experiment_log_records"] = []

def log_experiment_run():
    run_time = datetime.now().strftime("%H:%M:%S")
    lr = st.session_state.get("inp_learning_rate", 0.001)
    arch = st.session_state.get("inp_architecture", "Transformer")
    
    st.session_state["experiment_log_records"].append({
        "Timestamp": run_time,
        "Architecture": arch,
        "Learning Rate": lr,
        "Status": "Completed"
    })

col_log1, col_log2 = st.columns(2)

with col_log1:
    st.markdown("**💻 Python Source Code (State Update Callback)**")
    code_log = '''def log_run():
    new_entry = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Architecture": st.session_state["inp_arch"],
        "LR": st.session_state["inp_lr"]
    }
    st.session_state["run_history"].append(new_entry)

st.selectbox("Architecture", ["Transformer", "Mamba"], key="inp_arch")
st.number_input("Learning Rate", value=0.001, key="inp_lr")
st.button("Log Experiment", on_click=log_run)'''
    st.code(code_log, language="python")

with col_log2:
    st.markdown("**🖥️ Live Interactive Experiment Logger**")
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        st.selectbox("Architecture", ["Transformer", "Mamba", "RWKV", "MoE"], key="inp_architecture")
    with c_in2:
        st.number_input("Learning Rate", min_value=1e-5, max_value=0.1, value=0.0003, step=1e-4, format="%.5f", key="inp_learning_rate")

    st.button("📝 Record Experiment Run (`on_click`)", on_click=log_experiment_run, key="btn_record_exp")

    if st.session_state["experiment_log_records"]:
        st.dataframe(pd.DataFrame(st.session_state["experiment_log_records"]), width="stretch", hide_index=True)
    else:
        st.caption("No experiments recorded yet. Click button above to append.")

st.divider()

# ==============================================================================
# 7. Form Submission Callbacks
# ==============================================================================
st.header("7. Form Submission Callbacks (`st.form_submit_button`)", divider="rainbow")
st.markdown(
    """
In forms, individual input widgets do not rerun the app on change. Instead, execution batches until 
the user clicks the submit button. Attaching an `on_click` callback to **`st.form_submit_button`** 
allows you to validate and process all batched fields in one atomic step!
"""
)

if "submitted_user_profile" not in st.session_state:
    st.session_state["submitted_user_profile"] = None

def handle_form_submission():
    uname = st.session_state.get("form_username", "").strip()
    role = st.session_state.get("form_role", "Developer")
    gpu_quota = st.session_state.get("form_gpu_quota", 1)

    if not uname:
        st.session_state["form_error"] = "Username cannot be empty!"
        st.session_state["submitted_user_profile"] = None
    else:
        st.session_state["form_error"] = None
        st.session_state["submitted_user_profile"] = {
            "Username": uname,
            "Role": role,
            "H100 Quota": f"{gpu_quota} GPUs"
        }

col_frm1, col_frm2 = st.columns(2)

with col_frm1:
    st.markdown("**💻 Python Source Code (Form Callback)**")
    code_form_cb = '''def process_user_form():
    uname = st.session_state["form_user"].strip()
    if not uname:
        st.session_state["form_err"] = "Username required!"
    else:
        st.session_state["user_profile"] = {"name": uname}

with st.form("user_registration_form"):
    st.text_input("Username", key="form_user")
    st.selectbox("Role", ["Researcher", "Engineer"], key="form_role")
    st.form_submit_button("Submit", on_click=process_user_form)'''
    st.code(code_form_cb, language="python")

with col_frm2:
    st.markdown("**🖥️ Live Interactive Form Callback Demo**")
    with st.form("demo_user_registration_form"):
        st.text_input("Engineer Username", placeholder="e.g. adanani", key="form_username")
        st.selectbox("Engineering Role", ["AI Research Scientist", "MLOps Engineer", "Data Analyst"], key="form_role")
        st.slider("Requested H100 GPU Quota", 1, 8, 2, key="form_gpu_quota")
        st.form_submit_button("Submit Registration", on_click=handle_form_submission)

    if st.session_state.get("form_error"):
        st.error(st.session_state["form_error"])
    elif st.session_state["submitted_user_profile"]:
        st.success("✅ Profile Processed via Callback:")
        st.json(st.session_state["submitted_user_profile"])

st.divider()

# ==============================================================================
# 8. Avoiding Callback-Related Bugs & Anti-Patterns
# ==============================================================================
st.header("8. Avoiding Callback Bugs & Common Anti-Patterns", divider="rainbow")
st.markdown(
    r"""
### ⚠️ Common Callback Anti-Patterns

1. **Calling the Function Instead of Passing the Reference**:
   - ❌ `st.button("Click", on_click=my_function())` *(Calls function immediately at render time!)*
   - ✅ `st.button("Click", on_click=my_function)` *(Passes callable reference).*

2. **Missing Trailing Comma in Single-Item `args` Tuple**:
   - ❌ `args=("llama")` *(Python treats `("llama")` as a string, unpacking characters `'l', 'l', 'a', 'm', 'a'` into separate args!)*
   - ✅ `args=("llama",)` *(Trailing comma ensures it is parsed as a single-element tuple).*

3. **Rendering UI Directly Inside Callback Functions**:
   - ❌ Calling `st.write()` or `st.error()` inside the callback itself causes elements to render at the very top of the page before page configuration.
   - ✅ Always mutate session state inside the callback, then render UI in the main script flow based on that state.

4. **Calling `st.rerun()` Inside Callbacks**:
   - ❌ Calling `st.rerun()` inside a callback is redundant and can cause recursion warnings because Streamlit *automatically reruns* after callbacks finish!
"""
)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Callbacks API Reference Guide", divider="rainbow")
st.markdown(
    r"""
| Parameter / Concept | Type | Behavior & Purpose |
| :--- | :--- | :--- |
| **`on_change`** | `Callable` | Executes immediately when input widget value is modified |
| **`on_click`** | `Callable` | Executes immediately when button / form submit is clicked |
| **`args`** | `tuple` | Positional parameters passed to callback (e.g. `args=(val1, val2)`) |
| **`kwargs`** | `dict` | Keyword parameters passed to callback (e.g. `kwargs={"key": val}`) |
| **Execution Timing** | *Pre-Rerun* | Executes **before** the main script top-to-bottom rerun |
| **Golden Rule** | *Decouple UI* | Mutate `st.session_state` in callback; render UI in main script |
"""
)
