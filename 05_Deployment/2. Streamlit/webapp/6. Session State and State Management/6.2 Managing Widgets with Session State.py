import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Managing Widgets with Session State",
    page_icon="🎛️",
    layout="wide",
)

st.title("🎛️ Managing Widgets with Session State — Complete Guide")
st.markdown(
    """
When you assign a `key` to any Streamlit widget, it automatically establishes **bi-directional synchronization** 
with `st.session_state`. Mastering this relationship is essential for synchronizing paired controls, 
building cascading dependent dropdowns, implementing programmatic form resets, 
and avoiding widget lifecycle initialization conflicts.
"""
)

st.divider()

# ==============================================================================
# 1. Assigning Widget Keys & 2. Accessing Widget Values
# ==============================================================================
st.header("1. Assigning Keys & 2. Accessing Widget Values", divider="rainbow")
st.markdown(
    """
Assigning `key="my_key"` does two things simultaneously:
1. **Registers State**: Exposes the widget's current live value at `st.session_state["my_key"]`.
2. **Access Duality**: You can read the value from either the **direct return variable** or from **`st.session_state`**.
- Reading from `st.session_state` allows you to access a widget's value **earlier in the script**, before that widget's code is reached!
"""
)

col_key1, col_key2 = st.columns(2)

with col_key1:
    st.markdown("**💻 Python Source Code (Key Registration & Dual Access)**")
    code_keys_demo = '''# 1. Assign explicit unique key:
model_name = st.text_input(
    label="Foundation Model Name",
    value="meta-llama/Llama-3.3-70B",
    key="cfg_model_name"
)

# 2. Access via direct return:
st.write(f"Direct Return: `{model_name}`")

# 3. Access via session state dictionary:
st.write(f"Session State: `{st.session_state['cfg_model_name']}`")'''
    st.code(code_keys_demo, language="python")

with col_key2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    m_name = st.text_input(
        label="Foundation Model Name",
        value="meta-llama/Llama-3.3-70B",
        key="demo_cfg_model_name"
    )
    st.caption(f"Direct Return: `{m_name}`")
    st.caption(f"Session State: `{st.session_state.get('demo_cfg_model_name')}`")

st.divider()

# ==============================================================================
# 3. Synchronizing Widgets (Bi-Directional Slider & Number Input)
# ==============================================================================
st.header("3. Synchronizing Paired Widgets (Two-Way Binding)", divider="rainbow")
st.markdown(
    """
A common UI pattern in ML consoles is pairing a **slider** and a **number input** for the same hyperparameter.
Using `on_change` callbacks, updating either widget immediately synchronizes the other!
"""
)

# State initialization for paired widgets
if "shared_batch_size" not in st.session_state:
    st.session_state["shared_batch_size"] = 64

def sync_from_slider():
    st.session_state["shared_batch_size"] = st.session_state["slider_batch_size"]

def sync_from_number():
    st.session_state["shared_batch_size"] = st.session_state["num_batch_size"]

col_sync1, col_sync2 = st.columns(2)

with col_sync1:
    st.markdown("**💻 Python Source Code (Bi-Directional Callback Sync)**")
    code_sync = '''# Central state variable:
if "batch_size" not in st.session_state:
    st.session_state["batch_size"] = 64

# Callback functions:
def update_from_slider():
    st.session_state["batch_size"] = st.session_state["bs_slider"]

def update_from_number():
    st.session_state["batch_size"] = st.session_state["bs_num"]

c1, c2 = st.columns(2)
with c1:
    st.slider("Batch Size (Slider)", 8, 256, 
              value=st.session_state["batch_size"],
              key="bs_slider", on_change=update_from_slider)
with c2:
    st.number_input("Batch Size (Numeric)", 8, 256, 
                    value=st.session_state["batch_size"],
                    key="bs_num", on_change=update_from_number)'''
    st.code(code_sync, language="python")

with col_sync2:
    st.markdown("**🖥️ Live Interactive Synchronized Controls**")
    sync_c1, sync_c2 = st.columns(2)
    with sync_c1:
        st.slider(
            "Batch Size (Slider)", 8, 256,
            value=st.session_state["shared_batch_size"],
            key="slider_batch_size",
            on_change=sync_from_slider
        )
    with sync_c2:
        st.number_input(
            "Batch Size (Numeric)", 8, 256,
            value=st.session_state["shared_batch_size"],
            key="num_batch_size",
            on_change=sync_from_number
        )
    st.info(f"Target Training Batch Size: **{st.session_state['shared_batch_size']}**")

st.divider()

# ==============================================================================
# 4. Setting Initial Values & 5. Resetting Widget Values
# ==============================================================================
st.header("4. Setting Initial Values & 5. Programmatic Form Resets", divider="rainbow")
st.markdown(
    """
- **Setting Initial Values**: Pre-initialize `st.session_state[key]` at the top of your script.
- **Resetting Values**: Directly modifying `st.session_state[key]` *inside a callback* on a button click cleanly resets the widget.
- **The Golden Rule**: Never mutate a widget's session key *after* that widget has rendered on the same script execution pass; mutate inside callbacks before the rerun executes.
"""
)

# Initialize defaults
if "experiment_author" not in st.session_state:
    st.session_state["experiment_author"] = "Dr. Alan Turing"
if "experiment_seed" not in st.session_state:
    st.session_state["experiment_seed"] = 42

def reset_experiment_form():
    st.session_state["experiment_author"] = "Dr. Alan Turing"
    st.session_state["experiment_seed"] = 42

col_rst1, col_rst2 = st.columns(2)

with col_rst1:
    st.markdown("**💻 Python Source Code (Clean Reset Callback)**")
    code_reset = '''# 1. State Pre-Initialization:
if "author" not in st.session_state:
    st.session_state["author"] = "Dr. Alan Turing"
if "seed" not in st.session_state:
    st.session_state["seed"] = 42

# 2. Reset Callback Function:
def reset_form_callback():
    st.session_state["author"] = "Dr. Alan Turing"
    st.session_state["seed"] = 42

# 3. Widgets bound to session keys:
st.text_input("Author", key="author")
st.number_input("Random Seed", key="seed")

# 4. Reset Button triggers callback:
st.button("↺ Reset to Defaults", on_click=reset_form_callback)'''
    st.code(code_reset, language="python")

with col_rst2:
    st.markdown("**🖥️ Live Interactive Form Reset Demo**")
    st.text_input("Author Name", key="experiment_author")
    st.number_input("Random Seed", min_value=1, max_value=99999, key="experiment_seed")
    st.button("↺ Reset to Standard Defaults", on_click=reset_experiment_form, key="btn_do_reset")
    st.caption(f"Current State: Author = **{st.session_state['experiment_author']}** | Seed = **{st.session_state['experiment_seed']}**")

st.divider()

# ==============================================================================
# 6. Managing Dependent (Cascading) Widgets
# ==============================================================================
st.header("6. Managing Dependent Widgets (Cascading Dropdowns)", divider="rainbow")
st.markdown(
    """
When options in Widget B depend on the choice made in Widget A (e.g. Cloud Provider → Available GPU Accelerators):
- When Widget A changes, Widget B's previously selected value might become invalid.
- Use an `on_change` callback on Widget A to automatically reset Widget B's selection to a safe default!
"""
)

# Cloud Hardware Hierarchical Catalog
hardware_catalog = {
    "AWS": ["p4de.24xlarge (8x A100)", "p5.48xlarge (8x H100)", "g5.12xlarge (4x A10G)"],
    "GCP": ["a2-highgpu-8g (8x A100)", "a3-highgpu-8g (8x H100)", "g2-standard-96 (8x L4)"],
    "Azure": ["ND96amsr_A100_v4", "ND96is_H100_v5", "NC96ads_A100_v4"]
}

if "selected_cloud" not in st.session_state:
    st.session_state["selected_cloud"] = "AWS"

def on_cloud_provider_change():
    # Reset instance selection when cloud provider changes
    new_cloud = st.session_state["selected_cloud"]
    st.session_state["selected_instance"] = hardware_catalog[new_cloud][0]

if "selected_instance" not in st.session_state:
    st.session_state["selected_instance"] = hardware_catalog["AWS"][0]

col_dep1, col_dep2 = st.columns(2)

with col_dep1:
    st.markdown("**💻 Python Source Code (Cascading Reset Pattern)**")
    code_cascading = '''def on_cloud_change():
    new_provider = st.session_state["cloud_provider"]
    # Safely reset downstream selection to first valid instance:
    st.session_state["gpu_instance"] = hardware_catalog[new_provider][0]

# Parent Widget:
cloud = st.selectbox(
    "Cloud Provider",
    options=list(hardware_catalog.keys()),
    key="cloud_provider",
    on_change=on_cloud_change
)

# Dependent Child Widget:
instance = st.selectbox(
    "Available GPU Instance",
    options=hardware_catalog[st.session_state["cloud_provider"]],
    key="gpu_instance"
)'''
    st.code(code_cascading, language="python")

with col_dep2:
    st.markdown("**🖥️ Live Cascading Dropdowns Demo**")
    st.selectbox(
        "Cloud Provider",
        options=list(hardware_catalog.keys()),
        key="selected_cloud",
        on_change=on_cloud_provider_change
    )
    
    current_instances = hardware_catalog[st.session_state["selected_cloud"]]
    st.selectbox(
        "Available GPU Instance",
        options=current_instances,
        key="selected_instance"
    )
    st.success(f"Provisioning Target: **{st.session_state['selected_cloud']}** → `{st.session_state['selected_instance']}`")

st.divider()

# ==============================================================================
# 7. Avoiding Conflicting Widget Initialization & 8. Persisting Selections
# ==============================================================================
st.header("7. Avoiding Initialization Conflicts & 8. Persisting Selections", divider="rainbow")
st.markdown(
    """
### ⚠️ The Conflicting Initialization Gotcha
If you initialize `st.session_state["my_key"] = "Foo"` AND also pass `st.text_input(..., value="Bar", key="my_key")`, 
Streamlit issues a warning:
> *The widget with key "my_key" was created with a default value but also had its value set in st.session_state.*

### ✅ The Clean Pattern
1. Pre-initialize in `st.session_state` once at the top of the file:
   ```python
   if "my_key" not in st.session_state:
       st.session_state["my_key"] = "Foo"
   ```
2. Omit the `value=` parameter entirely when declaring the widget:
   ```python
   st.text_input("Enter Value", key="my_key")  # Inherits from session state!
   ```
"""
)

if "clean_prompt_template" not in st.session_state:
    st.session_state["clean_prompt_template"] = "Act as an expert PyTorch performance engineer."

col_con1, col_con2 = st.columns(2)

with col_con1:
    st.markdown("**💻 Python Source Code (Clean Initialization Pattern)**")
    code_clean_pattern = '''# Step 1: Pre-initialize in session state:
if "clean_prompt" not in st.session_state:
    st.session_state["clean_prompt"] = "Act as an expert PyTorch engineer."

# Step 2: Declare widget WITHOUT value parameter (Clean!):
st.text_area(
    label="System Prompt Template",
    key="clean_prompt",  # Automatically pulls and syncs with session state
    height=90
)

st.write("Value persists across tab switches and button clicks!")'''
    st.code(code_clean_pattern, language="python")

with col_con2:
    st.markdown("**🖥️ Live Interactive Persistent Input**")
    st.text_area(
        label="System Prompt Template",
        key="clean_prompt_template",
        height=90
    )
    st.caption("💡 *Type changes above, click any button on this page, and notice how text persists seamlessly.*")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Widget State Management Rules of Thumb", divider="rainbow")
st.markdown(
    r"""
| Scenario | Golden Rule / Pattern |
| :--- | :--- |
| **Two-Way Sync** | Use `on_change` callbacks on both widgets to update a shared session key |
| **Programmatic Reset** | Modify `st.session_state[key]` inside a button callback (`on_click`), NOT after widget render |
| **Cascading Dropdowns** | Use `on_change` on parent dropdown to reset child widget key to first valid option |
| **Avoid Init Warnings** | Pre-initialize in `st.session_state` and **omit the `value` parameter** in the widget call |
| **Read Value Before Widget** | `st.session_state.get("key", default)` can be read anywhere in the script pass |
"""
)
