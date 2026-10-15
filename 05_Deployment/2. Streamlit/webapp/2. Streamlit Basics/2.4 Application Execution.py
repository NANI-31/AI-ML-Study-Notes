import streamlit as st
import time
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Application Execution & Lifecycle",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Streamlit Application Execution & Lifecycle — Complete Guide")
st.markdown(
    """
Unlike traditional event-driven GUI frameworks (like React, Qt, or Tkinter) or request-response web servers 
(like Flask or FastAPI), Streamlit employs a unique **Linear Script Execution & Rerun Model**.
Understanding how your Python code executes top-to-bottom, how widget interactions trigger script reruns, 
and how state survives across runs is fundamental to building reliable, high-performance applications.
"""
)

st.divider()

# ==============================================================================
# 1. Understanding Top-to-Bottom Execution
# ==============================================================================
st.header("1. Understanding Top-to-Bottom Execution", divider="rainbow")
st.markdown(
    """
### 🧠 The Mental Model: Pure Sequential Python
In Streamlit, **your script is the webpage**. There is no hidden event loop or separate rendering engine:
- Every time a user loads the app, Python executes the script sequentially from **Line 1 down to the last line**.
- Variables are declared, calculated, and discarded during the execution flow.
- UI elements appear in the exact chronological order in which their corresponding `st.*` functions are called.
"""
)

col_t2b1, col_t2b2 = st.columns(2)

with col_t2b1:
    st.markdown("**💻 Python Source Code**")
    code_t2b = '''# Sequential Top-to-Bottom Execution Flow:
st.write("1. 🚀 Application Initialized")

# Step 2: Intermediate computation
calc_result = 42 * 2
st.write(f"2. ⚙️ Intermediate Calculation: 42 * 2 = {calc_result}")

# Step 3: Conditional execution
if calc_result > 50:
    st.success("3. ✅ Condition Met: Result exceeds threshold!")

st.write("4. 🏁 Script Execution Finished")'''
    st.code(code_t2b, language="python")

with col_t2b1:
    st.caption("Each line executes strictly in order. Variables exist only in their local scope during the pass.")

with col_t2b2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.write("1. 🚀 Application Initialized")
    calc_res = 42 * 2
    st.write(f"2. ⚙️ Intermediate Calculation: 42 * 2 = {calc_res}")
    if calc_res > 50:
        st.success("3. ✅ Condition Met: Result exceeds threshold!")
    st.write("4. 🏁 Script Execution Finished")

st.divider()

# ==============================================================================
# 2. Understanding Reruns: Local Variables vs Session State
# ==============================================================================
st.header("2. Understanding Reruns & State Lifecycle", divider="rainbow")
st.markdown(
    """
### 🔄 What Triggers a Rerun?
Whenever a user interacts with **any widget** (clicks a button, drags a slider, enters text, or changes a dropdown), 
Streamlit immediately **reruns the entire Python script from Line 1**.

### ⚠️ The Critical State Pitfall:
- **Local Variables (`x = 0`)**: Reset back to their initial default value on every rerun!
- **`st.session_state`**: Survives across reruns, retaining user modifications and cumulative calculations!
- **Programmatic Control**: Use `st.rerun()` to restart execution immediately, or `st.stop()` to halt further execution early.
"""
)

st.markdown("#### 📐 Formal Signatures: `st.rerun()` and `st.stop()`")
st.code(
    """st.rerun(*, scope: "app" | "fragment" = "app") -> NoReturn  # Re-executes the script immediately
st.stop() -> NoReturn                                      # Halts execution at the exact line called""",
    language="python"
)

# Initialize session state counter if not present
if "persistent_counter" not in st.session_state:
    st.session_state["persistent_counter"] = 0

col_rr1, col_rr2 = st.columns(2)

with col_rr1:
    st.markdown("**💻 Python Source Code (Local vs. Session State)**")
    code_rr = '''# A. Local variable (Resets to 0 on EVERY rerun):
local_counter = 0

# B. Session State variable (Persists across reruns):
if "persistent_counter" not in st.session_state:
    st.session_state["persistent_counter"] = 0

btn_clicked = st.button("Increment Counters", key="btn_inc_demo")
if btn_clicked:
    local_counter += 1
    st.session_state["persistent_counter"] += 1

st.write(f"Local Variable Value: {local_counter}")
st.write(f"Session State Value: {st.session_state['persistent_counter']}")'''
    st.code(code_rr, language="python")

with col_rr2:
    st.markdown("**🖥️ Live Interactive Counter Demo *(Click to observe reset!)*:**")
    
    local_counter = 0
    if st.button("⚡ Click to Trigger Rerun & Increment", key="btn_inc_demo"):
        local_counter += 1
        st.session_state["persistent_counter"] += 1

    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Local Variable (Resets)", f"{local_counter}", delta="Never exceeds 1", delta_color="inverse", border=True)
    c_m2.metric("Session State (Persists)", f"{st.session_state['persistent_counter']}", delta=f"+{st.session_state['persistent_counter']} Total", border=True)

    if st.button("🔄 Reset Persistent Counter", key="btn_reset_demo"):
        st.session_state["persistent_counter"] = 0
        st.rerun()

st.divider()

# ==============================================================================
# 3. Using If and Else in Apps (Conditional Rendering & Routing)
# ==============================================================================
st.header("3. Conditional Rendering with `if`, `elif`, and `else`", divider="rainbow")
st.markdown(
    """
Because Streamlit executes top-to-bottom, standard Python control flow statements (`if`, `elif`, `else`) 
function as **declarative view switchers and navigation routers**.
You don't need complex routing frameworks or URL dispatchers:
"""
)

col_if1, col_if2 = st.columns(2)

with col_if1:
    st.markdown("**💻 Python Source Code**")
    code_if = '''# User Mode Switcher via st.radio:
user_tier = st.radio(
    "Select Subscription Tier:",
    options=["Free Community", "Developer Pro", "Enterprise Dedicated"],
    horizontal=True
)

if user_tier == "Free Community":
    st.info("ℹ️ Community: Rate limited to 10 queries / minute. Standard CPU.")
elif user_tier == "Developer Pro":
    st.success("⚡ Developer Pro: 1,000 queries / minute. NVIDIA A10G GPU access.")
else:
    st.warning("🏢 Enterprise: Dedicated H100 Cluster, 99.99% SLA & Custom VPC Peering.")'''
    st.code(code_if, language="python")

with col_if2:
    st.markdown("**🖥️ Rendered Interactive Output**")
    user_tier_select = st.radio(
        "Select Subscription Tier:",
        options=["Free Community", "Developer Pro", "Enterprise Dedicated"],
        horizontal=True,
        key="sub_tier_radio"
    )

    if user_tier_select == "Free Community":
        st.info("ℹ️ **Community**: Rate limited to 10 queries / minute. Standard CPU shared queue.")
    elif user_tier_select == "Developer Pro":
        st.success("⚡ **Developer Pro**: 1,000 queries / minute. Dedicated NVIDIA A10G GPU access.")
    else:
        st.warning("🏢 **Enterprise**: Multi-node H100 Cluster, 99.99% SLA & Custom VPC Peering enabled.")

st.divider()

# ==============================================================================
# 4. Using Loops in Streamlit & The Widget Key Collision Rule
# ==============================================================================
st.header("4. Using Loops in Streamlit (`for` & `while`)", divider="rainbow")
st.markdown(
    """
Loops dynamically render repeated UI structures like metric cards, expanders, or tables from lists and dictionaries.

> [!CAUTION]
> **The Golden Rule of Streamlit Loops: Unique Widget Keys**  
> If you instantiate an interactive widget (e.g. `st.button`, `st.text_input`, `st.checkbox`) inside a loop, 
> Streamlit requires every instance to have a **unique `key` parameter**. Failing to provide a unique key 
> will throw a `DuplicateWidgetID` exception!
"""
)

sample_models = [
    {"name": "Llama-3-8B", "type": "Open Weights", "ram": "16 GB", "speed": "120 tps"},
    {"name": "Mistral-7B", "type": "Open Weights", "ram": "14 GB", "speed": "145 tps"},
    {"name": "Phi-3-Mini", "type": "Edge SLM", "ram": "8 GB", "speed": "210 tps"},
]

col_loop1, col_loop2 = st.columns(2)

with col_loop1:
    st.markdown("**💻 Python Source Code (Loop with Unique Keys)**")
    code_loop = '''models = [
    {"name": "Llama-3-8B", "type": "Open Weights", "ram": "16 GB", "speed": "120 tps"},
    {"name": "Mistral-7B", "type": "Open Weights", "ram": "14 GB", "speed": "145 tps"},
    {"name": "Phi-3-Mini", "type": "Edge SLM", "ram": "8 GB", "speed": "210 tps"},
]

cols = st.columns(len(models))

for i, model in enumerate(models):
    with cols[i]:
        st.markdown(f"### {model['name']}")
        st.caption(f"Category: {model['type']}")
        st.write(f"RAM: `{model['ram']}` | Speed: `{model['speed']}`")
        
        # MUST supply unique key=f"btn_deploy_{i}" to avoid DuplicateWidgetID:
        if st.button("Deploy Model", key=f"btn_deploy_{i}"):
            st.success(f"Deployed {model['name']}!")'''
    st.code(code_loop, language="python")

with col_loop2:
    st.markdown("**🖥️ Rendered Dynamic Grid Output**")
    cols_grid = st.columns(len(sample_models))
    for idx, m_info in enumerate(sample_models):
        with cols_grid[idx]:
            with st.container(border=True):
                st.markdown(f"**{m_info['name']}**")
                st.caption(f"Category: {m_info['type']}")
                st.write(f"RAM: `{m_info['ram']}`")
                st.write(f"Speed: `{m_info['speed']}`")
                if st.button("Deploy Model", key=f"btn_deploy_{idx}"):
                    st.toast(f"✅ Initiated deployment for {m_info['name']}!")

st.divider()

# ==============================================================================
# 5. Defining Functions in Streamlit Applications
# ==============================================================================
st.header("5. Defining Functions (Separation of Concerns)", divider="rainbow")
st.markdown(
    """
Structuring code into modular Python functions brings two major benefits in Streamlit:
1. **Clean Code Architecture**: Separates pure business logic / data transformations from UI rendering.
2. **Caching Compatibility**: Functions can be wrapped with `@st.cache_data` or `@st.cache_resource` 
   so heavy calculations are computed **once** and reused across reruns!
"""
)

col_fn1, col_fn2 = st.columns(2)

with col_fn1:
    st.markdown("**💻 Python Source Code**")
    code_fn = '''# 1. Pure computational function (cached to prevent rerun lag)
@st.cache_data
def calculate_model_pricing(tokens_millions: float, cost_per_m: float) -> dict:
    total_cost = tokens_millions * cost_per_m
    discount = 0.15 if tokens_millions >= 10.0 else 0.0
    final_cost = total_cost * (1.0 - discount)
    return {
        "raw_cost": total_cost,
        "discount_pct": discount * 100,
        "final_cost": final_cost
    }

# 2. Invocation in main UI script
res = calculate_model_pricing(tokens_millions=15.0, cost_per_m=2.50)
st.write(f"Estimated Monthly Bill: **${res['final_cost']:,.2f}**")'''
    st.code(code_fn, language="python")

with col_fn2:
    st.markdown("**🖥️ Rendered Function Execution**")
    
    @st.cache_data
    def calculate_model_pricing_demo(tokens_m: float, rate_per_m: float) -> dict:
        total = tokens_m * rate_per_m
        disc = 0.15 if tokens_m >= 10.0 else 0.0
        final = total * (1.0 - disc)
        return {"raw": total, "disc": disc * 100, "final": final}

    tokens_input = st.slider("Monthly Token Volume (Millions):", min_value=1.0, max_value=50.0, value=15.0, step=1.0, key="tok_slider")
    pricing_res = calculate_model_pricing_demo(tokens_input, rate_per_m=2.50)

    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("Base Cost", f"${pricing_res['raw']:,.2f}", border=True)
    f_col2.metric("Volume Discount", f"{pricing_res['disc']:.0f}%", border=True)
    f_col3.metric("Final Projected Bill", f"${pricing_res['final']:,.2f}", delta="-15%" if pricing_res['disc'] > 0 else "No Discount", border=True)

st.divider()

# ==============================================================================
# 6. Creating Reusable UI Functions (Component Pattern)
# ==============================================================================
st.header("6. Creating Reusable UI Functions (Component Pattern)", divider="rainbow")
st.markdown(
    """
You can encapsulate recurring visual layouts (cards, status banners, metric rows) into **reusable UI helper functions**.
Using `with st.container(border=True):` creates clean card boundaries:
"""
)

col_uifn1, col_uifn2 = st.columns(2)

with col_uifn1:
    st.markdown("**💻 Python Source Code (Reusable UI Card Function)**")
    code_uifn = '''def render_service_card(service_name: str, status: str, latency_ms: float, qps: int):
    """Reusable visual card component for system services."""
    status_color = ":green[HEALTHY]" if status == "UP" else ":red[DEGRADED]"
    with st.container(border=True):
        col_hdr1, col_hdr2 = st.columns([3, 1])
        col_hdr1.markdown(f"#### 🖥️ {service_name}")
        col_hdr2.markdown(f"**{status_color}**")
        
        c1, c2 = st.columns(2)
        c1.metric("Latency", f"{latency_ms} ms")
        c2.metric("Throughput", f"{qps:,} QPS")

# Reusing the component across multiple services:
render_service_card("Embeddings Service", "UP", 14.2, 4200)
render_service_card("Vector Search Engine", "UP", 8.6, 6800)
render_service_card("Reranker Microservice", "DOWN", 450.0, 120)'''
    st.code(code_uifn, language="python")

with col_uifn2:
    st.markdown("**🖥️ Rendered Reusable Component Cards**")
    
    def render_service_card_demo(service_name: str, status: str, latency_ms: float, qps: int):
        status_color = ":green[HEALTHY]" if status == "UP" else ":red[DEGRADED]"
        with st.container(border=True):
            col_hdr1, col_hdr2 = st.columns([3, 1])
            col_hdr1.markdown(f"##### 🖥️ {service_name}")
            col_hdr2.markdown(f"**{status_color}**")
            
            c1, c2 = st.columns(2)
            c1.metric("Latency", f"{latency_ms} ms")
            c2.metric("Throughput", f"{qps:,} QPS")

    render_service_card_demo("Embeddings Service", "UP", 14.2, 4200)
    render_service_card_demo("Vector Search Engine", "UP", 8.6, 6800)
    render_service_card_demo("Reranker Microservice", "DOWN", 450.0, 120)

st.divider()

# ==============================================================================
# 7. Understanding Script Errors in Streamlit
# ==============================================================================
st.header("7. Understanding & Handling Script Errors", divider="rainbow")
st.markdown(
    """
Common errors encountered during Streamlit development include:
1. **`DuplicateWidgetID`**: Occurs when two widgets share the same label and type without distinct `key` arguments.
2. **`KeyError` / `AttributeError`**: Attempting to read a missing key in `st.session_state` before initializing it.
3. **`StreamlitAPIException`**: Calling Streamlit commands from an asynchronous background thread without a `ScriptRunContext`.
4. **Unhandled Runtime Exceptions**: Crashes the page and presents an unsightly stack trace to end users.
"""
)

col_err1, col_err2 = st.columns(2)

with col_err1:
    st.markdown("**💻 Python Source Code (Graceful Exception Handling)**")
    code_err = '''# Robust error handling pattern:
def execute_database_query(query: str):
    if "DROP" in query.upper():
        raise PermissionError("Destructive query rejected by security policy.")
    return [{"id": 1, "status": "ACTIVE"}]

user_query = "DROP TABLE users;"

try:
    results = execute_database_query(user_query)
    st.write(results)
except PermissionError as pe:
    st.error(f"⛔ Security Violation: {pe}", icon="🚨")
except Exception as e:
    st.exception(e)  # Renders styled interactive traceback'''
    st.code(code_err, language="python")

with col_err2:
    st.markdown("**🖥️ Rendered Error Display (`st.error` & `st.exception`)**")
    
    test_error_mode = st.radio("Simulate Error Condition:", ["Clean Execution", "Security Violation", "Unexpected Bug"], horizontal=True, key="err_sim_radio")
    
    if test_error_mode == "Clean Execution":
        st.success("✅ Query executed successfully: 48 rows returned.")
    elif test_error_mode == "Security Violation":
        st.error("⛔ Security Violation: Destructive DROP TABLE query rejected by security policy.", icon="🚨")
    else:
        try:
            # Simulate zero division
            zero_div = 100 / 0
        except ZeroDivisionError as err:
            st.exception(err)

st.divider()

# ==============================================================================
# 8. Debugging Basic Applications (st.echo, Session State Inspector)
# ==============================================================================
st.header("8. Debugging Basic Applications (`st.echo` & State Inspector)", divider="rainbow")
st.markdown(
    """
Streamlit provides built-in utilities designed specifically for rapid visual debugging:
- **`st.echo()`**: A unique context manager that displays the exact Python source code and executes it simultaneously!
- **`st.session_state` JSON Inspector**: Expanding the current session dictionary to verify what state keys currently exist.
- **`st.stop()`**: Temporarily halts script execution to freeze and inspect variables up to a given line.
"""
)

st.markdown("#### 📐 Formal Signature: `st.echo()`")
st.code(
    """with st.echo(code_location: "above" | "below" = "above"):
    # Code written here is printed as a code block AND executed!""",
    language="python"
)

st.markdown("#### 🧪 Live Demonstration of `st.echo()`")

# Using st.echo live
with st.echo():
    # Everything inside this block is both printed to the user AND executed!
    raw_payload = {"model": "llama-3-8b", "batch_size": 32, "precision": "bfloat16"}
    computed_memory_gb = raw_payload["batch_size"] * 0.45
    st.write(f"Calculated GPU VRAM required: **{computed_memory_gb:.2f} GB**")

st.markdown("#### 🔍 Live Application State Inspector")
with st.expander("🛠️ Inspect Current `st.session_state` Dictionary (Debug Tool)", expanded=False):
    st.markdown("All active session keys and values in memory for your browser session:")
    st.json(dict(st.session_state))

st.divider()

# ==============================================================================
# Summary Checklist
# ==============================================================================
st.header("📋 Execution Model Summary Checklist", divider="rainbow")
st.markdown(
    """
| Concept | Behavior in Streamlit | Best Practice |
| :--- | :--- | :--- |
| **Top-to-Bottom Execution** | Script runs linearly from line 1 to EOF. | Keep main script modular; avoid heavy top-level calculations. |
| **Widget Reruns** | Any widget interaction restarts the whole script. | Wrap expensive functions in `@st.cache_data` or `@st.cache_resource`. |
| **Local Variables** | Reset to defaults on every single rerun. | Store values that must persist across clicks in `st.session_state`. |
| **Loops & Dynamic UI** | Renders items in loop order sequentially. | Always provide unique `key=f"item_{i}"` on widgets inside loops. |
| **Conditional Flow** | `if/elif/else` acts as page/view routers. | Use radio buttons or selectboxes to conditionally render distinct views. |
| **Debugging Tools** | Built-in `st.echo()`, `st.exception()`, `st.stop()`. | Inspect `dict(st.session_state)` to verify state health during development. |
"""
)
