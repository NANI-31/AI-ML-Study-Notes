import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Understanding Session State Tutorial",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Understanding Session State (`st.session_state`) — Complete Guide")
st.markdown(
    """
Streamlit's reactive execution model reruns the entire Python script from top to bottom on every user interaction. 
Without state management, **local Python variables are wiped and reset on every single rerun**. 
**`st.session_state`** is Streamlit's official mechanism to persist data across reruns, maintain user-specific memory, 
accumulate conversation histories, track multi-step workflows, and coordinate widget interactions.
"""
)

st.divider()

# ==============================================================================
# 1. What is Session State? & 2. Why is Session State Needed?
# ==============================================================================
st.header("1. What is Session State? & 2. Why is It Needed?", divider="rainbow")
st.markdown(
    """
- **Definition**: `st.session_state` is a dictionary-like container that retains its values across multiple script runs for an individual browser tab.
- **Why It's Needed**: In standard web frameworks (Flask, Django), server state is preserved between HTTP requests via sessions/databases. In Streamlit, each click or widget change triggers a fresh script rerun. Without session state, stateful features like shopping carts, multi-turn LLM chat histories, or counters are impossible.
"""
)

st.divider()

# ==============================================================================
# 3. Understanding Reruns and Lost Variables (The Counter Comparison)
# ==============================================================================
st.header("3. Understanding Reruns & Lost Variables: Ephemeral vs. Persistent Memory", divider="rainbow")
st.markdown(
    """
Observe the profound difference below between a **naive local variable** and **`st.session_state`**:
- **Naive Variable**: Every time you click the button, the script reruns, resets `local_count = 0`, adds 1, and prints 1. It can **never exceed 1**!
- **Session State Variable**: Preserved across runs. Each click increments the persistent memory counter.
"""
)

# Initialize persistent counter
if "persistent_counter" not in st.session_state:
    st.session_state["persistent_counter"] = 0

col_cnt_code, col_cnt_live = st.columns(2)

with col_cnt_code:
    st.markdown("**💻 Python Source Code (Side-by-Side Comparison)**")
    code_counter_comp = '''# ❌ BROKEN: Naive local variable resets on every rerun!
naive_count = 0
if st.button("Increment Naive Variable"):
    naive_count += 1
st.write(f"Naive Count: {naive_count}")  # Always prints 1!

# ✅ CORRECT: Session state persists across reruns!
if "persistent_counter" not in st.session_state:
    st.session_state["persistent_counter"] = 0

if st.button("Increment Session State"):
    st.session_state["persistent_counter"] += 1
st.write(f"Persistent Count: {st.session_state['persistent_counter']}")'''
    st.code(code_counter_comp, language="python")

with col_cnt_live:
    st.markdown("**🖥️ Live Interactive Comparison Demo**")
    
    col_demo_a, col_demo_b = st.columns(2)
    
    with col_demo_a:
        with st.container(border=True):
            st.markdown("##### ❌ Naive Variable")
            naive_count = 0
            if st.button("Click: Naive Counter", key="btn_naive"):
                naive_count += 1
            st.metric("Naive Count", naive_count)
            st.caption("⚠️ Notice: Resets to 0 then becomes 1 every time!")

    with col_demo_b:
        with st.container(border=True):
            st.markdown("##### ✅ Session State Variable")
            if st.button("Click: Persistent Counter", key="btn_persist"):
                st.session_state["persistent_counter"] += 1
            st.metric("Session Count", st.session_state["persistent_counter"])
            st.caption("✅ Persists and increments correctly across reruns.")

st.divider()

# ==============================================================================
# 4. st.session_state API & 5. Session-Specific Data Isolation
# ==============================================================================
st.header("4. `st.session_state` API & 5. Session-Specific Isolation", divider="rainbow")
st.markdown(
    """
- **Dual Syntax**: Access via dictionary notation (`st.session_state["key"]`) or attribute notation (`st.session_state.key`).
- **Session Isolation**: State is strictly scoped to the **individual user's browser tab session**. 
  If User A increments their counter to 50, User B opening the app in another browser or tab starts fresh at 0. 
  Session state never leaks across different users.
"""
)

col_api1, col_api2 = st.columns(2)

with col_api1:
    st.markdown("**💻 Python Source Code (Dict vs. Attribute Syntax)**")
    code_api_syntax = '''# Dictionary-style access:
st.session_state["model_name"] = "Claude-3.5-Sonnet"
current_model = st.session_state["model_name"]

# Attribute-style access (Identical):
st.session_state.max_tokens = 4096
token_limit = st.session_state.max_tokens

# Inspect all registered session state keys:
st.write("Current Keys:", list(st.session_state.keys()))'''
    st.code(code_api_syntax, language="python")

with col_api2:
    st.markdown("**🖥️ Live Registered Session State Snapshot**")
    st.json({k: str(v) for k, v in st.session_state.items() if not k.startswith("_")})
    st.caption("💡 *Above is a live JSON dump of all keys currently registered in your unique session state!*")

st.divider()

# ==============================================================================
# 6. Initializing Session State
# ==============================================================================
st.header("6. The Canonical Initialization Pattern", divider="rainbow")
st.markdown(
    """
Attempting to read `st.session_state["key"]` before it has been set raises an `AttributeError` or `KeyError`.
The **canonical initialization pattern** checks for key existence at the top of your script:
```python
if "my_key" not in st.session_state:
    st.session_state["my_key"] = default_value
```
"""
)

col_init1, col_init2 = st.columns(2)

with col_init1:
    st.markdown("**💻 Python Source Code (Initializing Complex Structures)**")
    code_init = '''# Initialize primitives:
if "user_authenticated" not in st.session_state:
    st.session_state["user_authenticated"] = False

# Initialize collections (e.g. Chat History):
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

# Initialize DataFrames or experiment logs:
if "experiment_log" not in st.session_state:
    st.session_state["experiment_log"] = []'''
    st.code(code_init, language="python")

with col_init2:
    st.markdown("**🖥️ Rendered Architecture Best Practices**")
    st.info(
        """
        - **Placement**: Place initialization logic at the top of the script or inside a dedicated `init_state()` function.
        - **Data Types**: Session state can store any Python object (integers, strings, dicts, lists, Pandas DataFrames, PyTorch tensors, custom classes).
        - **Lifecycle**: State persists until the user closes or hard-refreshes the browser tab.
        """
    )

st.divider()

# ==============================================================================
# 7. Checking Whether a Key Exists & Safe Retrieval
# ==============================================================================
st.header("7. Checking Key Existence & Safe Retrieval (`in` & `.get`)", divider="rainbow")
st.markdown(
    """
There are two idiomatic ways to safely check and read session keys without throwing exceptions:
1. **The `in` Operator**: `if "api_key" in st.session_state:`
2. **The `.get()` Method**: `st.session_state.get("api_key", default="fallback")`
"""
)

col_chk1, col_chk2 = st.columns(2)

with col_chk1:
    st.markdown("**💻 Python Source Code**")
    code_safe_get = '''# 1. Membership test with 'in':
if "user_token" in st.session_state:
    st.success("Session Token Active!")
else:
    st.warning("Session Token Missing. Please authenticate.")

# 2. Safe retrieval with default fallback using .get():
theme = st.session_state.get("ui_theme", "Dark Mode (Default)")
st.write(f"Active Theme: {theme}")'''
    st.code(code_safe_get, language="python")

with col_chk2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if "user_auth_token" in st.session_state:
            st.success("Token Active: `jwt_live_99a8bc`")
            if st.button("Log Out (Clear Token)", key="btn_logout"):
                del st.session_state["user_auth_token"]
                st.rerun()
        else:
            st.warning("Token Missing: Unauthenticated")
            if st.button("Log In (Set Token)", key="btn_login"):
                st.session_state["user_auth_token"] = "jwt_live_99a8bc"
                st.rerun()

    with col_t2:
        safe_theme = st.session_state.get("custom_theme_mode", "High-Contrast Dark")
        st.caption(f"Retrieved via `.get()`: **{safe_theme}**")

st.divider()

# ==============================================================================
# 8. Reading, Updating, and Deleting Stored Values
# ==============================================================================
st.header("8. Reading, Updating & Mutating Stored Collections", divider="rainbow")
st.markdown(
    """
Session state structures can be mutated in-place or deleted:
- **In-place updates**: `st.session_state.chat_history.append(...)`
- **Arithmetic updates**: `st.session_state.token_spend += 0.05`
- **Key deletion**: `del st.session_state["temporary_key"]`
- **State reset**: Resetting all session keys to default values.
"""
)

# Initialize chat buffer demo
if "interactive_history" not in st.session_state:
    st.session_state["interactive_history"] = [
        "System: Initialized Claude-3.5 Assistant session.",
        "Assistant: Ready to assist with machine learning code."
    ]

col_mut1, col_mut2 = st.columns(2)

with col_mut1:
    st.markdown("**💻 Python Source Code (Mutating & Clearing Collections)**")
    code_mutating = '''# Append new item to list inside session state:
new_msg = st.text_input("Enter log message:")
if st.button("Append Message"):
    st.session_state["interactive_history"].append(f"User: {new_msg}")

# Clear collection:
if st.button("Clear Log History"):
    st.session_state["interactive_history"] = []
    st.rerun()'''
    st.code(code_mutating, language="python")

with col_mut2:
    st.markdown("**🖥️ Live Interactive List Mutation Demo**")
    
    user_entry = st.text_input("Append to conversation buffer:", placeholder="Type a message...", key="txt_msg_entry")
    
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("➕ Append to History", key="btn_append_msg"):
            if user_entry:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state["interactive_history"].append(f"User ({timestamp}): {user_entry}")
                st.rerun()
    with col_act2:
        if st.button("🗑️ Reset History", key="btn_clear_history"):
            st.session_state["interactive_history"] = [
                "System: Initialized Claude-3.5 Assistant session."
            ]
            st.rerun()

    with st.container(height=140, border=True):
        for msg in st.session_state["interactive_history"]:
            st.text(msg)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Session State Cheat Sheet", divider="rainbow")
st.markdown(
    r"""
| Task | Syntax | Example |
| :--- | :--- | :--- |
| **Check Existence** | `"key" in st.session_state` | `if "logged_in" not in st.session_state:` |
| **Safe Read** | `st.session_state.get(k, default)` | `temp = st.session_state.get("temp", 0.7)` |
| **Direct Read/Write** | `st.session_state["key"] = val` | `st.session_state["epochs"] = 50` |
| **Attribute Access** | `st.session_state.key = val` | `st.session_state.epochs += 1` |
| **Delete Key** | `del st.session_state["key"]` | `del st.session_state["auth_token"]` |
| **List Mutation** | `st.session_state.list.append(v)`| `st.session_state.messages.append(msg)` |
| **Session Isolation** | Scoped per browser tab | Independent memory per user/tab |
"""
)
