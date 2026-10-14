import streamlit as st
import re

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Text Input Widgets Tutorial",
    page_icon="⌨️",
    layout="wide",
)

st.title("⌨️ Text Input Widgets — Complete Syntax & Validation Guide")
st.markdown(
    """
Text input widgets allow users to enter free-form text, credentials, prompts, and configuration parameters. 
Streamlit provides two primary primitives: **`st.text_input()`** for single-line inputs and 
**`st.text_area()`** for multi-line documents and prompt templates.
This guide details syntax, parameters, validation algorithms, character limits, and empty-state guard clauses.
"""
)

st.divider()

# ==============================================================================
# 1. st.text_input() (Single-Line Input)
# ==============================================================================
st.header("1. Single-Line Text Input (`st.text_input`)", divider="rainbow")
st.markdown(
    """
`st.text_input()` renders an HTML `<input type="text">` element. 
In Streamlit, text changes are submitted to Python when the user presses **Enter** or **clicks outside (blur)**.
"""
)

st.markdown("#### 📐 Formal Signature: `st.text_input()`")
st.code(
    """st.text_input(
    label: str,
    value: str = "",
    max_chars: int | None = None,
    key: str | int | None = None,
    type: "default" | "password" | "email" | "url" | "search" = "default",
    help: str | None = None,
    autocomplete: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    placeholder: str | None = None,
    disabled: bool = False,
    required: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    icon: str | None = None,
    width: "stretch" | "content" = "stretch"
) -> str""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.text_input()`")
st.markdown(
    """
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`label`** | `str` | *Required* | Label displayed above the input element. |
| **`value`** | `str` | `""` | Initial pre-populated text value. |
| **`placeholder`** | `str` | `None` | Faint hint text shown when the field is empty. |
| **`type`** | `str` | `"default"` | `"default"`, `"password"` (masks text), `"email"`, `"url"`, `"search"`. |
| **`max_chars`** | `int` | `None` | Maximum character limit (displays live counter). |
| **`required`** | `bool` | `False` | Renders an asterisk (`*`) indicating mandatory input. |
| **`label_visibility`** | `str` | `"visible"` | `"visible"`, `"hidden"` (accessible to screen readers), `"collapsed"`. |
| **`help`** | `str` | `None` | Tooltip icon explaining the expected format. |
"""
)

col_ti1, col_ti2 = st.columns(2)

with col_ti1:
    st.markdown("**💻 Python Source Code**")
    code_text_input = '''# Single-line text input:
model_endpoint = st.text_input(
    label="Inference API Endpoint URL",
    value="https://api.openai.com/v1",
    placeholder="https://api.your-domain.com/v1",
    help="Target host URL for model completions",
    icon="🌐"
)

st.write(f"Configured Endpoint: `{model_endpoint}`")'''
    st.code(code_text_input, language="python")

with col_ti2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    model_endpoint = st.text_input(
        label="Inference API Endpoint URL",
        value="https://api.openai.com/v1",
        placeholder="https://api.your-domain.com/v1",
        help="Target host URL for model completions",
        icon="🌐",
        key="ti_demo_endpoint"
    )
    st.caption(f"Active Target: `{model_endpoint}`")

st.divider()

# ==============================================================================
# 2. st.text_area() (Multi-Line Text Input)
# ==============================================================================
st.header("2. Multi-Line Text Input (`st.text_area`)", divider="rainbow")
st.markdown(
    """
`st.text_area()` renders an expandable `<textarea>` element for long-form content like prompts, 
code snippets, Markdown notes, and CSV payloads:
- In text areas, pressing **Enter** inserts a newline.
- Pressing **Ctrl + Enter** (or **Cmd + Enter**) commits the text and triggers an immediate script rerun.
- Use the **`height`** parameter to specify initial pixel height (default is ~100px).
"""
)

st.markdown("#### 📐 Formal Signature: `st.text_area()`")
st.code(
    """st.text_area(
    label: str,
    value: str = "",
    height: int | None = None,
    max_chars: int | None = None,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    placeholder: str | None = None,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    width: "stretch" | "content" = "stretch"
) -> str""",
    language="python",
)

col_ta1, col_ta2 = st.columns(2)

with col_ta1:
    st.markdown("**💻 Python Source Code**")
    code_text_area = '''# Multi-line system prompt configuration:
default_prompt = """You are an expert Machine Learning engineer.
Always output verified, production-grade Python code.
Include formal type annotations and docstrings."""

sys_prompt = st.text_area(
    label="System Prompt Template",
    value=default_prompt,
    height=140,
    placeholder="Enter custom instructions for the LLM...",
    help="Defines foundational behavior and persona"
)

st.write(f"Character count: {len(sys_prompt)} characters")'''
    st.code(code_text_area, language="python")

with col_ta2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    default_prompt_sample = """You are an expert Machine Learning engineer.
Always output verified, production-grade Python code.
Include formal type annotations and docstrings."""

    sys_prompt = st.text_area(
        label="System Prompt Template",
        value=default_prompt_sample,
        height=140,
        placeholder="Enter custom instructions for the LLM...",
        help="Defines foundational behavior and persona",
        key="ta_demo_prompt"
    )
    st.caption(f"Payload Size: **{len(sys_prompt)}** characters | Lines: **{len(sys_prompt.splitlines())}**")

st.divider()

# ==============================================================================
# 3. Placeholder Text vs Default Value
# ==============================================================================
st.header("3. Placeholder Text vs. Default Value", divider="rainbow")
st.markdown(
    """
Understanding the distinction between `placeholder` and `value` is critical:
- **`placeholder`**: Faint gray hint text that appears only when the field is completely empty. 
  It is **never returned** by the widget (the function returns `""`).
- **`value`**: The actual pre-populated initial content. If the user does not edit it, 
  the function returns this value as a valid string.
"""
)

col_pv1, col_pv2 = st.columns(2)

with col_pv1:
    st.markdown("**💻 Python Source Code**")
    code_pv = '''# A. Field with placeholder only (Returns empty string until typed)
input_ph = st.text_input("User Name (Placeholder Only)", placeholder="e.g. Jane Doe")
st.write("Return value A:", repr(input_ph))

# B. Field with default value (Returns 'gpt-4o' immediately)
input_val = st.text_input("Model ID (Default Value)", value="gpt-4o")
st.write("Return value B:", repr(input_val))'''
    st.code(code_pv, language="python")

with col_pv2:
    st.markdown("**🖥️ Rendered Comparison**")
    input_ph = st.text_input("User Name (Placeholder Only)", placeholder="e.g. Jane Doe", key="demo_ph_only")
    st.code(f"Return value: {repr(input_ph)} (Boolean: {bool(input_ph)})", language="python")

    input_val = st.text_input("Model ID (Default Value)", value="gpt-4o", key="demo_val_only")
    st.code(f"Return value: {repr(input_val)} (Boolean: {bool(input_val)})", language="python")

st.divider()

# ==============================================================================
# 4. Default Values & Programmatic State Resetting
# ==============================================================================
st.header("4. Default Values & Two-Way State Binding", divider="rainbow")
st.markdown(
    """
You can read and update input field values programmatically through **`st.session_state`**.
Setting `st.session_state[key]` updates the displayed text when paired with `st.rerun()`:
"""
)

if "search_query" not in st.session_state:
    st.session_state["search_query"] = "Transformer Attention Heads"

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**💻 Python Source Code (Programmatic Preset Buttons)**")
    code_presets = '''if "search_query" not in st.session_state:
    st.session_state["search_query"] = "Transformer Attention Heads"

# Preset action buttons:
p1, p2, p3 = st.columns(3)
if p1.button("Preset: RAG"):
    st.session_state["search_query"] = "Retrieval Augmented Generation"
    st.rerun()
if p2.button("Preset: LoRA"):
    st.session_state["search_query"] = "Low-Rank Adaptation Fine-Tuning"
    st.rerun()
if p3.button("Clear"):
    st.session_state["search_query"] = ""
    st.rerun()

query = st.text_input("Research Topic", key="search_query")'''
    st.code(code_presets, language="python")

with col_sb2:
    st.markdown("**🖥️ Rendered Interactive Presets**")
    p1, p2, p3 = st.columns(3)
    if p1.button("Preset: RAG", key="btn_rag"):
        st.session_state["search_query"] = "Retrieval Augmented Generation"
        st.rerun()
    if p2.button("Preset: LoRA", key="btn_lora"):
        st.session_state["search_query"] = "Low-Rank Adaptation Fine-Tuning"
        st.rerun()
    if p3.button("Clear Input", key="btn_clear_q"):
        st.session_state["search_query"] = ""
        st.rerun()

    query_val = st.text_input("Research Topic", key="search_query")
    st.caption(f"Active Query: **{query_val or '(Empty)'}**")

st.divider()

# ==============================================================================
# 5. Input Validation (Regex Patterns & Visual Feedback)
# ==============================================================================
st.header("5. Real-Time Input Validation", divider="rainbow")
st.markdown(
    """
Validating user input before sending it to APIs prevents runtime failures and security vulnerabilities:
- **Email Validation**: Using RFC 5322 regex pattern matching.
- **API Key Format**: Checking prefix patterns (e.g. `sk-proj-...`).
- **Visual Feedback**: Using `st.success()`, `st.warning()`, and `st.error()`.
"""
)

col_val1, col_val2 = st.columns(2)

with col_val1:
    st.markdown("**💻 Python Source Code (Regex Validation)**")
    code_val = r'''email_input = st.text_input("Work Email Address", placeholder="engineer@enterprise.ai")

# Regex for email validation:
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

if email_input:
    if re.match(EMAIL_REGEX, email_input):
        st.success("✅ Valid corporate email address format.")
    else:
        st.error("❌ Invalid email format. Expected: user@domain.com")
else:
    st.info("ℹ️ Please enter an email to validate.")'''
    st.code(code_val, language="python")

with col_val2:
    st.markdown("**🖥️ Rendered Live Validator**")
    email_test = st.text_input("Work Email Address", placeholder="engineer@enterprise.ai", key="val_email_test")
    EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    if email_test:
        if re.match(EMAIL_REGEX, email_test):
            st.success("✅ Valid corporate email address format.")
        else:
            st.error("❌ Invalid email format. Expected: user@domain.com")
    else:
        st.info("ℹ️ Please enter an email to test validation.")

st.divider()

# ==============================================================================
# 6. Maximum Character Lengths (max_chars)
# ==============================================================================
st.header("6. Maximum Character Lengths (`max_chars`)", divider="rainbow")
st.markdown(
    """
Setting **`max_chars=N`** enforces a strict length ceiling:
- The browser physically prevents typing beyond $N$ characters.
- Streamlit automatically displays an **interactive remaining-character counter** in the bottom-right corner.
- Essential for short descriptions, SMS prompts, metadata tags, and token-constrained queries.
"""
)

col_mc1, col_mc2 = st.columns(2)

with col_mc1:
    st.markdown("**💻 Python Source Code**")
    code_mc = '''# Enforce maximum character lengths:
short_summary = st.text_area(
    label="Model Card Executive Summary (Max 120 chars)",
    placeholder="Provide a concise 1-sentence synopsis...",
    max_chars=120,
    height=90
)

st.write(f"Remaining: {120 - len(short_summary)} characters")'''
    st.code(code_mc, language="python")

with col_mc2:
    st.markdown("**🖥️ Rendered Output with Live Character Counter**")
    short_summary_demo = st.text_area(
        label="Model Card Executive Summary (Max 120 chars)",
        placeholder="Provide a concise 1-sentence synopsis...",
        max_chars=120,
        height=90,
        key="mc_demo_area"
    )
    st.caption(f"Characters: **{len(short_summary_demo)} / 120** | Remaining: **{120 - len(short_summary_demo)}**")

st.divider()

# ==============================================================================
# 7. Password Input Fields (type="password")
# ==============================================================================
st.header("7. Password & Secret Inputs (`type='password'`)", divider="rainbow")
st.markdown(
    """
Setting **`type="password"`** masks typed characters behind privacy dots:
- Prevents shoulder-surfing of API keys, tokens, and database passwords.
- Renders an eye icon toggle button allowing users to reveal or hide typed text on demand.
"""
)

col_pwd1, col_pwd2 = st.columns(2)

with col_pwd1:
    st.markdown("**💻 Python Source Code (API Key Ingestion)**")
    code_pwd = '''# Secret API key input with masking:
api_key = st.text_input(
    label="OpenAI Secret API Key",
    type="password",
    placeholder="sk-proj-...",
    help="Your token is kept strictly in-memory during this session."
)

if api_key:
    # Safely show masked preview:
    masked_preview = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
    st.success(f"Key loaded: `{masked_preview}`")'''
    st.code(code_pwd, language="python")

with col_pwd2:
    st.markdown("**🖥️ Rendered Password Input**")
    api_key_test = st.text_input(
        label="OpenAI Secret API Key",
        type="password",
        placeholder="sk-proj-...",
        help="Your token is kept strictly in-memory during this session.",
        key="pwd_demo_input"
    )

    if api_key_test:
        masked = f"{api_key_test[:7]}...{api_key_test[-4:]}" if len(api_key_test) > 12 else "****"
        st.success(f"Key securely loaded: `{masked}`")
    else:
        st.caption("🔒 Characters are masked automatically with a reveal toggle.")

st.divider()

# ==============================================================================
# 8. Handling Empty Inputs & Guard Clauses (st.stop)
# ==============================================================================
st.header("8. Handling Empty Inputs & Guard Clauses (`st.stop`)", divider="rainbow")
st.markdown(
    """
Empty inputs can cause unhandled exceptions downstream. Best practices include:
1. **Button Disabling**: Disable action buttons when required inputs are blank (`disabled=not text.strip()`).
2. **Guard Clauses (`st.stop()`)**: Pause execution gracefully until mandatory input is provided.
"""
)

col_emp1, col_emp2 = st.columns(2)

with col_emp1:
    st.markdown("**💻 Python Source Code (Guard Clause Pattern)**")
    code_guard = '''user_prompt = st.text_input("Enter your prompt for inference:", placeholder="Ask a question...")

# 1. Guard clause: Stop execution early if input is missing
if not user_prompt.strip():
    st.warning("⚠️ Please provide a prompt before submitting.")
    st.stop()  # Prevents executing the code below!

# 2. Downstream execution (Only reached if prompt is NOT empty)
st.success(f"Sending prompt to LLM: '{user_prompt}'")
st.write("Generating streaming response...")'''
    st.code(code_guard, language="python")

with col_emp2:
    st.markdown("**🖥️ Live Interactive Guard Clause Demo**")
    guard_prompt = st.text_input("Enter your prompt for inference:", placeholder="Ask a question...", key="guard_demo_input")

    col_btn_sub, col_status = st.columns([1, 2])
    with col_btn_sub:
        submit_btn = st.button("Submit Query", disabled=not bool(guard_prompt.strip()), type="primary", key="btn_guard_sub")

    if not guard_prompt.strip():
        st.info("ℹ️ Button is disabled until you type at least 1 non-whitespace character.")
    else:
        if submit_btn:
            st.success(f"Query Dispatched: `{guard_prompt}`")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Text Input Widgets Cheat Sheet", divider="rainbow")
st.markdown(
    """
| Requirement | Recommended Component & Configuration |
| :--- | :--- |
| **Short String / Label** | `st.text_input("Name", placeholder="John Doe")` |
| **API Keys / Passwords** | `st.text_input("API Key", type="password")` |
| **Multi-line Text / Markdown** | `st.text_area("Notes", height=150)` |
| **Length Constraint** | `st.text_input("Slug", max_chars=30)` |
| **Pre-populated Preset** | `st.text_input("Search", value="Initial Query")` |
| **Mandatory Field Notice** | `st.text_input("Username", required=True)` |
| **Early Exit on Empty** | `if not text.strip(): st.warning("Required"); st.stop()` |
"""
)
