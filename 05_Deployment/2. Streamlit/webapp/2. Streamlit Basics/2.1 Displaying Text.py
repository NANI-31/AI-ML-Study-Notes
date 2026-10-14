import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Text Elements Tutorial",
    page_icon="📖",
    layout="wide",
)

st.title("📖 Streamlit Text Elements — Complete Syntax & Parameter Guide")
st.markdown(
    """
This interactive reference guides you through every Streamlit text display command.
For each command, you will find:
1. **Full Formal Function Signature**
2. **Comprehensive Parameter Reference Table** (Names, Types, Defaults, Allowed Values)
3. **Side-by-Side Example**: Python code on the left and live rendered output on the right.
"""
)

st.divider()

# ==============================================================================
# 1. Page Title & Headers (st.title, st.header, st.subheader)
# ==============================================================================
st.header("1. Typography Hierarchy: `st.title()`, `st.header()`, `st.subheader()`", divider="rainbow")

st.markdown("#### 📐 Formal Signatures")
st.code(
    """st.title(body, anchor=None, *, help=None, icon=None, width="stretch", text_alignment="left", wrap=True)
st.header(body, anchor=None, *, help=None, icon=None, divider=False, width="stretch", text_alignment="left", wrap=True)
st.subheader(body, anchor=None, *, help=None, icon=None, divider=False, width="stretch", text_alignment="left", wrap=True)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`body`** | `str` | *Required* | Any string / Markdown | The heading text to display (HTML `<h1>`, `<h2>`, or `<h3>`). |
| **`anchor`** | `str or False or None` | `None` | Custom string, `False`, or `None` | HTML anchor ID for URL deep-linking. Set to `False` to disable header anchor icon. |
| **`help`** | `str or None` | `None` | Any string | Tooltip text displayed on hover beside the heading. |
| **`icon`** | `str or None` | `None` | Emoji (e.g. `"🚀"`) or Material icon (e.g. `":material/home:"`) | Optional icon prefix placed before the heading text. |
| **`divider`** *(header & subheader)* | `bool or str` | `False` | `True`, `False`, `"blue"`, `"green"`, `"orange"`, `"red"`, `"violet"`, `"gray"`, `"rainbow"` | Renders a styled colored line directly below the heading. |
| **`width`** | `str` | `"stretch"` | `"stretch"` or `"content"` | Controls container width behavior. |
| **`text_alignment`** | `str` | `"left"` | `"left"`, `"center"`, `"right"`, `"justify"` | Text alignment within its container. |
| **`wrap`** | `bool` | `True` | `True`, `False` | Whether text wraps to the next line or clips. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_headers = '''# Title with custom icon, tooltip, and center alignment
st.title("Interactive Analytics Dashboard", icon="📊", help="Main Application Title", text_alignment="left")

# Header with built-in rainbow divider
st.header("Executive Summary", divider="rainbow", icon="📈")

# Subheader with blue divider
st.subheader("Quarterly Metrics Breakdown", divider="blue")'''
    st.code(code_headers, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.title("Interactive Analytics Dashboard", icon="📊", help="Main Application Title", text_alignment="left")
    st.header("Executive Summary", divider="rainbow", icon="📈")
    st.subheader("Quarterly Metrics Breakdown", divider="blue")

st.divider()

# ==============================================================================
# 2. Monospace & Captions (st.text, st.caption)
# ==============================================================================
st.header("2. Monospace Text & Captions: `st.text()`, `st.caption()`", divider="rainbow")

st.markdown("#### 📐 Formal Signatures")
st.code(
    """st.text(body, *, help=None, width="content", text_alignment="left", wrap=True)
st.caption(body, unsafe_allow_html=False, *, help=None, width="stretch", text_alignment="left", wrap=True)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Command | Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `st.text()` | **`body`** | `str` | *Required* | Raw literal text | Displays fixed-width monospace text without parsing Markdown or HTML. |
| `st.text()` | **`help`** | `str or None` | `None` | Any string | Tooltip text displayed on hover. |
| `st.text()` | **`width`** | `str` | `"content"` | `"content"`, `"stretch"` | Width of the container. |
| `st.text()` | **`text_alignment`** | `str` | `"left"` | `"left"`, `"center"`, `"right"`, `"justify"` | Monospace text alignment. |
| `st.text()` | **`wrap`** | `bool` | `True` | `True`, `False` | Wrap text or scroll horizontally. |
| `st.caption()` | **`body`** | `str` | *Required* | Markdown / Text | Displays small, subdued footnote or caption text (`<small>`). |
| `st.caption()` | **`unsafe_allow_html`** | `bool` | `False` | `True`, `False` | Allow raw HTML tags inside caption. |
| `st.caption()` | **`help`** | `str or None` | `None` | Any string | Hover tooltip for the caption. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_text = '''# st.text: Literal monospace (Markdown tags like **bold** are NOT rendered)
st.text("Training Log: Epoch 10/10 | Loss: 0.0234 | Val_Loss: 0.0251")
st.text("Raw string test: **This is NOT bold** | `no code styling`")

# st.caption: Subdued, small font for footnotes and metadata
st.caption("Source: Internal Telemetry DB (Updated: 2026-10-12) | Margin of error: ±1.2%")'''
    st.code(code_text, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.text("Training Log: Epoch 10/10 | Loss: 0.0234 | Val_Loss: 0.0251")
    st.text("Raw string test: **This is NOT bold** | `no code styling`")
    st.caption("Source: Internal Telemetry DB (Updated: 2026-10-12) | Margin of error: ±1.2%")

st.divider()

# ==============================================================================
# 3. Universal Display Command: st.write()
# ==============================================================================
st.header("3. The Universal Command: `st.write()`", divider="rainbow")

st.markdown("#### 📐 Formal Signature")
st.code(
    """st.write(*args, unsafe_allow_html=False)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`*args`** | `Any` (Variadic) | *Required (at least 1)* | Strings, numbers, booleans, dicts, lists, DataFrames, Matplotlib figures, exceptions | Automatically inspects argument type and delegates rendering to the optimal display method. |
| **`unsafe_allow_html`** | `bool` | `False` | `True`, `False` | When `True`, raw HTML passed inside string arguments will be rendered in the browser. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_write = '''# 1. Multiple mixed arguments in one call
st.write("Calculated metric:", 42 * 2, "| Flag active:", True)

# 2. Rich Markdown parsing by default
st.write("Supports **bold**, *italics*, and :green[colored text] natively.")

# 3. Dictionaries render as formatted, interactive JSON trees
model_params = {
    "algorithm": "XGBoost Classifier",
    "learning_rate": 0.05,
    "n_estimators": 250,
    "max_depth": 6
}
st.write(model_params)'''
    st.code(code_write, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.write("Calculated metric:", 42 * 2, "| Flag active:", True)
    st.write("Supports **bold**, *italics*, and :green[colored text] natively.")
    model_params = {
        "algorithm": "XGBoost Classifier",
        "learning_rate": 0.05,
        "n_estimators": 250,
        "max_depth": 6,
    }
    st.write(model_params)

st.divider()

# ==============================================================================
# 4. Formatted Markdown: st.markdown()
# ==============================================================================
st.header("4. Rich Markdown: `st.markdown()`", divider="rainbow")

st.markdown("#### 📐 Formal Signature")
st.code(
    """st.markdown(body, unsafe_allow_html=False, *, help=None, width="auto", text_alignment="left", anchors=True, wrap=True)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`body`** | `str` | *Required* | GitHub Flavored Markdown (GFM) | String formatted in Markdown syntax (headings, bold, lists, math, tables). |
| **`unsafe_allow_html`** | `bool` | `False` | `True`, `False` | By default (`False`), HTML tags are escaped. Setting to `True` renders raw HTML elements (e.g. `<span>`, `<div>`, `<style>`). |
| **`help`** | `str or None` | `None` | Any string | Tooltip text displayed on hover. |
| **`width`** | `str` | `"auto"` | `"auto"`, `"stretch"`, `"content"` | Width behavior of the Markdown container. |
| **`text_alignment`** | `str` | `"left"` | `"left"`, `"center"`, `"right"`, `"justify"` | Text alignment of the rendered block. |
| **`anchors`** | `bool` | `True` | `True`, `False` | Whether heading tags inside the Markdown automatically receive URL anchor links. |
| **`wrap`** | `bool` | `True` | `True`, `False` | Whether paragraphs wrap automatically. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_markdown = '''# Formatting and blockquotes
st.markdown("**Bold**, *Italics*, and ~~Strikethrough~~")
st.markdown("> 'Data is the new oil, but only if refined.'")

# Bulleted and nested lists
st.markdown("""
- **Data Prep**: Cleaning & Feature Engineering
- **Modeling**:
  1. Baseline Logistic Regression
  2. Champion Random Forest
""")

# Streamlit-specific color syntax: :color[text]
# Colors: red, blue, green, orange, violet, gray, rainbow
st.markdown(":red[Red text] | :blue[Blue text] | :green[Green text] | :orange[Orange text] | :violet[Violet text]")
st.markdown(":rainbow[🌈 Modern Streamlit Rainbow Gradient Text!]")'''
    st.code(code_markdown, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.markdown("**Bold**, *Italics*, and ~~Strikethrough~~")
    st.markdown("> 'Data is the new oil, but only if refined.'")
    st.markdown(
        """
- **Data Prep**: Cleaning & Feature Engineering
- **Modeling**:
  1. Baseline Logistic Regression
  2. Champion Random Forest
"""
    )
    st.markdown(
        ":red[Red text] | :blue[Blue text] | :green[Green text] | :orange[Orange text] | :violet[Violet text]"
    )
    st.markdown(":rainbow[🌈 Modern Streamlit Rainbow Gradient Text!]")

st.divider()

# ==============================================================================
# 5. Code Display: st.code()
# ==============================================================================
st.header("5. Code Blocks with Highlighting: `st.code()`", divider="rainbow")

st.markdown("#### 📐 Formal Signature")
st.code(
    """st.code(body, language="python", *, line_numbers=False, wrap_lines=False, height="content", width="stretch")""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`body`** | `str` | *Required* | Code string | The code text to display inside a formatted code container with a copy-to-clipboard button. |
| **`language`** | `str or None` | `"python"` | `"python"`, `"sql"`, `"bash"`, `"javascript"`, `"json"`, `"yaml"`, `"html"`, `"css"`, `"c"`, `"cpp"`, `"dockerfile"`, `None` | Programming or markup language used by Prism.js for syntax highlighting. |
| **`line_numbers`** | `bool` | `False` | `True`, `False` | If `True`, renders consecutive line numbers along the left margin. |
| **`wrap_lines`** | `bool` | `False` | `True`, `False` | If `True`, wraps long lines; if `False`, enables horizontal scrolling. |
| **`height`** | `int or "content" or None` | `"content"` | `"content"` or integer pixel count (e.g. `200`) | Height of the code block. If fixed, provides a vertical scrollbar. |
| **`width`** | `str` | `"stretch"` | `"stretch"`, `"content"` | Width of the container block. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_block = '''# Python with line numbers enabled
py_code = """import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))"""

st.code(py_code, language="python", line_numbers=True)

# SQL syntax highlighting
sql_code = """SELECT department, AVG(salary) AS mean_sal
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;"""

st.code(sql_code, language="sql", line_numbers=False)'''
    st.code(code_block, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    py_code = """import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))"""
    st.code(py_code, language="python", line_numbers=True)

    sql_code = """SELECT department, AVG(salary) AS mean_sal
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;"""
    st.code(sql_code, language="sql", line_numbers=False)

st.divider()

# ==============================================================================
# 6. Mathematical Formulas: st.latex()
# ==============================================================================
st.header("6. Mathematical Formulas: `st.latex()`", divider="rainbow")

st.markdown("#### 📐 Formal Signature")
st.code(
    """st.latex(body, *, help=None, width="stretch")""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`body`** | `str or sympy.Expr` | *Required* | Raw LaTeX string (e.g. `r"..."`) or SymPy expression | Mathematical expression rendered via KaTeX. Always prefix strings with `r` (raw string) to avoid backslash escaping issues. |
| **`help`** | `str or None` | `None` | Any string | Tooltip text displayed on hover. |
| **`width`** | `str` | `"stretch"` | `"stretch"`, `"content"` | Width of the equation container. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_latex = r'''# Mean Squared Error (MSE)
st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2")

# Normal Distribution Probability Density Function
st.latex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")'''
    st.code(code_latex, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2")
    st.latex(
        r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}"
    )

st.divider()

# ==============================================================================
# 7. Dividers & Section Breaks: st.divider()
# ==============================================================================
st.header("7. Visual Dividers: `st.divider()`", divider="rainbow")

st.markdown("#### 📐 Formal Signature")
st.code(
    """st.divider(*, width="stretch")""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`width`** | `str` | `"stretch"` | `"stretch"`, `"content"` | Width of the horizontal rule (`<hr>`). Used to cleanly separate logical sections. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_div = '''st.write("Section A: Data Preprocessing complete.")
st.divider()  # Draws a horizontal separating line
st.write("Section B: Model Evaluation results:")'''
    st.code(code_div, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.write("Section A: Data Preprocessing complete.")
    st.divider()
    st.write("Section B: Model Evaluation results:")

st.divider()

# ==============================================================================
# 8. Callout & Alert Banners: st.info, st.success, st.warning, st.error
# ==============================================================================
st.header("8. Callout & Alert Boxes: `st.info()`, `st.success()`, `st.warning()`, `st.error()`", divider="rainbow")

st.markdown("#### 📐 Formal Signatures")
st.code(
    """st.info(body, *, icon=None, width="stretch", title=None)
st.success(body, *, icon=None, width="stretch", title=None)
st.warning(body, *, icon=None, width="stretch", title=None)
st.error(body, *, icon=None, width="stretch", title=None)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification")
st.markdown(
    """
| Parameter | Type | Default Value | Allowed Values / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`body`** | `str` | *Required* | Text / Markdown | The message body text to display inside the banner. |
| **`icon`** | `str or None` | `None` | Emoji (e.g. `"ℹ️"`, `"✅"`) or Material icon (e.g. `":material/thumb_up:"`) | Replaces the default status icon on the left of the callout banner. |
| **`title`** | `str or None` | `None` | Any string | Optional bold title displayed prominently above the callout message body. |
| **`width`** | `str` | `"stretch"` | `"stretch"`, `"content"` | Width behavior of the callout container. |
"""
)

st.markdown("#### 🧪 Live Code vs Output Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💻 Python Code**")
    code_alerts = '''# Info callout (Blue) with custom title and icon
st.info("Cached features retrieved from memory in 12ms.", title="Cache Hit", icon="ℹ️")

# Success callout (Green)
st.success("Champion model exported to models/pipeline.joblib", title="Model Saved", icon="✅")

# Warning callout (Yellow/Orange)
st.warning("Collinearity detected between feature_a and feature_b (VIF > 10).", title="High VIF", icon="⚠️")

# Error callout (Red)
st.error("Connection timeout while attempting to connect to PostgreSQL DB.", title="DB Timeout", icon="🚨")'''
    st.code(code_alerts, language="python")

with col2:
    st.markdown("**🖥️ Rendered Output**")
    st.info("Cached features retrieved from memory in 12ms.", title="Cache Hit", icon="ℹ️")
    st.success("Champion model exported to models/pipeline.joblib", title="Model Saved", icon="✅")
    st.warning("Collinearity detected between feature_a and feature_b (VIF > 10).", title="High VIF", icon="⚠️")
    st.error("Connection timeout while attempting to connect to PostgreSQL DB.", title="DB Timeout", icon="🚨")

st.divider()

# ==============================================================================
# Comprehensive Parameter Comparison Matrix
# ==============================================================================
st.header("📊 Master Parameter Matrix Across All Text Commands", divider="rainbow")

matrix_markdown = """
| Command | `body` | `anchor` | `help` | `icon` | `divider` | `unsafe_allow_html` | `line_numbers` | `language` | `title` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`st.title()`** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **`st.header()`** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **`st.subheader()`** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **`st.text()`** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **`st.caption()`** | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **`st.write()`** | `*args` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **`st.markdown()`** | ✅ | `anchors` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **`st.code()`** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **`st.latex()`** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **`st.divider()`** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **`st.info()`** / Alerts | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
"""
st.markdown(matrix_markdown)