import streamlit as st
import json
import io
import wave
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import NamedTuple

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Displaying Different Data Types",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 Displaying Different Data Types — Complete Streamlit Guide")
st.markdown(
    """
Streamlit provides seamless, dynamic rendering for virtually every Python data type. 
This comprehensive guide explores how Streamlit handles **primitives, collections, custom objects, 
and JSON payloads**, detailing both the general-purpose `st.write()` dispatcher and dedicated 
high-performance display functions.
"""
)

st.divider()

# ==============================================================================
# 1. Displaying Strings (str)
# ==============================================================================
st.header("1. Displaying Strings (`str`)", divider="rainbow")
st.markdown(
    """
Python strings can be displayed using three distinct approaches depending on the formatting requirement:
1. **`st.markdown()` / `st.write()`**: Full Markdown, HTML (optional), emojis, LaTeX math, and semantic colors.
2. **`st.text()`**: Fixed-width monospace text that preserves verbatim whitespace, tabs, and newlines.
3. **`st.caption()`**: Smaller, muted typography for metadata, notes, and citations.
"""
)

st.markdown("#### 📋 Parameter Specification: String Display Functions")
st.markdown(
    """
| Function | Key Parameters | Default | Description |
| :--- | :--- | :--- | :--- |
| **`st.markdown(body, unsafe_allow_html=False)`** | `body: str`, `unsafe_allow_html: bool` | `False` | Renders formatted GitHub-flavored Markdown. Supports colored text like `:blue[text]`, `:green[text]`, and LaTeX `$x^2$`. |
| **`st.text(body)`** | `body: str` | *Required* | Renders preformatted monospace text (`<pre>`). Preserves raw indentation and does not parse markdown symbols. |
| **`st.caption(body, unsafe_allow_html=False)`** | `body: str`, `unsafe_allow_html: bool` | `False` | Muted 12px secondary text designed for labels, footnotes, and photo captions. |
"""
)

st.markdown("#### 🧪 Live Code vs. Rendered Output Comparison")
col_str1, col_str2 = st.columns(2)

with col_str1:
    st.markdown("**💻 Python Source Code**")
    code_str = '''# 1. Markdown with rich formatting & colored spans
st.markdown("**Bold text**, *italics*, and `:rainbow[rainbow gradient text]`!")
st.markdown("Supports inline LaTeX: $E = mc^2$ and emojis :rocket: :chart_with_upwards_trend:")

# 2. Raw monospace text (preserves indentation & newlines)
raw_ascii = """
+---------------------+
| System Status: OK   |
| Latency: 12ms       |
+---------------------+
"""
st.text(raw_ascii)

# 3. Caption for secondary footnotes
st.caption("📌 *Note: String measurements taken over a 100-batch rolling window.*")'''
    st.code(code_str, language="python")

with col_str2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown("**Bold text**, *italics*, and `:rainbow[rainbow gradient text]`!")
    st.markdown("Supports inline LaTeX: $E = mc^2$ and emojis :rocket: :chart_with_upwards_trend:")
    raw_ascii_sample = """+---------------------+
| System Status: OK   |
| Latency: 12ms       |
+---------------------+"""
    st.text(raw_ascii_sample)
    st.caption("📌 *Note: String measurements taken over a 100-batch rolling window.*")

with st.expander("🛠️ Interactive String Playground (Test Your Own String Formatting)"):
    user_string = st.text_input("Enter text to format:", value=":green[Streamlit] makes Python **web development** intuitive!")
    render_mode = st.radio("Display Mode:", ["st.markdown", "st.text", "st.caption"], horizontal=True)
    if render_mode == "st.markdown":
        st.markdown(user_string)
    elif render_mode == "st.text":
        st.text(user_string)
    else:
        st.caption(user_string)

st.divider()

# ==============================================================================
# 2. Displaying Integers and Floats (int, float)
# ==============================================================================
st.header("2. Displaying Integers and Floats (`int`, `float`)", divider="rainbow")
st.markdown(
    """
Numeric values can be rendered directly with `st.write()`, formatted using standard Python f-strings, 
or transformed into high-impact **Executive KPI Metric Cards** via `st.metric()`.
"""
)

st.markdown("#### 📐 Formal Signature: `st.metric()`")
st.code(
    """st.metric(
    label: str,
    value: str | int | float | None,
    delta: str | int | float | None = None,
    delta_color: "normal" | "inverse" | "off" = "normal",
    help: str | None = None,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    border: bool = False
)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.metric()`")
st.markdown(
    """
| Parameter | Type | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`label`** | `str` | *Required* | Any string | Title of the KPI metric displayed above the number. |
| **`value`** | `int`, `float`, `str` | *Required* | Numeric or formatted string | Primary numeric reading (e.g. `1420`, `"$12.4M"`). |
| **`delta`** | `int`, `float`, `str` | `None` | Numeric or formatted string | Indicator showing change or variance (e.g. `+12.4%`, `-350`). |
| **`delta_color`** | `str` | `"normal"` | `"normal"`, `"inverse"`, `"off"` | `"normal"`: green for positive / red for negative. `"inverse"`: red for positive / green for negative (useful for error rates). `"off"`: neutral gray. |
| **`border`** | `bool` | `False` | `True`, `False` | Encloses the metric inside a styled card border with padding. |
| **`help`** | `str` | `None` | Tooltip string | Hover tooltip explaining calculation methodology. |
"""
)

col_num1, col_num2 = st.columns(2)

with col_num1:
    st.markdown("**💻 Python Source Code**")
    code_num = '''# Direct numeric printing
revenue = 1425890.75
active_users = 48210
churn_rate = 0.024

# 1. Plain st.write()
st.write("Total Revenue (Raw):", revenue)

# 2. Formatted with Python f-strings
st.write(f"Formatted: **${revenue:,.2f}** | Users: **{active_users:,}** | Churn: **{churn_rate:.1%}**")

# 3. High-Impact KPI Cards (st.metric)
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Revenue (ARR)", f"${revenue/1e6:.2f}M", delta="+14.2%", border=True)
m_col2.metric("Active Sessions", f"{active_users:,}", delta="-340", border=True)
m_col3.metric("API Latency", "18.4 ms", delta="-4.2 ms", delta_color="inverse", border=True)'''
    st.code(code_num, language="python")

with col_num2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    revenue_val = 1425890.75
    users_val = 48210
    churn_val = 0.024
    st.write("Total Revenue (Raw):", revenue_val)
    st.write(f"Formatted: **${revenue_val:,.2f}** | Users: **{users_val:,}** | Churn: **{churn_val:.1%}**")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Revenue (ARR)", f"${revenue_val/1e6:.2f}M", delta="+14.2%", border=True)
    m_col2.metric("Active Sessions", f"{users_val:,}", delta="-340", border=True)
    m_col3.metric("API Latency", "18.4 ms", delta="-4.2 ms", delta_color="inverse", border=True, help="Inverse color: lower latency is green!")

st.divider()

# ==============================================================================
# 3. Displaying Lists (list)
# ==============================================================================
st.header("3. Displaying Lists (`list`)", divider="rainbow")
st.markdown(
    """
Python lists can be rendered in multiple representations based on UX needs:
- **Interactive JSON Array**: `st.write(my_list)` displays a collapsible array explorer.
- **Markdown Bullet / Ordered List**: Formatted programmatically via string joining.
- **Tabular Column**: Passing a list to `st.dataframe()` or `st.table()` displays clean tabular rows.
"""
)

col_list1, col_list2 = st.columns(2)

with col_list1:
    st.markdown("**💻 Python Source Code**")
    code_list = '''models = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B", "DeepSeek R1"]

# Option A: Direct st.write (Interactive JSON-style representation)
st.write("Raw List via st.write():", models)

# Option B: Programmatic Markdown Bullets
st.markdown("**Markdown Bullet List:**")
for i, model in enumerate(models, start=1):
    st.markdown(f"{i}. 🚀 **{model}**")

# Option C: Tabular View via DataFrame
df_models = pd.DataFrame(models, columns=["Frontier Foundation Model"])
st.dataframe(df_models, width="stretch", hide_index=True)'''
    st.code(code_list, language="python")

with col_list2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    models_demo = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B", "DeepSeek R1"]
    st.write("Raw List via st.write():", models_demo)
    
    st.markdown("**Markdown Bullet List:**")
    for idx, m_name in enumerate(models_demo[:3], start=1):
        st.markdown(f"{idx}. 🚀 **{m_name}**")
    st.caption("*(Showing top 3 models)*")

    df_models_demo = pd.DataFrame(models_demo, columns=["Frontier Foundation Model"])
    st.dataframe(df_models_demo, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 4. Displaying Dictionaries (dict)
# ==============================================================================
st.header("4. Displaying Dictionaries (`dict`)", divider="rainbow")
st.markdown(
    """
Dictionaries represent key-value mappings. Streamlit automatically parses dictionaries passed to 
`st.write()` and renders them into an interactive collapsible viewer with search and copy capabilities, 
identical to `st.json()`.
"""
)

sample_dict = {
    "server_name": "us-east-cluster-01",
    "environment": "production",
    "active_workers": 16,
    "cpu_utilization_pct": 42.8,
    "memory_gb": {"allocated": 64, "used": 28.5, "free": 35.5},
    "tags": ["web-tier", "ml-serving", "high-availability"],
    "health_status": "HEALTHY"
}

col_dict1, col_dict2 = st.columns(2)

with col_dict1:
    st.markdown("**💻 Python Source Code**")
    code_dict = '''system_metrics = {
    "server_name": "us-east-cluster-01",
    "environment": "production",
    "active_workers": 16,
    "cpu_utilization_pct": 42.8,
    "memory_gb": {"allocated": 64, "used": 28.5, "free": 35.5},
    "tags": ["web-tier", "ml-serving", "high-availability"],
    "health_status": "HEALTHY"
}

# 1. Native dict rendering (Collapsible Tree with copy button)
st.write(system_metrics)

# 2. Key-Value Summary Table
flat_summary = {k: str(v) for k, v in system_metrics.items() if not isinstance(v, dict)}
df_kv = pd.DataFrame(list(flat_summary.items()), columns=["Metric Key", "Current Value"])
st.dataframe(df_kv, width="stretch", hide_index=True)'''
    st.code(code_dict, language="python")

with col_dict2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown("**1. Collapsible Dictionary Viewer:**")
    st.write(sample_dict)
    
    st.markdown("**2. Key-Value Structured Table:**")
    flat_summary = {k: str(v) for k, v in sample_dict.items() if not isinstance(v, dict)}
    df_kv = pd.DataFrame(list(flat_summary.items()), columns=["Metric Key", "Current Value"])
    st.dataframe(df_kv, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 5. Displaying Tuples (tuple & NamedTuple)
# ==============================================================================
st.header("5. Displaying Tuples (`tuple`)", divider="rainbow")
st.markdown(
    """
Tuples are immutable sequences. In Streamlit:
- Standard tuples `(x, y)` render as array collections in `st.write()`.
- Named tuples (`collections.namedtuple` or `typing.NamedTuple`) preserve attribute names and provide self-documenting data structures.
"""
)

col_tup1, col_tup2 = st.columns(2)

with col_tup1:
    st.markdown("**💻 Python Source Code**")
    code_tup = '''from typing import NamedTuple

# 1. Standard Primitive Tuple (e.g. Geo coordinates)
geo_coordinates = (37.7749, -122.4194, "San Francisco, CA")
st.write("Raw Tuple:", geo_coordinates)

# 2. Structured NamedTuple
class ModelCheckpoint(NamedTuple):
    epoch: int
    validation_loss: float
    accuracy: float
    checkpoint_path: str

best_ckpt = ModelCheckpoint(
    epoch=25,
    validation_loss=0.0342,
    accuracy=0.9845,
    checkpoint_path="s3://models/vision_transformer_v2.pt"
)

# Render NamedTuple fields
st.write(best_ckpt)
st.markdown(f"""
- **Epoch**: `{best_ckpt.epoch}`
- **Validation Loss**: `{best_ckpt.validation_loss:.4f}`
- **Accuracy**: `{best_ckpt.accuracy:.2%}`
- **Artifact**: `{best_ckpt.checkpoint_path}`
""")'''
    st.code(code_tup, language="python")

with col_tup2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    geo_coords = (37.7749, -122.4194, "San Francisco, CA")
    st.write("Raw Tuple:", geo_coords)

    class ModelCheckpoint(NamedTuple):
        epoch: int
        validation_loss: float
        accuracy: float
        checkpoint_path: str

    best_ckpt = ModelCheckpoint(
        epoch=25,
        validation_loss=0.0342,
        accuracy=0.9845,
        checkpoint_path="s3://models/vision_transformer_v2.pt"
    )
    st.write("NamedTuple Instance:", best_ckpt)
    st.markdown(f"""
- **Epoch**: `{best_ckpt.epoch}`
- **Validation Loss**: `{best_ckpt.validation_loss:.4f}`
- **Accuracy**: `{best_ckpt.accuracy:.2%}`
- **Artifact**: `{best_ckpt.checkpoint_path}`
""")

st.divider()

# ==============================================================================
# 6. Displaying Python Objects (Custom Classes, Dataclasses, Functions)
# ==============================================================================
st.header("6. Displaying Python Objects (Dataclasses, Classes & `st.help`)", divider="rainbow")
st.markdown(
    """
Python objects can be inspected and displayed using three powerful techniques:
1. **`dataclasses.asdict(obj)`**: Converts a structured dataclass into a dictionary for JSON/Table rendering.
2. **`vars(obj)` / `obj.__dict__`**: Inspects attributes of custom class instances.
3. **`st.help(obj)`**: Streamlit's built-in interactive documentation and signature inspector. Works on classes, methods, modules, and built-in primitives.
"""
)

@dataclass
class LLMHyperparameters:
    model_id: str
    temperature: float
    max_tokens: int
    top_p: float
    stop_sequences: list

hyperparams = LLMHyperparameters(
    model_id="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
    top_p=0.95,
    stop_sequences=["<|endoftext|>", "###"]
)

col_obj1, col_obj2 = st.columns(2)

with col_obj1:
    st.markdown("**💻 Python Source Code**")
    code_obj = '''from dataclasses import dataclass, asdict

@dataclass
class LLMHyperparameters:
    model_id: str
    temperature: float
    max_tokens: int
    top_p: float
    stop_sequences: list

params = LLMHyperparameters(
    model_id="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
    top_p=0.95,
    stop_sequences=["<|endoftext|>", "###"]
)

# 1. Render dataclass converted to dict
st.write("Dataclass as Dict (JSON format):")
st.json(asdict(params))

# 2. Interactive Object Inspection via st.help()
st.help(LLMHyperparameters)'''
    st.code(code_obj, language="python")

with col_obj2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.write("Dataclass as Dict (`asdict`):")
    st.json(asdict(hyperparams))
    
    st.write("Interactive Class Docstring via `st.help()`:")
    st.help(LLMHyperparameters)

st.divider()

# ==============================================================================
# 7. Displaying JSON Data (st.json)
# ==============================================================================
st.header("7. Displaying JSON Data (`st.json`)", divider="rainbow")
st.markdown(
    """
`st.json()` provides a dedicated, publication-ready interactive JSON tree explorer. 
It supports collapsible objects, nested arrays, automatic syntax highlighting, an integrated search bar, 
and a **one-click copy button**.
"""
)

st.markdown("#### 📐 Formal Signature: `st.json()`")
st.code(
    """st.json(
    body: str | dict | list | object,
    expanded: bool | int = True,
    *,
    width: "stretch" | "content" = "stretch"
)""",
    language="python",
)

st.markdown("#### 📋 Parameter Specification: `st.json()`")
st.markdown(
    """
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`body`** | `dict`, `list`, `str`, `object` | *Required* | The object or JSON-encoded string to serialize and display. |
| **`expanded`** | `bool` or `int` | `True` | `True`: expand all nodes. `False`: collapse all nodes. Integer (e.g. `1` or `2`): expand nodes up to the specified depth level. |
| **`width`** | `str` | `"stretch"` | Controls container boundary (`"stretch"` fills parent width). |
"""
)

sample_llm_payload = {
    "id": "chatcmpl-9xyzABC123",
    "object": "chat.completion",
    "created": 1718102400,
    "model": "gpt-4o-2024-05-13",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Streamlit is an open-source Python framework that allows developers to build interactive web apps with minimal code."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 34,
        "completion_tokens": 28,
        "total_tokens": 62
    }
}

col_json_ctrl1, col_json_ctrl2 = st.columns(2)
with col_json_ctrl1:
    expand_depth = st.slider("JSON Expansion Depth Level:", min_value=0, max_value=3, value=2, key="json_depth")
with col_json_ctrl2:
    as_raw_string = st.checkbox("Pass as Raw JSON String (json.dumps)", value=False, key="json_raw_str")

col_json1, col_json2 = st.columns(2)

with col_json1:
    st.markdown("**💻 Python Source Code**")
    code_json = f'''import json

payload = {json.dumps(sample_llm_payload, indent=2)}

# Render with controlled depth
st.json(payload, expanded={expand_depth})'''
    st.code(code_json, language="python")

with col_json2:
    st.markdown("**🖥️ Rendered Interactive JSON Tree**")
    data_to_render = json.dumps(sample_llm_payload) if as_raw_string else sample_llm_payload
    st.json(data_to_render, expanded=expand_depth)

st.divider()

# ==============================================================================
# 8. Formatting Output with st.write() (The Swiss Army Knife)
# ==============================================================================
st.header("8. Master Formatting with `st.write()` (The Swiss Army Knife)", divider="rainbow")
st.markdown(
    """
`st.write()` is the universal display primitive in Streamlit. It uses dynamic runtime type inspection 
to dispatch any argument to its optimal specialized display component.
"""
)

st.markdown("#### 📐 Formal Signature: `st.write()`")
st.code(
    """st.write(*args: Any, unsafe_allow_html: bool = False, **kwargs: Any) -> None""",
    language="python"
)

st.markdown("#### 🧭 Internal Type Routing Algorithm")
st.markdown(
    """
When you invoke `st.write(arg)`, Streamlit evaluates the Python `type(arg)` using the following precedence rules:
"""
)

routing_matrix = {
    "Passed Python Type": [
        "str",
        "int, float, complex",
        "list, tuple",
        "dict",
        "pandas.DataFrame / Series",
        "matplotlib.figure.Figure",
        "altair.Chart / plotly.Figure",
        "Exception",
        "Custom Class / Object",
        "None"
    ],
    "Internal Streamlit Handler": [
        "st.markdown()",
        "st.markdown() (formatted number)",
        "st.json() (interactive array view)",
        "st.json() (collapsible tree view)",
        "st.dataframe()",
        "st.pyplot()",
        "st.altair_chart() / st.plotly_chart()",
        "st.exception()",
        "st.help() or str(obj)",
        "No output rendered"
    ],
    "When to Use Explicit Function Instead": [
        "Use `st.text()` for monospace logs; `st.caption()` for notes",
        "Use `st.metric()` for KPI cards with delta arrows",
        "Use `st.dataframe(df)` if you need custom column configurations",
        "Use `st.json(expanded=N)` for fine-grained depth control",
        "Use `st.dataframe()` with column config / styling",
        "Use explicit `st.pyplot(fig)` to customize margins",
        "Use `st.plotly_chart(fig, use_container_width=True)`",
        "Use `st.error()` if you want a clean user-facing error message",
        "Use `st.help(obj)` to view interactive docstrings",
        "N/A"
    ]
}

st.dataframe(routing_matrix, width="stretch", hide_index=True)

st.markdown("#### 🧪 Heterogeneous Multi-Argument Dispatch")
st.markdown(
    """
A unique power of `st.write()` is that it accepts **multiple arguments of different types simultaneously**, 
rendering each with its dedicated formatter sequentially:
"""
)

col_multi1, col_multi2 = st.columns(2)

with col_multi1:
    st.markdown("**💻 Python Source Code**")
    code_multi = '''# Passing multiple heterogeneous types in a single st.write() call:
st.write(
    "### Pipeline Audit Results",
    "Success Rate:",
    99.4,
    ["Worker-1 (OK)", "Worker-2 (OK)", "Worker-3 (Standby)"],
    {"active_threads": 48, "memory_cache_hit_rate": "94.2%"},
    pd.DataFrame({"Model": ["ResNet", "BERT"], "F1-Score": [0.94, 0.96]})
)'''
    st.code(code_multi, language="python")

with col_multi2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.write(
        "### Pipeline Audit Results",
        "Success Rate:",
        99.4,
        ["Worker-1 (OK)", "Worker-2 (OK)", "Worker-3 (Standby)"],
        {"active_threads": 48, "memory_cache_hit_rate": "94.2%"},
        pd.DataFrame({"Model": ["ResNet", "BERT"], "F1-Score": [0.94, 0.96]})
    )

st.markdown("#### 🪄 Streamlit Magic Commands")
st.markdown(
    """
Streamlit features **Magic Commands**: any variable, expression, or string literal written on its own line 
without being assigned or wrapped in a function call is automatically routed through `st.write()`:
"""
)

col_magic1, col_magic2 = st.columns(2)

with col_magic1:
    st.markdown("**💻 Python Code (Using Magic)**")
    code_magic = '''# Writing expressions directly on their own lines:
"""
# Autonomous Magic Header
Magic commands parse standalone strings as Markdown.
"""

score = 42 * 2
score  # Variable on its own line!

{"algorithm": "GradientBoosting", "n_estimators": 100}  # Standalone dict!'''
    st.code(code_magic, language="python")

with col_magic2:
    st.markdown("**🖥️ Equivalent Explicit Function Call**")
    st.markdown(
        """
        - Standalone string `\"\"\"# ...\"\"\"` $\\rightarrow$ `st.write(\"\"\"# ...\"\"\")`
        - Standalone variable `score` $\\rightarrow$ `st.write(score)`
        - Standalone dict `{...}` $\\rightarrow$ `st.write({...})`
        
        > 💡 **Best Practice**: Use Magic commands during rapid exploratory prototyping. In production production apps, prefer **explicit functions** (`st.markdown`, `st.metric`, `st.dataframe`) for codebase clarity and maintainability.
        """
    )

st.divider()

# ==============================================================================
# Advanced Display Deep Dives (Topics 2.3, 2.4, 2.5 via Expanders)
# ==============================================================================
st.header("🚀 Advanced Display Extensions (Interactive Deep Dives)", divider="rainbow")
st.markdown(
    """
Expand the sections below to explore advanced data display capabilities:
- **Topic 2.3**: Interactive DataFrames, static tables, two-way data editors, and column configurations.
- **Topic 2.4**: Built-in native charts vs. Plotly, Altair, and Seaborn/Matplotlib.
- **Topic 2.5**: Multimedia rendering (images, synthetic audio, video players, and live camera input).
"""
)

# ==============================================================================
# 9. Extended Topic 2.3 — Displaying Data in Tables and DataFrames
# ==============================================================================
with st.expander("📊 Topic 2.3 — Displaying Data in Tables and DataFrames (st.dataframe, st.table, st.data_editor, Column Configs)", expanded=False):
    st.markdown("### 📊 Comprehensive Guide: Tables, DataFrames & Interactive Data Editors")
    st.markdown(
        """
Streamlit offers three distinct tabular display primitives:
1. **`st.dataframe()`**: High-performance interactive grid supporting sorting, column resizing, full-text search, CSV downloads, and custom column formatters (`st.column_config`).
2. **`st.table()`**: Static HTML table. Renders entire content without client-side pagination or scrolling. Perfect for short summaries and non-interactive reports.
3. **`st.data_editor()`**: Bidirectional interactive spreadsheet that allows users to edit cells, add rows, and delete records directly inside the browser, passing edits back to Python.
"""
    )

    st.markdown("#### 📐 Formal Signatures")
    st.code(
        """st.dataframe(data=None, width=None, height=None, *, use_container_width=True, hide_index=None, column_order=None, column_config=None, row_height=None)
st.table(data=None)
st.data_editor(data=None, width=None, height=None, *, use_container_width=True, hide_index=None, column_order=None, column_config=None, num_rows="fixed", disabled=False, key=None)""",
        language="python"
    )

    st.markdown("#### 🎨 Advanced Column Configurations (`st.column_config`)")
    st.markdown(
        """
`st.column_config` transforms raw tabular data into rich visual representations without modifying your underlying pandas DataFrame:
- **`ProgressColumn`**: Renders numeric ratios as visual progress bars.
- **`NumberColumn`**: Applies currency/percentage formatting, bounds, and step increments.
- **`ImageColumn`**: Renders URLs as image thumbnails.
- **`LinkColumn`**: Renders clickable hyperlinks.
- **`CheckboxColumn`**: Displays booleans as interactive or read-only checkmarks.
- **`SelectboxColumn`**: Constrains edits to predefined dropdown options.
"""
    )

    models_data = {
        "Icon": [
            "https://raw.githubusercontent.com/streamlit/brand/main/icons/streamlit-mark-color.png",
            "https://raw.githubusercontent.com/streamlit/brand/main/icons/streamlit-mark-color.png",
            "https://raw.githubusercontent.com/streamlit/brand/main/icons/streamlit-mark-color.png",
            "https://raw.githubusercontent.com/streamlit/brand/main/icons/streamlit-mark-color.png"
        ],
        "Model Name": ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B"],
        "Provider": ["OpenAI", "Anthropic", "Google DeepMind", "Meta"],
        "Benchmark Accuracy": [0.92, 0.94, 0.91, 0.88],
        "Latency (ms)": [180.5, 210.0, 165.2, 95.8],
        "Cost per 1M Tokens ($)": [5.00, 3.00, 3.50, 0.70],
        "Production Ready": [True, True, True, True],
        "Documentation": [
            "https://platform.openai.com",
            "https://docs.anthropic.com",
            "https://ai.google.dev",
            "https://llama.meta.com"
        ]
    }
    df_models_registry = pd.DataFrame(models_data)

    st.markdown("##### 🧪 1. Live Interactive DataFrame with Rich Column Configs")
    code_df_configs = '''# Configure rich visual columns on a DataFrame:
st.dataframe(
    df_models_registry,
    column_config={
        "Icon": st.column_config.ImageColumn("Logo", help="Provider Icon", width="small"),
        "Benchmark Accuracy": st.column_config.ProgressColumn(
            "Benchmark Accuracy",
            help="MMLU Benchmark score (0.0 to 1.0)",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "Latency (ms)": st.column_config.NumberColumn(
            "Inference Latency",
            format="%.1f ms",
            help="Average token turnaround latency"
        ),
        "Cost per 1M Tokens ($)": st.column_config.NumberColumn(
            "Cost / 1M Tokens",
            format="$%.2f",
        ),
        "Production Ready": st.column_config.CheckboxColumn(
            "Status",
            help="Is model active in production?",
        ),
        "Documentation": st.column_config.LinkColumn(
            "Docs Link",
            display_text="View Documentation ↗"
        )
    },
    width="stretch",
    hide_index=True
)'''
    col_df1, col_df2 = st.columns([1, 1])
    with col_df1:
        st.markdown("**💻 Python Source Code**")
        st.code(code_df_configs, language="python")
    with col_df2:
        st.markdown("**🖥️ Rendered Interactive DataFrame**")
        st.dataframe(
            df_models_registry,
            column_config={
                "Icon": st.column_config.ImageColumn("Logo", help="Provider Icon", width="small"),
                "Benchmark Accuracy": st.column_config.ProgressColumn(
                    "Benchmark Accuracy",
                    help="MMLU Benchmark score (0.0 to 1.0)",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Latency (ms)": st.column_config.NumberColumn(
                    "Inference Latency",
                    format="%.1f ms",
                    help="Average token turnaround latency"
                ),
                "Cost per 1M Tokens ($)": st.column_config.NumberColumn(
                    "Cost / 1M Tokens",
                    format="$%.2f",
                ),
                "Production Ready": st.column_config.CheckboxColumn(
                    "Status",
                    help="Is model active in production?",
                ),
                "Documentation": st.column_config.LinkColumn(
                    "Docs Link",
                    display_text="View Documentation ↗"
                )
            },
            width="stretch",
            hide_index=True
        )

    st.markdown("##### 🧪 2. Direct Comparison: `st.dataframe()` vs `st.table()`")
    code_df_vs_tbl = '''# 1. Interactive Grid (sortable, searchable, downloadable CSV)
st.dataframe(summary_slice, width="stretch", hide_index=True)

# 2. Static HTML Table (zero client-side scroll, full table visible)
st.table(summary_slice)'''
    st.markdown("**💻 Python Source Code**")
    st.code(code_df_vs_tbl, language="python")

    st.markdown("**🖥️ Rendered Comparison**")
    col_t1, col_t2 = st.columns(2)
    summary_slice = df_models_registry[["Model Name", "Provider", "Benchmark Accuracy", "Latency (ms)"]]
    with col_t1:
        st.markdown("**`st.dataframe()` (Interactive Grid with Sorting & Search):**")
        st.dataframe(summary_slice, width="stretch", hide_index=True)
    with col_t2:
        st.markdown("**`st.table()` (Static HTML Table, No Scrollbars):**")
        st.table(summary_slice)

    st.markdown("##### 🧪 3. Bidirectional Spreadsheet: `st.data_editor()`")
    initial_editor_data = pd.DataFrame({
        "Feature": ["Vector Search", "Hybrid Re-ranking", "Query Caching", "Streaming Tokens"],
        "Priority": ["High", "Critical", "Medium", "High"],
        "Weight": [0.85, 0.95, 0.60, 0.90],
        "Enabled": [True, True, False, True]
    })

    code_editor = '''# Bidirectional spreadsheet with dropdowns and numbers:
edited_df = st.data_editor(
    initial_editor_data,
    num_rows="dynamic",  # Allows adding/deleting rows live
    column_config={
        "Priority": st.column_config.SelectboxColumn(
            "Priority Level",
            options=["Low", "Medium", "High", "Critical"],
            required=True
        ),
        "Weight": st.column_config.NumberColumn(
            "Feature Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f"
        ),
        "Enabled": st.column_config.CheckboxColumn("Active Flag")
    },
    width="stretch",
    key="data_editor_sample"
)'''
    col_ed1, col_ed2 = st.columns([1, 1])
    with col_ed1:
        st.markdown("**💻 Python Source Code**")
        st.code(code_editor, language="python")
    with col_ed2:
        st.markdown("**🖥️ Rendered Interactive Editor *(Try editing cells!)*:**")
        edited_df = st.data_editor(
            initial_editor_data,
            num_rows="dynamic",
            column_config={
                "Priority": st.column_config.SelectboxColumn(
                    "Priority Level",
                    options=["Low", "Medium", "High", "Critical"],
                    required=True
                ),
                "Weight": st.column_config.NumberColumn(
                    "Feature Weight",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f"
                ),
                "Enabled": st.column_config.CheckboxColumn("Active Flag")
            },
            width="stretch",
            key="data_editor_sample"
        )
        st.caption(f"Reactive State: Total configured active features: **{int(edited_df['Enabled'].sum())}** / **{len(edited_df)}**")

# ==============================================================================
# 10. Extended Topic 2.4 — Displaying Charts and Visualizations
# ==============================================================================
with st.expander("📈 Topic 2.4 — Displaying Charts & Visualizations (Native Charts vs. Plotly, Altair, Seaborn)", expanded=False):
    st.markdown("### 📈 Complete Charting Ecosystem: Native vs. Specialized Libraries")
    st.markdown(
        """
Streamlit bridges both lightweight built-in charting primitives and full-fledged scientific visualization packages:
- **Native Charts (`st.line_chart`, `st.bar_chart`, `st.scatter_chart`, `st.area_chart`)**: Zero-dependency wrappers over Altair that automatically conform to Streamlit light/dark themes.
- **Plotly (`st.plotly_chart`)**: Best for rich interactive charts with hover tooltips, 3D rotations, and responsive zoom/pan.
- **Altair (`st.altair_chart`)**: Declarative grammar of graphics with complex layerings, statistical aggregations, and facets.
- **Matplotlib & Seaborn (`st.pyplot`)**: Industry standards for publication-grade static heatmaps, confusion matrices, and distribution curves.
"""
    )

    chart_tabs = st.tabs(["⚡ Native Streamlit Charts", "📊 Plotly Interactive", "📉 Altair Grammar", "🔬 Seaborn & Matplotlib"])

    # Sample Time-Series Data
    np.random.seed(42)
    epochs = list(range(1, 21))
    train_loss = np.exp(-np.array(epochs) / 5) + 0.05 * np.random.randn(20)
    val_loss = np.exp(-np.array(epochs) / 6) + 0.12 + 0.04 * np.random.randn(20)
    df_learning_curve = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss": np.clip(train_loss, 0.05, 1.5),
        "Validation Loss": np.clip(val_loss, 0.10, 1.5),
        "GPU Utilization (%)": np.random.uniform(70, 98, size=20)
    })

    with chart_tabs[0]:
        st.markdown("#### 1. Native Streamlit Charts (`st.line_chart`, `st.bar_chart`, `st.scatter_chart`)")
        
        # A. Line Chart
        col_lc1, col_lc2 = st.columns(2)
        with col_lc1:
            st.markdown("**💻 Python Code: `st.line_chart`**")
            code_line = '''# Multi-metric convergence line chart
st.line_chart(
    df_learning_curve,
    x="Epoch",
    y=["Train Loss", "Validation Loss"]
)'''
            st.code(code_line, language="python")
        with col_lc2:
            st.markdown("**🖥️ Rendered `st.line_chart`**")
            st.line_chart(df_learning_curve, x="Epoch", y=["Train Loss", "Validation Loss"])

        # B. Bar Chart
        col_bc1, col_bc2 = st.columns(2)
        with col_bc1:
            st.markdown("**💻 Python Code: `st.bar_chart`**")
            code_bar = '''# GPU compute utilization bar chart
st.bar_chart(
    df_learning_curve,
    x="Epoch",
    y="GPU Utilization (%)"
)'''
            st.code(code_bar, language="python")
        with col_bc2:
            st.markdown("**🖥️ Rendered `st.bar_chart`**")
            st.bar_chart(df_learning_curve, x="Epoch", y="GPU Utilization (%)")

        # C. Scatter Chart
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            st.markdown("**💻 Python Code: `st.scatter_chart`**")
            code_scatter = '''# Multi-dimensional scatter plot (color & size encodings)
st.scatter_chart(
    df_learning_curve,
    x="GPU Utilization (%)",
    y="Train Loss",
    color="Epoch",
    size="Validation Loss"
)'''
            st.code(code_scatter, language="python")
        with col_sc2:
            st.markdown("**🖥️ Rendered `st.scatter_chart`**")
            st.scatter_chart(
                df_learning_curve,
                x="GPU Utilization (%)",
                y="Train Loss",
                color="Epoch",
                size="Validation Loss"
            )

    with chart_tabs[1]:
        st.markdown("#### 2. Plotly Interactive Visualization (`st.plotly_chart`)")
        col_pl1, col_pl2 = st.columns(2)
        with col_pl1:
            st.markdown("**💻 Python Source Code**")
            code_plotly = '''import plotly.express as px
import streamlit as st

fig_plotly = px.line(
    df_learning_curve,
    x="Epoch",
    y=["Train Loss", "Validation Loss"],
    markers=True,
    title="Model Convergence Dynamics",
    labels={"value": "Loss Magnitude", "variable": "Partition"}
)
fig_plotly.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_plotly, width="stretch")'''
            st.code(code_plotly, language="python")

        with col_pl2:
            st.markdown("**🖥️ Rendered Plotly Chart *(Zoom, Pan, Hover!)*:**")
            fig_plotly = px.line(
                df_learning_curve,
                x="Epoch",
                y=["Train Loss", "Validation Loss"],
                markers=True,
                title="Model Convergence Dynamics",
                labels={"value": "Loss Magnitude", "variable": "Partition"}
            )
            fig_plotly.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_plotly, width="stretch")

    with chart_tabs[2]:
        st.markdown("#### 3. Altair Declarative Layered Chart (`st.altair_chart`)")
        col_alt1, col_alt2 = st.columns(2)
        with col_alt1:
            st.markdown("**💻 Python Source Code**")
            code_altair = '''import altair as alt
import streamlit as st

base = alt.Chart(df_learning_curve).encode(x="Epoch:Q")
line_train = base.mark_line(color="#38BDF8").encode(
    y=alt.Y("Train Loss:Q", title="Loss")
)
line_val = base.mark_line(color="#F43F5E", strokeDash=[4, 4]).encode(
    y="Validation Loss:Q"
)

altair_layered = (line_train + line_val).properties(
    title="Altair Layered Spec (Train vs Val with Custom Dashed Stroke)"
).interactive()

st.altair_chart(altair_layered, width="stretch")'''
            st.code(code_altair, language="python")

        with col_alt2:
            st.markdown("**🖥️ Rendered Altair Chart**")
            base = alt.Chart(df_learning_curve).encode(x="Epoch:Q")
            line_train = base.mark_line(color="#38BDF8").encode(y=alt.Y("Train Loss:Q", title="Loss"))
            line_val = base.mark_line(color="#F43F5E", strokeDash=[4, 4]).encode(y="Validation Loss:Q")
            altair_layered = (line_train + line_val).properties(
                title="Altair Layered Spec (Train vs Val with Custom Dashed Stroke)"
            ).interactive()
            st.altair_chart(altair_layered, width="stretch")

    with chart_tabs[3]:
        st.markdown("#### 4. Seaborn & Matplotlib Heatmap (`st.pyplot`)")
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            st.markdown("**💻 Python Source Code**")
            code_sns = '''import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fig_sns, ax_sns = plt.subplots(figsize=(6, 3.5), facecolor="#0F172A")
ax_sns.set_facecolor("#0F172A")
corr_matrix = df_learning_curve.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="mako",
    fmt=".2f",
    ax=ax_sns,
    cbar=True,
    annot_kws={"color": "white", "size": 10}
)
ax_sns.tick_params(colors="white")
for spine in ax_sns.spines.values():
    spine.set_color("#334155")

st.pyplot(fig_sns)
plt.close(fig_sns)'''
            st.code(code_sns, language="python")

        with col_sb2:
            st.markdown("**🖥️ Rendered Matplotlib/Seaborn Figure**")
            fig_sns, ax_sns = plt.subplots(figsize=(6, 3.5), facecolor="#0F172A")
            ax_sns.set_facecolor("#0F172A")
            corr_matrix = df_learning_curve.corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="mako",
                fmt=".2f",
                ax=ax_sns,
                cbar=True,
                annot_kws={"color": "white", "size": 10}
            )
            ax_sns.tick_params(colors="white")
            for spine in ax_sns.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig_sns)
            plt.close(fig_sns)

# ==============================================================================
# 11. Extended Topic 2.5 — Displaying Media
# ==============================================================================
with st.expander("🎬 Topic 2.5 — Displaying Media (st.image, st.audio, st.video, and Camera Input)", expanded=False):
    st.markdown("### 🎬 Multimedia Integration: Images, Audio, Video & Webcams")
    st.markdown(
        """
Streamlit handles images, synthesized audio, streamed video, and client-side hardware inputs seamlessly:
- **`st.image()`**: Handles URLs, local files, NumPy RGB matrices, and PIL `Image` objects.
- **`st.audio()`**: Audio player accepting binary audio bytes, local files, and web streams.
- **`st.video()`**: Responsive video player supporting local MP4/WebM files, YouTube, and Vimeo URLs.
- **`st.camera_input()`**: Activates the user's browser webcam to capture live frames for inference.
"""
    )

    media_tabs = st.tabs(["🖼️ Images (PIL & Arrays)", "🎵 Audio (Synthesized Waves)", "📹 Video (Embeds & Local)", "📷 Camera Input (Webcam)"])

    with media_tabs[0]:
        st.markdown("#### 1. Displaying Images (`st.image`)")
        col_img1, col_img2 = st.columns(2)
        
        # In-memory PIL Image Generation
        pil_canvas = Image.new("RGB", (480, 240), color=(15, 23, 42))
        draw = ImageDraw.Draw(pil_canvas)
        draw.rectangle([20, 20, 460, 220], outline=(56, 189, 248), width=3)
        draw.ellipse([60, 50, 160, 150], fill=(99, 102, 241), outline=(129, 140, 248), width=2)
        draw.rectangle([220, 60, 420, 140], fill=(16, 185, 129))
        draw.text((75, 175), "Generated via PIL (In-Memory Canvas)", fill=(248, 250, 252))

        with col_img1:
            st.markdown("**💻 Python Source Code**")
            code_img = '''from PIL import Image, ImageDraw
import streamlit as st

# Generate canvas in memory
canvas = Image.new("RGB", (480, 240), color=(15, 23, 42))
draw = ImageDraw.Draw(canvas)
draw.ellipse([60, 50, 160, 150], fill=(99, 102, 241))

# Render with caption
st.image(canvas, caption="Dynamically Generated Vector Shapes", width="stretch")'''
            st.code(code_img, language="python")

        with col_img2:
            st.markdown("**🖥️ Rendered Image Output**")
            st.image(pil_canvas, caption="Dynamically Generated Vector Shapes (PIL Canvas)", width="stretch")

    with media_tabs[1]:
        st.markdown("#### 2. Displaying Audio (`st.audio`)")
        st.markdown("Generate and synthesize an acoustic sine wave completely in-memory using `numpy` and `wave`:")
        
        col_aud1, col_aud2 = st.columns(2)
        with col_aud1:
            st.markdown("**💻 Python Source Code**")
            code_aud = '''import numpy as np
import io, wave
import streamlit as st

# Synthesize acoustic sine wave in memory
sample_rate = 44100
duration = 1.2
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
waveform = (np.sin(2 * np.pi * 440 * t) * np.exp(-2.0 * t) * 32767).astype(np.int16)

wav_io = io.BytesIO()
with wave.open(wav_io, "wb") as wav_out:
    wav_out.setnchannels(1)
    wav_out.setsampwidth(2)
    wav_out.setframerate(sample_rate)
    wav_out.writeframes(waveform.tobytes())

# Render audio player with in-memory bytes
st.audio(wav_io.getvalue(), format="audio/wav")'''
            st.code(code_aud, language="python")

        with col_aud2:
            st.markdown("**🖥️ Rendered Audio Player**")
            freq_input = st.slider("Select Synthesizer Frequency (Hz):", min_value=220, max_value=880, value=440, step=20, key="synth_freq")
            sample_rate = 44100
            duration = 1.2
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            envelope = np.exp(-2.0 * t)
            waveform = (np.sin(2 * np.pi * freq_input * t) * envelope * 32767).astype(np.int16)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(sample_rate)
                wav_out.writeframes(waveform.tobytes())
            synth_bytes = wav_io.getvalue()
            st.audio(synth_bytes, format="audio/wav")
            st.caption(f"Playing synthesized tone at **{freq_input} Hz** with exponential decay envelope.")

    with media_tabs[2]:
        st.markdown("#### 3. Displaying Video (`st.video`)")
        st.markdown("Streamlit supports both direct video file buffers and public streaming links:")
        col_vid1, col_vid2 = st.columns(2)
        with col_vid1:
            st.markdown("**💻 Python Source Code**")
            code_vid = '''# Stream YouTube or MP4 video URL:
st.video(
    "https://www.youtube.com/watch?v=B2iAODr0fOo",
    start_time=10
)'''
            st.code(code_vid, language="python")
        with col_vid2:
            st.markdown("**🖥️ Rendered Video Player**")
            st.video("https://www.youtube.com/watch?v=B2iAODr0fOo", start_time=10)

    with media_tabs[3]:
        st.markdown("#### 4. Hardware Camera Input (`st.camera_input`)")
        st.markdown(
            """
            `st.camera_input()` accesses the browser's webcam permission dialog. 
            When a user takes a photo, the frame bytes are returned as an `UploadedFile` object 
            ready for computer vision and multimodal LLM inference!
            """
        )
        
        col_cam_c1, col_cam_c2 = st.columns(2)
        with col_cam_c1:
            st.markdown("**💻 Python Source Code**")
            code_cam = '''# Capture webcam frame from browser:
cam_photo = st.camera_input("Take a snapshot for Computer Vision Analysis")

if cam_photo is not None:
    st.success("Snapshot captured!")
    st.image(cam_photo, caption="Captured Frame", width="stretch")
    
    # Process with PIL / OpenCV:
    img_obj = Image.open(cam_photo)
    st.write(f"Resolution: {img_obj.width}x{img_obj.height}")'''
            st.code(code_cam, language="python")

        with col_cam_c2:
            st.markdown("**🖥️ Rendered Camera Input Widget**")
            cam_photo = st.camera_input("Take a snapshot for Computer Vision Analysis", key="cv_camera")
            if cam_photo is not None:
                st.success("Snapshot captured successfully!")
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    st.image(cam_photo, caption="Captured Frame", width="stretch")
                with c_c2:
                    img_obj = Image.open(cam_photo)
                    st.markdown(f"""
                    **Metadata Extraction:**
                    - **Image Dimensions**: `{img_obj.width} x {img_obj.height}` pixels
                    - **Color Format**: `{img_obj.format}` ({img_obj.mode})
                    - **Buffer Size**: `{cam_photo.size / 1024:.2f} KB`
                    """)


