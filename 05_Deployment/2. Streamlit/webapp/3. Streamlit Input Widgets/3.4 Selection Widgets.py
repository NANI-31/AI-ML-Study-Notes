import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Selection Widgets Tutorial",
    page_icon="🔘",
    layout="wide",
)

st.title("🔘 Selection Widgets — Complete Syntax & Architecture Guide")
st.markdown(
    """
Selection widgets enable users to choose single or multiple items from a defined collection. 
Streamlit offers classic form inputs (**`st.selectbox`**, **`st.multiselect`**, **`st.radio`**) 
alongside modern, sleek UI primitives (**`st.segmented_control`**, **`st.pills`**).
This guide covers formal signatures, default indices, multi-selection handling, and dynamic chained dependencies.
"""
)

st.divider()

# ==============================================================================
# 1. st.selectbox() (Dropdown Selector)
# ==============================================================================
st.header("1. Dropdown Selection (`st.selectbox`)", divider="rainbow")
st.markdown(
    """
`st.selectbox()` displays a searchable dropdown list for choosing a **single option**:
- Supports type-to-search with fuzzy filtering.
- Set `index=None` with a `placeholder` to force the user to make an explicit choice.
- Accepts lists, tuples, dictionaries, or DataFrame columns.
"""
)

st.markdown("#### 📐 Formal Signature: `st.selectbox()`")
st.code(
    """st.selectbox(
    label: str,
    options: Sequence[T],
    index: int | None = 0,
    format_func: Callable[[T], str] = str,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    placeholder: str | None = None,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    accept_new_options: bool = False,
    width: "stretch" | "content" = "stretch"
) -> T | None""",
    language="python",
)

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**💻 Python Source Code**")
    code_selectbox = '''# Single selectbox with placeholder and no default:
models = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B", "DeepSeek R1"]

chosen_model = st.selectbox(
    label="Choose Foundation Model",
    options=models,
    index=None,
    placeholder="Select an LLM from registry...",
    help="Target LLM for generating completions"
)

st.write("Selected:", chosen_model)'''
    st.code(code_selectbox, language="python")

with col_sb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    models_list = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B", "DeepSeek R1"]
    chosen_model = st.selectbox(
        label="Choose Foundation Model",
        options=models_list,
        index=None,
        placeholder="Select an LLM from registry...",
        help="Target LLM for generating completions",
        key="sb_demo_model"
    )
    st.caption(f"Active Selection: **{chosen_model or '(No selection made)'}**")

st.divider()

# ==============================================================================
# 2. st.multiselect() (Tag & Multi-Item Selector)
# ==============================================================================
st.header("2. Tag-Based Multi-Selection (`st.multiselect`)", divider="rainbow")
st.markdown(
    """
`st.multiselect()` allows users to pick **multiple items** rendered as dismissible badge tags:
- Returns a Python `list[T]` of all selected elements.
- **`default`**: Pre-selects a list of initial items.
- **`max_selections`**: Enforces a hard upper bound on allowed selections.
"""
)

st.markdown("#### 📐 Formal Signature: `st.multiselect()`")
st.code(
    """st.multiselect(
    label: str,
    options: Sequence[T],
    default: Sequence[T] | None = None,
    format_func: Callable[[T], str] = str,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    max_selections: int | None = None,
    placeholder: str | None = None,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    width: "stretch" | "content" = "stretch"
) -> list[T]""",
    language="python",
)

col_ms1, col_ms2 = st.columns(2)

with col_ms1:
    st.markdown("**💻 Python Source Code**")
    code_multiselect = '''benchmarks = ["MMLU", "GSM8K", "HumanEval", "MATH", "HellaSwag", "ARC-Challenge"]

selected_evals = st.multiselect(
    label="Select Evaluation Benchmarks (Max 3)",
    options=benchmarks,
    default=["MMLU", "HumanEval"],
    max_selections=3,
    help="Choose benchmarks to calculate composite benchmark score"
)

st.write("Active Benchmarks:", selected_evals)'''
    st.code(code_multiselect, language="python")

with col_ms2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    benchmarks_list = ["MMLU", "GSM8K", "HumanEval", "MATH", "HellaSwag", "ARC-Challenge"]
    selected_evals = st.multiselect(
        label="Select Evaluation Benchmarks (Max 3)",
        options=benchmarks_list,
        default=["MMLU", "HumanEval"],
        max_selections=3,
        help="Choose benchmarks to calculate composite benchmark score",
        key="ms_demo_evals"
    )
    st.caption(f"Selected Count: **{len(selected_evals)} / 3** $\\rightarrow$ `{selected_evals}`")

st.divider()

# ==============================================================================
# 3. st.radio() (Radio Buttons with Captions & Horizontal Mode)
# ==============================================================================
st.header("3. Radio Button Groups (`st.radio`)", divider="rainbow")
st.markdown(
    """
`st.radio()` displays mutually exclusive single-choice options:
- **`horizontal=True`**: Arranges options side-by-side in a responsive row.
- **`captions`**: Renders secondary explanatory subtitle text directly beneath each option.
- **`index=None`**: Starts with no option selected.
"""
)

st.markdown("#### 📐 Formal Signature: `st.radio()`")
st.code(
    """st.radio(
    label: str,
    options: Sequence[T],
    index: int | None = 0,
    format_func: Callable[[T], str] = str,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    horizontal: bool = False,
    captions: Sequence[str] | None = None,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible"
) -> T | None""",
    language="python",
)

col_rd1, col_rd2 = st.columns(2)

with col_rd1:
    st.markdown("**💻 Python Source Code (with Captions)**")
    code_radio = '''precision_modes = ["FP32 (Full)", "FP16 (Half)", "INT8 (Quantized)", "INT4 (Extreme)"]
mode_captions = [
    "Highest fidelity, 4 bytes/param",
    "Standard GPU serving, 2 bytes/param",
    "Near-lossless, 1 byte/param",
    "Minimum VRAM, edge inference"
]

selected_prec = st.radio(
    label="Quantization Precision",
    options=precision_modes,
    captions=mode_captions,
    index=1,
    horizontal=True
)

st.write("Active Precision:", selected_prec)'''
    st.code(code_radio, language="python")

with col_rd2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    precision_modes = ["FP32 (Full)", "FP16 (Half)", "INT8 (Quantized)", "INT4 (Extreme)"]
    mode_captions = [
        "Highest fidelity, 4 bytes/param",
        "Standard GPU serving, 2 bytes/param",
        "Near-lossless, 1 byte/param",
        "Minimum VRAM, edge inference"
    ]
    selected_prec = st.radio(
        label="Quantization Precision",
        options=precision_modes,
        captions=mode_captions,
        index=1,
        horizontal=True,
        key="rd_demo_prec"
    )
    st.caption(f"Active Mode: **{selected_prec}**")

st.divider()

# ==============================================================================
# 4. st.segmented_control() (Modern Segmented Button Bar)
# ==============================================================================
st.header("4. Modern Segmented Control (`st.segmented_control`)", divider="rainbow")
st.markdown(
    """
`st.segmented_control()` renders a cohesive horizontal button bar:
- **`selection_mode="single"`**: Functions as a visual alternative to radio buttons.
- **`selection_mode="multi"`**: Allows toggling multiple segments concurrently.
- Highly recommended for view switchers, chart timescale pickers (1D, 1W, 1M, 1Y), and inference modes.
"""
)

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**💻 Python Source Code**")
    code_seg = '''# Single-choice segmented control:
engine_options = ["vLLM", "Triton", "Ollama", "TGI"]
selected_engine = st.segmented_control(
    label="Inference Runtime Engine",
    options=engine_options,
    default="vLLM",
    selection_mode="single"
)

st.write("Selected Engine:", selected_engine)'''
    st.code(code_seg, language="python")

with col_sc2:
    st.markdown("**🖥️ Rendered Segmented Control**")
    engine_options = ["vLLM", "Triton", "Ollama", "TGI"]
    selected_engine = st.segmented_control(
        label="Inference Runtime Engine",
        options=engine_options,
        default="vLLM",
        selection_mode="single",
        key="sc_demo_engine"
    )
    st.caption(f"Serving Stack: **{selected_engine}**")

st.divider()

# ==============================================================================
# 5. st.pills() (Clickable Tag Badges)
# ==============================================================================
st.header("5. Clickable Pill Badges (`st.pills`)", divider="rainbow")
st.markdown(
    """
`st.pills()` renders rounded badge chips designed for filtering and categorical tagging:
- Compact, aesthetic visual presentation.
- Supports both `selection_mode="single"` and `selection_mode="multi"`.
- Supports emojis and custom icons embedded directly in option labels.
"""
)

col_pil1, col_pil2 = st.columns(2)

with col_pil1:
    st.markdown("**💻 Python Source Code (Multi-Select Pills)**")
    code_pills = '''task_categories = ["💬 Chatbot", "🔍 RAG Search", "💻 Code Gen", "🖼️ Vision", "📊 Analytics"]

selected_tasks = st.pills(
    label="Filter GenAI Capabilities",
    options=task_categories,
    default=["💬 Chatbot", "🔍 RAG Search"],
    selection_mode="multi"
)

st.write("Active Filters:", selected_tasks)'''
    st.code(code_pills, language="python")

with col_pil2:
    st.markdown("**🖥️ Rendered Pill Badges**")
    task_categories = ["💬 Chatbot", "🔍 RAG Search", "💻 Code Gen", "🖼️ Vision", "📊 Analytics"]
    selected_tasks = st.pills(
        label="Filter GenAI Capabilities",
        options=task_categories,
        default=["💬 Chatbot", "🔍 RAG Search"],
        selection_mode="multi",
        key="pills_demo_tasks"
    )
    st.caption(f"Selected Capabilities ({len(selected_tasks)}): `{selected_tasks}`")

st.divider()

# ==============================================================================
# 6. Default Selections & 7. Selection Indices
# ==============================================================================
st.header("6. Default Selections & 7. Selection Indices", divider="rainbow")
st.markdown(
    """
Streamlit handles defaults differently depending on the widget architecture:
- **`st.selectbox` and `st.radio`**: Use **`index: int | None`** (0-based integer position in the `options` sequence).
- **`st.multiselect`, `st.pills`, `st.segmented_control`**: Use **`default: Sequence[T]`** (actual list of value items).
"""
)

col_idx1, col_idx2 = st.columns(2)

with col_idx1:
    st.markdown("**💻 Python Source Code (Index vs Value Defaults)**")
    code_indices = '''tiers = ["Bronze Tier", "Silver Tier", "Gold Tier", "Platinum Enterprise"]

# 1. Using index=2 selects 'Gold Tier' (0-indexed position 2)
selected_tier = st.selectbox("Customer Tier (Default by Index)", options=tiers, index=2)

# 2. Finding the active index programmatically:
active_index = tiers.index(selected_tier)
st.write(f"Active Item: {selected_tier} (Index: {active_index})")'''
    st.code(code_indices, language="python")

with col_idx2:
    st.markdown("**🖥️ Rendered Output**")
    tiers = ["Bronze Tier", "Silver Tier", "Gold Tier", "Platinum Enterprise"]
    selected_tier = st.selectbox("Customer Tier (Default by Index)", options=tiers, index=2, key="sb_demo_index")
    active_index = tiers.index(selected_tier)
    st.caption(f"Resolved Selection: **{selected_tier}** | Integer Index: **{active_index}**")

st.divider()

# ==============================================================================
# 8. Handling Multiple Selections (Filtering DataFrames)
# ==============================================================================
st.header("8. Handling Multiple Selections (DataFrame Filtering)", divider="rainbow")
st.markdown(
    """
The return value of multi-select widgets is a standard Python `list`. 
The most common production use case is filtering pandas DataFrames with `.isin()`:
"""
)

# Sample Model Catalog
models_catalog = pd.DataFrame({
    "Model": ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.3 70B", "Mistral Large", "Qwen 2.5 72B"],
    "Provider": ["OpenAI", "Anthropic", "Google", "Meta", "Mistral", "Alibaba"],
    "Modality": ["Multimodal", "Multimodal", "Multimodal", "Text-Only", "Text-Only", "Text-Only"],
    "MMLU": [92.0, 94.0, 91.0, 88.5, 84.0, 86.2]
})

col_df_filt1, col_df_filt2 = st.columns(2)

with col_df_filt1:
    st.markdown("**💻 Python Source Code (Dynamic Filtering)**")
    code_filter = '''# Multi-select options derived from unique DataFrame column:
all_providers = models_catalog["Provider"].unique().tolist()
selected_provs = st.multiselect("Filter by Provider", options=all_providers, default=all_providers)

# Dynamic slice using .isin():
filtered_df = models_catalog[models_catalog["Provider"].isin(selected_provs)]
st.dataframe(filtered_df, width="stretch", hide_index=True)'''
    st.code(code_filter, language="python")

with col_df_filt2:
    st.markdown("**🖥️ Rendered Filterable Catalog**")
    all_providers = models_catalog["Provider"].unique().tolist()
    selected_provs = st.multiselect("Filter by Provider", options=all_providers, default=all_providers, key="ms_filter_provs")

    filtered_df = models_catalog[models_catalog["Provider"].isin(selected_provs)]
    st.dataframe(filtered_df, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 9. Dynamic Chained Selection Options (Dependent Dropdowns)
# ==============================================================================
st.header("9. Dynamic Chained Selection Options (Dependent Dropdowns)", divider="rainbow")
st.markdown(
    """
In hierarchical configurations, the options available in a secondary dropdown depend on the choice made 
in the primary dropdown (e.g., **Provider $\\rightarrow$ Available Models**):
- When the parent dropdown changes, the script reruns.
- The child dropdown receives the updated filtered options list automatically!
"""
)

model_hierarchy = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "o1-preview", "o3-mini"],
    "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"],
    "Meta (Open Weights)": ["llama-3.3-70b-instruct", "llama-3.1-8b-instruct"],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
}

col_ch1, col_ch2 = st.columns(2)

with col_ch1:
    st.markdown("**💻 Python Source Code (Chained Dependent Selectors)**")
    code_chained = '''hierarchy = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "o1-preview"],
    "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
    "Google": ["gemini-1.5-pro", "gemini-2.0-flash"]
}

# 1. Primary dropdown (Provider)
provider = st.selectbox("1. Select Cloud Provider", options=list(hierarchy.keys()))

# 2. Secondary dropdown (Dynamically populated from parent choice!)
available_models = hierarchy[provider]
model = st.selectbox("2. Select Model", options=available_models)

st.write(f"Deploying: **{provider}** / `{model}`")'''
    st.code(code_chained, language="python")

with col_ch2:
    st.markdown("**🖥️ Rendered Chained Dropdowns**")
    selected_parent = st.selectbox("1. Select Cloud Provider", options=list(model_hierarchy.keys()), key="chain_parent")
    child_options = model_hierarchy[selected_parent]
    selected_child = st.selectbox("2. Select Architecture Model", options=child_options, key="chain_child")

    st.success(f"Deployment Target: **{selected_parent}** $\\rightarrow$ `{selected_child}`")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Selection Widgets Comparison Matrix", divider="rainbow")
st.markdown(
    """
| Widget | Selection Cardinality | Best Visual Use Case | Key Differentiator |
| :--- | :--- | :--- | :--- |
| **`st.selectbox`** | Single | Long lists (> 6 items) | Searchable dropdown with fuzzy filtering |
| **`st.multiselect`** | Multiple | Multi-tag filters (1 to N items) | Dismissible chip tags, search, `max_selections` |
| **`st.radio`** | Single | Short lists (2–5 items) | Supports item `captions` and `horizontal=True` |
| **`st.segmented_control`** | Single or Multi | Timescale / Mode switchers | Connected button bar (Apple segment style) |
| **`st.pills`** | Single or Multi | Categorical badge filters | Rounded chip pills with optional icons |
"""
)
