import streamlit as st
import pandas as pd
from datetime import date, datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit DataFrame Customization Tutorial",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 DataFrame Customization (`st.column_config`) — Complete Guide")
st.markdown(
    """
Streamlit's **`st.column_config`** API transforms basic tabular grids into rich, publication-grade analytical dashboards. 
Instead of modifying your underlying Pandas DataFrame schemas or converting numbers to string strings, 
`st.column_config` allows you to customize column display types, rename headers, control widths, 
embed visual progress bars, render avatars/logos, build clickable hyperlinks, and format temporal dates.
"""
)

st.divider()

# ==============================================================================
# 1. Column Configuration Core Architecture
# ==============================================================================
st.header("1. Core Architecture & `column_config` Dictionary Mapping", divider="rainbow")
st.markdown(
    """
To customize columns, pass a dictionary to the `column_config` parameter of `st.dataframe()` or `st.data_editor()`:
- **Keys**: Must match existing column names in your DataFrame (or index).
- **Values**: Instances of `st.column_config.<Type>Column(...)` or `None` (to hide a column).
- **Non-Destructive**: The original DataFrame data types (e.g. `float64`, `datetime64`) remain pristine for downstream computations.
"""
)

st.markdown("#### 📐 Formal Pattern: `column_config` Mapping")
st.code(
    """st.dataframe(
    data=df,
    column_config={
        "col_name_1": st.column_config.NumberColumn("Display Label", format="$%.2f"),
        "col_name_2": st.column_config.ProgressColumn("Benchmark Score", min_value=0, max_value=100),
        "col_name_3": None  # Setting to None hides the column entirely!
    },
    width="stretch"
)""",
    language="python",
)

st.divider()

# ==============================================================================
# 2. Column Labels & 3. Column Widths
# ==============================================================================
st.header("2. Custom Column Labels & 3. Column Widths", divider="rainbow")
st.markdown(
    """
- **Column Labels (`label`)**: Rename database/raw column names (e.g. `m_name_v2` → `🤖 Model Name`) without renaming DataFrame columns in Python. Supports emojis and GitHub-flavored markdown.
- **Column Widths (`width`)**: Constrain or expand column sizes using preset sizes (`"small"`, `"medium"`, `"large"`) or specific pixel widths.
"""
)

raw_spec_df = pd.DataFrame({
    "m_id": ["GPT4O", "CL35S", "DSV3"],
    "ctx_len": [128000, 200000, 128000],
    "tok_rate": [112.5, 95.0, 140.2]
})

col_lw1, col_lw2 = st.columns(2)

with col_lw1:
    st.markdown("**💻 Python Source Code**")
    code_labels_widths = '''# Custom labels and widths without touching DataFrame columns:
st.dataframe(
    raw_spec_df,
    column_config={
        "m_id": st.column_config.TextColumn(
            label="🤖 Model Identifier",
            width="medium",
            help="Unique production model deployment code"
        ),
        "ctx_len": st.column_config.NumberColumn(
            label="🧠 Context Window",
            width="small",
            format="%d tokens"
        ),
        "tok_rate": st.column_config.NumberColumn(
            label="⚡ Throughput",
            width="small",
            format="%.1f tok/s"
        )
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_labels_widths, language="python")

with col_lw2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        raw_spec_df,
        column_config={
            "m_id": st.column_config.TextColumn(
                label="🤖 Model Identifier",
                width="medium",
                help="Unique production model deployment code"
            ),
            "ctx_len": st.column_config.NumberColumn(
                label="🧠 Context Window",
                width="small",
                format="%d tokens"
            ),
            "tok_rate": st.column_config.NumberColumn(
                label="⚡ Throughput",
                width="small",
                format="%.1f tok/s"
            )
        },
        width="stretch",
        hide_index=True,
        key="demo_labels_widths"
    )
    st.caption("💡 *Notice: Column headers have emojis and friendly labels, while underlying column names stay clean.*")

st.divider()

# ==============================================================================
# 4. Column Visibility & Ordering
# ==============================================================================
st.header("4. Controlling Column Visibility & Display Order", divider="rainbow")
st.markdown(
    """
There are two powerful mechanisms to control which columns appear:
1. **`column_config={"col_name": None}`**: Completely hides specific columns (e.g. internal primary keys, backend hashes, sensitive metadata).
2. **`column_order=[...]`**: Explicitly defines the left-to-right display order of columns.
"""
)

sensitive_df = pd.DataFrame({
    "internal_id": ["UUID-9918", "UUID-4421", "UUID-1034"],
    "model_name": ["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B"],
    "mmlu": [68.4, 64.2, 71.3],
    "secret_api_key": ["sk-proj-xyz99", "sk-proj-abc12", "sk-proj-def44"]
})

col_vis1, col_vis2 = st.columns(2)

with col_vis1:
    st.markdown("**💻 Python Source Code**")
    code_vis = '''# Hide sensitive columns using None and enforce custom column order:
st.dataframe(
    sensitive_df,
    column_order=["model_name", "mmlu"],  # Left-to-right order
    column_config={
        "secret_api_key": None,  # Hides column from the grid!
        "internal_id": None,     # Hides internal UUID
        "model_name": st.column_config.TextColumn("Model"),
        "mmlu": st.column_config.NumberColumn("MMLU Score", format="%.1f%%")
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_vis, language="python")

with col_vis2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        sensitive_df,
        column_order=["model_name", "mmlu"],
        column_config={
            "secret_api_key": None,
            "internal_id": None,
            "model_name": st.column_config.TextColumn("Model"),
            "mmlu": st.column_config.NumberColumn("MMLU Score", format="%.1f%%")
        },
        width="stretch",
        hide_index=True,
        key="demo_visibility"
    )
    st.caption("🔒 *`secret_api_key` and `internal_id` are safely hidden from the rendered user table.*")

st.divider()

# ==============================================================================
# 5. Number Formatting & 6. Date Formatting
# ==============================================================================
st.header("5. Number Formatting & 6. Date Formatting", divider="rainbow")
st.markdown(
    """
- **`st.column_config.NumberColumn`**: Applies printf formatting (e.g. `"$%.2f"`, `"%.1f%%"`, `"%,d"`), sets min/max validation bounds, and defines step sizes.
- **`st.column_config.DateColumn` / `DatetimeColumn`**: Formats timestamps into human-readable strings (e.g. `"YYYY-MM-DD"`, `"MMM DD, YYYY"`, `"DD/MM/YYYY, hh:mm a"`).
"""
)

financial_df = pd.DataFrame({
    "Release_Date": [date(2024, 5, 13), date(2024, 6, 20), date(2024, 12, 26)],
    "Timestamp": [datetime(2024, 5, 13, 17, 30), datetime(2024, 6, 20, 14, 15), datetime(2024, 12, 26, 9, 0)],
    "Input_Cost_USD": [2.50, 3.00, 0.27],
    "Token_Count": [15420000, 8920000, 45100000],
    "Error_Rate": [0.012, 0.008, 0.015]
})

col_fmt1, col_fmt2 = st.columns(2)

with col_fmt1:
    st.markdown("**💻 Python Source Code**")
    code_fmt = '''st.dataframe(
    financial_df,
    column_config={
        "Release_Date": st.column_config.DateColumn(
            "Release Date",
            format="YYYY-MM-DD"
        ),
        "Timestamp": st.column_config.DatetimeColumn(
            "Deployment Timestamp",
            format="DD/MM/YYYY, hh:mm a"
        ),
        "Input_Cost_USD": st.column_config.NumberColumn(
            "Cost / 1M Tokens",
            format="$%.2f"
        ),
        "Token_Count": st.column_config.NumberColumn(
            "Total Ingested Tokens",
            format="%,d"
        ),
        "Error_Rate": st.column_config.NumberColumn(
            "Failure Rate",
            format="%.2f%%"
        )
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_fmt, language="python")

with col_fmt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        financial_df,
        column_config={
            "Release_Date": st.column_config.DateColumn(
                "Release Date",
                format="YYYY-MM-DD"
            ),
            "Timestamp": st.column_config.DatetimeColumn(
                "Deployment Timestamp",
                format="DD/MM/YYYY, hh:mm a"
            ),
            "Input_Cost_USD": st.column_config.NumberColumn(
                "Cost / 1M Tokens",
                format="$%.2f"
            ),
            "Token_Count": st.column_config.NumberColumn(
                "Total Ingested Tokens",
                format="%,d"
            ),
            "Error_Rate": st.column_config.NumberColumn(
                "Failure Rate",
                format="%.2f%%"
            )
        },
        width="stretch",
        hide_index=True,
        key="demo_number_date_fmt"
    )

st.divider()

# ==============================================================================
# 7. Progress Columns — Visual Percentage & Rating Bars
# ==============================================================================
st.header("7. Progress Columns (`st.column_config.ProgressColumn`)", divider="rainbow")
st.markdown(
    """
`st.column_config.ProgressColumn` transforms numeric columns into interactive visual progress bars:
- **`min_value` & `max_value`**: Scale bounds (e.g. `0.0` to `1.0` or `0` to `100`).
- **`format`**: Displays the numeric percentage or count alongside the bar.
- Ideal for model accuracy scores, GPU memory utilization, quota consumption, and progress bars.
"""
)

progress_df = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Llama-3.3-70B"],
    "MMLU_Score": [88.7, 88.6, 88.5, 86.4],
    "GPU_VRAM_GB": [68.5, 78.2, 54.0, 92.4]
})

col_pr1, col_pr2 = st.columns(2)

with col_pr1:
    st.markdown("**💻 Python Source Code**")
    code_progress = '''st.dataframe(
    progress_df,
    column_config={
        "Model": st.column_config.TextColumn("Foundation Model"),
        "MMLU_Score": st.column_config.ProgressColumn(
            label="MMLU Benchmark Score",
            help="General reasoning benchmark (0 to 100%)",
            format="%.1f%%",
            min_value=0.0,
            max_value=100.0
        ),
        "GPU_VRAM_GB": st.column_config.ProgressColumn(
            label="VRAM Allocation (80 GB A100)",
            format="%.1f GB",
            min_value=0.0,
            max_value=80.0
        )
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_progress, language="python")

with col_pr2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        progress_df,
        column_config={
            "Model": st.column_config.TextColumn("Foundation Model"),
            "MMLU_Score": st.column_config.ProgressColumn(
                label="MMLU Benchmark Score",
                help="General reasoning benchmark (0 to 100%)",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0
            ),
            "GPU_VRAM_GB": st.column_config.ProgressColumn(
                label="VRAM Allocation (80 GB A100)",
                format="%.1f GB",
                min_value=0.0,
                max_value=80.0
            )
        },
        width="stretch",
        hide_index=True,
        key="demo_progress_col"
    )

st.divider()

# ==============================================================================
# 8. Image Columns & 9. Link Columns
# ==============================================================================
st.header("8. Image Columns & 9. Link Columns", divider="rainbow")
st.markdown(
    """
- **`st.column_config.ImageColumn`**: Renders image URLs (PNG, JPEG, WebP, SVG) as compact cell thumbnails.
- **`st.column_config.LinkColumn`**: Renders clickable hyperlinks. Use `display_text` to show friendly anchor text (e.g. `"View Docs"`) instead of raw, verbose URLs.
"""
)

multimedia_df = pd.DataFrame({
    "Provider": ["Streamlit", "Python", "GitHub"],
    "Logo": [
        "https://raw.githubusercontent.com/streamlit/brand/main/mark/streamlit-mark-color.png",
        "https://www.python.org/static/community_logos/python-logo-generic.svg",
        "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
    ],
    "Documentation": [
        "https://docs.streamlit.io",
        "https://docs.python.org/3",
        "https://docs.github.com"
    ]
})

col_ml1, col_ml2 = st.columns(2)

with col_ml1:
    st.markdown("**💻 Python Source Code**")
    code_img_link = '''st.dataframe(
    multimedia_df,
    column_config={
        "Logo": st.column_config.ImageColumn(
            label="Avatar / Logo",
            help="Official organization badge"
        ),
        "Documentation": st.column_config.LinkColumn(
            label="Official Documentation",
            display_text="Open Docs ↗",
            help="Click to open external reference documentation"
        )
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_img_link, language="python")

with col_ml2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        multimedia_df,
        column_config={
            "Logo": st.column_config.ImageColumn(
                label="Avatar / Logo",
                help="Official organization badge"
            ),
            "Documentation": st.column_config.LinkColumn(
                label="Official Documentation",
                display_text="Open Docs ↗",
                help="Click to open external reference documentation"
            )
        },
        width="stretch",
        hide_index=True,
        key="demo_img_link_col"
    )

st.divider()

# ==============================================================================
# 10. Checkbox & Selection Columns
# ==============================================================================
st.header("10. Checkbox & Selection Columns", divider="rainbow")
st.markdown(
    """
- **`st.column_config.CheckboxColumn`**: Renders boolean values (`True`/`False`) as visual checkmarks inside table cells.
- **`on_select="rerun"` & `selection_mode`**: Turns standard `st.dataframe()` grids into interactive row/column selectors returning `st.session_state` selection indices!
"""
)

selection_demo_df = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Mistral-Large-2"],
    "In_Production": [True, True, True, False],
    "Approved_for_HIPAA": [True, False, False, True],
    "Monthly_Requests": [1250000, 980000, 1850000, 420000]
})

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    st.markdown("**💻 Python Source Code**")
    code_selection = '''# 1. Custom Checkbox columns:
# 2. Interactive row selection callback:
event = st.dataframe(
    selection_demo_df,
    column_config={
        "In_Production": st.column_config.CheckboxColumn("Production Ready"),
        "Approved_for_HIPAA": st.column_config.CheckboxColumn("HIPAA Certified"),
        "Monthly_Requests": st.column_config.NumberColumn("Volume", format="%,d")
    },
    on_select="rerun",
    selection_mode="multi-row",
    width="stretch",
    hide_index=True
)

selected_rows = event.selection.rows
if selected_rows:
    st.write("Selected Rows:", selected_rows)'''
    st.code(code_selection, language="python")

with col_sel2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    event_data = st.dataframe(
        selection_demo_df,
        column_config={
            "In_Production": st.column_config.CheckboxColumn("Production Ready"),
            "Approved_for_HIPAA": st.column_config.CheckboxColumn("HIPAA Certified"),
            "Monthly_Requests": st.column_config.NumberColumn("Volume", format="%,d")
        },
        on_select="rerun",
        selection_mode="multi-row",
        width="stretch",
        hide_index=True,
        key="demo_checkbox_col"
    )

    selected_rows = event_data.selection.rows
    if selected_rows:
        chosen_names = selection_demo_df.iloc[selected_rows]["Model"].tolist()
        st.success(f"Selected Models ({len(chosen_names)}): **{', '.join(chosen_names)}**")
    else:
        st.caption("💡 *Click on any row checkbox on the left to select models and trigger reactive reruns.*")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Master `st.column_config` Reference Guide", divider="rainbow")
st.markdown(
    """
| Column Class | Primary Options & Parameters | Example Usage |
| :--- | :--- | :--- |
| **`st.column_config.Column`** | `label`, `width`, `help`, `disabled` | Base class for general column configuration |
| **`st.column_config.TextColumn`** | `label`, `width`, `help`, `max_chars` | Clean strings, codes, and identifiers |
| **`st.column_config.NumberColumn`** | `format` (`"$%.2f"`, `"%,d"`), `min_value`, `max_value`, `step` | Pricing, token counts, metrics |
| **`st.column_config.ProgressColumn`** | `format`, `min_value`, `max_value`, `color` | Benchmark percentages, GPU utilization |
| **`st.column_config.DateColumn`** | `format="YYYY-MM-DD"`, `min_value`, `max_value` | Release dates, birthdates, filter bounds |
| **`st.column_config.DatetimeColumn`** | `format="DD/MM/YYYY, hh:mm a"`, `timezone` | Telemetry event logs, checkpoint timestamps |
| **`st.column_config.ImageColumn`** | `label`, `help`, `width` | Avatars, brand logos, bounding box snippets |
| **`st.column_config.LinkColumn`** | `display_text`, `validate`, `max_chars` | Clickable papers, documentation, dashboard URLs |
| **`st.column_config.CheckboxColumn`** | `default=False`, `help` | Boolean statuses, flags, approval marks |
| **`Column Visibility`** | `column_config={"col": None}` | Hiding sensitive IDs, keys, or internal metadata |
"""
)
