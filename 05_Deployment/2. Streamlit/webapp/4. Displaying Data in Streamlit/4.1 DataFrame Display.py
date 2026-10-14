import streamlit as st
import pandas as pd
import numpy as np
import time

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit DataFrame Display Tutorial",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DataFrame Display (`st.dataframe` & `st.table`) — Complete Guide")
st.markdown(
    """
Displaying structured data is central to data science and machine learning applications. 
Streamlit provides two core primitives for tabular presentation: **`st.dataframe()`** (an interactive, virtualized data grid) 
and **`st.table()`** (a static HTML table). 
This guide covers both primitives, multiple input data structures (DataFrames, Series, NumPy arrays, dictionaries), 
large-scale dataset rendering performance, and precise geometry sizing.
"""
)

st.divider()

# ==============================================================================
# 1. st.dataframe() — Interactive Data Grid
# ==============================================================================
st.header("1. Interactive Data Grid (`st.dataframe`)", divider="rainbow")
st.markdown(
    """
`st.dataframe()` renders data using the high-performance **Glide Data Grid** engine:
- **Interactive features**: Column sorting (click headers), column resizing (drag borders), full-screen expansion, in-grid search, and CSV download.
- **Virtualized Rendering**: Only DOM elements currently in the viewport are rendered, allowing instant scrolling even across millions of data points.
- **Selection Support**: Supports row and column selection callbacks using `on_select="rerun"`.
"""
)

st.markdown("#### 📐 Formal Signature: `st.dataframe()`")
st.code(
    """st.dataframe(
    data: Data = None,
    width: "stretch" | "content" | int | None = "stretch",
    height: int | "auto" | None = "auto",
    *,
    use_container_width: bool | None = None,  # Deprecated in favor of width="stretch"
    hide_index: bool | None = None,
    column_order: Iterable[str] | None = None,
    column_config: dict | None = None,
    key: str | int | None = None,
    on_select: "ignore" | "rerun" | Callable = "ignore",
    selection_mode: "multi-row" | "single-row" | "multi-column" | "single-column" = "multi-row",
    row_height: int | None = None
) -> DeltaGenerator | DataframeState""",
    language="python",
)

# Demo DataFrame
sample_leaderboard = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Llama-3.3-70B", "Gemini-1.5-Pro"],
    "Provider": ["Anthropic", "OpenAI", "DeepSeek", "Meta", "Google"],
    "MMLU (%)": [88.7, 88.6, 88.5, 86.4, 85.9],
    "HumanEval (%)": [92.0, 90.2, 89.1, 81.7, 84.1],
    "Price / 1M Input ($)": [3.00, 2.50, 0.27, 0.70, 1.25]
})

col_df1, col_df2 = st.columns(2)

with col_df1:
    st.markdown("**💻 Python Source Code**")
    code_df = '''import pandas as pd
import streamlit as st

# Render interactive grid with custom width and hidden index:
st.dataframe(
    data=sample_leaderboard,
    width="stretch",
    hide_index=True,
    column_order=["Model", "Provider", "MMLU (%)", "HumanEval (%)", "Price / 1M Input ($)"]
)'''
    st.code(code_df, language="python")

with col_df2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(
        data=sample_leaderboard,
        width="stretch",
        hide_index=True,
        column_order=["Model", "Provider", "MMLU (%)", "HumanEval (%)", "Price / 1M Input ($)"],
        key="demo_df_basic"
    )
    st.caption("💡 *Tip: Hover over headers to sort ascending/descending or click the top-right fullscreen button.*")

st.divider()

# ==============================================================================
# 2. st.table() — Static HTML Table
# ==============================================================================
st.header("2. Static HTML Table (`st.table`)", divider="rainbow")
st.markdown(
    """
`st.table()` renders a classic, static HTML `<table>`:
- **Zero Interactivity**: Cannot be sorted, scrolled, resized, or filtered by users.
- **Full Content Rendering**: Displays all rows at once without scrollbars or virtualization.
- **When to Use**: Ideal for small executive summaries, fixed KPI cards, pricing tiers, or printable reports (typically ≤ 15–20 rows).
"""
)

st.markdown("#### 📐 Formal Signature: `st.table()`")
st.code(
    """st.table(
    data: Data = None,
    *,
    border: bool | "horizontal" = True,
    width: "stretch" | "content" | int = "stretch",
    height: int | "content" = "content",
    hide_index: bool | None = None,
    hide_header: bool | None = None
) -> DeltaGenerator""",
    language="python",
)

summary_stats = pd.DataFrame({
    "Metric": ["Total Evaluated Models", "Average MMLU Score", "Median Inference Latency", "Best Value Provider"],
    "Value": ["48 Models", "82.4%", "142 ms", "DeepSeek ($0.27 / 1M)"]
})

col_tb1, col_tb2 = st.columns(2)

with col_tb1:
    st.markdown("**💻 Python Source Code**")
    code_table = '''# Render static HTML table without scrollbars:
st.table(
    data=summary_stats,
    hide_index=True
)'''
    st.code(code_table, language="python")

with col_tb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.table(
        data=summary_stats,
        hide_index=True
    )
    st.caption("💡 *Notice: The table is completely static with fixed styling and no hover column sorting.*")

st.divider()

# ==============================================================================
# 3. Displaying Pandas DataFrames & Pandas Styler
# ==============================================================================
st.header("3. Displaying Pandas DataFrames with Pandas Styler", divider="rainbow")
st.markdown(
    """
Streamlit natively accepts `pandas.io.formats.style.Styler` objects:
- Apply background gradients with `.style.background_gradient(cmap=...)`.
- Highlight maximum or minimum values with `.style.highlight_max()`.
- Format numeric values (e.g., percentages, currency) with `.style.format(...)`.
"""
)

benchmark_metrics = pd.DataFrame({
    "Precision": [0.942, 0.887, 0.915, 0.783],
    "Recall": [0.912, 0.934, 0.894, 0.821],
    "F1-Score": [0.927, 0.910, 0.904, 0.801],
    "Latency (ms)": [45, 120, 85, 210]
}, index=["LightGBM", "XGBoost", "CatBoost", "RandomForest"])

styled_df = (
    benchmark_metrics.style
    .background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="Greens")
    .highlight_max(subset=["F1-Score"], color="#bbf7d0")
    .highlight_min(subset=["Latency (ms)"], color="#bfdbfe")
    .format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "Latency (ms)": "{} ms"
    })
)

col_styler1, col_styler2 = st.columns(2)

with col_styler1:
    st.markdown("**💻 Python Source Code**")
    code_styler = '''styled = (
    benchmark_metrics.style
    .background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="Greens")
    .highlight_max(subset=["F1-Score"], color="#bbf7d0")
    .highlight_min(subset=["Latency (ms)"], color="#bfdbfe")
    .format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "Latency (ms)": "{} ms"
    })
)

st.dataframe(styled, width="stretch")'''
    st.code(code_styler, language="python")

with col_styler2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(styled_df, width="stretch", key="demo_styled_df")
    st.caption("💡 *Green gradient represents metric intensity; light green highlights max F1-score; light blue marks lowest latency.*")

st.divider()

# ==============================================================================
# 4. Displaying Pandas Series
# ==============================================================================
st.header("4. Displaying Pandas Series", divider="rainbow")
st.markdown(
    """
Pandas `Series` are 1-dimensional labeled arrays. When passed to `st.dataframe()` or `st.table()`, 
Streamlit renders them with the index as the first column and values as the second column:
- Common examples: Model feature importances, value count frequencies, or class distributions.
"""
)

# Feature importance series
feature_importance = pd.Series(
    data=[0.382, 0.245, 0.178, 0.112, 0.083],
    index=["Attention_Dim", "Num_Heads", "Layer_Norm_Eps", "Dropout_Rate", "Vocabulary_Size"],
    name="Gini_Importance"
)

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("**💻 Python Source Code**")
    code_series = '''import pandas as pd

feature_importance = pd.Series(
    data=[0.382, 0.245, 0.178, 0.112, 0.083],
    index=["Attention_Dim", "Num_Heads", "Layer_Norm_Eps", "Dropout_Rate", "Vocabulary_Size"],
    name="Gini_Importance"
)

# Render Series directly:
st.dataframe(feature_importance, width="stretch")'''
    st.code(code_series, language="python")

with col_s2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(feature_importance, width="stretch", key="demo_series_view")
    st.caption(f"Series Name: `{feature_importance.name}` | Total Features: **{len(feature_importance)}**")

st.divider()

# ==============================================================================
# 5. Displaying NumPy Arrays
# ==============================================================================
st.header("5. Displaying NumPy Arrays", divider="rainbow")
st.markdown(
    """
Streamlit seamlessly converts 1D and 2D NumPy arrays into tabular structures without requiring manual DataFrame conversion:
- **1D Arrays**: Rendered as a single-column table.
- **2D Arrays**: Rendered as matrix grids with integer column headers (`0, 1, 2, ...`).
- **Use Cases**: Correlation matrices, embeddings projections, confusion matrices, and weight tensors.
"""
)

# 2D Confusion Matrix
np.random.seed(42)
confusion_matrix_arr = np.array([
    [142, 8, 2, 0],
    [5, 138, 11, 3],
    [1, 9, 150, 4],
    [0, 2, 6, 148]
])

col_np1, col_np2 = st.columns(2)

with col_np1:
    st.markdown("**💻 Python Source Code**")
    code_numpy = '''import numpy as np

# 4x4 Multi-class Confusion Matrix Array:
confusion_matrix = np.array([
    [142, 8, 2, 0],
    [5, 138, 11, 3],
    [1, 9, 150, 4],
    [0, 2, 6, 148]
])

# Streamlit renders 2D NumPy array automatically:
st.dataframe(confusion_matrix, width="stretch")'''
    st.code(code_numpy, language="python")

with col_np2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(confusion_matrix_arr, width="stretch", key="demo_numpy_view")
    st.caption(f"NumPy Shape: `{confusion_matrix_arr.shape}` | Data Type: `{confusion_matrix_arr.dtype}`")

st.divider()

# ==============================================================================
# 6. Displaying Dictionaries
# ==============================================================================
st.header("6. Displaying Dictionaries (Key-Value & Records)", divider="rainbow")
st.markdown(
    """
Dictionaries can be presented in two distinct formats:
1. **Columnar / Record Dictionary**: A dictionary of lists or list of dictionaries is automatically parsed as a tabular DataFrame.
2. **Scalar Key-Value Dictionary**: Single-level key-value mappings render as key-value property tables.
"""
)

# Dictionary representations
dict_records = {
    "Hyperparameter": ["learning_rate", "batch_size", "optimizer", "weight_decay", "scheduler"],
    "Configured Value": ["0.0003", "64", "AdamW", "0.01", "CosineAnnealingLR"],
    "Tunable Range": ["[1e-5, 1e-2]", "[16, 256]", "AdamW / SGD", "[0, 0.1]", "Step / Cosine"]
}

col_dc1, col_dc2 = st.columns(2)

with col_dc1:
    st.markdown("**💻 Python Source Code**")
    code_dict = '''# 1. Columnar Dictionary of lists:
hyperparams_dict = {
    "Hyperparameter": ["learning_rate", "batch_size", "optimizer", "weight_decay", "scheduler"],
    "Configured Value": ["0.0003", "64", "AdamW", "0.01", "CosineAnnealingLR"],
    "Tunable Range": ["[1e-5, 1e-2]", "[16, 256]", "AdamW / SGD", "[0, 0.1]", "Step / Cosine"]
}
st.dataframe(hyperparams_dict, width="stretch", hide_index=True)

# 2. Key-Value Scalar Dictionary:
system_status = {
    "PyTorch Version": "2.4.1+cu121",
    "CUDA Available": "True (NVIDIA RTX 4090)",
    "Available VRAM": "24,576 MB"
}
st.dataframe(system_status, width="stretch")'''
    st.code(code_dict, language="python")

with col_dc2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown("##### 1. Columnar Dictionary Table:")
    st.dataframe(dict_records, width="stretch", hide_index=True, key="demo_dict_columnar")
    
    st.markdown("##### 2. Key-Value Property Dictionary:")
    system_status_demo = {
        "PyTorch Version": "2.4.1+cu121",
        "CUDA Available": "True (NVIDIA RTX 4090)",
        "Available VRAM": "24,576 MB"
    }
    st.dataframe(system_status_demo, width="stretch", key="demo_dict_kv")
    st.caption("💡 *Streamlit automatically recognizes both dictionary of lists and key-value mapping formats.*")

st.divider()

# ==============================================================================
# 7. Displaying Large Datasets (Virtual Scrolling Performance)
# ==============================================================================
st.header("7. High-Performance Display of Large Datasets", divider="rainbow")
st.markdown(
    """
`st.dataframe()` relies on an **Apache Arrow** serialization bridge and client-side canvas virtualization:
- Handles **100,000+ rows** smoothly without freezing the browser DOM.
- Only renders visible cells on screen (O(1) memory overhead on client viewport).
- In contrast, calling `st.table()` on 50,000 rows will attempt to render 50,000 HTML `<tr>` tags and crash the browser tab!
"""
)

# Generate large synthetic telemetry dataset
row_count = 50_000
large_df_demo = pd.DataFrame({
    "Transaction_ID": [f"TX-{100000 + i}" for i in range(row_count)],
    "Device_ID": np.random.randint(100, 999, size=row_count),
    "Anomaly_Score": np.round(np.random.beta(2, 5, size=row_count), 4),
    "Latency_ms": np.round(np.random.exponential(scale=35, size=row_count), 2),
    "Flagged": np.random.choice([False, True], size=row_count, p=[0.97, 0.03])
})

col_lg1, col_lg2 = st.columns(2)

with col_lg1:
    st.markdown("**💻 Python Source Code**")
    code_large = f'''import pandas as pd
import numpy as np

# Synthetic telemetry dataset with {row_count:,} rows:
large_df = pd.DataFrame({{
    "Transaction_ID": [f"TX-{{100000 + i}}" for i in range({row_count})],
    "Device_ID": np.random.randint(100, 999, size={row_count}),
    "Anomaly_Score": np.round(np.random.beta(2, 5, size={row_count}), 4),
    "Latency_ms": np.round(np.random.exponential(scale=35, size={row_count}), 2),
    "Flagged": np.random.choice([False, True], size={row_count}, p=[0.97, 0.03])
}})

# Virtualized rendering via Apache Arrow & Glide Data Grid:
st.dataframe(large_df, height=300, width="stretch")'''
    st.code(code_large, language="python")

with col_lg2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.dataframe(large_df_demo, height=300, width="stretch", key="demo_large_df")
    st.caption(f"⚡ *Dataset Size: **{row_count:,} rows** × **{large_df_demo.shape[1]} columns** rendered in milliseconds via virtual scrolling.*")

st.divider()

# ==============================================================================
# 8. Configuring Table Height, Width & Geometry
# ==============================================================================
st.header("8. Configuring Table Geometry (Height & Width)", divider="rainbow")
st.markdown(
    """
Fine-tune table dimensions using `height` and `width`:
- **`width="stretch"`**: Fills the entire horizontal container width (standard modern replacement for deprecated `use_container_width=True`).
- **`width="content"`**: Shrinks the table horizontally to fit its exact column widths.
- **Fixed Pixel Width**: Pass an integer like `width=500` to constrain dimensions.
- **`height="auto"`**: Automatically sizes vertical height to fit all rows without inner scrolling.
- **Fixed Pixel Height**: Pass an integer like `height=220` to cap vertical footprint and enforce vertical scrolling.
"""
)

col_geom1, col_geom2 = st.columns(2)

with col_geom1:
    st.markdown("##### 📏 Fixed Height (`height=200`) with Scrollbar")
    code_fixed_h = '''# Constrain table height to 200px:
st.dataframe(
    data=sample_leaderboard,
    height=200,
    width="stretch",
    hide_index=True
)'''
    st.code(code_fixed_h, language="python")
    st.dataframe(
        data=sample_leaderboard,
        height=200,
        width="stretch",
        hide_index=True,
        key="demo_fixed_h"
    )

with col_geom2:
    st.markdown("##### 📐 Content Width (`width='content'`)")
    code_content_w = '''# Tight content fit without stretching:
st.dataframe(
    data=sample_leaderboard[["Model", "MMLU (%)"]],
    width="content",
    hide_index=True
)'''
    st.code(code_content_w, language="python")
    st.dataframe(
        data=sample_leaderboard[["Model", "MMLU (%)"]],
        width="content",
        hide_index=True,
        key="demo_content_w"
    )

st.divider()

# ==============================================================================
# Summary Comparison Reference
# ==============================================================================
st.header("📋 Comparison: `st.dataframe` vs. `st.table`", divider="rainbow")
st.markdown(
    """
| Feature / Characteristic | `st.dataframe()` | `st.table()` |
| :--- | :--- | :--- |
| **Engine** | Glide Data Grid (Canvas & DOM virtualization) | Native HTML `<table>` tag |
| **Interactivity** | ✅ Sorting, resizing, search, CSV download, selection | ❌ None (completely static) |
| **Scalability** | 🚀 Millions of rows via virtual scrolling | ⚠️ Small datasets only (max ~20-50 rows) |
| **Pandas Styler Support** | ✅ Yes (`.style.background_gradient()`, etc.) | ✅ Yes |
| **Column Configuration** | ✅ Yes (`st.column_config` support) | ❌ No |
| **Scrollbars** | ✅ Configurable horizontal and vertical scrollbars | ❌ No scrollbars (renders all rows) |
| **Best Used For** | Analytical dashboards, data exploration, leaderboards | Executive KPI summaries, static invoices, printable reports |
"""
)
