import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Interactive Data Tables Tutorial",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Interactive Data Tables (`st.data_editor`) — Complete Guide")
st.markdown(
    """
Streamlit's **`st.data_editor()`** widget transforms passive tabular displays into fully interactive, 
spreadsheet-like editable interfaces. Users can edit cell contents, insert and delete rows, 
pick values from in-grid dropdowns, and trigger reactive recalculations upon every modification. 
This master guide covers API signatures, row lifecycle controls, column lockdown, state diffs, validation, 
and persistence.
"""
)

st.divider()

# ==============================================================================
# 1. st.data_editor() Core Architecture
# ==============================================================================
st.header("1. Core Architecture & `st.data_editor` API", divider="rainbow")
st.markdown(
    """
Unlike `st.dataframe()` which is strictly read-only, `st.data_editor()` enables bi-directional data flow:
1. **Direct Return Value**: Returns the complete, modified DataFrame reflecting all current user edits.
2. **Session State State Diffs**: If assigned a `key`, `st.session_state[key]` stores a granular delta dictionary containing:
   - `edited_rows`: Map of changed row indices and their updated cell values.
   - `added_rows`: List of newly inserted rows.
   - `deleted_rows`: List of removed row indices.
"""
)

st.markdown("#### 📐 Formal Signature: `st.data_editor()`")
st.code(
    """st.data_editor(
    data: Data,
    *,
    width: "stretch" | "content" | int | None = "stretch",
    height: int | "auto" | None = "auto",
    use_container_width: bool | None = None,  # Deprecated in favor of width="stretch"
    hide_index: bool | None = None,
    column_order: Iterable[str] | None = None,
    column_config: dict | None = None,
    num_rows: "fixed" | "dynamic" | "add" | "delete" = "fixed",
    disabled: bool | Iterable[str] = False,
    key: str | int | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    row_height: int | None = None
) -> pd.DataFrame""",
    language="python",
)

st.divider()

# ==============================================================================
# 2. Basic In-Cell Editing of DataFrames
# ==============================================================================
st.header("2. Basic In-Cell Editing of DataFrames", divider="rainbow")
st.markdown(
    """
Double-click any cell (or press Enter) to edit text, numbers, or dates:
- Boolean columns render as clickable checkboxes.
- Number columns validate numeric data entry automatically.
- Edited cells display a subtle highlight indicator.
"""
)

init_models_df = pd.DataFrame({
    "Model Name": ["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B"],
    "Max Tokens": [8192, 4096, 8192],
    "Temperature": [0.70, 0.50, 0.80],
    "Active": [True, True, False]
})

col_ed1, col_ed2 = st.columns(2)

with col_ed1:
    st.markdown("**💻 Python Source Code**")
    code_basic_edit = '''# Basic editable DataFrame:
edited_df = st.data_editor(
    data=init_models_df,
    width="stretch",
    hide_index=True,
    key="basic_editor_demo"
)

# Display real-time updated summary:
st.write(f"Active Models Count: {edited_df['Active'].sum()} / {len(edited_df)}")'''
    st.code(code_basic_edit, language="python")

with col_ed2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    edited_basic = st.data_editor(
        data=init_models_df,
        width="stretch",
        hide_index=True,
        key="basic_editor_demo"
    )
    active_count = int(edited_basic["Active"].sum())
    st.caption(f"📊 Live Summary: **{active_count}** active models configured.")

st.divider()

# ==============================================================================
# 3. Adding and Deleting Rows (`num_rows="dynamic"`)
# ==============================================================================
st.header("3. Dynamic Row Insertion & Deletion (`num_rows`)", divider="rainbow")
st.markdown(
    """
By default, table row count is fixed (`num_rows="fixed"`). Setting `num_rows="dynamic"` enables spreadsheet-like row operations:
- **Add Row**: Users click the empty row at the bottom with the `+` icon or press `Ctrl + Enter`.
- **Delete Row**: Users select row checkboxes on the far left and press `Delete` on their keyboard (or click the trash icon).
- **Options**: `"fixed"` (default), `"dynamic"` (add & delete), `"add"` (add only), `"delete"` (delete only).
"""
)

prompt_templates_df = pd.DataFrame({
    "Role": ["System", "User", "Assistant"],
    "Template Key": ["sys_persona", "user_query", "assistant_prefill"],
    "Sample Text": ["You are an AI research assistant.", "Analyze this dataset.", "Based on the findings:"]
})

col_dyn1, col_dyn2 = st.columns(2)

with col_dyn1:
    st.markdown("**💻 Python Source Code**")
    code_dynamic = '''# Allow users to add and delete prompt templates:
updated_prompts = st.data_editor(
    data=prompt_templates_df,
    num_rows="dynamic",  # Enables row addition & deletion!
    width="stretch",
    hide_index=True,
    key="dynamic_prompts_editor"
)

st.write(f"Total Configured Prompts: {len(updated_prompts)}")'''
    st.code(code_dynamic, language="python")

with col_dyn2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    updated_prompts_demo = st.data_editor(
        data=prompt_templates_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="dynamic_prompts_editor"
    )
    st.caption(f"📝 Total Rows in Session: **{len(updated_prompts_demo)}** rows (try adding a new row via `+` at bottom).")

st.divider()

# ==============================================================================
# 4. Configuring Editable Columns & Locking Columns
# ==============================================================================
st.header("4. Configuring Editable Columns & Read-Only Locks", divider="rainbow")
st.markdown(
    """
Control which columns are editable and constrain input options:
- **`disabled=["col_a", "col_b"]`**: Locks specific columns from being edited by users.
- **`st.column_config.SelectboxColumn(options=[...])`**: Restricts cell input to a predefined dropdown list.
- **`st.column_config.NumberColumn(min_value=..., max_value=...)`**: Enforces numerical boundaries on user edits.
"""
)

experiment_grid_df = pd.DataFrame({
    "Exp_ID": ["EXP-001", "EXP-002", "EXP-003", "EXP-004"],
    "Architecture": ["Transformer", "Mamba", "Transformer", "MoE"],
    "Learning_Rate": [0.0003, 0.0001, 0.0005, 0.0002],
    "Optimizer": ["AdamW", "AdamW", "Lion", "AdamW"]
})

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    st.markdown("**💻 Python Source Code**")
    code_config_editable = '''# Lock Exp_ID, provide dropdown for Optimizer:
edited_experiments = st.data_editor(
    data=experiment_grid_df,
    disabled=["Exp_ID"],  # Exp_ID is read-only!
    column_config={
        "Exp_ID": st.column_config.TextColumn("Experiment ID", help="Primary key (Immutable)"),
        "Architecture": st.column_config.SelectboxColumn(
            "Model Architecture",
            options=["Transformer", "Mamba", "RWKV", "MoE"],
            required=True
        ),
        "Learning_Rate": st.column_config.NumberColumn(
            "Learning Rate",
            min_value=1e-5,
            max_value=1e-1,
            step=1e-4,
            format="%.5f"
        ),
        "Optimizer": st.column_config.SelectboxColumn(
            "Optimizer",
            options=["AdamW", "Lion", "SGD", "Sophia"],
            required=True
        )
    },
    width="stretch",
    hide_index=True,
    key="configured_experiments_editor"
)'''
    st.code(code_config_editable, language="python")

with col_cfg2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    edited_exp_demo = st.data_editor(
        data=experiment_grid_df,
        disabled=["Exp_ID"],
        column_config={
            "Exp_ID": st.column_config.TextColumn("Experiment ID", help="Primary key (Immutable)"),
            "Architecture": st.column_config.SelectboxColumn(
                "Model Architecture",
                options=["Transformer", "Mamba", "RWKV", "MoE"],
                required=True
            ),
            "Learning_Rate": st.column_config.NumberColumn(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-1,
                step=1e-4,
                format="%.5f"
            ),
            "Optimizer": st.column_config.SelectboxColumn(
                "Optimizer",
                options=["AdamW", "Lion", "SGD", "Sophia"],
                required=True
            )
        },
        width="stretch",
        hide_index=True,
        key="configured_experiments_editor"
    )
    st.caption("🔒 *`Exp_ID` is locked. Click `Architecture` or `Optimizer` cells to choose from dropdowns.*")

st.divider()

# ==============================================================================
# 5. Handling Edited Data & State Diffs (`st.session_state`)
# ==============================================================================
st.header("5. Inspecting Changes via `st.session_state` Diffs", divider="rainbow")
st.markdown(
    """
When you specify a `key`, Streamlit tracks exact cell mutations in `st.session_state[key]`:
- **`edited_rows`**: `{row_index: {column_name: new_value}}`
- **`added_rows`**: `[{column_name: value, ...}]`
- **`deleted_rows`**: `[row_index_1, ...]`
This enables fine-grained change detection (e.g. running an `UPDATE` SQL statement only for modified rows!).
"""
)

diff_test_df = pd.DataFrame({
    "User": ["alice@ai.org", "bob@ai.org", "charlie@ai.org"],
    "Role": ["Researcher", "Engineer", "Analyst"],
    "Quota_GB": [50, 100, 25]
})

col_diff1, col_diff2 = st.columns(2)

with col_diff1:
    st.markdown("**💻 Python Source Code (Accessing State Diffs)**")
    code_diffs = '''# Data editor with explicit session key:
st.data_editor(
    data=diff_test_df,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key="quota_editor"
)

# Access the delta changes dictionary:
editor_state = st.session_state.get("quota_editor", {})
st.json({
    "edited_cells": editor_state.get("edited_rows", {}),
    "added_records": editor_state.get("added_rows", []),
    "deleted_indices": editor_state.get("deleted_rows", [])
})'''
    st.code(code_diffs, language="python")

with col_diff2:
    st.markdown("**🖥️ Live Interactive State Diff Inspector**")
    st.data_editor(
        data=diff_test_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="quota_editor"
    )
    
    diff_state = st.session_state.get("quota_editor", {})
    st.markdown("##### 🔍 Real-Time Session State Delta:")
    st.json({
        "edited_rows": diff_state.get("edited_rows", {}),
        "added_rows": diff_state.get("added_rows", []),
        "deleted_rows": diff_state.get("deleted_rows", [])
    })

st.divider()

# ==============================================================================
# 6. Validating User Changes & 7. Saving Edited Data
# ==============================================================================
st.header("6. Validating User Changes & 7. Persistence & Downloads", divider="rainbow")
st.markdown(
    """
1. **Validation**: Test custom domain invariants (e.g. weights sum to 100%, budgets do not exceed quota limits).
2. **Persistence**: Store the validated DataFrame in `st.session_state` so edits survive page navigation.
3. **Export**: Provide instant CSV download capabilities using `st.download_button()`.
"""
)

# Portfolio weights DataFrame
if "portfolio_data" not in st.session_state:
    st.session_state["portfolio_data"] = pd.DataFrame({
        "Model Tier": ["Tier 1 (Frontier LLM)", "Tier 2 (Mid-size Task Model)", "Tier 3 (Edge SLM)"],
        "Allocation (%)": [50, 30, 20],
        "Hourly Cost Limit ($)": [25.0, 10.0, 5.0]
    })

col_val1, col_val2 = st.columns(2)

with col_val1:
    st.markdown("**💻 Python Source Code (Validation & Export)**")
    code_val_save = '''# 1. Edit allocation weights:
edited_portfolio = st.data_editor(
    data=st.session_state["portfolio_data"],
    width="stretch",
    hide_index=True,
    key="portfolio_editor"
)

# 2. Domain Validation: Sum must equal 100%
total_weight = edited_portfolio["Allocation (%)"].sum()

if total_weight == 100:
    st.success(f"Validation Passed! Total Allocation: {total_weight}%")
    # Save back to session state:
    st.session_state["portfolio_data"] = edited_portfolio
    
    # 3. CSV Export:
    csv_bytes = edited_portfolio.to_csv(index=False).encode("utf-8")
    st.download_button("Download Config (CSV)", csv_bytes, "model_budget.csv", "text/csv")
else:
    st.error(f"Validation Error: Total allocation is {total_weight}% (Must sum to exactly 100%)")'''
    st.code(code_val_save, language="python")

with col_val2:
    st.markdown("**🖥️ Live Validated Editor & Exporter**")
    edited_port_demo = st.data_editor(
        data=st.session_state["portfolio_data"],
        width="stretch",
        hide_index=True,
        key="portfolio_editor"
    )

    total_weight_val = edited_port_demo["Allocation (%)"].sum()

    if total_weight_val == 100:
        st.success(f"✅ Allocation Validated: **{total_weight_val}%**")
        st.session_state["portfolio_data"] = edited_port_demo
        csv_data = edited_port_demo.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Verified Config (CSV)",
            data=csv_data,
            file_name="model_budget_allocation.csv",
            mime="text/csv",
            key="btn_download_portfolio"
        )
    else:
        st.error(f"❌ Invariant Violation: Total allocation equals **{total_weight_val}%** (Requires exactly **100%**). Adjust weights above!")

st.divider()

# ==============================================================================
# 8. Building Spreadsheet-Like Applications (AI Experiment Planner)
# ==============================================================================
st.header("8. Real-World Project: AI Experiment & Budget Planner", divider="rainbow")
st.markdown(
    """
Below is a full spreadsheet-like application combining in-cell formulas, dynamic rows, locked columns, 
and real-time aggregate KPI scorecards:
"""
)

if "planner_records" not in st.session_state:
    st.session_state["planner_records"] = pd.DataFrame({
        "Model": ["Claude-3.5-Sonnet", "GPT-4o-Mini", "DeepSeek-V3"],
        "Daily Calls": [15000, 80000, 25000],
        "Input Tokens / Call": [800, 500, 1200],
        "Cost / 1M Input ($)": [3.00, 0.15, 0.27],
        "Approved": [True, True, True]
    })

col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("##### ⚙️ Interactive Model Run Sheet")
    edited_planner = st.data_editor(
        data=st.session_state["planner_records"],
        num_rows="dynamic",
        column_config={
            "Model": st.column_config.TextColumn("Model Name", required=True),
            "Daily Calls": st.column_config.NumberColumn("Daily Calls", min_value=1, format="%,d"),
            "Input Tokens / Call": st.column_config.NumberColumn("Avg Tokens", min_value=1, format="%,d"),
            "Cost / 1M Input ($)": st.column_config.NumberColumn("Price / 1M", format="$%.2f", min_value=0.0),
            "Approved": st.column_config.CheckboxColumn("Approved")
        },
        width="stretch",
        hide_index=True,
        key="app_planner_editor"
    )

with col_app2:
    st.markdown("##### 📈 Real-Time Reactive Financial Analytics")
    
    # Calculate derived column
    total_tokens = (edited_planner["Daily Calls"] * edited_planner["Input Tokens / Call"]).sum()
    daily_spend = (
        (edited_planner["Daily Calls"] * edited_planner["Input Tokens / Call"] / 1_000_000) * 
        edited_planner["Cost / 1M Input ($)"]
    ).sum()
    monthly_est = daily_spend * 30
    
    metric_c1, metric_c2 = st.columns(2)
    with metric_c1:
        st.metric("Total Ingested Tokens / Day", f"{total_tokens / 1e6:.2f} M Tokens")
        st.metric("Estimated Daily Spend", f"${daily_spend:.2f}")
    with metric_c2:
        st.metric("Active Models In Fleet", f"{len(edited_planner)}")
        st.metric("Projected Monthly Spend", f"${monthly_est:.2f}")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 `st.data_editor` Feature Checklist", divider="rainbow")
st.markdown(
    """
| Requirement | Syntax / Parameter | Explanation |
| :--- | :--- | :--- |
| **Enable Row Addition & Deletion** | `num_rows="dynamic"` | Displays `+` bottom row and enables checkbox row deletions |
| **Lock Specific Columns** | `disabled=["col1", "col2"]` | Prevents cell editing on key/id columns |
| **In-Cell Dropdowns** | `st.column_config.SelectboxColumn(options=[...])` | Enforces categorical selection in table cells |
| **Range Validation** | `st.column_config.NumberColumn(min_value=..., max_value=...)` | Restricts numerical bounds during in-cell typing |
| **Access Delta Diffs** | `st.session_state[key]["edited_rows"]` | Direct access to changed cells, added rows, and deleted rows |
| **Data Persistence** | `st.session_state["df"] = edited_df` | Stores mutations across navigation and reruns |
| **Data Export** | `st.download_button(data=edited_df.to_csv())` | Instant one-click CSV export of user-modified data |
"""
)
