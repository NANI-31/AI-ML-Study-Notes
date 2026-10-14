import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime

st.set_page_config(
    page_title="14.2 Pandas Integration",
    page_icon="🐼",
    layout="wide"
)

st.title("14.2 Pandas Integration in Streamlit")
st.caption("Industrial data engineering: DataFrame rendering with Arrow, CSV/Excel ingestion, dynamic filtering, multi-level aggregations, missingness treatment, relational joins, and exports.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Pandas in the Streamlit Ecosystem

Pandas `DataFrame` is the core tabular data structure in Python. 
Streamlit integrates natively with Pandas through **Apache Arrow**, serializing DataFrames into a binary columnar format for zero-copy transmission to the browser.

""")

st.graphviz_chart(
    r'''
    digraph PandasWorkflow {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        df [label="1. Pandas DataFrame\nTabular metrics & time series", fillcolor="#DBEAFE", color="#1D4ED8"];
        config [label="2. Column Configuration\nNumberColumn, ProgressColumn, LineChartColumn", fillcolor="#FEF3C7", color="#D97706"];
        grid [label="3. Interactive Grid\nst.dataframe(df, column_config=..., width='stretch')", fillcolor="#D1FAE5", color="#059669"];

        df -> config [label="Configures types"];
        config -> grid [label="Renders styled table"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Displaying DataFrames",
    "2. Reading CSV Files",
    "3. Reading Excel Files",
    "4. Dynamic Filtering",
    "5. Sorting Records",
    "6. Grouping & Aggregations",
    "7. Handling Missing Values",
    "8. Merging Datasets",
    "9. Exporting Data"
])

# =============================================================================
# 1. DISPLAYING DATAFRAMES
# =============================================================================
with tabs[0]:
    st.header("1. Displaying DataFrames (`st.dataframe`, `st.table`, `st.data_editor`)")
    st.markdown(r"""
    Streamlit provides three distinct ways to render tabular data:
    - `st.dataframe`: Interactive, scrollable table with search, sort, and `st.column_config`.
    - `st.table`: Static HTML table (renders all rows directly into the DOM; avoid for $>100$ rows).
    - `st.data_editor`: Interactive, editable grid allowing in-place edits and row additions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Model benchmark dataset
df_models = pd.DataFrame({
    "Model Name": ["Llama-3-70B", "Mistral-Large", "Claude-3.5-Sonnet", "GPT-4o"],
    "MMLU Score": [0.820, 0.812, 0.887, 0.885],
    "Context (k)": [128, 32, 200, 128],
    "Latency (ms)": [340, 290, 410, 260],
    "Open Weights": [True, False, False, False]
})

st.dataframe(
    df_models,
    column_config={
        "MMLU Score": st.column_config.ProgressColumn(
            "MMLU Accuracy", min_value=0.5, max_value=1.0, format="%.3f"
        ),
        "Latency (ms)": st.column_config.NumberColumn(
            "Latency (ms)", format="%d ms"
        ),
        "Open Weights": st.column_config.CheckboxColumn("Open Source?")
    },
    width="stretch",
    hide_index=True
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_models = pd.DataFrame({
                "Model Name": ["Llama-3-70B", "Mistral-Large", "Claude-3.5-Sonnet", "GPT-4o"],
                "MMLU Score": [0.820, 0.812, 0.887, 0.885],
                "Context (k)": [128, 32, 200, 128],
                "Latency (ms)": [340, 290, 410, 260],
                "Open Weights": [True, False, False, False]
            })

            st.dataframe(
                df_models,
                column_config={
                    "MMLU Score": st.column_config.ProgressColumn(
                        "MMLU Accuracy", min_value=0.5, max_value=1.0, format="%.3f"
                    ),
                    "Latency (ms)": st.column_config.NumberColumn(
                        "Latency (ms)", format="%d ms"
                    ),
                    "Open Weights": st.column_config.CheckboxColumn("Open Source?")
                },
                width="stretch",
                hide_index=True
            )

# =============================================================================
# 2. READING CSV FILES
# =============================================================================
with tabs[1]:
    st.header("2. Reading CSV Files (`pd.read_csv` & Buffers)")
    st.markdown(r"""
    Ingesting CSV data in Streamlit handles both local filesystem paths and uploaded files (`UploadedFile` buffer).
    Always specify explicit `encoding`, `dtype`, and `parse_dates` to optimize memory.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import io

# Simulated uploaded CSV buffer
sample_csv = """CustomerID,Region,SpendUSD,SignupDate
C-101,North America,450.50,2024-01-15
C-102,Europe,1200.00,2024-02-20
C-103,Asia-Pacific,890.25,2024-03-05
C-104,North America,340.00,2024-03-12
C-105,Europe,2150.75,2024-04-01
"""

# Parsing from in-memory stream or st.file_uploader
csv_buffer = io.StringIO(sample_csv)
df_csv = pd.read_csv(
    csv_buffer,
    parse_dates=["SignupDate"],
    dtype={"CustomerID": "string", "Region": "category", "SpendUSD": "float32"}
)

st.write(f"Loaded **{len(df_csv)}** records. Memory Footprint: **{df_csv.memory_usage().sum()}** bytes")
st.dataframe(df_csv, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sample_csv = """CustomerID,Region,SpendUSD,SignupDate
C-101,North America,450.50,2024-01-15
C-102,Europe,1200.00,2024-02-20
C-103,Asia-Pacific,890.25,2024-03-05
C-104,North America,340.00,2024-03-12
C-105,Europe,2150.75,2024-04-01
"""
            csv_buffer = io.StringIO(sample_csv)
            df_csv = pd.read_csv(
                csv_buffer,
                parse_dates=["SignupDate"],
                dtype={"CustomerID": "string", "Region": "category", "SpendUSD": "float32"}
            )
            st.write(f"Loaded **{len(df_csv)}** records. Memory Footprint: **{df_csv.memory_usage().sum()}** bytes")
            st.dataframe(df_csv, width="stretch", hide_index=True)

# =============================================================================
# 3. READING EXCEL FILES
# =============================================================================
with tabs[2]:
    st.header("3. Reading Excel Files (`pd.read_excel` & Sheets)")
    st.markdown(r"""
    Multi-sheet Excel workbooks (`.xlsx`) require parsing sheet names with `pd.ExcelFile` 
    and allowing interactive sheet selection via `st.selectbox`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import io

# Creating a simulated in-memory 2-sheet Excel file
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    pd.DataFrame({"Q1_Revenue": [15000, 22000], "Department": ["AI", "Cloud"]}).to_excel(writer, sheet_name="Q1_Financials", index=False)
    pd.DataFrame({"Headcount": [45, 80], "Department": ["AI", "Cloud"]}).to_excel(writer, sheet_name="Staffing", index=False)
excel_bytes = output.getvalue()

# Reading multi-sheet workbook
excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
sheet_selected = st.selectbox("Select Sheet to Inspect", excel_file.sheet_names)

df_sheet = pd.read_excel(excel_file, sheet_name=sheet_selected)
st.dataframe(df_sheet, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                pd.DataFrame({"Q1_Revenue": [15000, 22000], "Department": ["AI", "Cloud"]}).to_excel(writer, sheet_name="Q1_Financials", index=False)
                pd.DataFrame({"Headcount": [45, 80], "Department": ["AI", "Cloud"]}).to_excel(writer, sheet_name="Staffing", index=False)
            excel_bytes = output.getvalue()

            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            sheet_selected = st.selectbox("Select Sheet to Inspect", excel_file.sheet_names, key="excel_sheet_sel")

            df_sheet = pd.read_excel(excel_file, sheet_name=sheet_selected)
            st.dataframe(df_sheet, width="stretch", hide_index=True)

# =============================================================================
# 4. DYNAMIC FILTERING
# =============================================================================
with tabs[3]:
    st.header("4. Dynamic Record Filtering with Boolean Masks")
    st.markdown(r"""
    Filter datasets interactively by binding user widget values to boolean mask predicates.
    Combine multiple predicates using bitwise operators (`&`, `|`, `~`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Base employee roster
df_staff = pd.DataFrame({
    "Employee": ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona"],
    "Department": ["Research", "Engineering", "Engineering", "Research", "Sales", "Engineering"],
    "Experience_Years": [7, 3, 12, 5, 2, 8],
    "Rating": [4.8, 4.2, 4.9, 4.5, 3.8, 4.7]
})

c_dept, c_exp = st.columns(2)
with c_dept:
    selected_depts = st.multiselect(
        "Filter Departments", 
        df_staff["Department"].unique(), 
        default=df_staff["Department"].unique()
    )
with c_exp:
    min_exp = st.slider("Minimum Experience (Years)", 0, 15, 3)

# Compound Boolean Mask
mask = (df_staff["Department"].isin(selected_depts)) & (df_staff["Experience_Years"] >= min_exp)
filtered_df = df_staff[mask]

st.caption(f"Showing {len(filtered_df)} of {len(df_staff)} employees")
st.dataframe(filtered_df, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_staff = pd.DataFrame({
                "Employee": ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona"],
                "Department": ["Research", "Engineering", "Engineering", "Research", "Sales", "Engineering"],
                "Experience_Years": [7, 3, 12, 5, 2, 8],
                "Rating": [4.8, 4.2, 4.9, 4.5, 3.8, 4.7]
            })

            c_dept, c_exp = st.columns(2)
            with c_dept:
                sel_depts = st.multiselect(
                    "Filter Departments", 
                    df_staff["Department"].unique(), 
                    default=df_staff["Department"].unique(),
                    key="filter_dept_sel"
                )
            with c_exp:
                min_exp_val = st.slider("Minimum Experience (Years)", 0, 15, 3, key="filter_exp_slider")

            mask = (df_staff["Department"].isin(sel_depts)) & (df_staff["Experience_Years"] >= min_exp_val)
            filtered_df = df_staff[mask]

            st.caption(f"Showing {len(filtered_df)} of {len(df_staff)} employees")
            st.dataframe(filtered_df, width="stretch", hide_index=True)

# =============================================================================
# 5. SORTING RECORDS
# =============================================================================
with tabs[4]:
    st.header("5. Multi-Column Sorting (`sort_values`)")
    st.markdown(r"""
    Sorting tabular data across multiple keys with ascending/descending toggles.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

df_sales = pd.DataFrame({
    "Product": ["Server A", "Server B", "GPU Workstation", "Edge Device", "Rack C"],
    "Units_Sold": [45, 120, 30, 210, 15],
    "Revenue_k": [450, 600, 750, 315, 225]
})

col_sort_by, col_order = st.columns(2)
with col_sort_by:
    sort_column = st.selectbox("Sort By", ["Revenue_k", "Units_Sold", "Product"])
with col_order:
    sort_ascending = st.toggle("Ascending Order", value=False)

sorted_df = df_sales.sort_values(by=sort_column, ascending=sort_ascending).reset_index(drop=True)
st.dataframe(sorted_df, width="stretch")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_sales = pd.DataFrame({
                "Product": ["Server A", "Server B", "GPU Workstation", "Edge Device", "Rack C"],
                "Units_Sold": [45, 120, 30, 210, 15],
                "Revenue_k": [450, 600, 750, 315, 225]
            })

            cs_by, cs_ord = st.columns(2)
            with cs_by:
                sort_col = st.selectbox("Sort By", ["Revenue_k", "Units_Sold", "Product"], key="sort_col_sel")
            with cs_ord:
                sort_asc = st.toggle("Ascending Order", value=False, key="sort_asc_tog")

            sorted_df = df_sales.sort_values(by=sort_col, ascending=sort_asc).reset_index(drop=True)
            st.dataframe(sorted_df, width="stretch")

# =============================================================================
# 6. GROUPING & AGGREGATIONS
# =============================================================================
with tabs[5]:
    st.header("6. Grouping & Named Aggregations (`groupby` & `pivot_table`)")
    st.markdown(r"""
    Summarizing subsets of data using SQL-like aggregation operations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

df_transactions = pd.DataFrame({
    "Region": ["Americas", "EMEA", "APAC", "Americas", "EMEA", "APAC", "Americas"],
    "Tier": ["Enterprise", "SMB", "Enterprise", "SMB", "Enterprise", "SMB", "Enterprise"],
    "Contract_Value": [120000, 35000, 95000, 28000, 140000, 42000, 180000]
})

# Named Aggregation
summary = df_transactions.groupby("Region").agg(
    Total_Pipeline=("Contract_Value", "sum"),
    Average_Deal=("Contract_Value", "mean"),
    Deal_Count=("Contract_Value", "count")
).reset_index()

st.caption("Grouped Pipeline Summary:")
st.dataframe(summary.style.format({"Total_Pipeline": "${:,.0f}", "Average_Deal": "${:,.0f}"}), width="stretch")

# Pivot Table
pivot = df_transactions.pivot_table(
    index="Region", columns="Tier", values="Contract_Value", aggfunc="sum", fill_value=0
)
st.caption("Cross-Tabulation Pivot Table:")
st.dataframe(pivot.style.format("${:,.0f}"), width="stretch")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_transactions = pd.DataFrame({
                "Region": ["Americas", "EMEA", "APAC", "Americas", "EMEA", "APAC", "Americas"],
                "Tier": ["Enterprise", "SMB", "Enterprise", "SMB", "Enterprise", "SMB", "Enterprise"],
                "Contract_Value": [120000, 35000, 95000, 28000, 140000, 42000, 180000]
            })

            summary = df_transactions.groupby("Region").agg(
                Total_Pipeline=("Contract_Value", "sum"),
                Average_Deal=("Contract_Value", "mean"),
                Deal_Count=("Contract_Value", "count")
            ).reset_index()

            st.caption("Grouped Pipeline Summary:")
            st.dataframe(summary.style.format({"Total_Pipeline": "${:,.0f}", "Average_Deal": "${:,.0f}"}), width="stretch")

            pivot = df_transactions.pivot_table(
                index="Region", columns="Tier", values="Contract_Value", aggfunc="sum", fill_value=0
            )
            st.caption("Cross-Tabulation Pivot Table:")
            st.dataframe(pivot.style.format("${:,.0f}"), width="stretch")

# =============================================================================
# 7. HANDLING MISSING VALUES
# =============================================================================
with tabs[6]:
    st.header("7. Handling Missing Values (`isna`, `dropna`, `fillna`)")
    st.markdown(r"""
    Auditing missingness patterns and applying domain-appropriate imputation strategies (mean, median, mode, forward-fill).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

# DataFrame with simulated NaN values
df_dirty = pd.DataFrame({
    "Age": [25.0, np.nan, 32.0, 47.0, np.nan, 29.0],
    "Income": [55000.0, 62000.0, np.nan, 105000.0, 78000.0, 68000.0],
    "Score": [88.0, 92.0, 85.0, np.nan, 79.0, 95.0]
})

strategy = st.selectbox("Imputation Strategy", ["Mean Imputation", "Median Imputation", "Drop Rows (dropna)"])

if strategy == "Mean Imputation":
    df_clean = df_dirty.fillna(df_dirty.mean())
elif strategy == "Median Imputation":
    df_clean = df_dirty.fillna(df_dirty.median())
else:
    df_clean = df_dirty.dropna()

m1, m2 = st.columns(2)
m1.metric("Original Null Count", int(df_dirty.isna().sum().sum()))
m2.metric("Post-Treatment Null Count", int(df_clean.isna().sum().sum()))

st.dataframe(df_clean.style.format("{:.1f}"), width="stretch")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_dirty = pd.DataFrame({
                "Age": [25.0, np.nan, 32.0, 47.0, np.nan, 29.0],
                "Income": [55000.0, 62000.0, np.nan, 105000.0, 78000.0, 68000.0],
                "Score": [88.0, 92.0, 85.0, np.nan, 79.0, 95.0]
            })

            strat = st.selectbox("Imputation Strategy", ["Mean Imputation", "Median Imputation", "Drop Rows (dropna)"], key="imp_strat")
            if strat == "Mean Imputation":
                df_clean = df_dirty.fillna(df_dirty.mean())
            elif strat == "Median Imputation":
                df_clean = df_dirty.fillna(df_dirty.median())
            else:
                df_clean = df_dirty.dropna()

            nm1, nm2 = st.columns(2)
            nm1.metric("Original Null Count", int(df_dirty.isna().sum().sum()))
            nm2.metric("Post-Treatment Nulls", int(df_clean.isna().sum().sum()))

            st.dataframe(df_clean.style.format("{:.1f}"), width="stretch")

# =============================================================================
# 8. MERGING DATASETS
# =============================================================================
with tabs[7]:
    st.header("8. Merging Datasets (`pd.merge`)")
    st.markdown(r"""
    Joining relational datasets using primary and foreign keys with inner, left, and outer join semantics.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

df_users = pd.DataFrame({
    "UserID": [1, 2, 3, 4],
    "Name": ["Sarah", "Liam", "Maya", "Noah"]
})

df_orders = pd.DataFrame({
    "OrderID": [501, 502, 503, 504],
    "UserID": [2, 1, 2, 5],  # User 5 is unregistered
    "Amount": [120.50, 45.00, 210.00, 85.00]
})

join_type = st.selectbox("Join How", ["inner", "left", "right", "outer"])
df_merged = pd.merge(df_users, df_orders, on="UserID", how=join_type)

st.caption(f"Result of `{join_type}` join ({len(df_merged)} rows):")
st.dataframe(df_merged, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_users = pd.DataFrame({
                "UserID": [1, 2, 3, 4],
                "Name": ["Sarah", "Liam", "Maya", "Noah"]
            })
            df_orders = pd.DataFrame({
                "OrderID": [501, 502, 503, 504],
                "UserID": [2, 1, 2, 5],
                "Amount": [120.50, 45.00, 210.00, 85.00]
            })

            j_type = st.selectbox("Join How", ["inner", "left", "right", "outer"], key="join_type_sel")
            df_merged = pd.merge(df_users, df_orders, on="UserID", how=j_type)

            st.caption(f"Result of `{j_type}` join ({len(df_merged)} rows):")
            st.dataframe(df_merged, width="stretch", hide_index=True)

# =============================================================================
# 9. EXPORTING PROCESSED DATA
# =============================================================================
with tabs[8]:
    st.header("9. Exporting Processed Data (`st.download_button`)")
    st.markdown(r"""
    Enabling users to export cleaned, filtered DataFrames back to CSV or Excel binary buffers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import io

df_export = pd.DataFrame({
    "SKU": ["SKU-001", "SKU-002", "SKU-003"],
    "Price": [24.99, 89.50, 14.00],
    "Status": ["Active", "Active", "Archived"]
})

# 1. CSV Byte Export
csv_data = df_export.to_csv(index=False).encode("utf-8")

# 2. Excel Binary Export
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="CleanData")
excel_data = excel_buffer.getvalue()

c_csv, c_xlsx = st.columns(2)
with c_csv:
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="processed_inventory.csv",
        mime="text/csv"
    )
with c_xlsx:
    st.download_button(
        label="📥 Download as Excel (.xlsx)",
        data=excel_data,
        file_name="processed_inventory.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            df_export = pd.DataFrame({
                "SKU": ["SKU-001", "SKU-002", "SKU-003"],
                "Price": [24.99, 89.50, 14.00],
                "Status": ["Active", "Active", "Archived"]
            })

            csv_data = df_export.to_csv(index=False).encode("utf-8")

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False, sheet_name="CleanData")
            excel_data = excel_buffer.getvalue()

            c_csv, c_xlsx = st.columns(2)
            with c_csv:
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="processed_inventory.csv",
                    mime="text/csv",
                    key="dl_csv_btn"
                )
            with c_xlsx:
                st.download_button(
                    label="📥 Download as Excel (.xlsx)",
                    data=excel_data,
                    file_name="processed_inventory.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx_btn"
                )
            st.caption("Binary buffer exports ready for user download.")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Pandas & Streamlit Integration Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Table Sizing** | Using `st.table(df)` on a 10,000-row DataFrame | Use `st.dataframe(df, width="stretch")` with virtualized Arrow scrolling. |
| **Repeated File Reads** | Reading disk CSVs on every slider rerun | Wrap file parsing inside `@st.cache_data`. |
| **Object Dtypes** | Keeping strings as unbounded Python `object` | Convert low-cardinality strings to `category` or PyArrow `string[pyarrow]`. |
| **Excel Export** | Writing temporary `.xlsx` files to the local disk | Use `io.BytesIO()` memory streams with `openpyxl`. |
| **Missing Values** | Ignoring nulls, causing front-end formatting crashes | Audit with `df.isna().sum()` and impute or drop explicitly before rendering. |
""")
