import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(
    page_title="14.3 Interactive Data Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("14.3 Interactive Data Analysis in Streamlit")
st.caption("Building reactive Exploratory Data Analysis (EDA) pipelines: dynamic projections, temporal slicing, multi-dimensional aggregations, dynamic plotting, and data cleaning workflows.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Interactive EDA Pipeline Architecture

Interactive analysis decouples the raw input dataset from the rendered output through an immutable, multi-stage transformation pipeline:

""")

st.graphviz_chart(
    r'''
    digraph AnalysisPipeline {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        raw [label="1. Raw Data Source\nCSV, Parquet, or SQL Database", fillcolor="#DBEAFE", color="#1D4ED8"];
        cached [label="2. In-Memory Cache\n@st.cache_data def load_data()", fillcolor="#FEF3C7", color="#D97706"];
        filters [label="3. Reactive Sidebar Filters\nst.multiselect(), st.slider(), st.selectbox()", fillcolor="#FEF3C7", color="#D97706"];
        ui [label="4. Presentation & Export\nst.dataframe(width='stretch')\nst.download_button()", fillcolor="#D1FAE5", color="#059669"];

        raw -> cached [label="Loads once"];
        cached -> filters [label="Full dataset"];
        filters -> ui [label="Reactive filtered slice"];
    }
    '''
)

st.markdown("""
""")

# Shared synthetic dataset generator for reproducible interactive exploration
@st.cache_data
def get_ecommerce_telemetry():
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = ["AI Compute", "Cloud Storage", "API Gateway", "Vector DB"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    
    return pd.DataFrame({
        "TransactionID": [f"TX-{1000 + i}" for i in range(n)],
        "Date": dates,
        "Category": np.random.choice(categories, n),
        "Region": np.random.choice(regions, n),
        "Revenue": np.random.uniform(150, 4500, n).round(2),
        "Units": np.random.randint(1, 25, n),
        "Customer_Rating": np.random.choice([3.5, 4.0, 4.5, 5.0, np.nan], n, p=[0.1, 0.25, 0.35, 0.25, 0.05]),
        "Support_Tickets": np.random.poisson(1.2, n)
    })

base_df = get_ecommerce_telemetry()

tabs = st.tabs([
    "1. Column Selection",
    "2. Row Filtering",
    "3. Date-Range Filtering",
    "4. Category Filtering",
    "5. Numeric Filtering",
    "6. Group-By & Pivot",
    "7. Aggregations",
    "8. Dynamic Charts",
    "9. Cleaning Workflows"
])

# =============================================================================
# 1. COLUMN SELECTION
# =============================================================================
with tabs[0]:
    st.header("1. Dynamic Column Selection & Projection")
    st.markdown(r"""
    Allow users to project arbitrary subsets of columns from large tables, eliminating visual clutter 
    and optimizing browser memory rendering.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Let users choose visible attributes
all_cols = list(df.columns)
default_cols = ["TransactionID", "Date", "Category", "Revenue"]

selected_cols = st.multiselect(
    "Select Columns to Display",
    options=all_cols,
    default=default_cols
)

if not selected_cols:
    st.warning("Please select at least one column.")
else:
    projected_df = df[selected_cols]
    st.dataframe(projected_df, width="stretch", hide_index=True)
    st.caption(f"Projecting {len(selected_cols)} of {len(all_cols)} columns.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            all_cols = list(base_df.columns)
            def_cols = ["TransactionID", "Date", "Category", "Revenue"]

            sel_cols = st.multiselect(
                "Select Columns to Display",
                options=all_cols,
                default=def_cols,
                key="col_proj_sel"
            )

            if not sel_cols:
                st.warning("Please select at least one column.")
            else:
                proj_df = base_df[sel_cols]
                st.dataframe(proj_df.head(6), width="stretch", hide_index=True)
                st.caption(f"Projecting {len(sel_cols)} of {len(all_cols)} columns (first 6 rows shown).")

# =============================================================================
# 2. ROW FILTERING
# =============================================================================
with tabs[1]:
    st.header("2. Free-Text & Regex Row Searching")
    st.markdown(r"""
    Filtering rows based on substring searches across text and identifier columns.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

search_term = st.text_input("Search Transaction ID or Category", "")
case_sensitive = st.checkbox("Case Sensitive", value=False)

if search_term:
    mask = (
        df["TransactionID"].str.contains(search_term, case=case_sensitive, na=False) |
        df["Category"].str.contains(search_term, case=case_sensitive, na=False)
    )
    filtered = df[mask]
else:
    filtered = df

st.metric("Matching Rows", f"{len(filtered)} / {len(df)}")
st.dataframe(filtered, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            s_term = st.text_input("Search Transaction ID or Category", "", key="row_search_in")
            c_sens = st.checkbox("Case Sensitive", value=False, key="row_cs_chk")

            if s_term:
                s_mask = (
                    base_df["TransactionID"].str.contains(s_term, case=c_sens, na=False) |
                    base_df["Category"].str.contains(s_term, case=c_sens, na=False)
                )
                filtered_s = base_df[s_mask]
            else:
                filtered_s = base_df

            st.metric("Matching Rows", f"{len(filtered_s)} / {len(base_df)}")
            st.dataframe(filtered_s.head(6), width="stretch", hide_index=True)

# =============================================================================
# 3. DATE-RANGE FILTERING
# =============================================================================
with tabs[2]:
    st.header("3. Date-Range Temporal Filtering")
    st.markdown(r"""
    Slicing time-series records via `st.date_input(..., value=(start, end))`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import datetime

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.date_input(
    "Select Temporal Window",
    value=(min_date, min_date + datetime.timedelta(days=30)),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    time_mask = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
    time_filtered = df[time_mask]
    
    st.caption(f"Filtered {len(time_filtered)} rows between {start_d} and {end_d}.")
    st.line_chart(time_filtered.set_index("Date")["Revenue"], height=200)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            min_d = base_df["Date"].min().date()
            max_d = base_df["Date"].max().date()

            d_range = st.date_input(
                "Select Temporal Window",
                value=(min_d, min_d + datetime.timedelta(days=30)),
                min_value=min_d,
                max_value=max_d,
                key="date_range_picker"
            )

            if isinstance(d_range, tuple) and len(d_range) == 2:
                start_d, end_d = d_range
                t_mask = (base_df["Date"].dt.date >= start_d) & (base_df["Date"].dt.date <= end_d)
                t_filt = base_df[t_mask]
                st.caption(f"Filtered {len(t_filt)} rows between {start_d} and {end_d}.")
                st.line_chart(t_filt.set_index("Date")["Revenue"], height=180)

# =============================================================================
# 4. CATEGORY FILTERING
# =============================================================================
with tabs[3]:
    st.header("4. Categorical Filtering (Hierarchical & Multi-Select)")
    st.markdown(r"""
    Filtering records by categorical factors with parent-child cascade dependencies.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

c1, c2 = st.columns(2)
with c1:
    selected_regions = st.multiselect(
        "Select Regions", 
        options=df["Region"].unique(),
        default=df["Region"].unique()[:2]
    )
with c2:
    selected_categories = st.multiselect(
        "Select Categories",
        options=df["Category"].unique(),
        default=df["Category"].unique()
    )

cat_mask = (df["Region"].isin(selected_regions)) & (df["Category"].isin(selected_categories))
df_cat_filtered = df[cat_mask]

st.metric("Records Found", f"{len(df_cat_filtered)} / {len(df)}")
st.dataframe(df_cat_filtered[["TransactionID", "Region", "Category", "Revenue"]], width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                sel_regs = st.multiselect(
                    "Select Regions", 
                    options=base_df["Region"].unique(),
                    default=base_df["Region"].unique()[:2],
                    key="cat_reg_sel"
                )
            with c2:
                sel_cats = st.multiselect(
                    "Select Categories",
                    options=base_df["Category"].unique(),
                    default=base_df["Category"].unique(),
                    key="cat_cat_sel"
                )

            cat_mask = (base_df["Region"].isin(sel_regs)) & (base_df["Category"].isin(sel_cats))
            df_cat_filtered = base_df[cat_mask]

            st.metric("Records Found", f"{len(df_cat_filtered)} / {len(base_df)}")
            st.dataframe(df_cat_filtered[["TransactionID", "Region", "Category", "Revenue"]].head(6), width="stretch", hide_index=True)

# =============================================================================
# 5. NUMERIC FILTERING
# =============================================================================
with tabs[4]:
    st.header("5. Numeric-Range Threshold Filtering")
    st.markdown(r"""
    Restricting records using continuous range sliders or dual min/max numeric inputs.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

min_rev = float(df["Revenue"].min())
max_rev = float(df["Revenue"].max())

rev_range = st.slider(
    "Revenue Window ($)",
    min_value=min_rev,
    max_value=max_rev,
    value=(500.0, 3000.0),
    step=50.0
)

ticket_cap = st.slider("Max Support Tickets", 0, int(df["Support_Tickets"].max()), 3)

num_mask = (
    (df["Revenue"] >= rev_range[0]) & 
    (df["Revenue"] <= rev_range[1]) & 
    (df["Support_Tickets"] <= ticket_cap)
)
df_num_filt = df[num_mask]

st.metric("Total Revenue in Scope", f"${df_num_filt['Revenue'].sum():,.2f}")
st.dataframe(df_num_filt[["TransactionID", "Revenue", "Support_Tickets"]], width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            min_rev = float(base_df["Revenue"].min())
            max_rev = float(base_df["Revenue"].max())

            rev_range = st.slider(
                "Revenue Window ($)",
                min_value=min_rev,
                max_value=max_rev,
                value=(500.0, 3000.0),
                step=50.0,
                key="num_rev_slider"
            )
            ticket_cap = st.slider("Max Support Tickets", 0, int(base_df["Support_Tickets"].max()), 3, key="num_tick_slider")

            num_mask = (
                (base_df["Revenue"] >= rev_range[0]) & 
                (base_df["Revenue"] <= rev_range[1]) & 
                (base_df["Support_Tickets"] <= ticket_cap)
            )
            df_num_filt = base_df[num_mask]

            st.metric("Total Revenue in Scope", f"${df_num_filt['Revenue'].sum():,.2f}")
            st.dataframe(df_num_filt[["TransactionID", "Revenue", "Support_Tickets"]].head(6), width="stretch", hide_index=True)

# =============================================================================
# 6. GROUP-BY & PIVOT
# =============================================================================
with tabs[5]:
    st.header("6. Interactive Group-By Dimension Selection")
    st.markdown(r"""
    Allow users to select grouping dimensions dynamically to view high-level summaries.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

group_dim = st.selectbox("Group Dimension", ["Region", "Category"])
metric_col = st.selectbox("Measure Column", ["Revenue", "Units", "Support_Tickets"])

grouped_df = df.groupby(group_dim)[metric_col].agg(["sum", "mean", "count"]).reset_index()
grouped_df.columns = [group_dim, "Total", "Average", "Count"]

st.dataframe(grouped_df.style.format({"Total": "{:,.2f}", "Average": "{:,.2f}"}), width="stretch", hide_index=True)
st.bar_chart(grouped_df.set_index(group_dim)["Total"], height=200)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            g_dim = st.selectbox("Group Dimension", ["Region", "Category"], key="grp_dim_sel")
            m_col = st.selectbox("Measure Column", ["Revenue", "Units", "Support_Tickets"], key="grp_col_sel")

            grouped_df = base_df.groupby(g_dim)[m_col].agg(["sum", "mean", "count"]).reset_index()
            grouped_df.columns = [g_dim, "Total", "Average", "Count"]

            st.dataframe(grouped_df.style.format({"Total": "{:,.2f}", "Average": "{:,.2f}"}), width="stretch", hide_index=True)
            st.bar_chart(grouped_df.set_index(g_dim)["Total"], height=160)

# =============================================================================
# 7. AGGREGATIONS
# =============================================================================
with tabs[6]:
    st.header("7. Metric Aggregation Function Selection")
    st.markdown(r"""
    Dynamically apply statistical functions (`sum`, `mean`, `median`, `std`, `min`, `max`) across subsets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

agg_fn = st.selectbox("Select Aggregation Function", ["mean", "sum", "median", "std", "min", "max"])

summary_series = df[["Revenue", "Units", "Support_Tickets"]].agg(agg_fn)
summary_df = pd.DataFrame(summary_series, columns=[f"Aggregated ({agg_fn})"]).T

st.dataframe(summary_df.style.format("{:,.2f}"), width="stretch")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            a_fn = st.selectbox("Select Aggregation Function", ["mean", "sum", "median", "std", "min", "max"], key="agg_fn_sel")
            sum_s = base_df[["Revenue", "Units", "Support_Tickets"]].agg(a_fn)
            sum_df = pd.DataFrame(sum_s, columns=[f"Aggregated ({a_fn})"]).T
            st.dataframe(sum_df.style.format("{:,.2f}"), width="stretch")

# =============================================================================
# 8. DYNAMIC CHARTS
# =============================================================================
with tabs[7]:
    st.header("8. Dynamic Chart Generation")
    st.markdown(r"""
    Allow users to select chart types (Line, Bar, Area, Scatter) and bind X/Y axes dynamically.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"])
x_axis = st.selectbox("X-Axis", ["Date", "Region", "Category"])
y_axis = st.selectbox("Y-Axis", ["Revenue", "Units", "Support_Tickets"])

chart_data = df.groupby(x_axis)[y_axis].sum().reset_index()

if chart_type == "Line Chart":
    st.line_chart(chart_data.set_index(x_axis)[y_axis], height=240)
elif chart_type == "Bar Chart":
    st.bar_chart(chart_data.set_index(x_axis)[y_axis], height=240)
elif chart_type == "Area Chart":
    st.area_chart(chart_data.set_index(x_axis)[y_axis], height=240)
elif chart_type == "Scatter Plot":
    st.scatter_chart(df, x="Units", y="Revenue", color="Category", height=240)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            c_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"], key="dyn_chart_type")
            x_ax = st.selectbox("X-Axis", ["Date", "Region", "Category"], key="dyn_x_ax")
            y_ax = st.selectbox("Y-Axis", ["Revenue", "Units", "Support_Tickets"], key="dyn_y_ax")

            c_data = base_df.groupby(x_ax)[y_ax].sum().reset_index()

            if c_type == "Line Chart":
                st.line_chart(c_data.set_index(x_ax)[y_ax], height=200)
            elif c_type == "Bar Chart":
                st.bar_chart(c_data.set_index(x_ax)[y_ax], height=200)
            elif c_type == "Area Chart":
                st.area_chart(c_data.set_index(x_ax)[y_ax], height=200)
            else:
                st.scatter_chart(base_df, x="Units", y="Revenue", color="Category", height=200)

# =============================================================================
# 9. CLEANING WORKFLOWS
# =============================================================================
with tabs[8]:
    st.header("9. Interactive Data Cleaning Workflows")
    st.markdown(r"""
    A multi-step preprocessing console allowing users to inspect missingness, deduplicate records, 
    and fill missing numeric fields with verifiable audit feedback.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Interactive cleaning console
clean_df = df.copy()

st.caption("Step 1: Audit Missing Values")
null_counts = clean_df.isna().sum()
st.write(null_counts[null_counts > 0])

st.caption("Step 2: Cleaning Actions")
do_dedup = st.checkbox("Remove Duplicate Rows", value=True)
impute_col = st.selectbox("Impute Missing Values For", ["Customer_Rating"])
impute_val = st.radio("Impute Value", ["Median", "Mean", "Zero"])

if do_dedup:
    clean_df = clean_df.drop_duplicates()

if impute_val == "Median":
    clean_df[impute_col] = clean_df[impute_col].fillna(clean_df[impute_col].median())
elif impute_val == "Mean":
    clean_df[impute_col] = clean_df[impute_col].fillna(clean_df[impute_col].mean())
else:
    clean_df[impute_col] = clean_df[impute_col].fillna(0)

st.success(f"Cleaned DataFrame: {clean_df.isna().sum().sum()} remaining nulls.")
st.dataframe(clean_df.head(5), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            clean_df = base_df.copy()
            st.caption("Step 1: Audit Missing Values")
            null_s = clean_df.isna().sum()
            st.write(null_s[null_s > 0].to_dict())

            st.caption("Step 2: Cleaning Actions")
            do_dedup = st.checkbox("Remove Duplicate Rows", value=True, key="clean_dedup_chk")
            imp_val = st.radio("Impute Customer_Rating With", ["Median", "Mean", "Zero"], horizontal=True, key="clean_imp_radio")

            if do_dedup:
                clean_df = clean_df.drop_duplicates()

            if imp_val == "Median":
                clean_df["Customer_Rating"] = clean_df["Customer_Rating"].fillna(clean_df["Customer_Rating"].median())
            elif imp_val == "Mean":
                clean_df["Customer_Rating"] = clean_df["Customer_Rating"].fillna(clean_df["Customer_Rating"].mean())
            else:
                clean_df["Customer_Rating"] = clean_df["Customer_Rating"].fillna(0)

            st.success(f"Remaining nulls: {clean_df.isna().sum().sum()}")
            st.dataframe(clean_df.head(5), width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Interactive Data Analysis Best Practices

| Technique | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Filtering Chains** | Creating 10 copies of DataFrames per widget | Compute a single boolean mask `mask = mask1 & mask2 & mask3` and slice once. |
| **Temporal Filters** | Converting dates back and forth to strings | Keep `pd.to_datetime()` timestamp objects and compare with `.dt.date`. |
| **Group-by Reruns** | Recomputing heavy multi-column group-bys on every keystroke | Wrap the heavy transform in `@st.cache_data`. |
| **Missing Imputation** | Imputing on raw source dataset in-place | Create explicit immutable copies (`clean_df = df.copy()`) so resets work reliably. |
""")
