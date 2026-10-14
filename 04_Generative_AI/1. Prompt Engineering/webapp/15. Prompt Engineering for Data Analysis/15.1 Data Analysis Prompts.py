import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="15.1 Data Analysis Prompts | GenAI Roadmap",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .data-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("📊 15.1 Data Analysis Prompts — Summarization, Cleaning, EDA & Statistics")
st.markdown(
    """
Welcome to **Lesson 15.1: Data Analysis Prompts**.  
Data science and business intelligence rely on rigorous empirical facts. However, LLMs are fundamentally language models—they are not statistical calculators. Asking an LLM to calculate the standard deviation of 10,000 numbers in its head leads directly to hallucinated numbers.

Production data analysis prompting bridges this gap: prompting models to **generate executable Python / Pandas scripts**, interpret **pre-computed statistical summaries**, diagnose **data quality defects**, and synthesize **executive narrative insights**.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 5 Pillars of Data Analysis Prompting
# ==============================================================================
st.header("1. The 5 Pillars of Production Data Analysis Prompting", divider="rainbow")
st.markdown("Structuring analytical prompts across the standard data science lifecycle:")

pillars_df = pd.DataFrame({
    "Pillar": [
        "1. Data Summarization",
        "2. Data Cleaning & Quality Audit",
        "3. Exploratory Data Analysis (EDA)",
        "4. Statistical Inference & Testing",
        "5. Pattern & Anomaly Detection"
    ],
    "Core Analytical Task": [
        "Distilling multi-table schemas, metadata dimensions, and statistical distributions",
        "Identifying missing values, duplicate records, outliers, and type corruptions",
        "Uncovering feature distributions, skewness, correlations, and segment breakdowns",
        "Formulating null hypotheses, calculating p-values, verifying statistical significance",
        "Detecting multivariate anomalies, sudden trend drift, and customer churn clusters"
    ],
    "Best Practice Prompting Strategy": [
        "Pass `df.describe().to_markdown()` rather than raw raw unaggregated rows",
        "Ask for a pandas data cleaning script that imputes nulls and casts types",
        "Prompt the model to generate Matplotlib / Seaborn visualization code",
        "Require the model to state confidence intervals and sample size caveats",
        "Instruct the model to flag root-cause drivers rather than raw correlations"
    ]
})
st.dataframe(pillars_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Preventing Hallucinated Arithmetic
# ==============================================================================
st.header("2. Critical Principle: Never Ask LLMs to Do Mental Math", divider="rainbow")

col_mm1, col_mm2 = st.columns(2)

with col_mm1:
    with st.container(border=True):
        st.subheader("❌ Flawed Approach (Mental Arithmetic)")
        st.markdown(
            """
**Naive User Prompt:**
```text
Here is a list of 50 customer order amounts:
[14.20, 99.50, 120.00, 4.99, 55.40, ...]
What is the exact median order value and standard deviation?
```

**Why It Fails:**
- Autoregressive token prediction cannot perform floating-point square roots or sorting algorithms in token space.
- The model outputs a plausible-sounding hallucinated number (e.g., *"The median is $62.30"*), which is statistically incorrect.
"""
        )

with col_mm2:
    with st.container(border=True):
        st.subheader("✅ Production Approach (Code Generation / Aggregated Stats)")
        st.markdown(
            """
**Production Prompt Pattern:**
```text
<dataset_summary>
Shape: (10000, 5)
Columns: order_id (int), amount_usd (float), customer_tier (str)
amount_usd describe:
count: 10000 | mean: 84.21 | std: 32.14 | min: 4.99
25%: 54.10 | 50%: 81.50 | 75%: 110.20 | max: 495.00
</dataset_summary>

<instructions>
Interpret the pre-computed distribution metrics above.
Explain skewness and recommend outliers thresholds.
</instructions>
```
**Why It Succeeds:**
- The numbers are mathematically calculated by NumPy/Pandas beforehand.
- The LLM focuses purely on reasoning, interpretation, and strategic recommendations.
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Grounded Analytical Agent Pipeline", divider="rainbow")
st.markdown("How enterprise data analysis prompts interact with Python execution environments:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        raw_df [label="Raw Tabular Data\n(SQL / CSV / Parquet)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        profiler [label="Pandas Profiler\n(df.info() + df.describe())", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        prompt [label="Analytical Prompt\n(Grounded with Summary)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="LLM Code Reasoner\n(Generates Analysis Code)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        sandbox [label="Python REPL Sandbox\n(Executes Pandas/Seaborn)", fillcolor="#EAB308", fontcolor="#000000"]
        report [label="Executive Dashboard\n& Visual Charts", fillcolor="#10B981", fontcolor="#FFFFFF"]

        raw_df -> profiler -> prompt -> llm -> sandbox -> report
        raw_df -> sandbox
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Data Analysis Workbench
# ==============================================================================
st.header("4. Interactive Data Analysis Workbench", divider="rainbow")
st.markdown("Select a business dataset and explore how structured prompt engineering generates grounded insights:")

dataset_type = st.selectbox(
    "Select Business Scenario",
    [
        "E-Commerce Customer Churn Analysis",
        "SaaS Server Latency & Outage Telemetry",
        "Healthcare Patient Readmission Statistics"
    ]
)

col_dw1, col_dw2 = st.columns([1, 1])

with col_dw1:
    st.subheader("📋 Pre-Computed Statistical Profile")
    if dataset_type == "E-Commerce Customer Churn Analysis":
        summary_stats = pd.DataFrame({
            "Metric": ["Total Users", "Churn Rate (%)", "Avg Days Since Last Login", "Avg Lifetime Spend ($)", "Support Tickets Filed"],
            "Active Tier": ["8,500", "0.0%", "4.2 days", "$420.50", "0.8 tickets/yr"],
            "Churned Tier": ["1,500", "100.0%", "48.6 days", "$85.20", "4.6 tickets/yr"]
        })
    elif dataset_type == "SaaS Server Latency & Outage Telemetry":
        summary_stats = pd.DataFrame({
            "Cluster Region": ["us-east-1", "eu-central-1", "ap-southeast-1"],
            "p50 Latency (ms)": ["42 ms", "38 ms", "65 ms"],
            "p99 Latency (ms)": ["180 ms", "120 ms", "850 ms"],
            "Error Rate (%)": ["0.02%", "0.01%", "3.40%"]
        })
    else:
        summary_stats = pd.DataFrame({
            "Patient Cohort": ["Standard Discharge", "Early Follow-Up (Within 7d)", "No Follow-Up"],
            "Cohort Size": ["5,200", "3,100", "2,100"],
            "30-Day Readmission Rate": ["18.4%", "8.2%", "29.7%"],
            "Avg Medication Adherence": ["72%", "91%", "54%"]
        })
    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

with col_dw2:
    st.subheader("💡 Grounded Analytical Narrative")
    if dataset_type == "E-Commerce Customer Churn Analysis":
        st.markdown(
            """
**Key Empirical Findings:**
1. **Support Ticket Friction as Leading Indicator**: Churned users filed **5.7x more support tickets** (4.6 vs 0.8) prior to leaving.
2. **Inactivity Threshold**: Inactivity exceeding **30 days** indicates 92% churn probability.
3. **Recommended Action**: Implement an automated retention intervention whenever a customer logs a 3rd support ticket within a 60-day window.
"""
        )
    elif dataset_type == "SaaS Server Latency & Outage Telemetry":
        st.markdown(
            """
**Key Empirical Findings:**
1. **Severe Tail Latency Degradation in APAC**: `ap-southeast-1` exhibits a **13x spike** from p50 (65ms) to p99 (850ms), with a critical 3.4% error rate.
2. **US and EU Stability**: `us-east-1` and `eu-central-1` remain healthy with error rates under 0.02%.
3. **Recommended Action**: Investigate database read-replica contention or regional edge CDN cache misses in the Singapore cluster.
"""
        )
    else:
        st.markdown(
            """
**Key Empirical Findings:**
1. **Follow-Up Protective Factor**: Scheduling a physician follow-up within 7 days reduces 30-day readmissions from 29.7% to **8.2% (a 72% relative risk reduction)**.
2. **Adherence Correlation**: Medication adherence jumps from 54% to 91% with early clinical contact.
3. **Recommended Action**: Mandate hospital discharge workflow automation to lock in 7-day follow-up appointments before patient discharge.
"""
        )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Grounded DataFrame Prompting", divider="rainbow")
st.markdown("Python utility to extract statistical summaries and format grounded analysis prompts:")

st.code(
    '''import pandas as pd
from typing import Dict, Any

def create_grounded_analysis_prompt(df: pd.DataFrame, target_goal: str) -> str:
    """
    Extracts deterministic statistical summaries from a DataFrame
    and embeds them into an analytical prompt to eliminate hallucinated arithmetic.
    """
    buffer = []
    buffer.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    buffer.append("\\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        buffer.append(f"- {col}: {dtype}")
        
    buffer.append("\\nDescriptive Statistics (Pre-computed):")
    buffer.append(df.describe().to_markdown())
    
    context_str = "\\n".join(buffer)
    
    prompt = f"""<system>
You are a Principal Data Scientist and Business Intelligence Strategist.
Base your conclusions STRICTLY on the pre-computed mathematical summary below.
Do not hallucinate external metrics.
</system>

<data_profile>
{context_str}
</data_profile>

<analytical_goal>
{target_goal}
</analytical_goal>

<output_format>
1. Executive Summary (2 sentences)
2. Top 3 Empirical Findings (grounded with numbers from data_profile)
3. Actionable Strategic Recommendations
4. Recommended Visualizations (Chart type + X/Y axis definitions)
</output_format>
"""
    return prompt

# Demonstration
sample_df = pd.DataFrame({
    "revenue": [100.0, 150.0, 200.0, 80.0, 500.0],
    "marketing_spend": [20.0, 30.0, 45.0, 15.0, 90.0]
})
print(create_grounded_analysis_prompt(sample_df, "Evaluate marketing ROI and revenue leverage."))
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why should you never pass raw unaggregated rows (e.g. 5,000 CSV lines) directly to an LLM for statistical analysis?", expanded=True):
        st.markdown(
            """
1. **Severe Token Consumption & Cost**: 5,000 CSV rows can consume 50,000 to 100,000 tokens, resulting in massive API charges and high latency.
2. **Context Window Attention Dilution**: Transformers cannot reliably calculate sums, correlations, or medians over long text sequences. Attention gets diluted, and arithmetic calculations will be inaccurate.
3. **Best Practice Solution**: Compute statistical aggregations (`mean`, `median`, `IQR`, correlation matrices) using Pandas / Polars in Python first, and pass only the compact statistical summary table (~300 tokens) to the model.
"""
        )
    with st.expander("Q2: How do you prompt an LLM to recommend the most effective data visualizations?"):
        st.markdown(
            """
Instruct the model to evaluate **data dimensions and visual grammar**:
- 1 Continuous Variable -> Histogram or KDE plot (to inspect distribution)
- 2 Continuous Variables -> Scatter plot (to inspect correlation)
- 1 Categorical + 1 Continuous -> Box plot or Violin plot (to inspect variance across cohorts)
- Time Series -> Line chart (to inspect seasonality and trend)
Require the prompt to output exact Seaborn / Plotly code snippets rather than abstract advice.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we trust an LLM to determine whether an A/B test is statistically significant?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, give it conversion rates and sample sizes and ask if p < 0.05."*
- **Winning Response**: *"No. Asking an LLM to mentally compute t-tests, chi-square statistics, or z-scores produces hallucinated p-values. In production, prompt the LLM to write the exact `scipy.stats` or `statsmodels` code to compute the hypothesis test in a Python sandbox, and feed the resulting deterministic p-value and confidence interval back to the model for business interpretation."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Grounded Statistical Prompting**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Pre-computing mathematical metrics with Pandas/NumPy and passing the summary table to the LLM to prevent hallucinated math.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Code Interpreter Pattern**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Prompting an LLM to write executable Python data analysis code that executes in a sandboxed runtime rather than doing math in token space.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Descriptive Profiling**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Feeding `df.describe()` and schema metrics into the prompt context to ground qualitative reasoning in empirical distributions.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 15.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **No Mental Math**: Never ask LLMs to compute arithmetic or statistical distributions in token space.
2. **Pre-Compute Summaries**: Feed compact Pandas descriptive summaries into the context window.
3. **Code Interpreter Architecture**: Prompt the model to write executable Python/SQL scripts for deterministic computation.
4. **Visual Recommendations**: Prompt for exact Seaborn/Plotly visual specifications based on variable types.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why LLMs hallucinate calculations when given raw data points.", value=True, key="chk_15_1_1")
    st.checkbox("I know how to extract and format pre-computed Pandas profiles for prompt grounding.", value=True, key="chk_15_1_2")
    st.checkbox("I can implement the Code Interpreter pattern for statistical hypothesis testing.", value=True, key="chk_15_1_3")
    st.checkbox("I know how to prompt for specific data visualization recommendations.", value=True, key="chk_15_1_4")
