import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="15.3 Analytical Workflows | GenAI Roadmap",
    page_icon="📈",
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
    .flow-badge {
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
st.title("📈 15.3 Analytical Workflows — Ask → Analyze → Explain & Reporting")
st.markdown(
    """
Welcome to **Lesson 15.3: Analytical Workflows**.  
Single-prompt data analysis is insufficient for enterprise decision-making. Asking an LLM to simultaneously write code, run statistics, detect anomalies, and compose an executive narrative in one turn leads to shallow reasoning and calculation errors.

Production data intelligence systems decompose analysis into the **Canonical "Ask → Analyze → Explain" Pipeline**: planning mathematical inquiries, executing deterministic data transformations, auditing statistical significance, and synthesizing executive deliverables.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Canonical "Ask → Analyze → Explain" Workflow
# ==============================================================================
st.header("1. The Canonical 3-Stage Analytical Framework", divider="rainbow")

col_wf1, col_wf2, col_wf3 = st.columns(3)

with col_wf1:
    with st.container(border=True):
        st.subheader("Stage 1: ASK (Decomposition)")
        st.markdown(
            """
- **Goal:** Dissect the business question into quantitative testable hypotheses.
- **Actions:**
  - Identify target metrics (e.g. CAC, LTV, churn rate)
  - Map required tables and columns
  - Define time horizons and cohort segmentations
- **Output:** Structured Analytical Plan & SQL/Python pseudocode.
"""
        )

with col_wf2:
    with st.container(border=True):
        st.subheader("Stage 2: ANALYZE (Execution)")
        st.markdown(
            """
- **Goal:** Execute deterministic computation in a sandboxed runtime.
- **Actions:**
  - Run SQL aggregation queries
  - Execute Pandas / NumPy data cleaning & transforms
  - Compute statistical metrics (p-values, IQR, correlation)
- **Output:** Raw statistical outputs, verification tables, chart data.
"""
        )

with col_wf3:
    with st.container(border=True):
        st.subheader("Stage 3: EXPLAIN (Synthesis)")
        st.markdown(
            """
- **Goal:** Translate mathematical findings into executive narrative.
- **Actions:**
  - Ground conclusions strictly in Stage 2 outputs
  - Highlight business impact and strategic levers
  - Recommend tailored visualization layouts
- **Output:** Executive brief, KPIs, and actionable next steps.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Core Analytical Workflows Matrix
# ==============================================================================
st.header("2. Core Production Analytical Workflows", divider="rainbow")
st.markdown("Specialized multi-stage pipelines tailored to enterprise data science problems:")

workflows_df = pd.DataFrame({
    "Workflow Pipeline": [
        "1. Automated Anomaly Detection",
        "2. Multi-Period Trend Analysis",
        "3. Data Validation & Quality Gate",
        "4. Executive Board Report Generator",
        "5. Visualization Recommendation Engine"
    ],
    "Input Signals": [
        "Hourly metric time-series streams (traffic, revenue, error rates)",
        "Quarterly or monthly financial/cohort performance tables",
        "Raw inbound batch feeds, customer CSV uploads, ETL landing zones",
        "Multi-department KPI summaries + quarterly strategic goals",
        "Dataset dimensions, cardinality, skewness, variable types"
    ],
    "Analytical Execution": [
        "Computes rolling z-scores, IQR fences, or seasonal decomposition (STL)",
        "Calculates Year-over-Year (YoY) and Month-over-Month (MoM) compound growth",
        "Asserts schema completeness, null percentage, and foreign key integrity",
        "Aggregates variance between projected vs actual financial targets",
        "Applies visual grammar rules (continuous vs categorical vs temporal)"
    ],
    "Generated Deliverable": [
        "PagerDuty incident alert with root-cause candidate hypotheses",
        "Decomposition summary attributing growth to specific cohorts/regions",
        "Data quality score card + automated cleaning/imputation script",
        "Executive PDF briefing with narrative takeaways and risks",
        "Production Plotly / Seaborn chart generation code"
    ]
})
st.dataframe(workflows_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The 3-Stage Analytical State Machine", divider="rainbow")
st.markdown("How autonomous data analytics agents transition between planning, computation, and reporting:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        biz_q [label="Business Question\n(e.g., Why did churn spike?)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        stage1 [label="Stage 1: ASK\n(Hypothesis & Query Plan)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        stage2 [label="Stage 2: ANALYZE\n(Pandas / SQL Execution)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        verifier [label="Quality Gate\n(Check Nulls & Signif)", fillcolor="#EAB308", fontcolor="#000000"]
        stage3 [label="Stage 3: EXPLAIN\n(Executive Storytelling)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        charts [label="Interactive Dashboard\n& Action Plan", fillcolor="#3B82F6", fontcolor="#FFFFFF"]

        biz_q -> stage1 -> stage2 -> verifier
        verifier -> stage3 [label=" Verified", color="#10B981"]
        verifier -> stage2 [label=" Re-run Query", style="dashed", color="#EF4444"]
        stage3 -> charts
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Analytical Workflow Simulator
# ==============================================================================
st.header("4. Interactive Analytical Workflow Simulator", divider="rainbow")
st.markdown("Walk through the 3-stage pipeline analyzing an enterprise SaaS churn spike:")

step_slider = st.select_slider(
    "Select Active Workflow Stage",
    options=["1. ASK (Decompose)", "2. ANALYZE (Execute)", "3. EXPLAIN (Report)"],
    value="1. ASK (Decompose)"
)

if step_slider == "1. ASK (Decompose)":
    st.subheader("🔍 Stage 1: Hypothesis Decomposition & Query Planning")
    st.markdown(
        """
**Business Trigger:** *"Enterprise SaaS customer churn jumped from 2.1% in Q2 to 4.8% in Q3. Diagnose why."*

**Stage 1 Prompt Plan Generated by LLM:**
1. **Hypothesis A (Price Sensitivity):** Did the August pricing tier change drive cancellations among Pro users?
2. **Hypothesis B (Product Reliability):** Did the major v4.0 release introduce critical API outages impacting enterprise accounts?
3. **Hypothesis C (Support SLA Degradation):** Did ticket resolution times increase following customer support reorganization?

**Generated Execution SQL / Code Plan:**
```sql
-- Query 1: Churn rate segmented by customer tier before and after August 1
-- Query 2: Correlation between customer churn and P0 incidents experienced
-- Query 3: Average time-to-first-response on Zendesk tickets for churned vs retained accounts
```
"""
    )
elif step_slider == "2. ANALYZE (Execute)":
    st.subheader("💻 Stage 2: Sandboxed Data Execution & Statistical Output")
    st.markdown(
        """
**Sandboxed Python / Pandas Execution Telemetry:**
```python
# Executed in sandbox:
df_churn = pd.read_sql("SELECT * FROM churn_events_q3", conn)

# Test correlation between incident count and churn
corr_incidents = df_churn['experienced_p0_outages'].corr(df_churn['churned_boolean'])
p0_churn_rate = df_churn[df_churn['experienced_p0_outages'] > 0]['churned_boolean'].mean()
no_p0_churn_rate = df_churn[df_churn['experienced_p0_outages'] == 0]['churned_boolean'].mean()

print(f"P0 Outage Churn Rate: {p0_churn_rate * 100:.1f}%")
print(f"No Outage Churn Rate: {no_p0_churn_rate * 100:.1f}%")
print(f"Pricing Shift Churn Delta: +0.2% (Statistically Insignificant, p=0.42)")
```

**Empirical Verification Matrix:**
- Accounts experiencing >= 1 P0 outage churned at **14.2%** (vs **1.8%** baseline).
- Pricing shift churn delta was only **+0.2%** (p-value: 0.42 — rejected as primary cause).
"""
    )
else:
    st.subheader("📊 Stage 3: Executive Synthesis & Visual Recommendation")
    st.markdown(
        """
**Executive Brief (Delivered to VP of Engineering & CEO):**

### Root Cause Diagnosis:
The Q3 churn spike (2.1% -> 4.8%) was **not caused by pricing tier changes**, which accounted for only a negligible 0.2% delta. 
Instead, the churn was overwhelmingly concentrated among enterprise clients impacted by the **August v4.0 infrastructure outages**. Clients experiencing at least one P0 outage experienced a **nearly 8x surge in churn probability (14.2% vs 1.8%)**.

### Strategic Action Items:
1. **Infrastructure Stability SLA**: Institute a temporary feature freeze until the database connection pooling issue from v4.0 is resolved.
2. **High-Touch Account Outreach**: Trigger immediate executive sponsor outreach to all 42 enterprise accounts that suffered outages in Q3.
3. **Recommended Visualization**: Dual-axis line chart plotting weekly P0 outage frequency against cancellation notice submissions.
"""
    )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: 3-Stage Analytical Agent", divider="rainbow")
st.markdown("Python orchestration script managing the end-to-end Ask -> Analyze -> Explain state machine:")

st.code(
    '''from typing import Dict, Any

class AnalyticalWorkflowAgent:
    """
    Production agent orchestrating the Ask -> Analyze -> Explain lifecycle.
    """
    def __init__(self, db_schema: str):
        self.db_schema = db_schema

    def stage_ask(self, business_question: str) -> str:
        """Decomposes question into quantitative hypotheses and query plan."""
        return f"""<stage>ASK</stage>
<schema>{self.db_schema}</schema>
<business_question>{business_question}</business_question>
<instructions>
Formulate 3 testable hypotheses. Generate exact SQL queries to test each hypothesis.
</instructions>"""

    def stage_explain(self, business_question: str, query_results_summary: str) -> str:
        """Synthesizes deterministic query results into executive recommendations."""
        return f"""<stage>EXPLAIN</stage>
<business_question>{business_question}</business_question>
<empirical_results>
{query_results_summary}
</empirical_results>
<instructions>
Write an executive brief.
Highlight the primary driver, quantify business impact, and recommend 2 actions.
Do not hallucinate numbers not present in empirical_results.
</instructions>"""

# Example instantiation
agent = AnalyticalWorkflowAgent(db_schema="TABLE churn (user_id UUID, tier TEXT, outages INT, churned BOOL);")
print("Analytical Agent prepared with Stage 1 (Ask) and Stage 3 (Explain) prompts.")
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
    with st.expander("Q1: Why is separating the 'Analyze' stage from the 'Explain' stage critical for eliminating data hallucinations?", expanded=True):
        st.markdown(
            """
When an LLM is asked to analyze and explain in a single prompt, it performs two conflicting cognitive tasks: **symbolic calculation** and **rhetorical storytelling**. 
Language models will unconsciously bend numbers to fit a compelling narrative (e.g. inventing a 25% churn drop to support a recommendation).
By enforcing a strict **Analyze stage** (which outputs pure Python/SQL numbers verified in a sandbox) followed by an **Explain stage** (which is strictly grounded in the verified table), numbers cannot be fabricated.
"""
        )
    with st.expander("Q2: How do you design an automated 'Data Quality Gate' before running analysis pipelines?"):
        st.markdown(
            """
Before executing analysis queries, run an automated assertion suite:
1. **Null Checks**: Assert critical IDs and timestamps have 0% missingness.
2. **Outlier Fences**: Verify numeric ranges (e.g. prices are positive, dates are in the past).
3. **Cardinality & Distribution**: Check that categorical columns have expected distinct values.
If any assertion fails, halt the pipeline and prompt an alert rather than generating misleading conclusions on corrupt data.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we let the LLM directly suggest business strategy without showing the statistical calculation?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Executives only care about high-level recommendations, so skip the math."*
- **Winning Response**: *"Never. An AI recommendation without auditable statistical grounding is a corporate liability. In enterprise BI, every strategic takeaway must cite an exact empirical metric from the analysis phase (e.g., 'p=0.002, 14.2% vs 1.8% rate'). The intermediate calculation trace must always be saved for auditability and compliance."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Ask → Analyze → Explain**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The 3-stage framework separating hypothesis planning, sandboxed calculation, and executive narrative synthesis.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Auditability Trace**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Persisting intermediate SQL queries and execution outputs so business leaders can verify how an AI conclusion was derived.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Data Quality Gate**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Pre-analysis assertions that check schema, nulls, and outliers to prevent running analysis on corrupt data.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 15.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Multi-Stage Decomposition**: Always decouple hypothesis planning, code calculation, and reporting.
2. **Grounding in Computation**: Ensure narrative claims strictly reference verified numbers from sandbox execution.
3. **Auditability First**: Preserve all intermediate queries and outputs for compliance.
4. **Data Quality Gates**: Assert dataset integrity before running analytical pipelines.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know the distinct stages of the Ask → Analyze → Explain framework.", value=True, key="chk_15_3_1")
    st.checkbox("I understand why single-prompt data analysis leads to calculation hallucinations.", value=True, key="chk_15_3_2")
    st.checkbox("I can implement an automated Data Quality Gate in Python.", value=True, key="chk_15_3_3")
    st.checkbox("I understand how to design auditable analytical agents for executive leadership.", value=True, key="chk_15_3_4")
