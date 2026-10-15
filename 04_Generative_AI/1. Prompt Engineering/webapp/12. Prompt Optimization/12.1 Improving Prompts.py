import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="12.1 Improving Prompts | GenAI Roadmap",
    page_icon="🔧",
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
    .iteration-badge {
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
st.title("🔧 12.1 Improving Prompts — Systematic Iteration & Ambiguity Removal")
st.markdown(
    """
Welcome to **Lesson 12.1: Improving Prompts**.  
Writing an effective production prompt is rarely a one-shot effort. In professional engineering, prompt creation is an **iterative optimization loop** akin to test-driven software development.

Unrefined prompts suffer from lexical ambiguity, underspecified constraints, and edge-case fragility. By methodically identifying weaknesses and applying structured refinements, we transform unpredictable drafts into hardened, production-grade instructions.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 5-Step Engineering Iteration Loop
# ==============================================================================
st.header("1. The 5-Step Prompt Refinement Cycle", divider="rainbow")

col_it1, col_it2, col_it3, col_it4, col_it5 = st.columns(5)

with col_it1:
    with st.container(border=True):
        st.markdown("#### 1. Baseline Draft")
        st.caption("Initial Hypothesis")
        st.markdown("Write a minimal functional prompt stating the primary task objective.")

with col_it2:
    with st.container(border=True):
        st.markdown("#### 2. Stress Test")
        st.caption("Adversarial Eval")
        st.markdown("Run prompt against 20-50 edge-case test inputs (empty fields, huge text, conflicting claims).")

with col_it3:
    with st.container(border=True):
        st.markdown("#### 3. Root Cause")
        st.caption("Failure Audit")
        st.markdown("Categorize failures: was it lexical ambiguity, missing constraints, or format drifting?")

with col_it4:
    with st.container(border=True):
        st.markdown("#### 4. Targeted Fix")
        st.caption("Hypothesis Update")
        st.markdown("Add explicit negative constraints, few-shot edge examples, or schema bounds.")

with col_it5:
    with st.container(border=True):
        st.markdown("#### 5. Regression Check")
        st.caption("Benchmark Run")
        st.markdown("Verify the fix solved edge cases without degrading performance on standard cases.")

st.divider()

# ==============================================================================
# Section 2: Diagnosing & Fixing Weak Prompts
# ==============================================================================
st.header("2. Catalog of Common Prompt Defects & Targeted Fixes", divider="rainbow")
st.markdown("Recognizing specific failure modes allows surgical, high-leverage prompt revisions:")

defects_df = pd.DataFrame({
    "Prompt Defect": [
        "1. Lexical Ambiguity",
        "2. Unbounded Length",
        "3. Missing Edge-Case Guidance",
        "4. Tone / Format Drift",
        "5. Under-Constrained Logic",
        "6. Context Clutter"
    ],
    "Symptom in Generation": [
        "Model guesses intent; outputs vary wildly across temperature seeds",
        "Runaway verbosity; wastes tokens and spikes API latency",
        "Hallucinates fabricated facts when source data lacks answer",
        "Generates conversational filler, markdown formatting unpredictably",
        "Model takes unwarranted creative liberties with critical data",
        "Model ignores crucial instructions buried inside bloated essays"
    ],
    "Surgical Engineering Fix": [
        "Replace subjective adjectives ('good', 'fast') with quantifiable metrics",
        "Specify exact token caps, sentence limits, or structured bullet counts",
        "Explicit fallback rule: 'If datum is missing, output UNKNOWN'",
        "Provide explicit JSON Schema or strict output format delimiter",
        "Inject explicit negative constraints ('Do NOT assume or infer')",
        "Prune redundant filler; use XML tags to highlight key directives"
    ]
})
st.dataframe(defects_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Evaluation-Driven Prompt Refinement Pipeline", divider="rainbow")
st.markdown("How enterprise teams iterate on prompts using regression test suites:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9,color="white", fontcolor="white"]

        cand_prompt [label="Candidate Prompt\n(Version N)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        test_suite [label="Golden Test Suite\n(50+ Curated Cases)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        eval_runner [label="Automated Eval Runner\n(Asserts + LLM Judge)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        metrics [label="Score Card\n(Accuracy, Latency, Cost)", fillcolor="#EAB308", fontcolor="#000000"]
        prod_release [label="Production Deploy\n(Passes 95% SLA)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        diff_engine [label="Failure Diagnostics\n& Surgical Edit", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        cand_prompt -> test_suite -> eval_runner -> metrics
        metrics -> prod_release [label=" Meets SLA", color="#10B981"]
        metrics -> diff_engine [label=" Fails Regressions", color="#EF4444"]
        diff_engine -> cand_prompt [label=" Updated Prompt V(N+1)", style="dashed"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Prompt Diagnostic & Iteration Workbench
# ==============================================================================
st.header("4. Interactive Prompt Refinement Workbench", divider="rainbow")
st.markdown("Observe the dramatic difference between a naive draft and an iteratively engineered prompt:")

task_selection = st.selectbox(
    "Select Real-World Use Case",
    [
        "Customer Support Ticket Escalation",
        "Executive Financial Summary",
        "Code Refactoring & Bug Fixing"
    ]
)

col_w1, col_w2 = st.columns(2)

with col_w1:
    with st.container(border=True):
        st.subheader("❌ Version 1.0: Naive Prompt")
        if task_selection == "Customer Support Ticket Escalation":
            v1_text = """Classify if this ticket is urgent or not.
Ticket: "Our production cluster in us-east-1 went down 10 minutes ago."
"""
        elif task_selection == "Executive Financial Summary":
            v1_text = """Summarize the company's Q3 financial report nicely.
Report: [10 pages of balance sheets and EBITDA tables]
"""
        else:
            v1_text = """Find bugs in this Python code and fix them.
def calculate(data): return [x*2 for x in data if x > 0]
"""
        st.code(v1_text, language="text")
        st.error("🚨 Weaknesses Identified:")
        st.markdown(
            """
- No definition of what constitutes 'urgent' vs 'standard'
- Output format unspecified (could return a paragraph, a word, or a poem)
- No guidance for ambiguous or missing data
- Subjective qualifiers ('nicely', 'bugs')
"""
        )

with col_w2:
    with st.container(border=True):
        st.subheader("✅ Version 3.0: Iteratively Refined Production Prompt")
        if task_selection == "Customer Support Ticket Escalation":
            v3_text = """<instructions>
Classify incoming customer support tickets into an escalation priority.
Evaluate the ticket enclosed in <ticket> tags against the SLA matrix.
</instructions>

<sla_matrix>
- P0_CRITICAL: Complete production outage, data loss, or active security breach.
- P1_HIGH: Major feature broken for multiple users with no immediate workaround.
- P2_MEDIUM: Isolated bug or performance degradation with an available workaround.
- P3_LOW: General inquiry, cosmetic UI glitch, or feature request.
</sla_matrix>

<constraints>
- If the ticket mentions production downtime in any region, classify strictly as P0_CRITICAL.
- Do NOT include conversational pleasantries.
- Return ONLY valid JSON matching the specified schema.
</constraints>

<ticket>
Our production cluster in us-east-1 went down 10 minutes ago.
</ticket>

<output_format>
{
  "priority": "P0_CRITICAL" | "P1_HIGH" | "P2_MEDIUM" | "P3_LOW",
  "reasoning_summary": str (max 20 words),
  "escalate_to_pagerduty": bool
}
</output_format>"""
        elif task_selection == "Executive Financial Summary":
            v3_text = """<instructions>
Extract executive financial performance metrics from the quarterly report inside <report> tags.
</instructions>

<constraints>
- Quote exact monetary values; do NOT round or extrapolate.
- If a metric is not explicitly stated in the text, assign value null.
- Exactly 3 bullet points for key strategic risks.
</constraints>

<output_format>
{
  "quarter": "Q3",
  "revenue_usd_millions": float,
  "ebitda_usd_millions": float,
  "year_over_year_growth_pct": float,
  "top_risks": [str, str, str]
}
</output_format>"""
        else:
            v3_text = """<instructions>
Perform a security and performance code review on the function inside <code> tags.
</instructions>

<constraints>
- Check specifically for: TypeErrors, NoneType crashes, unbound memory growth, and edge cases (empty list, NaN).
- Output must provide the corrected Python code inside triple backticks and a one-sentence rationale.
</constraints>

<output_format>
```python
# Refactored function with type hints and defensive guard clauses
```
Rationale: [Explanation of fixes]
</output_format>"""
        st.code(v3_text, language="xml")
        st.success("🟢 Production Hardening:")
        st.markdown(
            """
- Explicit categorical rubric (P0 through P3 SLA definition)
- Strict typed JSON schema or bounded output format
- Clear edge-case rules ('If missing, assign null')
- Modular XML boundaries isolating data from instructions
"""
        )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Prompt Regression Testing", divider="rainbow")
st.markdown("Automate prompt evaluation with deterministic unit test suites to protect against regressions:")

st.code(
    '''import json
from typing import List, Dict

# 1. Golden Evaluation Dataset
EVAL_CASES: List[Dict[str, str]] = [
    {
        "input": "All servers in eu-central are returning 500 errors. 10,000 users offline.",
        "expected_priority": "P0_CRITICAL",
        "expected_pagerduty": True
    },
    {
        "input": "Where can I update my billing credit card on file?",
        "expected_priority": "P3_LOW",
        "expected_pagerduty": False
    },
    {
        "input": "Export to CSV button takes 45 seconds to download 100 rows.",
        "expected_priority": "P2_MEDIUM",
        "expected_pagerduty": False
    }
]

def run_prompt_regression_eval(candidate_prompt_template, mock_llm_fn) -> float:
    """
    Executes a candidate prompt against the golden test suite.
    Calculates passing accuracy and flags regression anomalies.
    """
    passed = 0
    total = len(EVAL_CASES)
    
    for case in EVAL_CASES:
        assembled_prompt = candidate_prompt_template.format(ticket_text=case["input"])
        raw_response = mock_llm_fn(assembled_prompt)
        
        try:
            parsed = json.loads(raw_response)
            if (parsed.get("priority") == case["expected_priority"] and
                parsed.get("escalate_to_pagerduty") == case["expected_pagerduty"]):
                passed += 1
            else:
                print(f"Regression on input: '{case['input'][:30]}...' -> Got {parsed}")
        except Exception as e:
            print(f"Parse failure: {e}")
            
    accuracy = (passed / total) * 100.0
    print(f"Evaluation Complete. Accuracy: {accuracy:.1f}% ({passed}/{total})")
    return accuracy
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
    with st.expander("Q1: What is 'Prompt Drift' and how do you monitor for it in production systems?", expanded=True):
        st.markdown(
            """
- **Prompt Drift**: The phenomenon where a prompt that previously performed well begins producing lower-quality or corrupted outputs over time.
- **Root Causes**:
  1. **Upstream Model Updates**: LLM providers roll out silently updated model checkpoints or quantization passes behind the same API model identifier.
  2. **Data Distribution Shift**: Real-world user queries change vocabulary, slang, length, or intent distributions.
- **Production Defense**: Continuous synthetic canary evaluations running daily against a static 'Golden Benchmark' dataset, paired with automated alerts if output schema parse rates drop below 99%.
"""
        )
    with st.expander("Q2: Why is adding negative constraints ('Don't do X') often less reliable than positive demonstrations ('Do Y')?"):
        st.markdown(
            """
Autoregressive transformers predict the most likely continuation based on tokens present in the prompt. When you write *"Do not mention price"*, the token *"price"* is placed directly into the model's attention context, increasing the activation of price-related semantic representations. Providing **positive demonstrations** showing what the output should look like, or phrasing rules constructively (*"Focus exclusively on technical specs"*), avoids priming the model with forbidden tokens.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'When a prompt gives bad output, should you immediately write a 5-paragraph explanation of what not to do?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, write more rules explaining every mistake the model made."*
- **Winning Response**: *"No. Bloating prompts with long paragraphs of negative rules increases prompt token cost, inflates latency, and dilutes the model's attention on the core task. The best optimization strategy is surgical: first identify if the ambiguity stems from missing definitions or bad formatting, then add 1-2 targeted few-shot input/output examples demonstrating the desired behavior. Examples teach transformers far more effectively than abstract rules."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Iterative Prompting**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Systematically refining prompts through test cases, error analysis, and regression benchmarking rather than ad-hoc guessing.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Golden Benchmark Suite**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A curated, version-controlled set of test inputs and expected outputs used to validate prompt revisions and detect regressions.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt Drift**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Performance degradation caused by silent provider model checkpoint updates or shifting user input distributions.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 12.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Iterate with Test Cases**: Never optimize prompts without a fixed evaluation dataset.
2. **Remove Ambiguity**: Replace subjective adjectives with quantifiable criteria and categorical rubrics.
3. **Prefer Positive Framing**: Use few-shot demonstrations over paragraphs of negative constraints.
4. **Guard Against Regressions**: Test new prompt versions against previous successful cases before deploying.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the 5-step engineering prompt iteration cycle.", value=True, key="chk_12_1_1")
    st.checkbox("I know how to identify and diagnose the 6 common prompt defects.", value=True, key="chk_12_1_2")
    st.checkbox("I can build a golden evaluation test suite in Python.", value=True, key="chk_12_1_3")
    st.checkbox("I understand why prompt drift occurs and how to detect it.", value=True, key="chk_12_1_4")
