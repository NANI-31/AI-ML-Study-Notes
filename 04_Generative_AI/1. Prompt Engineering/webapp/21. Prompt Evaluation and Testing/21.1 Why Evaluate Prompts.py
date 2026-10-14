import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="21.1 Why Evaluate Prompts | GenAI Roadmap",
    page_icon="⚖️",
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
    .eval-badge {
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
st.title("⚖️ 21.1 Why Evaluate Prompts — From 'Vibes' to Empirical Engineering")
st.markdown(
    """
Welcome to **Lesson 21.1: Why Evaluate Prompts**.  
In early prototype development, engineers often rely on "vibe-based evaluations"—tweaking a prompt, testing it on 2 or 3 manual queries in the playground, and assuming it works if the output "feels right".

In enterprise software engineering, this approach is disastrous. Changing a single word in a prompt might fix one specific customer issue while silently **breaking 20% of previously working edge cases**. Prompt evaluation is the discipline of applying **rigorous, reproducible, and automated testing** to generative AI systems.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Failure of "Vibe-Based" Prompting
# ==============================================================================
st.header("1. The Failure of 'Vibe-Based' Prompting", divider="rainbow")

col_vb1, col_vb2 = st.columns(2)

with col_vb1:
    with st.container(border=True):
        st.subheader("❌ Vibe-Based Prompting (Ad-Hoc)")
        st.markdown(
            """
- **Test Set Size:** 2-3 subjective examples manually tested in an interactive chat UI.
- **Evaluation Criteria:** "Looks reasonable to me."
- **Regression Detection:** Zero. Changes are deployed without checking historical failure modes.
- **Result:** Constant production fire-drills, unexpected customer hallucinations, and unstable API outputs.
"""
        )

with col_vb2:
    with st.container(border=True):
        st.subheader("✅ Empirical Prompt Engineering (Scientific)")
        st.markdown(
            """
- **Test Set Size:** 100+ version-controlled "Golden Benchmark" test inputs representing real edge cases.
- **Evaluation Criteria:** Deterministic code assertions + LLM-as-a-Judge semantic rubrics.
- **Regression Detection:** Automated CI/CD test runners block deployment if accuracy drops even 1%.
- **Result:** Auditable reliability, predictable performance, and confident prompt iteration.
"""
        )

st.divider()

# ==============================================================================
# Section 2: The 6 Core Quality Dimensions of Enterprise Evaluation
# ==============================================================================
st.header("2. The 6 Core Quality Dimensions of Enterprise Evaluation", divider="rainbow")
st.markdown("Evaluating prompts across diverse, orthogonal axes of performance:")

dimensions_df = pd.DataFrame({
    "Quality Dimension": [
        "1. Factual Accuracy",
        "2. Semantic Relevance",
        "3. Output Consistency",
        "4. Schema Reliability",
        "5. Policy Safety & Guardrails",
        "6. Latency & Token Efficiency"
    ],
    "What It Measures": [
        "Are the extracted numbers, dates, and domain conclusions factually true?",
        "Does the answer directly address the user's intent without drifting into tangential filler?",
        "Does the model produce identical conclusions across diverse random temperature seeds?",
        "Does the output strictly adhere to required JSON, XML, or Markdown table schemas without parsing errors?",
        "Does the prompt resist adversarial jailbreaks, prevent PII leakage, and uphold brand rules?",
        "Does the prompt keep Time-to-First-Token (TTFT) low and maintain cost within SLA budgets?"
    ],
    "Automated Measurement Technique": [
        "Exact Match (EM), F1-Score, Ground-Truth Semantic Cosine Similarity",
        "LLM-as-a-Judge relevance scoring (1-5 Likert scale)",
        "Monte Carlo variance scoring: running 5 runs per prompt at temp=0.7",
        "Pydantic `model_validate_json()` success rate (Target: 99.9%)",
        "Automated Red-Teaming attack pass rates (Garak / PyRIT)",
        "Profiler benchmarking: TTFT (ms), Output Tokens, Total API Cost ($)"
    ]
})
st.dataframe(dimensions_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Automated Prompt Regression Test Suite", divider="rainbow")
st.markdown("How enterprise prompt revisions are gated through continuous integration:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        cand_prompt [label="Candidate Prompt (V2.1)\n(Git Pull Request)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        golden_set [label="Golden Benchmark Suite\n(150 Curated Edge Cases)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        eval_runner [label="CI/CD Eval Runner\n(Asserts + LLM Judge)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        diff_report [label="Regression Scorecard\n(Accuracy: +4%, Regressions: 0)", fillcolor="#EAB308", fontcolor="#000000"]
        deploy [label="Production Release\n(Passes 98% Threshold)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        reject [label="PR Blocked / Rejected\n(Flagged Regression on Case #42)", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        cand_prompt -> eval_runner
        golden_set -> eval_runner
        eval_runner -> diff_report
        diff_report -> deploy [label=" Meets SLA", color="#10B981"]
        diff_report -> reject [label=" Regression Found", color="#EF4444"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Prompt Evaluation Diagnostic Workbench
# ==============================================================================
st.header("4. Interactive Prompt Regression Diagnostic Workbench", divider="rainbow")
st.markdown("Compare Version 1.0 (Baseline) vs Version 2.0 (Candidate) across an evaluation test suite:")

eval_data = [
    {"Test ID": "Case #01", "Input Scenario": "Standard invoice with single line item", "V1 Score": "PASS", "V2 Score": "PASS", "Notes": "Both versions parse simple inputs cleanly"},
    {"Test ID": "Case #02", "Input Scenario": "Invoice with $0 subtotal and 100% promo code", "V1 Score": "FAIL (Division by Zero)", "V2 Score": "PASS", "Notes": "V2 handles zero amounts gracefully"},
    {"Test ID": "Case #03", "Input Scenario": "Missing vendor tax ID (Nullable field)", "V1 Score": "FAIL (Hallucinated 999-99)", "V2 Score": "PASS", "Notes": "V2 correctly sets vendor_tax_id=null"},
    {"Test ID": "Case #04", "Input Scenario": "French receipt with Euro currency (14,50 EUR)", "V1 Score": "FAIL (Treated comma as text)", "V2 Score": "PASS", "Notes": "V2 normalizes European decimal commas"},
    {"Test ID": "Case #05", "Input Scenario": "Adversarial prompt injection inside invoice notes", "V1 Score": "FAIL (Prompt Leaked)", "V2 Score": "PASS", "Notes": "V2 XML isolation neutralized injection"}
]

col_ew1, col_ew2 = st.columns([1, 1])

with col_ew1:
    st.subheader("📊 Regression Test Matrix")
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df[["Test ID", "Input Scenario", "V1 Score", "V2 Score"]], use_container_width=True, hide_index=True)

with col_ew2:
    st.subheader("📈 Quantitative Scorecard Comparison")
    m1, m2 = st.columns(2)
    m1.metric("Version 1.0 Accuracy", "20.0%", delta="1 / 5 Passed")
    m2.metric("Version 2.0 Accuracy", "100.0%", delta="+80% Improvement")
    
    st.success("🏆 Deployment Recommendation: APPROVE CANDIDATE (V2.0)")
    st.markdown(
        """
- **Schema Parse Rate:** Improved from 40% to 100%
- **Edge-Case Resilience:** Handled European decimal formatting and zero-dollar carts
- **Security Audit:** Neutralized injection payloads in Case #05
"""
    )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Golden Benchmark Evaluator", divider="rainbow")
st.markdown("Python script running an automated prompt evaluation harness over a golden test suite:")

st.code(
    '''import json
from typing import List, Dict, Any

# 1. Golden Evaluation Dataset
GOLDEN_TEST_SUITE: List[Dict[str, Any]] = [
    {
        "input": "Patient has BP 120/80 and reports no pain. Follow up in 1 month.",
        "expected_triage": "ROUTINE",
        "expected_flag": False
    },
    {
        "input": "Patient SpO2 dropped to 86%, severe dyspnea, HR 135 bpm.",
        "expected_triage": "CRITICAL",
        "expected_flag": True
    },
    {
        "input": "Patient reports mild cough for 2 days. Temp 98.6F.",
        "expected_triage": "ROUTINE",
        "expected_flag": False
    }
]

def evaluate_prompt_candidate(prompt_template_fn, mock_llm_fn) -> Dict[str, Any]:
    """
    Executes candidate prompt against golden test suite.
    Calculates overall pass rate and lists specific regression failures.
    """
    total = len(GOLDEN_TEST_SUITE)
    passed = 0
    failures = []
    
    for idx, test_case in enumerate(GOLDEN_TEST_SUITE):
        prompt = prompt_template_fn(test_case["input"])
        raw_output = mock_llm_fn(prompt)
        
        try:
            parsed = json.loads(raw_output)
            matches_triage = parsed.get("triage") == test_case["expected_triage"]
            matches_flag = parsed.get("flag") == test_case["expected_flag"]
            
            if matches_triage and matches_flag:
                passed += 1
            else:
                failures.append({"case_idx": idx, "got": parsed, "expected": test_case})
        except Exception as exc:
            failures.append({"case_idx": idx, "error": str(exc)})
            
    accuracy_pct = (passed / total) * 100.0
    return {
        "accuracy_pct": accuracy_pct,
        "total_cases": total,
        "passed_cases": passed,
        "failures": failures
    }

print("Prompt Evaluation Harness ready for CI/CD regression testing.")
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
    with st.expander("Q1: Why is testing a prompt on 10 examples in a playground completely inadequate for enterprise production?", expanded=True):
        st.markdown(
            """
1. **Sampling Bias**: Humans instinctively test "happy path" inputs that reflect their own mental model of the feature, ignoring real-world edge cases (foreign characters, empty strings, contradictory inputs).
2. **Hidden Regressions**: Prompt optimization is often non-monotonic: tweaking phrasing to fix edge case A frequently degrades performance on baseline case B. Without a regression benchmark covering both baseline and edge cases, regressions escape to production unnoticed.
3. **Statistical Insignificance**: 10 test cases has a margin of error of +/- 30%. A 100+ example benchmark is necessary for statistical confidence.
"""
        )
    with st.expander("Q2: How do you assemble a high-quality 'Golden Dataset' for prompt evaluation?"):
        st.markdown(
            """
A production Golden Dataset is curated from 3 sources:
1. **Production Edge Cases**: Anonymized customer failure logs and edge cases extracted from bug reports.
2. **Synthetic Data Augmentation**: Using a frontier model to generate diverse linguistic variations, typos, and edge cases.
3. **Human Gold Standard Annotations**: Verified by senior domain experts to ensure ground-truth labels are 100% unambiguous.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we evaluate prompts purely on cost and latency without measuring accuracy?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, if a new prompt cuts latency by 50%, we should deploy it immediately."*
- **Winning Response**: *"Never. The fastest, cheapest prompt is one that outputs nothing at all. Aggressive prompt compression or switching to a smaller model to save money frequently triggers subtle accuracy degradations, hallucinations, and schema invalidation. Optimization must always be evaluated as a **Pareto Frontier**: latency and cost improvements are only valid if accuracy stays above the established SLA threshold."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Golden Benchmark Dataset**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A curated, version-controlled set of representative test cases and ground-truth labels used to evaluate prompt revisions in CI/CD.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt Regression**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** When an edit to a prompt fixes a new problem but accidentally causes previously passing test cases to fail.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Pareto Frontier in Prompts**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The optimal balance where latency and token costs are minimized without sacrificing required accuracy and safety thresholds.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 21.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Ditch the Vibes**: Replace subjective playground testing with automated empirical benchmarks.
2. **The 6 Dimensions**: Evaluate accuracy, relevance, consistency, reliability, safety, and efficiency.
3. **Golden Datasets**: Curate version-controlled test suites from production logs and expert labels.
4. **Guard Against Regressions**: Block prompt deployments in CI/CD if benchmark accuracy declines.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why 'vibe-based' prompt testing fails in enterprise production.", value=True, key="chk_21_1_1")
    st.checkbox("I know the 6 core quality dimensions of prompt evaluation.", value=True, key="chk_21_1_2")
    st.checkbox("I can build an automated regression test harness in Python.", value=True, key="chk_21_1_3")
    st.checkbox("I understand how to evaluate trade-offs along the accuracy-cost Pareto frontier.", value=True, key="chk_21_1_4")
