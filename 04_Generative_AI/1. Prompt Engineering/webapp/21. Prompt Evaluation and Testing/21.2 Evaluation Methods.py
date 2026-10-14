import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="21.2 Evaluation Methods | GenAI Roadmap",
    page_icon="🧪",
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
    .method-badge {
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
st.title("🧪 21.2 Evaluation Methods — Deterministic Rules, Human Audits & LLM-as-a-Judge")
st.markdown(
    """
Welcome to **Lesson 21.2: Evaluation Methods**.  
How do you measure whether a 500-word generated explanation is "good"? For simple classification, exact match works; but for complex summaries, reasoning, and code generation, evaluation requires a hybrid methodology.

Modern enterprise systems combine three complementary evaluation layers: **Deterministic Rule-Based Assertions** (fast, cheap, objective), **Human-in-the-Loop Expert Audits** (high ground truth, low throughput), and **LLM-as-a-Judge** (scalable semantic evaluation using frontier models).
"""
)

st.divider()

# ==============================================================================
# Section 1: The Triad of Evaluation Methodologies
# ==============================================================================
st.header("1. The Triad of Evaluation Methodologies", divider="rainbow")
st.markdown("Comparing deterministic, human, and model-based evaluation paradigms:")

triad_df = pd.DataFrame({
    "Evaluation Methodology": [
        "1. Deterministic Rule-Based Asserts",
        "2. LLM-as-a-Judge (Automated Semantic)",
        "3. Human Expert Evaluation"
    ],
    "How It Works": [
        "Python code checks exact string matches, regex patterns, JSON schema validity, and numerical ranges",
        "A frontier model (e.g. GPT-4o, Claude 3.5) grades candidate outputs against a detailed rubric",
        "Domain specialists (doctors, lawyers, engineers) manually review and score sample outputs"
    ],
    "Strengths": [
        "100% deterministic, zero API cost, executes in microseconds, perfect for CI/CD unit tests",
        "Scalable, highly correlated with human preference, evaluates nuance, tone, and reasoning quality",
        "Gold-standard ground truth; detects subtle domain errors that AI evaluators miss"
    ],
    "Limitations": [
        "Blind to semantic meaning, style, or factual plausibility beyond strict regex rules",
        "Subject to evaluator biases (length bias, position bias, self-enhancement bias); incurs API cost",
        "Extremely slow, expensive ($50-$200/hr), does not scale to thousands of daily regression runs"
    ],
    "Ideal Role in Pipeline": [
        "First-line CI/CD gatekeeper: catches schema breaks, syntax crashes, and basic regressions",
        "Automated continuous evaluation: grades quality, helpfulness, and faithfulness across 500+ test cases",
        "Periodic calibration: establishes golden datasets and audits LLM-as-a-judge alignment quarterly"
    ]
})
st.dataframe(triad_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: LLM-as-a-Judge: Best Practices & Bias Mitigation
# ==============================================================================
st.header("2. LLM-as-a-Judge: Biases & Mitigation Strategies", divider="rainbow")

col_jb1, col_jb2 = st.columns(2)

with col_jb1:
    with st.container(border=True):
        st.subheader("⚠️ Known Evaluator Biases")
        st.markdown(
            """
1. **Length Bias (Verbosity):** Judge models naturally assign higher ratings to long, verbose answers even if they contain redundant fluff.
2. **Position Bias:** In pairwise comparison (*"Which is better: Model A or Model B?"*), judges overwhelmingly favor whichever response is presented first.
3. **Self-Enhancement Bias:** LLMs tend to favor generations produced by their own model family over competitor models.
"""
        )

with col_jb2:
    with st.container(border=True):
        st.subheader("🛡️ Production Mitigation Strategies")
        st.markdown(
            """
1. **Position Swapping:** In pairwise tests, evaluate both `(A, B)` and `(B, A)` and take the average or enforce a tie if inconsistent.
2. **Explicit Conciseness Penalties:** Instruct the judge to dock points for unnecessary wordiness or filler.
3. **Few-Shot Anchored Rubrics:** Provide concrete few-shot examples demonstrating exact score boundaries (e.g. what an exact 'Score 3' looks like vs 'Score 5').
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Hybrid Evaluation Funnel", divider="rainbow")
st.markdown("How enterprise evaluation routes candidate outputs through deterministic and semantic gates:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        cand [label="Candidate LLM Generation", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        gate1 [label="Gate 1: Code Asserts\n(json.loads, regex, types)", fillcolor="#EAB308", fontcolor="#000000"]
        gate2 [label="Gate 2: LLM-as-a-Judge\n(Rubric Scoring 1-5)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        human_sample [label="Gate 3: Human Audit\n(Sample 5% of runs)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        deploy [label="Deployment Approved", fillcolor="#10B981", fontcolor="#FFFFFF"]
        reject [label="Rejected / Blocked", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        cand -> gate1
        gate1 -> reject [label=" Syntax Error", color="#EF4444"]
        gate1 -> gate2 [label=" Valid Syntax", color="#10B981"]
        gate2 -> reject [label=" Score < 4.0", color="#EF4444"]
        gate2 -> deploy [label=" High Score", color="#10B981"]
        deploy -> human_sample [style="dashed"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive LLM-as-a-Judge Workbench
# ==============================================================================
st.header("4. Interactive LLM-as-a-Judge Workbench", divider="rainbow")
st.markdown("Test an automated LLM judge grading a candidate customer support response:")

eval_metric = st.selectbox(
    "Select Evaluation Dimension",
    ["Faithfulness (Strict adherence to reference doc)", "Professional Empathy & Clarity", "Conciseness (No filler)"]
)

candidate_response_type = st.radio(
    "Choose Candidate Response to Grade",
    [
        "Candidate A: Clear, grounded, concise response",
        "Candidate B: Overly verbose with mild hallucinations",
        "Candidate C: Rude, unhelpful, or completely inaccurate"
    ]
)

col_jw1, col_jw2 = st.columns([1, 1])

with col_jw1:
    st.subheader("📋 Evaluation Prompt Sent to Judge LLM")
    judge_prompt_template = f"""<system>
You are an impartial Expert Quality Evaluator. Grade the candidate response on a 1-5 integer scale.
Dimension: {eval_metric}
Rubric:
- 5 (Exceptional): Perfect compliance, highly accurate, zero errors.
- 4 (Good): Factually correct with minor phrasing suboptimality.
- 3 (Mediocre): Partially accurate, but omits key details or includes slight fluff.
- 2 (Poor): Significant inaccuracies, irrelevant details, or policy violations.
- 1 (Unacceptable): Harmful, completely incorrect, or unreadable.
Output format: JSON with "score" (int) and "rationale" (str).
</system>"""
    st.code(judge_prompt_template, language="xml")

with col_jw2:
    st.subheader("⚖️ Simulated Judge Ruling")
    if "Candidate A" in candidate_response_type:
        st.success("🏆 Judge Verdict: SCORE 5 / 5 (PASSED)")
        st.json({
            "dimension": eval_metric,
            "score": 5,
            "passed": True,
            "rationale": "The candidate strictly answered all user points, cited documentation accurately, and contained zero conversational filler."
        })
    elif "Candidate B" in candidate_response_type:
        st.warning("⚠️ Judge Verdict: SCORE 3 / 5 (BORDERLINE)")
        st.json({
            "dimension": eval_metric,
            "score": 3,
            "passed": False,
            "rationale": "The candidate answered the primary question but included 150 words of unnecessary fluff and stated an unsubstantiated policy guarantee."
        })
    else:
        st.error("🚨 Judge Verdict: SCORE 1 / 5 (FAILED)")
        st.json({
            "dimension": eval_metric,
            "score": 1,
            "passed": False,
            "rationale": "The response was abrasive, failed to address the user request, and hallucinated inaccurate technical directives."
        })

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: LLM-as-a-Judge Evaluator", divider="rainbow")
st.markdown("Python class implementing structured judge scoring with Pydantic validation:")

st.code(
    '''from pydantic import BaseModel, Field
import json
from typing import Dict, Any

class JudgeEvaluationResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Integer rating from 1 to 5")
    reasoning: str = Field(description="Step-by-step justification based on the rubric")
    policy_compliant: bool = Field(description="Whether output violates safety constraints")

JUDGE_SYSTEM_PROMPT = """You are an expert AI Benchmark Judge.
Evaluate the candidate response against the user prompt and reference ground truth.
Grade strictly on a 1-5 scale. Output ONLY valid JSON matching the schema:
{
  "score": int (1-5),
  "reasoning": str,
  "policy_compliant": bool
}
"""

def evaluate_with_llm_judge(user_prompt: str, candidate_output: str, reference_text: str, judge_llm_fn) -> JudgeEvaluationResult:
    eval_input = f"""<user_prompt>
{user_prompt}
</user_prompt>

<reference_ground_truth>
{reference_text}
</reference_ground_truth>

<candidate_output>
{candidate_output}
</candidate_output>
"""
    raw_judge_response = judge_llm_fn(JUDGE_SYSTEM_PROMPT, eval_input)
    parsed_result = JudgeEvaluationResult.model_validate_json(raw_judge_response)
    return parsed_result

# Example execution demonstration
print("LLM-as-a-Judge evaluator initialized with Pydantic schema validation.")
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
    with st.expander("Q1: What is 'Self-Consistency / Position Swapping' in pairwise LLM evaluation?", expanded=True):
        st.markdown(
            """
When comparing two models (Model A vs Model B), judges suffer from **Position Bias**—they vote for Candidate 1 up to 65% of the time simply because it appears first in the prompt.
**Position Swapping Methodology**:
1. Run Evaluation 1: Present `[Option 1: Model A, Option 2: Model B]`.
2. Run Evaluation 2: Swap order: `[Option 1: Model B, Option 2: Model A]`.
3. If the judge votes for Model A in both runs, Model A is declared the definitive winner. If the judge flips its vote based purely on position, the comparison is marked as an invalid tie.
"""
        )
    with st.expander("Q2: How do you calibrate an LLM-as-a-Judge against human domain experts?"):
        st.markdown(
            """
Measure the **Cohen's Kappa** or **Spearman Rank Correlation** between judge scores and human scores across a 200-sample test set:
1. Have 3 human experts independently grade 200 outputs.
2. Have the LLM judge grade the same 200 outputs.
3. Calculate correlation ($\rho \ge 0.80$ is typically target threshold).
4. Iterate on the judge prompt rubric and few-shot calibration examples until correlation aligns with human consensus.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we use a smaller, cheaper 7B model to judge outputs from a frontier 70B model?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, using a 7B model as judge saves money and is fast."*
- **Winning Response**: *"No. An evaluation model cannot reliably judge reasoning that exceeds its own cognitive capabilities. If a 7B model evaluates nuanced legal or mathematical outputs generated by GPT-4o or Claude 3.5, the smaller model will fail to detect subtle logical fallacies and hallucinate false evaluations. The judge model must ALWAYS be equal to or more capable than the model being evaluated."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**LLM-as-a-Judge**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Using a capable foundation model to evaluate and score the quality of candidate LLM responses against an explicit rubric.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Position Bias**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The systematic tendency of LLM judges to prefer whichever candidate response is placed first in a pairwise comparison prompt.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Few-Shot Rubric Anchors**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Providing concrete input-output examples in the judge prompt that demonstrate exact boundary cases for scores (1 vs 3 vs 5).")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 21.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Triad Architecture**: Combine code asserts (fast/cheap), LLM judges (scalable), and human audits (gold truth).
2. **Mitigate Evaluator Biases**: Use position swapping and conciseness penalties to overcome length/position bias.
3. **Judge Model Sizing**: The judge model must be at least as capable as the generator model.
4. **Statistical Calibration**: Measure Spearman correlation against human experts to validate rubrics.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know when to use rule-based asserts vs. LLM-as-a-judge vs. human audits.", value=True, key="chk_21_2_1")
    st.checkbox("I understand the 3 primary biases of LLM judges (length, position, self-enhancement).", value=True, key="chk_21_2_2")
    st.checkbox("I can implement position swapping to eliminate pairwise comparison bias.", value=True, key="chk_21_2_3")
    st.checkbox("I can write a structured LLM-as-a-Judge evaluation harness in Python.", value=True, key="chk_21_2_4")
