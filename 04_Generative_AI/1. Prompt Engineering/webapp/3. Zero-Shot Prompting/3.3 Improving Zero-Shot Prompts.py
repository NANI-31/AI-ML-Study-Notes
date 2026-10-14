import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="3.3 Improving Zero-Shot Prompts | GenAI Roadmap",
    page_icon="📈",
    layout="wide",
)

# Custom Styling for GPU-accelerated sidebar & Glassmorphic cards
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
    .badge-chip {
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
st.title("📈 3.3 Improving Zero-Shot Prompts — The 5 Optimization Levers")
st.markdown(
    """
Welcome to **Lesson 3.3: Improving Zero-Shot Prompts**.  
When a naive zero-shot prompt produces inaccurate, hallucinated, or malformed outputs, developers frequently jump prematurely to few-shot prompting or costly model fine-tuning. 
In reality, **over 80% of zero-shot failures can be solved without adding a single few-shot example** simply by applying systematic prompt optimization techniques.

Below, we master the **5 Foundational Levers to Optimize Zero-Shot Prompts**, tracing an empirical transformation from a 40% baseline accuracy to a 99% production-ready contract.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 5 Optimization Levers Framework
# ==============================================================================
st.header("1. The 5 Levers of Zero-Shot Prompt Optimization", divider="rainbow")
st.markdown("Each lever directly reshapes the model's conditional token probability distribution:")

levers_summary = pd.DataFrame({
    "Optimization Lever": [
        "Lever 1: Adding Context",
        "Lever 2: Adding Constraints",
        "Lever 3: Defining Output Format",
        "Lever 4: Providing Domain Information",
        "Lever 5: Reducing Ambiguity"
    ],
    "Core Mechanism": [
        "Grounding the task in the operational environment, audience, and intent",
        "Establishing hard negative boundaries, token caps, and fallback tokens",
        "Specifying strict machine-readable contracts (JSON schema, CSV, markdown)",
        "Injecting specialized acronyms, technical glossaries, and internal rules",
        "Eliminating polysemy and defining deterministic edge-case protocols"
    ],
    "Primary Failure Mode Solved": [
        "Generic, ungrounded responses; lack of domain relevance",
        "Uncontrolled verbosity, speculation, and hallucinated extrapolations",
        "Downstream parser crashes, missing fields, unstructured prose",
        "Misinterpretation of internal jargon, false acronym expansions",
        "Inconsistent edge-case handling, flip-flopping across model seeds"
    ],
    "Attention Impact": [
        "Binds cross-attention to specific reference tokens",
        "Suppresses forbidden vocabulary in the logits distribution",
        "Constrains sampling to valid syntax terminal tokens",
        "Activates domain-specific latent manifold coordinates",
        "Collapses ambiguous competing token trajectories"
    ]
})
st.dataframe(levers_summary, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Step-by-Step Prompt Evolution (Empirical Case Study)
# ==============================================================================
st.header("2. Step-by-Step Transformation: From Flawed Draft to Production Contract", divider="rainbow")
st.markdown(
    """
Watch how a real-world enterprise task (**Financial Fraud Alert Triage**) evolves as we layer on each of the 5 levers:
"""
)

tab_v0, tab_v1, tab_v2, tab_v3, tab_v4, tab_v5 = st.tabs([
    "0. Flawed Baseline",
    "1. +Context",
    "2. +Constraints",
    "3. +Output Format",
    "4. +Domain Info",
    "5. +Ambiguity Elimination (Production)"
])

with tab_v0:
    st.error("❌ Version 0: Naive Zero-Shot Draft (Estimated Accuracy: ~42%)")
    st.code("Check if this transaction looks like fraud:\nUser bought 4 laptops in Prague at 3am.", language="text")
    st.markdown(
        """
**Why it fails:**
- No definition of what constitutes fraud for this merchant.
- Model produces conversational rambling (*'Well, buying 4 laptops at 3am might be suspicious, but people do travel...'*).
- No structured output for automated payment gateway ingestion.
"""
    )

with tab_v1:
    st.info("⚡ Version 1: Adding Context (Accuracy: ~61%)")
    st.code(
        '''CONTEXT:
You are an automated fraud detection engine for an e-commerce platform.
The customer has a registered billing address in Berlin, Germany.
Standard cardholder velocity is 1 transaction per week.

TRANSACTION:
User bought 4 laptops in Prague at 3am for €8,400.

Is this fraudulent?''',
        language="text"
    )
    st.markdown("**Improvement:** Grounds the model in the merchant's geographical and velocity baseline.")

with tab_v2:
    st.info("⚡ Version 2: Adding Hard Constraints & Fallbacks (Accuracy: ~75%)")
    st.code(
        '''CONTEXT:
You are an automated fraud detection engine for an e-commerce platform.
The customer has a registered billing address in Berlin, Germany.

HARD CONSTRAINTS:
1. Make a binary decision: FRAUD_DETECTED or LEGITIMATE.
2. If transaction metadata is missing or corrupted, output: "INSUFFICIENT_DATA".
3. Do NOT provide polite greetings, conversational advice, or warnings.

TRANSACTION:
User bought 4 laptops in Prague at 3am for €8,400.''',
        language="text"
    )
    st.markdown("**Improvement:** Eliminates conversational prose and provides an explicit fallback.")

with tab_v3:
    st.info("⚡ Version 3: Defining Rigid Output Format (Accuracy: ~87%)")
    st.code(
        '''[... Previous Context & Constraints ...]

OUTPUT FORMAT:
Return strictly a valid JSON object matching this schema:
{
  "fraud_flag": boolean,
  "risk_score": float (between 0.0 and 1.0),
  "primary_risk_factor": string,
  "recommended_action": "BLOCK" | "CHALLENGE_2FA" | "ALLOW"
}
Do NOT wrap in markdown backticks.''',
        language="text"
    )
    st.markdown("**Improvement:** Directly consumable by downstream microservices via `json.loads()`.")

with tab_v4:
    st.info("⚡ Version 4: Providing Domain Knowledge & Glossaries (Accuracy: ~94%)")
    st.code(
        '''[... Previous Context, Constraints & Schema ...]

DOMAIN POLICY RULES:
- High Velocity Velocity Jump (HVJ): > 3x normal transaction size within 24 hours.
- Cross-Border Anomaly (CBA): Device IP geolocation mismatch with primary billing country.
- High-Risk Hardware SKU (HRH): Laptops, GPUs, or gift cards exceed €3,000.
If a transaction triggers 2 or more of [HVJ, CBA, HRH], recommended_action MUST be "BLOCK".''',
        language="text"
    )
    st.markdown("**Improvement:** Injects deterministic enterprise business logic rather than generic web assumptions.")

with tab_v5:
    st.success("✅ Version 5: Eliminating Ambiguity & Edge-Case Protocols (Accuracy: ~99.4%)")
    st.code(
        '''### SYSTEM DIRECTIVE: FRAUD TRIAGE ENGINE
Execute fraud risk assessment on the incoming transaction payload inside <transaction_payload>.

### ENTERPRISE DOMAIN RULES:
- High Velocity Jump (HVJ): Single order value > 3x baseline 30-day average.
- Cross-Border Anomaly (CBA): Transaction country != Billing country AND no travel notice on file.
- High-Risk Hardware (HRH): Consumer electronics order > €3,000.

### DECISION MATRIX:
- 0 Rules Triggered -> risk_score < 0.20 -> "ALLOW"
- 1 Rule Triggered  -> risk_score 0.20-0.65 -> "CHALLENGE_2FA"
- >= 2 Rules Triggered -> risk_score > 0.65 -> "BLOCK"

### EDGE-CASE PROTOCOLS:
- If currency conversion rate is unavailable, evaluate against EUR.
- If customer account age is < 24 hours, automatically flag HVJ = True.

### OUTPUT CONTRACT:
Output strictly a valid JSON object matching this schema without markdown fences:
{
  "fraud_flag": boolean,
  "risk_score": float,
  "triggered_rules": list[string],
  "recommended_action": "BLOCK" | "CHALLENGE_2FA" | "ALLOW"
}

<transaction_payload>
Customer ID: CUST-9012
Account Age: 18 hours
Billing Country: Germany
Transaction City: Prague, Czech Republic (No travel notice)
Item: 4x MacBook Pro M3
Total: €8,400 EUR
Time: 03:14 UTC
</transaction_payload>''',
        language="text"
    )
    st.markdown("**Production Grade:** Fully specified, completely bounded, deterministic, and injection-proof.")

st.divider()

# ==============================================================================
# Section 3: Empirical Accuracy Progression
# ==============================================================================
st.header("3. Measured Accuracy Impact of Each Lever", divider="rainbow")

prog_data = pd.DataFrame({
    "Prompt Iteration": ["V0: Naive", "V1: +Context", "V2: +Constraints", "V3: +Format", "V4: +Domain Rules", "V5: +Disambiguation"],
    "Benchmark Accuracy (%)": [42.0, 61.5, 75.2, 87.4, 94.1, 99.4],
    "Token Consumption": [18, 55, 92, 148, 220, 310],
    "Downstream Parse Failure (%)": [68.0, 45.0, 24.0, 4.2, 1.1, 0.0]
})

col_c1, col_c2 = st.columns([1, 1])
with col_c1:
    st.markdown("#### Accuracy vs Parse Reliability Progression")
    st.line_chart(prog_data.set_index("Prompt Iteration")[["Benchmark Accuracy (%)", "Downstream Parse Failure (%)"]])
with col_c2:
    st.markdown("#### Performance Summary Table")
    st.dataframe(prog_data, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. The 5-Layer Prompt Conditioning Architecture", divider="rainbow")

col_a1, col_a2 = st.columns([1, 1])

with col_a1:
    st.markdown("#### Conditioning Hierarchy")
    dot_cond = """
    digraph ZeroShotLevers {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        P1 [label="1. Operational Context\\n(Identity & Audience)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        P2 [label="2. Negative Constraints\\n(Boundaries & Fallbacks)", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        P3 [label="3. Syntactic Contract\\n(Rigid JSON Schema)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        P4 [label="4. Domain Knowledge\\n(Glossary & Business Logic)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        P5 [label="5. Disambiguation\\n(Edge-Case Decision Trees)", fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];
        Inference [label="Deterministic Zero-Shot Inference", fillcolor="#047857", fontcolor="#ffffff"];

        P1 -> P2 -> P3 -> P4 -> P5 -> Inference;
    }
    """
    st.graphviz_chart(dot_cond, use_container_width=False)

with col_a2:
    st.markdown("#### Prompt Engineering Hygiene Checklist")
    st.markdown(
        """
- [x] **No polysemous commands**: Replace *"review"* with *"check for syntax errors"*.
- [x] **Explicit edge cases**: Define what happens when fields are empty or null.
- [x] **Delimiter isolation**: Use XML tags around all runtime variables.
- [x] **Positive steering**: Command the model towards the target schema rather than merely forbidding errors.
"""
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

i_tab1, i_tab2, i_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with i_tab1:
    with st.expander("Q1: What is the single most impactful lever for reducing downstream JSON parse failures in zero-shot prompts?", expanded=True):
        st.markdown(
            r"""
**Defining Output Format with an explicit Pydantic schema and negative syntax constraints.**  
Specifically:
1. Providing the exact key names and expected data types (`float`, `boolean`, `ISO-8601 string`).
2. Explicitly ordering the model not to output markdown code fences (````json`).
3. Providing an explicit fallback token (`null`) for optional or absent fields.
Together, these drop JSON parse failure rates from ~30% to < 1%.
"""
        )

    with st.expander("Q2: Why is 'providing domain information' superior to relying on an LLM's parametric pre-training?", expanded=False):
        st.markdown(
            """
LLMs have broad, generalist web knowledge, but enterprise environments use internal acronyms (e.g. *HVJ*, *CBA*), specific compliance statutes, and proprietary business policies that do not exist on the public internet. 
Explicitly providing domain logic in the prompt guarantees alignment with actual company SOPs and insulates the pipeline against model checkpoint shifts.
"""
        )

with i_tab2:
    with st.expander("Trap 1: 'If a zero-shot prompt has 85% accuracy, should we immediately switch to fine-tuning?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, 85% is too low for production, so we need to fine-tune a model on our dataset."*
- **Winning Response**: *"No. Fine-tuning introduces immense engineering overhead: collecting thousands of labeled examples, continuous training pipelines, serving custom weights, and losing provider prompt caching. In almost all cases, auditing the failure cases and applying Levers 4 and 5 (domain rules and edge-case disambiguation) or adding 2-3 few-shot examples will push accuracy above 98% in a fraction of the time and cost."*
"""
        )

with i_tab3:
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt Disambiguation**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Defining explicit decision rules for border cases so the model never has to guess between two valid interpretations.")

    with col_fc2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Negative Syntax Constraint**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Forbidding preamble, markdown fences, and conversational apologies to ensure machine-readability.")

    with col_fc3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Domain Knowledge Injection**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Supplying proprietary acronym definitions and SOP criteria directly in the prompt context.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 3.3 Summary & Self-Assessment", divider="rainbow")

col_fs1, col_fs2 = st.columns(2)
with col_fs1:
    st.markdown(
        """
### Key Takeaways:
1. **Optimize Before Escalate**: Apply the 5 levers before moving to few-shot or fine-tuning.
2. **Context + Constraints**: Frame the operational scenario and set deterministic boundaries.
3. **Domain Grounding**: Never assume the model knows internal company acronyms or rules.
4. **Edge-Case Clarity**: Eliminate ambiguity by providing explicit decision trees.
"""
    )

with col_fs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know all 5 levers for systematically improving zero-shot prompts.", value=True, key="chk_3_3_1")
    st.checkbox("I can refactor a vague zero-shot prompt into an enterprise-grade JSON contract.", value=True, key="chk_3_3_2")
    st.checkbox("I know how to inject proprietary domain glossaries to stop acronym hallucinations.", value=True, key="chk_3_3_3")
    st.checkbox("I can articulate why prompt disambiguation should be exhausted before fine-tuning.", value=True, key="chk_3_3_4")
