import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="4.2 Designing Examples | GenAI Roadmap",
    page_icon="🎨",
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
st.title("🎨 4.2 Designing Examples — The Anatomy of High-Impact Exemplars")
st.markdown(
    """
Welcome to **Lesson 4.2: Designing Examples**.  
The effectiveness of Few-Shot Prompting depends entirely on the **quality, distribution, and structural integrity of your demonstrative exemplars**. 
Carelessly selected examples inject label skew, prompt injection vulnerabilities, and format drift directly into the LLM's attention heads.

Below, we master the **6 Foundational Dimensions of Exemplar Engineering**: Input-Output mapping, Positive demonstrations, Negative contrastive examples, Lexical diversity, Production representativeness, and Edge-case coverage.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Dimensions of Exemplar Engineering
# ==============================================================================
st.header("1. The 6 Dimensions of Exemplar Engineering", divider="rainbow")
st.markdown("Each dimension optimizes a distinct facet of in-context attention alignment:")

dimensions_df = pd.DataFrame({
    "Exemplar Dimension": [
        "1. Input-Output Structure",
        "2. Positive Exemplars",
        "3. Negative & Counter-Examples",
        "4. Syntactic & Lexical Diversity",
        "5. Production Representativeness",
        "6. Edge-Case & Boundary Coverage"
    ],
    "Architectural Objective": [
        "Enforce rigid delimiter symmetry (Input: -> Output:) across all samples",
        "Anchor target reasoning steps, schema serialization, and tone",
        "Demonstrate how to handle invalid, malicious, or unanswerable queries",
        "Prevent spurious correlation with sentence length or vocabulary cues",
        "Reflect real-world user noise (typos, slang, unstructured OCR artifacts)",
        "Teach the model when to execute deterministic fallback routes"
    ],
    "Common Anti-Pattern": [
        "Asymmetric formatting (using JSON in ex 1, markdown in ex 2)",
        "Showing only easy, trivial examples that fail on real queries",
        "Forbidding behaviors without showing how an error response looks",
        "Using identically structured sentences of identical word length",
        "Using sanitized, pristine academic sentences divorced from reality",
        "Leaving edge cases to chance, inducing catastrophic hallucinations"
    ]
})
st.dataframe(dimensions_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Deep Dive into Each Dimension with Production Code
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1. Input-Output Structure
with st.container(border=True):
    st.subheader("1. Input-Output Symmetrical Formatting")
    st.markdown(
        """
Every exemplar must mirror the exact delimiter structure expected of the final completion. 
If your production query uses `<input>...</input>`, every single few-shot example must use `<input>...</input>`. 
Asymmetry destroys the attention induction head pattern.
"""
    )
    col_io1, col_io2 = st.columns(2)
    with col_io1:
        st.markdown("**❌ Anti-Pattern (Asymmetric Formatting):**")
        st.code(
            '''Ex 1: "User: Hello -> Bot: Hi"
Ex 2: "Query: Buy item | Response: Confirmed"
Target: <user_input>Cancel order</user_input>''',
            language="text"
        )
    with col_io2:
        st.markdown("**✅ Production Standard (Strict Delimiter Symmetry):**")
        st.code(
            '''<example>
<raw_query>Hello</raw_query>
<parsed_intent>GREETING</parsed_intent>
</example>
<example>
<raw_query>Buy item</raw_query>
<parsed_intent>PURCHASE</parsed_intent>
</example>''',
            language="text"
        )

# 2 & 3. Positive vs Negative Exemplars
with st.container(border=True):
    st.subheader("2 & 3. Positive vs Negative & Counter-Examples")
    st.markdown(
        """
- **Positive Exemplars**: Demonstrate the ideal target execution path.
- **Negative / Counter-Exemplars**: Demonstrate how the model should behave when confronted with **invalid inputs, malicious injections, or out-of-scope requests**.
"""
    )
    st.code(
        '''prompt = """
INSTRUCTION:
Extract stock tickers mentioned in financial news snippets.

<example type="positive">
<text>Apple Inc announced earnings while Microsoft gained 2% in premarket trading.</text>
<tickers>["AAPL", "MSFT"]</tickers>
</example>

<example type="negative_spurious_name">
<text>John Apple and Mary Microsoft attended the quarterly real estate seminar.</text>
<tickers>[]</tickers>
<rationale>Personal names matching company brand words must NOT be tagged as tickers.</rationale>
</example>

<example type="edge_case_unsupported">
<text>Bitcoin surged past $80,000 following spot ETF inflows.</text>
<tickers>[]</tickers>
<rationale>Cryptocurrencies are not equities; return empty list.</rationale>
</example>

<target>
<text>CEO Tim Cook spoke with Amazon leadership regarding server procurement.</text>
<tickers>"""''',
        language="python"
    )

# 4 & 5. Diversity & Representativeness
with st.container(border=True):
    st.subheader("4 & 5. Lexical Diversity & Production Representativeness")
    st.markdown(
        """
If every few-shot example starts with *"Please..."* and contains exactly 12 words, the model's attention heads correlate sentence length and conversational openings with the task logic.
- **Vary Sentence Lengths**: Mix short 3-word inputs with multi-sentence paragraphs.
- **Inject Real-World Noise**: Include natural typos (*"calcel subscription"*, *"wont load"*) so the model is robust to imperfect user text.
"""
    )
    st.code(
        '''# Real-World Diverse Exemplar Set for IT Ticket Classification
exemplars = [
    # Short & Slangy (noisy)
    {"ticket": "app crashed again wtf", "category": "APPLICATION_BUG", "urgency": "HIGH"},
    # Verbose & Polite (formal)
    {"ticket": "Good morning IT team, could you kindly grant read access for user ID 401 to the sales database?", "category": "ACCESS_REQUEST", "urgency": "LOW"},
    # Punctuation Heavy & Truncated
    {"ticket": "VPN disconnected (error 0x80070002)... cannot reach internal Jira", "category": "NETWORK_VPN", "urgency": "HIGH"}
]''',
        language="python"
    )

# 6. Edge-Case Coverage
with st.container(border=True):
    st.subheader("6. Edge-Case & Fallback Coverage")
    st.markdown(
        """
Never assume the model knows how to fail gracefully. 
Include an explicit exemplar demonstrating what to output when mandatory data is missing or corrupted.
"""
    )
    st.code(
        '''<example type="edge_case_missing_data">
<invoice_text>"Generic Receipt - Thank you for your business!"</invoice_text>
<parsed_json>
{
  "vendor": null,
  "total_usd": null,
  "invoice_number": null,
  "status": "UNPROCESSABLE_DOCUMENT"
}
</parsed_json>
</example>''',
        language="text"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Exemplar Audit Simulator
# ==============================================================================
st.header("3. Interactive Workbench: Audit a Few-Shot Exemplar Set", divider="rainbow")
st.markdown("Audit a sample candidate exemplar set for the classic **Customer Support Ticket Classifier**:")

audit_scenario = st.selectbox(
    "Choose Exemplar Set to Audit:",
    [
        "Candidate Set A: Naive & Skewed",
        "Candidate Set B: Balanced & Production-Ready"
    ]
)

if audit_scenario.startswith("Candidate Set A"):
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        st.error("🚨 Candidate Set A (Flawed)")
        st.code(
            '''Ex 1: "Love the UI!" -> POSITIVE
Ex 2: "Great checkout speed" -> POSITIVE
Ex 3: "Fast delivery service" -> POSITIVE
Ex 4: "Buggy login page" -> NEGATIVE''',
            language="text"
        )
    with col_a2:
        st.markdown(
            """
### ⚠️ Audit Findings & Violations:
1. **Severe Class Imbalance**: 3 Positive vs 1 Negative (75% positive skew). The model will suffer from majority-class frequency bias and classify ambiguous cases as POSITIVE.
2. **Zero Neutral / Fallback**: No examples of neutral statements or unanswerable gibberish.
3. **Lexical Monotony**: All inputs are short 3-word praises. Zero complex or real-world tickets.
"""
        )
else:
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        st.success("✅ Candidate Set B (Engineered)")
        st.code(
            '''<ex label="POSITIVE">
<text>Seamless onboarding! Integrated Stripe in under 20 minutes.</text>
<sentiment>POSITIVE</sentiment>
</ex>

<ex label="NEGATIVE">
<text>Refund button throws 500 server error. Terrible experience.</text>
<sentiment>NEGATIVE</sentiment>
</ex>

<ex label="NEUTRAL">
<text>What is the current rate limit for the batch extraction endpoint?</text>
<sentiment>NEUTRAL</sentiment>
</ex>

<ex label="EDGE_CASE_GIBBERISH">
<text>asdfkjh123 ???</text>
<sentiment>UNPARSABLE_INPUT</sentiment>
</ex>''',
            language="text"
        )
    with col_b2:
        st.markdown(
            """
### 🎯 Audit Approval & Highlights:
1. **Perfect 1:1:1:1 Ratio**: Equal representation across Positive, Negative, Neutral, and Edge Case.
2. **Syntactic Delimiters**: Strict XML boundary tagging.
3. **Real-World Nuance**: Covers technical errors (500 error), technical queries, and noise fallback.
"""
        )

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. Exemplar Pipeline Architecture", divider="rainbow")

col_g1, col_g2 = st.columns([1, 1])

with col_g1:
    st.markdown("#### Exemplar Curation & Ingestion Lifecycle")
    dot_curation = """
    digraph ExemplarPipeline {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Logs [label="Production Audit Logs\\n(Real User Queries)", fillcolor="#334155", fontcolor="#ffffff"];
        Filter [label="1. Sanitize & Anonymize\\n(Strip PII & Keys)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Balance [label="2. Balance Classes (1:1:1)\\n(Stratified Sampling)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Edge [label="3. Inject Boundary Cases\\n(Nulls, Fallbacks)", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        XML [label="4. Standardize Delimiters\\n(<example> XML Framing)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Context [label="Final Prompt Context Stream", fillcolor="#047857", fontcolor="#ffffff"];

        Logs -> Filter -> Balance -> Edge -> XML -> Context;
    }
    """
    st.graphviz_chart(dot_curation, use_container_width=False)

with col_g2:
    st.markdown("#### The Exemplar Golden Ratio")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ TOTAL EXEMPLARY BUDGET: 4 EXAMPLES                      │
├────────────────────────────────────────────────────────┤
│  [25%] Positive Canonical Path                         │
│  [25%] Negative Class Path                             │
│  [25%] Neutral / Ambiguous Query                       │
│  [25%] Unanswerable / Malicious Injection Fallback     │
└────────────────────────────────────────────────────────┘""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

e_tab1, e_tab2, e_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with e_tab1:
    with st.expander("Q1: What happens if your few-shot exemplars have label imbalance?", expanded=True):
        st.markdown(
            """
LLMs are highly sensitive to label frequency within the prompt context window. 
If 80% of your examples demonstrate class A and 20% demonstrate class B, the model will skew its prior probability toward predicting class A, regardless of the user input's true semantic class. 
This is called **Majority-Class Frequency Bias** and can degrade classification F1 scores by 20–30%.
"""
        )

    with st.expander("Q2: Why are counter-examples (negative examples) difficult to implement correctly in few-shot prompting?", expanded=False):
        st.markdown(
            """
Because an LLM may copy the bad output rather than learning to avoid it if the framing is ambiguous. 
If you simply show a flawed input and a bad response, the attention induction heads might learn that generating the bad response is the desired task. 
To implement counter-examples correctly, always show the flawed input paired with the **correct error-handling response** (e.g. `<status>REJECTED_INVALID_INPUT</status>`), accompanied by an explicit `<rationale>` tag.
"""
        )

with e_tab2:
    with st.expander("Trap 1: 'Can we just use ChatGPT to generate all our few-shot examples synthetic-style?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, asking an LLM to generate 5 examples is fast and easy."*
- **Winning Response**: *"Synthetic examples generated by LLMs tend to be overly uniform, grammatically pristine, and devoid of the linguistic noise (typos, slang, unstructured fragments) found in real user traffic. While LLM generation is acceptable for a rapid prototype, production few-shot exemplars must be mined from real anonymized production logs to ensure the model learns to navigate real-world distribution shifts."*
"""
        )

with e_tab3:
    c_de1, c_de2, c_de3 = st.columns(3)
    with c_de1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Label Imbalance Risk**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Skewed exemplar ratios trick the model into predicting whichever label appeared most frequently.")

    with c_de2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Format Symmetry**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Enforcing identical delimiter tags (`<example>`) across all samples to anchor induction heads.")

    with c_de3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Counter-Exemplar**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Demonstrating an invalid input paired with a deterministic fallback response (`REJECTED`).")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 4.2 Summary & Self-Assessment", divider="rainbow")

col_es1, col_es2 = st.columns(2)
with col_es1:
    st.markdown(
        """
### Key Takeaways:
1. **Symmetry is King**: Maintain identical delimiter contracts across all exemplars.
2. **Balance Labels (1:1:1)**: Prevent majority-class frequency bias with equal class counts.
3. **Real-World Diversity**: Mine anonymized production logs rather than generic toy samples.
4. **Demonstrate Fallbacks**: Teach the model how to fail gracefully when inputs are corrupted.
"""
    )

with col_es2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain why label imbalance in few-shot examples biases model predictions.", value=True, key="chk_4_2_1")
    st.checkbox("I know how to structure positive, negative, and edge-case exemplars using XML tags.", value=True, key="chk_4_2_2")
    st.checkbox("I understand the danger of relying exclusively on synthetic LLM-generated examples.", value=True, key="chk_4_2_3")
    st.checkbox("I can audit a few-shot exemplar set for lexical diversity and formatting symmetry.", value=True, key="chk_4_2_4")
