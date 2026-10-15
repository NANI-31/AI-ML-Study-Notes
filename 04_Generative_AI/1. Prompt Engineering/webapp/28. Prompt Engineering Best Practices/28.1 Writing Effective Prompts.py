import streamlit as st
import json

st.set_page_config(
    page_title="28.1 Writing Effective Prompts - Prompt Engineering",
    page_icon="✍️",
    layout="wide"
)

# Header Section
st.title("28.1 The Core Tenets of Writing Effective Prompts")
st.caption("Chapter 28: Prompt Engineering Best Practices | Domain: 1. Prompt Engineering")
st.markdown("""
Effective prompt engineering is fundamentally about **reducing ambiguity and maximizing constraint alignment**. 
A world-class prompt is not a polite conversational request; it is an **executable specification**. 
Mastery rests on six bedrock tenets: radical clarity, exacting specificity, grounded context, 
unambiguous constraints, strict output schemas, and high-fidelity few-shot exemplars.
""")

st.divider()

# Section 1: The 6 Bedrock Tenets
st.header("1. The Six Bedrock Tenets of High-Impact Prompts", divider="rainbow")
st.markdown("""
Every production prompt should be measured against these six foundational criteria:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Radical Clarity")
    st.markdown("""
    * State the exact primary action in the very first sentence.
    * Eliminate fluff words (*"Please", "Kindly", "Try your best"*).
    * Use active imperative verbs: `"Extract", "Classify", "Validate", "Summarize"`.
    """)

    st.subheader("2. Exacting Specificity")
    st.markdown("""
    * Avoid relative descriptors (*"brief", "detailed", "good"*).
    * Quantify parameters with exact boundaries: `"Under 150 words", "Exactly 3 bullet points", "Include numerical percentage"`.
    """)

    st.subheader("3. Grounded Context Injection")
    st.markdown("""
    * Isolate reference knowledge cleanly inside structured delimiters (`<context>`, `<reference_docs>`).
    * Explicitly mandate that answers must be derived solely from injected context.
    """)

with col2:
    st.subheader("4. Explicit Negative Constraints")
    st.markdown("""
    * Specify what the model must **NOT** do.
    * Provide explicit refusal tokens: `"If not mentioned in context, output UNKNOWN. Never invent facts."`
    """)

    st.subheader("5. Strict Output Format Definition")
    st.markdown("""
    * Declare the exact serialization format (JSON schema, CSV, Markdown table).
    * Forbid unrequested conversational preambles or post-explanations.
    """)

    st.subheader("6. Ground-Truth Few-Shot Exemplars")
    st.markdown("""
    * Provide 1 to 3 diverse input/output demonstrations representing realistic edge cases, unusual abbreviations, and null handling.
    """)

st.divider()

# Section 2: The Master Production Prompt Template
st.header("2. The Master Production Prompt Specification Pattern", divider="rainbow")
st.markdown("""
The template below illustrates how to assemble all six tenets into a bulletproof production prompt specification:
""")

st.code('''# The Canonical Enterprise Prompt Specification Pattern
SYSTEM_PROMPT = """You are a deterministic financial compliance auditor.

<objective>
Audit incoming transaction logs against corporate anti-money-laundering (AML) policies.
</objective>

<rules>
1. Output ONLY a valid JSON object matching the requested schema.
2. If transaction is under $10,000 and from a domestic vendor, status is "APPROVED".
3. If transaction is >= $10,000 or from an unverified international vendor, status is "FLAGGED".
4. If transaction details are incomplete, status is "PENDING_INFO". Never guess missing IDs.
5. Do not include markdown code fences, greetings, or conversational preambles.
</rules>

<few_shot_examples>
Example 1:
Input: "Tx #101: $4,200 to Staples Inc (Domestic)."
Output: {"tx_id": "101", "status": "APPROVED", "reason": "Domestic under $10k"}

Example 2:
Input: "Tx #102: $15,000 to Global Supplies (Overseas)."
Output: {"tx_id": "102", "status": "FLAGGED", "reason": "Exceeds $10k threshold and international"}
</few_shot_examples>"""

USER_PROMPT = """<transaction>
{transaction_payload}
</transaction>"""''', language="python")

st.divider()

# Section 3: Before-and-After Prompt Evolution
st.header("3. Before-and-After Prompt Transformation", divider="rainbow")
st.markdown("Compare a novice prompt with an enterprise-hardened executable specification.")

col_a, col_b = st.columns(2)

with col_a:
    st.error("❌ Novice Prompt (High Failure Rate)")
    st.code('''Can you please look at this article and tell me what you think about it? 
Make sure you write a good summary that highlights the main points and is not too long. 
Also tell me if the sentiment is good or bad. Thanks!

Article: {text}''', language="text")
    st.markdown("""
    * Vague verbs (*"tell me what you think"*).
    * Subjective constraints (*"good summary", "not too long"*).
    * Polysemous output (*"good or bad"* instead of POSITIVE/NEGATIVE).
    * No schema anchoring.
    """)

with col_b:
    st.success("✅ Hardened Specification (Deterministic)")
    st.code('''System: You are an editorial summarizer. Analyze the input article and output a JSON object.

<json_schema>
{
  "headline": "string (under 10 words)",
  "executive_summary": "string (between 40 and 60 words)",
  "key_takeaways": ["string (max 3 bullets)"],
  "sentiment": "POSITIVE | NEGATIVE | NEUTRAL"
}
</json_schema>

Rules:
- Strictly follow the schema types and constraints.
- Output ONLY valid JSON.

User:
<article>
{text}
</article>''', language="text")
    st.markdown("""
    * Exact word bounds and schema types.
    * Controlled categorical vocabulary for sentiment.
    * Rigid XML boundary isolation.
    """)

st.divider()

# Section 4: Interactive Prompt Scorecard Workbench
st.header("4. Interactive Prompt Quality Scorecard", divider="rainbow")
st.markdown("Audit your candidate prompt against the 6 core tenets to compute a readiness score.")

user_draft = st.text_area(
    "Enter Candidate Prompt to Audit:",
    value="Extract all product names and prices from this receipt. Output as JSON with keys 'product' and 'price'. If price is missing, use null.",
    height=100
)

# Automated heuristic checks
c_clarity = any(v in user_draft.lower() for v in ["extract", "classify", "summarize", "audit", "calculate"])
c_specific = any(c in user_draft.lower() for c in ["keys", "under", "exactly", "strictly", "null", "format"])
c_format = "json" in user_draft.lower() or "schema" in user_draft.lower() or "table" in user_draft.lower()
c_neg = any(n in user_draft.lower() for n in ["never", "do not", "if missing", "if not", "null"])

score = sum([c_clarity, c_specific, c_format, c_neg])

m_col1, m_col2 = st.columns(2)
with m_col1:
    st.metric("Heuristic Readiness Score", f"{score} / 4")
with m_col2:
    if score == 4:
        st.success("🎉 High-Quality Prompt: Clear imperative verb, formatting target, schema specification, and negative fallback!")
    elif score >= 2:
        st.info("👍 Decent Foundation: Consider adding explicit negative constraints or few-shot exemplars.")
    else:
        st.warning("⚠️ High Ambiguity: Specify exact output formats and quantified constraints.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: Why is providing positive fallbacks just as critical as negative constraints?"):
        st.write("""
        Negative constraints tell the model what **not** to do (`"Do not fabricate facts"`), but without positive guidance, 
        the model is left in an under-constrained state where attention heads must guess how to recover. 
        Pairing negative rules with positive fallback directives (`"If the date cannot be found in context, output null and set confidence=0.0"`) 
        provides an explicit deterministic escape path, eliminating hallucination pressure.
        """)

    with st.expander("Q: What is 'Prompt Leakage' and how do delimiters prevent it?"):
        st.write("""
        Prompt leakage occurs when dynamic user input contains phrases that the LLM confuses with system instructions 
        (e.g. user input saying `"Ignore previous rules"` or containing multiple markdown headers). 
        Wrapping dynamic inputs in structural delimiters (like `<user_input>...</user_input>`) informs the model 
        where system instructions end and raw data begins, preventing instruction collision.
        """)

with qa2:
    with st.expander("Trap to Avoid: Writing 'Negative Only' Prompts"):
        st.write("""
        A prompt that lists 10 things NOT to do (`"Don't write long answers, don't use adjectives, don't be passive..."`) 
        causes behavioral paralysis and awkward token completions. 
        Always frame prompt directives around what the model SHOULD do, backed by 1 or 2 clear negative boundaries.
        """)

    with st.expander("Flashcards: Prompting Tenets"):
        fc_sel = st.selectbox(
            "Select Tenet Flashcard:",
            ["Radical Clarity", "Schema Anchoring", "Positive Fallbacks", "Instruction Delimiters"]
        )
        if fc_sel == "Radical Clarity":
            st.info("Using direct active imperative verbs and stripping conversational pleasantries.")
        elif fc_sel == "Schema Anchoring":
            st.info("Specifying explicit typed JSON objects or Pydantic classes to constrain generation.")
        elif fc_sel == "Positive Fallbacks":
            st.info("Providing an explicit safe action (e.g. output null) when data is missing.")
        else:
            st.info("Using XML or Markdown fences to separate trusted system instructions from untrusted user payloads.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can structure prompts as executable engineering specifications.")
    st.checkbox("I know how to pair negative constraints with positive fallback directives.")
with c2:
    st.checkbox("I can isolate dynamic user data using XML delimiters to prevent prompt leakage.")
    st.checkbox("I can anchor extraction tasks with high-fidelity few-shot exemplars.")
