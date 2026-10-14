import streamlit as st
import json

st.set_page_config(
    page_title="22.2 Debugging Techniques - Prompt Engineering",
    page_icon="🛠️",
    layout="wide"
)

# Header Section
st.title("22.2 Systematic Prompt Debugging Techniques")
st.caption("Chapter 22: Prompt Debugging and Troubleshooting | Domain: 1. Prompt Engineering")
st.markdown("""
Debugging a prompt is not a matter of random rephrasing or hoping for the best. 
Production engineering relies on a **deterministic, incremental refinement methodology**: 
stripping down complexity to minimal reproducers, task decomposition, anchoring with few-shot exemplars, 
tightening boundary constraints, and validating bidirectional I/O contracts.
""")

st.divider()

# Section 1: The 6 Core Debugging Techniques
st.header("1. The Six Core Engineering Techniques", divider="rainbow")
st.markdown("""
When a prompt misbehaves, apply these six progressive interventions in order of operational impact:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Simplify the Prompt (Minimal Reproducer)")
    st.markdown("""
    Strip away unnecessary background fluff, complex role adjectives, and convoluted multi-paragraph explanations. 
    Reduce the prompt down to the absolute bare minimum required to reproduce the failure.
    * If the minimal prompt succeeds: The bug was caused by instruction crowding or conflicting constraints.
    * If the minimal prompt still fails: The task itself requires decomposition or few-shot demonstration.
    """)

    st.subheader("2. Break the Task into Sequential Steps")
    st.markdown("""
    Single-turn monolithic prompts that attempt extraction, classification, validation, and generation simultaneously 
    overload attention heads.
    * Convert into explicit procedural steps: `"Step 1: Identify key clauses. Step 2: Compare against SLA terms. Step 3: Produce final verdict."`
    * Or split into an explicit pipeline of two smaller, specialized prompts.
    """)

    st.subheader("3. Anchor with Ground-Truth Examples (Few-Shot)")
    st.markdown("""
    Abstract verbal descriptions are inherently lossy. Adding 2 to 3 canonical input/output examples 
    clarifies edge cases (e.g. handling missing fields, unusual abbreviations, date formatting) with 10x higher fidelity 
    than paragraphs of prose rules.
    """)

with col2:
    st.subheader("4. Tighten Negative and Boundary Constraints")
    st.markdown("""
    Clearly define what the model must **NOT** do. Negative constraints prevent common leakage patterns:
    * `"If the requested entity is not explicitly mentioned, output null. Never guess or interpolate."`
    * `"Do not include any conversational pleasantries, introductory markdown, or trailing thoughts."`
    * `"Restrict output length to strictly under 100 characters."`
    """)

    st.subheader("5. Validate and Sanitize Inputs")
    st.markdown("""
    Bugs often arise not from the prompt template, but from pathological user inputs (blank strings, 50,000-character pastes, 
    or malformed JSON payloads).
    * Enforce token boundaries, string normalization, and input sanitization before string interpolation.
    """)

    st.subheader("6. Enforce Bidirectional Output Validation Contracts")
    st.markdown("""
    Never treat LLM raw string generation as safe code. Wrap all completions in deterministic programmatic validators:
    * Regex validation for IDs, dates, and currency.
    * Pydantic schema validation with automatic retry loops.
    * Type guards and fallback handlers.
    """)

st.divider()

# Section 2: Progressive Refinement Walkthrough
st.header("2. Progressive Refinement: From Buggy to Rock-Solid", divider="rainbow")
st.markdown("Observe how applying the debugging techniques transforms a flaky prompt into an enterprise-ready microservice.")

stage = st.slider(
    "Select Debugging Iteration Stage:",
    min_value=1,
    max_value=4,
    value=1,
    format="Stage %d"
)

if stage == 1:
    st.markdown("### Stage 1: The Monolithic Flaky Prompt (Failure Rate: ~40%)")
    st.error("Problem: Too many tasks crammed into one unanchored block with zero formatting constraints.")
    st.code('''System: You are an invoice analyzer. Review the provided invoice, check if the math adds up, flag any suspicious vendors, calculate the sales tax, extract all line items with prices, and write a summary for the CFO.

User: {raw_invoice_text}''', language="text")
    st.info("Failure modes: Skips tax calculation, hallucinates missing line items, varies format between markdown and bullet points.")

elif stage == 2:
    st.markdown("### Stage 2: Step Decomposition & Simplification (Failure Rate: ~20%)")
    st.warning("Improvement: Task decomposed into sequential execution steps.")
    st.code('''System: You are an invoice auditor. Process the input strictly in three steps:
Step 1: Extract line items as a list of item name, quantity, and unit price.
Step 2: Calculate line subtotal and verify tax is 8.25%.
Step 3: Output a 2-sentence executive summary.

User: {raw_invoice_text}''', language="text")
    st.info("Improvement: Logic is much clearer, but downstream code still cannot reliably parse the output.")

elif stage == 3:
    st.markdown("### Stage 3: Adding Schema & Negative Constraints (Failure Rate: ~5%)")
    st.warning("Improvement: Rigid JSON schema and strict negative constraints.")
    st.code('''System: You are a deterministic invoice parsing engine.
Rules:
1. Output ONLY a valid JSON object matching the provided schema.
2. If any field cannot be found, use null. Never invent or estimate prices.
3. No conversational prose or markdown formatting outside the JSON block.

<json_schema>
{
  "items": [{"name": "string", "quantity": 0, "price": 0.0}],
  "subtotal": 0.0,
  "tax_verified": true,
  "flagged": false
}
</json_schema>

User: {raw_invoice_text}''', language="text")
    st.info("Improvement: JSON parsing succeeds 95% of the time. Edge-case currency symbols ($ vs EUR) occasionally cause parse fails.")

else:
    st.markdown("### Stage 4: Production Hardening (Few-Shot Exemplar + Pydantic Contract) (Failure Rate: < 0.1%)")
    st.success("State of the Art: Few-shot exemplar anchors edge cases, Pydantic handles validation & auto-healing.")
    st.code('''System: You are a deterministic invoice parsing engine.

<few_shot_exemplar>
Input: "Inv #102: 2 widgets @ $15.00 each. Subtotal $30. Tax $2.48. Total $32.48."
Output: {"items": [{"name": "widgets", "quantity": 2, "price": 15.0}], "subtotal": 30.0, "tax_amount": 2.48, "currency": "USD"}
</few_shot_exemplar>

Rules:
- Strip currency symbols ($/€) into numeric floats.
- Use ISO currency codes.
- If quantity is omitted, default to 1.

User:
<invoice_payload>
{raw_invoice_text}
</invoice_payload>''', language="text")
    st.caption("Coupled with a Pydantic retry wrapper, achieves 99.9%+ reliability in production pipelines.")

st.divider()

# Section 3: Programmatic Output Validation & Auto-Healing
st.header("3. Output Validation & Auto-Healing Code Pattern", divider="rainbow")
st.markdown("""
A robust prompt engineering system never trusts raw completions. 
The code snippet below illustrates how to implement validation guards and automatic retry prompts upon failure:
""")

st.code('''# Self-Healing Prompt Validation Pattern
from pydantic import BaseModel, Field, ValidationError
import json

class CustomerClassification(BaseModel):
    intent: str = Field(description="Must be one of: BILLING, TECH, SALES")
    urgency_score: int = Field(ge=1, le=5, description="Integer urgency from 1 to 5")
    requires_human_escalation: bool

def robust_llm_call(prompt_text: str, max_retries: int = 2) -> CustomerClassification:
    current_prompt = prompt_text
    
    for attempt in range(max_retries + 1):
        # 1. Dispatch to LLM
        response_text = mock_llm_generate(current_prompt)
        
        # 2. Validate syntactic and schema contracts
        try:
            parsed_json = json.loads(response_text)
            validated_obj = CustomerClassification(**parsed_json)
            return validated_obj
        except (json.JSONDecodeError, ValidationError) as err:
            if attempt == max_retries:
                raise RuntimeError(f"Prompt failed schema validation after {max_retries} retries: {err}")
            
            # 3. Dynamic Auto-Healing Error Feedback Prompt
            current_prompt = f"""The previous output was invalid.
Error details: {str(err)}
Please regenerate the output strictly adhering to valid JSON and schema constraints.
Original Prompt:
{prompt_text}"""''', language="python")

st.divider()

# Section 4: Interactive Prompt Debugger Workbench
st.header("4. Interactive Prompt Debugger Workbench", divider="rainbow")
st.markdown("Simulate applying debugging techniques to fix common prompt failure scenarios.")

test_scenario = st.selectbox(
    "Choose Bug to Fix:",
    [
        "Fix 1: Add Few-Shot to stabilize Date Parsing",
        "Fix 2: Add Negative Constraint to eliminate conversational noise",
        "Fix 3: Step Decomposition to fix multi-step math"
    ]
)

if test_scenario == "Fix 1: Add Few-Shot to stabilize Date Parsing":
    st.markdown("**Original Flaky Output**: `'The date is roughly mid-May or maybe 5/14/24'`")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Without Few-Shot:")
        st.code("Extract the date in YYYY-MM-DD from 'Arriving next Tuesday after May 10, 2024'")
    with col_b:
        st.write("With Few-Shot Anchor:")
        st.code("""Extract dates into strictly YYYY-MM-DD.
Example: "Leaving this Friday, June 7th 2024" -> "2024-06-07"
Input: "Arriving next Tuesday after May 10, 2024" -> "2024-05-14\"""")
    st.success("Result: 100% adherence to ISO-8601 formatting.")

elif test_scenario == "Fix 2: Add Negative Constraint to eliminate conversational noise":
    st.markdown("**Original Flaky Output**: `'Sure! I am happy to help you with that. The sentiment is POSITIVE'`")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Without Negative Constraint:")
        st.code("Classify sentiment of: 'Great service!'")
    with col_b:
        st.write("With Negative Constraint:")
        st.code("""Classify sentiment of the text.
Output exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.
Do not include any conversational greeting, punctuation, or preamble.""")
    st.success("Result: Single-token clean response `'POSITIVE'`.")

else:
    st.markdown("**Original Flaky Output**: Fabricated final total due to mental leap in complex discounts.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Without Step Decomposition:")
        st.code("A $120 item has 15% discount, plus 8% tax on the discounted price, minus a $10 coupon. What is total?")
    with col_b:
        st.write("With Step Decomposition:")
        st.code("""Solve step by step:
Step 1: Calculate discounted price = original price * (1 - discount).
Step 2: Calculate tax = discounted price * tax rate.
Step 3: Subtract coupon from (discounted price + tax).
State each step clearly, then output Final Total on its own line.""")
    st.success("Result: Intermediate values calculated accurately step-by-step.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_col1, qa_col2 = st.columns(2)

with qa_col1:
    with st.expander("Q: What is 'Self-Healing Prompts' and what are the trade-offs in production?"):
        st.write("""
        Self-healing is an architectural pattern where failed outputs (syntax errors, validation exceptions) 
        are fed back into the LLM along with the exact parser error trace in an automatic retry loop.
        * **Benefits**: Recovers 70% to 90% of edge-case schema failures automatically without crashing downstream services.
        * **Trade-offs**: Multiplies latency by 2x to 3x on failures and increases token costs. Best reserved for asynchronous background jobs.
        """)

    with st.expander("Q: When should you decompose a prompt into multiple calls versus chaining steps in a single prompt?"):
        st.write("""
        * **Single Prompt with Steps**: Ideal when intermediate steps are small, budget is constrained, and latency must stay low (< 1 second).
        * **Multiple Chained Calls**: Necessary when intermediate steps require external tool lookup, when output exceeds token context, 
        or when individual sub-tasks require different specialized system instructions or safety evaluations.
        """)

with qa_col2:
    with st.expander("Trap to Avoid: Writing Long Negative Constraints Without Stating Allowed Actions"):
        st.write("""
        Telling an LLM only what NOT to do (`"Do not be rude, do not use slang, do not mention prices..."`) 
        creates behavioral paralysis. Models perform significantly better when negative constraints are paired 
        with explicit positive fallbacks: `"If price is unknown, output 'PRICING_UNAVAILABLE' and direct to sales@domain.com"`.
        """)

    with st.expander("Flashcards: Debugging Patterns"):
        sel_fc = st.selectbox(
            "Select Debugging Flashcard:",
            ["Minimal Reproducer", "Schema Anchoring", "Negative Grounding", "Pydantic Self-Correction"]
        )
        if sel_fc == "Minimal Reproducer":
            st.info("Stripping all non-essential instructions until the simplest prompt that reproduces the bug remains.")
        elif sel_fc == "Schema Anchoring":
            st.info("Providing an explicit structural template or JSON schema so the attention mechanism focuses on key-value filling.")
        elif sel_fc == "Negative Grounding":
            st.info("Explicitly stating safe fallback tokens (e.g. 'UNKNOWN') to avoid forcing the LLM to hallucinate.")
        else:
            st.info("Catching ValidationError and feeding the exact Pydantic error dictionary back to the LLM to fix formatting.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know how to construct a minimal reproducer to isolate prompt bugs.")
    st.checkbox("I can implement an auto-healing validation retry loop in Python.")
with c2:
    st.checkbox("I know how to balance negative constraints with positive fallback directives.")
    st.checkbox("I can decompose a monolithic prompt into sequential operational steps.")
