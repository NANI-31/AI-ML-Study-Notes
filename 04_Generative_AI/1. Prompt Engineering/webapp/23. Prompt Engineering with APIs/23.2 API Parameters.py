import streamlit as st
import json

st.set_page_config(
    page_title="23.2 API Parameters - Prompt Engineering",
    page_icon="🎛️",
    layout="wide"
)

# Header Section
st.title("23.2 API Hyperparameters & Decoding Controls")
st.caption("Chapter 23: Prompt Engineering with APIs | Domain: 1. Prompt Engineering")
st.markdown("""
Prompt text provides instructions, but **inference hyperparameters govern decoding behavior**. 
Configuring temperature, sampling thresholds (top_p, top_k), penalty coefficients, stop sequences, 
and constrained response formats determines whether an LLM acts as a creative ideator or a deterministic state machine.
""")

st.divider()

# Section 1: Temperature and Top-p Sampling
st.header("1. Temperature, Top-p, and Nucleus Sampling", divider="rainbow")
st.markdown("""
At each generation step, the language model produces unnormalized logits for every token in its vocabulary. 
Decoding hyperparameters transform these logits into probability distributions:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Temperature (T)")
    st.markdown("""
    Scales the logit distribution before softmax:
    * **T = 0.0 (Greedy Decoding)**: Always selects the single token with highest probability. Crucial for classification, JSON extraction, coding, and math.
    * **T = 0.2 - 0.5**: Focused, deterministic generation with minor lexical variety. Ideal for enterprise customer service and RAG synthesis.
    * **T = 0.7 - 1.0+**: Flattens distribution, increasing likelihood of selecting lower-ranked tokens. Used for creative storytelling, marketing copy, and brainstorming.
    """)

with col2:
    st.subheader("Top-p (Nucleus Sampling)")
    st.markdown("""
    Restricts candidate tokens to the smallest set whose cumulative probability reaches threshold `p`:
    * **Top-p = 0.1**: Retains only the top ~10% probability mass (strictly conservative).
    * **Top-p = 0.9 (Standard Default)**: Truncates the long improbable tail of tokens while retaining natural diversity.
    * **Production Rule of Thumb**: Modify **either** Temperature or Top-p; never tune both simultaneously to prevent unpredictable compounding.
    """)

st.divider()

# Section 2: Penalties and Token Boundaries
st.header("2. Maximum Tokens, Penalties, and Stop Sequences", divider="rainbow")
st.markdown("""
Fine-tuning execution boundaries prevents runaway generation and repetitive loops:
""")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.subheader("max_tokens / max_completion_tokens")
    st.markdown("""
    Hard ceiling on generated tokens. 
    * If hit before completion, response abruptly cuts off (`finish_reason="length"`).
    * Always provide a buffer (e.g. 500 tokens for a 150-token expected response) to avoid malformed JSON syntax.
    """)

with col_p2:
    st.subheader("Presence & Frequency Penalties")
    st.markdown("""
    * **Frequency Penalty (-2.0 to 2.0)**: Penalizes tokens based on how many times they have already appeared in the output (prevents phrase looping).
    * **Presence Penalty (-2.0 to 2.0)**: Flat penalty applied once if a token has appeared at all (encourages introducing new topics).
    """)

with col_p3:
    st.subheader("Stop Sequences (`stop`)")
    st.markdown("""
    Array of up to 4 strings that instantly halt generation when emitted:
    * Example: `["\\n\\n", "User:", "###", "```"]`.
    * Indispensable for preventing few-shot prompt leakage where the model continues hallucinating fake user turns.
    """)

st.divider()

# Section 3: Response Format & Tool Configuration
st.header("3. Constrained Response Formats and Tool Configuration", divider="rainbow")
st.markdown("""
Modern APIs provide deterministic grammar and schema anchors directly in API parameters:
""")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.subheader("response_format (JSON Mode vs Structured Outputs)")
    st.markdown("""
    * **`{"type": "json_object"}`**: Guarantees the output will parse as valid JSON. Requires mentioning 'JSON' somewhere in the prompt.
    * **`{"type": "json_schema", "json_schema": {"strict": true, ...}}`**: Enforces exact Pydantic schema alignment with 100% mathematical guarantee via grammar-constrained decoding.
    """)

with col_cfg2:
    st.subheader("tool_choice Controls")
    st.markdown("""
    Directs how the model interacts with defined functions:
    * **`"auto"`**: Model decides autonomously whether to call a tool or reply with text.
    * **`"required"`**: Forces model to call at least one tool.
    * **`{"type": "function", "function": {"name": "lookup_user"}}`**: Pinpoints an exact tool that must be called.
    * **`"none"`**: Disables tool execution completely.
    """)

st.code('''# Comprehensive Production API Parameter Payload
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a customer verification service."},
        {"role": "user", "content": "Verify account #8911 for user Alice."}
    ],
    temperature=0.0,              # Deterministic execution
    max_tokens=250,               # Hard safety cap
    frequency_penalty=0.0,        # No penalty needed for structured data
    stop=["###", "User:"],        # Stop triggers
    response_format={             # Guaranteed schema adherence
        "type": "json_schema",
        "json_schema": {
            "name": "VerificationResult",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "verified": {"type": "boolean"}
                },
                "required": ["account_id", "verified"],
                "additionalProperties": False
            }
        }
    }
)''', language="python")

st.divider()

# Section 4: Interactive Parameter Playground
st.header("4. Interactive Parameter Decoding Simulator", divider="rainbow")
st.markdown("Experiment with how temperature, top-p, and penalties alter the output distribution.")

col_ctrl, col_out = st.columns(2)

with col_ctrl:
    temp_val = st.slider("Temperature (0.0 = Deterministic, 1.2 = Highly Creative)", 0.0, 1.5, 0.2, step=0.1)
    top_p_val = st.slider("Top-p Nucleus Threshold", 0.1, 1.0, 0.9, step=0.05)
    stop_seq = st.multiselect("Active Stop Sequences", ["\\n\\n", "---", "Observation:", "END"], default=["---"])
    preset = st.selectbox("Apply Recommended Production Preset:", [
        "Custom",
        "Strict JSON Extraction (Temp=0.0, Top-p=1.0)",
        "Enterprise RAG Q&A (Temp=0.2, Top-p=0.9)",
        "Creative Marketing Copy (Temp=0.8, Top-p=0.95)"
    ])

with col_out:
    st.subheader("Parameter Analysis")
    if preset == "Strict JSON Extraction (Temp=0.0, Top-p=1.0)":
        st.success("Preset: Deterministic extraction. Logits locked to argmax. Zero variance.")
    elif preset == "Enterprise RAG Q&A (Temp=0.2, Top-p=0.9)":
        st.info("Preset: Stable RAG response. Natural phrasing without hallucinations.")
    elif preset == "Creative Marketing Copy (Temp=0.8, Top-p=0.95)":
        st.warning("Preset: High entropy. Explores novel analogies and varied vocabulary.")
    else:
        st.write(f"Current Config: Temperature={temp_val}, Top-p={top_p_val}, Stop={stop_seq}")
        
    st.markdown("#### Sample Model Behavior:")
    if temp_val == 0.0:
        st.code('{"status": "CONFIRMED", "code": 200, "details": "Record retrieved successfully."}', language="json")
    elif temp_val < 0.5:
        st.code('The verification was completed successfully. The account is currently active.', language="text")
    else:
        st.code('Welcome aboard! Your account is humming along smoothly and all verifications look fantastic!', language="text")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: Why is tuning both Temperature and Top-p simultaneously considered an anti-pattern?"):
        st.write("""
        Both parameters control the same underlying phenomenon: the shape and truncation of the token probability distribution. 
        Temperature scales the relative logit distances, while Top-p cuts off the probability tail. 
        Adjusting both creates confounded effects, making it impossible to systematically isolate which parameter caused an improvement or regression. 
        Industry best practice is to hold one fixed (e.g. `top_p=1.0` or `temperature=0.0`) and vary the other.
        """)

    with st.expander("Q: What is the architectural difference between 'JSON Mode' and 'Structured Outputs (Strict)'?"):
        st.write("""
        * **JSON Mode (`{"type": "json_object"}`)**: Tells the model to emit valid JSON syntax, but the model can still emit any keys or omit required fields.
        * **Structured Outputs (`strict: true`)**: The inference engine compiles the provided JSON Schema into a context-free grammar (CFG) mask. 
        During autoregressive decoding, tokens that would violate the schema are mathematically masked out at the logit level, guaranteeing 100% adherence to types and required keys.
        """)

with qa2:
    with st.expander("Trap to Avoid: Setting max_tokens Equal to Expected Response Length"):
        st.write("""
        If an expected summary is 200 words (~270 tokens), setting `max_tokens=270` will frequently cause premature cutoff 
        because LLM tokenization varies with punctuation and word choice. When output cuts off, JSON braces are unclosed, 
        invalidating downstream parsers. Always provide a generous buffer (e.g. 500-800 tokens). Token billing is based on 
        actual tokens generated, not the `max_tokens` ceiling!
        """)

    with st.expander("Flashcards: Parameter Controls"):
        fc_sel = st.selectbox(
            "Select Parameter Flashcard:",
            ["Greedy Decoding", "Nucleus Sampling (Top-p)", "Frequency Penalty", "CFG Masking"]
        )
        if fc_sel == "Greedy Decoding":
            st.info("Selecting the argmax token (T=0.0) at every step for deterministic outputs.")
        elif fc_sel == "Nucleus Sampling (Top-p)":
            st.info("Dynamically sampling only from the smallest set of tokens summing to probability p.")
        elif fc_sel == "Frequency Penalty":
            st.info("Proportionally penalizes tokens based on how frequently they have already appeared in the output.")
        else:
            st.info("Context-Free Grammar logit masking used in Structured Outputs to enforce schema rules.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know when to use Temperature 0.0 vs 0.7.")
    st.checkbox("I understand why tuning both Temperature and Top-p at once is discouraged.")
with c2:
    st.checkbox("I can configure stop sequences to prevent multi-turn few-shot hallucinations.")
    st.checkbox("I understand the difference between JSON Mode and Structured Outputs.")
