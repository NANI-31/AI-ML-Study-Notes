import streamlit as st
import json

st.set_page_config(
    page_title="27.1 Token Optimization - Prompt Engineering",
    page_icon="✂️",
    layout="wide"
)

# Header Section
st.title("27.1 Token Optimization & Prompt Compression")
st.caption("Chapter 27: Performance and Cost Optimization | Domain: 1. Prompt Engineering")
st.markdown("""
In commercial generative AI applications, tokens are the fundamental unit of billing, memory, and latency. 
An unoptimized prompt template with bloated markdown, redundant explanations, and unconstrained completions 
wastes millions of tokens annually. Engineering teams apply **ruthless token pruning, structural minification, 
context compression, strict output bounds, and prompt caching prefix alignments** to trim excess token weight.
""")

st.divider()

# Section 1: The Token Anatomy & Waste Factors
st.header("1. Anatomy of Token Bloat in Production Prompts", divider="rainbow")
st.markdown("""
Token bloat typically sneaks into production prompts across five key vectors:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Conversational Fluff & Politeness")
    st.markdown("""
    Phrases like *"Please be so kind as to carefully review this text and provide your best thoughtful summary..."* 
    consume 15+ tokens without adding semantic constraint. 
    Replacing with `"Summarize text concisely."` saves 75% of instruction tokens while actually improving model adherence.
    """)

    st.subheader("2. Redundant Multi-Paragraph Constraints")
    st.markdown("""
    Repeating the same instruction three times in different paragraphs (*"Remember to output JSON. You must use JSON format. Do not forget JSON."*) 
    dilutes attention and multiplies token usage. Use one clear, unambiguous delimiter block instead.
    """)

with col2:
    st.subheader("3. Unbounded Generation Defaults")
    st.markdown("""
    Leaving `max_tokens` unconstrained allows the model to generate verbose apologies, conversational pleasantries, 
    and unsolicited commentary, inflating completion token costs (which are 3x to 4x more expensive than input tokens!).
    """)

    st.subheader("4. Formatting Overhead in Schemas")
    st.markdown("""
    Overly verbose JSON examples with long keys (`"customer_account_identification_number_string": ...`) 
    can be minified to concise keys (`"cust_id": ...`) without loss of downstream utility.
    """)

st.divider()

# Section 2: Before-and-After Token Minification
st.header("2. Real-World Token Minification Case Study", divider="rainbow")
st.markdown("Examine how minifying a customer ticket triage prompt cuts token consumption by 68% while improving accuracy.")

col_bad, col_good = st.columns(2)

with col_bad:
    st.error("❌ Bloated Prompt (142 Tokens)")
    st.code('''System: You are an extremely helpful, polite, and diligent customer support classification assistant. Your job is to take the incoming customer message and carefully evaluate what department it belongs to. The departments are BILLING, TECH_SUPPORT, or GENERAL_INQUIRY. Please think very carefully and provide only the department name as your final output. Thank you so much for your help!

User: Hello support team! I noticed an extra charge of $45 on my invoice this morning. Can someone please look into this?''', language="text")
    st.caption("Token Count: 142 tokens | Verbose, conversational, unanchored.")

with col_good:
    st.success("✅ Minified Optimized Prompt (45 Tokens)")
    st.code('''System: Classify ticket into: BILLING, TECH_SUPPORT, GENERAL. Output label only.

User: <ticket>Extra charge of $45 on my invoice this morning.</ticket>''', language="text")
    st.caption("Token Count: 45 tokens (68% Reduction!) | Zero ambiguity, razor-sharp focus.")

st.divider()

# Section 3: Algorithmic Token Trimmer
st.header("3. Algorithmic Token Pruning Patterns", divider="rainbow")
st.markdown("""
Production microservices run static text normalizers before prompt interpolation to eliminate whitespace and comment bloat:
""")

st.code('''# Programmatic Token Minifier for Ingested Text
import re

def minify_text_payload(text: str) -> str:
    """
    Strips redundant whitespace, empty lines, and conversational filler
    to minimize prompt token overhead.
    """
    # 1. Normalize multiple spaces and tabs to a single space
    text = re.sub(r'[ \\t]+', ' ', text)
    
    # 2. Collapse 3+ newlines to max 2 newlines
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    
    # 3. Strip leading/trailing line whitespace
    lines = [line.strip() for line in text.split('\\n') if line.strip()]
    
    return '\\n'.join(lines)''', language="python")

st.divider()

# Section 4: Interactive Token & Cost Savings Calculator
st.header("4. Interactive Token & Budget Savings Calculator", divider="rainbow")
st.markdown("Calculate the annual ROI of pruning prompt templates across your organization's request volume.")

col_c1, col_c2 = st.columns(2)

with col_c1:
    daily_reqs = st.number_input("Daily Requests", value=100000, step=25000)
    toks_before = st.slider("Average Prompt Tokens (Before Pruning)", 500, 4000, 1800, step=100)
    toks_after = st.slider("Average Prompt Tokens (After Optimization)", 200, 2000, 600, step=50)

with col_c2:
    # Model pricing: $2.50 per 1M input tokens
    toks_saved_per_req = toks_before - toks_after
    daily_toks_saved = daily_reqs * toks_saved_per_req
    annual_toks_saved = daily_toks_saved * 365
    
    annual_savings_dollars = (annual_toks_saved / 1_000_000) * 2.50
    pct_reduction = (toks_saved_per_req / toks_before) * 100
    
    st.subheader("Financial Impact Assessment")
    m1, m2 = st.columns(2)
    m1.metric("Token Reduction", f"{pct_reduction:.1f}%")
    m2.metric("Annual Tokens Saved", f"{annual_toks_saved / 1e9:.2f} Billion")
    st.metric("Annual Budget Saved ($)", f"${annual_savings_dollars:,.2f}", delta="Annualized Savings")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: Why are output tokens significantly more expensive than input tokens across virtually all LLM providers?"):
        st.write("""
        * **Input Tokens (Prefill Phase)**: Processed in **parallel** across GPU tensor cores. The entire prompt matrix is computed in parallel matrix multiplications, maximizing GPU FLOP utilization.
        * **Output Tokens (Decode Phase)**: Must be generated **autoregressively, one token at a time** in a sequential loop. 
        Each token requires loading model weights from VRAM to SRAM, causing GPU memory bandwidth bottlenecks. 
        Because decoding holds GPU resources hostage for longer durations per token, providers bill output tokens at 3x to 4x higher rates.
        """)

    with st.expander("Q: How does Tokenizer whitespace handling impact prompt efficiency?"):
        st.write("""
        Byte-Pair Encoding (BPE) tokenizers (like tiktoken `cl100k_base` or `o200k_base`) treat leading spaces as part of words 
        (e.g., `' hello'` is 1 token, but multiple spaces `'   '` or non-standard tabs can fragment into multiple individual byte tokens). 
        Stripping redundant trailing spaces and normalizing indentation prevents token fragmentation.
        """)

with qa2:
    with st.expander("Trap to Avoid: Over-pruning Semantic Context into Vagueness"):
        st.write("""
        While cutting filler words is vital, pruning essential domain constraints or edge-case handling rules 
        leads to downstream quality failures and customer escalations. 
        Always validate minified prompt versions against automated golden evaluation datasets (CI/CD eval suites) 
        to ensure accuracy does not drop along with token count.
        """)

    with st.expander("Flashcards: Token Optimization"):
        fc_sel = st.selectbox(
            "Select Token Term:",
            ["Prefill vs Decode", "BPE Fragmentation", "Minification", "Prompt Token Delta"]
        )
        if fc_sel == "Prefill vs Decode":
            st.info("Prefill processes inputs in parallel; decode generates outputs sequentially, driving higher per-token cost.")
        elif fc_sel == "BPE Fragmentation":
            st.info("When erratic whitespace or symbols break into multiple expensive byte tokens.")
        elif fc_sel == "Minification":
            st.info("Stripping conversational adjectives, excessive formatting, and redundant filler from system prompts.")
        else:
            st.info("The net change in token count achieved by refactoring and structuring instructions.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand why output tokens are priced 3x to 4x higher than input tokens.")
    st.checkbox("I know how to minify bloated system prompts without degrading task performance.")
with c2:
    st.checkbox("I can implement programmatic whitespace and delimiter sanitization.")
    st.checkbox("I understand how BPE tokenization fragments erratic spacing.")
