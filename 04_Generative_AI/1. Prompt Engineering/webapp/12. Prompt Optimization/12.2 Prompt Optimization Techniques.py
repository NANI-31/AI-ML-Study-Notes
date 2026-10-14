import streamlit as st
import pandas as pd
import re

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="12.2 Prompt Optimization Techniques | GenAI Roadmap",
    page_icon="⚡",
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
    .opt-chip {
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
st.title("⚡ 12.2 Prompt Optimization Techniques — Compression & Prioritization")
st.markdown(
    """
Welcome to **Lesson 12.2: Prompt Optimization Techniques**.  
In high-throughput enterprise pipelines handling millions of LLM requests daily, prompt quality is only half the battle. **Token efficiency, API latency, and inference cost** dictate whether an AI system is financially sustainable.

This lesson covers advanced optimization techniques: **prompt compression**, **instruction prioritization**, **dynamic example pruning**, and **context compaction**—enabling you to cut token usage by 30-50% while preserving benchmark accuracy.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Core Optimization Vectors
# ==============================================================================
st.header("1. The 6 Production Optimization Vectors", divider="rainbow")
st.markdown("Optimization targets latency reduction, token cost minimization, and maximum attention focus:")

vectors_df = pd.DataFrame({
    "Optimization Vector": [
        "1. Prompt Rewriting",
        "2. Prompt Compression",
        "3. Instruction Prioritization",
        "4. Example Optimization",
        "5. Output Constraint Compaction",
        "6. Context Compaction"
    ],
    "Engineering Methodology": [
        "Eliminating conversational pleasantries and passive voice; using concise imperative verbs",
        "Removing redundant syntactic filler, stop words, and whitespace using semantic pruners (e.g. LLMLingua)",
        "Placing mission-critical invariants in the system role and late recency position",
        "Selecting only the 1-2 most informative boundary few-shot examples using embedding similarity",
        "Specifying terse JSON key schemas without prose explanations",
        "Pre-summarizing lengthy reference docs and dropping irrelevant metadata tables"
    ],
    "Token & Latency Impact": [
        "Cuts prompt tokens by 15-25%; reduces time-to-first-token (TTFT)",
        "Cuts prompt tokens by 30-50%; drastically lowers API billing",
        "Prevents instruction forgetfulness in long context windows",
        "Saves 200-800 tokens per call while maintaining formatting alignment",
        "Reduces both input and output token consumption",
        "Prevents context window overflow and reduces attention dispersion"
    ]
})
st.dataframe(vectors_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Before & After Optimization Case Study
# ==============================================================================
st.header("2. Production Case Study: Verbose vs. Optimized Prompt", divider="rainbow")

col_ba1, col_ba2 = st.columns(2)

with col_ba1:
    with st.container(border=True):
        st.subheader("❌ Verbose Enterprise Prompt (318 Tokens)")
        st.code(
            '''Hello! You are an AI assistant tasked with acting as a legal contract assistant.
We would really appreciate it if you could carefully read through the employment contract 
that is provided below in full detail. Please take your time and do a thorough review.

When you are reading it, please look out for non-compete clauses. If you happen to find any 
non-compete clauses, please tell us if they are enforceable or not. In addition to that, 
could you please extract the governing jurisdiction, the termination notice period, and 
the base salary if mentioned? 

Please format your response in a clear and organized manner, preferably using JSON, 
so that our engineering team can easily parse it and save it into our internal databases. 
Thank you so much for your assistance!

Contract:
{{contract_text}}''',
            language="text",
        )
        st.markdown("**Issues:** Conversational filler, vague requests ('preferably JSON'), unbounded output, high token cost.")

with col_ba2:
    with st.container(border=True):
        st.subheader("✅ Optimized Production Prompt (124 Tokens — 61% Savings)")
        st.code(
            '''<system>
Legal contract entity extractor. Output strictly valid JSON.
</system>

<instructions>
Extract contract covenants from <contract>.
If a field is missing, assign null.
</instructions>

<contract>
{{contract_text}}
</contract>

<output_format>
{
  "non_compete_present": bool,
  "jurisdiction": str or null,
  "termination_notice_days": int or null,
  "base_salary_usd": float or null
}
</output_format>''',
            language="xml",
        )
        st.markdown("**Gains:** 61% token reduction, zero conversational drift, 100% deterministic JSON structure, faster inference.")

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Algorithmic Prompt Pruning Pipeline", divider="rainbow")
st.markdown("How semantic compression engines prune tokens before dispatching to frontier models:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9, color="white", fontcolor="white"]

        raw_prompt [label="Raw Application Prompt\n(5,000 Tokens)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        pruner [label="Semantic Pruner\n(e.g., LLMLingua / Heuristics)", fillcolor="#EAB308", fontcolor="#000000"]
        prioritizer [label="Attention Prioritizer\n(System + Recency Tags)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        compact_prompt [label="Compact Prompt\n(2,500 Tokens — 50% cut)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="Frontier LLM API\n(Faster TTFT & Lower Cost)", fillcolor="#10B981", fontcolor="#FFFFFF"]

        raw_prompt -> pruner -> prioritizer -> compact_prompt -> llm
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Prompt Compression Calculator
# ==============================================================================
st.header("4. Interactive Prompt Compression Workbench", divider="rainbow")
st.markdown("Test the impact of heuristic token pruning on enterprise prompt templates:")

sample_verbose_prompt = """Hello there! Could you please be so kind as to analyze the following customer support ticket for me?
I would really appreciate it if you could carefully read every single line and determine whether the customer seems angry or happy.
In addition to that, please try to extract the user's account identifier number if it happens to be present in the text.
Thank you very much in advance for your help! Here is the ticket:
[TICKET_PAYLOAD]"""

col_pr1, col_pr2 = st.columns([1, 1])

with col_pr1:
    st.subheader("📥 Verbose Input Prompt")
    user_prompt = st.text_area("Prompt Text", value=sample_verbose_prompt, height=180)
    apply_filler_strip = st.checkbox("Strip Conversational Filler & Politeness", value=True)
    apply_xml_tagging = st.checkbox("Encapsulate into XML Schema", value=True)

with col_pr2:
    st.subheader("📤 Optimized Output & Efficiency Metrics")
    
    # Calculate word count estimations
    orig_words = len(user_prompt.split())
    orig_tokens_est = int(orig_words * 1.3)
    
    # Prune
    cleaned = user_prompt
    if apply_filler_strip:
        filler_phrases = [
            r"Hello there!", r"Could you please be so kind as to", r"for me\?",
            r"I would really appreciate it if you could", r"carefully read every single line and",
            r"In addition to that, please try to", r"if it happens to be present in the text",
            r"Thank you very much in advance for your help!", r"Here is the ticket:"
        ]
        for fp in filler_phrases:
            cleaned = re.sub(fp, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if apply_xml_tagging:
        cleaned = f"""<instructions>Analyze ticket sentiment and extract account_id.</instructions>
<input>[TICKET_PAYLOAD]</input>
<output_format>{{"sentiment": "ANGRY" | "HAPPY" | "NEUTRAL", "account_id": str or null}}</output_format>"""

    opt_words = len(cleaned.split())
    opt_tokens_est = int(opt_words * 1.3)
    savings_pct = max(0, int(((orig_tokens_est - opt_tokens_est) / orig_tokens_est) * 100)) if orig_tokens_est > 0 else 0

    st.code(cleaned, language="xml" if apply_xml_tagging else "text")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original Tokens (est)", orig_tokens_est)
    m2.metric("Optimized Tokens (est)", opt_tokens_est)
    m3.metric("Token Savings", f"{savings_pct}%", delta=f"-{orig_tokens_est - opt_tokens_est} tokens")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Programmatic Token Pruning", divider="rainbow")
st.markdown("Python function to automate prompt minification and stop-phrase stripping across automated pipelines:")

st.code(
    '''import re

FILLER_PATTERNS = [
    r"\\b(please|could you|kindly|can you|would you mind|in order to|i want you to)\\b",
    r"\\b(thank you|thanks in advance|much appreciated|let me know if)\\b",
    r"\\b(carefully|thoroughly|step by step|in a detailed manner)\\b",
]

def prune_prompt_verbosity(raw_prompt: str) -> str:
    """
    Automated pre-processing pipeline to remove polite conversational filler,
    normalize repeated whitespace, and reduce input token footprint by 20-35%.
    """
    pruned = raw_prompt
    for pattern in FILLER_PATTERNS:
        pruned = re.sub(pattern, "", pruned, flags=re.IGNORECASE)
    
    # Normalize multiple whitespace, tabs, and duplicate newlines
    pruned = re.sub(r' +', ' ', pruned)
    pruned = re.sub(r'\\n\\s*\\n+', '\\n\\n', pruned)
    return pruned.strip()

# Demonstration
sample = "Please kindly analyze this text and thoroughly extract all entities, thank you!"
compacted = prune_prompt_verbosity(sample)
print(f"Original:  {sample}")
print(f"Compacted: {compacted}")
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
    with st.expander("Q1: What is LLMLingua and how does it compress prompts without degrading reasoning?", expanded=True):
        st.markdown(
            """
- **Mechanism**: LLMLingua uses a smaller, lightweight model (like LLaMA-2-7B or GPT-2-small) to compute token-level perplexity across the prompt. Tokens with low perplexity (highly predictable, redundant grammatical words) contain minimal task-critical information and are safely dropped.
- **Dynamic Budgeting**: It allocates token budgets dynamically between instructions, demonstrations, and context, achieving 2x to 5x prompt compression with minimal loss in downstream benchmark accuracy.
"""
        )
    with st.expander("Q2: Why does reducing prompt token length directly improve time-to-first-token (TTFT)?"):
        st.markdown(
            """
In transformer inference, the **prefill phase** (processing the prompt tokens) requires quadratic self-attention across the entire input sequence. Reducing prompt length from 4,000 tokens to 1,500 tokens drastically cuts the matrix computation required to compute the initial KV-cache, resulting in a substantially faster Time-to-First-Token (TTFT) and reduced memory pressure on GPU VRAM.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Does aggressive prompt compression ever hurt performance?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"No, fewer tokens always means better, faster prompts."*
- **Winning Response**: *"Yes, over-compression can destroy semantic nuances. Pruning essential domain terminology, stripping disambiguating context, or removing few-shot examples can trigger hallucinations or formatting drift. Production compression must always be validated against an automated regression evaluation benchmark to verify that accuracy SLAs remain uncompromised."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt Compression**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The technique of reducing prompt token count (via semantic pruners or syntactic minification) while preserving downstream output accuracy.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prefill Phase / TTFT**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The initial transformer phase that processes all prompt tokens to populate the KV-cache, directly determining Time-to-First-Token.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Imperative Phrasing**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Replacing conversational politeness ('Please would you kindly...') with direct imperative commands ('Extract...', 'Classify...').")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 12.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Cost & Latency**: Prompt optimization is critical for financial sustainability and fast API response times.
2. **Strip Politeness**: Models are mathematical algorithms; polite conversational filler wastes tokens.
3. **Structured Schemas**: Terse XML and JSON schemas save tokens compared to verbose explanatory paragraphs.
4. **Validate with Benchmarks**: Always ensure compression does not cause accuracy regression on core tasks.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to identify and prune conversational filler from prompt templates.", value=True, key="chk_12_2_1")
    st.checkbox("I understand how prompt compression reduces Time-to-First-Token (TTFT).", value=True, key="chk_12_2_2")
    st.checkbox("I can implement programmatic regex token pruners in Python.", value=True, key="chk_12_2_3")
    st.checkbox("I understand the trade-offs of semantic compression algorithms like LLMLingua.", value=True, key="chk_12_2_4")
