import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="7.1 Understanding Context | GenAI Roadmap",
    page_icon="🌐",
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
st.title("🌐 7.1 Understanding Context — Attention Dynamics & Working Memory")
st.markdown(
    """
Welcome to **Lesson 7.1: Understanding Context**.  
While training weights represent an LLM's frozen, long-term parametric knowledge, the **Context Window** represents its active **working memory**. 
Everything a model reasons about at runtime—instructions, reference documents, few-shot examples, and user queries—must compete for attention within this finite window.

Below, we master the fundamental mechanics of **Context Engineering**, analyze context density and ordering, examine the famous **'Lost in the Middle'** phenomenon, and explore practical limits.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is Context? Working Memory vs Frozen Weights
# ==============================================================================
st.header("1. What is Context? Working Memory vs Frozen Weights", divider="rainbow")

col_c1, col_c2 = st.columns([1, 1])

with col_c1:
    st.markdown(
        """
### Definition
**Context** is the sequence of input tokens currently loaded into a Transformer's self-attention mechanism during an inference forward pass.

### Parametric Memory vs In-Context Memory:
- **Parametric Memory (Model Weights)**: Frozen at pre-training time. Massive (billions of parameters), but static, prone to out-of-date facts, and expensive to modify.
- **In-Context Memory (Prompt Window)**: Dynamic, live working memory. Allows instant grounding with current facts, proprietary documents, and transient conversation state with **zero model retraining**.

### The Cost of Context:
Every token loaded into context incurs:
1. **Computational Cost**: Attention scales with context length; processing long contexts consumes VRAM and GPU compute.
2. **Key-Value (KV) Cache Memory**: Storing attention keys and values consumes gigabytes of server memory per concurrent user.
3. **Financial Cost**: API providers charge per input token.
"""
    )

with col_c2:
    st.markdown("### The Transformer Working Memory Stack (ASCII)")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ CONTEXT WINDOW (Live Working Memory: e.g. 128k Tokens) │
│                                                        │
│  [STATIC SYSTEM INSTRUCTIONS]  (Priority 1: Top)       │
│  [FEW-SHOT DEMONSTRATIONS]     (Priority 2: Upper)     │
│  [RETRIEVED RAG DOCUMENTS]     (Priority 3: Middle)    │
│  [RUNTIME USER QUERY]          (Priority 4: Bottom)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ MULTI-HEAD SELF-ATTENTION                              │
│ Every token computes attention weights with all others │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ PARAMETRIC MEMORY (Static Weights: Frozen Knowledge)   │
└────────────────────────────────────────────────────────┘""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 2: The 5 Dimensions of Context Engineering
# ==============================================================================
st.header("2. The 5 Dimensions of Context Engineering", divider="rainbow")
st.markdown("Managing context is an optimization problem across five architectural dimensions:")

dims_df = pd.DataFrame({
    "Dimension": [
        "1. Context Window Size",
        "2. Context Relevance",
        "3. Context Density",
        "4. Context Ordering",
        "5. Effective Retrieval Limits"
    ],
    "Architectural Reality": [
        "Advertised window (e.g. 128K, 1M, 2M tokens) vs real compute latency",
        "Ratio of task-critical signal tokens to irrelevant conversational noise",
        "Information packed per token (dense JSON vs verbose conversational transcripts)",
        "Positioning of key facts relative to the start, middle, and end of prompt",
        "The difference between what fits in context and what the model can actually recall"
    ],
    "Best Practice Rule": [
        "Keep contexts as lean as possible; do not stuff unneeded tokens simply because they fit",
        "Rigorously filter RAG chunks; irrelevant chunks degrade answer accuracy by up to 35%",
        "Strip markdown fluff, boilerplate headers, and CSS before context injection",
        "Place critical instructions at the very beginning and target query at the very end",
        "Test with synthetic 'Needle in a Haystack' audits before relying on massive windows"
    ]
})
st.dataframe(dims_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: The 'Lost in the Middle' Phenomenon
# ==============================================================================
st.header("3. The 'Lost in the Middle' Phenomenon", divider="rainbow")
st.markdown(
    """
Empirical research (Liu et al., Stanford / UC Berkeley) reveals that **LLM attention is NOT uniform across the context window**. 
Instead, retrieval performance follows a **U-shaped curve**:
- **Primacy Effect**: Models recall information placed at the **very beginning** of the prompt with high accuracy (~95%).
- **Recency Effect**: Models recall information placed at the **very end** of the prompt with high accuracy (~98%).
- **The Trough of Forgetting**: Information buried in the **middle 40%–70%** of a long context window suffers from significant retrieval degradation, dropping accuracy to as low as 50–60%!
"""
)

# Simulated U-Shaped Curve Chart
st.markdown("#### Empirical Retrieval Accuracy vs Token Position (U-Shaped Curve):")
positions = ["Start (0-10%)", "Early (10-30%)", "Middle (30-60%)", "Late (60-85%)", "End (85-100%)"]
accuracy = [96.2, 78.4, 52.8, 74.1, 97.5]

u_curve_data = pd.DataFrame({"Token Position in Context": positions, "Retrieval Recall Rate (%)": accuracy})
st.bar_chart(u_curve_data.set_index("Token Position in Context"))

st.divider()

# ==============================================================================
# Section 4: Production Ordering Strategy (The Sandwich Pattern)
# ==============================================================================
st.header("4. Production Context Ordering: The Sandwich Pattern", divider="rainbow")
st.markdown(
    """
To defeat the 'Lost in the Middle' problem, enterprise prompt engineers use the **Sandwich Pattern**:
1. **Top (Primacy Anchor)**: System instructions, primary task directives, and safety constraints.
2. **Middle (Passive Data Sponge)**: Reference documents, RAG chunks, and historical transcripts.
3. **Bottom (Recency Anchor)**: Re-state the primary instruction, specify output format, and supply the runtime user query!
"""
)

st.code(
    '''# The Production Sandwich Pattern Layout:
prompt_sandwich = """
### [TOP ANCHOR: SYSTEM DIRECTIVES & CONSTRAINTS]
You are a Principal Compliance Auditor.
Answer the audit query strictly using the documents enclosed in <reference_corpus>.
If the answer is absent from the corpus, output: "NOT_FOUND".

### [MIDDLE SPONGE: UNSTRUCTURED DATA PAYLOAD]
<reference_corpus>
[Doc 1: Financial Report 2025...]
[Doc 2: Vendor Risk Register...]
[Doc 3: Executive Minutes...]
</reference_corpus>

### [BOTTOM ANCHOR: QUERY & FORMAT CONTRACT]
REMINDER: Answer strictly from <reference_corpus>. No assumptions.
Format strictly as JSON: {"audit_finding": string, "citation": string}

AUDIT QUERY:
What was the maximum liability cap agreed upon in Vendor Agreement B?

JSON RESPONSE:"""''',
    language="text"
)

st.divider()

# ==============================================================================
# Section 5: Interactive Context Ordering Workbench
# ==============================================================================
st.header("5. Interactive Workbench: The Position Sensitivity Tester", divider="rainbow")
st.markdown("Experiment with how placing a critical fact at different positions impacts attention focus:")

target_fact_pos = st.radio(
    "Where should the secret target fact ('Server Admin Password: Alpha99#') be positioned?",
    ["Option A: Beginning of Context (Top 5%)", "Option B: Buried in Middle (50% mark)", "Option C: End of Context (Bottom 5%)"],
    index=1,
    horizontal=True
)

if "Beginning" in target_fact_pos:
    st.success("🎯 **Beginning Position**: High Primacy Weight. The attention heads bind to the secret token during initial prefix encoding. Expected Retrieval Rate: ~96%.")
elif "Middle" in target_fact_pos:
    st.error("⚠️ **Middle Position (The Trough of Forgetting)**: Surrounding distractor documents dilute attention. High probability of hallucination or 'Information not found' false negative. Expected Retrieval Rate: ~53%.")
else:
    st.success("🎯 **End Position**: Maximum Recency Weight. Autoregressive attention heads attend directly to the most recent tokens. Expected Retrieval Rate: ~98%.")

st.divider()

# ==============================================================================
# Section 6: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("6. Context Ingestion & Allocation Topology", divider="rainbow")

col_ci1, col_ci2 = st.columns([1, 1])

with col_ci1:
    st.markdown("#### The Context Budget Allocation Architecture")
    dot_ctx = """
    digraph ContextBudget {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Budget [label="Total Context Budget\\n(e.g. 16,000 Tokens Max)", fillcolor="#334155", fontcolor="#ffffff"];
        Top [label="Top Anchor (10%)\\nSystem Prompt & Rules", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        RAG [label="Middle Sponge (70%)\\nReranked RAG Chunks", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Bottom [label="Bottom Anchor (20%)\\nQuery & Output Contract", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Inference [label="KV-Cached LLM Forward Pass", fillcolor="#047857", fontcolor="#ffffff"];

        Budget -> Top;
        Budget -> RAG;
        Budget -> Bottom;
        Top -> Inference;
        RAG -> Inference;
        Bottom -> Inference;
    }
    """
    st.graphviz_chart(dot_ctx, use_container_width=False)

with col_ci2:
    st.markdown("#### The 3 Rules of Context Hygiene")
    st.markdown(
        """
1. **Never stuff raw HTML or JSON dumps**: Strip unnecessary whitespace, style tags, and tracking scripts before context loading.
2. **Sort by Relevance**: When injecting RAG chunks, sort the **most relevant chunk at the top** and the **second most relevant at the very bottom**.
3. **Filter aggressively**: 3 highly relevant chunks outperform 20 noisy chunks every time.
"""
    )

st.divider()

# ==============================================================================
# Section 7: Technical Interview Preparation
# ==============================================================================
st.header("7. Technical Interview Preparation", divider="rainbow")

ct_tab1, ct_tab2, ct_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with ct_tab1:
    with st.expander("Q1: Explain the 'Lost in the Middle' phenomenon and why it happens in Transformers.", expanded=True):
        st.markdown(
            """
The 'Lost in the Middle' phenomenon refers to the empirical finding that LLMs are significantly worse at retrieving and reasoning about information located in the middle of long input contexts compared to the beginning or end.  
**Root Causes**:
1. **Positional Encoding Bias**: Positional embeddings (RoPE, ALiBi) decay attention scores over long distances, making tokens far from the query less salient.
2. **Pre-training Data Distribution**: Natural documents (articles, papers, emails) naturally place key theses at the introduction and conclusions, biasing the model's learned attention priors.
3. **Instruction Recency**: In autoregressive generation, tokens immediately preceding the output generation step exert the strongest immediate conditional influence.
"""
        )

    with ct_tab2:
        with st.expander("Trap 1: 'Since models have 1M token context windows, we no longer need RAG or vector databases.'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, with 1M tokens you can just dump the entire company documentation into the prompt."*
- **Winning Response**: *"No. First, sending 1M tokens on every query creates massive financial cost ($5–$20 per call) and unacceptable latency (30–60s TTFT). Second, 'Needle in a Haystack' performance degrades significantly when multi-hop reasoning or aggregate analysis is required across 1M tokens. Third, RAG drastically reduces noise, ensures deterministic source citations, and allows sub-second query retrieval. Large context windows complement RAG by allowing richer retrieved passages; they do not eliminate the need for retrieval architectures."*
"""
            )

    with ct_tab3:
        c_ct1, c_ct2, c_ct3 = st.columns(3)
        with c_ct1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Lost in the Middle**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The empirical drop in LLM retrieval accuracy for facts buried in the middle 40-70% of a long context window.")

        with c_ct2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**The Sandwich Pattern**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Context ordering strategy: Static instructions at the TOP, reference documents in the MIDDLE, query and schema at the BOTTOM.")

        with c_ct3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Context Density**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The ratio of actionable factual signal to unneeded conversational formatting tokens in a prompt payload.")

st.divider()

# ==============================================================================
# Section 8: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("8. Lesson 7.1 Summary & Self-Assessment", divider="rainbow")

col_cs1, col_cs2 = st.columns(2)
with col_cs1:
    st.markdown(
        """
### Key Takeaways:
1. **Context is Working Memory**: In-context memory grounds reasoning with zero weight updates.
2. **Attention is U-Shaped**: Facts at the beginning and end are recalled with highest fidelity.
3. **The Sandwich Pattern**: Frame large document payloads between top and bottom anchors.
4. **Signal Over Noise**: High context density outperforms bloated context stuffing.
"""
    )

with col_cs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain the Transformer mechanics behind the 'Lost in the Middle' phenomenon.", value=True, key="chk_7_1_1")
    st.checkbox("I know how to design a prompt using the Sandwich Pattern to protect against middle-retrieval decay.", value=True, key="chk_7_1_2")
    st.checkbox("I understand why million-token context windows do not eliminate the architectural need for RAG.", value=True, key="chk_7_1_3")
    st.checkbox("I can optimize context density by stripping boilerplate and non-informative tokens.", value=True, key="chk_7_1_4")
