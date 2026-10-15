import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="7.3 Context Optimization | GenAI Roadmap",
    page_icon="⚡",
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
st.title("⚡ 7.3 Context Optimization — Compression, Pruning & Memory Management")
st.markdown(
    """
Welcome to **Lesson 7.3: Context Optimization**.  
In high-throughput enterprise GenAI applications, context window bloat is the primary driver of **excessive API latency, high cloud expenditure, and degraded attention quality**. 
Just because a model accepts 128k or 1M tokens does not mean you should fill it.

Below, we master the **6 Essential Context Optimization Techniques**: Semantic compression, Boilerplate pruning, Cross-encoder prioritization, Hierarchical summarization, Intelligent chunking, and Long-context memory management.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Pillars of Context Optimization
# ==============================================================================
st.header("1. The 6 Pillars of Context Optimization", divider="rainbow")
st.markdown("Context optimization balances token savings against semantic fidelity:")

opt_summary_df = pd.DataFrame({
    "Optimization Pillar": [
        "1. Context Compression",
        "2. Irrelevant Token Pruning",
        "3. Context Prioritization (Reranking)",
        "4. Hierarchical Summarization",
        "5. Semantic Chunking",
        "6. Sliding Window Memory"
    ],
    "Core Technique": [
        "Semantic pruning (LLMLingua) removing redundant filler tokens",
        "Stripping raw HTML, tracking scripts, duplicate headers, legal footers",
        "Cross-encoder reranking (Cohere Rerank / BGE-Reranker) to select Top-K",
        "Map-reduce hierarchical compression for multi-document pipelines",
        "Splitting text at sentence, heading, or thematic semantic boundaries",
        "Maintaining fixed token buffers in conversational multi-turn sessions"
    ],
    "Typical Token Reduction": [
        "30% - 60% token reduction",
        "20% - 40% token reduction",
        "50% - 80% token reduction",
        "70% - 90% token reduction",
        "Prevents fragmented chunk cutoff",
        "Caps memory growth at fixed ceiling"
    ],
    "Impact on Latency & Cost": [
        "Slashes TTFT latency by ~40%; cuts input token bill in half",
        "Eliminates distractor noise; improves answer precision",
        "Dramatically reduces hallucination from low-relevance chunks",
        "Allows processing 500-page book corpora in small LLM calls",
        "Ensures complete semantic thoughts remain intact for embedding",
        "Prevents context window overflow in unbounded chat dialogues"
    ]
})
st.dataframe(opt_summary_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1 & 2: Pruning & Compression
with st.container(border=True):
    st.subheader("1 & 2. Boilerplate Stripping & Semantic Token Pruning")
    st.markdown(
        """
Raw documents scraped from web pages or converted from PDFs contain massive amounts of non-informational noise: CSS styles, navigation bars, tracking URLs, and repeated disclaimers. 
Stripping this before context injection yields immediate 30-50% token savings with **zero loss of information**.
"""
    )
    st.code(
        '''import re

def prune_document_noise(raw_text: str) -> str:
    """Removes boilerplate, duplicate whitespace, and structural noise."""
    # 1. Strip HTML tags and script elements
    cleaned = re.sub(r'<script.*?</script>', '', raw_text, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    
    # 2. Strip standard corporate legal footers
    cleaned = re.sub(r'This email and any files transmitted with it are confidential.*', '', cleaned, flags=re.DOTALL)
    
    # 3. Collapse multiple newlines and spaces into single breaks
    cleaned = re.sub(r'\\n\\s*\\n', '\\n\\n', cleaned)
    cleaned = re.sub(r'[ \\t]+', ' ', cleaned)
    
    return cleaned.strip()''',
        language="python"
    )

# 3: Reranking & Prioritization
with st.container(border=True):
    st.subheader("3. Context Prioritization via Cross-Encoder Reranking")
    st.markdown(
        """
Standard bi-encoder vector search (cosine similarity on embeddings) retrieves chunks fast, but often returns false positives. 
A **Cross-Encoder Reranker** scores the joint probability of `(Query, Chunk)` together, allowing you to discard the bottom 70% of retrieved chunks and keep only the true high-signal passages.
"""
    )
    st.code(
        '''from sentence_transformers import CrossEncoder

# Load specialized cross-encoder reranking model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_and_filter_chunks(query: str, raw_chunks: list[str], top_k: int = 3) -> list[str]:
    # Form pairs: [(query, chunk1), (query, chunk2), ...]
    pairs = [(query, chunk) for chunk in raw_chunks]
    scores = reranker.predict(pairs)
    
    # Sort chunks by cross-encoder score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # Select only top_k highest quality chunks
    return [raw_chunks[i] for i in ranked_indices[:top_k]]''',
        language="python"
    )

# 5: Semantic Chunking
with st.container(border=True):
    st.subheader("5. Semantic & Markdown-Aware Chunking")
    st.markdown(
        """
Naive fixed-character chunking (e.g. splitting every 500 characters) cuts words, code blocks, and sentences in half. 
**Markdown-Aware Chunking** splits along structural boundaries (`### Headings`, table borders, paragraphs), ensuring self-contained semantic units.
"""
    )
    st.code(
        '''from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# Payment Architecture
The payment gateway handles credit card tokenization via Stripe.

## Webhook Dispatcher
Webhooks dispatch asynchronously to Redis queues. Retry limit is 3.

## Refund Service
Refunds require manager approval if amount exceeds $500.
"""

headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = markdown_splitter.split_text(markdown_document)
# Each chunk retains its parent header metadata!''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Context Pruning Simulator
# ==============================================================================
st.header("3. Interactive Workbench: The Context Compression Engine", divider="rainbow")
st.markdown("Test an untamed, bloated document snippet and observe the quantitative token reduction:")

raw_bloated_input = """<div>
  <p>CONFIDENTIALITY NOTICE: This transmission is intended only for Acme Corp employees.</p>
  <style>.header { font-size: 14px; color: #333; }</style>
  <h2>API Gateway Rate Limiting Update (Effective Oct 2026)</h2>
  <br/>
  <p>The rate limit for Tier-1 Enterprise endpoints is officially capped at 500 requests per second.</p>
  <br/><br/>
  <p>Please note: All requests exceeding this threshold will receive an HTTP 429 Too Many Requests response.</p>
  <footer>Page 1 of 12 - Acme Internal Infrastructure Guidelines - Do Not Distribute</footer>
</div>"""

col_b1, col_b2 = st.columns(2)
with col_b1:
    st.markdown("**Bloated Raw Input Payload (Raw HTML + Legal Noise):**")
    st.code(raw_bloated_input, language="html")
    est_raw_tokens = len(raw_bloated_input.split()) * 1.3
    st.caption(f"Estimated Input Footprint: ~{int(est_raw_tokens)} tokens")

with col_b2:
    st.markdown("**Optimized & Pruned Context Payload:**")
    # Apply simulated pruning
    pruned_output = """### API Gateway Rate Limiting (Oct 2026):
- Enterprise Tier-1 Rate Limit: 500 req/sec.
- Threshold Breach Behavior: Returns HTTP 429 Too Many Requests."""
    st.code(pruned_output, language="text")
    est_pruned_tokens = len(pruned_output.split()) * 1.3
    st.caption(f"Estimated Optimized Footprint: ~{int(est_pruned_tokens)} tokens")

reduction_pct = (1 - (est_pruned_tokens / est_raw_tokens)) * 100
st.success(f"🎉 **Compression Efficiency**: Reduced prompt tokens by **{reduction_pct:.1f}%** with zero loss of technical meaning!")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. Context Optimization Pipeline Architecture", divider="rainbow")

col_op1, col_op2 = st.columns([1, 1])

with col_op1:
    st.markdown("#### The 4-Stage Optimization Funnel")
    dot_opt = """
    digraph ContextFunnel {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Raw [label="Raw Ingested Documents\\n(15,000 Tokens)", fillcolor="#334155", fontcolor="#ffffff"];
        Clean [label="1. Boilerplate & HTML Pruning\\n(Strip styles, scripts, disclaimers)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Chunk [label="2. Semantic Chunking\\n(Markdown & paragraph boundaries)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Rerank [label="3. Cross-Encoder Reranker\\n(Keep Top-3 Relevant Passages)", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        Lean [label="Optimized Context Stream\\n(850 Tokens -> 94% Savings!)", fillcolor="#047857", fontcolor="#ffffff"];

        Raw -> Clean -> Chunk -> Rerank -> Lean;
    }
    """
    st.graphviz_chart(dot_opt, use_container_width=False)

with col_op2:
    st.markdown("#### The Cost of Unoptimized Context")
    cost_data = pd.DataFrame({
        "Daily Query Volume": ["1,000 queries", "10,000 queries", "100,000 queries"],
        "Unoptimized Cost (8k ctx)": ["$20.00 / day", "$200.00 / day", "$2,000.00 / day"],
        "Optimized Cost (1k ctx)": ["$2.50 / day", "$25.00 / day", "$250.00 / day"],
        "Monthly Savings": ["~$525 / mo", "~$5,250 / mo", "~$52,500 / mo"]
    })
    st.dataframe(cost_data, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

co_tab1, co_tab2, co_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with co_tab1:
    with st.expander("Q1: What is the architectural difference between a Bi-Encoder (Dense Retrieval) and a Cross-Encoder (Reranker)?", expanded=True):
        st.markdown(
            """
- **Bi-Encoder (Vector Search)**: Computes vector embeddings of the Query and Document *separately*. Comparison is a fast dot product or cosine distance. Extremely fast ($O(1)$ with ANN indices like HNSW), but cannot capture token-to-token cross-attention interactions between query and document.
- **Cross-Encoder (Reranker)**: Passes the Query and Document *together* through full multi-head cross-attention layers simultaneously: `[CLS] Query [SEP] Document [SEP]`. Significantly more accurate at judging true relevance, but computationally too slow to search a million documents.  
**Production Standard**: Use Bi-Encoder to retrieve top 25 candidates, then Cross-Encoder to rerank and filter down to the top 3.
"""
        )

    with co_tab2:
        with st.expander("Trap 1: 'Can we compress context by simply deleting all vowels or stopwords?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, removing words like 'the', 'is', 'at' saves tokens without changing meaning."*
- **Winning Response**: *"No. Aggressive mechanical stopword removal corrupts grammatical structures and creates out-of-distribution token combinations that confuse Transformer attention layers. Modern semantic compression (e.g. LLMLingua) uses a small language model to calculate token perplexity and mutual information, pruning only tokens with low information entropy while preserving syntactic coherence."*
"""
            )

    with co_tab3:
        c_co1, c_co2, c_co3 = st.columns(3)
        with c_co1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Cross-Encoder Reranker**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Jointly attends to Query + Document pairs to score true relevance, discarding low-signal candidates.")

        with c_co2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Semantic Chunking**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Splitting text along conceptual/markdown heading boundaries rather than arbitrary character offsets.")

        with c_co3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Boilerplate Pruning**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Stripping HTML, tracking URLs, and legal disclaimers before context loading to eliminate token waste.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 7.3 Summary & Self-Assessment", divider="rainbow")

col_so1, col_so2 = st.columns(2)
with col_so1:
    st.markdown(
        """
### Key Takeaways:
1. **Lean Contexts Win**: Higher token density yields lower latency, lower costs, and higher precision.
2. **Two-Stage Retrieval**: Combine Bi-Encoder candidate search with Cross-Encoder reranking.
3. **Markdown-Aware Splitting**: Never cut sentences or tables in half with fixed character limits.
4. **Strip the Noise**: Pre-process text to remove HTML boilerplate and corporate disclaimers.
"""
    )

with col_so2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain why Cross-Encoder rerankers outperform pure vector search in RAG.", value=True, key="chk_7_3_1")
    st.checkbox("I know how to implement markdown-aware chunking to protect structural boundaries.", value=True, key="chk_7_3_2")
    st.checkbox("I understand the computational cost of bloated contexts on Time-to-First-Token (TTFT).", value=True, key="chk_7_3_3")
    st.checkbox("I can design a two-stage context retrieval and pruning pipeline in Python.", value=True, key="chk_7_3_4")
