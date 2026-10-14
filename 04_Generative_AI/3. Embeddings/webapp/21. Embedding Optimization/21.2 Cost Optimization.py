import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="21.2 Cost Optimization - Embeddings Curriculum",
    page_icon="💰",
    layout="wide"
)

# Title & Overview
st.title("21.2 Cost Optimization: Deduplication, Token Pruning, Query Caching, & Model Selection")
st.caption("Chapter 21: Embedding Optimization | Domain: 3. Vector Embeddings")
st.markdown("""
At prototype scale (100 documents), embedding API costs are negligible (fractions of a cent).
However, at enterprise production scale—indexing **10,000,000 documents across corporate drives** and serving **500,000 user queries daily**—unoptimized embedding pipelines can drain thousands of dollars every month in redundant API fees.

The primary cost leaks in enterprise AI systems are:
* **The Boilerplate Tax**: 30% to 50% of raw HTML, PDF headers, legal disclaimers, and navigation bars provide zero semantic signal but consume paid tokens.
* **The Re-Embedding Trap**: Re-indexing updated folders re-embeds millions of unchanged documents.
* **The Query Redundancy Drain**: Thousands of users ask the exact same FAQ questions repeatedly (*"How do I reset my password?"*, *"Where are my tax forms?"*), burning query embedding tokens every time.
* **Model Oversizing**: Using expensive 3072-dimensional flagship models where lightweight or open-source local models achieve identical accuracy.

This lesson explores how to implement a multi-layered **Cost Shield** to reduce embedding expenditure by **60% to 80%**.
""")

st.divider()

# Section 1: The 5 Cost Reduction Pillars
st.header("1. The Multi-Tier Cost Shield Architecture", divider="rainbow")

col_cost1, col_cost2 = col_cost1, col_cost2 = st.columns([1, 1])

with col_cost1:
    st.markdown("""
    ### The 5 Cost Reduction Pillars
    * **1. Boilerplate & Whitespace Pruning**:
      * Stripping HTML tags, CSS blocks, base64 strings, page headers, footers, and copyright disclaimers before tokenization. Typically cuts total token volume by **30% to 45%**!
    * **2. Content Hash Deduplication**:
      * Ingesting documents by SHA-256 content hashes. Skips re-embedding identical files across department shared drives.
    * **3. Semantic Query Caching (GPTCache / Redis)**:
      * Caching common user queries. If an incoming query has $> 0.95$ cosine similarity to a recently cached query, return the cached result instantly ($0 cost, 0ms API time).
    * **4. Model Tier Right-Sizing**:
      * Using `text-embedding-3-small` ($0.02 / 1M tokens) instead of `text-embedding-3-large` ($0.13 / 1M tokens) when 1536 dimensions suffice, saving **85% on API bills**.
      * Alternatively, deploying open-source models (`bge-small-en` via FastEmbed/ONNX) locally at **$0 API cost**.
    * **5. Bulk Batching Tiers**:
      * OpenAI and cloud providers offer batch APIs at a **50% discount** for non-urgent offline ingestion jobs processed within 24 hours.
    """)

with col_cost2:
    st.markdown("### The Enterprise Cost Shield Pipeline")
    cost_dot = """
    digraph CostShield {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserPrompt [label="Incoming User Query\\n'How to reset VPN password?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_query_shield {
            label="Layer 1: Semantic Query Cache"; style="dashed"; color="#10B981";
            QCacheCheck [label="Semantic Cache Check (Redis)\\nSimilarity >= 0.95?", fillcolor="#D1FAE5"];
            QCacheHit [label="Cache Hit: Instant Return\\n($0 API Cost | 0ms Network)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        subgraph cluster_ingest_shield {
            label="Layer 2: Ingestion Pruning & Batching"; style="dashed"; color="#8B5CF6";
            Prune [label="Text Pruning & Cleaning\\nStrip HTML, disclaimers (-35% tokens)", fillcolor="#EDE9FE"];
            BatchCall [label="Batch API Call\\n(50% Discount Batch Tier)", fillcolor="#C4B5FD", color="#7C3AED", style="filled,bold"];
        }

        UserPrompt -> QCacheCheck;
        QCacheCheck -> QCacheHit [label="Hit (Frequent FAQ)"];
        QCacheCheck -> Prune [label="Miss (Unique Query)"];
        Prune -> BatchCall;
    }
    """
    st.graphviz_chart(cost_dot, use_container_width=False)

st.markdown("### Model Pricing & Cost Comparison (10 Million Tokens)")
st.markdown("""
| Model Architecture | Provider | Cost / 1M Tokens | Cost for 10M Tokens | Relative Cost Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **`text-embedding-3-large`** | OpenAI Cloud | $0.13 | **$1.30** | 6.5x (Expensive) |
| **`text-embedding-3-small`** | OpenAI Cloud | $0.02 | **$0.20** | 1.0x (Baseline) |
| **`text-embedding-3-small (Batch API)`** | OpenAI Batch (24h) | $0.01 | **$0.10** | **0.5x (50% Off)** |
| **`Cohere embed-english-v3.0`** | Cohere Cloud | $0.10 | **$1.00** | 5.0x |
| **`BGE-base-en-v1.5 (Local ONNX)`** | Self-Hosted / FastEmbed | **$0.00** | **$0.00 (Free)** | **0.0x (100% Free)** |
""")

st.divider()

# Section 2: Production Code: Cost-Optimized Gateway
st.header("2. Production Code: Cost Shield Gateway & Text Pruner", divider="rainbow")

st.markdown("""
The production utility below demonstrates an enterprise Cost Shield Gateway: pruning redundant HTML/disclaimer boilerplate, caching query embeddings in memory, and tracking token savings.
""")

st.code(r'''
import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class CostOptimizedEmbeddingGateway:
    """
    Minimizes embedding API costs by:
    - Pruning boilerplate, HTML, and duplicate whitespace
    - Caching exact and semantic query vectors
    - Tracking dollar and token savings
    """
    BOILERPLATE_PATTERNS = [
        r'<[^>]+>',                         # HTML tags
        r'CONFIDENTIALITY NOTICE:.*',       # Email confidentiality disclaimers
        r'Copyright © \d{4}.*reserved\.',   # Copyright footers
        r'http[s]?://\S+',                  # Raw URLs
        r'\s+'                              # Multiple spaces/newlines
    ]

    def __init__(self, cost_per_1m_tokens: float = 0.02):
        self.cost_per_1m = cost_per_1m_tokens
        self.query_cache: Dict[str, np.ndarray] = {}
        self.tokens_saved = 0
        self.tokens_billed = 0

    def prune_boilerplate(self, raw_text: str) -> Tuple[str, int]:
        """Strips useless boilerplate and returns cleaned text with saved tokens."""
        original_words = len(raw_text.split())
        cleaned = raw_text

        # 1. Strip HTML tags
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        # 2. Strip standard legal disclaimers
        cleaned = re.sub(r'CONFIDENTIALITY NOTICE:.*', ' ', cleaned, flags=re.IGNORECASE)
        # 3. Strip copyright notices
        cleaned = re.sub(r'Copyright © \d{4}.*all rights reserved\.?', ' ', cleaned, flags=re.IGNORECASE)
        # 4. Collapse whitespace
        cleaned = " ".join(cleaned.split())

        cleaned_words = len(cleaned.split())
        # Estimate ~1.3 tokens per word
        tokens_original = int(original_words * 1.3)
        tokens_cleaned = int(cleaned_words * 1.3)
        saved = max(0, tokens_original - tokens_cleaned)

        self.tokens_saved += saved
        self.tokens_billed += tokens_cleaned
        return cleaned, saved

    def get_query_embedding(self, query: str) -> Tuple[np.ndarray, bool]:
        """Returns embedding from cache if exists; otherwise computes and caches."""
        q_norm = " ".join(query.strip().lower().split())
        q_hash = hashlib.sha256(q_norm.encode("utf-8")).hexdigest()

        if q_hash in self.query_cache:
            # Cache hit: 0 tokens billed!
            saved_tokens = int(len(q_norm.split()) * 1.3)
            self.tokens_saved += saved_tokens
            return self.query_cache[q_hash], True

        # Cache miss: simulate embedding computation
        np.random.seed(abs(hash(q_norm)) % (2**31 - 1))
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.maximum(np.linalg.norm(vec), 1e-12)

        self.query_cache[q_hash] = vec
        self.tokens_billed += int(len(q_norm.split()) * 1.3)
        return vec, False
''', language="python")

st.divider()

# Section 3: Interactive Cost Optimization & Savings Calculator
st.header("3. Interactive Workbench: Enterprise RAG ROI & Cost Calculator", divider="rainbow")

st.markdown("""
Calculate how much money and compute your organization will save by implementing the 4 cost reduction levers on an enterprise corpus.
""")

col_calc1, col_calc2 = st.columns([1, 1])

with col_calc1:
    st.subheader("1. Organization Scale Parameters")
    doc_corpus_count = st.number_input("Corpus Chunk Count:", min_value=10000, max_value=50000000, value=500000, step=50000)
    avg_chunk_tokens = st.slider("Average Raw Tokens per Chunk:", min_value=100, max_value=1000, value=400, step=50)
    daily_queries = st.number_input("Daily User Queries:", min_value=1000, max_value=5000000, value=50000, step=5000)
    avg_query_tokens = st.slider("Average Tokens per Query:", min_value=5, max_value=50, value=15, step=5)

    st.subheader("2. Model Selection")
    chosen_model_cost = st.selectbox(
        "Embedding Provider & Tier:",
        [
            ("OpenAI text-embedding-3-small ($0.02 / 1M)", 0.02),
            ("OpenAI text-embedding-3-large ($0.13 / 1M)", 0.13),
            ("Cohere embed-english-v3.0 ($0.10 / 1M)", 0.10),
            ("OpenAI Batch API Tier 50% Off ($0.01 / 1M)", 0.01),
            ("Self-Hosted Open-Source BGE ($0.00 / 1M API)", 0.00)
        ],
        format_func=lambda x: x[0]
    )[1]

    st.subheader("3. Optimization Levers Active")
    enable_pruning = st.checkbox("Enable Boilerplate & HTML Pruning (-35% tokens)", value=True)
    enable_dedup = st.checkbox("Enable Content Hash Deduplication (-20% chunks)", value=True)
    enable_qcache = st.checkbox("Enable Semantic Query Caching (40% cache hit rate)", value=True)

with col_calc2:
    st.subheader("4. Monthly Cost & Savings Dashboard")

    # Raw unoptimized calculations (Monthly)
    raw_ingest_tokens = doc_corpus_count * avg_chunk_tokens
    raw_query_tokens_monthly = daily_queries * avg_query_tokens * 30
    total_unoptimized_tokens = raw_ingest_tokens + raw_query_tokens_monthly
    cost_unoptimized = (total_unoptimized_tokens / 1000000.0) * chosen_model_cost

    # Optimized calculations
    eff_chunks = doc_corpus_count * (0.80 if enable_dedup else 1.0)
    eff_chunk_tokens = avg_chunk_tokens * (0.65 if enable_pruning else 1.0)
    opt_ingest_tokens = eff_chunks * eff_chunk_tokens

    eff_queries = daily_queries * (0.60 if enable_qcache else 1.0)
    opt_query_tokens_monthly = eff_queries * avg_query_tokens * 30
    total_optimized_tokens = opt_ingest_tokens + opt_query_tokens_monthly
    cost_optimized = (total_optimized_tokens / 1000000.0) * chosen_model_cost

    saved_dollars = max(0.0, cost_unoptimized - cost_optimized)
    savings_pct = (saved_dollars / max(1e-6, cost_unoptimized)) * 100.0 if cost_unoptimized > 0 else 0.0

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Unoptimized Cost", f"${cost_unoptimized:.2f} / mo")
    c_m2.metric("Optimized Cost", f"${cost_optimized:.2f} / mo", delta=f"-${saved_dollars:.2f}")
    c_m3.metric("Cost Reduction", f"{savings_pct:.1f}%")

    st.markdown("#### Monthly Token Volume Breakdown")
    df_token_breakdown = pd.DataFrame([
        {"Category": "Ingestion Tokens", "Unoptimized": f"{raw_ingest_tokens:,}", "Optimized": f"{int(opt_ingest_tokens):,}", "Tokens Saved": f"{int(raw_ingest_tokens - opt_ingest_tokens):,}"},
        {"Category": "Query Tokens (Monthly)", "Unoptimized": f"{raw_query_tokens_monthly:,}", "Optimized": f"{int(opt_query_tokens_monthly):,}", "Tokens Saved": f"{int(raw_query_tokens_monthly - opt_query_tokens_monthly):,}"},
        {"Category": "TOTAL BILLED TOKENS", "Unoptimized": f"{total_unoptimized_tokens:,}", "Optimized": f"{int(total_optimized_tokens):,}", "Tokens Saved": f"{int(total_unoptimized_tokens - total_optimized_tokens):,}"}
    ])
    st.dataframe(df_token_breakdown, use_container_width=True)

    st.success(f"🎉 By implementing these 3 levers, your organization saves **${saved_dollars:.2f} every month** (**${saved_dollars * 12:.2f} annually**)!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Cost Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does Semantic Query Caching (e.g. GPTCache) work and how does it differ from exact match caching?**
    * **Answer**:
      * **Exact Match Caching**: Uses a hash of the query string (`hash("how to reset password")`). If a user asks *"where do I change my password?"*, it results in a cache miss despite identical semantic intent.
      * **Semantic Query Caching**: Stores previous queries as embedding vectors in a small in-memory index. When a new query arrives:
        1. It embeds the query and searches the query cache.
        2. If the cosine similarity to an existing cached query exceeds a high threshold (e.g. $\\ge 0.96$), it retrieves the cached answer directly.
        3. This bypasses the entire main document vector search and LLM inference loop, delivering $0 cost and $< 10\\text{ms}$ latency for common questions.

    **Q2: What is the 'Boilerplate Tax' in document ingestion?**
    * **Answer**: Raw enterprise documents (PDFs, HTML scrapes, email threads) frequently contain recurring non-informative text: legal confidentiality disclaimers, cookie banners, navigation menus, and corporate copyright footers.
    These repetitive headers and footers consume up to 40% of chunk tokens without contributing semantic signal, unnecessarily driving up both embedding API costs and LLM prompt generation costs.

    **Q3: When should you use Cloud Embedding Batch APIs (e.g. OpenAI Batch)?**
    * **Answer**: Cloud providers offer asynchronous Batch endpoints that execute large-scale embedding jobs within 24 hours at a **50% discount** compared to standard synchronous endpoints. They are ideal for initial corpus ingestions, weekly database re-indexing, or migration pipelines that do not require real-time interactive responses.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Defaulting to Flagship Models Without Benchmarking**:
      * *Issue*: Blindly deploying `text-embedding-3-large` ($0.13/1M) across millions of documents when `text-embedding-3-small` ($0.02/1M) scores within 1% on your validation set.
      * *Consequence*: Spending 6.5x more on API bills with zero perceptible user benefit.
      * *Fix*: Always benchmark small models first; only upgrade if validation recall demonstrably improves.

    * **Trap 2: Cache Key Collisions Across Embedding Models**:
      * *Issue*: Caching vectors under a plain content hash without prefixing the embedding model name.
      * *Consequence*: When you switch from model A (384 dim) to model B (1536 dim), the cache returns stale 384-dim vectors, crashing the vector store.
      * *Fix*: Always construct cache keys as `Key = ModelName + ":" + Version + ":" + ContentHash`.

    * **Trap 3: Caching Outdated Query Results Indefinitely**:
      * *Issue*: Setting infinite TTL on query caches for documents that change frequently.
      * *Consequence*: Users receive outdated answers to company policy questions even after policies were updated.
      * *Fix*: Set conservative TTLs (e.g. 24 hours) or invalidate cache keys when related documents are modified.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Query Caching**
      * *Definition*: Caching search queries based on vector cosine proximity (e.g. $> 0.95$) to return cached answers for semantically equivalent prompts.
    * **Flashcard 2: Boilerplate Tax**
      * *Definition*: The financial waste of paying embedding and LLM token fees for non-informative legal disclaimers, headers, and footers.
    * **Flashcard 3: Cloud Batch Tier**
      * *Definition*: Asynchronous 24-hour API endpoints offering a 50% discount for non-urgent bulk document ingestion.
    * **Flashcard 4: Content Hash Deduplication**
      * *Definition*: Hashing chunk strings with SHA-256 to ensure identical paragraphs across different documents are only embedded once.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Cost Optimization:

- [ ] I can articulate the 5 pillars of the Enterprise Cost Shield.
- [ ] I understand how boilerplate pruning reduces token volume by up to 45%.
- [ ] I know how Semantic Query Caching differs from exact-match caching.
- [ ] I can calculate monthly RAG costs and ROI savings across model tiers.
- [ ] I know when to leverage 50% off cloud Batch API tiers for bulk ingestion.
""")
