import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.3 Cost Optimization - RAG Master Curriculum",
    page_icon="💰",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class CostOptimizerEngine:
    """
    Enterprise FinOps Cost Optimizer & Budget Modeling Engine for RAG Systems.
    Models unit economics across embedding generation, asymmetric LLM token consumption,
    multi-tier model cascading, prompt prefix caching, context pruning, and batch processing.
    """
    def __init__(self):
        # Pricing rates per 1,000,000 tokens (USD)
        self.pricing_catalog = {
            "Frontier_LLM": {"input_per_m": 2.50, "cached_input_per_m": 1.25, "output_per_m": 10.00},
            "SLM_Cascade": {"input_per_m": 0.15, "cached_input_per_m": 0.075, "output_per_m": 0.60},
            "SelfHosted_vLLM": {"input_per_m": 0.05, "cached_input_per_m": 0.05, "output_per_m": 0.08},
            "Embedding_Commercial": {"cost_per_m": 0.02},
            "Embedding_Batch_API": {"cost_per_m": 0.01}
        }

    def compute_query_cost(
        self,
        model_tier: str,
        input_tokens: int,
        output_tokens: int,
        cached_prefix_tokens: int = 0
    ) -> float:
        """
        Computes single query inference cost accounting for cached prefix discounts.
        """
        if model_tier not in self.pricing_catalog:
            model_tier = "Frontier_LLM"

        rates = self.pricing_catalog[model_tier]
        standard_input = max(0, input_tokens - cached_prefix_tokens)

        cost_uncached = (standard_input / 1_000_000.0) * rates["input_per_m"]
        cost_cached = (cached_prefix_tokens / 1_000_000.0) * rates["cached_input_per_m"]
        cost_output = (output_tokens / 1_000_000.0) * rates["output_per_m"]

        return round(cost_uncached + cost_cached + cost_output, 6)

    def simulate_monthly_rag_budget(
        self,
        monthly_queries: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        cache_hit_rate: float,
        cascade_slm_pct: float,
        prompt_compression_pct: float,
        provider_prompt_caching_pct: float,
        use_batch_ingestion: bool,
        docs_to_index: int,
        chunk_size_tokens: int
    ) -> Dict[str, Any]:
        """
        Simulates enterprise monthly RAG budget comparing Unoptimized Baseline
        against Fully Optimized Cascaded Architecture.
        """
        # Baseline: All queries go to Frontier Model with 0% caching, 0% compression
        baseline_cost_per_q = self.compute_query_cost("Frontier_LLM", avg_input_tokens, avg_output_tokens, 0)
        baseline_monthly_llm = monthly_queries * baseline_cost_per_q

        # Baseline Ingestion: Synchronous commercial embedding
        total_chunks = docs_to_index
        total_ingest_tokens = total_chunks * chunk_size_tokens
        baseline_ingest_cost = (total_ingest_tokens / 1_000_000.0) * self.pricing_catalog["Embedding_Commercial"]["cost_per_m"]
        baseline_total = baseline_monthly_llm + baseline_ingest_cost

        # Optimized Architecture Calculations
        effective_queries = monthly_queries * (1.0 - cache_hit_rate)
        cache_saved_queries = monthly_queries * cache_hit_rate

        # Compressed input tokens
        compressed_input = int(avg_input_tokens * (1.0 - (prompt_compression_pct / 100.0)))
        compressed_input = max(100, compressed_input)

        cached_prefix = int(compressed_input * (provider_prompt_caching_pct / 100.0))

        # Cascading Routing: Portion to SLM vs Frontier
        slm_queries = effective_queries * (cascade_slm_pct / 100.0)
        frontier_queries = effective_queries * (1.0 - (cascade_slm_pct / 100.0))

        cost_slm_single = self.compute_query_cost("SLM_Cascade", compressed_input, avg_output_tokens, cached_prefix)
        cost_frontier_single = self.compute_query_cost("Frontier_LLM", compressed_input, avg_output_tokens, cached_prefix)

        optimized_slm_total = slm_queries * cost_slm_single
        optimized_frontier_total = frontier_queries * cost_frontier_single
        optimized_monthly_llm = optimized_slm_total + optimized_frontier_total

        # Ingestion Optimization: Deduplication + Batch API (50% discount)
        dedup_factor = 0.40  # 40% of chunks are unchanged in incremental updates
        new_chunks = total_chunks * (1.0 - dedup_factor)
        new_tokens = new_chunks * chunk_size_tokens

        embed_rate = self.pricing_catalog["Embedding_Batch_API"]["cost_per_m"] if use_batch_ingestion else self.pricing_catalog["Embedding_Commercial"]["cost_per_m"]
        optimized_ingest_cost = (new_tokens / 1_000_000.0) * embed_rate

        optimized_total = optimized_monthly_llm + optimized_ingest_cost
        total_savings = baseline_total - optimized_total
        pct_savings = (total_savings / baseline_total) * 100.0 if baseline_total > 0 else 0.0

        return {
            "baseline_total": round(baseline_total, 2),
            "baseline_llm": round(baseline_monthly_llm, 2),
            "baseline_ingest": round(baseline_ingest_cost, 2),
            "optimized_total": round(optimized_total, 2),
            "optimized_llm": round(optimized_monthly_llm, 2),
            "optimized_ingest": round(optimized_ingest_cost, 2),
            "total_savings": round(total_savings, 2),
            "pct_savings": round(pct_savings, 1),
            "cache_bypassed_queries": int(cache_saved_queries),
            "slm_queries": int(slm_queries),
            "frontier_queries": int(frontier_queries)
        }


# --- HERO SECTION ---
st.title("23.3 Cost Optimization")
st.caption("Embedding Economics, LLM Token Burn Rates, Semantic Caching, Dynamic Model Cascading & Batch Processing")
st.markdown("""
Deploying a RAG proof-of-concept with 50 internal beta users costs pennies; scaling that exact same architecture
to **5 million monthly customer queries across 500,000 corporate documents** can generate a **$65,000 monthly cloud invoice** if left unmanaged.
**Cost Optimization** in enterprise RAG is not about degrading accuracy—it is about **eliminating redundant compute and right-sizing model capacity**:
directing 75% of queries to high-speed Small Language Models (SLMs), caching reusable prompt prefixes and embeddings,
compressing verbose context chunks, and utilizing asynchronous Batch APIs for massive ingestion pipelines.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise FinOps Cost Optimization & Model Cascading Architecture", divider="rainbow")

st.markdown("""
The architecture below illustrates the transformation from an **Unoptimized High-Cost RAG Pipeline**
(where 100% of queries invoke expensive frontier models and uncompressed contexts) to an **Enterprise FinOps Optimized Architecture**
featuring semantic vector caching, dynamic complexity routing, model cascading, provider prompt caching, and batch ingestion.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    cost_blueprint_dot = """
    digraph CostArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_ingress_cache {
            label="1. Ingress & Zero-Cost Caching Tier ($0.00 / query)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            UserQuery [label="User Query Ingress\nHigh-Volume Traffic Stream", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            SemanticCache [label="Semantic Vector Cache\nCosine Match >= 0.90\n(Bypasses LLM Generation)", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            DirectReturn [label="Zero-Cost Answer Return\nCache Hit Rate: ~30%\nCost: $0.00", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_complexity_routing {
            label="2. Complexity Classification & Model Cascading Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            ComplexityRouter [label="Query Complexity Router\nFast DeBERTa / Regex Intent Classifier", shape=diamond, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            Tier1_SLM [label="Tier 1: Economical SLM\nGPT-4o-mini / Haiku 3.5\nInput: $0.15/M | Output: $0.60/M", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#34D399"];
            Tier2_Frontier [label="Tier 2: Flagship Frontier LLM\nGPT-4o / Claude 3.5 Sonnet\nInput: $2.50/M | Output: $10.00/M", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
        }

        subgraph cluster_context_reduction {
            label="3. Context Pruning & Prefix Prompt Caching Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            ContextPruner [label="Context Pruning (LLMLingua)\nStrip Boilerplate & Redundant Tokens\n(-35% Input Token Burn)", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            StaticPrefix [label="Provider Prompt Cache (KV-Cache)\nStatic System & Doc Prefix\n(-50% to -80% Token Cost Discount)", shape=box, fillcolor="#451A03", fontcolor="#FED7AA", color="#F97316"];
        }

        subgraph cluster_batch_ingest {
            label="4. Offline Document Ingestion Tier (Batch API -50%)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#A855F7";
            fontcolor="#D8B4FE";
            fontsize=11;

            DocStream [label="Corpus Ingestion Stream\nPDFs, Wikis, Jira, Confluence", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#A855F7"];
            HashDedup [label="Content Hash Deduplication\nSHA-256 Checksum (Re-embed 0% Unchanged)", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#A855F7"];
            BatchAPI [label="OpenAI / Anthropic Batch API\n24-Hour SLA Ingestion Queue\nFlat 50% Ingestion Discount", shape=component, fillcolor="#581C87", fontcolor="#F3E8FF", color="#C084FC"];
        }

        UserQuery -> SemanticCache;
        SemanticCache -> DirectReturn [label="Hit (30%)", color="#10B981"];
        SemanticCache -> ComplexityRouter [label="Miss (70%)"];

        ComplexityRouter -> Tier1_SLM [label="75% Simple Queries\n(Factoid, FAQ, Lookup)"];
        ComplexityRouter -> Tier2_Frontier [label="25% Complex Queries\n(Multi-Hop, Code, Synthesis)"];

        Tier1_SLM -> ContextPruner;
        Tier2_Frontier -> ContextPruner;
        ContextPruner -> StaticPrefix;

        DocStream -> HashDedup;
        HashDedup -> BatchAPI [label="Modified Chunks Only"];
    }
    """
    st.graphviz_chart(cost_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: EMBEDDING COSTS
# ==============================================================================
st.header("2. Embedding Costs (Batch Vectorization, Dimension Pruning & Open-Source Serving)", divider="rainbow")

st.markdown("""
Embedding generation occurs across two distinct lifecycle phases:
1. **Offline Ingestion**: Embedding millions of historical documents, enterprise wikis, and tickets into the vector database.
2. **Online Query Inference**: Real-time vectorization of incoming user queries.

While query embedding is economically lightweight (e.g., 20 tokens per query @ $0.02 / 1M tokens is negligible),
**bulk offline ingestion** can become extremely costly if full corpuses are re-embedded naively on schedule.
""")

col_e1, col_e2 = st.columns([1, 1])

with col_e1:
    st.markdown("### Embedding Economics by Provider Tier")
    emb_pricing_df = pd.DataFrame({
        "Embedding Provider / Model": ["OpenAI text-embedding-3-small", "OpenAI text-embedding-3-large", "Cohere embed-v3 (Multilingual)", "Self-Hosted BGE-M3 (vLLM / TEI)"],
        "Dimensions": [1536, 3072, 1024, 1024],
        "Cost per 1M Tokens": ["$0.020", "$0.130", "$0.100", "~$0.006 (GPU electricity/compute)"],
        "Batch API Cost / 1M": ["$0.010 (-50%)", "$0.065 (-50%)", "N/A", "N/A"]
    })
    st.dataframe(emb_pricing_df, use_container_width=True)

with col_e2:
    st.markdown("### Dimension Pruning (Matryoshka Embeddings)")
    st.markdown("""
    * **Matryoshka Representation Learning (MRL)**: Modern models like `text-embedding-3-small` allow truncating 1536-dimensional embeddings to 512 dimensions while preserving over 98% of retrieval accuracy.
    * **Downstream Cost Savings**:
      * Truncating from 1536 to 512 dimensions slashes **Vector Database RAM footprint by 66.7%**.
      * Significantly lowers managed vector cloud hosting fees (Pinecone, Milvus Cloud, Qdrant Cloud), where pricing scales directly with index RAM footprint.
    """)

st.divider()

# ==============================================================================
# SECTION 3: LLM TOKEN COSTS
# ==============================================================================
st.header("3. LLM Token Costs (Asymmetric Input/Output Pricing, Token Burn Rates & Daily Budget Caps)", divider="rainbow")

st.markdown("""
In generative RAG architectures, LLM inference accounts for **85% to 95% of total operating expenses**.
Language model APIs enforce an **asymmetric pricing model**, where generated output tokens cost between **3x to 4x more** than input prompt tokens.
However, in RAG systems, the input prompt is vastly larger than the output answer because of retrieved document chunks.
""")

col_tk1, col_tk2 = st.columns([1, 1])

with col_tk1:
    st.markdown("### Anatomy of a 10-Chunk RAG Request Token Burn:")
    st.markdown("""
    * **System Prompt & Instructions**: 350 tokens
    * **Retrieved Chunks (10 chunks @ 400 tokens)**: 4,000 tokens
    * **Conversation History (Last 3 turns)**: 800 tokens
    * **User Question**: 30 tokens
    * **Total Input Prompt**: **5,180 tokens**
    * **Generated Answer**: **250 tokens**
    """)
    st.markdown("""
    Notice that **95.4% of all tokens consumed** are input prompt tokens!
    Therefore, reducing input prompt tokens via context compression and caching yields the highest dollar-for-dollar financial returns.
    """)

with col_tk2:
    st.markdown("### Monthly LLM Cost at Scale (1,000,000 Queries)")
    scale_calc_df = pd.DataFrame({
        "Model Tier": ["Frontier (GPT-4o / Claude 3.5 Sonnet)", "Mid-Tier (GPT-4o-mini / Haiku)", "Open-Source 8B Self-Hosted"],
        "Input Rate / 1M": ["$2.50", "$0.15", "$0.05"],
        "Output Rate / 1M": ["$10.00", "$0.60", "$0.08"],
        "Cost per 1,000 Queries": ["$15.45", "$0.93", "$0.28"],
        "Cost per 1M Queries": ["$15,450.00", "$927.00", "$279.00"]
    })
    st.dataframe(scale_calc_df, use_container_width=True)
    st.caption("Routing standard queries to mid-tier models slashes monthly generation costs from $15,450 down to $927 (94.0% reduction).")

st.divider()

# ==============================================================================
# SECTION 4: CACHING EMBEDDINGS
# ==============================================================================
st.header("4. Caching Embeddings (Redis Vector Cache, Query Hash Invalidation & Cache Hit Yields)", divider="rainbow")

st.markdown("""
A primary FinOps waste pattern in corporate RAG is re-embedding identical document chunks upon every sync cycle.
When documentation pipelines sync nightly from Jira or Confluence, thousands of pages are re-parsed.
Without chunk-level content hashing, entire corpuses are sent to embedding APIs repeatedly.
""")

col_cemb1, col_cemb2 = st.columns([1, 1])

with col_cemb1:
    st.markdown("### Content Hash Deduplication Workflow:")
    st.markdown("""
    1. **Chunk-Level SHA-256 Signature**:
       During document parsing, generate a deterministic hash of the chunk text:
       `chunk_hash = sha256(chunk_text.strip().encode('utf-8')).hexdigest()`.
    2. **Redis Hash Key Check**:
       Query the embedding metadata store. If `chunk_hash` exists, reuse the existing vector directly without calling the embedding API.
    3. **Ingestion Cost Avoidance**:
       In typical enterprise document repositories, **less than 5% of documents change each week**. Content hashing avoids 95% of re-indexing embedding expenses.
    """)

with col_cemb2:
    st.markdown("### Deduplication Ingestion Architecture")
    dot_dedup = """
    digraph EmbedDedup {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        NewDoc [label="Nightly Corpus Sync\n100,000 Chunks Ingested", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        HashCalc [label="Compute SHA-256 Hash\nper Text Chunk", shape=ellipse, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        HashLookup [label="Check Hash in Redis Catalog", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#6366F1"];
        SkipEmbed [label="Reuse Existing Vector\nCost: $0.00 (95,000 chunks)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        CallAPI [label="Call Embedding API\nCost: $0.10 (5,000 chunks)", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];

        NewDoc -> HashCalc -> HashLookup;
        HashLookup -> SkipEmbed [label="Hash Exists (95%)"];
        HashLookup -> CallAPI [label="New / Modified (5%)"];
    }
    """
    st.graphviz_chart(dot_dedup, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: SMALLER MODELS
# ==============================================================================
st.header("5. Smaller Models (Dynamic Model Cascading: SLMs vs Frontier Models & Distillation)", divider="rainbow")

st.markdown("""
Enterprise queries follow a steep power-law distribution:
* **70% - 80% are Simple Queries**: Direct factual lookups, definitions, policy checks, or entity extractions (e.g., *"What is the HR contact number?"*).
* **20% - 30% are Complex Queries**: Multi-document synthesis, cross-source contradiction resolution, code refactoring, or multi-step reasoning.

**Model Cascading** deploys a lightweight query router that sends simple queries to an ultra-fast, low-cost Small Language Model (SLM),
reserving flagship frontier models exclusively for queries that genuinely require high reasoning capacity.
""")

col_cas1, col_cas2 = st.columns([1, 1])

with col_cas1:
    st.markdown("### Complexity Classifier Criteria:")
    st.markdown("""
    * **Heuristic Filter**:
      Queries with token count < 15, no comparative words ('compare', 'contrast', 'synthesize'), and targeting single-entity documents are tagged as `Simple`.
    * **Lightweight Classifier Model**:
      A fine-tuned DeBERTa-v3-small or Llama-3.2-1B model scores query complexity in **~8ms** with zero API fee.
    * **Verification & Fallback Guardrail**:
      If the SLM responds with low confidence or fails self-reflection checks, the orchestrator automatically re-dispatches the request to the Frontier model.
    """)

with col_cas2:
    st.markdown("### Model Cascading Routing Topology")
    dot_cascade = """
    digraph CascadeTopology {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        IncomingQ [label="Retrieved Chunks + User Query", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        Classifier [label="Complexity Evaluator\n(DeBERTa / Prompt Rule)", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        SLM_Worker [label="Tier 1: GPT-4o-mini / Haiku\nCost: $0.15/M | Speed: 110 tok/s\n(Answers 75% of Queries)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        Frontier_Worker [label="Tier 2: Flagship GPT-4o / Sonnet\nCost: $2.50/M | Complex Synthesis\n(Answers 25% of Queries)", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
        ConfidenceCheck [label="SLM Confidence >= 0.85?", shape=diamond, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#6366F1"];
        ReturnUser [label="Deliver Verified Response", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

        IncomingQ -> Classifier;
        Classifier -> SLM_Worker [label="Simple Intent (75%)"];
        Classifier -> Frontier_Worker [label="Complex Intent (25%)"];
        SLM_Worker -> ConfidenceCheck;
        ConfidenceCheck -> ReturnUser [label="Pass"];
        ConfidenceCheck -> Frontier_Worker [label="Low Confidence Fallback"];
        Frontier_Worker -> ReturnUser;
    }
    """
    st.graphviz_chart(dot_cascade, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: CONTEXT COMPRESSION
# ==============================================================================
st.header("6. Context Compression (Prompt Token Trimming, LLMLingua & Extractive Pruning)", divider="rainbow")

st.markdown("""
Since input prompt tokens constitute 95% of RAG token volume, reducing the size of retrieved contexts directly translates into linear dollar savings.
Standard chunking retrieves fixed 500-token blocks, many of which contain irrelevant narrative filler, legal boilerplate, and repeated conversational greetings.

**Context Compression Techniques**:
""")

col_cc1, col_cc2 = st.columns([1, 1])

with col_cc1:
    st.markdown("### 1. Extractive Re-ranking Pruning")
    st.markdown("""
    Instead of passing all Top-10 retrieved chunks to the generator, a fast cross-encoder (or FlashRank ONNX) scores chunks and only passes the Top-3 chunks that exceed a strict relevancy threshold (`relevance_score >= 0.65`).
    **Token Reduction**: Eliminates 7 out of 10 chunks (**70% prompt token reduction**).
    """)

    st.markdown("### 2. Information Entropy Trimming (LLMLingua)")
    st.markdown("""
    LLMLingua calculates the perplexity of each token in the context. Tokens with low conditional perplexity (e.g., articles, predictable connectors, repetitive nouns) are pruned without degrading syntactic semantic clarity.
    **Token Reduction**: Compresses retained chunks by an additional **30% - 40%**.
    """)

with col_cc2:
    st.markdown("### 3. Provider Prompt Caching (KV-Cache Optimization)")
    st.markdown("""
    Modern model providers (Anthropic Claude and OpenAI) offer **Prompt Caching discounts**:
    * If a prompt prefix exceeds a minimum length (e.g., 1,024 tokens) and remains static across requests, subsequent requests that share this prefix receive a **50% to 80% discount on input tokens**.
    * **FinOps Implementation**:
      Structure your prompt deterministically:
      1. Fixed System Prompt (Cached)
      2. Static Documentation Index (Cached)
      3. Dynamic User Query (Uncached, at the bottom)
    """)

st.divider()

# ==============================================================================
# SECTION 7: BATCH PROCESSING
# ==============================================================================
st.header("7. Batch Processing (Offline Embedding Pipelines, Bulk Ingestion & High-Throughput Job Queues)", divider="rainbow")

st.markdown("""
When processing non-real-time workloads—such as historical document backfilling, monthly FAQ regeneration,
or offline evaluation datasets—using synchronous real-time API endpoints wastes 50% of your budget.

Major LLM and embedding providers provide **Batch APIs** (OpenAI Batch API, Anthropic Message Batches) designed for high-throughput, non-urgent workloads:
""")

col_b1, col_b2 = st.columns([1, 1])

with col_b1:
    st.markdown("### Synchronous vs. Batch Processing Economics:")
    batch_comp_df = pd.DataFrame({
        "Attribute": ["API Discount", "Turnaround SLA", "Rate Limit Quotas", "Ideal Use Case"],
        "Synchronous Real-Time API": ["0% (Standard Rates)", "Immediate (< 2,000ms)", "Strict TPM / RPM Quotas", "Live User Chat & Real-Time Q&A"],
        "Asynchronous Batch API": ["50% Flat Discount", "Completed within 24 hours", "Separate, Vastly Higher Batch Quotas", "Corpus Embedding, Nightly Syncs, Eval"]
    })
    st.dataframe(batch_comp_df, use_container_width=True)

with col_b2:
    st.markdown("### High-Throughput Batch Ingestion Queue")
    dot_batch = """
    digraph BatchPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        RawDocs [label="New Document Corpus\n(10,000,000 Tokens)", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        JSONL_Builder [label="Generate JSONL Batch Request File\nChunk Embeddings & Metadata", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        SubmitBatch [label="Submit to Provider Batch API\n(POST /v1/batches)", shape=ellipse, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        PollQueue [label="Webhook / 24h Completion Poller", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        LoadVectorDB [label="Load Vectors into VectorDB\nCost: 50% Lower ($100 vs $200)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        RawDocs -> JSONL_Builder -> SubmitBatch -> PollQueue -> LoadVectorDB;
    }
    """
    st.graphviz_chart(dot_batch, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Enterprise FinOps Cost Optimizer & Budget Enforcer (CostOptimizerEngine)", divider="rainbow")

st.markdown("""
Below is the complete production class `CostOptimizerEngine`. It provides single-query inference cost modeling,
monthly budget simulations comparing unoptimized baselines against optimized multi-tier cascading architectures,
and unit economics forecasting.
""")

st.code('''
class CostOptimizerEngine:
    """
    Enterprise FinOps Cost Optimizer & Budget Modeling Engine for RAG Systems.
    Models unit economics across embedding generation, asymmetric LLM token consumption,
    multi-tier model cascading, prompt prefix caching, context pruning, and batch processing.
    """
    def __init__(self):
        self.pricing_catalog = {
            "Frontier_LLM": {"input_per_m": 2.50, "cached_input_per_m": 1.25, "output_per_m": 10.00},
            "SLM_Cascade": {"input_per_m": 0.15, "cached_input_per_m": 0.075, "output_per_m": 0.60},
            "SelfHosted_vLLM": {"input_per_m": 0.05, "cached_input_per_m": 0.05, "output_per_m": 0.08},
            "Embedding_Commercial": {"cost_per_m": 0.02},
            "Embedding_Batch_API": {"cost_per_m": 0.01}
        }

    def compute_query_cost(
        self,
        model_tier: str,
        input_tokens: int,
        output_tokens: int,
        cached_prefix_tokens: int = 0
    ) -> float:
        rates = self.pricing_catalog.get(model_tier, self.pricing_catalog["Frontier_LLM"])
        standard_input = max(0, input_tokens - cached_prefix_tokens)

        cost_uncached = (standard_input / 1_000_000.0) * rates["input_per_m"]
        cost_cached = (cached_prefix_tokens / 1_000_000.0) * rates["cached_input_per_m"]
        cost_output = (output_tokens / 1_000_000.0) * rates["output_per_m"]

        return round(cost_uncached + cost_cached + cost_output, 6)

    def simulate_monthly_rag_budget(
        self,
        monthly_queries: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        cache_hit_rate: float,
        cascade_slm_pct: float,
        prompt_compression_pct: float,
        provider_prompt_caching_pct: float,
        use_batch_ingestion: bool,
        docs_to_index: int,
        chunk_size_tokens: int
    ) -> Dict[str, Any]:
        baseline_cost_per_q = self.compute_query_cost("Frontier_LLM", avg_input_tokens, avg_output_tokens, 0)
        baseline_monthly_llm = monthly_queries * baseline_cost_per_q

        total_chunks = docs_to_index
        total_ingest_tokens = total_chunks * chunk_size_tokens
        baseline_ingest_cost = (total_ingest_tokens / 1_000_000.0) * self.pricing_catalog["Embedding_Commercial"]["cost_per_m"]
        baseline_total = baseline_monthly_llm + baseline_ingest_cost

        effective_queries = monthly_queries * (1.0 - cache_hit_rate)
        compressed_input = max(100, int(avg_input_tokens * (1.0 - (prompt_compression_pct / 100.0))))
        cached_prefix = int(compressed_input * (provider_prompt_caching_pct / 100.0))

        slm_queries = effective_queries * (cascade_slm_pct / 100.0)
        frontier_queries = effective_queries * (1.0 - (cascade_slm_pct / 100.0))

        cost_slm_single = self.compute_query_cost("SLM_Cascade", compressed_input, avg_output_tokens, cached_prefix)
        cost_frontier_single = self.compute_query_cost("Frontier_LLM", compressed_input, avg_output_tokens, cached_prefix)

        optimized_monthly_llm = (slm_queries * cost_slm_single) + (frontier_queries * cost_frontier_single)

        new_chunks = total_chunks * 0.60
        embed_rate = self.pricing_catalog["Embedding_Batch_API"]["cost_per_m"] if use_batch_ingestion else self.pricing_catalog["Embedding_Commercial"]["cost_per_m"]
        optimized_ingest_cost = ((new_chunks * chunk_size_tokens) / 1_000_000.0) * embed_rate

        optimized_total = optimized_monthly_llm + optimized_ingest_cost
        total_savings = baseline_total - optimized_total
        pct_savings = (total_savings / baseline_total) * 100.0 if baseline_total > 0 else 0.0

        return {
            "baseline_total": round(baseline_total, 2),
            "optimized_total": round(optimized_total, 2),
            "total_savings": round(total_savings, 2),
            "pct_savings": round(pct_savings, 1)
        }
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
cost_engine_inst = CostOptimizerEngine()
single_q_cost = cost_engine_inst.compute_query_cost(
    model_tier="Frontier_LLM",
    input_tokens=4000,
    output_tokens=300,
    cached_prefix_tokens=1000
)
assert single_q_cost > 0, "Query cost calculation must be strictly positive"

budget_sim_test = cost_engine_inst.simulate_monthly_rag_budget(
    monthly_queries=100_000,
    avg_input_tokens=4000,
    avg_output_tokens=300,
    cache_hit_rate=0.30,
    cascade_slm_pct=75.0,
    prompt_compression_pct=35.0,
    provider_prompt_caching_pct=40.0,
    use_batch_ingestion=True,
    docs_to_index=20_000,
    chunk_size_tokens=500
)
assert budget_sim_test["total_savings"] > 0, "Optimized architecture must generate cost savings"
assert budget_sim_test["pct_savings"] >= 50.0, "Multi-tier optimization should yield at least 50% savings"

st.success("Automated Verification Passed: CostOptimizerEngine successfully verified across asymmetric token pricing, cascading ratios, and batch ingestion discounts.")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Tier RAG Cost Calculator & Model Cascading Simulator", divider="rainbow")

st.markdown("""
Model and predict your enterprise RAG unit economics and monthly cloud invoices.
Configure traffic volumes, cascading percentages, semantic cache hit rates, prompt pruning ratios, and ingestion patterns.
""")

wbc_col1, wbc_col2, wbc_col3 = st.columns([1, 1, 1])

with wbc_col1:
    st.markdown("#### Workload & Traffic Volumes")
    in_monthly_queries = st.slider("Monthly Query Volume", min_value=10_000, max_value=2_000_000, value=250_000, step=25_000)
    in_input_tokens = st.slider("Average Input Prompt Tokens", min_value=1000, max_value=12000, value=4500, step=500)
    in_output_tokens = st.slider("Average Output Tokens", min_value=100, max_value=1500, value=300, step=50)

with wbc_col2:
    st.markdown("#### Cascading & Cache Parameters")
    in_cache_hit_pct = st.slider("Semantic Cache Hit Rate (%)", min_value=0, max_value=60, value=30, step=5)
    in_cascade_slm_pct = st.slider("Queries Routed to SLM Tier (%)", min_value=0, max_value=100, value=75, step=5)
    in_prompt_compression_pct = st.slider("Prompt Context Compression (%)", min_value=0, max_value=60, value=35, step=5)
    in_prefix_caching_pct = st.slider("Provider Prompt Prefix Caching (%)", min_value=0, max_value=80, value=40, step=5)

with wbc_col3:
    st.markdown("#### Document Corpus Ingestion")
    in_docs_indexed = st.slider("Total Corpus Chunks Indexed", min_value=5000, max_value=500000, value=50000, step=5000)
    in_chunk_size = st.slider("Chunk Size (Tokens)", min_value=200, max_value=1500, value=500, step=50)
    in_use_batch_api = st.toggle("Enable Batch API for Ingestion (-50%)", value=True)

sim_engine = CostOptimizerEngine()
sim_results = sim_engine.simulate_monthly_rag_budget(
    monthly_queries=in_monthly_queries,
    avg_input_tokens=in_input_tokens,
    avg_output_tokens=in_output_tokens,
    cache_hit_rate=in_cache_hit_pct / 100.0,
    cascade_slm_pct=float(in_cascade_slm_pct),
    prompt_compression_pct=float(in_prompt_compression_pct),
    provider_prompt_caching_pct=float(in_prefix_caching_pct),
    use_batch_ingestion=in_use_batch_api,
    docs_to_index=in_docs_indexed,
    chunk_size_tokens=in_chunk_size
)

st.markdown("### Projected Monthly Financial Comparison")

fc_m1, fc_m2, fc_m3, fc_m4 = st.columns(4)
fc_m1.metric("Unoptimized Baseline", f"${sim_results['baseline_total']:,.2f}")
fc_m2.metric("FinOps Optimized Total", f"${sim_results['optimized_total']:,.2f}")
fc_m3.metric("Net Monthly Savings", f"${sim_results['total_savings']:,.2f}", delta=f"{sim_results['pct_savings']}% Lower")
fc_m4.metric("Queries Bypassed via Cache", f"{sim_results['cache_bypassed_queries']:,} queries")

st.markdown("#### Monthly Expense Breakdown by Subsystem")
expense_summary_df = pd.DataFrame([
    {
        "Subsystem Category": "LLM Inference Generation",
        "Baseline (Frontier Only)": f"${sim_results['baseline_llm']:,.2f}",
        "Optimized (Cascaded + Pruned)": f"${sim_results['optimized_llm']:,.2f}",
        "Dollar Savings": f"${sim_results['baseline_llm'] - sim_results['optimized_llm']:,.2f}"
    },
    {
        "Subsystem Category": "Embedding Corpus Ingestion",
        "Baseline (Frontier Only)": f"${sim_results['baseline_ingest']:,.2f}",
        "Optimized (Cascaded + Pruned)": f"${sim_results['optimized_ingest']:,.2f}",
        "Dollar Savings": f"${sim_results['baseline_ingest'] - sim_results['optimized_ingest']:,.2f}"
    },
    {
        "Subsystem Category": "Total Operating Budget",
        "Baseline (Frontier Only)": f"${sim_results['baseline_total']:,.2f}",
        "Optimized (Cascaded + Pruned)": f"${sim_results['optimized_total']:,.2f}",
        "Dollar Savings": f"${sim_results['total_savings']:,.2f}"
    }
])
st.dataframe(expense_summary_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: Why is input token optimization generally far more impactful for RAG FinOps than output token optimization?**"):
    st.markdown("""
    **Answer**:
    In typical enterprise RAG requests, the input prompt contains system instructions, conversation context, and multiple retrieved chunks (often 3,000 to 8,000 tokens), whereas the synthesized output response is concise (typically 150 to 350 tokens).
    Even though providers price output tokens at a 3x to 4x premium over input tokens, input tokens constitute **90% to 96% of the absolute token volume**.
    Therefore, reducing input tokens by 40% through context pruning, chunk deduplication, and prefix caching delivers vastly greater aggregate dollar savings than attempting to truncate output responses.
    """)

with st.expander("**Q2: How does Provider Prompt Caching work under the hood, and what architectural rule must your prompt builder follow?**"):
    st.markdown("""
    **Answer**:
    Provider Prompt Caching (e.g., Anthropic Prompt Caching and OpenAI Cached Prompts) stores the Key-Value (KV) attention matrices computed during the prompt prefill phase across requests that share identical prefix token sequences.
    * **The Golden Prompt Architecture Rule**:
      Dynamic and variable content must never be placed at the beginning of the prompt.
      You must order your prompt strictly as:
      1. Static System Instructions (Identical across all calls)
      2. Static / Stable Corporate Knowledge Documents (Cached)
      3. Dynamic Session History & Specific User Question (Uncached, at the very end).
      Placing the user's timestamp or dynamic user query at the top invalidates the entire KV cache prefix for all subsequent tokens.
    """)

with st.expander("**Q3: What are the criteria used by a Model Cascading Router to decide between an SLM (e.g., GPT-4o-mini) and a Frontier LLM (e.g., GPT-4o)?**"):
    st.markdown("""
    **Answer**:
    Production cascading routers evaluate multiple signals:
    1. **Lexical / Intent Classification**: Does the query contain analytical keywords (*'compare'*, *'synthesize'*, *'critique'*, *'code'*) or is it a direct factoid (*'what is'*, *'who is'*, *'policy for'*).
    2. **Retrieval Dispersion**: If retrieved chunks originate from a single document with high semantic similarity, an SLM suffices. If chunks originate from multiple conflicting sources requiring complex reconciliation, route to the Frontier model.
    3. **Confidence Scoring / Self-Reflection**: The SLM generates a candidate answer along with a confidence logit. If confidence falls below 0.85, the request cascades to the Frontier model.
    """)

with st.expander("**Q4: How do Matryoshka Embeddings reduce vector database hosting costs without re-indexing from scratch?**"):
    st.markdown("""
    **Answer**:
    Matryoshka Representation Learning (MRL) trains embedding models so that the earlier dimensions capture the highest-entropy semantic signals.
    A 1536-dimensional vector can be sliced down to 512 or 256 dimensions simply by truncating the vector array (`vec[:512]`) and re-normalizing.
    Because managed vector databases (such as Pinecone, Qdrant Cloud, Milvus) charge primarily for RAM to hold vectors and graph indexes in memory, reducing dimensions by 66% slashes the cluster's memory footprint and operational cloud hosting bill by approximately two-thirds.
    """)

with st.expander("**Q5: When should an enterprise engineering team transition from commercial API models to self-hosted open-source models (vLLM)?**"):
    st.markdown("""
    **Answer**:
    The crossover point is governed by query volume and GPU utilization economics:
    * At low or bursty volumes (< 100,000 queries/month), commercial serverless APIs are vastly cheaper because you pay strictly for consumed tokens with zero idle infrastructure cost.
    * When monthly traffic exceeds **1.5M - 3M queries**, continuous GPU compute instances (e.g., an AWS `g5.12xlarge` or `g6.4xlarge` running vLLM with Llama-3.1-8B) reach >70% steady utilization.
    * At high sustained scale, open-source self-hosting drops the effective cost per million tokens from ~$0.75 down to ~$0.08, achieving massive ROI even after accounting for DevOps maintenance overhead.
    """)

st.markdown("### Production Traps & Failure Modes")

ctr_col1, ctr_col2 = st.columns(2)

with ctr_col1:
    st.error("""
    **Trap 1: Dynamic Timestamps Invalidating Prompt Prefix Caching**
    * **Failure**: Injecting `f"Current Time: {datetime.now()}"` at the top of the system prompt template.
    * **Consequence**: The prompt prefix changes every second, completely destroying the KV cache hit rate across all users and forfeiting the 50-80% provider token discount.
    * **Remedy**: Always isolate dynamic timestamps and user session IDs at the very bottom of the prompt payload, keeping the upper 2,000+ tokens completely static.
    """)

    st.error("""
    **Trap 2: The Unbounded Re-Indexing Billing Shock**
    * **Failure**: Setting up a cron job that drops and re-embeds the entire corporate Confluence space (15M tokens) nightly without chunk hashing.
    * **Consequence**: The organization incurs hundreds of dollars in unnecessary embedding API charges every month for documentation that had zero edits.
    * **Remedy**: Enforce chunk-level SHA-256 hash checks in Redis. Only modified or newly added paragraphs should ever trigger embedding API calls.
    """)

with ctr_col2:
    st.warning("""
    **Trap 3: Over-Cascading without Fallback Guardrails**
    * **Failure**: Forcing 95% of queries to an SLM to artificially meet aggressive FinOps targets without hallucination verification.
    * **Consequence**: Subtle multi-hop queries receive degraded, hallucinated answers, destroying enterprise trust.
    * **Remedy**: Always implement secondary logit confidence verification or consistency checks with automated fallback to the frontier model.
    """)

    st.warning("""
    **Trap 4: Using Real-Time APIs for Massive Offline Evaluation Runs**
    * **Failure**: Running large offline RAGAS evaluation suites (50,000 sample queries) against synchronous commercial endpoints.
    * **Consequence**: Full price is billed and client jobs hit API rate limit quotas (HTTP 429), resulting in failed evaluation runs.
    * **Remedy**: Route all non-interactive evaluation and batch ingestion through Provider Batch APIs for an automatic flat 50% discount and dedicated quotas.
    """)

st.markdown("### Core Technical Flashcards")

fcc_c1, fcc_c2 = st.columns(2)

with fcc_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary economic difference between RAG input tokens and output tokens?")
        st.markdown("**Back**: Output tokens are priced 3x to 4x higher per token by providers, but input prompt tokens account for 90% - 96% of total token volume in RAG. Context pruning therefore drives the highest absolute dollar savings.")

    with st.container(border=True):
        st.markdown("**Front**: How does Model Cascading achieve up to 90% cost reduction without sacrificing answer quality?")
        st.markdown("**Back**: By routing 70% - 80% of routine single-hop queries to low-cost Small Language Models ($0.15/M tokens), reserving expensive Frontier models ($2.50+/M tokens) exclusively for complex multi-hop synthesis.")

with fcc_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why is chunk content hashing (SHA-256) critical for embedding cost optimization?")
        st.markdown("**Back**: It allows incremental updates where only newly added or modified paragraphs trigger embedding API calls, eliminating wasteful re-indexing of static documents.")

    with st.container(border=True):
        st.markdown("**Front**: What financial advantage do Provider Batch APIs offer for offline RAG workloads?")
        st.markdown("**Back**: They provide an automatic flat 50% discount on both input and output token pricing for non-urgent requests completed within a 24-hour SLA.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of RAG Cost Optimization & FinOps principles:")

c_chk1 = st.checkbox("I understand the unit economics of RAG token burn and why input prompt tokens dominate operating expenses.", value=True)
c_chk2 = st.checkbox("I know how to design a Model Cascading architecture with complexity classifiers and confidence fallbacks.", value=True)
c_chk3 = st.checkbox("I can implement deterministic prompt ordering to maximize Provider Prompt Prefix Caching discounts.", value=True)
c_chk4 = st.checkbox("I understand how SHA-256 chunk deduplication eliminates redundant embedding costs during scheduled re-indexing.", value=True)
c_chk5 = st.checkbox("I know how Matryoshka dimension truncation reduces downstream managed Vector Database RAM hosting costs.", value=True)
c_chk6 = st.checkbox("I can leverage asynchronous Batch APIs to secure 50% flat discounts on large-scale offline ingestion and evaluation.", value=True)

if c_chk1 and c_chk2 and c_chk3 and c_chk4 and c_chk5 and c_chk6:
    st.success("Mastery Certified: You have mastered production RAG cost optimization, FinOps token economics, model cascading, and batch processing.")
