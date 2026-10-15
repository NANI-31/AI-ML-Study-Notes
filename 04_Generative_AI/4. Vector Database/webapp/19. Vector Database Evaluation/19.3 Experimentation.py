import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="19.3 Experimentation - Vector DB Curriculum",
    page_icon="🔬",
    layout="wide"
)

# Title & Overview
st.title("19.3 RAG Experimentation: Comparing Embedding Models, Chunk Sizes, Top-K, Retrieval & Rerankers")
st.caption("Chapter 19: Vector Database Evaluation | Domain: 4. Vector Database")
st.markdown("""
Building an enterprise RAG system is not a one-time configuration; it is an **empirical science of systematic experimentation**. 

There is no single "magic" set of hyperparameters that works across all domains. A medical knowledge base requires 
fundamentally different chunk sizes, embedding models, and retrieval strategies than a GitHub code search engine.

To maximize retrieval accuracy while controlling cost and latency, engineers must conduct controlled **A/B Matrix Experiments** 
across five core architectural levers:
1. **Embedding Models**: Evaluating dimensionality, domain vocabulary, and latency across providers.
2. **Chunk Sizes & Overlaps**: Balancing atomic precision (256 tokens) against semantic completeness (1024 tokens).
3. **Top-K Retrieval Windows**: Tuning candidate volume to optimize the recall-to-noise ratio.
4. **Retrieval Strategies**: Comparing Pure Dense vs. Hybrid BM25+Dense vs. MMR.
5. **Reranking Strategies**: Measuring the precision uplift of Stage-2 Cross-Encoders vs. cloud reranking APIs.

In this lesson, we explore experimental methodologies, analyze empirical trade-off curves, and test an interactive 
A/B RAG Configuration Arena.
""")

st.divider()

# Section 1: The Five Experimental Levers of RAG Architecture
st.header("1. The Five Experimental Levers of RAG Architecture", divider="rainbow")

col_lev1, col_lev2 = col_lev1, col_lev2 = st.columns([1.1, 0.9])

with col_lev1:
    st.markdown("""
    ### The Systematic Hyperparameter Grid
    
    * **Lever 1: Embedding Model Selection**:
      * *Candidates*: OpenAI `text-embedding-3-small` (1536d) vs BGE `bge-large-en-v1.5` (1024d) vs `all-MiniLM-L6-v2` (384d).
      * *Evaluation*: MTEB Retrieval benchmarks, embedding latency, and API costs.
    
    * **Lever 2: Chunk Size & Overlap Boundary**:
      * *Small (200 - 350 tokens)*: Laser-focused facts; ideal for FAQ lookups; risks losing surrounding context.
      * *Medium (400 - 600 tokens)*: **Industry Sweet Spot**; balances factual specificity and structural context.
      * *Large (800 - 1500 tokens)*: Broad context; ideal for summarization; dilutes specific entity vectors.
    
    * **Lever 3: Top-K Context Window**:
      * Tuning K between 3 and 15 to find the exact point of diminishing returns where adding more chunks increases prompt cost without improving answer quality.
    
    * **Lever 4: Retrieval Engine Mode**:
      * Pure Dense Vector vs. Hybrid BM25+Dense (RRF) vs. Maximal Marginal Relevance (MMR).
    
    * **Lever 5: Reranker Integration**:
      * None (Baseline) vs. Sentence-Transformers MiniLM vs. BGE-Reranker-Large vs. Cohere Rerank v3.
    """)

with col_lev2:
    st.markdown("### The Experimental Matrix Topology")
    exp_dot = """
    digraph ExperimentGrid {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Dataset [label="Golden Benchmark Suite\\n(500 Labeled Questions)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_varA {
            label="Variant A (Baseline)"; style="dashed"; color="#DC2626";
            ConfA [label="OpenAI small (1536d)\\nChunk: 800 tokens\\nDense Top-5 (No Rerank)", fillcolor="#FEE2E2"];
        }

        subgraph cluster_varB {
            label="Variant B (Optimized)"; style="dashed"; color="#059669";
            ConfB [label="BGE-large (1024d)\\nChunk: 450 tokens\\nHybrid + Rerank Top-3", fillcolor="#D1FAE5", style="filled,bold"];
        }

        EvalEngine [label="Automated Evaluator\\nRecall@5 | NDCG | Faithfulness | Latency | Cost", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        Dataset -> ConfA;
        Dataset -> ConfB;
        ConfA -> EvalEngine;
        ConfB -> EvalEngine;
    }
    """
    st.graphviz_chart(exp_dot, use_container_width=False)

st.divider()

# Section 2: Empirical Findings from Industry Benchmarks
st.header("2. Empirical Findings from Industry Benchmarks", divider="rainbow")

st.markdown("""
Across thousands of real-world production evaluations, key empirical patterns consistently emerge:
""")

EMPIRICAL_FINDINGS = [
    {
        "Experimental Dimension": "Chunk Size (256 vs 512 vs 1024)",
        "Key Finding": "512 tokens with 15% overlap consistently outperforms 1024 tokens on factual question answering by +14% NDCG@10. 1024 tokens causes vector dilution, reducing cosine separation between distinct topics."
    },
    {
        "Experimental Dimension": "Dense Only vs Hybrid (BM25 + Dense)",
        "Key Finding": "Adding BM25 via Reciprocal Rank Fusion improves overall recall by +8% to +18%, entirely eliminating catastrophic misses on part numbers, error codes, and technical acronyms."
    },
    {
        "Experimental Dimension": "Top-K Sizing (K=3 vs K=10)",
        "Key Finding": "Increasing K from 3 to 10 improves Recall by ~6%, but degrades downstream LLM Faithfulness by ~11% due to Lost-in-the-Middle context distraction, while increasing prompt token costs by 230%."
    },
    {
        "Experimental Dimension": "Adding Cross-Encoder Reranking",
        "Key Finding": "Deploying a Stage-2 Reranker (BGE or Cohere) delivers the single largest accuracy gain in the entire pipeline (+22% to +35% NDCG@10), easily justifying its ~20ms latency addition."
    }
]

st.dataframe(pd.DataFrame(EMPIRICAL_FINDINGS), use_container_width=True, hide_index=True)

st.divider()

# Section 3: Architecture for Running Automated Matrix Sweeps
st.header("3. Architecture for Running Automated Matrix Sweeps", divider="rainbow")

col_code_a, col_code_b = col_code_a, col_code_b = st.columns([1, 1])

with col_code_a:
    st.markdown("""
    ### The Systematic Evaluation Protocol
    To evaluate a new RAG configuration without human bias:
    
    1. **Golden Dataset**: Maintain a fixed validation dataset of 200–500 `(query, ground_truth_chunk_ids, reference_answer)` triplets.
    2. **Grid Execution**: Programmatically loop through parameter combinations:
       ```text
       for chunk_size in [256, 512]:
         for retrieval_mode in ['dense', 'hybrid']:
           for reranker in [None, 'bge-large']:
             run_eval_suite(...)
       ```
    3. **Composite Quality Index**: Plot the **Pareto Frontier** of Accuracy (NDCG/Faithfulness) vs. Total Latency vs. Operational Cost.
    """)

with col_code_b:
    st.markdown("### Python Grid Sweep Framework")
    st.code("""from itertools import product
import pandas as pd

# Define hyperparameter grid
EXPERIMENT_GRID = {
    "chunk_size": [300, 500, 1000],
    "retrieval_strategy": ["dense", "hybrid_rrf", "mmr"],
    "reranker": [None, "bge-reranker-large", "cohere-v3"],
    "top_k": [3, 5]
}

def execute_grid_sweep(benchmark_suite):
    results = []
    keys, values = zip(*EXPERIMENT_GRID.items())
    
    for combo in product(*values):
        config = dict(zip(keys, combo))
        # Build pipeline for current configuration
        pipeline = build_rag_pipeline(config)
        
        # Run automated evaluation
        metrics = benchmark_suite.evaluate(pipeline)
        results.append({**config, **metrics})
        
    return pd.DataFrame(results)
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live RAG A/B Configuration Arena
st.header("4. Interactive Workbench: Live RAG A/B Configuration Arena", divider="rainbow")

st.markdown("""
Configure two competing RAG pipeline variants (Variant A vs. Variant B) to project and compare 
their retrieval accuracy, latency, prompt token volume, and operational costs.
""")

col_var_a, col_var_b = st.columns(2)

with col_var_a:
    st.subheader("🔴 Pipeline Variant A (Baseline)")
    emb_a = st.selectbox("Embedding Model A:", ["OpenAI text-embedding-3-small (1536d)", "all-MiniLM-L6-v2 (384d)"], index=0)
    chunk_a = st.selectbox("Chunk Size A:", [300, 500, 1000], index=2)
    strat_a = st.selectbox("Retrieval Mode A:", ["Pure Dense Vector", "Hybrid BM25+Dense", "MMR"], index=0)
    rerank_a = st.selectbox("Reranker A:", ["None (Direct Top-K)", "BGE-Reranker-Large", "Cohere Rerank v3"], index=0)
    k_a = st.slider("Final Top-K Context Chunks A:", min_value=2, max_value=8, value=5)

with col_var_b:
    st.subheader("🟢 Pipeline Variant B (Optimized)")
    emb_b = st.selectbox("Embedding Model B:", ["OpenAI text-embedding-3-small (1536d)", "BAAI BGE-large (1024d)"], index=1)
    chunk_b = st.selectbox("Chunk Size B:", [300, 500, 1000], index=1)
    strat_b = st.selectbox("Retrieval Mode B:", ["Pure Dense Vector", "Hybrid BM25+Dense", "MMR"], index=1)
    rerank_b = st.selectbox("Reranker B:", ["None (Direct Top-K)", "BGE-Reranker-Large", "Cohere Rerank v3"], index=1)
    k_b = st.slider("Final Top-K Context Chunks B:", min_value=2, max_value=8, value=3)

# Performance Modeling
def score_variant(emb, chunk, strat, rerank, k):
    # Base Recall
    recall = 78.0
    if "BGE" in emb: recall += 4.0
    if chunk == 500: recall += 5.0
    elif chunk == 300: recall += 2.0
    if "Hybrid" in strat: recall += 8.0
    if rerank != "None (Direct Top-K)": recall += 7.0
    
    # Base NDCG
    ndcg = 0.58
    if rerank != "None (Direct Top-K)": ndcg += 0.22
    if "Hybrid" in strat: ndcg += 0.08
    if chunk == 500: ndcg += 0.04
    
    # Latency (ms)
    lat = 2.0
    if "OpenAI" in emb: lat += 14.0
    else: lat += 5.0
    if "Hybrid" in strat: lat += 1.5
    if rerank != "None (Direct Top-K)": lat += 18.0
    llm_lat = 400.0 + (k * (chunk // 4) * 0.35)
    total_lat = lat + llm_lat
    
    # Tokens & Cost per 1k queries
    prompt_tokens = 350 + (k * (chunk // 4)) + 80
    cost_1k = (prompt_tokens * 1000 / 1e6) * 2.50
    if "Cohere" in rerank: cost_1k += 1.00
    
    return min(98.5, recall), min(0.96, ndcg), total_lat, prompt_tokens, cost_1k

rec_a, ndcg_a, lat_a, tok_a, cost_a = score_variant(emb_a, chunk_a, strat_a, rerank_a, k_a)
rec_b, ndcg_b, lat_b, tok_b, cost_b = score_variant(emb_b, chunk_b, strat_b, rerank_b, k_b)

st.divider()

st.subheader("Head-to-Head Architectural Scorecard")

comp_df = pd.DataFrame([
    {"Metric": "Expected Recall@K", "Variant A (Baseline)": f"{rec_a:.1f}%", "Variant B (Optimized)": f"{rec_b:.1f}%", "Delta": f"{rec_b - rec_a:+.1f}%"},
    {"Metric": "Ranking Precision (NDCG@K)", "Variant A (Baseline)": f"{ndcg_a:.3f}", "Variant B (Optimized)": f"{ndcg_b:.3f}", "Delta": f"{ndcg_b - ndcg_a:+.3f}"},
    {"Metric": "Prompt Context Tokens", "Variant A (Baseline)": f"{tok_a:,} tokens", "Variant B (Optimized)": f"{tok_b:,} tokens", "Delta": f"{tok_b - tok_a:+,} tokens"},
    {"Metric": "End-to-End Latency", "Variant A (Baseline)": f"{lat_a:.0f} ms", "Variant B (Optimized)": f"{lat_b:.0f} ms", "Delta": f"{lat_b - lat_a:+.0f} ms"},
    {"Metric": "Cost per 1,000 Queries", "Variant A (Baseline)": f"${cost_a:.2f}", "Variant B (Optimized)": f"${cost_b:.2f}", "Delta": f"${cost_b - cost_a:+.2f}"}
])

st.dataframe(comp_df, use_container_width=True, hide_index=True)

if ndcg_b > ndcg_a and cost_b < cost_a:
    st.success("🏆 **Pareto Superiority Achieved**: Variant B achieves higher ranking precision (+{:.1f}% NDCG) while saving prompt tokens and cutting operating costs!".format((ndcg_b - ndcg_a)*100))
elif ndcg_b > ndcg_a:
    st.info("💡 **Precision Trade-Off**: Variant B achieves superior retrieval fidelity (+{:.1f}% NDCG) with acceptable latency characteristics.".format((ndcg_b - ndcg_a)*100))

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why is chunk size 500 tokens generally considered superior to 1000 tokens for factual Q&A?"):
    st.markdown("""
    **Answer**:
    * **Vector Dilution**: An embedding vector has fixed dimensionality (e.g. 1536 floats). When you compress 1,000 tokens into that vector, diverse concepts become blended together, diluting specific technical numbers or clauses.
    * **Cosine Distance Flattening**: Long chunks appear moderately similar to almost everything, reducing geometric separation between exact answers and generic background text.
    * **Signal-to-Noise**: A 500-token chunk provides sufficient context for sentence comprehension while ensuring that the vector remains focused on the primary concept.
    """)

with st.expander("Q2: How do you determine the optimal Top-K context window empirically?"):
    st.markdown("""
    **Answer**:
    Plot a **Recall vs. Faithfulness Trade-Off Curve**:
    1. Evaluate your test benchmark across varying K values: `K = [1, 2, 3, 5, 8, 12, 20]`.
    2. Track **Recall@K** (which increases monotonically with K) and **LLM Faithfulness** (which initially rises, peaks, and then drops as context noise increases).
    3. The optimal K is the **inflection point** where Recall begins to plateau while Faithfulness remains at its maximum peak (typically `K = 3 to 5` when using a reranker).
    """)

with st.expander("Q3: When is upgrading to a larger embedding model NOT worth the engineering effort?"):
    st.markdown("""
    **Answer**:
    If your pipeline is missing a **Stage-2 Cross-Encoder Reranker** or **Hybrid BM25 keyword search**:
    * Upgrading from a 384-dim to a 3,072-dim embedding model typically yields a modest 2% to 4% improvement in NDCG.
    * In contrast, adding Hybrid BM25 provides an immediate **+10% to +15% boost**, and adding a Cross-Encoder provides a **+20% to +30% boost**!
    * **Engineering Heuristic**: Never spend weeks fine-tuning or scaling embedding models until you have already implemented hybrid retrieval and cross-encoder reranking.
    """)

with st.expander("Q4: What is the 'Pareto Frontier' in RAG optimization?"):
    st.markdown("""
    **Answer**:
    The Pareto Frontier represents the set of configurations where **no metric can be improved without degrading another**:
    * A configuration is Pareto-optimal if there is no alternative that has higher accuracy, lower latency, AND lower cost simultaneously.
    * Engineers plot **Accuracy (NDCG) vs. Total Latency** and select configurations along the Pareto frontier that satisfy their specific business SLA (e.g. 95% accuracy with p95 latency < 500ms).
    """)

with st.expander("Q5: How do you prevent Data Leakage when evaluating chunk size experiments?"):
    st.markdown("""
    **Answer**:
    * If ground-truth test questions are generated from specific 500-token chunks, re-evaluating with 250-token chunks might artificially split the ground-truth sentence across a boundary, causing an apparent drop in recall.
    * **Remedy**: Ground truth should be defined at the **document or canonical fact level** (e.g. *"Fact: Database WAL retention is 35 days"*), not tied to arbitrary chunk boundary offsets.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Tuning Multiple Levers Simultaneously**  
    Changing the embedding model, chunk size, and reranker at the same time prevents identifying which modification caused an accuracy gain or regression. Change one variable at a time.
    """)
    st.warning("""
    **Trap 2: Evaluating on Fewer than 100 Queries**  
    Running benchmarks on 15 queries produces high statistical variance where a single lucky match swings accuracy metrics by 7%. Evaluate on at least 200–500 queries.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Increasing Top-K to Mask Poor Retrieval Precision**  
    Setting K=20 because the retriever ranks the answer at Rank 15 inflates prompt costs by 400% and increases hallucinations. Deploy a Stage-2 reranker instead.
    """)
    st.warning("""
    **Trap 4: Ignoring Embedding Generation Latency in A/B Tests**  
    Focusing only on vector search speed while ignoring that a 3,072-dim embedding API call takes 3x longer to generate leads to surprising production latency regressions.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the optimal chunk size for Q&A?"):
        st.markdown("""
        **400 to 600 Tokens**:  
        Sweet spot: preserves sentence context while preventing semantic vector dilution.
        """)
    with st.expander("🗂️ Flashcard: What delivers the biggest accuracy gain?"):
        st.markdown("""
        **Stage-2 Reranker**:  
        Adding a Cross-Encoder yields +20% to +35% NDCG, far outpacing embedding model upgrades.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What happens if Top-K is too large?"):
        st.markdown("""
        **Context Dilution**:  
        Wastes prompt tokens, inflates API bills, and increases hallucinations via Lost-in-the-Middle.
        """)
    with st.expander("🗂️ Flashcard: What is the Pareto Frontier?"):
        st.markdown("""
        **Optimal Trade-Off Set**:  
        Configurations where accuracy cannot be improved without increasing latency or cost.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the five core experimental levers of RAG architecture.", value=True)
chk2 = st.checkbox("I know why 500-token chunks outperform 1000-token chunks on factual retrieval.", value=True)
chk3 = st.checkbox("I understand how to determine the optimal Top-K context window empirically.", value=True)
chk4 = st.checkbox("I can design and execute an automated hyperparameter grid sweep across a golden dataset.", value=True)
chk5 = st.checkbox("I know how to identify Pareto-optimal configurations balancing accuracy, latency, and cost.", value=True)
chk6 = st.checkbox("I can lead a systematic A/B experimentation program for enterprise vector search optimization.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completely mastered Chapter 19: Vector Database Evaluation and are ready for Chapter 20: Vector Database Security!")
