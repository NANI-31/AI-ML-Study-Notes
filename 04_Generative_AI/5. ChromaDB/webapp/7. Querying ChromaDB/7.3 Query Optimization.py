import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="7.3 Query Optimization - ChromaDB Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("7.3 Query Optimization: Top-K Tuning, Threshold Gating, Query Expansion & RRF Fusion")
st.caption("Chapter 7: Querying ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
In naive RAG implementations, a user's single natural language question is passed directly into `collection.query()`, 
and whatever Top-K documents return are blindly injected into the LLM context prompt.

This naive approach fails frequently in production:
* **Vocabulary Mismatch**: The user asks *"How do I fix database restarts?"* but the technical runbook is titled *"Configuring Kubernetes OOMKilled Container Memory Limits"*. Direct vector similarity alone may rank this document poorly.
* **Hallucination by Irrelevant Context**: Out-of-domain questions retrieve the "least-bad" irrelevant chunks, poisoning the LLM prompt.
* **Lost in the Middle**: Sizing `n_results` too high causes LLMs to ignore facts buried in lengthy prompt context.

Production-grade search pipelines solve these challenges through **Query Optimization Strategies**:
1. **Dynamic Similarity Threshold Gating**: Discarding low-confidence matches (`distance > cutoff`) to guarantee grounded context.
2. **Query Expansion & HyDE**: Generating complementary perspectives and sub-queries using LLMs before search.
3. **Multi-Query Reciprocal Rank Fusion (RRF)**: Merging candidate lists across multiple queries into a single, high-confidence ranking.

In this lesson, we break down query optimization techniques without LaTeX math, implement a full RRF fusion pipeline in Python, 
and test an interactive Query Expansion & Reciprocal Rank Fusion Arena.
""")

st.divider()

# Section 1: The Three Levers of Query Optimization
st.header("1. The Three Levers of Query Optimization", divider="rainbow")

col_opt1, col_opt2 = col_opt1, col_opt2 = st.columns([1.1, 0.9])

with col_opt1:
    st.markdown("""
    ### Optimization Taxonomy
    
    * **1. Dynamic Similarity Threshold Gating**:
      * *Problem*: In a standard Top-5 query, Chroma always returns 5 documents even if all 5 are completely unrelated.
      * *Solution*: Enforce a strict distance cutoff (e.g., `distance < 0.35` for cosine). If zero documents pass, return a fallback message immediately.
    
    * **2. Query Expansion & Sub-Query Generation**:
      * *Problem*: Natural human questions are often ambiguous or omit critical domain keywords.
      * *Solution*: Prompt an LLM to generate 3 alternative formulations of the user's question, capturing synonyms and technical terminology.
    
    * **3. Reciprocal Rank Fusion (RRF)**:
      * *Problem*: When executing 3 expanded queries, documents appear at different ranks across the 3 result sets.
      * *Solution*: Merge rankings using standard RRF scoring: `Score = sum(1.0 / (60 + rank))`. Documents retrieved consistently across multiple queries receive massive ranking boosts!
    """)

with col_opt2:
    st.markdown("### The Optimized Retrieval Pipeline")
    opt_dot = """
    digraph OptimizedRetrieval {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="User Query: 'Database crashing'\\nAmbiguous user phrasing", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        LLMExpand [label="LLM Query Expansion\\nGenerates 3 Technical Variants\\n(OOMKilled, Disk Exhaustion, Segfault)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        BatchSearch [label="ChromaDB Batch Query\\ncollection.query(query_texts=[Q1, Q2, Q3])", fillcolor="#FEF3C7", color="#D97706"];

        RRFFusion [label="Reciprocal Rank Fusion (RRF)\\nMerge & Re-score Candidates", fillcolor="#FDE68A", color="#D97706"];

        ThresholdFilter [label="Distance Cutoff Gate\\nDiscard items with distance > 0.35", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        CleanContext [label="Final Precision Context\\nPassed to LLM for Synthesis", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserQ -> LLMExpand -> BatchSearch -> RRFFusion -> ThresholdFilter -> CleanContext;
    }
    """
    st.graphviz_chart(opt_dot, use_container_width=False)

st.divider()

# Section 2: Reciprocal Rank Fusion (RRF) Formulation
st.header("2. Reciprocal Rank Fusion (RRF) Formulation", divider="rainbow")

col_rrf1, col_rrf2 = col_rrf1, col_rrf2 = st.columns([1, 1])

with col_rrf1:
    st.markdown("""
    ### Why RRF Outperforms Score Averaging
    * When merging results from multiple queries or hybrid search models (dense vectors + sparse BM25), **raw distance scores cannot be averaged directly** because different queries produce different distance distributions.
    * **Reciprocal Rank Fusion (RRF)** bypasses raw distance values and operates strictly on **rank positions**:
      * For each document, its RRF score across all query runs is:
        `RRF_Score = sum( 1.0 / (k_constant + rank_in_query) )`
      * The industry standard smoothing constant is `k_constant = 60`.
    
    ### Properties of RRF:
    * Documents ranked #1 receive the highest score (`1 / 61 = 0.0164`).
    * Documents that appear across **multiple queries** compound their scores, easily outranking documents that appeared in only one query.
    * Highly resilient to outliers and requires zero hyperparameter tuning.
    """)

with col_rrf2:
    st.markdown("### Clean Python Implementation of RRF")
    st.code("""from typing import List, Dict, Any
from collections import defaultdict

def reciprocal_rank_fusion(
    batch_results: Dict[str, Any],
    k_constant: int = 60
) -> List[Dict[str, Any]]:
    \"\"\"Merges multi-query ChromaDB results using Reciprocal Rank Fusion.\"\"\"
    rrf_scores = defaultdict(float)
    doc_lookup = {}
    
    num_queries = len(batch_results["ids"])
    
    for q_idx in range(num_queries):
        query_ids = batch_results["ids"][q_idx]
        query_docs = batch_results["documents"][q_idx]
        
        for rank, (doc_id, text) in enumerate(zip(query_ids, query_docs), start=1):
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = text
                
    # Sort documents by descending RRF score
    ranked_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {"id": doc_id, "rrf_score": round(score, 6), "text": doc_lookup[doc_id]}
        for doc_id, score in ranked_fused
    ]
""", language="python")

st.divider()

# Section 3: Full End-to-End Optimized Query Pipeline
st.header("3. Full End-to-End Optimized Query Pipeline", divider="rainbow")

st.code("""import chromadb

def optimized_search_pipeline(
    collection,
    original_query: str,
    expanded_queries: List[str],
    top_k: int = 3,
    max_distance_cutoff: float = 0.35
):
    \"\"\"Executes multi-query expansion, batch retrieval, and distance gating.\"\"\"
    all_queries = [original_query] + expanded_queries
    
    # 1. Execute batch query in a single round-trip
    results = collection.query(
        query_texts=all_queries,
        n_results=top_k * 2  # Over-fetch for fusion
    )
    
    # 2. Apply distance threshold filter & deduplication
    valid_candidates = {}
    for q_idx in range(len(all_queries)):
        for doc_id, doc, dist in zip(
            results["ids"][q_idx],
            results["documents"][q_idx],
            results["distances"][q_idx]
        ):
            if dist <= max_distance_cutoff:
                if doc_id not in valid_candidates or dist < valid_candidates[doc_id]["best_dist"]:
                    valid_candidates[doc_id] = {
                        "id": doc_id,
                        "text": doc,
                        "best_dist": dist,
                        "similarity": 1.0 - dist
                    }
                    
    # 3. Sort by best similarity and slice final Top-K
    final_hits = sorted(valid_candidates.values(), key=lambda x: x["best_dist"])[:top_k]
    return final_hits
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Query Expansion & RRF Arena
st.header("4. Interactive Workbench: Query Expansion & RRF Arena", divider="rainbow")

st.markdown("""
Simulate query expansion and Reciprocal Rank Fusion. Enter an initial user query, 
inspect the generated sub-queries, observe which documents are retrieved across queries, 
and see how RRF elevates multi-hit documents to the top.
""")

col_wb_q, col_wb_cfg = col_wb_q, col_wb_cfg = st.columns([1.1, 0.9])

with col_wb_q:
    st.subheader("1. Base Query & LLM Expansion")
    base_q = st.text_input("User Initial Query:", value="How do I fix database pod crashes?")
    exp_q1 = st.text_input("Expanded Variant 1 (Technical):", value="Kubernetes PostgreSQL container OOMKilled exit code 137 resolution.")
    exp_q2 = st.text_input("Expanded Variant 2 (Alternative):", value="StatefulSet NVMe disk exhaustion troubleshooting procedures.")

with col_wb_cfg:
    st.subheader("2. Fusion Parameters")
    rrf_k_param = st.slider("RRF Constant (k_constant):", min_value=10, max_value=100, value=60, step=10)
    cutoff_gate = st.slider("Max Allowed Distance (Cutoff Gate):", min_value=0.10, max_value=0.80, value=0.40, step=0.05)

# Mock retrieval across 3 queries
MOCK_EXP_RETRIEVAL = {
    "ids": [
        ["doc_oom_01", "doc_nvme_01", "doc_net_01"],       # Query 0
        ["doc_oom_01", "doc_mem_02", "doc_oom_03"],        # Query 1
        ["doc_nvme_01", "doc_disk_02", "doc_oom_01"]       # Query 2
    ],
    "documents": [
        ["PostgreSQL pod OOMKilled due to high maintenance_work_mem buffer allocation.", "StatefulSet PVC volume disk full remediation steps.", "Calico CNI IP pool exhaustion."],
        ["PostgreSQL pod OOMKilled due to high maintenance_work_mem buffer allocation.", "Kernel cgroup v2 memory limit configuration guide.", "Linux OOM killer process scores."],
        ["StatefulSet PVC volume disk full remediation steps.", "PostgreSQL WAL directory disk space cleanup.", "PostgreSQL pod OOMKilled due to high maintenance_work_mem buffer allocation."]
    ],
    "distances": [
        [0.22, 0.35, 0.65],
        [0.12, 0.28, 0.31],
        [0.15, 0.29, 0.38]
    ]
}

# Compute RRF
from collections import defaultdict
scores = defaultdict(float)
doc_texts = {}
best_distances = defaultdict(lambda: 99.0)
appearance_counts = defaultdict(int)

for q_i in range(3):
    for r, (d_id, text, dist) in enumerate(zip(
        MOCK_EXP_RETRIEVAL["ids"][q_i],
        MOCK_EXP_RETRIEVAL["documents"][q_i],
        MOCK_EXP_RETRIEVAL["distances"][q_i]
    ), start=1):
        if dist <= cutoff_gate:
            scores[d_id] += 1.0 / (rrf_k_param + r)
            doc_texts[d_id] = text
            appearance_counts[d_id] += 1
            if dist < best_distances[d_id]:
                best_distances[d_id] = dist

fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

st.divider()

st.subheader("Fused & Re-Ranked Search Results (RRF)")

fused_rows = []
for rank, (doc_id, score) in enumerate(fused_results, start=1):
    fused_rows.append({
        "Final Rank": f"#{rank}",
        "Document ID": doc_id,
        "RRF Score": f"{score:.6f}",
        "Query Hits": f"{appearance_counts[doc_id]} / 3 queries",
        "Best Cosine Dist": f"{best_distances[doc_id]:.4f}",
        "Document Text Preview": doc_texts[doc_id][:65] + "..."
    })

st.dataframe(pd.DataFrame(fused_rows), use_container_width=True, hide_index=True)

if fused_results:
    champion = fused_results[0][0]
    st.success(f"🏆 **Champion Match**: `{champion}` appeared across {appearance_counts[champion]} out of 3 queries and won Rank #1 via RRF Fusion! High confidence context achieved.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is Query Expansion, and how does it solve the vocabulary mismatch problem?"):
    st.markdown("""
    **Answer**:
    * **Vocabulary Mismatch**: Users frequently express technical queries using lay terms, typos, or partial acronyms (e.g. *"website is slow"* instead of *"P99 latency degradation due to TCP window exhaustion"*).
    * **Query Expansion**: Uses an LLM to generate 2–4 diverse alternative phrasings, sub-questions, or hypothetical documents (HyDE) prior to database search.
    * Searching with all variants dramatically improves candidate recall by covering multiple semantic angles in vector space.
    """)

with st.expander("Q2: How does Reciprocal Rank Fusion (RRF) work, and why is `k = 60` standard?"):
    st.markdown("""
    **Answer**:
    * `RRF_score(d) = sum( 1.0 / (k + rank(d)) )` across all query result sets.
    * It scores documents based on their position relative to other candidates rather than raw unnormalized similarity scores.
    * The constant `k = 60` (originally proposed by Cormack, Clarke, and Buettcher in SIGIR 2009) acts as a smoothing factor that prevents top-1 hits in a single query from completely dominating candidates that ranked consistently high (e.g. #2 and #3) across multiple queries.
    """)

with st.expander("Q3: What is HyDE (Hypothetical Document Embeddings)?"):
    st.markdown("""
    **Answer**:
    * In standard RAG, the system embeds a **short question** (e.g. 5 words) and searches against **long document passages** (e.g. 250 words).
    * Because questions and answers have different grammatical and semantic structures, their vectors can be misaligned.
    * **HyDE**: Instructs an LLM to generate a hypothetical answer passage first (even if hallucinated!).
    * The system then embeds the *hypothetical answer* and searches ChromaDB. Because document passages are compared against document passages, retrieval recall increases substantially.
    """)

with st.expander("Q4: When should you NOT apply Query Expansion in a search pipeline?"):
    st.markdown("""
    **Answer**:
    1. **Strict Real-Time Latency Budgets (< 50ms)**: Calling an LLM to expand queries introduces 200–500ms of extra generation latency before search begins.
    2. **Exact Identifier Lookups**: Queries searching for specific SKU codes, UUIDs, or commit hashes do not benefit from semantic expansion; direct metadata filtering is superior.
    3. **Token Cost Constraints**: Expanding every inbound user query with an LLM multiplies LLM API token consumption.
    """)

with st.expander("Q5: How does combining distance cutoff thresholds with RRF eliminate hallucinations?"):
    st.markdown("""
    **Answer**:
    * RRF merges candidate ranks, but if an entire query is out-of-domain, even the highest-ranked document will have a terrible geometric distance (e.g. `0.75`).
    * By applying a **Distance Cutoff Filter** (`dist <= 0.35`) **before or during RRF aggregation**, low-confidence candidate vectors are discarded.
    * If no candidates survive the filter, the pipeline terminates safely without invoking the LLM, achieving zero hallucination risk!
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Averaging Raw Cosine Distances Across Queries**  
    Averaging uncalibrated distances across different queries is mathematically invalid. Use Reciprocal Rank Fusion (RRF) based on rank positions.
    """)
    st.warning("""
    **Trap 2: Sequential Loop Queries for Expanded Variants**  
    Running 3 expanded queries in a Python `for` loop triples network round-trip time. Pass all variants simultaneously in `query_texts=[Q1, Q2, Q3]`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Omitting Distance Cutoffs on Fused Results**  
    Running RRF without distance thresholds will still crown a #1 champion even on completely irrelevant out-of-domain queries, inducing hallucinations.
    """)
    st.warning("""
    **Trap 4: Excessive Query Expansion (> 5 Variants)**  
    Generating 10 query variants creates diminishing returns, increases LLM pre-search latency to >1 second, and floods the candidate pool with noise. Use 2 to 3 variants.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is RRF formula?"):
        st.markdown("""
        **`sum(1.0 / (60 + rank))`**:  
        Ranks candidates by consensus across multiple search queries.
        """)
    with st.expander("🗂️ Flashcard: What is HyDE?"):
        st.markdown("""
        **Hypothetical Document Embeddings**:  
        Embeds a generated answer to search doc-to-doc instead of question-to-doc.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why batch expanded queries?"):
        st.markdown("""
        **Single Round-Trip**:  
        Passing `query_texts=[Q1, Q2, Q3]` executes in parallel via vectorized SIMD.
        """)
    with st.expander("🗂️ Flashcard: Why filter before fusion?"):
        st.markdown("""
        **Hallucination Shield**:  
        Prevents low-confidence noise from accumulating RRF points.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the three query optimization levers: Top-K tuning, threshold gating, and query expansion.", value=True)
chk2 = st.checkbox("I know why Reciprocal Rank Fusion (RRF) operates on rank positions rather than uncalibrated distance averages.", value=True)
chk3 = st.checkbox("I understand how HyDE bridges the semantic gap between short questions and long document passages.", value=True)
chk4 = st.checkbox("I know how to batch expanded queries into a single `collection.query()` execution for sub-10ms retrieval.", value=True)
chk5 = st.checkbox("I can implement an end-to-end RRF search pipeline with distance threshold gating in Python.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 8: Getting and Retrieving Data in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 7: Querying ChromaDB!")
