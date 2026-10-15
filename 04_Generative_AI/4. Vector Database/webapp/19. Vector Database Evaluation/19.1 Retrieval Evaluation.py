import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="19.1 Retrieval Evaluation - Vector DB Curriculum",
    page_icon="📊",
    layout="wide"
)

# Title & Overview
st.title("19.1 Retrieval Evaluation: Hit Rate, Precision@K, Recall@K, MRR & NDCG")
st.caption("Chapter 19: Vector Database Evaluation | Domain: 4. Vector Database")
st.markdown("""
A common failure in generative AI development is evaluating only the final LLM response while treating 
the vector database retrieval layer as an opaque black box. 

If the retrieval engine returns poor or misordered context, even a frontier model will hallucinate. 
To build reliable AI systems, engineers must establish **Component-Level Retrieval Evaluation** using 
standardized Information Retrieval (IR) metrics.

In this lesson, we explore the mathematics, interpretations, and trade-offs of the five foundational retrieval metrics:
1. **Hit Rate (Success@K)**: The binary probability that at least one relevant document appears in the top K.
2. **Precision@K**: The fraction of retrieved passages that are actually relevant.
3. **Recall@K**: The fraction of all relevant ground-truth passages captured in the top K.
4. **Mean Reciprocal Rank (MRR@K)**: How quickly the search engine surfaces the *first* relevant passage.
5. **Normalized Discounted Cumulative Gain (NDCG@K)**: The industry gold standard for multi-graded relevance ranking.

We will examine how to build golden evaluation benchmarks and test an interactive Live IR Metric Calculator.
""")

st.divider()

# Section 1: The Information Retrieval Metric Suite
st.header("1. The Information Retrieval Metric Suite", divider="rainbow")

col_suite1, col_suite2 = col_suite1, col_suite2 = st.columns([1.1, 0.9])

with col_suite1:
    st.markdown("""
    ### The Five Core Metrics Defined
    
    1. **Hit Rate@K**:
       * `Hit Rate@K = 1 if any relevant doc in Top-K else 0`.
       * *Best For*: Quick triage benchmarks and FAQ search where a single correct hit suffices.
    
    2. **Precision@K**:
       * `Precision@K = (Relevant Docs in Top-K) / K`.
       * *Best For*: Measuring noise. Low precision means the LLM prompt is being polluted with irrelevant text.
    
    3. **Recall@K**:
       * `Recall@K = (Relevant Docs in Top-K) / (Total Relevant Docs in Ground Truth)`.
       * *Best For*: Multi-hop research where missing a key reference breaks downstream synthesis.
    
    4. **Mean Reciprocal Rank (MRR@K)**:
       * `RR = 1.0 / Rank_first_relevant_doc`. (MRR is the average across all test queries).
       * *Best For*: Question answering and chatbots where the user reads the first top result.
    
    5. **Normalized Discounted Cumulative Gain (NDCG@K)**:
       * Evaluates documents on a multi-graded scale (e.g. 0=Irrelevant, 1=Partial, 2=Relevant, 3=Perfect).
       * Rewards engines that place Grade 3 documents at Rank 1 and penalizes placing them at Rank 5.
    """)

with col_suite2:
    st.markdown("### Metric Application Decision Tree")
    tree_dot = """
    digraph MetricDecisionTree {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Start [label="Evaluation Objective", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Q1 [label="Is relevance binary (0 or 1)\\nor multi-graded (0, 1, 2, 3)?", shape=diamond, fillcolor="#FEF3C7"];

        BinaryBranch [label="Does order/rank position\\nmatter to the LLM?", shape=diamond, fillcolor="#EDE9FE"];

        NDCG [label="Use NDCG@K\\n(Gold standard for graded ranking)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        MRR [label="Use MRR@K\\n(Prioritizes finding 1st match fast)", fillcolor="#D1FAE5", color="#059669"];

        PR [label="Use Precision@K & Recall@K\\n(Measures noise and completeness)", fillcolor="#D1FAE5", color="#059669"];

        Start -> Q1;
        Q1 -> NDCG [label="Multi-graded"];
        Q1 -> BinaryBranch [label="Binary"];
        BinaryBranch -> MRR [label="Rank matters"];
        BinaryBranch -> PR [label="Set presence"];
    }
    """
    st.graphviz_chart(tree_dot, use_container_width=False)

st.divider()

# Section 2: Mathematical Formulations: MRR vs NDCG
st.header("2. Mathematical Formulations: MRR vs NDCG", divider="rainbow")

col_math_a, col_math_b = col_math_a, col_math_b = st.columns([1, 1])

with col_math_a:
    st.markdown("""
    ### Mean Reciprocal Rank (MRR)
    For a test suite of `Q` queries:
    ```text
    MRR = (1 / Q) * sum_{i=1}^{Q} ( 1 / rank_i )
    ```
    Where `rank_i` is the rank position of the **first relevant document** for query `i`.
    * If the first relevant document is at Rank 1: `RR = 1 / 1 = 1.0`.
    * If at Rank 2: `RR = 1 / 2 = 0.5`.
    * If at Rank 5: `RR = 1 / 5 = 0.2`.
    * If absent from Top-K: `RR = 0.0`.
    """)

with col_math_b:
    st.markdown("""
    ### Discounted Cumulative Gain (DCG & NDCG)
    Discounted Cumulative Gain (DCG) rewards highly relevant documents placed high in the ranking:
    ```text
    DCG@K = sum_{i=1}^{K} [ (2^{rel_i} - 1) / log2(i + 1) ]
    ```
    Where `rel_i` is the graded relevance score of document at rank `i`.
    
    * **Ideal DCG (IDCG@K)**: The DCG score achieved if candidates were sorted in perfect descending order of relevance.
    * **Normalized DCG**:
      ```text
      NDCG@K = DCG@K / IDCG@K
      ```
    * **Properties**: Bounded strictly between `0.0` (worst) and `1.0` (flawless optimal ranking).
    """)

st.divider()

# Section 3: Clean Python Implementation of IR Metrics
st.header("3. Clean Python Implementation of IR Metrics", divider="rainbow")

st.code("""import numpy as np
from typing import List

def compute_retrieval_metrics(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> dict:
    \"\"\"Computes binary IR metrics for a single query.\"\"\"
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    
    # 1. Hit Rate
    hits = [1 if doc_id in gt_set else 0 for doc_id in top_k]
    hit_rate = 1.0 if any(hits) else 0.0
    
    # 2. Precision@K
    precision_at_k = sum(hits) / k
    
    # 3. Recall@K
    recall_at_k = sum(hits) / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # 4. Reciprocal Rank (RR)
    reciprocal_rank = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit == 1:
            reciprocal_rank = 1.0 / idx
            break
            
    return {
        f"HitRate@{k}": hit_rate,
        f"Precision@{k}": precision_at_k,
        f"Recall@{k}": recall_at_k,
        f"MRR@{k}": reciprocal_rank
    }

def compute_ndcg_at_k(relevance_grades: List[int], k: int = 5) -> float:
    \"\"\"Computes NDCG@K given multi-graded relevance integers [0..3].\"\"\"
    grades = relevance_grades[:k]
    if not grades or sum(grades) == 0:
        return 0.0
        
    # Calculate DCG
    dcg = sum((2**rel - 1) / np.log2(idx + 1) for idx, rel in enumerate(grades, start=1))
    
    # Calculate IDCG (Ideal DCG)
    ideal_grades = sorted(grades, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(idx + 1) for idx, rel in enumerate(ideal_grades, start=1))
    
    return float(dcg / idcg) if idcg > 0 else 0.0
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live IR Metric Calculator
st.header("4. Interactive Workbench: Live IR Metric Calculator", divider="rainbow")

st.markdown("""
Assign graded relevance ratings to five retrieved candidate documents to observe how 
Hit Rate, Precision@K, Recall@K, MRR, and NDCG respond in real time.
""")

col_wb_cand, col_wb_res = st.columns([1.1, 0.9])

with col_wb_cand:
    st.subheader("1. Retrieved Candidate Rankings (Top 5)")
    st.caption("Rate relevance: 0 = Irrelevant | 1 = Partial | 2 = Relevant | 3 = Perfect Match")
    
    rel_r1 = st.select_slider("Rank 1 Document Relevance:", options=[0, 1, 2, 3], value=2)
    rel_r2 = st.select_slider("Rank 2 Document Relevance:", options=[0, 1, 2, 3], value=0)
    rel_r3 = st.select_slider("Rank 3 Document Relevance:", options=[0, 1, 2, 3], value=3)
    rel_r4 = st.select_slider("Rank 4 Document Relevance:", options=[0, 1, 2, 3], value=1)
    rel_r5 = st.select_slider("Rank 5 Document Relevance:", options=[0, 1, 2, 3], value=0)
    
    total_relevant_gt = st.slider("Total Relevant Documents in Entire Ground Truth:", min_value=1, max_value=8, value=3)

# Metric Calculations
grades = [rel_r1, rel_r2, rel_r3, rel_r4, rel_r5]
binary_hits = [1 if g >= 2 else 0 for g in grades]

hit_rate_val = 1.0 if any(binary_hits) else 0.0
precision_val = sum(binary_hits) / 5.0
recall_val = sum(binary_hits) / total_relevant_gt

# RR calculation
mrr_val = 0.0
for idx, h in enumerate(binary_hits, start=1):
    if h == 1:
        mrr_val = 1.0 / idx
        break

# NDCG calculation
dcg_val = sum((2**g - 1) / np.log2(i + 1) for i, g in enumerate(grades, start=1))
ideal_grades = sorted(grades, reverse=True)
idcg_val = sum((2**g - 1) / np.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))
ndcg_val = (dcg_val / idcg_val) if idcg_val > 0 else 0.0

with col_wb_res:
    st.subheader("2. Calculated Information Retrieval Metrics")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Hit Rate@5", value=f"{hit_rate_val * 100:.0f}%")
        st.metric(label="Precision@5", value=f"{precision_val * 100:.1f}%")
        st.metric(label="Recall@5", value=f"{recall_val * 100:.1f}%")
    with m_col2:
        st.metric(label="Reciprocal Rank (MRR)", value=f"{mrr_val:.3f}")
        st.metric(label="NDCG@5 Score", value=f"{ndcg_val:.4f}", delta="Optimal Ranking" if ndcg_val >= 0.95 else "Suboptimal Rank Order")
    
    st.divider()
    
    if rel_r1 < rel_r3 and rel_r3 == 3:
        st.warning("⚠️ Ranking Penalty: Your best document (Grade 3) was ranked at Rank 3 rather than Rank 1. Notice how NDCG dropped below 1.0 to penalize the delay!")
    elif ndcg_val == 1.0:
        st.success("🎉 Flawless Ranking: Documents are sorted in perfect order of relevance (IDCG = DCG).")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the primary difference between Recall@K and Precision@K in RAG retrieval evaluation?"):
    st.markdown("""
    **Answer**:
    * **Precision@K**: Measures the proportion of retrieved chunks that are actually relevant: `(Relevant Chunks) / K`. High precision ensures that the prompt context is dense and noise-free.
    * **Recall@K**: Measures the proportion of all relevant information in the ground truth that was captured: `(Relevant Chunks) / (Total Relevant in Corpus)`. High recall guarantees that multi-hop questions have all necessary facts.
    * **The Trade-off**: Increasing `K` from 3 to 15 almost always **increases Recall@K** (more chances to catch the document) but **decreases Precision@K** (introduces irrelevant noise chunks).
    """)

with st.expander("Q2: Why is NDCG widely considered superior to MRR for evaluating search engines and Rerankers?"):
    st.markdown("""
    **Answer**:
    * **MRR Limitation**: MRR is a **binary, single-target metric**. It only tracks the position of the *first* relevant document, completely ignoring whether documents at Rank 2, 3, and 4 are high quality or useless garbage.
    * **NDCG Superiority**: NDCG supports **multi-graded relevance** (0=Irrelevant, 1=Marginal, 2=Relevant, 3=Vital) and evaluates the quality of the **entire ranked list**. 
    * It applies a logarithmic position discount (`log2(rank + 1)`), reflecting human cognitive behavior: users (and LLM attention mechanisms) scrutinize early ranks far more carefully than later ranks.
    """)

with st.expander("Q3: How do you construct a Synthetic Golden Test Set to evaluate a vector database when human labels are unavailable?"):
    st.markdown("""
    **Answer**:
    Use the **LLM-Based Question Generation Pattern**:
    1. Sample 500 representative chunks from your production corpus.
    2. Prompt a frontier model (GPT-4o or Claude 3.5):  
       *"Given this passage, generate 3 realistic, nuanced questions that can ONLY be answered by this passage. Output as JSON."*
    3. Pair each generated question with the source chunk ID as ground truth: `(Question, Ground_Truth_Chunk_ID)`.
    4. Query your vector database with the generated questions and measure Hit Rate@K and MRR@K.
    5. This provides an automated, reproducible regression test suite for CI/CD pipelines.
    """)

with st.expander("Q4: What causes a high Hit Rate@5 but low RAG answer accuracy?"):
    st.markdown("""
    **Answer**:
    This indicates a **Lost-in-the-Middle or Context Dilution failure**:
    * The correct chunk is successfully retrieved (Hit Rate is 100%), but it is placed at Rank 4 or Rank 5 surrounded by 4 irrelevant chunks.
    * The LLM suffers from attention degradation over the irrelevant text and misses the crucial sentence in Chunk 4.
    * **Remedy**: Deploy a Stage-2 Cross-Encoder reranker to elevate the correct chunk to Rank 1, maximizing MRR.
    """)

with st.expander("Q5: What is the difference between Macro-Averaging and Micro-Averaging across test queries?"):
    st.markdown("""
    **Answer**:
    * **Macro-Averaging**: Computes the metric for each query individually, then takes the unweighted average across all queries. Every query contributes equally.
    * **Micro-Averaging**: Aggregates all relevant documents and hits across all queries globally before computing the metric. Queries with many relevant documents dominate the score.
    * **Best Practice**: In IR benchmarking, **Macro-Averaging is the standard** because every user question deserves equal evaluation weight.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Evaluating RAG Only at the LLM Response Level**  
    Evaluating only final generated answers hides whether mistakes are caused by bad vector retrieval or LLM reasoning failure. Always evaluate retrieval independently.
    """)
    st.warning("""
    **Trap 2: Using Top-K = 1 to Benchmark Recall**  
    Setting K=1 measures Hit Rate@1, not recall. True recall requires setting K to match your pipeline's downstream retrieval window (e.g. K=5).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Uncalibrated Synthetic Ground Truth Questions**  
    Generating synthetic questions that contain exact string keywords from the chunk artificially inflates retrieval scores. Instruct the LLM to paraphrase queries using synonyms.
    """)
    st.warning("""
    **Trap 4: Forgetting the Logarithmic Discount in NDCG**  
    Using a linear discount (`1 / rank`) instead of `1 / log2(rank + 1)` penalizes lower ranks too harshly, distorting model comparisons. Stick to standard log2 discounting.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Hit Rate@K?"):
        st.markdown("""
        **Success Metric**:  
        Binary indicator (0 or 1) of whether at least one relevant document appears in the top K.
        """)
    with st.expander("🗂️ Flashcard: What is MRR?"):
        st.markdown("""
        **Reciprocal Rank**:  
        `1.0 / rank` of the first relevant document. Rewards engines that surface the primary answer fast.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What makes NDCG unique?"):
        st.markdown("""
        **Graded & Discounted**:  
        Supports multi-graded relevance (0-3) and penalizes misordered ranks using logarithmic decay (`log2(i+1)`).
        """)
    with st.expander("🗂️ Flashcard: Why evaluate retrieval separately?"):
        st.markdown("""
        **Isolating Failures**:  
        Prevents blaming the LLM when the root cause was missing or noisy retrieved context.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why component-level retrieval evaluation is essential for production RAG.", value=True)
chk2 = st.checkbox("I know how to calculate Hit Rate@K, Precision@K, and Recall@K.", value=True)
chk3 = st.checkbox("I understand the Mean Reciprocal Rank (MRR) formula and its focus on the first relevant hit.", value=True)
chk4 = st.checkbox("I know how Discounted Cumulative Gain (DCG) and NDCG evaluate multi-graded rankings.", value=True)
chk5 = st.checkbox("I can construct a synthetic golden evaluation dataset using LLM question generation.", value=True)
chk6 = st.checkbox("I can interpret IR evaluation benchmarks to identify context dilution and recall bottlenecks.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Retrieval Evaluation and are ready for Chapter 19.2: RAG Evaluation & The RAG Triad!")
