import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="6.4 Chunk Optimization - RAG Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class ChunkOptimizerSimulator:
    """Simulates retrieval recall and token cost across different chunk parameter combinations."""
    def __init__(self):
        pass

    def evaluate_configuration(self, chunk_size: int, overlap_pct: int) -> Dict[str, Any]:
        # Simulated trade-off curve
        # Optimum is around 350-500 tokens with 15% overlap
        overlap_tokens = int(chunk_size * (overlap_pct / 100.0))
        
        # Penalize if too small (<150) due to context loss, or too large (>800) due to noise
        if chunk_size < 150:
            recall = 0.62 + (chunk_size / 500.0) * 0.15
            noise_ratio = 0.05
        elif chunk_size <= 500:
            recall = 0.85 + (chunk_size / 1000.0) * 0.12
            noise_ratio = 0.12 + (chunk_size / 2000.0) * 0.1
        else:
            recall = 0.90 - ((chunk_size - 500) / 1000.0) * 0.18
            noise_ratio = 0.25 + ((chunk_size - 500) / 1000.0) * 0.35

        # Overlap bonus
        if overlap_pct >= 10 and overlap_pct <= 25:
            recall = min(0.98, recall + 0.06)
        elif overlap_pct > 40:
            noise_ratio += 0.15

        estimated_chunks = int(10000 / max(1, (chunk_size - overlap_tokens)))
        monthly_cost_usd = round(estimated_chunks * 0.00002 * 100, 3)

        return {
            "chunk_size": chunk_size,
            "overlap_tokens": overlap_tokens,
            "overlap_pct": f"{overlap_pct}%",
            "estimated_chunks_per_10k_words": estimated_chunks,
            "simulated_recall_at_5": round(recall, 3),
            "context_noise_ratio": round(noise_ratio, 3),
            "storage_cost_index": monthly_cost_usd
        }

# Title & Overview
st.title("6.4 Chunk Optimization: Tuning Size, Overlap & Evaluation Metrics")
st.caption("Chapter 6: Text Chunking | Domain: 6. RAG")
st.markdown("""
Configuring chunk size and overlap is not a matter of guesswork or copying default tutorials. 
An arbitrary `chunk_size = 1000` chosen in a prototype often becomes a multi-million-dollar bottleneck 
in production, causing either severe context dilution or catastrophic retrieval failure.

**Chunk Optimization** is an empirical systems engineering process: treating chunk size and overlap as 
hyperparameters evaluated through systematic grid sweeps against domain-specific test questions, 
measuring **Hit Rate@K**, **Mean Reciprocal Rank (MRR)**, and **Faithfulness**.

In this lesson, you will master:
1. **The Optimization Trade-Off Landscape**: Navigating precision, recall, latency, and vector storage costs.
2. **Empirical Overlap Guidelines**: Why the 10–20% overlap rule balances boundary continuity without index bloat.
3. **Preventing Context Loss & Noise**: Combating the *Lost-in-the-Middle* effect in large LLM context windows.
4. **Automated Evaluation Pipelines**: Measuring chunking quality using RAGAS and TruLens frameworks.
5. **Production Failure Traps**: Avoiding unmonitored index bloat, micro-chunk fragmentation, and evaluation data leakage.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Chunk Optimization & Evaluation Feedback Loop", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChunkOptimizationBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_grid {
            label="Tier 1: Chunk Parameter Grid Search Sweeper";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            corpus [label="Golden Benchmark Knowledge Corpus\\n50 representative technical documents", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sweeper [label="Grid Parameter Permutator\\nSizes: [256, 512, 1024] | Overlaps: [0%, 10%, 20%]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            corpus -> sweeper;
        }

        subgraph cluster_eval {
            label="Tier 2: Dual-Stage Retrieval & Generation Evaluation Harness";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            retrieval_eval [label="Retrieval Metrics Auditor\\nMeasures Hit Rate@5, MRR, NDCG", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            generation_eval [label="Generation Faithfulness Judge\\nRAGAS LLM-as-a-Judge: Faithfulness, Context Relevancy", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            cost_auditor [label="Infrastructure Cost & Latency Auditor\\nVector DB RAM consumption, QPS throughput SLA", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            sweeper -> retrieval_eval;
            sweeper -> generation_eval;
            sweeper -> cost_auditor;
        }

        subgraph cluster_decision {
            label="Tier 3: Pareto-Optimal Configuration Selection";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            pareto_matrix [label="Pareto Frontier Selection Engine\\nMaximizes Recall@K while minimizing Context Noise Ratio", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            prod_config [label="Locked Production Configuration\\nChunk Size: 400 tokens | Overlap: 15% (60 tokens)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            retrieval_eval -> pareto_matrix;
            generation_eval -> pareto_matrix;
            cost_auditor -> pareto_matrix;
            pareto_matrix -> prod_config;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Choosing Chunk Size
st.header("2. Choosing Chunk Size: Matching Corpus Structure to Query Modality", divider="rainbow")
st.markdown("""
The optimal chunk size is governed by two empirical variables: **The nature of the knowledge corpus** and 
**The user's query intent**:

* **Short, Atomic Queries (e.g., *"What is the timeout threshold in section 4.2?"*)**:
  * Demands **small chunk sizes (150–250 tokens)**.
  * A compact chunk matches the query embedding tightly without being dragged down by irrelevant surrounding paragraphs.
* **Complex, Multi-Hop Synthesis Queries (e.g., *"Compare our European and Asian privacy compliance architectures."*)**:
  * Demands **larger chunk sizes (600–1000 tokens)** or hierarchical Parent-Document retrieval.
  * Small fragments fail because the LLM requires the overarching reasoning chain to synthesize comparative conclusions.
* **The Rule of Thumb**: For standard enterprise documentation, begin experiments at **400–500 tokens**, sweeping down to 250 for FAQ lookups and up to 800 for legal contracts.
""")
st.divider()

# Section 3: Choosing Overlap
st.header("3. Choosing Overlap: The Empirical 10–20% Rule", divider="rainbow")
st.markdown("""
Why is **10% to 20%** the universal industry consensus for chunk overlap?

* **0% Overlap (Disastrous)**: Boundary sentences are split arbitrarily. Information crossing a page boundary suffers complete semantic severance.
* **10–20% Overlap (Optimal Sweet Spot)**: In a 500-token chunk, a 15% overlap corresponds to 75 tokens (~55 words, or roughly 2 complete sentences). This provides sufficient room to capture any phrase, clause, or noun reference that spans the boundary without bloating the database.
* **>40% Overlap (Severe Diminishing Returns)**: Increases vector index size by 2x–3x, inflates embedding generation costs, and wastes valuable LLM prompt context by passing near-duplicate text passages.
""")
st.divider()

# Section 4: Avoiding Excessive Chunks
st.header("4. Avoiding Excessive Chunks: Combating Vector Bloat and Dilution", divider="rainbow")
st.markdown("""
When documents are split into tens of thousands of tiny micro-chunks (e.g., 50 tokens each), several severe production failures occur:
* **Vector Database Memory Bloat**: Every chunk generates an HNSW graph node and edge pointers. Indexing 1,000,000 micro-chunks consumes 4x more RAM than 250,000 well-sized chunks.
* **Query Dilution**: During top-K search, all top-5 results may come from the exact same paragraph, crowding out other diverse perspectives across the document.
* **Embedding Model Overhead**: Generating embeddings for millions of tiny fragments incurs substantial GPU compute or API billing costs.
""")
st.divider()

# Section 5: Avoiding Context Loss
st.header("5. Avoiding Context Loss: The Lost-in-the-Middle Phenomenon", divider="rainbow")
st.markdown("""
Research (*Liu et al., Stanford, 2023*) demonstrated that Large Language Models suffer from the 
**Lost-in-the-Middle Phenomenon**:
* LLMs exhibit highest recall when critical facts appear at the **very beginning** or **very end** of their prompt context window.
* When critical facts are buried in the middle of long retrieved context stuffing (chunks 4 through 8 in a 10-chunk prompt), LLM retrieval accuracy drops by over 30%!
* **Optimization Implication**: Packing huge 1500-token chunks into the prompt forces the LLM into the low-recall middle zone. Tighter, optimized chunks (350–500 tokens) keep the total context concise, positioning facts near high-attention prompt borders.
""")
st.divider()

# Section 6: Chunk Quality Evaluation
st.header("6. Chunk Quality Evaluation: Quantitative Metric Frameworks", divider="rainbow")
st.markdown("""
Never rely on subjective intuition when tuning chunking configurations. Deploy automated quantitative evaluation:
""")

st.markdown("""
| Evaluation Metric | What It Measures | Target Production SLA |
| :--- | :--- | :--- |
| **Hit Rate@K** | Percentage of test queries where the correct ground-truth chunk appears in top-K results. | `>= 0.88` (at K=5) |
| **Mean Reciprocal Rank (MRR)**| Measures how close the correct chunk was to the #1 rank position (`1 / rank`). | `>= 0.75` |
| **Context Noise Ratio** | Proportion of tokens in retrieved chunks that are irrelevant to the query. | `<= 0.25` |
| **Faithfulness (RAGAS)** | Whether the LLM's generated answer is mathematically grounded in the retrieved chunks. | `>= 0.92` |
""")
st.divider()

# Section 7: Chunking Experiments
st.header("7. Chunking Experiments: Systematic A/B Grid Sweeps", divider="rainbow")
st.markdown("""
To optimize chunking for an enterprise corpus, execute a **Grid Search Experiment**:

1. **Curate a Golden Test Set**: Compile 50–100 representative user queries paired with verified ground-truth answer passages.
2. **Define the Parameter Matrix**:
   * Chunk Sizes: `[250, 500, 750, 1000]`
   * Overlap Percentages: `[0%, 10%, 20%]`
   * Strategies: `[RecursiveCharacter, Sentence, Semantic]`
3. **Automated Sweep Execution**: Programmatically ingest the corpus into temporary vector collections across all 12 combinations.
4. **Evaluate & Chart the Pareto Frontier**: Plot Retrieval Hit Rate@5 against Storage Cost and Latency to select the optimal operating point.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production platforms integrate **Continuous Chunk Experimentation in CI/CD pipelines** using tools like 
**LangSmith** or **Ragas**. Every time documentation changes, an automated workflow tests whether the current chunking 
parameters maintain the required Hit Rate@5 SLA before deploying indices to production vector clusters.
""")
st.divider()

# Section 8: Production Implementation
st.header("8. Production Implementation: Grid Search Chunk Evaluation Harness", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating an automated evaluation harness that sweeps 
chunk configurations and computes simulated retrieval metrics.
""")

st.code('''
from typing import List, Dict, Any

class ChunkGridSearchHarness:
    """Automates grid search evaluation across chunk size and overlap configurations."""
    def __init__(self, test_queries: List[Dict[str, str]]):
        self.test_queries = test_queries

    def run_sweep(
        self, 
        chunk_sizes: List[int], 
        overlap_ratios: List[float]
    ) -> List[Dict[str, Any]]:
        results = []
        for size in chunk_sizes:
            for overlap in overlap_ratios:
                overlap_tokens = int(size * overlap)
                
                # Mock evaluation: calculate trade-off score
                # Optimal balance typically around 400-500 tokens with 15% overlap
                if size < 200:
                    hit_rate = 0.72 + (overlap * 0.2)
                elif size <= 600:
                    hit_rate = 0.88 + (overlap * 0.3)
                else:
                    hit_rate = 0.82 - ((size - 600) / 2000.0)

                hit_rate = min(0.98, round(hit_rate, 3))
                mrr = round(hit_rate * 0.86, 3)

                results.append({
                    "chunk_size": size,
                    "overlap_ratio": overlap,
                    "overlap_tokens": overlap_tokens,
                    "hit_rate_at_5": hit_rate,
                    "mrr": mrr
                })

        return sorted(results, key=lambda x: x["hit_rate_at_5"], reverse=True)
''', language="python")
st.divider()

# Section 9: Interactive Workbench
st.header("9. Interactive Workbench: Chunk Optimization Parameter Sweeper & Recall Simulator", divider="rainbow")
st.markdown("""
Experiment with different chunk size and overlap configurations to observe simulated recall, context noise ratios, 
and vector storage index overhead.
""")

opt_sim = ChunkOptimizerSimulator()

col_sw1, col_sw2 = st.columns(2)
with col_sw1:
    sim_size = st.select_slider("Target Chunk Size (Tokens):", options=[100, 200, 350, 500, 750, 1000, 1500], value=500)
with col_sw2:
    sim_overlap_pct = st.select_slider("Overlap Percentage:", options=[0, 10, 15, 20, 30, 40, 50], value=15)

sim_metrics = opt_sim.evaluate_configuration(sim_size, sim_overlap_pct)

st.subheader("Performance & Quality Trade-Off Metrics")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Simulated Hit Rate@5", f"{sim_metrics['simulated_recall_at_5'] * 100:.1f}%")
col_m2.metric("Context Noise Ratio", f"{sim_metrics['context_noise_ratio'] * 100:.1f}%")
col_m3.metric("Chunks per 10k Words", f"{sim_metrics['estimated_chunks_per_10k_words']}")
col_m4.metric("Storage Cost Index", f"${sim_metrics['storage_cost_index']}")

if sim_size >= 350 and sim_size <= 600 and sim_overlap_pct in [10, 15, 20]:
    st.success("🎯 **Optimal Pareto Operating Point**: Exceptional balance between high retrieval recall and low prompt context noise!")
elif sim_size < 200:
    st.warning("⚠️ **Under-sized Chunks**: High precision, but severe risk of context loss and anaphoric pronoun severance.")
elif sim_size > 800:
    st.warning("⚠️ **Over-sized Chunks**: Noticeable context noise ratio. Vulnerable to the Lost-in-the-Middle phenomenon.")
st.divider()

# Section 10: Technical Interview Preparation & Production Traps
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is the Lost-in-the-Middle phenomenon, and how does chunk optimization combat it?", expanded=True):
    st.markdown("""
    **Answer:**
    The Lost-in-the-Middle phenomenon describes how transformer language models exhibit high retrieval attention at the very beginning 
    and end of long prompt context windows, but experience sharp degradation (up to 30% drop in factual recall) when relevant facts are 
    positioned in the middle of long context passages.
    
    Chunk optimization combats this by tuning chunk size to produce compact, high-density passages (300–500 tokens). This keeps the total 
    retrieved prompt context small (e.g., 3 chunks = 1200 tokens), preventing the context window from expanding into the low-recall middle territory.
    """)

with st.expander("Q2: Why should you avoid 0% overlap in production chunking pipelines?"):
    st.markdown("""
    **Answer:**
    At 0% overlap, sentences crossing chunk boundaries are severed arbitrarily. If a critical statement begins on line 100 and finishes on line 101, 
    splitting between them creates two incomplete fragments. Neither fragment contains both the premise and the conclusion, causing vector similarity 
    searches to fail to retrieve either chunk with confidence. Overlapping by 10–20% guarantees boundary continuity.
    """)

with st.expander("Q3: How do you choose between optimizing for Hit Rate@K versus optimizing for Faithfulness?"):
    st.markdown("""
    **Answer:**
    * **Hit Rate@K** measures retrieval: did the vector search successfully retrieve the chunk containing the ground truth? (Driven by appropriate chunk size, semantic specificity, and overlap).
    * **Faithfulness** measures generation: did the LLM generate an answer derived *strictly* from that retrieved chunk without hallucinating? (Driven by context sufficiency and low noise).
    In production, a Pareto-optimal configuration achieves high Hit Rate@5 (>= 0.88) while maintaining high Faithfulness (>= 0.92) without bloating token costs.
    """)

with st.expander("Q4: What causes Vector Database bloat during chunking experiments?"):
    st.markdown("""
    **Answer:**
    Vector database bloat occurs when developers use micro-chunking (e.g., 50–100 tokens) combined with excessive overlap (>40%). 
    A document that would normally produce 50 cohesive chunks generates 400 near-identical micro-chunks. This consumes 8x more RAM 
    for HNSW graph edges, degrades search QPS, and crowds the top-K results with redundant copies of the same sentence.
    """)

with st.expander("Q5: What is the difference between Mean Reciprocal Rank (MRR) and Hit Rate@K?"):
    st.markdown("""
    **Answer:**
    * **Hit Rate@K**: A binary measure of whether the correct ground-truth chunk appeared anywhere within the top-K retrieved results (0 or 1).
    * **Mean Reciprocal Rank (MRR)**: Evaluates the *position* of the first relevant chunk (`1 / rank`). If the correct chunk is at rank 1, score is 1.0; at rank 2, score is 0.5; at rank 5, score is 0.2. MRR rewards systems that place the most relevant chunk at the very top.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Deploying Default Tutorial Chunk Sizes to Production Without Benchmarking (Accuracy Failure)**
    Using `chunk_size = 1000` from quickstart tutorials on dense medical or legal corpora. The 1000-token chunks bundle 4 separate contractual obligations into a single vector, diluting specific query retrieval and cutting Hit Rate@5 by over 40%.
    """)
    st.error("""
    **Trap 2: Micro-Chunking Creating Massive Index Bloat and Memory OOM (Infrastructure Cost Disaster)**
    Setting chunk size to 60 tokens with 30-token overlap across a 500,000-page document repository. The system generates over 15 million vectors, crashing HNSW memory limits in the vector database and inflating cloud hosting costs by 10x.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Data Contamination in Chunking Evaluation Benchmarks (False Confidence)**
    Evaluating chunking configurations using test questions generated by the same LLM from the exact chunks being evaluated. The evaluation demonstrates 99% hit rate in testing, but collapses when real human users submit conversational queries in production.
    """)
    st.warning("""
    **Trap 4: Setting Overlap Above 40% Flooding Prompt Context with Redundant Duplicates (Context Waste)**
    Configuring overlap to 50%. The vector database retrieves top-3 chunks that share 70% of the identical sentences, wasting 60% of the LLM's prompt token budget on duplicate text instead of providing diverse perspectives.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Lost-in-the-Middle phenomenon?")
        st.markdown("**Back**: LLMs recall information accurately from the beginning and end of long context prompts, but recall drops sharply for facts in the middle.")
    with st.container(border=True):
        st.markdown("**Front**: What is the recommended baseline chunk size for enterprise RAG?")
        st.markdown("**Back**: 350 to 500 tokens with 10% to 20% overlap, balancing semantic focus with sufficient context.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What does Mean Reciprocal Rank (MRR) evaluate?")
        st.markdown("**Back**: The positional rank of the first relevant chunk (1/rank), rewarding systems that place correct chunks at rank 1.")
    with st.container(border=True):
        st.markdown("**Front**: What is the danger of setting chunk overlap above 40%?")
        st.markdown("**Back**: It causes vector index bloat and floods LLM prompts with redundant, duplicate sentences.")

st.divider()

# Section 11: Self-Assessment Checklist
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Chunk Optimization and Evaluation before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the trade-offs between chunk size, semantic specificity, and context noise.", value=True, key="chk_opt_1")
    chk2 = st.checkbox("I know why the 10-20% overlap rule balances boundary continuity without database bloat.", value=True, key="chk_opt_2")
    chk3 = st.checkbox("I can explain the Lost-in-the-Middle phenomenon and how compact chunking combats it.", value=True, key="chk_opt_3")
with c2:
    chk4 = st.checkbox("I know how to evaluate chunk configurations using Hit Rate@K, MRR, and Faithfulness.", value=True, key="chk_opt_4")
    chk5 = st.checkbox("I understand how to design and execute systematic A/B grid search chunking sweeps.", value=True, key="chk_opt_5")
    chk6 = st.checkbox("I can identify production traps such as uncalibrated tutorial parameters and test set contamination.", value=True, key="chk_opt_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Chunk Optimization Mastery Demonstrated!** You have completed Chapter 6 of the RAG curriculum.")
