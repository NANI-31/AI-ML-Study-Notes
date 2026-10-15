import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="10.3 Top-K Retrieval - Embeddings Curriculum",
    page_icon="🔢",
    layout="wide"
)

# Title & Overview
st.title("10.3 Top-K Retrieval: Tuning Retrieval Depth, Token Budgets, & Two-Stage Reranking")
st.caption("Chapter 10: Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
In Information Retrieval and RAG, the parameter **K** dictates how many candidate documents are returned by the nearest-neighbor search:
* Set **K too small** ($K=1$), and the system risks missing the crucial piece of information if the single top vector has slight lexical divergence (**Low Recall Failure**).
* Set **K too large** ($K=20$), and the prompt becomes bloated with second-rate distractors that dilute the LLM's attention, inflate API billing, and slow down generation (**Lost in the Middle / Distractor Failure**).

Selecting the optimal $K$ is an engineering balancing act between **Recall Coverage**, **LLM Prompt Token Budgets**, and **Downstream Latency**.

This lesson explores the trade-offs across retrieval depths ($K=1, 3, 5, 10$), the economics of prompt token consumption, and the industry-standard **Two-Stage Retrieval Architecture** (Retrieve Top-50 $\\rightarrow$ Rerank to Top-3).
""")

st.divider()

# Section 1: The Retrieval Depth Spectrum: K=1 vs K=3 vs K=5 vs K=10
st.header("1. The Retrieval Depth Spectrum: Trade-offs Across Values of K", divider="rainbow")

col_k1, col_k2 = st.columns([1, 1])

with col_k1:
    st.markdown("""
    ### Analyzing Values of K:
    * **Top-1 Retrieval ($K = 1$)**:
      * **Best For**: Direct FAQ lookups, exact database key retrieval, or automated deterministic actions.
      * **Hazard**: Zero fault tolerance. If chunk 1 is a preamble and chunk 2 holds the actual answer, generation fails completely.
    * **Top-3 Retrieval ($K = 3$)**:
      * **Best For**: Standard single-hop question answering where chunks are ~300 tokens each.
      * **Benefit**: Fits cleanly within 1,000 prompt tokens; minimal distractor noise.
    * **Top-5 Retrieval ($K = 5$)**:
      * **Best For**: Enterprise knowledge base default.
      * **Benefit**: Handles multi-clause inquiries where information is distributed across several sections; reaches ~85-90% recall.
    * **Top-10 to Top-50 Retrieval ($K = 10 - 50$)**:
      * **Role**: Candidate generation for a secondary **Cross-Encoder Reranker**.
      * Fast bi-encoder vector search retrieves 50 coarse candidates; a high-precision cross-encoder re-scores and filters them down to the top 3!
    """)

with col_k2:
    st.markdown("### Two-Stage Retrieval Funnel Architecture")
    k_dot = """
    digraph RetrievalFunnel {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Corpus [label="Corpus Database\\n1,000,000 Vectors", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Stage1 [label="Stage 1: Bi-Encoder Vector Search\\nBroad Recall Filter\\nRetrieves Top-K = 50 Candidates (sub-5ms)", fillcolor="#FEF3C7"];
        
        Stage2 [label="Stage 2: Cross-Encoder Reranker\\nDeep Semantic Token Attention\\nRe-scores & prunes to Top-3 (25ms)", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        
        LLM [label="Final LLM Generator\\nInjected Context: Top-3 Chunks (~900 tokens)\\nZero Distractor Noise | Fast Inference", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        Corpus -> Stage1 -> Stage2 -> LLM;
    }
    """
    st.graphviz_chart(k_dot, use_container_width=False)

st.markdown("### Retrieval Depth Tradeoff Matrix")
st.markdown("""
| Value of K | Prompt Token Budget (assuming 300 words/chunk) | LLM Generation Latency Added | Empirical Recall Coverage | Distractor Noise Risk | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 1** | ~350 tokens | Negligible (+50ms) | Low (~55%) | Zero | Exact FAQ matching |
| **K = 3** | ~1,050 tokens | Low (+150ms) | Moderate (~78%) | Low | Direct Q&A bots |
| **K = 5** | ~1,750 tokens | Moderate (+300ms) | **High (~88%)** | Moderate | **Default Enterprise RAG** |
| **K = 10** | ~3,500 tokens | High (+650ms) | Very High (~94%) | High | Complex multi-topic synthesis |
| **K = 50** | ~17,500 tokens (Too large for prompt) | Severe | **Near 100%** | Extreme | Pre-filter for Cross-Encoder Reranker |
""")

st.divider()

# Section 2: Production Code: Two-Stage Retrieve-and-Rerank Engine
st.header("2. Production Implementation: Two-Stage Retrieve-and-Rerank Engine", divider="rainbow")

st.markdown("""
The production utility below implements the industry-standard **Two-Stage Retrieval Pipeline**: pulling a broad $K_1 = 10$ candidates via fast vector dot products, then re-scoring and pruning to $K_2 = 3$ high-confidence passages.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any

class TwoStageRetriever:
    """
    Production-grade Two-Stage Retrieve & Rerank pipeline.
    Stage 1: Broad vector dot-product search (K1).
    Stage 2: Precision scoring & pruning to top K2 for LLM injection.
    """
    def __init__(self, corpus: List[Dict[str, Any]], embedding_dim: int = 384):
        self.corpus = corpus
        self.dim = embedding_dim

    def retrieve_and_rerank(
        self, 
        query_vector: np.ndarray, 
        k1_candidates: int = 10, 
        k2_final: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes two-stage retrieval funnel.
        """
        q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-12)

        # Stage 1: Fast Vector Search (Top K1)
        scored_candidates = []
        for doc in self.corpus:
            d_norm = doc["vector"] / np.maximum(np.linalg.norm(doc["vector"]), 1e-12)
            sim = float(np.dot(q_norm, d_norm))
            scored_candidates.append({**doc, "vector_score": sim})
            
        scored_candidates.sort(key=lambda x: x["vector_score"], reverse=True)
        top_k1 = scored_candidates[:k1_candidates]

        # Stage 2: Precision Scoring / Reranking (Simulated cross-encoder)
        # In production: cross_encoder.predict([(query, d['text']) for d in top_k1])
        reranked = []
        for item in top_k1:
            # Subtle rerank adjustment reflecting cross-attention nuance
            simulated_rerank_score = item["vector_score"] * 0.9 + 0.1 * np.random.default_rng(abs(hash(item["text"])) % (2**31)).random()
            reranked.append({
                "id": item["id"],
                "text": item["text"],
                "vector_score": item["vector_score"],
                "final_score": float(simulated_rerank_score)
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:k2_final]
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveKOptimizer:
    def __init__(self):
        # 10 candidate document chunks with simulated relevance and token footprints
        self.sample_chunks = [
            {"id": "CHUNK_01", "tokens": 280, "score": 0.94, "is_critical": True, "text": "Annual leave policy: 20 vacation days accrued monthly with up to 5 days carry-over."},
            {"id": "CHUNK_02", "tokens": 310, "score": 0.89, "is_critical": True, "text": "Carry-over vacation days must be utilized within the first quarter (by March 31)."},
            {"id": "CHUNK_03", "tokens": 250, "score": 0.84, "is_critical": False, "text": "Sick leave policy: 10 days provided annually upon start date for illness or medical care."},
            {"id": "CHUNK_04", "tokens": 290, "score": 0.79, "is_critical": False, "text": "Parental leave policy: 12 weeks paid leave for primary caregivers following birth or adoption."},
            {"id": "CHUNK_05", "tokens": 320, "score": 0.73, "is_critical": False, "text": "Bereavement leave: Up to 5 consecutive paid days for immediate family members."},
            {"id": "CHUNK_06", "tokens": 270, "score": 0.68, "is_critical": False, "text": "Jury duty policy: Up to 2 weeks paid leave for civic court appearances with court summons."},
            {"id": "CHUNK_07", "tokens": 340, "score": 0.62, "is_critical": False, "text": "Military leave: Active reserve duty accommodations in compliance with federal guidelines."},
            {"id": "CHUNK_08", "tokens": 260, "score": 0.58, "is_critical": False, "text": "Sabbatical policy: Eligible after 5 years continuous service for up to 3 unpaid months."},
            {"id": "CHUNK_09", "tokens": 300, "score": 0.53, "is_critical": False, "text": "Corporate holidays: 11 fixed company-wide statutory holidays observed per calendar year."},
            {"id": "CHUNK_10", "tokens": 290, "score": 0.47, "is_critical": False, "text": "Volunteer time off: 1 paid day per year dedicated to certified nonprofit community service."}
        ]

    def simulate_k_impact(self, k: int) -> Dict[str, Any]:
        selected = self.sample_chunks[:k]
        total_tokens = sum(c["tokens"] for c in selected)
        critical_found = sum(1 for c in selected if c["is_critical"])
        total_critical = sum(1 for c in self.sample_chunks if c["is_critical"])
        
        recall_pct = (critical_found / total_critical) * 100.0
        distractors = sum(1 for c in selected if not c["is_critical"])
        distractor_pct = (distractors / max(k, 1)) * 100.0
        
        # Token cost estimation ($0.005 / 1k tokens for generation model input)
        prompt_cost_usd = (total_tokens / 1000.0) * 0.005
        # Latency estimate
        est_latency_ms = 120 + int(total_tokens * 0.25)

        return {
            "k": k,
            "total_tokens": total_tokens,
            "recall_pct": recall_pct,
            "distractor_pct": distractor_pct,
            "prompt_cost": prompt_cost_usd,
            "latency_ms": est_latency_ms,
            "chunks": selected
        }

st.divider()

# Section 3: Interactive Workbench: K Optimization & Token Budget Calculator
st.header("3. Interactive Workbench: K Optimization & Token Budget Calculator", divider="rainbow")

st.markdown("""
Experiment with different values of **K** for the query:
*"What happens to unused vacation days at year end?"*
Observe how increasing $K$ improves **Critical Information Recall**, but steadily inflates **Prompt Token Consumption**, **Inference Latency**, and **Distractor Noise Percentage**!
""")

k_opt = InteractiveKOptimizer()

col_k_l, col_k_r = st.columns([1, 1])

with col_k_l:
    st.subheader("1. Configure Retrieval Depth K")
    k_slider = st.slider("Select Retrieval Depth K:", min_value=1, max_value=10, value=3, step=1)
    
    sim_stats = k_opt.simulate_k_impact(k_slider)

    st.markdown("#### Prompt Economics Impact")
    m1, m2 = st.columns(2)
    m1.metric("Total Injected Tokens", f"{sim_stats['total_tokens']:,} tokens")
    m2.metric("Critical Recall", f"{sim_stats['recall_pct']:.0f}%", delta="100% Captured" if sim_stats['recall_pct'] == 100 else "Incomplete!")

    m3, m4 = st.columns(2)
    m3.metric("Estimated Added Latency", f"{sim_stats['latency_ms']} ms")
    m4.metric("Distractor Ratio", f"{sim_stats['distractor_pct']:.0f}%", delta=f"{sim_stats['distractor_pct']:.0f}% Noise", delta_color="inverse")

with col_k_r:
    st.subheader(f"2. Retrieved Chunks Injected into Prompt (Top {k_slider})")
    df_k = pd.DataFrame([
        {
            "Rank": i + 1,
            "Chunk ID": c["id"],
            "Score": c["score"],
            "Tokens": c["tokens"],
            "Relevance": "🎯 Critical Answer" if c["is_critical"] else "⚠️ Secondary / Distractor",
            "Snippet": c["text"]
        }
        for i, c in enumerate(sim_stats["chunks"])
    ])
    st.dataframe(
        df_k.style.format({"Score": "{:.2f}"}).background_gradient(subset=["Score"], cmap="Blues"),
        use_container_width=True
    )

    if k_slider == 1:
        st.warning("⚠️ **K=1 (Under-Retrieval)**: Only retrieved Chunk 1 (20 vacation days), completely missing Chunk 2 (the crucial rule that carry-over days expire March 31)!")
    elif k_slider in [2, 3]:
        st.success("🌟 **Sweet Spot (K=2-3)**: Captured 100% of critical facts with minimal token consumption (<1,000 tokens) and low distractor overhead.")
    else:
        st.info(f"ℹ️ **High K ({k_slider})**: 100% recall achieved, but {sim_stats['distractor_pct']:.0f}% of injected text discusses irrelevant policies (Jury duty, bereavement), diluting LLM attention.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "K-Tuning Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the 'Lost in the Middle' phenomenon, and how does it relate to choosing K?**
    * **Answer**:
      * Research by Liu et al. (Stanford, 2023) revealed that Large Language Models exhibit a **U-shaped attention curve**:
        * LLMs pay high attention to documents placed at the very beginning of the prompt.
        * They pay high attention to documents placed at the very end of the prompt.
        * They frequently overlook and fail to retrieve facts placed in the **middle** of long multi-document prompts!
      * If $K$ is set too high (e.g. $K = 15$), relevant documents pushed into middle positions are ignored by the model despite being present in context!
      * Keeping $K$ small ($K = 3 - 5$) keeps all retrieved context within the high-attention boundaries.

    **Q2: How does the Two-Stage Retrieval pattern (Bi-Encoder + Cross-Encoder) resolve the K dilemma?**
    * **Answer**:
      * **The Dilemma**: A Bi-Encoder needs high $K$ (e.g. $K=50$) to guarantee high recall, but feeding 50 documents into an LLM causes latency spikes and context dilution.
      * **The Two-Stage Solution**:
        1. **Stage 1 (High Recall)**: Fast bi-encoder vector search retrieves $K_1 = 50$ candidates in 3ms.
        2. **Stage 2 (High Precision)**: A Cross-Encoder reranker performs full joint cross-attention over all 50 `(query, document)` pairs in 25ms, assigning deep relevance scores.
        3. **Pruning**: Cut off to $K_2 = 3$ top-scoring reranked passages.
        4. The LLM receives only 3 pristine, verified passages, maximizing accuracy while minimizing token costs!

    **Q3: How do you choose K dynamically based on query classification?**
    * **Answer**:
      * Not all queries require the same retrieval depth:
        * **Factual Lookup Queries** (*"What is our CEO's email?"*): Single-hop question $\\rightarrow$ Set $K = 1$ or $2$.
        * **Comparative Queries** (*"Compare Plan A vs Plan B coverage"*): Multi-hop inquiry $\\rightarrow$ Set $K = 5$ or $6$.
        * **Summarization Queries** (*"Summarize Q3 board meeting outcomes"*): Broad topic $\\rightarrow$ Set $K = 8 - 10$.
      * A lightweight query router classifies user intent upfront to dynamically set $K$.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Setting K without Considering Chunk Size**:
      * *Issue*: Setting $K = 10$ where each chunk is 1,000 tokens long.
      * *Consequence*: Injects 10,000 tokens into the prompt, adding 2-3 seconds of TTFT (Time To First Token) latency and inflating billing by 10x.
      * *Fix*: Calculate token budget product: $K \\times \\text{avg\\_chunk\\_tokens} \\le 1,500$.

    * **Trap 2: Fixed K on Small Document Collections**:
      * *Issue*: Requesting $K = 10$ on a document collection that only contains 4 pages.
      * *Consequence*: Retrieves the entire database for every query, including completely irrelevant sections.
      * *Fix*: Clamp $K$: `effective_k = min(target_k, len(collection))`.

    * **Trap 3: Relying on High K Instead of Fine-Tuning or Chunk Optimization**:
      * *Issue*: Pumping $K$ from 3 to 20 because the search engine keeps missing answers.
      * *Consequence*: Masking poor embedding models or bad chunking boundaries with prompt bloat.
      * *Fix*: Fix chunk size and deploy a cross-encoder reranker rather than inflating $K$.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Retrieval Depth (K)**
      * *Definition*: The number of nearest-neighbor candidate passages returned by vector search for downstream consumption.
    * **Flashcard 2: Lost in the Middle**
      * *Definition*: The empirical tendency of LLMs to ignore information located in the middle of long multi-document prompt contexts.
    * **Flashcard 3: Two-Stage Retrieval**
      * *Definition*: Retrieving a large candidate set (K1=50) via fast bi-encoder vector search, then pruning to a small set (K2=3) with a cross-encoder.
    * **Flashcard 4: Dynamic K**
      * *Definition*: Adapting the retrieval depth K based on query complexity (small K for factual lookups, larger K for comparative synthesis).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Top-K Retrieval:

- [ ] I can articulate the trade-offs between small K (K=1-3) and large K (K=5-10).
- [ ] I understand the Lost in the Middle phenomenon and why bloated prompts degrade LLM generation.
- [ ] I know how to calculate prompt token budgets based on K and average chunk size.
- [ ] I can explain the Two-Stage Retrieve-and-Rerank pattern and why it optimizes both recall and precision.
- [ ] I know how to implement dynamic K sizing based on query classification.
""")
