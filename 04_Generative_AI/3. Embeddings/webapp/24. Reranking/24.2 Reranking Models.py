import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="24.2 Reranking Models - Embeddings Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("24.2 Reranking Models: Cross-Encoders, Late-Interaction (ColBERT), & LLM Reranking (RankGPT)")
st.caption("Chapter 24: Reranking | Domain: 3. Vector Embeddings")
st.markdown("""
Once your Stage 1 retrieval engine extracts 30 to 100 candidate chunks, the next critical architectural decision is **selecting the right reranking model family**.
Reranking is not a one-size-fits-all problem; different production environments require different balances of latency, inference cost, and reasoning depth:

* **Dedicated Cross-Encoders (BGE-Reranker, Cohere Rerank v3)**:
  * Compact BERT-style encoders trained with binary classification heads on query-passage pairs.
  * Extremely fast (15-30ms) and cost-effective.
* **Late-Interaction Models (ColBERT / ColBERTv2)**:
  * Preserves multi-vector token embeddings and computes **MaxSim token alignment**.
  * Delivers near-Cross-Encoder accuracy at 10x faster inference speed!
* **LLM-Based Rerankers (RankGPT / Listwise Prompting)**:
  * Prompts generative instruction models (GPT-4o-mini, Llama 3) to rank candidate passages listwise.
  * Highest reasoning capability, but higher latency (500-1500ms) and API token costs.

This lesson explores the inner mechanics, attention architectures, and engineering trade-offs across these three modern reranking paradigms.
""")

st.divider()

# Section 1: Taxonomy of Modern Reranking Architectures
st.header("1. The Three Modern Reranking Paradigms", divider="rainbow")

col_par1, col_par2 = col_par1, col_par2 = st.columns([1, 1])

with col_par1:
    st.markdown("""
    ### The 3 Core Architectural Paradigms
    * **1. Dedicated Cross-Encoder (Pairwise Regression)**:
      * Concatenates query and passage: `[CLS] Q [SEP] D [SEP]`.
      * Cross-attention computes interaction across all tokens in all layers.
      * Final `[CLS]` token is mapped through a linear projection layer with Sigmoid to yield a calibrated relevance score in `[0.0, 1.0]`.
      * *Examples*: `bge-reranker-large`, `ms-marco-MiniLM-L-6-v2`, `Cohere Rerank v3`.
    * **2. Late-Interaction Architecture (ColBERT)**:
      * Retains individual token vectors for query and document.
      * Computes a **MaxSim Operator**: for every query token, finds the maximum cosine similarity among all document tokens, then sums them up!
      * Eliminates full transformer cross-attention overhead while preserving fine-grained token matching.
    * **3. Generative LLM-as-a-Reranker (RankGPT)**:
      * Injects 20 candidate passages into a generative LLM prompt.
      * Prompts the model to output a ranked permutation list (`[4] > [1] > [2] > ...`).
      * Excels at complex, multi-hop reasoning questions where passage synthesis is required.
    """)

with col_par2:
    st.markdown("### Architectural Attention Comparison")
    attn_dot = """
    digraph RerankerArchitectures {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_cross {
            label="1. Cross-Encoder (Full Cross-Attention)"; style="dashed"; color="#10B981";
            CE_In [label="Concatenated: [CLS] Query [SEP] Doc", fillcolor="#D1FAE5"];
            CE_Attn [label="All-to-All Token Cross-Attention\\n(12 Transformer Layers)", fillcolor="#A7F3D0"];
            CE_Out [label="Sigmoid Score: 0.94", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
            CE_In -> CE_Attn -> CE_Out;
        }

        subgraph cluster_colbert {
            label="2. ColBERT (Late Interaction MaxSim)"; style="dashed"; color="#3B82F6";
            CB_In [label="Multi-Vector Token Matrices", fillcolor="#DBEAFE"];
            CB_Max [label="MaxSim Operator (Fast CPU SIMD)\\nSum(Max(Q_i . D_j))", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
            CB_In -> CB_Max;
        }

        subgraph cluster_llm {
            label="3. RankGPT (Generative Listwise)"; style="dashed"; color="#8B5CF6";
            LLM_In [label="Prompt: 'Rank the following 10 passages...'", fillcolor="#EDE9FE"];
            LLM_Out [label="Output Permutation: [3] > [1] > [5]", fillcolor="#C4B5FD", color="#7C3AED", style="filled,bold"];
            LLM_In -> LLM_Out;
        }
    }
    """
    st.graphviz_chart(attn_dot, use_container_width=False)

st.markdown("### Reranking Paradigm Engineering Matrix")
st.markdown("""
| Model Family | Canonical Implementations | Typical Latency (50 docs) | Inference Cost | Reasoning Capability | Best Production Fit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dedicated Cross-Encoder** | `bge-reranker-large`, `Cohere Rerank v3` | **15 - 30 ms** | Low ($0.002 / 1k queries) | High (Fine-grained syntax) | **Default Enterprise RAG Choice.** |
| **Late Interaction** | `ColBERTv2`, `RAGatouille`, `Jina-ColBERT` | **~5 - 12 ms** | Ultra-Low (Local CPU/GPU) | High (Token-level matching) | Real-time high-throughput search (> 500 QPS). |
| **Generative LLM (RankGPT)** | `GPT-4o-mini`, `Llama 3 8B` (Permutation) | 400 - 1200 ms | High ($0.05 / 1k queries) | **Extraordinary (Deep logic)** | High-stakes legal/medical research where latency is secondary. |
""")

st.divider()

# Section 2: Production Code: Multi-Reranker Simulation Engine
st.header("2. Production Code: Multi-Reranker Simulation Engine", divider="rainbow")

st.markdown("""
The production utility below implements simulated Cross-Encoder scoring, ColBERT MaxSim operator token alignment, and LLM permutation parsing.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class RerankingModelSuite:
    """
    Simulates Cross-Encoder, ColBERT Late-Interaction,
    and RankGPT Listwise Reranking implementations.
    """
    @staticmethod
    def cross_encoder_score(query: str, doc_text: str) -> float:
        """Simulates full cross-attention score [0.0, 1.0]."""
        q_tokens = set(query.lower().split())
        d_tokens = set(doc_text.lower().split())
        
        # Word overlap + bigram continuity check
        overlap = len(q_tokens.intersection(d_tokens)) / max(1, len(q_tokens))
        bigrams = [f"{w1} {w2}" for w1, w2 in zip(query.lower().split()[:-1], query.lower().split()[1:])]
        bigram_hit = sum(1 for bg in bigrams if bg in doc_text.lower()) / max(1, len(bigrams)) if bigrams else 0.0

        raw_score = 0.4 * overlap + 0.6 * bigram_hit
        return min(0.99, max(0.01, raw_score))

    @staticmethod
    def colbert_maxsim_score(query: str, doc_text: str, dim: int = 32) -> float:
        """
        Simulates ColBERT MaxSim operator:
        For each query token vector, finds max dot product with doc token vectors.
        """
        q_words = query.lower().split()
        d_words = doc_text.lower().split()
        
        # Mock token embeddings
        q_vecs = [np.random.randn(dim) for _ in q_words]
        d_vecs = [np.random.randn(dim) for _ in d_words]
        q_vecs = [v / np.linalg.norm(v) for v in q_vecs]
        d_vecs = [v / np.linalg.norm(v) for v in d_vecs]

        # MaxSim calculation
        total_sim = 0.0
        for q_v in q_vecs:
            max_sim = max([float(np.dot(q_v, d_v)) for d_v in d_vecs], default=0.0)
            total_sim += max_sim

        normalized_score = total_sim / max(1, len(q_words))
        return (normalized_score + 1.0) / 2.0

    @staticmethod
    def rankgpt_listwise_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
        """Formats candidates into a listwise permutation prompt for an LLM."""
        items_str = "\n".join([f"[{i+1}] {c['text'][:120]}..." for i, c in enumerate(candidates)])
        prompt = f"""You are RankGPT. Rank the following candidate passages from most relevant to least relevant for the query:
QUERY: {query}

CANDIDATES:
{items_str}

Respond with only the ranked identifier list, for example: [3] > [1] > [4] > [2]"""
        return prompt
''', language="python")

st.divider()

# Section 3: Interactive Reranker Benchmark Studio
st.header("3. Interactive Workbench: Reranking Paradigm Benchmark Arena", divider="rainbow")

st.markdown("""
Compare how a **Cross-Encoder**, **ColBERT**, and **RankGPT** re-rank candidate documents for a complex technical question.
""")

test_candidates = [
    {"id": "doc_01", "text": "PostgreSQL streaming replication uses Write-Ahead Logs (WAL) to synchronize standby replicas asynchronously.", "label": "Partial Match"},
    {"id": "doc_02", "text": "To resolve connection pool exhaustion in PostgreSQL, deploy PgBouncer in transaction pooling mode.", "label": "EXACT TARGET ANSWER"},
    {"id": "doc_03", "text": "Max_connections in postgresql.conf sets the upper limit for concurrent client connections.", "label": "Supporting Fact"},
    {"id": "doc_04", "text": "MySQL connection pooling with ProxySQL provides query routing and read-write split capabilities.", "label": "Distractor (MySQL)"},
    {"id": "doc_05", "text": "Redis connection pools in Python can be managed using the redis-py ConnectionPool class.", "label": "Distractor (Redis)"}
]

col_m_in, col_m_res = st.columns([1, 1])

with col_m_in:
    st.subheader("1. Test Query & Candidates")
    user_q = st.text_input(
        "Technical Troubleshooting Query:",
        value="How do I fix PostgreSQL connection pool exhaustion errors?"
    )

    st.subheader("2. Select Reranking Model")
    selected_model = st.radio(
        "Reranking Architecture:",
        ["1. Dedicated Cross-Encoder (bge-reranker-large / Cohere)", "2. Late-Interaction ColBERT (ColBERTv2)", "3. Generative LLM RankGPT (Listwise Permutation)"],
        index=0
    )

with col_m_res:
    st.subheader(f"3. Reranker Execution Telemetry")

    if "1. Dedicated Cross-Encoder" in selected_model:
        lat = "22 ms"
        cost = "$0.002 / 1k queries"
        throughput = "45 req/sec"
        # Scores
        scored_docs = [
            (test_candidates[1], 0.96, "Exact causal match for connection exhaustion via PgBouncer."),
            (test_candidates[2], 0.74, "Relevant parameter configuration (max_connections)."),
            (test_candidates[0], 0.45, "Postgres topic but discusses replication WALs."),
            (test_candidates[3], 0.12, "Distractor: describes MySQL ProxySQL."),
            (test_candidates[4], 0.04, "Distractor: describes Redis.")
        ]
        st.info("ℹ️ **Cross-Encoder Model**: Full cross-attention between every query and passage token.")

    elif "2. Late-Interaction ColBERT" in selected_model:
        lat = "7 ms (Fastest!)"
        cost = "$0.000 (Local CPU SIMD)"
        throughput = "140 req/sec"
        scored_docs = [
            (test_candidates[1], 0.93, "High MaxSim token alignment on 'PgBouncer' and 'exhaustion'."),
            (test_candidates[2], 0.76, "Strong token overlap with 'connections' and 'PostgreSQL'."),
            (test_candidates[0], 0.42, "Moderate overlap on 'PostgreSQL'."),
            (test_candidates[3], 0.18, "Matches 'connection pool' but penalizes 'MySQL'."),
            (test_candidates[4], 0.08, "Penalizes 'Redis'.")
        ]
        st.info("ℹ️ **ColBERT Model**: Fast token-level MaxSim dot products without transformer cross-attention.")

    else: # RankGPT
        lat = "750 ms (Slowest)"
        cost = "$0.045 / 1k queries"
        throughput = "1.3 req/sec"
        scored_docs = [
            (test_candidates[1], 0.99, "LLM Reasoning: Direct solution to connection pool exhaustion."),
            (test_candidates[2], 0.80, "LLM Reasoning: Key background parameter for connection limits."),
            (test_candidates[0], 0.35, "LLM Reasoning: Unrelated replication topic."),
            (test_candidates[3], 0.05, "LLM Reasoning: Incorrect database technology (MySQL)."),
            (test_candidates[4], 0.01, "LLM Reasoning: Incorrect database technology (Redis).")
        ]
        st.info("ℹ️ **RankGPT Model**: Listwise prompt reasoning over all 5 passages simultaneously.")

    m1, m2, m3 = st.columns(3)
    m1.metric("P95 Latency", lat)
    m2.metric("Inference Cost", cost)
    m3.metric("Throughput", throughput)

    st.markdown("#### Re-Ranked Candidate List")
    for rank, (doc, sc, explanation) in enumerate(scored_docs, 1):
        with st.expander(f"Rank #{rank} | {doc['id']} | Score: {sc:.2f} ({doc['label']})", expanded=(rank <= 2)):
            st.markdown(f"> {doc['text']}")
            st.caption(f"**Model Diagnostic**: {explanation}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Reranker Model Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does ColBERT's 'Late Interaction' MaxSim operator achieve Cross-Encoder quality at Bi-Encoder speeds?**
    * **Answer**:
      * In a Bi-Encoder, an entire passage is collapsed into a single vector.
      * In a Cross-Encoder, query and passage tokens are cross-attended across all layers (slow).
      * **ColBERT keeps multi-vector token representations**: a 10-token query has 10 vectors; a 100-token passage has 100 vectors.
      * At query time, it evaluates the **MaxSim Operator**: for each query token vector $q_i$, it finds the single maximum dot product across all document vectors $d_j$: `sum_i max_j (q_i . d_j)`.
      * Because MaxSim is a simple matrix dot product, it runs via fast CPU SIMD instructions in $< 10\\text{ms}$ while preserving token-level syntactic alignment!

    **Q2: What is the 'Listwise Prompting' technique used by RankGPT?**
    * **Answer**: Rather than scoring passages one by one (pointwise) or in pairs (pairwise), RankGPT passes a list of candidate passages (e.g. 10 to 20) in a single LLM prompt.
    The LLM reasons across all candidates simultaneously, observing differences and cross-referencing claims before outputting an ordered permutation string (`[3] > [1] > [4] > ...`).
    For large candidate pools (e.g. 50 items), RankGPT uses a **Sliding Window** (e.g. window size 10, step size 5) scanning backward from bottom to top to bubbled-up top candidates.

    **Q3: When should an enterprise deploy Cohere Rerank API versus hosting an open-source Cross-Encoder (e.g. `bge-reranker-large`)?**
    * **Answer**:
      * **Deploy Cohere Rerank API when**: You want zero GPU infrastructure to manage, support for 100+ multilingual languages out of the box, and instant scaling.
      * **Self-Host `bge-reranker-large` (Local) when**:
        1. Strict data privacy (HIPAA / GDPR / on-premise air-gapped environments) prohibits sending data to external APIs.
        2. High query volume (> 1,000 QPS) makes commercial reranker API billing cost-prohibitive.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Using RankGPT for Real-Time Autocomplete Search**:
      * *Issue*: Deploying an LLM-based reranker with 800ms latency on interactive user search bars.
      * *Consequence*: Sluggish UI lag and massive OpenAI token bills.
      * *Fix*: Use lightweight Cross-Encoders (15ms) or ColBERT (5ms) for interactive search; reserve RankGPT for offline workflows.

    * **Trap 2: Ignoring Max Sequence Length Truncation in Rerankers**:
      * *Issue*: Passing a 1,000-word document into a Cross-Encoder with a 512-token context window.
      * *Consequence*: The model silently truncates the second half of the document, failing to cross-attend to facts at the bottom.
      * *Fix*: Always verify `max_length` and only pass concise chunks to rerankers.

    * **Trap 3: Reranking Unnormalized Candidates from Incompatible Retrievers**:
      * *Issue*: Manually blending Cross-Encoder scores with Stage 1 vector scores without calibration.
      * *Consequence*: Distorted final rankings.
      * *Fix*: Use the Cross-Encoder score strictly as the final ground-truth ranking authority.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Dedicated Cross-Encoder**
      * *Definition*: A transformer classification model that evaluates full cross-attention across concatenated query-document token pairs.
    * **Flashcard 2: ColBERT MaxSim Operator**
      * *Definition*: The late-interaction formula `sum_i max_j (q_i . d_j)` finding the highest similarity token match for each query token.
    * **Flashcard 3: RankGPT Listwise Reranking**
      * *Definition*: Prompting a generative LLM to evaluate multiple candidate passages simultaneously and return an optimal permutation order.
    * **Flashcard 4: Pointwise vs. Listwise Reranking**
      * *Definition*: Scoring documents individually in isolation (pointwise) vs. ranking documents jointly in context of the entire candidate pool (listwise).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Reranking Models:

- [ ] I can articulate the differences between Dedicated Cross-Encoders, ColBERT, and RankGPT.
- [ ] I understand how ColBERT's MaxSim operator achieves near-Cross-Encoder accuracy at 10x speed.
- [ ] I know how RankGPT performs listwise permutation ranking using sliding windows.
- [ ] I can select the optimal reranking model based on latency, cost, and throughput requirements.
- [ ] I understand why Cross-Encoder scores should serve as the final ranking authority.
""")
