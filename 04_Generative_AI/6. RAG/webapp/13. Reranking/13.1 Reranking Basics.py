import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class TwoStageRerankingEngine:
    """
    Simulates a production two-stage retrieve-and-rerank pipeline:
    - Stage 1: Fast Bi-Encoder approximate nearest neighbor retrieval (Top 10 candidates).
    - Stage 2: High-precision Cross-Encoder joint attention reranking (Top 3 finalists).
    Demonstrates how cross-encoders resolve bi-encoder blind spots like negation and fine-grained entity overlap.
    """
    CORPUS = [
        {
            "id": "doc_med_01",
            "title": "Clinical Trial Alpha - Drug X Adverse Events",
            "text": "Drug X is FDA-approved for severe bronchial asthma and is known to cause cardiac arrhythmia in 14% of patients.",
            "bi_encoder_score": 0.89,
            "cross_encoder_score": 0.12,
            "category": "Medical Safety"
        },
        {
            "id": "doc_med_02",
            "title": "Clinical Trial Beta - Drug Y Safety Profile",
            "text": "Drug Y is FDA-approved for bronchial asthma and demonstrates zero cardiac side effects or rhythm anomalies in clinical trials.",
            "bi_encoder_score": 0.83,
            "cross_encoder_score": 0.96,
            "category": "Medical Safety"
        },
        {
            "id": "doc_db_01",
            "title": "PostgreSQL Connection Pooling with PgBouncer",
            "text": "Transaction pooling mode in PgBouncer reuses server connections as soon as a transaction ends, maximizing connection throughput.",
            "bi_encoder_score": 0.88,
            "cross_encoder_score": 0.94,
            "category": "Database Architecture"
        },
        {
            "id": "doc_db_02",
            "title": "PostgreSQL Session Pooling Limitations",
            "text": "Session pooling maintains server connections for the entire client session lifetime, limiting max concurrent clients.",
            "bi_encoder_score": 0.85,
            "cross_encoder_score": 0.72,
            "category": "Database Architecture"
        },
        {
            "id": "doc_db_03",
            "title": "General Relational Database Intro",
            "text": "Relational databases use SQL to query structured data tables with foreign keys and ACID transactions.",
            "bi_encoder_score": 0.71,
            "cross_encoder_score": 0.18,
            "category": "Database Architecture"
        }
    ]

    PRESET_QUERIES = {
        "Negation Challenge (Medical Safety)": {
            "query": "Asthma treatments that do NOT cause cardiac arrhythmia",
            "doc_ids": ["doc_med_01", "doc_med_02"],
            "explanation": "Bi-encoder ranks Doc 1 #1 due to massive topical token overlap, ignoring the negation. Cross-encoder catches the negation 'do NOT cause', flipping Doc 2 to #1!"
        },
        "Fine-Grained Technical Specificity": {
            "query": "PgBouncer transaction pooling connection reuse mechanism",
            "doc_ids": ["doc_db_01", "doc_db_02", "doc_db_03"],
            "explanation": "Bi-encoder retrieves all three database docs with similar scores. Cross-encoder sharply distinguishes transaction pooling from session pooling."
        }
    }

    @classmethod
    def simulate_pipeline(
        cls,
        preset_key: str,
        stage1_cutoff: float = 0.70,
        stage2_top_k: int = 2
    ) -> Dict[str, Any]:
        preset = cls.PRESET_QUERIES[preset_key]
        target_docs = [d for d in cls.CORPUS if d["id"] in preset["doc_ids"]]

        # Stage 1: Bi-encoder ranking
        stage1_ranked = sorted(target_docs, key=lambda x: x["bi_encoder_score"], reverse=True)
        stage1_filtered = [d for d in stage1_ranked if d["bi_encoder_score"] >= stage1_cutoff]

        # Stage 2: Cross-encoder reranking
        stage2_reranked = sorted(stage1_filtered, key=lambda x: x["cross_encoder_score"], reverse=True)[:stage2_top_k]

        return {
            "query": preset["query"],
            "explanation": preset["explanation"],
            "stage1_results": stage1_filtered,
            "stage2_results": stage2_reranked
        }

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="13.1 Reranking Basics - RAG Master Curriculum",
    page_icon="⚖️",
    layout="wide"
)

st.title("13.1 Reranking Basics: Two-Stage Retrieval, Cross-Encoders, & Relevance Scoring")
st.caption("Chapter 13: Reranking | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In large-scale RAG systems, retrieval speed and retrieval precision are in constant tension:
* **First-Stage Retrievers (Bi-Encoders / HNSW)** are ultra-fast: they search across millions of documents 
  in under 10 milliseconds, but operate with coarse semantic granularity. Because query and document are encoded independently, 
  bi-encoders frequently suffer from **negation blindness** and subtle context confusion.
* **Second-Stage Rerankers (Cross-Encoders)** compute full joint cross-attention between every query token 
  and every document token simultaneously. They possess extraordinary scoring precision, but are computationally 
  too expensive to run against an entire database.

The industry-standard solution is the **Two-Stage Retrieve-and-Rerank Architecture**:
1. An inexpensive bi-encoder retrieves a candidate pool of **Top 20–50 coarse candidates**.
2. A high-precision cross-encoder reranker scores the candidate pairs, reordering them to output the definitive **Top 3–5 context chunks** for the LLM.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Two-Stage Retrieve-and-Rerank Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    rerank_bp_dot = """
    digraph TwoStageRerankBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_stage1 {
            label = "1. Stage 1: Coarse High-Recall Retrieval Tier (Fast & Broad)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="User Natural Language Query\\ne.g. 'Asthma drugs that do NOT cause arrhythmia'", fillcolor="#0F172A", color="#38BDF8"];
            vdb [label="Vector Store / Hybrid Index\\nMillions of Indexed Document Chunks", fillcolor="#0F172A", color="#06B6D4"];
            stage1_candidates [label="Top 25 Coarse Candidates\\nBi-Encoder Cosine Proximity | Latency < 15ms", fillcolor="#334155", color="#F59E0B"];

            user_query -> vdb [label="ANN Inner Product Walk"];
            vdb -> stage1_candidates [label="Over-Fetch K=25"];
        }

        subgraph cluster_stage2 {
            label = "2. Stage 2: Fine High-Precision Reranking Tier (Deep Joint Attention)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            cross_encoder [label="Cross-Encoder Model (e.g. BGE-Reranker-Large, Cohere Rerank v3)\\nFull All-to-All Token Cross-Attention Layer", fillcolor="#334155", color="#EC4899"];
            relevance_scores [label="Calibrated Relevance Logits\\nTrue Relevance Scores [0.0 - 1.0]", fillcolor="#0F172A", color="#10B981"];
            cutoff_gate [label="Relevance Threshold Gate\\nDiscards Chunks with Score < 0.70", fillcolor="#334155", color="#F59E0B"];

            user_query -> cross_encoder [label="Concat: [CLS] Query [SEP] Doc [SEP]"];
            stage1_candidates -> cross_encoder [label="25 Document Candidates"];
            cross_encoder -> relevance_scores [label="Compute Joint Matrix"];
            relevance_scores -> cutoff_gate [label="Filter Low Scores"];
        }

        subgraph cluster_downstream {
            label = "3. Downstream LLM Generation Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            final_topk [label="Top 3–5 High-Precision Finalists\\nZero Distractor Noise | Exact Context", fillcolor="#0F172A", color="#10B981"];
            llm_gen [label="LLM Synthesis & Citation Generator\\nGrounded Answer Free of Hallucinations", fillcolor="#334155", color="#38BDF8"];

            cutoff_gate -> final_topk [label="Hydrate Top 3"];
            final_topk -> llm_gen [label="Prompt Context Injection"];
        }
    }
    """
    st.graphviz_chart(rerank_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG architectures, single-stage retrieval is considered an anti-pattern for mission-critical tasks. 
Every production query executes a two-stage cascade: an initial fast vector search fetches 20 to 50 coarse candidates, 
which are immediately streamed to a cross-encoder model (such as **BGE-Reranker-v2-m3**, **Cohere Rerank v3**, 
or a locally hosted **Cross-Encoder/ms-marco-MiniLM-L-6-v2**). 
The reranker scores all 25 candidates in under 35 milliseconds, sorting them by true semantic relevance and 
discarding false positives before prompt assembly.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHAT IS RERANKING?
# ==============================================================================

st.header("2. What is Reranking? The Concept, Architecture, and Role in Modern RAG", divider="rainbow")

st.markdown("""
**Reranking** is the process of re-evaluating and re-ordering a subset of retrieved document candidates 
using a more expressive, computationally intensive scoring function than was feasible during initial search.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Fundamental Trade-Off in Information Retrieval")
    st.markdown("""
    In computer science, information retrieval algorithms always trade off between **Search Space Scalability** 
    and **Scoring Expressiveness**:
    * **High Scalability, Coarse Scoring**:
      * *Bi-Encoders & Inverted Indices*: Encode documents into static vectors offline. 
        Enables searching over 100 million documents in 5 milliseconds via HNSW graphs. 
        However, the model cannot see the query and document together at encoding time.
    * **Low Scalability, Deep Scoring**:
      * *Cross-Encoders*: Concat query and document into a single sequence, passing them together through all Transformer layers. 
        Computes all-to-all cross-attention between every query word and every document word. 
        Provides nearly flawless relevance scoring, but cannot scale to searching millions of documents.
    * **Reranking bridges the gap**: Use the fast retriever to prune 1,000,000 documents down to 25, 
      then use the deep reranker to re-order those 25 documents!
    """)

with col2:
    st.subheader("Bi-Encoder vs. Cross-Encoder Contrast")
    contrast_df = pd.DataFrame([
        {
            "Architectural Metric": "Encoding Mechanism",
            "Bi-Encoder (Retriever)": "Dual independent towers: f(Q) and f(D)",
            "Cross-Encoder (Reranker)": "Joint single sequence: f([CLS] Q [SEP] D)"
        },
        {
            "Architectural Metric": "Cross-Attention",
            "Bi-Encoder (Retriever)": "Zero (Dot product between pooled vectors)",
            "Cross-Encoder (Reranker)": "Full all-to-all token cross-attention"
        },
        {
            "Architectural Metric": "Offline Pre-Computation",
            "Bi-Encoder (Retriever)": "Yes (All document vectors pre-indexed)",
            "Cross-Encoder (Reranker)": "No (Requires real-time forward pass per pair)"
        },
        {
            "Architectural Metric": "Search Scalability",
            "Bi-Encoder (Retriever)": "Billion-scale (O(log N) ANN graph)",
            "Cross-Encoder (Reranker)": "Candidate window only (O(K) where K <= 50)"
        },
        {
            "Architectural Metric": "Scoring Accuracy",
            "Bi-Encoder (Retriever)": "Moderate (Blind to subtle negation)",
            "Cross-Encoder (Reranker)": "State-of-the-art (Captures fine nuances)"
        }
    ])
    st.dataframe(contrast_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: RETRIEVER VS RERANKER
# ==============================================================================

st.header("3. Retriever vs. Reranker: Algorithmic, Computational, and Operational Differences", divider="rainbow")

st.markdown("""
Understanding the complementary operational roles of Retrievers and Rerankers prevents architectural misconfigurations.
""")

col_rv1, col_rv2 = st.columns([1, 1])

with col_rv1:
    st.subheader("Retriever: The High-Recall Scout")
    st.markdown("""
    * **Primary Responsibility**:
      * Maximize **Recall**: Ensure that the true relevant answer is captured somewhere within the top 25–50 candidates.
    * **Tolerates False Positives**:
      * It is acceptable if documents at rank 4, 7, and 12 are completely irrelevant distractor text, 
        provided the golden document is present at rank 9.
    * **Infrastructure**:
      * Vector databases (Qdrant, Pinecone, Milvus), Elasticsearch inverted indices.
    * **Cost**:
      * Inexpensive vector dot product operations on CPU memory.
    """)

with col_rv2:
    st.subheader("Reranker: The High-Precision Filter")
    st.markdown("""
    * **Primary Responsibility**:
      * Maximize **Precision at Rank 1–3**: Push the golden document from rank 9 up to rank 1.
    * **Rejects Distractors**:
      * Identifies misleading documents that fooled the bi-encoder due to superficial keyword overlap, 
        penalizing their score down to near zero.
    * **Infrastructure**:
      * Dedicated GPU/CPU microservices running cross-encoder models, or hosted API endpoints (Cohere Rerank).
    * **Cost**:
      * Higher compute per document pair, but bounded by small candidate window `K`.
    """)

st.divider()

# ==============================================================================
# SECTION 4: WHY RERANKING IMPROVES RAG
# ==============================================================================

st.header("4. Why Reranking Improves RAG: Overcoming Negation Blindness and Lost-in-the-Middle", divider="rainbow")

st.markdown("""
Deploying a reranker is often the single most cost-effective performance upgrade for enterprise RAG systems.
""")

col_why1, col_why2 = st.columns([1, 1])

with col_why1:
    st.subheader("1. Eliminating Negation Blindness")
    st.markdown("""
    * Bi-encoders compute dot products on pooled sentence representations. 
    * Sentences that negate a concept share almost all the same words:
      * *Sentence A*: *"Drug X is safe and approved for pediatric patients."*
      * *Sentence B*: *"Drug X is NOT safe and NOT approved for pediatric patients."*
    * Bi-encoders assign both sentences cosine similarities > 0.88 to the query *"Is Drug X safe for pediatric patients?"*.
    * A Cross-Encoder computes full self-attention between the word *"NOT"* and the query words *"safe"* and *"approved"*, 
      immediately recognizing the contradiction and scoring Sentence B appropriately.
    """)

with col_why2:
    st.subheader("2. Solving the Lost-in-the-Middle Effect")
    st.markdown("""
    * Research (Liu et al., Stanford) proves that LLMs suffer from **attention degradation**: 
      models pay heavy attention to the beginning and end of long prompts, but routinely ignore facts located in the middle.
    * If a bi-encoder places the true relevant document at rank #8 out of 10, the LLM will hallucinate even though 
      the answer was technically provided in the prompt!
    * Reranking promotes the golden document to rank #1 or #2, placing it right at the top of the prompt where 
      LLM attention heads are sharpest.
    """)

st.divider()

# ==============================================================================
# SECTION 5: CANDIDATE RETRIEVAL
# ==============================================================================

st.header("5. Candidate Retrieval: Setting the Optimal First-Stage Over-Fetch Window", divider="rainbow")

st.markdown("""
How many documents should the first-stage retriever fetch before handing off to the reranker?
""")

col_cand1, col_cand2 = st.columns([1, 1])

with col_cand1:
    st.subheader("The Over-Fetching Window Trade-Off")
    st.markdown("""
    * **Candidate Window Too Small (K = 5)**:
      * If the retriever only fetches 5 candidates, the reranker can only reorder those 5. 
        If the true relevant document was at rank #7 in the vector database, the reranker never sees it!
    * **Candidate Window Too Large (K = 200)**:
      * Scoring 200 documents with a cross-encoder requires 200 forward passes through a deep Transformer. 
        Latency explodes to 500+ milliseconds, blowing past search SLAs.
    * **The Golden Zone (K = 20 to 50)**:
      * Empirical benchmarks demonstrate that fetching **20 to 35 candidates** captures 95% of all potential recall gains 
        while keeping cross-encoder inference latency strictly under 30 milliseconds on modern hardware.
    """)

with col_cand2:
    st.subheader("Candidate Window Benchmark Guidelines")
    cand_df = pd.DataFrame([
        {"Window Size (K)", "Recall Capture Rate", "Cross-Encoder Latency", "Production Verdict"},
        {"K = 5", "68.2%", "8 ms", "Too small; misses consensus candidates"},
        {"K = 15", "84.5%", "18 ms", "Acceptable for ultra-low latency SLAs (<50ms)"},
        {"K = 25 - 35", "96.4%", "32 ms", "Optimal production standard for enterprise RAG"},
        {"K = 100", "98.1%", "145 ms", "Diminishing returns; excessive latency"},
        {"K = 250", "98.8%", "380 ms", "Prohibitive compute overhead"}
    ])
    st.dataframe(cand_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: RELEVANCE SCORING
# ==============================================================================

st.header("6. Relevance Scoring: Logits, Sigmoid Normalization, and Cutoff Calibration", divider="rainbow")

st.markdown("""
Cross-encoders output uncalibrated classification logits. Understanding how to normalize and threshold 
these scores is vital for applying quality gates.
""")

col_sc1, col_sc2 = st.columns([1, 1])

with col_sc1:
    st.subheader("Logits to Calibrated Probabilities")
    st.markdown("""
    * **Raw Cross-Encoder Output**:
      * The final layer of a cross-encoder is a binary classification head outputting a scalar logit `z` 
        representing the query-document relevance log-odds.
      * Logits can be negative or positive (e.g., `-6.4` to `+8.2`).
    * **Sigmoid Transformation**:
      * To convert logits into an intuitive probability scale between `0.0` and `1.0`:
        `relevance_probability = 1.0 / (1.0 + exp(-z))`
    * **Unlike Cosine Similarity**:
      * A cross-encoder sigmoid score of `0.85` represents a true, calibrated confidence probability! 
        Documents scoring below `0.30` can be safely discarded as verified non-relevant noise.
    """)

with col_sc2:
    st.subheader("Dynamic Cutoff Strategy")
    st.markdown("""
    * **Fixed Top-K vs. Quality Cutoff**:
      * In naive pipelines, systems always return a fixed `Top-K = 5` documents, even if 4 of them are noise.
      * In enterprise two-stage pipelines, the system applies a **Quality Cutoff Filter**:
        ```python
        # Filter out candidates failing minimum relevance confidence
        clean_context = [doc for doc in reranked_docs if doc.score >= 0.65]
        ```
      * If all candidates score below `0.65`, the system triggers an empty context fallback, preventing hallucinations entirely!
    """)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Two-Stage Reranking Orchestrator", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module demonstrating a two-stage retrieve-and-rerank pipeline with 
candidate over-fetching, cross-encoder joint scoring simulation, and score threshold gating.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseRetrieveAndRerankOrchestrator:
    """
    Two-stage retrieval orchestrator:
    Stage 1: Over-fetches coarse candidates from vector store (K=25).
    Stage 2: Cross-Encoder joint scoring re-orders candidates and applies quality cutoffs.
    """
    def __init__(
        self,
        first_stage_k: int = 25,
        final_top_k: int = 3,
        relevance_cutoff: float = 0.60
    ):
        self.first_stage_k = first_stage_k
        self.final_top_k = final_top_k
        self.relevance_cutoff = relevance_cutoff

    def execute_pipeline(
        self,
        query: str,
        retriever_client: Any,
        reranker_client: Any
    ) -> List[Dict[str, Any]]:
        # 1. Stage 1: Coarse candidate retrieval (Over-fetch)
        coarse_candidates = retriever_client.search(query, top_k=self.first_stage_k)
        if not coarse_candidates:
            return []

        # 2. Stage 2: Fine cross-encoder scoring
        # Pair query with each candidate text
        pairs = [[query, doc["text"]] for doc in coarse_candidates]
        cross_scores = reranker_client.compute_scores(pairs)

        # 3. Associate cross-encoder scores with documents
        reranked_pool = []
        for doc, score in zip(coarse_candidates, cross_scores):
            # Sigmoid normalization if scores are raw logits
            prob_score = 1.0 / (1.0 + np.exp(-score)) if score < 0 or score > 1.0 else score
            item = dict(doc)
            item["rerank_score"] = float(round(prob_score, 4))
            reranked_pool.append(item)

        # 4. Sort descending by fine-grained reranker score
        reranked_pool.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 5. Apply quality cutoff filter and truncate to final top-k
        final_selected = [
            d for d in reranked_pool if d["rerank_score"] >= self.relevance_cutoff
        ][:self.final_top_k]

        return final_selected
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Retrieve-and-Rerank Simulation Console", divider="rainbow")

st.markdown("""
Experience how a **Two-Stage Retrieve-and-Rerank** pipeline rescues RAG systems from bi-encoder failure modes. 
Select a scenario below to observe how the cross-encoder reorders candidates and filters distractors.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    scenario_choice = st.selectbox(
        "Select Test Scenario:",
        list(TwoStageRerankingEngine.PRESET_QUERIES.keys())
    )
    cutoff_slider = st.slider("Cross-Encoder Quality Cutoff Score:", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    final_k_slider = st.slider("Final Document Candidates to LLM:", min_value=1, max_value=3, value=2)

    pipeline_out = TwoStageRerankingEngine.simulate_pipeline(
        preset_key=scenario_choice,
        stage1_cutoff=0.60,
        stage2_top_k=final_k_slider
    )

    st.markdown("#### 🎯 Query & Challenge Analysis")
    st.info(f"**Query**: *\"{pipeline_out['query']}\"*")
    st.caption(f"💡 {pipeline_out['explanation']}")

with wb_col2:
    st.markdown("### 📊 Stage 1 (Bi-Encoder) vs Stage 2 (Cross-Encoder)")

    st.markdown("**Stage 1: Coarse Bi-Encoder Hits (Unordered)**")
    s1_df = pd.DataFrame(pipeline_out["stage1_results"])
    st.dataframe(s1_df[["id", "title", "bi_encoder_score", "category"]], use_container_width=True)

    st.markdown("**Stage 2: Cross-Encoder Reranked Finalists (Post-Filter)**")
    s2_df = pd.DataFrame(pipeline_out["stage2_results"])
    st.dataframe(s2_df[["id", "title", "cross_encoder_score", "bi_encoder_score", "text"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is the fundamental architectural difference between a Bi-Encoder and a Cross-Encoder?"):
    st.markdown("""
    **Answer**: 
    * **Bi-Encoder**: Encodes the query and document independently into fixed-length dense vectors using two separate forward passes: 
      `Score = dot(f(Q), f(D))`. Because document vectors are pre-computed offline, retrieval at query time requires only sub-millisecond 
      vector similarity search across millions of documents. However, because tokens in the query never cross-attend to tokens in the document, 
      fine-grained interactions and negations are lost.
    * **Cross-Encoder**: Feeds the query and document simultaneously into a single Transformer sequence: `Input = [CLS] Q [SEP] D [SEP]`. 
      Full all-to-all cross-attention is computed between every query token and every document token across all layers. 
      This yields state-of-the-art relevance scoring accuracy, but requires a live forward pass for every candidate pair at runtime, 
      limiting its application to small candidate pools (K <= 50).
    """)

with st.expander("Q2: Why does adding a cross-encoder reranker reduce LLM token costs in RAG pipelines?"):
    st.markdown("""
    **Answer**: In a single-stage retrieval pipeline, because bi-encoders have lower precision, engineers often pass 10 to 15 document chunks 
    into the LLM context prompt to ensure the correct answer is present somewhere. This consumes thousands of prompt tokens per turn. 
    By inserting a cross-encoder reranker, the system reliably promotes the single best document to rank #1 and filters out noise. 
    The prompt can safely be shrunk from 15 chunks down to 3 high-precision chunks, reducing downstream LLM prompt token consumption by 70% 
    while improving answer quality.
    """)

with st.expander("Q3: What is the recommended latency budget for the reranking stage in production search?"):
    st.markdown("""
    **Answer**: In real-time conversational search, the total end-to-end latency budget is typically 800–1200ms (mostly consumed by generative LLM time-to-first-token). 
    The retrieval and reranking phase should complete in **under 40 to 60 milliseconds**. 
    A modern 6-layer or 12-layer Cross-Encoder (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) scoring 25 document pairs 
    takes approximately 15–30ms on a T4/A10G GPU, fitting comfortably within the operational latency budget.
    """)

with st.expander("Q4: How does a cross-encoder resolve negation queries that fool vector embedding models?"):
    st.markdown("""
    **Answer**: In dense vector embeddings, sentences like *"Drug X is recommended"* and *"Drug X is NOT recommended"* share 95% of identical vocabulary 
    and syntactic structure. Contrastive embedding models pull them into almost identical vector coordinates. 
    In a Cross-Encoder, the query and document are processed jointly in the same self-attention layers. 
    The attention heads directly compute weights between the query token *"NOT"* and the document predicate, 
    allowing the model to recognize the semantic inversion and dramatically depress the relevance logit.
    """)

with st.expander("Q5: What is ColBERT (Late Interaction) and how does it fit between Bi-Encoders and Cross-Encoders?"):
    st.markdown("""
    **Answer**: **ColBERT (Contextualized Late Interaction over BERT)** is a hybrid architecture designed to achieve Cross-Encoder quality 
    at Bi-Encoder retrieval speeds. Instead of compressing a document into a single pooled vector, ColBERT keeps token-level embeddings 
    for all words in both query and document. At search time, it computes the **MaxSim operator** (maximum inner product across token pairs). 
    Because document token vectors can be pre-computed offline, ColBERT executes in under 25ms over millions of documents while achieving 
    near-Cross-Encoder ranking precision.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Feeding Unchunked 2,000-Word Documents into a Cross-Encoder**
    * **Symptom**: Cross-encoder latency balloons from 25ms to 800ms, or crashes with CUDA Out-Of-Memory errors.
    * **Root Cause**: Concatenating a query with a 2,000-word document exceeds the model's 512-token context window. All-to-all attention scales quadratically O(N^2) with sequence length.
    * **Remedy**: Always ensure document chunks are strictly bounded between 200 and 400 tokens before feeding candidate pairs into the reranker.
    """)

    st.error("""
    **Trap 2: Setting the First-Stage Candidate Window Too Narrow (K < 10)**
    * **Symptom**: Reranker fails to improve overall RAG answer accuracy.
    * **Root Cause**: If the true relevant document is ranked at position #14 in the vector database, fetching only K=5 candidates means the reranker never sees the true document.
    * **Remedy**: Over-fetch candidate pools: fetch Top 25 to 50 candidates in Stage 1 so the reranker has high recall coverage to work with.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Sequential Scoring of Candidate Pairs in a Python Loop**
    * **Symptom**: Scoring 25 candidate documents takes 250ms on GPU.
    * **Root Cause**: Calling `reranker.predict([query, doc])` in a synchronous `for` loop 25 times.
    * **Remedy**: Batch all 25 candidate pairs into a single tensor batch: `reranker.predict([[q, d1], [q, d2], ...], batch_size=32)`, leveraging full GPU tensor core parallelization.
    """)

    st.warning("""
    **Trap 4: Treating Cross-Encoder Output as Linear Probabilities Without Sigmoid**
    * **Symptom**: Quality threshold filters behave erratically across different reranker model architectures.
    * **Root Cause**: Many cross-encoders output raw unbounded classification logits (-10 to +10). Comparing raw logits to a 0.70 probability cutoff discards valid candidates.
    * **Remedy**: Apply the standard logistic sigmoid transformation `1 / (1 + exp(-logit))` before evaluating cutoff thresholds.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Two-Stage Retrieve-and-Rerank architecture?")
        st.markdown("**Back**: An information retrieval pattern where an inexpensive bi-encoder retrieves a broad candidate pool (Top 25), and a high-precision cross-encoder reranks the candidates to output the top 3–5 finalists for the LLM.")

    with st.container(border=True):
        st.markdown("**Front**: Why can Cross-Encoders not be used for initial database retrieval?")
        st.markdown("**Back**: Because cross-encoders require a joint forward pass over concatenated query-document token pairs. Computing this across millions of documents at query time would take minutes instead of milliseconds.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: How does Reranking mitigate the Lost-in-the-Middle effect in LLMs?")
        st.markdown("**Back**: By reliably promoting the golden relevant document from rank #8 or #12 up to rank #1, placing it at the very top of the prompt context where LLM attention is strongest.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What is the recommended Stage 1 over-fetch window size for production RAG?")
            st.markdown("**Back**: Top 20 to 35 candidate documents. This window captures ~95% of all potential recall gains while keeping cross-encoder batch inference under 35 milliseconds.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Reranking Basics, Cross-Encoder Mechanics, and Two-Stage Pipelines:
""")

chk1 = st.checkbox("I understand the trade-offs between Bi-Encoder scalability and Cross-Encoder scoring expressiveness.", value=True)
chk2 = st.checkbox("I can articulate how Cross-Encoders resolve negation blindness through joint all-to-all attention.", value=True)
chk3 = st.checkbox("I know why the first-stage retriever must over-fetch candidates (K = 20 to 35) to provide high recall.", value=True)
chk4 = st.checkbox("I understand how reranking solves the Lost-in-the-Middle attention degradation effect in generator LLMs.", value=True)
chk5 = st.checkbox("I know how to transform raw cross-encoder classification logits into calibrated sigmoid probabilities.", value=True)
chk6 = st.checkbox("I can batch candidate pairs into a single tensor forward pass to maintain sub-35ms reranker latency.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Masterful! You have conquered Reranking Basics and Two-Stage Retrieval architectures.")
