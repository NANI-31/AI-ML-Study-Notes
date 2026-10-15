import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class AdvancedRetrievalItem:
    """Represents a candidate document in an advanced retrieval pipeline."""
    def __init__(
        self,
        doc_id: str,
        title: str,
        snippet: str,
        keywords: List[str],
        semantic_profile: str
    ):
        self.doc_id = doc_id
        self.title = title
        self.snippet = snippet
        self.keywords = [k.lower() for k in keywords]
        self.semantic_profile = semantic_profile

class MockAdvancedRetrievalStudio:
    """
    Simulates a complete advanced retrieval lifecycle:
    Query Rewriting (HyDE / Step-back) -> Sparse (BM25) + Dense -> RRF Fusion -> Cross-Encoder Reranking.
    """
    def __init__(self):
        self.corpus: List[AdvancedRetrievalItem] = [
            AdvancedRetrievalItem(
                doc_id="doc-adv-01",
                title="LangChain BaseRetriever Interface & Runnables",
                snippet="BaseRetriever adheres to the Runnable protocol, exposing invoke, ainvoke, and batch methods returning List[Document].",
                keywords=["baseretriever", "runnable", "invoke", "lcel", "interface"],
                semantic_profile="architecture"
            ),
            AdvancedRetrievalItem(
                doc_id="doc-adv-02",
                title="Cohere & BGE Cross-Encoder Reranking Architecture",
                snippet="Cross-encoders compute full cross-attention between the query and candidate document tokens, eliminating bi-encoder compression losses.",
                keywords=["cross-encoder", "reranker", "cohere", "bge", "attention"],
                semantic_profile="reranking"
            ),
            AdvancedRetrievalItem(
                doc_id="doc-adv-03",
                title="Reciprocal Rank Fusion (RRF) for Hybrid Ensembles",
                snippet="RRF merges sparse BM25 and dense vector rank orders using score = 1 / (60 + rank), ensuring robust calibration without score normalization.",
                keywords=["rrf", "reciprocal rank fusion", "hybrid", "bm25", "dense"],
                semantic_profile="fusion"
            ),
            AdvancedRetrievalItem(
                doc_id="doc-adv-04",
                title="Hypothetical Document Embeddings (HyDE) Prompting",
                snippet="HyDE generates a synthetic hypothetical answer using an LLM, embedding that answer to search for real documents with matching vector topology.",
                keywords=["hyde", "hypothetical", "query rewriting", "expansion", "prompt"],
                semantic_profile="rewriting"
            ),
            AdvancedRetrievalItem(
                doc_id="doc-adv-05",
                title="RAG Triad & Ragas Retrieval Evaluation Metrics",
                snippet="Context Precision measures whether relevant chunks are placed at the top, while Context Recall verifies if all ground-truth facts were fetched.",
                keywords=["ragas", "context precision", "context recall", "evaluation", "hit rate"],
                semantic_profile="evaluation"
            )
        ]

    def rewrite_query(self, query: str, mode: str) -> str:
        """Simulates HyDE and Step-Back query rewriting."""
        q_lower = query.lower()
        if mode == "HyDE (Hypothetical Document Embeddings)":
            return f"Hypothetical Answer: In enterprise RAG, high precision retrieval requires {query}. Specifically, algorithms utilize cross-attention scoring and reciprocal rank fusion to isolate authoritative facts."
        elif mode == "Step-Back Prompting":
            if "rerank" in q_lower or "cross" in q_lower:
                return "What are the fundamental architectural principles of neural information retrieval and two-stage ranking?"
            elif "eval" in q_lower or "precision" in q_lower:
                return "How is information retrieval quality evaluated mathematically using precision and recall?"
            return f"High-level conceptual overview of: {query}"
        return query

    def _score_bm25(self, query: str, doc: AdvancedRetrievalItem) -> float:
        q_tokens = query.lower().split()
        matches = sum(1 for kw in doc.keywords if any(kw in t or t in kw for t in q_tokens))
        return round(min(1.0, (matches * 0.35) + (0.05 if any(t in doc.snippet.lower() for t in q_tokens) else 0.0)), 4)

    def _score_dense(self, query: str, doc: AdvancedRetrievalItem) -> float:
        q_lower = query.lower()
        # Synthetic dense similarity
        score = 0.30
        if doc.semantic_profile in q_lower or any(kw in q_lower for kw in doc.keywords):
            score += 0.45
        if "retriev" in q_lower and "retriever" in doc.title.lower():
            score += 0.20
        return round(min(0.97, max(0.20, score)), 4)

    def _score_cross_encoder(self, query: str, doc: AdvancedRetrievalItem) -> float:
        """Simulates a high-precision Cross-Encoder with full token interaction."""
        q_lower = query.lower()
        # Cross encoders heavily reward exact context alignment
        dense = self._score_dense(query, doc)
        bm25 = self._score_bm25(query, doc)
        
        # Cross-encoder joint reasoning boost
        joint_boost = 0.0
        if "cross" in q_lower and "cross-encoder" in doc.keywords:
            joint_boost = 0.45
        elif "hyde" in q_lower and "hyde" in doc.keywords:
            joint_boost = 0.45
        elif "eval" in q_lower and "evaluation" in doc.keywords:
            joint_boost = 0.45
        elif "rrf" in q_lower and "rrf" in doc.keywords:
            joint_boost = 0.45
        elif "interface" in q_lower and "baseretriever" in doc.keywords:
            joint_boost = 0.45

        final_score = (dense * 0.4) + (bm25 * 0.3) + joint_boost
        return round(min(0.99, max(0.15, final_score)), 4)

    def execute_advanced_pipeline(
        self,
        query: str,
        rewrite_mode: str,
        pipeline_stage: str,
        k: int,
        rerank_threshold: float
    ) -> Dict[str, Any]:
        # 1. Query Rewriting
        effective_query = self.rewrite_query(query, rewrite_mode)

        # 2. Candidate Scoring
        candidates = []
        for d in self.corpus:
            bm25_s = self._score_bm25(effective_query, d)
            dense_s = self._score_dense(effective_query, d)
            ce_s = self._score_cross_encoder(effective_query, d)
            candidates.append({
                "doc": d,
                "bm25_score": bm25_s,
                "dense_score": dense_s,
                "ce_score": ce_s
            })

        # Dense ranking
        candidates_dense = sorted(candidates, key=lambda x: x["dense_score"], reverse=True)
        for r, item in enumerate(candidates_dense, 1):
            item["dense_rank"] = r

        # BM25 ranking
        candidates_bm25 = sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)
        for r, item in enumerate(candidates_bm25, 1):
            item["bm25_rank"] = r

        # RRF Fusion
        for item in candidates:
            rrf = (1.0 / (60 + item["dense_rank"])) + (1.0 / (60 + item["bm25_rank"]))
            item["rrf_score"] = round(rrf, 5)

        candidates_rrf = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)
        for r, item in enumerate(candidates_rrf, 1):
            item["rrf_rank"] = r

        # Cross-Encoder Reranking
        candidates_reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        for r, item in enumerate(candidates_reranked, 1):
            item["final_ce_rank"] = r

        # Select stage output
        if pipeline_stage == "Dense Only":
            final_list = candidates_dense[:k]
        elif pipeline_stage == "BM25 Only":
            final_list = candidates_bm25[:k]
        elif pipeline_stage == "Hybrid (RRF Fusion)":
            final_list = candidates_rrf[:k]
        else: # "Hybrid + Cross-Encoder Reranker"
            filtered = [c for c in candidates_reranked if c["ce_score"] >= rerank_threshold]
            final_list = filtered[:k]

        return {
            "effective_query": effective_query,
            "results": final_list,
            "all_candidates": candidates_reranked
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.3 Advanced Retrieval — LangChain Course",
    page_icon="⚡",
    layout="wide"
)

st.title("15.3 Advanced Retrieval")
st.caption("Deep-Dive into Hybrid Search, Cross-Encoder Reranking, Reciprocal Rank Fusion, Query Rewriting, and Retrieval Evaluation")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: Advanced Retrieval, RRF & Precision Reranking Pipeline", divider="rainbow")
st.markdown("""
Modern enterprise RAG pipelines abandon single-stage vector lookups in favor of a **multi-stage retrieval and reranking funnel**. 
This state-of-the-art architecture couples query reformulation with hybrid sparse-dense candidate generation, 
reciprocal rank fusion, and heavy cross-encoder neural reranking to deliver pinpoint accuracy.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedRetrievalArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_stage0 {
            label = "Stage 0: Query Pre-Processing & Expansion";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            raw_q [label="User Query: 'How to score cross encoders?'", fillcolor="#1E293B", color="#0284C7"];
            rewrite_node [label="Query Rewriter (HyDE / Step-Back / Sub-Queries)\\nExpands vocabulary and generates hypothetical answers", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_stage1 {
            label = "Stage 1: Multi-Channel Candidate Retrieval (Broad Recall, k=50)";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            sparse_branch [label="Sparse Lexical Channel (BM25)\\nExact keyword matching, acronyms, IDs", fillcolor="#1E293B", color="#64748B"];
            dense_branch [label="Dense Semantic Channel (Bi-Encoder HNSW)\\nConceptual semantic proximity embeddings", fillcolor="#1E293B", color="#64748B"];
        }

        subgraph cluster_stage2 {
            label = "Stage 2: Rank Fusion Engine (k=25)";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            rrf_node [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#0369A1" colspan="2"><b>Reciprocal Rank Fusion (RRF)</b></td></tr>
                <tr><td bgcolor="#1E293B">Formula</td><td bgcolor="#1E293B">RRF_score = sum( w_i / (60 + rank_i) )</td></tr>
                <tr><td bgcolor="#1E293B">Properties</td><td bgcolor="#1E293B">Score-scale invariant, robust calibration</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_stage3 {
            label = "Stage 3: Deep Cross-Encoder Neural Reranking (High Precision, k=5)";
            style = "rounded";
            color = "#D97706";
            fontcolor = "#FBBF24";
            reranker_node [label="Cross-Encoder Reranker (Cohere / BAAI BGE-Reranker)\\nFull joint query-document cross-attention", fillcolor="#78350F", color="#F59E0B"];
            threshold_gate [label="Relevance Gating & Score Cutoff\\nDiscard chunks with score &lt; tau", fillcolor="#1E293B", color="#EF4444"];
        }

        subgraph cluster_stage4 {
            label = "Stage 4: Grounded Generation Context";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            final_context [label="Final Grounded Prompt Context (Top 3-5 Chunks)\\nMaximum signal-to-noise ratio", fillcolor="#064E3B", color="#10B981"];
        }

        raw_q -> rewrite_node;
        rewrite_node -> sparse_branch [label="Token Query"];
        rewrite_node -> dense_branch [label="Embedded Vector"];
        
        sparse_branch -> rrf_node [label="Top-50 Sparse"];
        dense_branch -> rrf_node [label="Top-50 Dense"];
        
        rrf_node -> reranker_node [label="Top-25 Fused Candidates"];
        reranker_node -> threshold_gate [label="Cross-Attention Scores"];
        threshold_gate -> final_context [label="Filtered Clean Chunks"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: HYBRID SEARCH ---
st.header("2. Hybrid Search", divider="rainbow")
st.markdown("""
In enterprise knowledge bases, neither pure dense vector retrieval nor pure lexical keyword search is sufficient on its own:
- **Dense Vector Search (Bi-Encoders)** maps queries and documents into continuous embedding spaces. It shines at understanding 
  paraphrases, high-level intent, and conceptual synonyms (e.g. associating *"physician"* with *"doctor"*). However, it frequently misses 
  exact error codes, model numbers, alphanumeric SKUs, or rare company acronyms.
- **Lexical Search (BM25 / Inverted Indexes)** evaluates term frequency and inverse document frequency. It has pinpoint precision on 
  exact technical terms, function names (`BaseRetriever`), and specific product codes, but fails completely when the user uses words 
  not literally present in the index.

**Hybrid Search** runs both pipelines simultaneously over the corpus, merging candidate lists to achieve both **high semantic recall** 
and **exact keyword precision**.
""")

st.divider()

# --- SECTION 3: KEYWORD + SEMANTIC SEARCH ---
st.header("3. Keyword + Semantic Search Calibration", divider="rainbow")
st.markdown("""
Combining keyword and semantic search requires resolving a fundamental mathematical incompatibility: **score distributions**.
BM25 generates unbounded positive scores (often ranging from 0.0 to 30.0+ depending on document length and term frequency), 
while cosine similarity produces scores bounded between -1.0 and 1.0 (or 0.0 and 1.0).
""")

comp_df = pd.DataFrame({
    "Search Paradigm": ["Lexical (BM25 / Lucene)", "Dense Vector (Bi-Encoder)"],
    "Underlying Index": ["Inverted index of stemmed token postings", "HNSW / IVF vector geometric graph"],
    "Score Range": ["Unbounded positive: [0, infinity)", "Cosine: [-1, 1] or L2 distance: [0, infinity)"],
    "Strengths": ["Exact keyword hits, acronyms, code identifiers, zero GPU cost", "Semantic synonyms, multi-lingual concepts, conceptual intent"],
    "Weaknesses": ["Zero synonym awareness, vocabulary mismatch", "Misses exact alphanumeric identifiers, high embedding compute cost"]
})
st.dataframe(comp_df, use_container_width=True)

st.markdown("""
### Fusion Techniques:
1. **Min-Max Score Normalization**: Scales both score sets to `[0, 1]` and computes a weighted average `alpha * dense + (1 - alpha) * bm25`. 
   Vulnerable to score clustering and outliers.
2. **Reciprocal Rank Fusion (RRF)**: Ignores raw score magnitudes entirely and operates purely on relative rank order. Recommended standard for enterprise production.
""")

st.divider()

# --- SECTION 4: RERANKING ---
st.header("4. Reranking with Cross-Encoders", divider="rainbow")
st.markdown("""
The single most impactful upgrade you can introduce to a RAG retrieval pipeline is a **Cross-Encoder Reranker** 
(such as Cohere Rerank, BAAI `bge-reranker-large`, or MS-MARCO MiniLM).
""")

col_bi, col_cross = st.columns(2)
with col_bi:
    st.info("""
    **Bi-Encoder (First-Stage Retriever)**
    - Embeds queries and documents **independently** into fixed-length vectors.
    - Ultra-fast: O(1) vector distance comparisons via pre-indexed HNSW graphs.
    - Limitation: Compresses an entire document into a single vector, losing fine-grained word-to-word interactions.
    """)
    st.code("""
# Independent encoding
q_vec = embed(query)
d_vec = embed(document)
score = cosine(q_vec, d_vec)
    """, language="python")

with col_cross:
    st.success("""
    **Cross-Encoder (Second-Stage Reranker)**
    - Feeds the query and document **jointly** into full multi-head cross-attention layers.
    - Captures deep token interactions across every word in the query and candidate chunk.
    - Computationally expensive, but only evaluated on the top-20 to top-50 candidates.
    """)
    st.code("""
# Joint full cross-attention
input_pair = f"{query} [SEP] {document}"
score = cross_encoder_model(input_pair)
    """, language="python")

st.divider()

# --- SECTION 5: RELEVANCE SCORING ---
st.header("5. Relevance Scoring and Reciprocal Rank Fusion", divider="rainbow")
st.markdown("""
**Reciprocal Rank Fusion (RRF)** is the mathematical cornerstone of robust hybrid search. Developed by Cormack et al., 
RRF merges ranked lists without requiring score calibration or probability normalization.
""")

st.markdown("""
### RRF Formulation:
For any document d present across M distinct retrieval result lists R_1, ..., R_M:

`RRF_score(d) = sum_{m=1}^{M} [ w_m / (k_rrf + r_m(d)) ]`

Where:
- `r_m(d)` is the 1-based rank position of document d in result list m (if document d is not present in list m, its rank is considered infinity).
- `k_rrf` is a smoothing constant, universally set to **60** to prevent top-1 candidates from dominating the fusion.
- `w_m` is the assigned channel weight (e.g. 0.5 for BM25, 0.5 for Dense).
""")

st.divider()

# --- SECTION 6: RETRIEVAL FILTERING ---
st.header("6. Retrieval Filtering & Post-Rerank Threshold Gating", divider="rainbow")
st.markdown("""
Retrieving documents is only half the battle; knowing when to **throw documents away** is what separates high-precision 
enterprise RAG from hallucination-prone prototypes.
""")

col_filt1, col_filt2 = st.columns(2)
with col_filt1:
    st.markdown("""
    ### Post-Reranker Score Cutoff (tau):
    Cross-encoders output well-calibrated relevance probabilities between 0.0 and 1.0. 
    By establishing a minimum relevance threshold (typically tau between 0.40 and 0.65):
    - Unrelated documents that only surfaced due to top-k quotas are pruned.
    - If no documents meet tau, the retriever returns an empty list, allowing the application to respond: 
      *"I cannot find authoritative information in the knowledge base to answer this question."*
    """)
with col_filt2:
    st.markdown("""
    ### Dynamic Top-k Pruning:
    Instead of passing a static k = 5 chunks to the LLM:
    - **Relative Score Dropoff**: Prune any chunk whose score drops by more than 30% relative to the #1 ranked chunk.
    - **Cumulative Score Budget**: Include chunks in descending order until a cumulative probability budget is exhausted.
    """)

st.divider()

# --- SECTION 7: QUERY REWRITING ---
st.header("7. Query Rewriting (HyDE & Step-Back)", divider="rainbow")
st.markdown("""
Raw user queries are often colloquial, terse, or phrased as questions rather than answers. Embedding a question vector 
and searching for answer paragraphs can fail due to asymmetric vector clustering.
""")

col_hyde, col_step = st.columns(2)
with col_hyde:
    st.subheader("HyDE (Hypothetical Document Embeddings)")
    st.markdown("""
    1. Prompt an LLM: *"Write a hypothetical passage that directly answers: '{query}'"*.
    2. The generated passage may contain factual hallucinations, but its **vector topology closely mirrors real authoritative answers**.
    3. Embed the hypothetical document and perform vector search against the corpus.
    """)
    st.code("""
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm,
    base_embeddings,
    prompt_key="web_search"
)
    """, language="python")

with col_step:
    st.subheader("Step-Back Prompting")
    st.markdown("""
    1. For hyper-specific questions, instruct the LLM to step back and ask a **broader, high-level conceptual question**.
    2. Retrieve documents for both the original question and the step-back question.
    3. Provides foundational context that prevents the LLM from missing fundamental principles.
    """)
    st.code("""
# Prompt instruction:
# "You are an expert at information retrieval. Given a user query,
# take a step back and synthesize a broader, more abstract question."
    """, language="python")

st.divider()

# --- SECTION 8: QUERY EXPANSION ---
st.header("8. Query Expansion & Sub-Query Decomposition", divider="rainbow")
st.markdown("""
Complex real-world inquiries frequently contain **multi-hop questions** that cannot be satisfied by a single retrieval pass.

For example:
> *"Compare the latency SLAs between our Aurora PostgreSQL database and DynamoDB clusters."*

A single retrieval call will either match Aurora documents or DynamoDB documents, but rarely both in the top-3.
""")

st.markdown("""
### Multi-Aspect Sub-Query Decomposition:
1. An LLM analyzes the query and decomposes it into independent, parallel sub-queries:
   - Sub-query 1: *"What is the latency SLA for Aurora PostgreSQL?"*
   - Sub-query 2: *"What is the latency SLA for DynamoDB?"*
2. Each sub-query is executed independently across the retriever.
3. The retrieved document sets are merged and deduplicated before prompt assembly.
""")

st.divider()

# --- SECTION 9: RETRIEVAL EVALUATION ---
st.header("9. Retrieval Evaluation (Ragas & Information Retrieval Metrics)", divider="rainbow")
st.markdown("""
You cannot optimize what you do not measure. Evaluating retrieval quality requires specialized Information Retrieval (IR) 
metrics and LLM-assisted evaluation frameworks like **Ragas**.
""")

eval_data = {
    "Metric Name": [
        "Hit Rate @ k",
        "Mean Reciprocal Rank (MRR)",
        "Context Precision (Ragas)",
        "Context Recall (Ragas)"
    ],
    "Mathematical Definition": [
        "Fraction of queries where the ground-truth document appears within the top-k results.",
        "Average of reciprocal rank: (1 / |Q|) * sum( 1 / rank_i ), where rank_i is the position of first relevant doc.",
        "Measures the signal-to-noise ratio: fraction of relevant chunks in context, weighted by their rank position.",
        "Measures completeness: whether all necessary facts from the ground truth answer are present in retrieved chunks."
    ],
    "Target Benchmark": [
        "> 0.85 @ k=5",
        "> 0.70",
        "> 0.80",
        "> 0.90"
    ]
}
st.dataframe(pd.DataFrame(eval_data), use_container_width=True)

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Advanced Hybrid RRF & Reranker Pipeline", divider="rainbow")
st.markdown("""
Here is a production-grade Python implementation of an advanced multi-stage retrieval pipeline in LangChain v0.3+. 
It combines BM25 keyword search, dense FAISS vectors via RRF, and a Cohere Cross-Encoder reranker with score threshold gating.
""")

st.code("""
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def build_advanced_reranked_pipeline(
    documents: List[Document],
    openai_api_key: str,
    cohere_api_key: str,
    final_k: int = 4,
    min_relevance_score: float = 0.60
) -> ContextualCompressionRetriever:
    \"\"\"
    Constructs an end-to-end production RAG retrieval pipeline:
    1. First Stage: Hybrid EnsembleRetriever (BM25 + FAISS) fetching top-25 broad candidates via RRF.
    2. Second Stage: CohereRerank cross-encoder evaluating full cross-attention.
    3. Gating: Filters candidates falling below min_relevance_score and returns top final_k chunks.
    \"\"\"
    # 1. First-Stage Retrievers
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 25

    # 2. Hybrid RRF Ensemble (Fetches top 25 diverse candidates)
    hybrid_ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 3. Second-Stage Cross-Encoder Reranker
    reranker = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-v3.5",
        top_n=final_k,
        score_threshold=min_relevance_score
    )

    # 4. Wrap into LCEL ContextualCompressionRetriever
    pipeline = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_ensemble
    )

    return pipeline
""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Advanced Retrieval Studio", divider="rainbow")
st.markdown("""
Experience how each stage of the advanced retrieval funnel refines candidate rankings. 
Compare **Dense-only**, **BM25-only**, **Hybrid RRF**, and **Hybrid + Cross-Encoder Reranking**, and observe how 
**HyDE query rewriting** alters candidate scores.
""")

# Instantiate the module-scope engine
adv_studio = MockAdvancedRetrievalStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    user_q = st.text_input(
        "Enter Retrieval Query:",
        value="How does a cross-encoder reranker calculate cross-attention scores?"
    )

    rewrite_option = st.selectbox(
        "Query Pre-Processing / Expansion Technique:",
        ["None (Raw Query)", "HyDE (Hypothetical Document Embeddings)", "Step-Back Prompting"]
    )

    pipeline_stage_select = st.radio(
        "Pipeline Execution Stage:",
        [
            "Dense Only (Bi-Encoder k-NN)",
            "BM25 Only (Lexical Keyword)",
            "Hybrid (RRF Fusion)",
            "Hybrid + Cross-Encoder Reranker"
        ],
        index=3
    )

    rerank_tau = 0.50
    if "Cross-Encoder" in pipeline_stage_select:
        rerank_tau = st.slider(
            "Cross-Encoder Relevance Cutoff (tau):",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05
        )

    k_slider = st.slider("Select Output k:", min_value=1, max_value=5, value=3)
    run_btn = st.button("🚀 Run Advanced Retrieval Funnel", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Funnel Execution Inspection")

    if run_btn or user_q:
        stage_key = "Hybrid + Cross-Encoder Reranker"
        if "Dense Only" in pipeline_stage_select:
            stage_key = "Dense Only"
        elif "BM25 Only" in pipeline_stage_select:
            stage_key = "BM25 Only"
        elif "Hybrid (RRF" in pipeline_stage_select:
            stage_key = "Hybrid (RRF Fusion)"

        execution = adv_studio.execute_advanced_pipeline(
            query=user_q,
            rewrite_mode=rewrite_option,
            pipeline_stage=stage_key,
            k=k_slider,
            rerank_threshold=rerank_tau
        )

        st.info(f"**Effective Query Processed:**\\n`{execution['effective_query']}`")

        results = execution["results"]
        if results:
            for idx, item in enumerate(results, 1):
                doc = item["doc"]
                with st.expander(f"Rank #{idx}: {doc.title} ({doc.doc_id})", expanded=True):
                    st.write(f"**Snippet**: `{doc.snippet}`")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Dense Rank", f"#{item['dense_rank']}", f"{item['dense_score']}")
                    m2.metric("BM25 Rank", f"#{item['bm25_rank']}", f"{item['bm25_score']}")
                    m3.metric("RRF Score", f"{item['rrf_score']}")
                    m4.metric("Cross-Encoder", f"#{item['final_ce_rank']}", f"{item['ce_score']}")
        else:
            st.warning("⚠️ No documents met the cross-encoder relevance threshold cutoff! Try lowering tau.")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is a two-stage retrieval pipeline (Retriever + Cross-Encoder Reranker) computationally optimal?",
     "Running a heavy cross-encoder over millions of documents in real time is computationally impossible because it requires joint token attention with latency proportional to O(N * (L_q + L_d)^2). A bi-encoder retriever rapidly prunes millions of documents down to 20-50 high-potential candidates in sub-5ms using vector indexing. The cross-encoder is then applied only to these 20-50 candidates, delivering high cross-attention accuracy with negligible latency impact."),
    ("Why is Reciprocal Rank Fusion (RRF) preferred over linear weighted sum in hybrid search?",
     "Linear score combination requires normalizing disparate score scales (e.g. BM25 scores from 0-30 and Cosine scores from -1 to 1). This normalization is highly unstable because a single outlier BM25 match can skew min-max boundaries and drown out dense signals. RRF operates purely on discrete rank indices (1 / (60 + rank)), making it completely invariant to raw score magnitudes and robust across diverse query types."),
    ("How does Hypothetical Document Embeddings (HyDE) overcome asymmetric retrieval?",
     "Queries are short, interrogative, and asymmetric compared to long, declarative document passages. Because embedding models cluster questions closer to other questions than to answers, vector search often retrieves related FAQ questions rather than factual answers. HyDE prompts an LLM to generate a hypothetical answer passage. Embedding this synthetic declarative text lands directly within the vector cluster of real document answers."),
    ("What is the difference between Context Precision and Context Recall in Ragas evaluation?",
     "Context Recall evaluates whether the retrieved chunks contain all the necessary ground-truth facts required to formulate the complete answer. Context Precision evaluates ranking quality and noise: it measures whether the relevant chunks are positioned at the very top of the retrieved context window rather than buried beneath irrelevant distractors."),
    ("When would you NOT recommend using HyDE in a production system?",
     "Do not use HyDE when: (1) Latency budgets are strict (HyDE introduces an initial LLM generation call adding 500ms+), (2) The query involves exact product IDs or codes that the LLM has never seen (causing the hypothetical text to fabricate fictional IDs), or (3) In high-throughput, low-cost RAG systems where an extra LLM call per search triples API expenses.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Passing Top-K Chunks to Cross-Encoder Without Truncating Chunk Lengths. Cross-encoders have strict token context limits (typically 512 or 1024 tokens). If your first-stage retriever returns full 2,000-token parent documents, the cross-encoder will silently truncate the text or throw an API token limit exception. Pass small child chunks or compressed snippets to the reranker.")
    st.error("**Trap 2 (Breaking Error)**: Using HyDE for Precision Keyword Queries. If a user searches for an exact part number like 'XR-9421-B', an LLM generating a hypothetical answer will almost certainly hallucinate a different code, causing the dense embedding search to miss the real document completely.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running Rerankers on Large Candidate Pools (k > 100). Cohere and other reranker APIs charge per 1,000 search units based on the number of candidates evaluated. Passing k=100 per query will deplete API credits 10x faster than passing k=20 with virtually zero recall gain.")
    st.warning("**Trap 4 (Quality Warning)**: Setting RRF Constant k_rrf Too Low. Setting k_rrf = 1 instead of the standard 60 causes rank 1 documents to receive an enormous relative boost over rank 2, effectively ignoring agreement across the secondary retriever.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Cross-Encoder Reranker"):
        st.markdown("**Evaluates joint query-document cross-attention to accurately re-score top candidate chunks.**")
    with st.expander("🎴 Flashcard 2: Reciprocal Rank Fusion"):
        st.markdown("**Merges sparse and dense ranked candidate lists using score = sum( 1 / (60 + rank) ).**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: HyDE"):
        st.markdown("**Hypothetical Document Embeddings: embeds an LLM-generated fake answer to search for real answers.**")
    with st.expander("🎴 Flashcard 4: Context Precision"):
        st.markdown("**Measures the proportion of relevant chunks in context, weighted by top-rank placement.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the two-stage retrieval architecture: bi-encoder recall followed by cross-encoder precision.", value=True)
chk2 = st.checkbox("I can explain why Reciprocal Rank Fusion (RRF) avoids score calibration issues in hybrid search.", value=True)
chk3 = st.checkbox("I know how Cross-Encoders utilize full joint cross-attention to eliminate bi-encoder compression loss.", value=True)
chk4 = st.checkbox("I understand the mechanics and trade-offs of HyDE (Hypothetical Document Embeddings).", value=True)
chk5 = st.checkbox("I can differentiate between Context Precision and Context Recall in Ragas evaluation.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 advanced retrieval interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 15.3 Advanced Retrieval!")
