import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class ProductionRerankingPipelineSimulator:
    """
    Simulates the end-to-end 5-step production reranking pipeline:
    1. Retrieve Top 20 coarse candidates (Stage 1 over-fetch).
    2. Rerank candidates using Cross-Encoder joint scoring.
    3. Select Top 5 highest-ranking documents.
    4. Remove irrelevant context via quality cutoff filtering (drops noise).
    5. Pass final validated context with source citations to Generator LLM.
    """
    SCENARIOS = {
        "Scenario A (Mixed Quality - Noise Filtering)": {
            "query": "How to configure PgBouncer transaction pooling in Kubernetes",
            "coarse_pool": [
                {"id": "c_01", "title": "PgBouncer Helm Chart & Transaction Pooling", "text": "Deploying PgBouncer on Kubernetes with pool_mode = transaction and max_client_conn = 5000.", "coarse_rank": 1, "cross_score": 0.95},
                {"id": "c_02", "title": "Kubernetes StatefulSets for Databases", "text": "PostgreSQL StatefulSets with persistent volume claims and automated failover.", "coarse_rank": 2, "cross_score": 0.88},
                {"id": "c_03", "title": "Database Connection Saturation Alerts", "text": "Prometheus alerting rules for active client connections exceeding threshold.", "coarse_rank": 3, "cross_score": 0.74},
                {"id": "c_04", "title": "General Kubernetes Ingress Guide", "text": "Nginx Ingress controller setup for routing HTTP traffic to cluster services.", "coarse_rank": 4, "cross_score": 0.32},
                {"id": "c_05", "title": "Linux Kernel TCP Memory Tuning", "text": "Optimizing net.ipv4.tcp_rmem and wmem buffer limits for high-bandwidth networks.", "coarse_rank": 5, "cross_score": 0.28},
                {"id": "c_06", "title": "MySQL Master-Slave Replication", "text": "Binlog replication configurations for distributed MySQL clusters.", "coarse_rank": 6, "cross_score": 0.15}
            ]
        },
        "Scenario B (Out-of-Domain - Hallucination Prevention)": {
            "query": "What is the corporate dental insurance reimbursement limit for 2026?",
            "coarse_pool": [
                {"id": "c_10", "title": "Kubernetes Pod Eviction Thresholds", "text": "Kubelet evicts pods when node memory falls below the eviction threshold.", "coarse_rank": 1, "cross_score": 0.08},
                {"id": "c_11", "title": "Postgres Vacuuming Best Practices", "text": "Autovacuum worker parameters for dead tuple reclamation.", "coarse_rank": 2, "cross_score": 0.05},
                {"id": "c_12", "title": "Kafka Partition Rebalancing", "text": "Consumer group coordinator triggers rebalance when heartbeats time out.", "coarse_rank": 3, "cross_score": 0.03}
            ]
        }
    }

    @classmethod
    def execute_pipeline(
        cls,
        scenario_key: str,
        cutoff_threshold: float = 0.65,
        max_final_docs: int = 3
    ) -> Dict[str, Any]:
        scenario = cls.SCENARIOS[scenario_key]
        coarse_pool = scenario["coarse_pool"]

        # Step 2: Rerank candidates by cross-encoder score
        reranked = sorted(coarse_pool, key=lambda x: x["cross_score"], reverse=True)

        # Step 3: Select top candidates (up to 5)
        top_candidates = reranked[:5]

        # Step 4: Remove irrelevant context (apply quality cutoff)
        filtered_context = [d for d in top_candidates if d["cross_score"] >= cutoff_threshold]
        pruned_distractors = [d for d in top_candidates if d["cross_score"] < cutoff_threshold]

        # Step 5: Format final context payload for LLM
        final_payload = filtered_context[:max_final_docs]
        formatted_prompt_chunks = []
        for idx, doc in enumerate(final_payload, start=1):
            formatted_prompt_chunks.append(
                f"[Source {idx} - {doc['title']} (Confidence: {doc['cross_score']:.2f})]\n{doc['text']}"
            )

        return {
            "query": scenario["query"],
            "total_coarse_fetched": len(coarse_pool),
            "top_selected": top_candidates,
            "passed_quality_filter": final_payload,
            "pruned_distractors": pruned_distractors,
            "formatted_prompt_context": "\n\n".join(formatted_prompt_chunks) if formatted_prompt_chunks else "[EMPTY CONTEXT - TRIGGER SAFE FALLBACK]"
        }

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="13.3 Reranking Pipeline - RAG Master Curriculum",
    page_icon="🛠️",
    layout="wide"
)

st.title("13.3 Reranking Pipeline: Candidate Retrieval, Cross-Encoder Scoring, & Noise Pruning")
st.caption("Chapter 13: Reranking | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A production retrieve-and-rerank system is not merely an isolated model call; it is a synchronized 
**Five-Step Operational Pipeline**:
1. **Retrieve Top 20 Documents**: Over-fetch coarse candidates from vector or hybrid search.
2. **Rerank Candidates**: Compute deep joint cross-attention scores across all candidates in a parallel batch.
3. **Select Top 5 Documents**: Truncate the sorted candidate pool to the highest-scoring candidate window.
4. **Remove Irrelevant Context**: Apply an absolute relevance cutoff threshold, pruning distracting noise.
5. **Pass Final Context to LLM**: Format the surviving high-confidence chunks with citations for the generator LLM.

This lesson explores each stage of the production pipeline, demonstrating how strict relevance gating 
eliminates hallucinations and reduces prompt token expenses.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: End-to-End Production Reranking Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    pipe_bp_dot = """
    digraph RerankingPipelineBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_step1 {
            label = "Step 1: Coarse Candidate Over-Fetch (K = 20)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_q [label="User Query\\ne.g. 'PgBouncer transaction pooling in K8s'", fillcolor="#0F172A", color="#38BDF8"];
            coarse_retriever [label="Vector Store / Hybrid Index\\nFetches Top 20 Candidates (< 15ms)", fillcolor="#334155", color="#10B981"];
            coarse_pool [label="20 Candidate Document Chunks\\nHigh Recall, High Noise", fillcolor="#0F172A", color="#F59E0B"];

            user_q -> coarse_retriever -> coarse_pool;
        }

        subgraph cluster_step2_3 {
            label = "Steps 2 & 3: Cross-Encoder Scoring & Top 5 Selection";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            cross_batch [label="Batched Cross-Encoder Inference\\nScores 20 [Q, Doc] Pairs in Parallel (< 30ms)", fillcolor="#334155", color="#EC4899"];
            reordered_list [label="Sorted Candidate List\\nOrdered Descending by Calibrated Score", fillcolor="#0F172A", color="#10B981"];
            top5_slice [label="Top 5 Candidate Window\\nCandidate Slicing Step", fillcolor="#334155", color="#8B5CF6"];

            coarse_pool -> cross_batch;
            cross_batch -> reordered_list -> top5_slice;
        }

        subgraph cluster_step4_5 {
            label = "Steps 4 & 5: Noise Pruning & Prompt Context Assembly";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            cutoff_filter [label="Relevance Cutoff Filter (Score >= 0.65)\\nPrunes Weak Matches & Distractor Noise", fillcolor="#334155", color="#F59E0B"];
            final_context [label="Final Context Payload\\nClean Chunks + Citations [1..3]", fillcolor="#0F172A", color="#10B981"];
            generator_llm [label="Downstream Generator LLM\\nSynthesizes Grounded Truth Answer", fillcolor="#334155", color="#38BDF8"];

            top5_slice -> cutoff_filter -> final_context -> generator_llm;
        }
    }
    """
    st.graphviz_chart(pipe_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems, Step 4 (**Removing Irrelevant Context**) is the critical safety barrier. 
If an out-of-domain query is submitted, all 20 retrieved candidates will receive cross-encoder scores below `0.30`. 
Instead of forwarding low-confidence distractors to the generator LLM (which forces hallucinations), 
the pipeline intercepts the empty context and immediately short-circuits to an authenticated fallback message:
*"I cannot find authoritative documentation matching your inquiry."*
""")

st.divider()

# ==============================================================================
# SECTION 2: RETRIEVE TOP 20 DOCUMENTS
# ==============================================================================

st.header("2. Retrieve Top 20 Documents: Balancing Candidate Window Recall and Latency", divider="rainbow")

st.markdown("""
The first operational step is pulling a sufficiently broad candidate pool from the underlying search index.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why K = 20 is the Industry Sweet Spot")
    st.markdown("""
    * **The Golden Zone of Over-Fetching**:
      * Fetching only 5 documents results in a high probability of missing the true answer if it ranked #7 in vector search.
      * Fetching 100 documents increases cross-encoder batch latency by 4x without meaningful recall gains.
      * Empirical research shows that **Top 20 candidates** captures over 94% of all relevant documents.
    * **Execution Protocol**:
      * The vector database or hybrid engine returns: `List[Candidate(id, text, metadata, raw_vector_score)]`.
      * Candidate retrieval completes in 5–15 milliseconds.
    """)

with col2:
    st.subheader("Candidate Pool Ingestion Code Pattern")
    st.code('''# Step 1: Candidate Over-Fetch in LangChain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}  # Over-fetch candidate window
)

coarse_candidates = retriever.invoke(user_query)
assert len(coarse_candidates) <= 20
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: RERANK CANDIDATES
# ==============================================================================

st.header("3. Rerank Candidates: Batched Cross-Encoder Scoring and In-Memory Sorting", divider="rainbow")

st.markdown("""
Step 2 passes the 20 candidate pairs into the cross-encoder in a single batched tensor forward pass.
""")

col_sc1, col_sc2 = st.columns([1, 1])

with col_sc1:
    st.subheader("Batched Cross-Encoder Execution")
    st.markdown("""
    * **Batch Construction**:
      * The query is paired with each candidate text:
        `pairs = [[query, doc.text] for doc in coarse_candidates]`
    * **Hardware Saturation**:
      * Passing all 20 pairs in a single tensor batch (`batch_size=32`) allows GPU CUDA tensor cores 
        to execute matrix multiplications in parallel, completing in ~20 milliseconds.
    * **Descendant Sorting**:
      * Documents are reordered based on their cross-encoder score:
        `candidates.sort(key=lambda x: x.cross_score, reverse=True)`
    """)

with col_sc2:
    st.subheader("Scoring Transformation Matrix")
    score_ex_df = pd.DataFrame([
        {"Doc ID", "Vector Rank (Stage 1)", "Cross-Encoder Score", "Reranked Rank (Stage 2)"},
        {"Doc A (True Answer with Negation)", "Rank 6", "0.96", "Rank 1 (Promoted +5)"},
        {"Doc B (Relevant Architecture)", "Rank 1", "0.89", "Rank 2 (Slight Shift)"},
        {"Doc C (Superficial Keyword Match)", "Rank 2", "0.22", "Rank 18 (Demoted -16)"},
        {"Doc D (Irrelevant Topic)", "Rank 3", "0.14", "Rank 19 (Demoted -16)"}
    ])
    st.dataframe(score_ex_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: SELECT TOP 5 DOCUMENTS
# ==============================================================================

st.header("4. Select Top 5 Documents: Truncating the Reordered Candidate List", divider="rainbow")

st.markdown("""
Once the 20 candidates are sorted by true relevance, the pipeline truncates the list to the top 5 candidates.
""")

col_top5_1, col_top5_2 = st.columns([1, 1])

with col_top5_1:
    st.subheader("Candidate Window Slicing")
    st.markdown("""
    * **Discarding the Lower Tail**:
      * Positions 6 through 20 are discarded immediately.
      * The bottom 15 candidates represent low-relevance distractor text that survived initial vector retrieval.
    * **Preserving Candidate Headroom**:
      * The top 5 candidates advance to Step 4 for absolute relevance threshold validation.
    """)

with col_top5_2:
    st.subheader("Token Cost Savings")
    st.markdown("""
    * Passing 20 chunks to an LLM context prompt consumes ~6,000 tokens per request.
    * Truncating down to 5 chunks cuts prompt token consumption to ~1,500 tokens (a **75% reduction** in LLM inference costs!).
    """)

st.divider()

# ==============================================================================
# SECTION 5: REMOVE IRRELEVANT CONTEXT
# ==============================================================================

st.header("5. Remove Irrelevant Context: Score Cutoffs and Hallucination Prevention", divider="rainbow")

st.markdown("""
Even within the top 5 reranked candidates, some documents may still be irrelevant if the user asked an 
out-of-domain question. Step 4 enforces a strict **Relevance Quality Cutoff**.
""")

col_cut1, col_cut2 = st.columns([1, 1])

with col_cut1:
    st.subheader("The Quality Cutoff Filter")
    st.markdown("""
    * **The Cutoff Threshold (e.g. Score >= 0.65)**:
      * Every candidate in the top 5 is checked against an absolute relevance cutoff.
      * *Example*:
        * Doc 1: Score `0.94` -> **Retained**
        * Doc 2: Score `0.88` -> **Retained**
        * Doc 3: Score `0.41` -> **Discarded**
        * Doc 4: Score `0.32` -> **Discarded**
        * Doc 5: Score `0.18` -> **Discarded**
    * **The Outcome**: Only 2 pristine, high-confidence chunks enter the final LLM prompt!
    """)

with col_cut2:
    st.subheader("The Empty Context Circuit Breaker")
    st.markdown("""
    * If a user asks a question with **zero matching records** in the knowledge base:
      * All candidates score below `0.30`.
      * The filter prunes all 5 candidates, yielding an **empty context set**.
    * **System Response**:
      The orchestration layer bypasses the generator LLM and outputs a deterministic refusal:
      *"No authoritative context found for this inquiry."*
      This completely eliminates generative hallucinations!
    """)

st.divider()

# ==============================================================================
# SECTION 6: PASS FINAL CONTEXT TO LLM
# ==============================================================================

st.header("6. Pass Final Context to LLM: Formatting Context Payloads and Citations", divider="rainbow")

st.markdown("""
The final surviving high-confidence chunks are assembled into an augmented prompt template with 
explicit source citations and structural bounding.
""")

col_llm1, col_llm2 = st.columns([1, 1])

with col_llm1:
    st.subheader("Context Assembly Best Practices")
    st.markdown("""
    * **Explicit Source Delimiters**:
      * Wrap each chunk with structured tags (`[Document 1: source_title (Score: 0.94)]`).
    * **Ordering Strategy**:
      * Place the highest-scoring chunk at the **very top** or **very bottom** of the context block to optimize LLM attention heads.
    * **Citation Grounding Instruction**:
      * Instruct the LLM to attribute every claim to specific document tags:
        `"Cite sources using [Doc X] notation. If facts are not directly stated in context, state that they are unknown."`
    """)

with col_llm2:
    st.subheader("Production Prompt Assembly Pattern")
    st.code('''# Step 5: Final Augmented Prompt Construction
PROMPT_TEMPLATE = """You are an authoritative enterprise assistant.
Answer the user query strictly using the verified context below.
Cite all facts using [Source X] references.

CONTEXT:
{formatted_context_chunks}

USER QUERY: {query}
GROUNDED ANSWER:"""
''', language="python")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Production Two-Stage Pipeline with Cutoff Filtering", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python class implementing the complete 5-step production reranking pipeline 
with over-fetching, cross-encoder batch scoring, candidate truncation, and noise pruning.
""")

st.code(r'''
from typing import List, Dict, Any, Optional

class ProductionRerankingPipeline:
    """
    End-to-end 5-step retrieval and reranking pipeline:
    1. Retrieve Top 20 candidates (Stage 1)
    2. Batch Cross-Encoder scoring (Stage 2)
    3. Select Top 5 candidates
    4. Apply relevance cutoff threshold (Prune noise)
    5. Assemble structured prompt payload for LLM
    """
    def __init__(
        self,
        retriever_client: Any,
        reranker_client: Any,
        coarse_k: int = 20,
        select_k: int = 5,
        cutoff_threshold: float = 0.65
    ):
        self.retriever = retriever_client
        self.reranker = reranker_client
        self.coarse_k = coarse_k
        self.select_k = select_k
        self.cutoff_threshold = cutoff_threshold

    def run(self, query: str) -> Dict[str, Any]:
        # Step 1: Retrieve Top 20 coarse candidates
        coarse_docs = self.retriever.search(query, top_k=self.coarse_k)
        if not coarse_docs:
            return {"status": "NO_DOCS_FOUND", "context_payload": ""}

        # Step 2: Batch Cross-Encoder scoring
        pairs = [[query, doc["text"]] for doc in coarse_docs]
        scores = self.reranker.predict(pairs, batch_size=32)

        scored_pool = []
        for doc, score in zip(coarse_docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(round(score, 4))
            scored_pool.append(item)

        # Step 3: Sort descending and select top candidates
        scored_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_candidates = scored_pool[:self.select_k]

        # Step 4: Remove irrelevant context (Cutoff filter)
        surviving_chunks = [d for d in top_candidates if d["rerank_score"] >= self.cutoff_threshold]

        if not surviving_chunks:
            return {
                "status": "CONFIDENCE_TOO_LOW",
                "message": "All retrieved candidates scored below relevance cutoff.",
                "context_payload": ""
            }

        # Step 5: Format context payload for LLM
        formatted_chunks = []
        for idx, chunk in enumerate(surviving_chunks, start=1):
            formatted_chunks.append(
                f"[Source {idx}: {chunk.get('title', 'Unknown')} (Score: {chunk['rerank_score']:.2f})]\n{chunk['text']}"
            )

        return {
            "status": "SUCCESS",
            "context_payload": "\n\n".join(formatted_chunks),
            "surviving_count": len(surviving_chunks),
            "top_candidates": top_candidates
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Production Reranking Pipeline Simulator", divider="rainbow")

st.markdown("""
Simulate how the 5-step production reranking pipeline processes queries, scores candidates, 
and applies quality cutoff filtering to eliminate noise.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    scenario_choice = st.selectbox(
        "Select Pipeline Test Scenario:",
        list(ProductionRerankingPipelineSimulator.SCENARIOS.keys())
    )
    cutoff_val = st.slider("Step 4 Cutoff Threshold (Score Filter):", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    max_docs_val = st.slider("Step 5 Max Chunks to Pass to LLM:", min_value=1, max_value=5, value=2)

    pipe_out = ProductionRerankingPipelineSimulator.execute_pipeline(
        scenario_key=scenario_choice,
        cutoff_threshold=cutoff_val,
        max_final_docs=max_docs_val
    )

    st.markdown("#### 📥 Query Info")
    st.info(f"**Query**: *\"{pipe_out['query']}\"*")
    st.write(f"**Step 1 Coarse Candidates Fetched**: `{pipe_out['total_coarse_fetched']}`")
    st.write(f"**Step 3 Top Selected Window**: `{len(pipe_out['top_selected'])} candidates`")
    st.write(f"**Step 4 Passed Quality Filter**: `{len(pipe_out['passed_quality_filter'])} clean chunks`")
    st.write(f"**Step 4 Pruned Distractor Noise**: `{len(pipe_out['pruned_distractors'])} chunks discarded`")

with wb_col2:
    st.markdown("### 📄 Step 5: Assembled Context Payload for LLM")
    st.text_area(
        "Formatted Context Prompt Block:",
        value=pipe_out["formatted_prompt_context"],
        height=220
    )

    if pipe_out["passed_quality_filter"]:
        st.success(f"✅ Context successfully prepared with {len(pipe_out['passed_quality_filter'])} verified chunks!")
    else:
        st.error("🛑 Circuit Breaker Triggered: All candidates scored below cutoff. Bypassing LLM generation to prevent hallucination!")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is Step 4 (Remove Irrelevant Context) considered the most vital stage of the reranking pipeline?"):
    st.markdown("""
    **Answer**: In traditional retrieval pipelines, systems unconditionally pass a fixed number of documents (e.g., K=5) 
    to the generator LLM regardless of their actual relevance. When a user asks an unanswerable or out-of-domain question, 
    the retriever returns the closest available distractor text. 
    Because the LLM receives these distractor chunks in its context, it feels compelled to answer, triggering severe hallucinations. 
    Step 4 enforces an absolute quality cutoff score: if all candidates score below `0.65`, the system discards them and returns an empty 
    context signal, enabling the application to refuse gracefully without hallucination.
    """)

with st.expander("Q2: How does setting the candidate window to Top 20 documents in Step 1 optimize the recall-latency curve?"):
    st.markdown("""
    **Answer**: If Step 1 fetches only 5 documents, the pipeline recall is artificially capped: any relevant document ranked between 
    positions 6 and 20 in vector search is permanently lost. 
    Conversely, fetching 100 documents increases cross-encoder batch inference time by 5x (from 25ms to 130ms), blowing through sub-50ms search SLAs. 
    Empirical research on the MS-MARCO and BEIR benchmarks proves that a candidate pool of **Top 20 documents** captures over 94% of all possible 
    recall gains while keeping cross-encoder inference latency under 30 milliseconds on modern GPU hardware.
    """)

with st.expander("Q3: What are the security benefits of applying the reranking pipeline before prompt construction?"):
    st.markdown("""
    **Answer**: Reranking mitigates **Prompt Injection and Indirect Poisoning Attacks**. 
    Attackers often inject adversarial text chunks into document stores designed to fool bi-encoders with repetitive trigger keywords. 
    Because cross-encoders evaluate the full semantic coherence and context of the passage against the query, 
    adversarial chunks that lack genuine conceptual relevance receive near-zero relevance scores, causing them to be pruned in Step 4 
    before ever reaching the LLM's system prompt.
    """)

with st.expander("Q4: Why should source citations be explicitly embedded in Step 5 context payloads?"):
    st.markdown("""
    **Answer**: Structuring context chunks with explicit delimiters (`[Source 1: Title (Score: 0.94)]`) provides two key benefits:
    1. It anchors the LLM's attention mechanism to reference explicit sources, enabling verifiable footnotes in the final answer.
    2. It facilitates automated post-generation validation: evaluation frameworks (like Ragas or TruLens) parse the generated citations 
       and compute **Faithfulness** and **Context Precision** metrics directly against the identified chunks.
    """)

with st.expander("Q5: What happens when all 20 retrieved candidates score below the cutoff threshold in Step 4?"):
    st.markdown("""
    **Answer**: When all candidates fail the threshold, the pipeline activates an **Empty Context Circuit Breaker**. 
    Rather than calling the expensive generator LLM with irrelevant text, the orchestration pipeline short-circuits execution: 
    it saves LLM token costs, saves 800ms of generation latency, and immediately replies: 
    *"I do not have sufficient verified information in my knowledge base to answer this question."*
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Omission of Step 4 (The Unconditional Fixed-K Forwarding Trap)**
    * **Symptom**: Enterprise assistant hallucinates confident answers to nonsensical or out-of-domain questions.
    * **Root Cause**: The pipeline blindly passes Top 5 documents to the LLM without checking if their rerank scores are above an absolute relevance threshold.
    * **Remedy**: Always implement a score threshold filter in Step 4, pruning any candidate with a score < 0.65.
    """)

    st.error("""
    **Trap 2: Sorting Candidates Ascending Instead of Descending**
    * **Symptom**: Reranking pipeline consistently promotes the worst documents to rank #1.
    * **Root Cause**: Python's `sort()` defaults to ascending order. Forgetting `reverse=True` places the lowest relevance scores at the top of the context prompt.
    * **Remedy**: Always verify sort polarity: `candidates.sort(key=lambda x: x["score"], reverse=True)`.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Over-Populating Context with Redundant Paragraphs**
    * **Symptom**: Final prompt contains 5 paragraphs from the exact same page, crowding out other relevant document sections.
    * **Root Cause**: Reranking without document-level deduplication allows 5 slightly different chunks from the same source file to consume all available top slots.
    * **Remedy**: Enforce a maximum chunk limit per parent document (e.g., maximum 2 chunks per source document ID).
    """)

    st.warning("""
    **Trap 4: Discarding Metadata and Source Attributes During Step 5 Formatting**
    * **Symptom**: Generator LLM outputs answers without knowing which document provided which fact, preventing accurate citations.
    * **Root Cause**: Stripping metadata dictionaries and passing only raw text strings into the prompt context payload.
    * **Remedy**: Maintain document metadata through all pipeline stages and inject source titles into chunk headers.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 5 operational steps of a production reranking pipeline?")
        st.markdown("**Back**: 1. Retrieve Top 20 candidates; 2. Rerank candidates with Cross-Encoder; 3. Select Top 5 candidates; 4. Remove irrelevant context via cutoff threshold; 5. Pass final validated context to LLM.")

    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of Step 4 (Remove Irrelevant Context)?")
        st.markdown("**Back**: To prune weak matches (e.g. score < 0.65) and trigger an empty context circuit breaker when queries are out-of-domain, preventing generative LLM hallucinations.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Top 20 the standard first-stage candidate retrieval window?")
        st.markdown("**Back**: It captures ~94% of all potential recall gains while keeping cross-encoder batch tensor inference strictly under 30 milliseconds.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What is the Empty Context Circuit Breaker in RAG?")
            st.markdown("**Back**: An architectural safety gate that intercepts cases where all retrieved documents fail relevance thresholds, bypassing LLM generation and outputting an authentic refusal.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of the Production Reranking Pipeline, 5-Stage Orchestration, and Noise Pruning:
""")

chk1 = st.checkbox("I understand the complete 5-step lifecycle of an enterprise reranking pipeline.", value=True)
chk2 = st.checkbox("I can explain why over-fetching Top 20 candidates balances high recall with sub-30ms batch inference latency.", value=True)
chk3 = st.checkbox("I understand how truncating to Top 5 candidates reduces downstream LLM prompt token consumption by 75%.", value=True)
chk4 = st.checkbox("I know how the Step 4 relevance cutoff filter prunes distractor noise and prevents hallucinations.", value=True)
chk5 = st.checkbox("I understand how the empty context circuit breaker safely handles out-of-domain inquiries.", value=True)
chk6 = st.checkbox("I know how to format structured context payloads with source citations for downstream LLM synthesis.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Phenomenal! You have mastered the Production Reranking Pipeline and Enterprise Noise Pruning architectures.")
