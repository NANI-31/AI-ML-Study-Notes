import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="24.3 Retrieval Pipeline - Embeddings Curriculum",
    page_icon="🚀",
    layout="wide"
)

# Title & Overview
st.title("24.3 The Complete Reranked Retrieval Pipeline: From Query to Grounded Generation")
st.caption("Chapter 24: Reranking | Domain: 3. Vector Embeddings")
st.markdown("""
A production-grade Retrieval-Augmented Generation (RAG) system integrates embedding models, vector databases, and rerankers into an **automated multi-stage pipeline**:

```
Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Top 20 Documents
  ↓
Reranker
  ↓
Top 5 Documents
  ↓
LLM
```

Each stage operates under strict latency budgets and mathematical contracts:
1. **Embedding**: Fast vector projection ($< 25\\text{ms}$).
2. **Vector Search**: Sub-millisecond ANN traversal over millions of vectors ($< 5\\text{ms}$).
3. **Top 20 Documents**: Casting an ample candidate net to guarantee high recall.
4. **Reranker**: Cross-encoder precision scoring to eliminate distractor chunks ($< 30\\text{ms}$).
5. **Top 5 Documents**: Feeding only the most informative, deduplicated context to the LLM.
6. **LLM**: Synthesizing a grounded, hallucination-free answer with source citations.

This lesson explores how to assemble, monitor, and optimize this canonical 7-stage retrieval pipeline.
""")

st.divider()

# Section 1: The Canonical 7-Stage Retrieval Pipeline
st.header("1. The Canonical 7-Stage Pipeline Architecture", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### Pipeline Stage Breakdown & Latency Budgets
    * **1. User Query**: Natural language question or search prompt.
    * **2. Query Embedding (15 - 25 ms)**:
      * Encodes user query using the collection's Bi-Encoder model.
    * **3. Vector Search (2 - 5 ms)**:
      * Navigates HNSW / IVF index to retrieve the nearest neighbor candidate IDs.
    * **4. Top 20 Candidates (< 2 ms)**:
      * Hydrates the top 20 candidate chunk payloads from document storage.
    * **5. Cross-Encoder Reranker (15 - 30 ms)**:
      * Evaluates full cross-attention on all 20 `(Query, Chunk)` pairs.
      * Discards chunks with relevance scores below confidence thresholds ($< 0.35$).
    * **6. Top 5 Clean Chunks (< 1 ms)**:
      * Assembles the top 3-5 pristine passages into an augmented prompt template.
    * **7. LLM Generation (400 - 1500 ms)**:
      * Generates final answer with citations. Streamed via SSE to the user.
    """)

with col_pipe2:
    st.markdown("### Exact Pipeline Progression Flowchart")
    pipe_dot = """
    digraph CanonicalPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Q [label="1. User Query\\n'How to configure S3 encryption?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        E [label="2. Query Embedding\\n(Bi-Encoder Vector d=1536)", fillcolor="#DBEAFE"];
        VS [label="3. Vector Search\\n(HNSW ANN Index)", fillcolor="#BFDBFE"];
        Top20 [label="4. Top 20 Candidates\\nHigh Recall Pool", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
        Rerank [label="5. Cross-Encoder Reranker\\n(Full Cross-Attention)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        Top5 [label="6. Top 5 Documents\\nHigh Precision Filtered", fillcolor="#A7F3D0"];
        LLM [label="7. Generative LLM\\nGrounded Answer + Citations", fillcolor="#FEF08A", color="#D97706", style="filled,bold"];

        Q -> E [label="Tokenize"];
        E -> VS [label="Dense vector"];
        VS -> Top20 [label="ANN scan"];
        Top20 -> Rerank [label="Candidate batch"];
        Q -> Rerank [label="Query string"];
        Rerank -> Top5 [label="Re-score & prune"];
        Top5 -> LLM [label="Augmented prompt"];
        Q -> LLM [label="User intent"];
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=False)

st.markdown("### Latency & Error Budget Across All 7 Stages")
st.markdown("""
| Step | Pipeline Stage | Typical Latency | SLA Target | Primary Failure Mode | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | User Query | < 1 ms | < 5 ms | Ambiguous / underspecified phrasing. | Query expansion / HyDE. |
| **2** | Query Embedding | 15 - 25 ms | < 30 ms | API network latency spike. | Local ONNX embedding / caching. |
| **3** | Vector Search | 2 - 5 ms | < 8 ms | Cold-start cache miss on HNSW graph. | Memory-mapped pre-warmed RAM index. |
| **4** | Top 20 Candidates | 1 - 2 ms | < 5 ms | High payload deserialization overhead. | In-memory key-value docstore. |
| **5** | Reranker | 15 - 30 ms | < 40 ms | GPU memory exhaustion on long batches. | Cap candidate batch size to 20-30 chunks. |
| **6** | Top 5 Clean Chunks | < 1 ms | < 2 ms | Duplicate sibling chunks from same doc. | Deduplicate parent document IDs. |
| **7** | LLM Generation | 400 - 1500 ms | < 2000 ms | Hallucination on contradictory context. | Grounded system prompt + citations. |
""")

st.divider()

# Section 2: Production Code: End-to-End Orchestrator
st.header("2. Production Code: End-to-End Pipeline Orchestrator with Telemetry", divider="rainbow")

st.markdown("""
The production utility below implements the full 7-stage pipeline orchestrator: tracking step-by-step latency telemetry, executing two-stage retrieval, and synthesizing an augmented LLM prompt.
""")

st.code(r'''
import time
import numpy as np
from typing import List, Dict, Any, Tuple

class CompleteRerankedRAGPipeline:
    """
    Orchestrates the canonical 7-stage RAG pipeline:
    Query -> Embedding -> Vector Search -> Top 20 -> Reranker -> Top 5 -> LLM
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        self.corpus: List[Dict[str, Any]] = []

    def mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.strip().lower())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def mock_rerank(self, query: str, doc_text: str) -> float:
        """Simulates Cross-Encoder cross-attention scoring."""
        q_words = set(query.lower().split())
        d_words = set(doc_text.lower().split())
        overlap = len(q_words.intersection(d_words)) / max(1, len(q_words))
        return min(0.99, max(0.01, 0.3 * overlap + np.random.uniform(0.1, 0.7)))

    def run_pipeline(
        self,
        query: str,
        stage1_k: int = 20,
        stage2_k: int = 5,
        min_rerank_threshold: float = 0.30
    ) -> Dict[str, Any]:
        telemetry = {}
        t_start = time.perf_counter()

        # Step 2: Query Embedding
        t0 = time.perf_counter()
        q_vec = self.mock_embed(query)
        telemetry["step_2_embed_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3 & 4: Vector Search (Top 20 Candidates)
        t0 = time.perf_counter()
        scored_stage1 = []
        for doc in self.corpus:
            sim = float(np.dot(doc["vector"], q_vec))
            scored_stage1.append((doc, sim))
        scored_stage1.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_stage1[:stage1_k]
        telemetry["step_3_4_vector_search_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: Cross-Encoder Reranking
        t0 = time.perf_counter()
        reranked_pool = []
        for doc, bi_score in top_candidates:
            cross_score = self.mock_rerank(query, doc["text"])
            doc_entry = doc.copy()
            doc_entry["stage1_score"] = bi_score
            doc_entry["rerank_score"] = cross_score
            reranked_pool.append(doc_entry)
        reranked_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        telemetry["step_5_reranker_ms"] = (time.perf_counter() - t0) * 1000

        # Step 6: Filter Top 5 Clean Chunks
        t0 = time.perf_counter()
        final_top_5 = [
            d for d in reranked_pool[:stage2_k]
            if d["rerank_score"] >= min_rerank_threshold
        ]
        telemetry["step_6_filter_ms"] = (time.perf_counter() - t0) * 1000

        # Step 7: Prompt Synthesis & Simulated LLM
        t0 = time.perf_counter()
        context_str = "\n".join([f"[{d['id']}]: {d['text']}" for d in final_top_5])
        augmented_prompt = f"CONTEXT:\n{context_str}\n\nQUERY:\n{query}\n\nANSWER:"
        # Simulated LLM generation
        answer = f"Synthesized answer grounded on {len(final_top_5)} verified context passages."
        telemetry["step_7_llm_ms"] = (time.perf_counter() - t0) * 1000

        telemetry["total_pipeline_ms"] = (time.perf_counter() - t_start) * 1000
        telemetry["final_docs"] = final_top_5
        telemetry["answer"] = answer
        telemetry["prompt"] = augmented_prompt

        return telemetry
''', language="python")

st.divider()

# Section 3: Interactive End-to-End Pipeline Stepper
st.header("3. Interactive Workbench: 7-Stage Pipeline Stepper & Latency Waterfall", divider="rainbow")

st.markdown("""
Step sequentially through each stage of the canonical retrieval pipeline.
Observe data representations and track per-stage latency budgets.
""")

class PipelineStepperEngine:
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.strip().lower())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

stepper_engine = PipelineStepperEngine()

demo_kb = [
    {"id": "doc_s3_01", "text": "Amazon S3 server-side encryption with AWS KMS keys (SSE-KMS) provides an audit trail of who used a key and when.", "is_key": True},
    {"id": "doc_s3_02", "text": "Default encryption for S3 buckets applies SSE-S3 automatically to all new objects with 256-bit AES encryption.", "is_key": True},
    {"id": "doc_s3_03", "text": "Customer-Provided Keys (SSE-C) allow clients to manage their own encryption keys while S3 manages encryption and decryption.", "is_key": True},
    {"id": "doc_cld_04", "text": "Bucket policies restrict access to objects based on IAM user roles, source IP ranges, or VPC endpoints.", "is_key": False},
    {"id": "doc_cld_05", "text": "S3 object lock prevents objects from being deleted or overwritten for a fixed retention period.", "is_key": False},
    {"id": "doc_cld_06", "text": "Glacier Flexible Archive retrieval options include Expedited (1-5 min), Standard (3-5 hours), and Bulk (5-12 hours).", "is_key": False},
    {"id": "doc_cld_07", "text": "Cross-region replication asynchronously copies objects across buckets in different AWS regions.", "is_key": False},
    {"id": "doc_cld_08", "text": "Amazon CloudFront distributes cached static assets using global edge points of presence.", "is_key": False}
]

col_st1, col_st2 = st.columns([1, 1])

with col_st1:
    st.subheader("1. Pipeline Input Setup")
    user_query = st.text_input(
        "User Query (Stage 1):",
        value="What encryption options are available for Amazon S3 buckets?"
    )

    stage_select = st.select_slider(
        "Active Stage to Inspect:",
        options=[
            "Stage 1: User Query",
            "Stage 2: Query Embedding",
            "Stage 3 & 4: Vector Search (Top 20)",
            "Stage 5: Cross-Encoder Reranker",
            "Stage 6: Top 5 Clean Chunks",
            "Stage 7: Grounded LLM Answer"
        ],
        value="Stage 5: Cross-Encoder Reranker"
    )

with col_st2:
    st.subheader(f"2. Stage Inspection: {stage_select.split(':')[1].strip()}")

    if "Stage 1" in stage_select:
        st.info("Stage 1: User Query Ingestion")
        st.code(f"Query: \"{user_query}\"")
        st.caption("Input validated, sanitized, and passed to vectorizer.")

    elif "Stage 2" in stage_select:
        st.info("Stage 2: Query Embedding Generation (Bi-Encoder)")
        q_vec = stepper_engine.embed(user_query)
        st.markdown(f"**Dense Vector (First 8 of {len(q_vec)} dimensions):**")
        st.code(str([round(float(x), 4) for x in q_vec[:8]]) + " ...")
        st.caption("Latency: ~18 ms (OpenAI API / Local BGE forward pass).")

    elif "Stage 3" in stage_select:
        st.info("Stage 3 & 4: HNSW Vector Search (Top Candidates)")
        st.markdown("**Retrieved Candidate Chunks (Broad Recall Pool):**")
        q_vec = stepper_engine.embed(user_query)
        stage1_scored = []
        for d in demo_kb:
            sim = float(np.dot(stepper_engine.embed(d["text"]), q_vec))
            stage1_scored.append((d, sim))
        stage1_scored.sort(key=lambda x: x[1], reverse=True)
        
        df_cands = pd.DataFrame([
            {"Rank": f"#{i+1}", "Doc ID": item[0]["id"], "Cosine Sim": f"{item[1]:.4f}", "Snippet": item[0]["text"][:60] + "..."}
            for i, item in enumerate(stage1_scored[:6])
        ])
        st.dataframe(df_cands, use_container_width=True)
        st.caption("Latency: ~3.5 ms (HNSW graph nearest neighbor traversal).")

    elif "Stage 5" in stage_select:
        st.info("Stage 5: Cross-Encoder Precision Reranking")
        st.markdown("**Cross-Attention Scores (Query <-> Passage Pairs):**")
        reranked = [
            (demo_kb[0], 0.96, "SSE-KMS audit key matches query encryption intent."),
            (demo_kb[1], 0.94, "Default SSE-S3 encryption matches query directly."),
            (demo_kb[2], 0.91, "SSE-C customer-managed keys matches encryption options."),
            (demo_kb[3], 0.38, "Bucket policy security distractor."),
            (demo_kb[4], 0.22, "Object lock immutability distractor."),
            (demo_kb[5], 0.12, "Glacier archive tier distractor.")
        ]
        df_rr = pd.DataFrame([
            {"Rerank Rank": f"#{i+1}", "Doc ID": r[0]["id"], "Cross-Encoder Score": f"{r[1]:.2f}", "Diagnostic": r[2]}
            for i, r in enumerate(reranked)
        ])
        st.dataframe(df_rr, use_container_width=True)
        st.caption("Latency: ~24 ms (Full cross-attention scoring across candidate pairs).")

    elif "Stage 6" in stage_select:
        st.info("Stage 6: Top 5 Filtered Context Delivery")
        st.success("Extracted Top 3 verified passages (Rerank Score >= 0.35). Pruned all distractors!")
        for idx in range(3):
            with st.expander(f"Prompt Context [{demo_kb[idx]['id']}]", expanded=True):
                st.markdown(f"> {demo_kb[idx]['text']}")
        st.caption("Latency: < 1 ms (Context prompt assembly).")

    else:
        st.info("Stage 7: Grounded LLM Response with Citations")
        aug_response = """Based on the verified documentation [doc_s3_01, doc_s3_02, doc_s3_03], Amazon S3 provides three primary server-side encryption options:

1. **SSE-S3**: Default server-side encryption managed with Amazon S3 keys using 256-bit AES [doc_s3_02].
2. **SSE-KMS**: Server-side encryption with AWS Key Management Service providing key usage audit logging [doc_s3_01].
3. **SSE-C**: Server-side encryption with customer-provided keys where the customer manages keys and S3 manages encryption [doc_s3_03]."""
        st.success(aug_response)
        st.caption("Latency: ~650 ms (LLM generation with 100% citation grounding).")

    # Waterfall Latency Chart
    st.markdown("#### End-to-End Latency Waterfall Budget")
    df_lat = pd.DataFrame({
        "Stage": ["2. Embed", "3. ANN Search", "5. Rerank", "7. LLM Generation"],
        "Latency (ms)": [20, 4, 25, 650]
    })
    st.bar_chart(df_lat.set_index("Stage"))

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Pipeline Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do you choose the candidate pool size for Stage 1 ($K=20$) versus the final prompt pool size ($K=5$) in a production RAG pipeline?**
    * **Answer**:
      * **Stage 1 Pool Size ($K=20$ to $50$)**: Must be large enough to guarantee that the true answer passage is captured even if the Bi-Encoder vector model ranks it somewhat low due to synonym or phrasing differences.
      * **Stage 2 Prompt Pool Size ($K=3$ to $5$)**: Must be small and concentrated to fit comfortably within the LLM prompt budget, minimize generation cost, and prevent the **Lost-in-the-Middle** hallucination effect.
      * Setting Stage 1 to 20-50 and Stage 2 to 3-5 provides the ideal balance between high recall and high precision.

    **Q2: What is 'Confidence Threshold Pruning' in the Reranker stage?**
    * **Answer**: If a user asks a question completely outside the corporate knowledge base (e.g. *"What is the recipe for chocolate cake?"*), Stage 1 will still return 20 documents (the closest irrelevant ones).
    * If those 20 documents are passed to the LLM, it may hallucinate an answer.
    * A **Confidence Threshold** (e.g. `Rerank Score >= 0.35`) discards all candidate chunks if none meet the minimum relevance bar. If zero chunks survive, the pipeline short-circuits and immediately replies: *"I do not have sufficient information in the knowledge base."* without calling the LLM!

    **Q3: Where does the majority of latency accumulate in this 7-stage pipeline and how is it optimized?**
    * **Answer**:
      * Stages 1–6 (Embedding, Search, Filtering, Reranking) typically take **less than 60 ms combined**.
      * Stage 7 (LLM Generation) consumes **80% to 90% of total latency** (500ms to 2000ms).
      * **Optimization**: Use token streaming via Server-Sent Events (SSE) so the user sees the first token in ~200ms, and use smaller, faster generative models (GPT-4o-mini, Claude 3.5 Haiku, Llama 3 8B) for simple extraction tasks.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Passing All Stage 1 Candidates to the LLM Without Pruning**:
      * *Issue*: Retrieving 20 chunks in Stage 1, reranking them, and then stuffing all 20 chunks into the LLM prompt.
      * *Consequence*: The bottom 15 chunks are distractor noise that increases prompt cost and induces hallucinations.
      * *Fix*: Always prune the candidate list to the top 3-5 highest scoring items.

    * **Trap 2: Cascading Timeout Failures on Reranker Latency Spikes**:
      * *Issue*: Setting a tight 100ms total API timeout when the reranker GPU experiences a momentary cold-start spike (120ms).
      * *Consequence*: The API aborts and returns a 504 error to the user.
      * *Fix*: Implement fallback logic: if the Reranker times out, fall back immediately to the Stage 1 Bi-Encoder Top-3 ranking.

    * **Trap 3: Uncached Repeated Queries Draining Pipeline Capacity**:
      * *Issue*: Re-running all 7 stages every time a user refreshes their browser.
      * *Consequence*: Saturated GPU workers and unnecessary API bills.
      * *Fix*: Deploy a Semantic Query Cache in front of Stage 1.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: The 7-Stage Retrieval Pipeline**
      * *Definition*: The industry-standard sequence: Query -> Embedding -> ANN Search -> Top 20 -> Reranker -> Top 5 -> LLM Answer.
    * **Flashcard 2: Confidence Threshold Pruning**
      * *Definition*: Discarding candidates whose Cross-Encoder score is below a threshold (e.g. 0.35) to prevent hallucinations on out-of-domain questions.
    * **Flashcard 3: Latency Waterfall**
      * *Definition*: The cumulative latency timeline of a RAG query, dominated (80-90%) by LLM token generation.
    * **Flashcard 4: Graceful Reranker Fallback**
      * *Definition*: The failover mechanism that bypasses the reranker and delivers Stage 1 top results directly to the LLM if the reranker endpoint times out.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of the Retrieval Pipeline:

- [ ] I can trace all 7 stages of the canonical reranked retrieval pipeline in order.
- [ ] I know how to allocate latency budgets across embedding, ANN search, reranking, and generation.
- [ ] I understand how Confidence Threshold Pruning prevents hallucinations on irrelevant queries.
- [ ] I know why Stage 1 candidates must be pruned down to 3-5 chunks before LLM injection.
- [ ] I can design a graceful failover mechanism if the reranker endpoint experiences a timeout.
""")
