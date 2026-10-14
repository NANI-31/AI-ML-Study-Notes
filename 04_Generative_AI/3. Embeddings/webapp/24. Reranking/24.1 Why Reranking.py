import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="24.1 Why Reranking - Embeddings Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("24.1 Why Reranking: The Two-Stage Funnel, Bi-Encoders vs. Cross-Encoders")
st.caption("Chapter 24: Reranking | Domain: 3. Vector Embeddings")
st.markdown("""
In naive RAG architectures, developers attempt to find the top 5 chunks directly using a single vector search call.
However, **Bi-Encoder vector search faces a fundamental mathematical limitation**:
* In a Bi-Encoder, the query and document are encoded **separately in total isolation**.
* Query tokens never directly interact with document tokens through transformer self-attention layers!
* As a result, subtle linguistic nuances—negations (*"with approval"* vs *"without approval"*), directional relationships, and fine-grained logic—often get lost in high-dimensional vector space.

To solve this without sacrificing speed, the industry universally adopts the **Two-Stage Retrieval Funnel**:
1. **Stage 1: Fast Candidate Retrieval (Bi-Encoder / Vector Search)**:
   * Casts a wide net over millions of documents in sub-5ms to extract the **Top 50 to 100 candidate chunks** with high recall.
2. **Stage 2: Precision Reranking (Cross-Encoder / Reranker)**:
   * Feeds the query and candidate document **together into a single transformer model**, allowing every query token to attend directly to every document token (**Full Cross-Attention**).
   * Re-scores and re-ranks the 50 candidates, distilling them into the **Top 3 to 5 pristine chunks** for the LLM prompt.

This lesson explores the architecture and justification of two-stage retrieval systems.
""")

st.divider()

# Section 1: The Two-Stage Funnel Architecture
st.header("1. Bi-Encoders vs. Cross-Encoders & The Two-Stage Funnel", divider="rainbow")

col_two1, col_two2 = col_two1, col_two2 = st.columns([1, 1])

with col_two1:
    st.markdown("""
    ### Bi-Encoders vs. Cross-Encoders
    * **Bi-Encoder (Stage 1 - Candidate Retrieval)**:
      * Query and document are projected into separate embedding vectors: `v_q = Model(q)` and `v_d = Model(d)`.
      * Similarity is computed via dot product: `Sim = dot(v_q, v_d)`.
      * **Speed**: Blazing fast ($O(1)$ with HNSW index).
      * **Limitation**: Zero cross-token attention; misses subtle syntactic nuances.
    * **Cross-Encoder (Stage 2 - Precision Reranking)**:
      * Query and document are concatenated into a single input sequence: `[CLS] Query [SEP] Document [SEP]`.
      * Every query token attends to every document token across all 12+ transformer attention layers!
      * **Precision**: State-of-the-art accuracy; understands negations, qualifying clauses, and word order.
      * **Limitation**: Computationally heavy ($O(N)$ forward passes); impossible to run on millions of documents, but blazing fast on 50 candidates ($< 25\\text{ms}$).
    """)

with col_two2:
    st.markdown("### The Two-Stage Retrieval Funnel")
    funnel_dot = """
    digraph TwoStageFunnel {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Corpus [label="Enterprise Corpus\\n(10,000,000 Chunks)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_stage1 {
            label="Stage 1: Broad Retrieval (Bi-Encoder)"; style="dashed"; color="#3B82F6";
            BiEncoder [label="HNSW Vector Search / Hybrid\\nEvaluates millions in 3ms", fillcolor="#DBEAFE"];
            Top50 [label="Top 50 Candidate Chunks\\nHigh Recall (Captures the needle)", fillcolor="#BFDBFE", color="#1D4ED8", style="filled,bold"];
            BiEncoder -> Top50;
        }

        subgraph cluster_stage2 {
            label="Stage 2: Precision Reranking (Cross-Encoder)"; style="dashed"; color="#10B981";
            CrossEnc [label="Full Cross-Attention Reranker\\n(Cohere Rerank / BGE-Reranker)\\nScores 50 pairs in 20ms", fillcolor="#D1FAE5"];
            Top5 [label="Top 5 Pristine Chunks\\nMaximum Precision (No Distractors)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            CrossEnc -> Top5;
        }

        LLMContext [label="LLM Generative Context\\n100% Grounded Answer", fillcolor="#FEF08A", color="#D97706", style="filled,bold"];

        Corpus -> BiEncoder;
        Top50 -> CrossEnc;
        Top5 -> LLMContext;
    }
    """
    st.graphviz_chart(funnel_dot, use_container_width=False)

st.markdown("### Architectural Trade-off Comparison")
st.markdown("""
| Architectural Property | Stage 1: Bi-Encoder (Dense Vector Search) | Stage 2: Cross-Encoder (Reranker) |
| :--- | :--- | :--- |
| **Input Format** | Query and Document encoded separately | Concatenated pair: `[CLS] Query [SEP] Doc` |
| **Attention Mechanism** | Intra-text only (No cross-attention) | **Full Cross-Attention across all tokens** |
| **Pre-computable Embeddings?**| **Yes (Pre-indexed in Vector Database)** | **No (Requires live query pair inference)** |
| **Complexity on $N$ Documents**| $O(\\log N)$ (Sub-linear HNSW graph) | $O(N)$ (Full neural forward passes) |
| **Optimal Search Scope** | 100,000 to Billions of vectors | **Top 20 to 100 candidate chunks** |
| **Latency** | 2 - 5 ms | 15 - 35 ms |
""")

st.divider()

# Section 2: Production Code: Two-Stage Reranking Pipeline
st.header("2. Production Code: Two-Stage Funnel Pipeline Simulator", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to construct a two-stage retrieval pipeline: generating a broad candidate pool using dense similarity, followed by Cross-Encoder scoring to re-rank the top items.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class TwoStageRetrievalPipeline:
    """
    Implements the canonical Two-Stage Retrieval Funnel:
    - Stage 1: Bi-Encoder ANN candidate retrieval (Broad recall)
    - Stage 2: Cross-Encoder cross-attention reranking (Precision)
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        self.corpus: List[Dict[str, Any]] = []

    def mock_bi_encoder(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.strip().lower())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def mock_cross_encoder_score(self, query: str, doc_text: str) -> float:
        """
        Simulates full cross-attention score:
        Evaluates exact phrase matching, negations, and syntactic nuance.
        """
        q_words = query.lower().split()
        d_words = doc_text.lower().split()
        
        # Cross-Encoder attends directly to word order and qualifiers:
        score = 0.20
        for i in range(len(q_words) - 1):
            bigram = f"{q_words[i]} {q_words[i+1]}"
            if bigram in doc_text.lower():
                score += 0.35
                
        # Heavy penalty if doc contradicts query negations
        if "without" in query.lower() and "with prior" in doc_text.lower():
            score -= 0.40
        if "without" in query.lower() and "without prior" in doc_text.lower():
            score += 0.50

        return max(0.01, min(0.99, score))

    def retrieve_and_rerank(
        self,
        query: str,
        stage1_candidates_count: int = 20,
        stage2_final_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        q_vec = self.mock_bi_encoder(query)

        # Stage 1: Fast Bi-Encoder Retrieval
        stage1_scored = []
        for doc in self.corpus:
            sim = float(np.dot(doc["vector"], q_vec))
            stage1_scored.append((doc, sim))
        stage1_scored.sort(key=lambda x: x[1], reverse=True)
        candidates = stage1_scored[:stage1_candidates_count]

        # Stage 2: Precision Cross-Encoder Reranking
        reranked = []
        for doc, bi_score in candidates:
            cross_score = self.mock_cross_encoder_score(query, doc["text"])
            doc_data = doc.copy()
            doc_data["bi_encoder_score"] = bi_score
            doc_data["rerank_score"] = cross_score
            reranked.append(doc_data)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:stage2_final_top_k]
''', language="python")

st.divider()

# Section 3: Interactive Two-Stage Funnel & Reranker Studio
st.header("3. Interactive Workbench: The Reranker Flip Studio", divider="rainbow")

st.markdown("""
Observe the **Reranker Flip Effect**:
In Stage 1 (Bi-Encoder), candidate documents that contain subtle negations or specific qualifiers are buried at lower ranks.
When the **Cross-Encoder Reranker** is activated, full cross-attention flips the ranking, bringing the true answer directly to **Rank 1**!
""")

sample_funnel_docs = [
    {
        "id": "doc_01",
        "title": "Overtime Approval SOP",
        "text": "Non-exempt employees may work overtime without prior manager approval strictly during declared site emergencies.",
        "bi_rank": 6,
        "is_true_answer": True
    },
    {
        "id": "doc_02",
        "title": "General Overtime Policy",
        "text": "All overtime hours worked by hourly employees must be approved in advance by the department manager.",
        "bi_rank": 1,
        "is_true_answer": False
    },
    {
        "id": "doc_03",
        "title": "Exempt Employee Hours",
        "text": "Salaried exempt employees do not qualify for overtime pay regardless of hours worked per week.",
        "bi_rank": 2,
        "is_true_answer": False
    },
    {
        "id": "doc_04",
        "title": "Manager Timesheet Review",
        "text": "Department managers must review and digitally sign off on electronic bi-weekly timesheets by Monday noon.",
        "bi_rank": 3,
        "is_true_answer": False
    },
    {
        "id": "doc_05",
        "title": "Meal Breaks and Rest Periods",
        "text": "Non-exempt hourly workers are entitled to an unpaid thirty-minute meal break for every five continuous hours worked.",
        "bi_rank": 4,
        "is_true_answer": False
    }
]

col_r1, col_r2 = st.columns([1, 1])

with col_r1:
    st.subheader("1. Test Policy Query")
    test_q = st.text_input(
        "User Question:",
        value="Can employees work overtime without prior manager approval?"
    )

    st.subheader("2. Reranker Control")
    enable_reranker = st.toggle("Enable Stage 2 Cross-Encoder Reranker", value=True)
    k_prompt = st.slider("Passages to send to LLM ($K$):", min_value=1, max_value=4, value=2)

    st.markdown("#### Notice the Nuance:")
    st.info("The query asks about working overtime **WITHOUT** prior approval. Notice that `doc_01` specifically answers this for declared emergencies, but Bi-Encoder vector search ranked `doc_02` at Rank 1 because doc 2 shares general overtime words!")

with col_r2:
    st.subheader("3. Ranked Context Delivered to LLM")

    if not enable_reranker:
        st.warning("⚠️ **Stage 1 Only (Naive Bi-Encoder)**: The true answer is buried at Rank 6! The LLM will receive the wrong policy.")
        # Sort by bi_rank
        stage1_sorted = sorted(sample_funnel_docs, key=lambda x: x["bi_rank"])
        for rank, item in enumerate(stage1_sorted[:k_prompt], 1):
            with st.expander(f"Rank {rank} | {item['id']} ({item['title']})", expanded=True):
                st.markdown(f"> {item['text']}")
                if item["is_true_answer"]:
                    st.success("Target Answer")
                else:
                    st.error("Incomplete Distractor (Requires approval)")
    else:
        st.success("✅ **Stage 2 Active (Cross-Encoder Re-ranked)**: Cross-attention detected the qualifier 'without prior manager approval' and flipped `doc_01` to **Rank 1**!")
        
        # Simulate reranker scores
        reranked_data = []
        for item in sample_funnel_docs:
            if item["is_true_answer"]:
                r_score = 0.94
            elif item["id"] == "doc_02":
                r_score = 0.42 # Contradicts query intent
            else:
                r_score = 0.15
            reranked_data.append((item, r_score))

        reranked_data.sort(key=lambda x: x[1], reverse=True)

        for rank, (item, sc) in enumerate(reranked_data[:k_prompt], 1):
            with st.expander(f"Rank {rank} | {item['id']} | Rerank Score: {sc:.2f}", expanded=True):
                st.markdown(f"> {item['text']}")
                if item["is_true_answer"]:
                    st.success("🎯 TRUE ANSWER PROMOTED TO RANK 1")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Reranking Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why can't we use Cross-Encoder models directly to search an entire vector database of 10 million documents?**
    * **Answer**:
      * In a Bi-Encoder, vectors are **pre-computed offline**. A runtime search evaluates simple dot products over an HNSW graph in sub-5ms.
      * In a Cross-Encoder, the query and document must be concatenated and fed through all 12+ layers of a multi-million parameter transformer.
      * Running a Cross-Encoder forward pass over 10,000,000 documents for a single query would take **hours of GPU compute and cost hundreds of dollars per query**!
      * The Two-Stage Funnel solves this: Bi-Encoders narrow the search space from 10,000,000 to 50 in 3ms; Cross-Encoders re-rank the 50 in 20ms.

    **Q2: What is the primary difference in attention mechanics between Bi-Encoders and Cross-Encoders?**
    * **Answer**:
      * **Bi-Encoder**: Computes self-attention within the query alone: `Attn(Q, Q)` and within the document alone: `Attn(D, D)`. There is **zero cross-attention** between query and document tokens.
      * **Cross-Encoder**: Computes joint cross-attention across the combined sequence: `Attn([Q; D], [Q; D])`. Every query token (e.g. *"without"*) directly attends to every document token (e.g. *"emergency"*), capturing subtle causal and conditional dependencies.

    **Q3: What is the optimal number of candidate documents to pass from Stage 1 into the Stage 2 Reranker?**
    * **Answer**: Empirically, between **20 and 100 candidate documents** (typically 30 to 50).
    * Passing fewer than 20 risks missing the needle document if the Bi-Encoder scored it at rank 25.
    * Passing more than 100 yields diminishing returns on recall while linearly increasing Cross-Encoder inference latency (e.g. 100ms+), degrading user experience.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Expecting Rerankers to Rescue Missing Documents**:
      * *Issue*: If the Bi-Encoder fails to retrieve the correct document into the top 50 candidates, expecting the Reranker to find it.
      * *Consequence*: The Reranker can only reorder the candidate documents it is given; it cannot summon documents from the database that were missed in Stage 1!
      * *Fix*: Maximize Stage 1 Recall using Hybrid Search (BM25 + Vectors) and an ample candidate pool ($K=50$).

    * **Trap 2: Feeding Entire Long Documents into Cross-Encoders**:
      * *Issue*: Passing 3,000-word documents directly into a Cross-Encoder.
      * *Consequence*: Cross-Encoder attention complexity scales quadratically $O(L^2)$ with token sequence length, causing latency to explode from 20ms to 800ms.
      * *Fix*: Only pass concise chunks (250-400 tokens) to the Cross-Encoder.

    * **Trap 3: Reranking in the Synchronous User Path on Large Batches**:
      * *Issue*: Reranking 500 documents on every keystroke in search autocomplete.
      * *Consequence*: UI freezes with 2-second lag spikes.
      * *Fix*: Restrict reranking to explicit form submissions or cap candidate pools at 25 items for interactive workflows.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Two-Stage Retrieval Funnel**
      * *Definition*: The architectural pattern pairing fast, broad candidate retrieval (Bi-Encoder) with slow, high-precision re-scoring (Cross-Encoder).
    * **Flashcard 2: Full Cross-Attention**
      * *Definition*: Joint transformer self-attention over concatenated query and document tokens allowing fine-grained syntactic interaction.
    * **Flashcard 3: The Reranker Flip**
      * *Definition*: The phenomenon where a Cross-Encoder promotes a subtly qualified document from a lower rank up to Rank 1 by resolving negations.
    * **Flashcard 4: Candidate Pool Depth ($K_{cand}$)**
      * *Definition*: The number of items retrieved in Stage 1 (typically 30-50) passed into the Stage 2 reranker.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Why Reranking:

- [ ] I can articulate the fundamental limitation of Bi-Encoder dot-product matching.
- [ ] I understand how Full Cross-Attention resolves negations and qualifying clauses.
- [ ] I can explain why Cross-Encoders cannot be run across millions of database vectors.
- [ ] I know the optimal candidate pool depth to pass between Stage 1 and Stage 2.
- [ ] I understand why Stage 1 recall bounds the maximum possible performance of Stage 2.
""")
