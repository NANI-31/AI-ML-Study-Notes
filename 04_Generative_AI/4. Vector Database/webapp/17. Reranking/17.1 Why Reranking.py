import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="17.1 Why Reranking - Vector DB Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("17.1 Why Reranking? The Bi-Encoder Bottleneck & The Two-Stage Funnel")
st.caption("Chapter 17: Reranking | Domain: 4. Vector Database")
st.markdown("""
In vector search engineering, a common misconception is that upgrading to a larger embedding model will solve all retrieval precision problems. 
However, all standard vector databases share a fundamental mathematical limitation: **The Bi-Encoder Bottleneck**.

* **Bi-Encoders (Embedding Models)** compress an entire 500-word passage into a single fixed-length vector *before* the query is even known. 
  They are fundamentally incapable of modeling fine-grained, token-to-token cross-attention between query words and document words.
* As a result, vector similarity search provides **high recall but mediocre precision**: the correct answer is almost always in the top 20 results, 
  but frequently sits at Rank 7 or Rank 12 rather than Rank 1!

**Reranking** solves this by implementing the industry-standard **Two-Stage Retrieval Funnel**:
1. **Stage 1 (Coarse Fast Retrieval)**: Use the vector database (HNSW) to retrieve a broad candidate pool of the top 20–50 passages in **1–2 milliseconds**.
2. **Stage 2 (Fine-Grained Cross-Attention)**: Pass the top candidates to a **Cross-Encoder Reranker**, which evaluates all-to-all cross-attention between query tokens and document tokens, reordering the candidates with near-human precision in **15–25 milliseconds**.

In this lesson, we dissect the bi-encoder bottleneck, explore the two-stage funnel architecture, examine empirical precision gains, 
and test an interactive Two-Stage Funnel Simulator.
""")

st.divider()

# Section 1: The Bi-Encoder Bottleneck vs Cross-Encoder Precision
st.header("1. The Bi-Encoder Bottleneck vs Cross-Encoder Precision", divider="rainbow")

col_bott1, col_bott2 = col_bott1, col_bott2 = st.columns([1.1, 0.9])

with col_bott1:
    st.markdown("""
    ### Why Bi-Encoders Cannot Model Token-Level Nuance
    
    * **Bi-Encoder Architecture (Vector Search)**:
      * Maps Query `Q` -> `vec(Q)` via Transformer 1.
      * Maps Document `D` -> `vec(D)` via Transformer 2 (or shared weights).
      * Evaluates: `score = dot_product(vec(Q), vec(D))`.
      * **The Information Bottleneck**: All rich interactions between individual words are collapsed into a single mathematical vector. Negations (*"not"*, *"never"*), prepositional nuances (*"with"* vs *"without"*), and subtle qualifier shifts are frequently lost.
    
    * **Cross-Encoder Architecture (Reranker)**:
      * Concatenates query and document into a single joint sequence:  
        `[CLS] Query Tokens [SEP] Document Tokens [SEP]`.
      * Passes the joint sequence through **all multi-head self-attention layers simultaneously**.
      * Every word in the query directly attends to every word in the document, capturing syntactic structure, negations, and exact semantic alignment with 10x higher precision.
    """)

with col_bott2:
    st.markdown("### Architectural Topology Comparison")
    arch_dot = """
    digraph ArchitectureComparison {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_bi {
            label="1. Bi-Encoder (Fast Vector Search)"; style="dashed"; color="#2563EB";
            Q1 [label="Query Text", fillcolor="#DBEAFE"];
            D1 [label="Doc Text", fillcolor="#DBEAFE"];
            EncQ [label="Encoder (BERT)", fillcolor="#BFDBFE"];
            EncD [label="Encoder (BERT)", fillcolor="#BFDBFE"];
            Dot [label="Dot Product / Cosine\\n(Independent Vectors)", fillcolor="#93C5FD", style="filled,bold"];
            Q1 -> EncQ; D1 -> EncD;
            EncQ -> Dot; EncD -> Dot;
        }

        subgraph cluster_cross {
            label="2. Cross-Encoder (Precise Reranker)"; style="dashed"; color="#059669";
            Concat [label="[CLS] Query [SEP] Doc [SEP]", fillcolor="#D1FAE5"];
            FullAttn [label="Full Multi-Head Cross Attention\\n(Every token attends to every token)", fillcolor="#A7F3D0", style="filled,bold"];
            RelevanceOut [label="Direct Relevance Logit [0..1]", fillcolor="#6EE7B7", style="filled,bold"];
            Concat -> FullAttn -> RelevanceOut;
        }
    }
    """
    st.graphviz_chart(arch_dot, use_container_width=False)

st.divider()

# Section 2: The Two-Stage Funnel Architecture
st.header("2. The Two-Stage Funnel Architecture", divider="rainbow")

col_fun1, col_fun2 = col_fun1, col_fun2 = st.columns([1, 1])

with col_fun1:
    st.markdown("""
    ### Why We Cannot Use Cross-Encoders for Everything
    If Cross-Encoders are so dramatically superior to Bi-Encoders, why not use them for the entire search process?
    
    * **Computational Complexity**:
      * A Cross-Encoder requires running full neural forward inference on `(Query, Doc)` **at query time for every single candidate**.
      * If you have a database of 1,000,000 documents, evaluating a single search query requires 1,000,000 deep transformer forward passes!
      * **Latency**: At 1 millisecond per pass, search would take **1,000 seconds (16 minutes)** per user query!
    
    * **The Funnel Compromise**:
      * **Stage 1 (HNSW Vector Index)**: Scans 1,000,000 documents via graph traversal in **1.8 milliseconds**, pruning 99.998% of the database to yield the top **25 candidates**.
      * **Stage 2 (Cross-Encoder)**: Scores only the 25 candidates in **18 milliseconds**.
      * **Total Latency**: ~20 ms! You get the sub-millisecond scalability of HNSW combined with the human-grade precision of full cross-attention.
    """)

with col_fun2:
    st.markdown("### The Two-Stage Funnel Flow")
    fun_dot = """
    digraph TwoStageFunnel {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        FullCorpus [label="Enterprise Corpus\\n1,000,000 Indexed Chunks", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold", width=3.5];

        Stage1 [label="Stage 1: HNSW Dense Search\\nFilters 1,000,000 -> 25 Candidates (~2ms)", fillcolor="#FEF3C7", color="#D97706", width=2.8];

        Stage2 [label="Stage 2: Cross-Encoder Reranker\\nReranks 25 Candidates (~18ms)", fillcolor="#D1FAE5", color="#059669", width=2.2];

        TopK [label="Final Grounded Context\\nTop-3 Precision Chunks -> LLM Prompt", fillcolor="#A7F3D0", color="#059669", style="filled,bold", width=1.6];

        FullCorpus -> Stage1 [label="Coarse Recall Filter"];
        Stage1 -> Stage2 [label="Candidate Pool (Top 25)"];
        Stage2 -> TopK [label="Precision Reordering"];
    }
    """
    st.graphviz_chart(fun_dot, use_container_width=False)

st.divider()

# Section 3: Empirical Precision Gains & Benchmark Comparisons
st.header("3. Empirical Precision Gains & Benchmark Comparisons", divider="rainbow")

st.markdown("""
Industry benchmarks across the BEIR (Benchmarking Information Retrieval) and MS MARCO evaluation suites 
consistently prove the dramatic impact of adding a Stage-2 Reranker:
""")

BENCHMARK_DATA = [
    {
        "Retrieval Architecture": "Naive Dense Vector (Bi-Encoder Only - Top 3)",
        "Recall@20": "84.2%",
        "NDCG@10": "0.582",
        "MRR@10": "0.610",
        "Average Query Latency": "2.1 ms",
        "Context Precision": "Moderate (Frequent false positives)"
    },
    {
        "Retrieval Architecture": "Hybrid (Dense HNSW + Sparse BM25 Top 3)",
        "Recall@20": "89.6%",
        "NDCG@10": "0.645",
        "MRR@10": "0.672",
        "Average Query Latency": "3.8 ms",
        "Context Precision": "Good on exact keywords"
    },
    {
        "Retrieval Architecture": "Two-Stage Funnel (HNSW Top 25 -> Cross-Encoder Top 3)",
        "Recall@20": "89.6%",
        "NDCG@10": "0.768 (+28%)",
        "MRR@10": "0.795 (+27%)",
        "Average Query Latency": "21.4 ms",
        "Context Precision": "Exceptional (Zero-shot domain accuracy)"
    }
]

st.dataframe(pd.DataFrame(BENCHMARK_DATA), use_container_width=True, hide_index=True)

st.info("""
💡 **Takeaway**: Adding a Cross-Encoder Stage 2 boosts **NDCG@10 (Normalized Discounted Cumulative Gain) by over 28%** 
and **MRR@10 (Mean Reciprocal Rank) by over 27%**, directly translating to fewer hallucinations in downstream RAG generation!
""")

st.divider()

# Section 4: Interactive Workbench: Two-Stage Funnel Simulator
st.header("4. Interactive Workbench: Two-Stage Funnel Simulator", divider="rainbow")

st.markdown("""
Observe how a subtle query causes Bi-Encoder vector search to assign the best document to Rank 4 or 5, 
and watch how a Cross-Encoder reranker elevates the true answer to Rank 1.
""")

col_scen_ctl, col_scen_view = st.columns([1, 1.2])

with col_scen_ctl:
    st.subheader("1. Test Search Scenario")
    inquiry_choice = st.selectbox(
        "Select User Inquiry:",
        [
            "Negation Nuance: 'How to configure Envoy WITHOUT mutual TLS client verification?'",
            "Exact Threshold Query: 'What is the maximum allowed database connection idle timeout?'",
            "Policy Exception: 'When does the 35-day backup retention policy NOT apply?'"
        ]
    )
    
    enable_stage2 = st.checkbox("Enable Stage-2 Cross-Encoder Reranker", value=True)
    top_candidates_pool = st.slider("Stage-1 Candidate Pool (fetch_k):", min_value=5, max_value=20, value=10)

# Mock Documents
if "WITHOUT" in inquiry_choice:
    CANDIDATES = [
        {"id": "DOC-A", "title": "Envoy mTLS Mutual Authentication Guide", "text": "To enforce mTLS, configure require_client_certificate: true on downstream listeners.", "bi_rank": 1, "bi_score": 0.94, "cross_score": 0.12},
        {"id": "DOC-B", "title": "Envoy TLS Ingress Overview", "text": "Envoy terminates TLS using standard certificates mounted in /etc/envoy/certs.", "bi_rank": 2, "bi_score": 0.91, "cross_score": 0.45},
        {"id": "DOC-C", "title": "Envoy TLS Passthrough & Client Cert Bypass", "text": "To disable client certificate verification, omit the require_client_certificate flag or set it explicitly to false.", "bi_rank": 4, "bi_score": 0.82, "cross_score": 0.97},
        {"id": "DOC-D", "title": "SPIFFE Pod Identity Setup", "text": "SPIRE agents issue workload SVIDs to validate pod identity.", "bi_rank": 3, "bi_score": 0.85, "cross_score": 0.18}
    ]
else:
    CANDIDATES = [
        {"id": "DOC-A", "title": "PostgreSQL Connection Pooling Guide", "text": "Connection pools maintain a pool of warm database workers for fast transactions.", "bi_rank": 1, "bi_score": 0.93, "cross_score": 0.35},
        {"id": "DOC-B", "title": "General Infrastructure Timeouts", "text": "Network timeouts are enforced across all load balancers at 60 seconds.", "bi_rank": 2, "bi_score": 0.90, "cross_score": 0.42},
        {"id": "DOC-C", "title": "PostgreSQL idle_in_transaction_session_timeout", "text": "The maximum allowed database connection idle timeout is strictly configured to 300 seconds (5 minutes).", "bi_rank": 5, "bi_score": 0.81, "cross_score": 0.98},
        {"id": "DOC-D", "title": "Redis Cache Expiration", "text": "Redis keys expire based on standard TTL policies.", "bi_rank": 3, "bi_score": 0.86, "cross_score": 0.15}
    ]

# Sort candidates
bi_sorted = sorted(CANDIDATES, key=lambda x: x["bi_score"], reverse=True)
cross_sorted = sorted(CANDIDATES, key=lambda x: x["cross_score"], reverse=True)

with col_scen_view:
    st.subheader("2. Retrieval Comparison Analysis")
    
    if enable_stage2:
        winner = cross_sorted[0]
        st.success(f"🏆 **Stage-2 Reranker Winner**: **{winner['id']} - {winner['title']}** (Score: `{winner['cross_score']:.2f}`)")
        st.info("💡 Notice: The Bi-Encoder placed the exact answer at Rank 4/5 because it failed to grasp the negation/specific keyword. The Cross-Encoder's full multi-head attention correctly promoted it to Rank 1!")
    else:
        winner = bi_sorted[0]
        st.error(f"⚠️ **Stage-1 Bi-Encoder Winner**: **{winner['id']} - {winner['title']}** (Sim: `{winner['bi_score']:.2f}`)")
        st.warning("🚨 Precision Failure: Vector similarity picked a document that talks about enforcing mTLS because it ignored the word 'WITHOUT'!")

st.divider()

col_col_s1, col_col_s2 = col_col_s1, col_col_s2 = st.columns([1, 1])

with col_col_s1:
    st.subheader("Stage 1: Bi-Encoder HNSW Ranking (Coarse Filter)")
    st.caption("Scored in ~1.8 ms via pure vector cosine similarity.")
    for idx, c in enumerate(bi_sorted):
        with st.expander(f"Rank #{idx+1}: {c['id']} (Bi-Score: {c['bi_score']:.2f})", expanded=True):
            st.markdown(f"**{c['title']}**\n\n*{c['text']}*")

with col_col_s2:
    st.subheader("Stage 2: Cross-Encoder Reranked Order (Fine Precision)")
    st.caption("Scored in ~16.5 ms via full multi-head token cross-attention.")
    if enable_stage2:
        for idx, c in enumerate(cross_sorted):
            is_promoted = c["id"] == "DOC-C"
            with st.expander(f"{'⭐ ' if is_promoted else ''}Rank #{idx+1}: {c['id']} (Cross-Score: {c['cross_score']:.2f})", expanded=True):
                st.markdown(f"**{c['title']}**\n\n*{c['text']}*")
                if is_promoted:
                    st.success("Promoted from Rank #4 to Rank #1!")
    else:
        st.info("Enable the Stage-2 checkbox above to view Cross-Encoder reranking.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the fundamental architectural difference between a Bi-Encoder and a Cross-Encoder?"):
    st.markdown("""
    **Answer**:
    * **Bi-Encoder**: Encodes Query `Q` and Document `D` independently into two isolated vector vectors `u` and `v`. The similarity score is simply their dot product `<u, v>`. Because document embeddings are pre-computed offline, vector search scales to billions of documents via HNSW index traversal. However, it cannot model token-to-token cross-attention.
    * **Cross-Encoder**: Takes `[CLS] Query [SEP] Document [SEP]` as a single concatenated input sequence and passes it through all transformer self-attention layers simultaneously. Every query token attends to every document token, capturing fine-grained syntax and negations. However, it cannot pre-compute embeddings offline, making it too slow to scan an entire database.
    """)

with st.expander("Q2: What is the 'Two-Stage Funnel' in enterprise search architecture?"):
    st.markdown("""
    **Answer**:
    The Two-Stage Funnel resolves the trade-off between speed and precision:
    1. **Stage 1 (High Recall, Fast)**: Vector database (or hybrid BM25 + Vector) scans the entire corpus of millions of documents in 1-3ms to retrieve the top 20–50 candidate passages.
    2. **Stage 2 (High Precision, Deep)**: A Cross-Encoder reranks only those 20–50 candidates in 15–25ms, outputting the definitive top 3–5 passages to inject into the LLM context prompt.
    This delivers the sub-millisecond scalability of vector databases with the near-human accuracy of full transformer cross-attention.
    """)

with st.expander("Q3: Why do Bi-Encoders struggle with queries containing negations (e.g. 'WITHOUT mutual TLS')?"):
    st.markdown("""
    **Answer**:
    When a bi-encoder compresses a text into a single vector, content words (like *"mutual"*, *"TLS"*, *"client"*, *"Envoy"*) carry massive statistical magnitude in the attention pooling layer. 
    Small grammatical function words and negations (*"without"*, *"not"*, *"except"*) have very small relative norms. 
    Consequently, the vector for *"configure Envoy without mTLS"* lands geometrically right beside *"configure Envoy with mTLS"*. 
    A Cross-Encoder, having full cross-attention across tokens, directly aligns the negation *"without"* with the configuration flag `require_client_certificate: false`, correctly separating them.
    """)

with st.expander("Q4: How does Reranking improve LLM generation cost and latency?"):
    st.markdown("""
    **Answer**:
    Without a reranker, engineers are forced to pass large top-K pools (e.g. `k = 12` chunks) to the LLM to ensure the correct answer is captured somewhere in the prompt.
    * Passing 12 chunks inflates prompt token size by 6,000+ tokens, driving up LLM API costs and slowing down generation time-to-first-token.
    * With a Stage-2 Reranker, you can confidently shrink the prompt context to **only the top 3 chunks**, cutting input token costs by **75%** while delivering higher factual grounding!
    """)

with st.expander("Q5: What are the primary production trade-offs when adding a Reranker?"):
    st.markdown("""
    **Answer**:
    1. **Latency Penalty**: Adding a Cross-Encoder adds **15ms to 35ms** of inference latency to the query path.
    2. **Compute / Financial Cost**: Cross-Encoders require either self-hosting a GPU/ONNX inference microservice (e.g. Triton Server / vLLM) or paying for cloud reranking APIs (e.g. Cohere Rerank API).
    3. **Verdict**: For 95% of enterprise applications, the 20ms latency addition is completely invisible compared to the 1,000ms taken by LLM generation, making reranking an essential architectural best practice.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing the Entire Database to a Cross-Encoder**  
    Attempting to run a Cross-Encoder over 50,000 documents causes query timeouts and GPU memory exhaustion. Always use a vector database for Stage 1 candidate generation.
    """)
    st.warning("""
    **Trap 2: Reranking an Insufficiently Sized Candidate Pool**  
    Setting Stage 1 to fetch only 3 candidates (`fetch_k = 3`) means the reranker can only reorder those 3 items. If the real answer was at Rank 7 in Stage 1, the reranker can never see it. Fetch 20–50 candidates.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Reranking Full 10-Page Documents Rather Than Chunks**  
    Feeding entire 4,000-word documents into a Cross-Encoder triggers model sequence length truncation (usually 512 tokens). Always rerank atomic chunks.
    """)
    st.warning("""
    **Trap 4: Expecting Rerankers to Fix Missing Data**  
    A reranker can only promote documents that exist in Stage 1's candidate pool. If Stage-1 retrieval fails to retrieve the correct document at all (Recall@50 = 0), reranking cannot help.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the Bi-Encoder bottleneck?"):
        st.markdown("""
        **Compression Bottleneck**:  
        Compresses documents into a single static vector before queries are known; cannot model fine-grained token cross-attention.
        """)
    with st.expander("🗂️ Flashcard: How does a Cross-Encoder work?"):
        st.markdown("""
        **Full Cross-Attention**:  
        Processes `[CLS] Query [SEP] Doc [SEP]` together so every query token attends directly to every document token.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Two-Stage Funnel?"):
        st.markdown("""
        **Two-Stage Pipeline**:  
        Stage 1: HNSW retrieves top 25 candidates in 2ms.  
        Stage 2: Cross-Encoder reranks top 25 down to top 3 in 20ms.
        """)
    with st.expander("🗂️ Flashcard: Why not use Cross-Encoders for everything?"):
        st.markdown("""
        **O(N) Complexity**:  
        Scoring 1,000,000 documents at query time would take 15+ minutes per search!
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the mathematical difference between Bi-Encoders and Cross-Encoders.", value=True)
chk2 = st.checkbox("I know why Bi-Encoders suffer from an information compression bottleneck on negations and syntax.", value=True)
chk3 = st.checkbox("I understand the Two-Stage Funnel architecture: Fast HNSW Stage 1 -> Deep Cross-Encoder Stage 2.", value=True)
chk4 = st.checkbox("I know how to tune the Stage-1 candidate pool size (`fetch_k = 20 - 50`).", value=True)
chk5 = st.checkbox("I understand how reranking cuts prompt token volume and reduces LLM hallucinations.", value=True)
chk6 = st.checkbox("I can evaluate the latency and compute trade-offs of deploying a production reranking microservice.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Why Reranking and are ready for Chapter 17.2: Reranking Models!")
