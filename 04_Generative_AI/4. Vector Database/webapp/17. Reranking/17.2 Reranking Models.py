import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="17.2 Reranking Models - Vector DB Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("17.2 Reranking Models: Cross-Encoders, BGE-Reranker, Cohere & LLM-as-Reranker")
st.caption("Chapter 17: Reranking | Domain: 4. Vector Database")
st.markdown("""
Once the need for a Stage-2 Reranker is established, engineers must select the appropriate **Reranking Model Architecture**.

The modern reranking ecosystem spans three distinct operational paradigms:
1. **Self-Hosted Open-Weight Cross-Encoders**:
   * *Sentence-Transformers (`ms-marco-MiniLM-L-6-v2`)*: Ultra-lightweight (22M params), CPU-runnable in ~6ms.
   * *BAAI BGE-Reranker (`bge-reranker-large` / `v2-m3`)*: State-of-the-art multilingual performance with 8k token sequence lengths.
2. **Managed Cloud Reranking APIs**:
   * *Cohere Rerank (`rerank-v3.5` / `multilingual-v3.0`)*: Enterprise-grade managed inference with native semi-structured JSON attribute scoring.
3. **Generative LLM-as-a-Judge (Listwise RankGPT)**:
   * Prompts a small autoregressive model to inspect all candidate chunks simultaneously and output an optimal ranked permutation.

In this lesson, we explore the architectures of modern rerankers, compare **Pointwise, Pairwise, and Listwise** ranking formulations, 
implement production integration patterns in Python, and test an interactive Reranking Model Arena.
""")

st.divider()

# Section 1: The Modern Reranking Model Landscape
st.header("1. The Modern Reranking Model Landscape", divider="rainbow")

col_land1, col_land2 = col_land1, col_land2 = st.columns([1.1, 0.9])

with col_land1:
    st.markdown("""
    ### Four Core Reranker Options
    
    1. **Sentence-Transformers MiniLM (`ms-marco-MiniLM-L-6-v2`)**:
       * *Size*: 22 Million parameters (~80 MB RAM).
       * *Inference Latency*: **4–8 ms on CPU** (no GPU required!).
       * *Best For*: Latency-critical edge devices, local development, and high-QPS budget-conscious APIs.
    
    2. **BAAI BGE-Reranker Family (`bge-reranker-large` & `v2-m3`)**:
       * *Size*: 560 Million parameters (large) / XLM-RoBERTa backbone (m3).
       * *Capabilities*: 8,192 token input window, 100+ languages, top of the MTEB Reranking leaderboard.
       * *Best For*: Enterprise multi-lingual search, long-context chunk re-ranking on private GPU clusters.
    
    3. **Cohere Rerank API (`rerank-english-v3.0` / `multilingual-v3.0`)**:
       * *Mechanism*: Cloud SaaS API; billed per 1,000 search queries.
       * *Key Feature*: Native support for semi-structured JSON objects, code snippets, and automated token budgeting.
       * *Best For*: Serverless architectures and teams seeking zero GPU infrastructure maintenance.
    
    4. **Listwise LLM Reranking (RankGPT / RankZephyr)**:
       * *Mechanism*: Feeds 10–20 candidates directly into an LLM prompt: *"Rank these 10 passages from most relevant to least relevant."*
       * *Best For*: Complex multi-hop reasoning where deep semantic comparison across candidates is mandatory.
    """)

with col_land2:
    st.markdown("### Model Architectural Comparison")
    land_dot = """
    digraph ModelSpectrum {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_open {
            label="Self-Hosted Open-Weight (Local GPU/CPU)"; style="dashed"; color="#059669";
            MiniLM [label="MiniLM-L6-v2\\n22M Params | ~6ms on CPU", fillcolor="#D1FAE5"];
            BGE [label="BGE-Reranker-Large / M3\\n560M Params | 8k tokens | GPU SOTA", fillcolor="#A7F3D0", style="filled,bold"];
        }

        subgraph cluster_cloud {
            label="Managed Cloud API"; style="dashed"; color="#2563EB";
            Cohere [label="Cohere Rerank v3.5\\nManaged SaaS | JSON support", fillcolor="#DBEAFE", style="filled,bold"];
        }

        subgraph cluster_llm {
            label="Generative Listwise"; style="dashed"; color="#7C3AED";
            RankGPT [label="RankGPT / Listwise LLM\\nContext window permutation", fillcolor="#EDE9FE"];
        }
    }
    """
    st.graphviz_chart(land_dot, use_container_width=False)

st.divider()

# Section 2: Ranking Paradigms: Pointwise vs Pairwise vs Listwise
st.header("2. Ranking Paradigms: Pointwise vs Pairwise vs Listwise", divider="rainbow")

col_par1, col_par2, col_par3 = st.columns(3)

with col_par1:
    st.markdown("""
    #### 1. Pointwise Reranking
    * **Mechanism**: Evaluates each candidate independently:  
      `Score = f(Query, Doc_i)`.
    * **Examples**: BGE-Reranker, MiniLM, Cohere.
    * **Pros**: Highly parallelizable across GPU threads; `O(N)` computational complexity.
    * **Cons**: Cannot model relative superiority between two adjacent candidates.
    """)

with col_par2:
    st.markdown("""
    #### 2. Pairwise Reranking
    * **Mechanism**: Evaluates candidate pairs:  
      `Is Doc_A better than Doc_B for Query?`
    * **Examples**: RankNet, LambdaMART.
    * **Pros**: Higher fidelity on subtle comparative distinctions.
    * **Cons**: `O(N^2)` combinatorial comparisons. Evaluating 20 candidates requires ~190 pair inferences!
    """)

with col_par3:
    st.markdown("""
    #### 3. Listwise Reranking
    * **Mechanism**: Inspects the entire list simultaneously:  
      `Permutation = f(Query, [Doc_1 ... Doc_N])`.
    * **Examples**: RankGPT, RankZephyr.
    * **Pros**: Global awareness of all candidates in context.
    * **Cons**: Vulnerable to "Lost-in-the-Middle" attention bias; higher token latency.
    """)

st.divider()

# Section 3: Production Code Patterns
st.header("3. Production Code Patterns", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. Self-Hosted BGE-Reranker via CrossEncoder")
    st.code("""from sentence_transformers import CrossEncoder

# Load pre-trained BGE reranker
reranker = CrossEncoder(
    "BAAI/bge-reranker-large",
    max_length=512,
    device="cuda"  # or "cpu"
)

query = "What is the Envoy proxy certificate TTL?"
candidates = [
    "Envoy proxies mount certificates in /etc/certs.",
    "Certificates have a strict 1-hour time-to-live (TTL) and rotate automatically.",
    "PostgreSQL transaction logs stream to S3 continuously."
]

# Construct (Query, Doc) pairs
pairs = [[query, doc] for doc in candidates]

# Predict pointwise relevance scores (higher = more relevant)
scores = reranker.predict(pairs)

# Sort candidates by score descending
ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked_results:
    print(f"Score: {score:.4f} | Content: {doc[:60]}...")
""", language="python")

with col_code2:
    st.markdown("### 2. LangChain + Cohere Rerank Integration")
    st.code("""from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# 1. Base Stage-1 Vector Retriever (Fetches top 20 candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 2. Stage-2 Compressor Reranker (Slices to top 3)
cohere_compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=3
)

# 3. Two-Stage Funnel Pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_compressor,
    base_retriever=base_retriever
)

# Executes: HNSW top-20 -> Cohere API rerank -> returns top-3
final_docs = compression_retriever.invoke("Envoy certificate TTL")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Reranking Model Arena
st.header("4. Interactive Workbench: Reranking Model Arena", divider="rainbow")

st.markdown("""
Compare how different reranking model architectures evaluate a candidate pool containing 
both exact technical answers and deceptive topical distractors.
""")

MOCK_ARENA_CANDIDATES = [
    {
        "id": "PASSAGE-1",
        "title": "Exact SLA Specification",
        "text": "The Recovery Point Objective (RPO) is strictly under 5 minutes, while the Recovery Time Objective (RTO) for full region restoration is 4 hours.",
        "bge_score": 0.982,
        "cohere_score": 0.965,
        "minilm_score": 0.912,
        "rankgpt_pos": 1
    },
    {
        "id": "PASSAGE-2",
        "title": "Topical Distractor (Mentions RPO/RTO but wrong tech)",
        "text": "In older legacy mainframe systems, RTO was commonly negotiated at 24 to 48 hours with tape backups.",
        "bge_score": 0.124,
        "cohere_score": 0.145,
        "minilm_score": 0.320,
        "rankgpt_pos": 3
    },
    {
        "id": "PASSAGE-3",
        "title": "Related Operational Context",
        "text": "PostgreSQL WAL archiving streams transactions continuously to Amazon S3 to satisfy Point-In-Time Recovery windows.",
        "bge_score": 0.745,
        "cohere_score": 0.780,
        "minilm_score": 0.680,
        "rankgpt_pos": 2
    },
    {
        "id": "PASSAGE-4",
        "title": "Irrelevant Distractor",
        "text": "Employee meal stipends during business travel are capped at $85 per day under corporate expense policies.",
        "bge_score": 0.002,
        "cohere_score": 0.001,
        "minilm_score": 0.015,
        "rankgpt_pos": 4
    }
]

col_arena_m, col_arena_q = col_arena_m, col_arena_q = st.columns([1, 1.2])

with col_arena_m:
    st.subheader("1. Choose Active Reranker Model")
    active_model = st.selectbox(
        "Reranker Architecture:",
        [
            "BAAI BGE-Reranker-Large (Pointwise Cross-Encoder)",
            "Cohere Rerank v3.5 (Cloud Managed Pointwise API)",
            "Sentence-Transformers MiniLM-L6 (Lightweight CPU Cross-Encoder)",
            "RankGPT (Generative Listwise LLM-as-a-Judge)"
        ]
    )

with col_arena_q:
    st.subheader("2. Target Evaluation Query")
    st.info("Query: **'What is our database disaster recovery RTO and RPO target?'**")
    st.caption("Candidate pool contains 4 passages (1 exact match, 1 topical distractor, 1 related runbook, 1 irrelevant).")

# Sorting based on model
if "BGE" in active_model:
    sorted_arena = sorted(MOCK_ARENA_CANDIDATES, key=lambda x: x["bge_score"], reverse=True)
    score_key = "bge_score"
    lat_est = "18 ms (GPU TensorRT)"
    cost_est = "$0.00 (Self-Hosted Pod)"
elif "Cohere" in active_model:
    sorted_arena = sorted(MOCK_ARENA_CANDIDATES, key=lambda x: x["cohere_score"], reverse=True)
    score_key = "cohere_score"
    lat_est = "24 ms (Cloud API)"
    cost_est = "$1.00 / 1,000 queries"
elif "MiniLM" in active_model:
    sorted_arena = sorted(MOCK_ARENA_CANDIDATES, key=lambda x: x["minilm_score"], reverse=True)
    score_key = "minilm_score"
    lat_est = "6 ms (CPU ONNX)"
    cost_est = "$0.00 (CPU Threaded)"
else:  # RankGPT
    sorted_arena = sorted(MOCK_ARENA_CANDIDATES, key=lambda x: x["rankgpt_pos"])
    score_key = "rankgpt_pos"
    lat_est = "320 ms (LLM Generation)"
    cost_est = "$2.50 / 1,000 queries"

st.divider()

col_arena_out, col_arena_metrics = col_arena_out, col_arena_metrics = st.columns([1.2, 0.8])

with col_arena_out:
    st.subheader(f"Ranked Candidate Output ({active_model.split()[0]})")
    for idx, c in enumerate(sorted_arena):
        with st.expander(f"🏅 Rank #{idx+1}: {c['title']}", expanded=True):
            st.markdown(f"*{c['text']}*")
            if score_key == "rankgpt_pos":
                st.caption(f"Listwise Permutation Position: #{c['rankgpt_pos']}")
            else:
                st.caption(f"Relevance Score: `{c[score_key]:.4f}`")

with col_arena_metrics:
    st.subheader("Model Profile & Operational Audit")
    st.metric("Estimated Rerank Latency", lat_est)
    st.metric("Operational Cost", cost_est)
    
    if sorted_arena[0]["id"] == "PASSAGE-1":
        st.success("✅ Perfect Alignment: Exact specification identified at Rank #1 with high confidence separation.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the difference between Pointwise, Pairwise, and Listwise reranking?"):
    st.markdown("""
    **Answer**:
    * **Pointwise**: Scores each document independently against the query: `Score = f(Q, D_i)`. Scales linearly (`O(N)`), easy to batch on GPUs. Used by BGE, MiniLM, and Cohere.
    * **Pairwise**: Compares document pairs: `f(Q, D_A, D_B) -> {D_A > D_B}`. Higher fidelity for fine-grained ranking, but scales quadratically (`O(N^2)`), making it too slow for real-time online retrieval.
    * **Listwise**: Takes all candidate documents in a single prompt and outputs an entire ordered list: `f(Q, [D_1 ... D_N]) -> [D_3, D_1, D_4, D_2]`. Used in RankGPT. Captures global inter-document context, but has higher latency and is susceptible to attention position biases.
    """)

with st.expander("Q2: Why is BAAI's BGE-Reranker-Large widely considered the gold standard for self-hosted reranking?"):
    st.markdown("""
    **Answer**:
    1. **Top MTEB Benchmark Scores**: Consistently leads open-source reranking benchmarks on MS MARCO and BEIR.
    2. **Multilingual Architecture (`v2-m3`)**: Pre-trained on 100+ languages with native cross-lingual transfer (e.g. querying in English can accurately rank Japanese documentation).
    3. **Long Context Support**: Supports up to 8,192 tokens, preventing sequence truncation on large document chunks.
    4. **ONNX / TensorRT Support**: Can be compiled to ONNX or TensorRT for sub-10ms GPU inference.
    """)

with st.expander("Q3: When is a lightweight model like `ms-marco-MiniLM-L-6-v2` preferred over BGE-large?"):
    st.markdown("""
    **Answer**:
    * **Infrastructure Constraints**: MiniLM has only 22M parameters and runs in ~6ms on a standard **multi-core CPU without requiring an expensive GPU**.
    * **Edge / Desktop Deployments**: Fits comfortably in memory on local laptops or edge microservices.
    * **Cost-Sensitive High QPS**: When serving 10,000 queries per second, running BGE-large requires a massive GPU cluster, whereas MiniLM can be horizontally scaled across cheap commodity CPU nodes.
    """)

with st.expander("Q4: How does Cohere Rerank handle structured JSON documents?"):
    st.markdown("""
    **Answer**:
    Standard Cross-Encoders accept only flat string text. 
    Cohere Rerank v3 natively accepts **structured JSON objects** (e.g. `{"title": "...", "author": "...", "body": "...", "price": 49}`):
    * Developers can specify `rank_fields=["title", "body"]` to focus the attention mechanism on primary content while ignoring auxiliary metadata.
    * This prevents boilerplate JSON metadata from distorting relevance scoring.
    """)

with st.expander("Q5: What is Position Bias in Listwise LLM Rerankers (RankGPT), and how is it mitigated?"):
    st.markdown("""
    **Answer**:
    * **Position Bias (Lost in the Middle)**: Autoregressive LLMs pay disproportionate attention to documents placed at the very beginning and very end of their prompt context, while under-scoring documents located in the middle.
    * **Mitigation (Sliding Window Permutation)**: RankGPT uses a sliding window approach:
      1. Rerank candidates 10–20.
      2. Slide the window up to evaluate candidates 5–15.
      3. Repeat in multiple passes to ensure every document is evaluated in both leading and trailing positions.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Deploying BGE-Large on CPU in the Web Request Path**  
    Running a 560M parameter Cross-Encoder on a single CPU core takes 250ms–500ms per query, destroying web response SLAs. Run on GPU or use MiniLM on CPU.
    """)
    st.warning("""
    **Trap 2: Ignoring Sequence Length Truncation**  
    MiniLM has a hard limit of 512 tokens. Passing 1,000-token chunks silently discards the tail 50% of the text. Use BGE (8k limit) or ensure chunks fit model limits.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Reranking in Pure Python Loops**  
    Iterating through candidates one-by-one with `for doc in candidates: score(doc)` destroys throughput. Always pass all candidate pairs in a single batched array call.
    """)
    st.warning("""
    **Trap 4: Unbounded Listwise Candidate Windows**  
    Feeding 50 candidates into an LLM prompt for listwise ranking consumes 25,000 tokens, inflating latency and cost. Limit listwise ranking to top 10–15 candidates.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Pointwise reranking?"):
        st.markdown("""
        **Pointwise**:  
        Scores each `(Query, Doc)` pair independently into a float. Fast, parallelizable, `O(N)` complexity (BGE, Cohere).
        """)
    with st.expander("🗂️ Flashcard: Why choose BGE-Reranker-Large?"):
        st.markdown("""
        **BGE Strengths**:  
        Top MTEB benchmark performance, 8k token sequence lengths, 100+ language support.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: When use MiniLM-L6-v2?"):
        st.markdown("""
        **CPU Efficiency**:  
        Tiny 22M params; evaluates candidates in 4-8ms on standard CPUs without a GPU.
        """)
    with st.expander("🗂️ Flashcard: What is Listwise reranking?"):
        st.markdown("""
        **Listwise (RankGPT)**:  
        Prompts an LLM to inspect all candidates at once and output a ranked permutation.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural trade-offs between Pointwise, Pairwise, and Listwise reranking.", value=True)
chk2 = st.checkbox("I know how to deploy BGE-Reranker-Large with GPU batching in production.", value=True)
chk3 = st.checkbox("I understand when to use lightweight CPU-friendly models like `ms-marco-MiniLM-L-6-v2`.", value=True)
chk4 = st.checkbox("I know how to integrate Cohere Rerank API using LangChain `ContextualCompressionRetriever`.", value=True)
chk5 = st.checkbox("I understand Position Bias in Listwise LLM ranking and how sliding windows mitigate it.", value=True)
chk6 = st.checkbox("I can architect a cost- and latency-optimized two-stage reranking pipeline.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Reranking Models and are ready for Chapter 17.3: The Complete Reranking Pipeline!")
