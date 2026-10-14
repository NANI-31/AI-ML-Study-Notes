import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="15.4 LangChain Retrieval - Vector DB Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("15.4 LangChain Retrieval: Similarity, MMR, Score Thresholds & Metadata Filters")
st.caption("Chapter 15: LangChain + Vector Databases | Domain: 4. Vector Database")
st.markdown("""
In a production RAG application, simply fetching the "top-3 most similar" vector chunks is often insufficient. 
Naive similarity search suffers from a critical flaw: **redundancy**. 

If a 50-page document mentions a topic repeatedly across adjacent paragraphs, standard vector search will retrieve 
three nearly identical variations of the same paragraph, wasting prompt tokens and blinding the LLM to other crucial sections 
of the document.

LangChain addresses this with advanced retrieval algorithms built into the **`VectorStoreRetriever`**:
1. **Standard Similarity Retrieval**: Fast, pure nearest-neighbor lookups.
2. **Maximal Marginal Relevance (MMR)**: Actively penalizes redundancy, selecting candidates that maximize both relevance to the query AND diversity relative to already-chosen documents.
3. **Similarity Score Thresholding**: Prunes low-confidence, out-of-domain chunks before they reach the LLM.
4. **Targeted Metadata Filtering**: Enforces strict database-level security and categorical scoping.

In this lesson, we explore LangChain's retrieval strategies, master the mathematics of **MMR diversity tuning**, 
and experiment with an interactive Similarity vs. MMR Comparison Lab.
""")

st.divider()

# Section 1: The Spectrum of LangChain Retrieval Strategies
st.header("1. The Spectrum of LangChain Retrieval Strategies", divider="rainbow")

col_spec1, col_spec2 = col_spec1, col_spec2 = st.columns([1.1, 0.9])

with col_spec1:
    st.markdown("""
    ### The 3 Primary Retriever Modes in `.as_retriever()`
    
    1. **Standard Similarity (`search_type="similarity"`)**:
       * Queries the HNSW index and returns the `k` closest vectors by geometric distance.
       * *Pros*: Lowest latency (< 2ms); strictly maximizes query-document cosine match.
       * *Cons*: Highly redundant when documents have repeated phrasing.
    
    2. **Maximal Marginal Relevance (`search_type="mmr"`)**:
       * Fetches an initial candidate pool of `fetch_k` vectors (e.g., 20 candidates).
       * Iteratively selects `k` final documents by balancing query similarity against similarity to previously selected chunks.
       * *Pros*: Eliminates duplicate information; surfaces complementary angles.
    
    3. **Similarity Score Threshold (`search_type="similarity_score_threshold"`)**:
       * Drops any chunk whose distance exceeds a configured threshold.
       * *Pros*: Completely suppresses hallucination-inducing noise on out-of-domain questions.
    """)

with col_spec2:
    st.markdown("### MMR Diversity Selection Architecture")
    mmr_dot = """
    digraph MMRFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Query [label="User Query Vector\\n'Envoy mTLS Configuration'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        FetchPool [label="Initial Candidate Pool (fetch_k=20)\\nTop 20 Nearest Neighbors", fillcolor="#FEF3C7", color="#D97706"];

        MMREngine [label="MMR Selection Loop\\nScore = lambda * Sim(Doc, Q) - (1-lambda) * max Sim(Doc, Selected)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        subgraph cluster_final {
            label="Diverse Top-3 Chunks (k=3)"; style="dashed"; color="#7C3AED";
            C1 [label="Chunk A: Core mTLS Architecture (Relevance: 0.94)", fillcolor="#EDE9FE"];
            C2 [label="Chunk B: Cert Rotation / SPIFFE (Relevance: 0.88, Diverse)", fillcolor="#EDE9FE"];
            C3 [label="Chunk C: Debugging Envoy TLS Drops (Relevance: 0.82, Diverse)", fillcolor="#EDE9FE"];
        }

        Query -> FetchPool;
        FetchPool -> MMREngine;
        MMREngine -> C1;
        MMREngine -> C2;
        MMREngine -> C3;
    }
    """
    st.graphviz_chart(mmr_dot, use_container_width=False)

st.divider()

# Section 2: The Mathematics of Maximal Marginal Relevance (MMR)
st.header("2. The Mathematics of Maximal Marginal Relevance (MMR)", divider="rainbow")

col_math1, col_math2 = col_math1, col_math2 = st.columns([1, 1])

with col_math1:
    st.markdown("""
    ### The MMR Objective Equation
    At each step, MMR chooses the next document `D_i` from the unselected candidate pool that maximizes:
    
    ```text
    MMR_Score(D_i) = lambda * Sim(D_i, Q) - (1 - lambda) * max_{D_j in Selected} Sim(D_i, D_j)
    ```
    
    Where:
    * **`Sim(D_i, Q)`**: Cosine similarity between candidate document `D_i` and the user query `Q`.
    * **`Sim(D_i, D_j)`**: Cosine similarity between candidate `D_i` and already-chosen documents `D_j` in the selected set.
    * **`lambda` (`lambda_mult`)**: A diversity weighting factor between `0.0` and `1.0`.
    """)

with col_math2:
    st.markdown("""
    ### Tuning the `lambda_mult` Parameter
    
    | `lambda_mult` Value | Behavior | Ideal Use Case |
    | :---: | :--- | :--- |
    | **`1.0`** | **Pure Similarity**: Identical to standard Top-K; zero diversity penalty. | Fact retrieval with short, distinct documents. |
    | **`0.7`** | **Industry Sweet Spot**: 70% query relevance, 30% diversity bonus. | General enterprise RAG, search, customer support. |
    | **`0.5`** | **Balanced Exploration**: Equal weight on novelty and relevance. | Broad research summaries, topic exploration. |
    | **`0.0`** | **Maximal Diversity**: Prioritizes chunks as different from each other as possible. | Clustering, exploratory data analysis. |
    """)

st.divider()

# Section 3: Python Implementation Reference
st.header("3. Python Implementation Reference", divider="rainbow")

col_py1, col_py2 = col_py1, col_py2 = st.columns([1, 1])

with col_py1:
    st.markdown("### 1. MMR Retriever Configuration")
    st.code("""from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="production_kb",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_store"
)

# MMR Retriever with custom candidate pool and lambda
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,               # Return top 3 final diverse chunks
        "fetch_k": 20,        # Initial candidate pool to evaluate
        "lambda_mult": 0.7    # 70% relevance, 30% diversity
    }
)

docs = mmr_retriever.invoke("How do I configure Envoy proxy?")
""", language="python")

with col_py2:
    st.markdown("### 2. Score Threshold + Metadata Filtering")
    st.code("""# Enforce minimum similarity cutoff and metadata filter
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.80,  # Drop low confidence matches
        "k": 5,
        "filter": {
            "$and": [
                {"department": "Security"},
                {"year": {"$gte": 2024}}
            ]
        }
    }
)

# If no documents pass the 0.80 threshold, returns empty list []
docs = threshold_retriever.invoke("What is our password policy?")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Similarity vs MMR Diversity Lab
st.header("4. Interactive Workbench: Similarity vs MMR Diversity Lab", divider="rainbow")

st.markdown("""
Observe how standard **Similarity Search** vs. **Maximal Marginal Relevance (MMR)** behaves on a realistic 
corpus containing multiple repetitive drafts.
""")

# Mock Corpus with Redundant Drafts
CORPUS_DOCS = [
    {
        "id": "DOC-ENV-01",
        "title": "Envoy mTLS Overview (v1)",
        "content": "Envoy proxy mTLS requires mounting client certificates in /etc/envoy/certs and setting require_client_certificate to true.",
        "query_sim": 0.95,
        "inter_doc_sim": 0.92  # High similarity to Doc 2
    },
    {
        "id": "DOC-ENV-02",
        "title": "Envoy mTLS Overview (v2 Draft)",
        "content": "To enable mTLS on Envoy listeners, mount certificates in /etc/envoy/certs and verify require_client_certificate: true.",
        "query_sim": 0.94,
        "inter_doc_sim": 0.92  # Near duplicate of Doc 1!
    },
    {
        "id": "DOC-SPIFFE-03",
        "title": "SPIFFE/SPIRE Certificate Rotation",
        "content": "Automated certificate rotation for Envoy uses SPIRE agents running as local Kubernetes DaemonSets with 1-hour TTLs.",
        "query_sim": 0.86,
        "inter_doc_sim": 0.35  # Distinct topic!
    },
    {
        "id": "DOC-DEBUG-04",
        "title": "Troubleshooting Envoy TLS Drops",
        "content": "Check Envoy admin endpoint /clusters and grep for ssl.connection_error to diagnose TLS handshake protocol mismatches.",
        "query_sim": 0.81,
        "inter_doc_sim": 0.28  # Distinct topic!
    }
]

col_lab_q, col_lab_mmr = st.columns([1, 1.2])

with col_lab_q:
    st.subheader("1. Test Search Query")
    st.info("Query: **'How do I configure and maintain Envoy mTLS certificates?'**")
    
    sim_k = st.slider("Number of Chunks to Retrieve (k):", min_value=2, max_value=3, value=2)
    ui_lambda = st.slider("MMR Diversity Lambda (`lambda_mult`):", min_value=0.1, max_value=1.0, value=0.6, step=0.1)

with col_lab_mmr:
    st.subheader("2. Retrieval Comparison Analysis")
    
    # Calculate Standard Similarity Top-K
    sorted_by_sim = sorted(CORPUS_DOCS, key=lambda x: x["query_sim"], reverse=True)[:sim_k]
    
    # Calculate MMR Top-K
    mmr_selected = [sorted_by_sim[0]]  # Pick top item first
    # Pick second item
    remaining = [d for d in CORPUS_DOCS if d["id"] != mmr_selected[0]["id"]]
    for item in remaining:
        # MMR score calculation
        penalty = item["inter_doc_sim"] if item["id"] == "DOC-ENV-02" else 0.30
        item["mmr_score"] = (ui_lambda * item["query_sim"]) - ((1.0 - ui_lambda) * penalty)
    
    remaining.sort(key=lambda x: x["mmr_score"], reverse=True)
    if sim_k == 2:
        mmr_selected.append(remaining[0])
    else:
        mmr_selected.append(remaining[0])
        mmr_selected.append(remaining[1])

st.divider()

col_out_sim, col_out_mmr = st.columns([1, 1])

with col_out_sim:
    st.subheader("🔴 Standard Similarity Search")
    st.caption("Picks purely by highest query similarity (zero diversity penalty).")
    
    for idx, doc in enumerate(sorted_by_sim):
        with st.expander(f"Chunk {idx+1}: {doc['title']} (Sim: {doc['query_sim']:.2f})", expanded=True):
            st.markdown(f"*{doc['content']}*")
            
    if sorted_by_sim[0]["id"] == "DOC-ENV-01" and sorted_by_sim[1]["id"] == "DOC-ENV-02":
        st.error("⚠️ Redundancy Failure: Retrieved two nearly identical copies of the same overview text! Wasted prompt tokens on duplicate information.")

with col_out_mmr:
    st.subheader("🟢 Maximal Marginal Relevance (MMR)")
    st.caption(f"Balances query similarity with diversity (lambda = {ui_lambda}).")
    
    for idx, doc in enumerate(mmr_selected):
        with st.expander(f"Chunk {idx+1}: {doc['title']} (Sim: {doc['query_sim']:.2f})", expanded=True):
            st.markdown(f"*{doc['content']}*")
            
    if "DOC-SPIFFE-03" in [d["id"] for d in mmr_selected]:
        st.success("✅ Diversity Win: MMR rejected the duplicate draft (v2) and selected the SPIRE Certificate Rotation runbook, giving the LLM complementary answers!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the primary problem solved by Maximal Marginal Relevance (MMR) in RAG retrieval?"):
    st.markdown("""
    **Answer**:
    Standard vector similarity search optimizes exclusively for query relevance. In real-world enterprise corpora:
    * Documents often contain repeated paragraphs across revisions, changelogs, or summary sections.
    * A standard similarity search for *"Kubernetes pod autoscaling"* will return 3 nearly identical sentences explaining what HPA is.
    * The LLM prompt gets filled with redundant tokens, leaving zero room for related operational details like metric server setup or stabilization windows.
    * **MMR penalizes redundancy**, ensuring that each retrieved chunk introduces novel, complementary semantic information.
    """)

with st.expander("Q2: How does the `fetch_k` parameter in MMR impact query latency and diversity quality?"):
    st.markdown("""
    **Answer**:
    * **`k`**: The final number of diverse chunks returned to the caller (e.g. `k = 4`).
    * **`fetch_k`**: The initial candidate pool retrieved from the vector database before MMR re-ranking begins (e.g. `fetch_k = 20`).
    * **Impact**:
      * If `fetch_k` is too small (e.g. `fetch_k = 5`), MMR has very few candidates to choose from, limiting diversity.
      * If `fetch_k` is too large (e.g. `fetch_k = 200`), computing pairwise cosine similarities between all 200 candidates in Python introduces unnecessary CPU latency.
      * **Rule of Thumb**: Set `fetch_k = 4 * k` to `5 * k` (e.g., fetch 20 for k=4).
    """)

with st.expander("Q3: When should you use `search_type='similarity_score_threshold'`?"):
    st.markdown("""
    **Answer**:
    Use `similarity_score_threshold` when your application requires a **hard gatekeeper against out-of-domain hallucinations**.
    * In standard similarity search, if a user asks about *"Quantum gravity"* in a corporate HR database, Chroma will still return the 3 "closest" HR documents (e.g. employee conduct policies) with terrible cosine scores (e.g. 0.35).
    * Feeding these irrelevant documents to an LLM triggers hallucinations.
    * By setting `score_threshold = 0.75`, the retriever discards all candidates and returns an empty list `[]`. The application can immediately respond: *"No relevant documentation found."*
    """)

with st.expander("Q4: How does LangChain execute metadata filtering under the hood with ChromaDB?"):
    st.markdown("""
    **Answer**:
    When you pass `search_kwargs={"filter": {"dept": "Security"}}` in LangChain:
    1. LangChain translates the dictionary into ChromaDB's native `where` parameter: `collection.query(..., where={"dept": "Security"})`.
    2. ChromaDB applies single-stage metadata filtering during HNSW graph exploration, ensuring that non-qualifying nodes are pruned before distance evaluation.
    3. LangChain wraps the resulting qualifying records into `List[Document]` with preserved metadata.
    """)

with st.expander("Q5: Can MMR be combined with metadata filtering in a single retriever call?"):
    st.markdown("""
    **Answer**:
    **Yes**. In LangChain:
    ```python
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.7,
            "filter": {"tenant_id": "org_42"}
        }
    )
    ```
    1. ChromaDB first filters the index to only include documents matching `tenant_id == 'org_42'`.
    2. Chroma retrieves the top 20 candidate vectors matching the query within that tenant partition.
    3. MMR reranks those 20 candidates in memory to select the top 3 most diverse and relevant chunks.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Setting `lambda_mult` Too Low (< 0.3)**  
    Setting lambda close to 0 causes MMR to ignore query relevance, selecting completely unrelated documents simply because they are mathematically distant from Chunk 1. Maintain lambda between 0.5 and 0.8.
    """)
    st.warning("""
    **Trap 2: Setting `fetch_k` Equal to `k`**  
    Setting `fetch_k = 3` and `k = 3` completely disables MMR diversity! MMR has zero alternative candidates to swap in. Always ensure `fetch_k >= 3 * k`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Arbitrary Score Thresholds Without Metric Calibration**  
    Setting `score_threshold = 0.8` assuming cosine similarity when your vector space is L2 distance causes the retriever to return 0 results for every query. Verify whether scores represent distance or similarity.
    """)
    st.warning("""
    **Trap 4: Discarding Distance Scores in Debugging**  
    Relying only on `.similarity_search()` hides why a particular document won. Use `.similarity_search_with_score()` during development to inspect exact numerical calibration.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Maximal Marginal Relevance?"):
        st.markdown("""
        **MMR Purpose**:  
        Selects documents that maximize query relevance while penalizing similarity to already-selected documents to eliminate redundancy.
        """)
    with st.expander("🗂️ Flashcard: What does lambda_mult control in MMR?"):
        st.markdown("""
        **Lambda Trade-Off**:  
        `1.0` = Pure query similarity; `0.0` = Maximal diversity. Industry sweet spot is `0.6 - 0.7`.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is fetch_k in MMR?"):
        st.markdown("""
        **Candidate Pool**:  
        The number of initial nearest neighbors retrieved before MMR applies the diversity filter (recommended: `4 * k`).
        """)
    with st.expander("🗂️ Flashcard: What does similarity_score_threshold do?"):
        st.markdown("""
        **Relevance Gatekeeper**:  
        Drops any document with similarity below the threshold, suppressing hallucinations on out-of-domain questions.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the difference between standard similarity search, MMR, and score thresholding.", value=True)
chk2 = st.checkbox("I know how the MMR mathematical objective equation balances relevance against diversity.", value=True)
chk3 = st.checkbox("I can configure `k`, `fetch_k`, and `lambda_mult` in LangChain's `.as_retriever()`.", value=True)
chk4 = st.checkbox("I know how to pass composite metadata filters (`where`) inside LangChain `search_kwargs`.", value=True)
chk5 = st.checkbox("I understand how score thresholding acts as a gatekeeper against out-of-domain hallucinations.", value=True)
chk6 = st.checkbox("I can implement an enterprise-grade diverse retrieval pipeline in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completely mastered Chapter 15: LangChain + Vector Databases and are ready for Chapter 16: Advanced Retrieval Techniques!")
