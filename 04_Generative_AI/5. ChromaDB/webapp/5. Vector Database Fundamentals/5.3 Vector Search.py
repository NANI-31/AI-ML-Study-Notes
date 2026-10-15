import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="5.3 Vector Search - ChromaDB Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("5.3 Vector Search: Exact vs Approximate Search, Top-K Retrieval, Ranking & Threshold Cutoffs")
st.caption("Chapter 5: Vector Database Fundamentals | Domain: 5. ChromaDB")
st.markdown("""
In the vector database lifecycle, all ingestion, embedding generation, and graph indexing operations exist 
to serve one ultimate objective: **Executing high-precision, low-latency Vector Search**.

Executing vector search in production requires understanding several foundational retrieval mechanics:
* **Exact k-NN vs. Approximate Nearest Neighbor (ANN)**: Balancing the mathematical certainty of exhaustive brute-force scans against the logarithmic speed of graph traversals.
* **Top-K Retrieval (`n_results=K`)**: Determining how many document candidates to retrieve for LLM context synthesis.
* **Dynamic Search Ranking**: How vectors are scored and ordered by ascending geometric distance.
* **Similarity Threshold Gating**: Pruning irrelevant candidates to protect LLMs from hallucinations when user queries drift outside the indexed knowledge base.

In this lesson, we analyze search algorithms, explore threshold calibration techniques, implement production 
search filters in Python, and test an interactive Vector Search Simulator and Threshold Gating Arena.
""")

st.divider()

# Section 1: Exact k-NN vs. Approximate Nearest Neighbor (ANN)
st.header("1. Exact k-NN vs. Approximate Nearest Neighbor (ANN)", divider="rainbow")

col_search1, col_search2 = col_search1, col_search2 = st.columns([1.1, 0.9])

with col_search1:
    st.markdown("""
    ### Algorithm Comparison
    
    * **1. Exact k-NN (Exhaustive Brute-Force Scan)**:
      * *Mechanism*: Computes the distance between the query vector and **every single vector in the entire database**.
      * *Recall*: **100.0% Exact Ground Truth**.
      * *Complexity*: `O(N * D)`.
      * *Production Reality*: Viable only for toy datasets (< 10,000 vectors). Completely unusable for interactive applications at enterprise scale.
    
    * **2. Approximate Nearest Neighbor (ANN via HNSW)**:
      * *Mechanism*: Traverses a multi-layer navigable small world graph, greedily hopping between neighboring nodes in direction of greatest similarity.
      * *Recall*: **96.0% – 99.8% Configurable Recall**.
      * *Complexity*: `O(log N)` Logarithmic Time.
      * *Production Reality*: Evaluates only 100–300 candidate nodes regardless of whether the collection has 100,000 or 10,000,000 vectors! Sub-10ms response time.
    """)

with col_search2:
    st.markdown("### Search Traversal Flowchart")
    search_dot = """
    digraph SearchTraversal {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Query [label="User Query: 'Database recovery'\\nVector: [0.12, -0.45, ...]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_hnsw {
            label="HNSW Graph Traversal (O(log N))"; style="dashed"; color="#059669";
            Layer2 [label="Layer 2: Expressway Hop\\nIdentifies Database Cluster", fillcolor="#D1FAE5"];
            Layer1 [label="Layer 1: Intermediate Hop\\nRefines Near Neighbors", fillcolor="#A7F3D0"];
            Layer0 [label="Layer 0: Dense Base Graph\\nPinpoints Exact Top-K Nodes", fillcolor="#6EE7B7", style="filled,bold"];
        }

        TopK [label="Top-K Candidates\\nSorted by Ascending Distance", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        ThresholdFilter [label="Threshold Gating Filter\\nDiscard any distance > 0.35!", fillcolor="#EDE9FE", color="#7C3AED"];

        LLMPrompt [label="Clean Grounded Context\\nPassed to Downstream LLM", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        Query -> Layer2 -> Layer1 -> Layer0 -> TopK -> ThresholdFilter -> LLMPrompt;
    }
    """
    st.graphviz_chart(search_dot, use_container_width=False)

st.divider()

# Section 2: Tuning Search: Top-K & Distance Threshold Gating
st.header("2. Tuning Search: Top-K & Distance Threshold Gating", divider="rainbow")

col_gate1, col_gate2 = col_gate1, col_gate2 = st.columns([1, 1])

with col_gate1:
    st.markdown("""
    ### Sizing `n_results` (Top-K)
    * **The Trade-Off**:
      * *Too Small (`K=1` or `2`)*: Risk missing critical context if the relevant answer was split across multiple chunks.
      * *Too Large (`K=20` or `50`)*: Suffers from the "Lost in the Middle" phenomenon where LLMs ignore facts buried deep in context; increases prompt token costs.
    * **Industry Rule of Thumb**:
      * General QA: **`K = 3 to 5`**.
      * Multi-document summarization: **`K = 8 to 12`**.
    
    ### The Similarity Threshold Shield
    By default, Chroma returns the K closest vectors even if the best match is completely unrelated (e.g., distance = `0.82`). 
    Applying client-side distance gating shields your application from hallucinations:
    """)

with col_gate2:
    st.markdown("### Production Search & Gating Pattern")
    st.code("""import chromadb

def search_with_threshold_gating(
    collection,
    query_text: str,
    top_k: int = 5,
    max_distance_threshold: float = 0.35
):
    \"\"\"Queries ChromaDB and filters out low-confidence hallucinations.\"\"\"
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    gated_results = []
    # Unpack 2D results list
    for doc_id, doc, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0]
    ):
        if dist <= max_distance_threshold:
            gated_results.append({
                "id": doc_id,
                "document": doc,
                "distance": dist,
                "similarity": 1.0 - dist
            })
            
    return gated_results
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Advanced Search Operations
st.header("3. Clean Python Implementation of Advanced Search Operations", divider="rainbow")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="enterprise_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# Ingest sample records
collection.upsert(
    ids=["k8s-01", "pg-01", "sec-01"],
    documents=[
        "Kubernetes StatefulSets maintain persistent NVMe storage on pod eviction.",
        "PostgreSQL WAL archives stream continuously to S3 for disaster recovery.",
        "Zero-trust network architecture requires mutual TLS on all internal RPCs."
    ],
    metadatas=[{"tier": "infra"}, {"tier": "database"}, {"tier": "security"}]
)

# 1. Execute Top-2 Vector Search with Metadata Pre-Filtering
search_results = collection.query(
    query_texts=["How do we recover databases after hardware failure?"],
    n_results=2,
    where={"tier": "database"},  # Pre-filter candidate nodes
    include=["documents", "metadatas", "distances"]  # Explicit payload inclusion
)

# 2. Iterate and display ranked results
print(f"Query returned {len(search_results['ids'][0])} candidates:")
for rank, (doc_id, text, dist) in enumerate(zip(
    search_results["ids"][0],
    search_results["documents"][0],
    search_results["distances"][0]
), start=1):
    sim = max(0.0, 1.0 - dist) * 100
    print(f"Rank {rank}: [{doc_id}] Dist={dist:.4f} (Sim={sim:.1f}%) | {text}")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Search Simulator & Threshold Gating Arena
st.header("4. Interactive Workbench: Search Simulator & Threshold Gating Arena", divider="rainbow")

st.markdown("""
Simulate querying a knowledge base with tunable Top-K (`n_results`) and a strict 
**Cosine Distance Cutoff Threshold**. Observe which candidate documents are accepted for LLM context 
and which are rejected.
""")

col_sim_q, col_sim_cfg = col_sim_q, col_sim_cfg = st.columns([1.1, 0.9])

with col_sim_q:
    st.subheader("1. Inbound Query & Dataset")
    user_query = st.text_input("Search Query (`query_texts`):", value="How do we back up databases to cloud storage?")
    
    DOCUMENTS_VAULT = [
        {"id": "doc_db_01", "text": "PostgreSQL WAL archives stream continuously to encrypted S3 for disaster recovery.", "dist": 0.082},
        {"id": "doc_db_02", "text": "Automated snapshot lifecycle schedules daily backup dumps of customer databases.", "dist": 0.145},
        {"id": "doc_k8s_01", "text": "Kubernetes StatefulSets mount persistent EBS volumes on pod restart.", "dist": 0.380},
        {"id": "doc_sec_01", "text": "OAuth 2.0 refresh tokens expire after 30 days of inactivity.", "dist": 0.620},
        {"id": "doc_cul_01", "text": "Classic Italian spaghetti bolognese recipe with slow-cooked ragu.", "dist": 0.890}
    ]

with col_sim_cfg:
    st.subheader("2. Retrieval & Gating Parameters")
    param_k = st.slider("Top-K Candidates (`n_results`):", min_value=1, max_value=5, value=4)
    param_cutoff = st.slider("Max Allowed Distance (Cutoff Threshold):", min_value=0.05, max_value=0.90, value=0.35, step=0.05)

# Slice Top-K
raw_top_k = DOCUMENTS_VAULT[:param_k]

st.divider()

col_raw_view, col_gated_view = col_raw_view, col_gated_view = st.columns([1, 1])

with col_raw_view:
    st.subheader(f"Raw Database Top-{param_k} Results")
    st.caption("What ChromaDB returns prior to application-level threshold gating.")
    for rank, doc in enumerate(raw_top_k, start=1):
        is_passed = doc["dist"] <= param_cutoff
        badge = "✅ PASSED" if is_passed else "❌ REJECTED"
        st.markdown(f"**#{rank}: [{doc['id']}]** (Dist: `{doc['dist']:.3f}`) — {badge}")
        st.text(doc["text"])

with col_gated_view:
    st.subheader("Grounded LLM Prompt Context")
    st.caption("Final clean context assembled after applying distance gating.")
    
    accepted_docs = [d for d in raw_top_k if d["dist"] <= param_cutoff]
    
    if accepted_docs:
        context_str = "\n\n".join([f"[{d['id']}]: {d['text']}" for d in accepted_docs])
        st.code(f"""You are a helpful assistant. Use ONLY the following context to answer:

{context_str}

User Question: {user_query}
""", language="markdown")
        st.success(f"🟢 **Safe Context Assembly**: Passed {len(accepted_docs)} high-confidence documents to LLM! Filtered out {len(raw_top_k) - len(accepted_docs)} irrelevant passages.")
    else:
        st.error("🔴 **ALL CANDIDATES REJECTED**: Distance exceeded threshold! System gracefully responds: *'I could not find relevant documentation to answer this question.'* (Zero Hallucination Risk!)")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the 'Lost in the Middle' problem, and how does it affect the choice of `n_results`?"):
    st.markdown("""
    **Answer**:
    * Research (Liu et al., Stanford) demonstrated that LLMs are most effective at retrieving and reasoning over facts located at the **very beginning** or **very end** of their input context window.
    * When engineers retrieve large numbers of documents (`n_results=20`), critical facts placed in the middle of the prompt are frequently overlooked by the LLM's attention mechanism.
    * **Best Practice**: Keep `n_results` lean (typically `3 to 5` high-precision chunks), or use a **Cross-Encoder Reranker** to reorder and condense context before prompting.
    """)

with st.expander("Q2: Why must distance threshold filtering happen on the client application layer rather than inside ChromaDB?"):
    st.markdown("""
    **Answer**:
    * Currently, ChromaDB's core `collection.query()` API accepts `n_results=K`, returning the Top-K nearest neighbors.
    * It does not currently expose a native `max_distance` argument in the query call.
    * Therefore, the standard engineering pattern is to fetch `n_results=K` and apply client-side list comprehension filtering:
      `[doc for doc, dist in zip(docs, dists) if dist <= threshold]`.
    """)

with st.expander("Q3: What happens if a user's query returns fewer results than `n_results`?"):
    st.markdown("""
    **Answer**:
    * If a collection contains 3 documents and you request `n_results=5`, ChromaDB does NOT throw an error.
    * It returns all 3 available documents without padding.
    * If metadata filtering (`where={...}`) matches fewer documents than `n_results`, Chroma returns only the subset of matching documents.
    """)

with st.expander("Q4: What is the difference between Pre-Filtering and Post-Filtering in vector databases?"):
    st.markdown("""
    **Answer**:
    * **Pre-Filtering (Supported by ChromaDB)**: The metadata filter is evaluated **before or during** graph traversal. Only nodes matching the metadata predicate are considered candidate neighbors. (Guarantees K valid results).
    * **Post-Filtering**: The vector index retrieves the Top-K nearest neighbors purely by geometry, and then discards any results that fail metadata filters.
      * *Major Flaw*: If 90% of the Top-K candidates fail the metadata filter, post-filtering returns an empty or insufficient result set!
    """)

with st.expander("Q5: How does the `include` parameter optimize search latency and memory?"):
    st.markdown("""
    **Answer**:
    * By default, `collection.query()` returns `include=["documents", "metadatas", "distances"]`.
    * If you only need document IDs to query another system, specifying `include=["distances"]` skips reading SQLite document text rows into memory, reducing JSON payload size and accelerating response times by 30%–50%.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Requesting `n_results=50` to Be 'Safe'**  
    Stuffing 50 chunks into an LLM prompt degrades attention accuracy (Lost in the Middle) and multiplies API token costs by 10x. Use Top-3 to 5.
    """)
    st.warning("""
    **Trap 2: Omitting Threshold Cutoffs on Open-Domain Queries**  
    Allowing out-of-domain queries to pass irrelevant vectors with distance > 0.6 into LLM prompts forces the model to hallucinate false answers.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Assuming Similarity Score is Uniform Across Models**  
    A cosine distance of `0.2` on OpenAI embeddings may correspond to a `0.35` distance on MiniLM. Calibrate thresholds for each model independently.
    """)
    st.warning("""
    **Trap 4: Missing Error Handling for Empty Gated Results**  
    If all retrieved documents are filtered out by your distance cutoff, ensure your application handles empty lists gracefully without crashing.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Top-K retrieval?"):
        st.markdown("""
        **Ranked Slicing**:  
        Retrieves the K nearest vector candidates ordered by ascending geometric distance.
        """)
    with st.expander("🗂️ Flashcard: What is Lost in the Middle?"):
        st.markdown("""
        **Attention Degradation**:  
        LLMs struggle to recall facts buried in the middle of excessively large prompt contexts.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why gate by distance threshold?"):
        st.markdown("""
        **Hallucination Shield**:  
        Prunes irrelevant vector matches before LLM prompt assembly to avoid generating false facts.
        """)
    with st.expander("🗂️ Flashcard: What is Pre-Filtering?"):
        st.markdown("""
        **Metadata Filtering**:  
        Applies SQL-like metadata filters during graph traversal, guaranteeing valid Top-K hits.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the performance differences between Exact k-NN scans and HNSW ANN search.", value=True)
chk2 = st.checkbox("I understand the trade-offs of sizing `n_results` and how 'Lost in the Middle' degrades LLM attention.", value=True)
chk3 = st.checkbox("I know how to implement client-side distance threshold gating to prevent hallucinations.", value=True)
chk4 = st.checkbox("I understand how Pre-Filtering differs from Post-Filtering in vector query execution.", value=True)
chk5 = st.checkbox("I know how to use the `include` parameter to optimize network payload sizes.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 6: Adding Data to ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 5: Vector Database Fundamentals in ChromaDB!")
