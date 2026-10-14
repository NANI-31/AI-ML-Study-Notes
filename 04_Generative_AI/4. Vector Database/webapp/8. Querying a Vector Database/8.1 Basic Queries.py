import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="8.1 Basic Queries - Vector DB Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("8.1 Basic Queries: Query by Text, Embedding, ID & Top-K Retrieval")
st.caption("Chapter 8: Querying a Vector Database | Domain: 4. Vector Database")
st.markdown("""
Querying is the primary purpose of any vector database. Once documents and vectors are indexed, how you query 
dictates retrieval precision, system latency, and downstream LLM generation quality.

In ChromaDB and modern vector databases, querying is not limited to passing text strings into a search bar. 
There are **three distinct query ingress modalities**:
1. **Query by Text (`query_texts`)**: The natural language interface where the database automatically vectorizes your query string.
2. **Query by Embedding (`query_embeddings`)**: Passing pre-computed vector arrays directly—mandatory for multimodal image search and decoupled GPU pipelines.
3. **Query by ID (`get` & "More Like This")**: Point lookups and chaining item-to-item similarity recommendations ("Find documents similar to Document X").
4. **Top-K Candidate Collection (`n_results`)**: Tuning candidate depth for RAG context hydration.

In this lesson, we master all three query pathways, analyze candidate collection mechanics, and build an interactive 
query workbench.
""")

st.divider()

# Section 1: The 3 Query Ingress Modalities
st.header("1. The 3 Query Ingress Modalities Dissected", divider="rainbow")

col_qm1, col_qm2 = col_qm1, col_qm2 = st.columns([1, 1])

with col_qm1:
    st.markdown("""
    ### 1. Query by Text (`query_texts=[...]`)
    * **How it works**:
      ```python
      results = collection.query(
          query_texts=["How to configure SAML SSO?"],
          n_results=3
      )
      ```
    * **Mechanism**:
      * The collection's bound `embedding_function` encodes the text into a dense vector `q in R^d`.
      * The vector navigates the HNSW graph to find the `n_results` nearest neighbors.
    * **Note**: Input is a **list of strings**, enabling multi-query batching in a single call.
    
    ### 2. Query by Embedding (`query_embeddings=[...]`)
    * **How it works**:
      ```python
      results = collection.query(
          query_embeddings=[query_vector],
          n_results=3
      )
      ```
    * **Mechanism**:
      * Bypasses the embedding function entirely! The raw float array is passed straight to the C++ HNSWlib index.
    * **Why It is Essential**:
      * **Multimodal Search**: Searching text documents using an image query embedding (via CLIP).
      * **Compute Decoupling**: Query vector is computed on a dedicated inference server.
    """)

with col_qm2:
    st.markdown("""
    ### 3. Query by ID (Point Lookup & "More Like This")
    * **Exact Point Lookup (`collection.get`)**:
      ```python
      # Fetches specific record in O(1) time without vector search
      doc = collection.get(ids=["doc_101"])
      ```
    * **"More Like This" Recommender Pattern**:
      * How do you find articles related to Document `doc_101`?
      1. Fetch Document `doc_101`'s embedding:
         `item = collection.get(ids=["doc_101"], include=["embeddings"])`.
      2. Query using that embedding:
         `results = collection.query(query_embeddings=item["embeddings"], n_results=4)`.
      3. Filter out the seed document itself (which will always match with distance 0.0) to get the top 3 recommendations!
    
    ### 4. Top-K Retrieval (`n_results`)
    * Controls the size of the result set.
    * Default is `10`. For RAG context injection, `3` to `5` is typically optimal.
    """)

st.markdown("### The 3 Query Ingress Paths Flowchart")
query_flow_dot = """
digraph QueryIngressFlow {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    subgraph cluster_path_text {
        label="Path 1: Text Query"; style="dashed"; color="#2563EB";
        TextIn [label="Query Text String\\n'How to configure SSO?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        EmbedFunc [label="Embedding Function\\n(MiniLM / OpenAI)", fillcolor="#DBEAFE"];
        TextIn -> EmbedFunc;
    }

    subgraph cluster_path_vec {
        label="Path 2: Direct Vector"; style="dashed"; color="#059669";
        VecIn [label="Pre-Computed Vector\\n[0.12, -0.44, ..., 0.89]", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
    }

    subgraph cluster_path_id {
        label="Path 3: Point Lookup / More Like This"; style="dashed"; color="#7C3AED";
        IDIn [label="Document ID\\n'doc_101'", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
        GetVector [label="collection.get(include=['embeddings'])\\nFetch seed vector", fillcolor="#DDD6FE"];
        IDIn -> GetVector;
    }

    HNSWIndex [label="HNSWlib Graph Index\\nSub-5ms ANN Search", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
    TopKOut [label="Top-K Nearest Records\\n{ids, distances, documents, metadatas}", fillcolor="#6EE7B7", color="#047857", style="filled,bold"];

    EmbedFunc -> HNSWIndex [label="query_texts"];
    VecIn -> HNSWIndex [label="query_embeddings"];
    GetVector -> HNSWIndex [label="'More Like This'"];
    HNSWIndex -> TopKOut;
}
"""
st.graphviz_chart(query_flow_dot, use_container_width=False)

st.divider()

# Section 2: Tuning Top-K (`n_results`) for Downstream RAG
st.header("2. Tuning Top-K (`n_results`): The Precision vs. Hallucination Trade-Off", divider="rainbow")

st.markdown("""
What is the optimal value for `n_results`? 
Setting Top-K too small causes **information deficiency**, while setting it too large causes **context poisoning**:
""")

col_k1, col_k2, col_k3 = st.columns(3)

with col_k1:
    st.markdown("""
    #### Under-fetching (`k = 1`)
    * **Risk**:
      * A single chunk rarely contains complete context for multi-part questions (e.g., comparing two features).
    * **Consequence**:
      * The LLM answers partially or hallucinates missing details.
    * **Best For**:
      * Strict FAQ deflection where a single definitive answer exists.
    """)

with col_k2:
    st.markdown("""
    #### Production Sweet Spot (`k = 3 - 5`)
    * **Advantage**:
      * Supplies complementary perspectives, background definitions, and edge cases.
    * **Token Economy**:
      * 3 chunks of 400 tokens = 1,200 tokens (fits comfortably within LLM budget).
    * **Best For**:
      * Standard enterprise RAG Q&A pipelines.
    """)

with col_k3:
    st.markdown("""
    #### Over-fetching (`k = 20 - 50`)
    * **Risk**:
      * Low-similarity noisy chunks are injected into the context window.
    * **"Lost in the Middle"**:
      * LLMs prioritize tokens at the very beginning and very end of prompts, ignoring facts buried in chunk 15!
    * **Best For**:
      * Initial candidate retrieval prior to Cross-Encoder Reranking.
    """)

st.divider()

# Section 3: Interactive Query Workbench
st.header("3. Interactive Chroma Query Workbench", divider="rainbow")

st.markdown("""
Test all three query pathways against an active technical documentation collection. 
Observe **text vectorization**, **direct vector queries**, and **"More Like This" recommendations**:
""")

# Sample Knowledge Base
query_kb_data = [
    {"id": "doc_sso_okta", "text": "Configure Single Sign-On (SSO) with Okta identity provider using SAML 2.0 assertions.", "dept": "Security", "year": 2024},
    {"id": "doc_sso_azure", "text": "Azure Active Directory enterprise SSO setup with OpenID Connect (OIDC).", "dept": "Security", "year": 2024},
    {"id": "doc_db_pg", "text": "PostgreSQL connection pool optimization using PgBouncer transaction pooling.", "dept": "Database", "year": 2023},
    {"id": "doc_db_mysql", "text": "MySQL InnoDB buffer pool sizing and dirty page flush tuning.", "dept": "Database", "year": 2023},
    {"id": "doc_k8s_hpa", "text": "Kubernetes Horizontal Pod Autoscaler deployment with custom Prometheus metrics.", "dept": "DevOps", "year": 2024}
]

col_q_in, col_q_out = st.columns([1, 1])

with col_q_in:
    st.subheader("Query Configuration")
    query_modality = st.radio("Choose Query Pathway", ["Path A: Query by Text (`query_texts`)", "Path B: Query by Embedding (`query_embeddings`)", "Path C: Query by ID ('More Like This' Recommender)"])
    k_choice = st.slider("Top-K Candidates (`n_results`)", 1, 4, 2)
    
    if "Path A" in query_modality:
        user_text = st.selectbox(
            "Natural Language Query Text",
            options=[
                "How do I set up SSO with Okta?",
                "Tuning database buffer pools and connections",
                "Autoscaling containers in Kubernetes clusters"
            ]
        )
    elif "Path B" in query_modality:
        st.markdown("**Simulated 4-Dimensional Vector Input:**")
        v_0 = st.slider("Dimension 1", -1.0, 1.0, 0.9, 0.1)
        v_1 = st.slider("Dimension 2", -1.0, 1.0, 0.1, 0.1)
        v_2 = st.slider("Dimension 3", -1.0, 1.0, 0.0, 0.1)
        v_3 = st.slider("Dimension 4", -1.0, 1.0, 0.0, 0.1)
    else: # Path C
        seed_id = st.selectbox("Select Seed Document ID to Find Similar Items", [d["id"] for d in query_kb_data])
        seed_doc = next(d for d in query_kb_data if d["id"] == seed_id)
        st.info(f"**Seed Document:** `{seed_doc['text']}`")

with col_q_out:
    st.subheader("Query Execution & Results")
    
    if "Path A" in query_modality:
        code_call = f"""results = collection.query(
    query_texts=["{user_text}"],
    n_results={k_choice}
)"""
        st.code(code_call, language="python")
        
        # Simulate scoring
        scored = []
        for d in query_kb_data:
            if "SSO" in user_text and "SSO" in d["text"]:
                sim = 0.94 if "Okta" in d["text"] and "Okta" in user_text else 0.88
            elif "database" in user_text and "pool" in d["text"]:
                sim = 0.92 if "PostgreSQL" in d["text"] else 0.89
            elif "Autoscaling" in user_text and "Autoscaler" in d["text"]:
                sim = 0.95
            else:
                sim = 0.25
            scored.append((d, 1.0 - sim, sim))
            
        scored.sort(key=lambda x: x[1]) # Ascending distance
        for idx, (doc, dist, sim) in enumerate(scored[:k_choice], 1):
            st.markdown(f"**Rank {idx}:** `{doc['id']}` | **Distance:** `{dist:.4f}` (Sim: `{sim:.4f}`)")
            st.markdown(f"*Text:* `{doc['text']}`")
            st.markdown("---")
            
    elif "Path B" in query_modality:
        code_call = f"""# Direct Vector Ingress
results = collection.query(
    query_embeddings=[[{v_0:.1f}, {v_1:.1f}, {v_2:.1f}, {v_3:.1f}]],
    n_results={k_choice}
)"""
        st.code(code_call, language="python")
        st.success("SIMD Distance Kernel evaluated directly against HNSW index without embedding inference!")
        for idx in range(min(k_choice, len(query_kb_data))):
            doc = query_kb_data[idx]
            st.markdown(f"**Rank {idx + 1}:** `{doc['id']}` | Simulated Distance: `0.{idx * 3 + 1:04d}`")
            st.markdown(f"*Text:* `{doc['text']}`")
            st.markdown("---")
            
    else: # Path C - More Like This
        code_call = f"""# 1. Fetch seed document vector
seed = collection.get(ids=["{seed_id}"], include=["embeddings"])

# 2. Query for similar items (excluding self)
results = collection.query(
    query_embeddings=seed["embeddings"],
    n_results={k_choice + 1} # Fetch k+1 to allow filtering self
)
recommendations = [id for id in results["ids"][0] if id != "{seed_id}"][:{k_choice}]"""
        st.code(code_call, language="python")
        
        # Display recommendations (excluding seed)
        rec_candidates = [d for d in query_kb_data if d["id"] != seed_id and d["dept"] == seed_doc["dept"]]
        if not rec_candidates:
            rec_candidates = [d for d in query_kb_data if d["id"] != seed_id]
            
        st.markdown(f"**Top Recommended Items Similar to `{seed_id}`:**")
        for idx, doc in enumerate(rec_candidates[:k_choice], 1):
            st.markdown(f"**Recommendation {idx}:** `{doc['id']}` (Dept: `{doc['dept']}`)")
            st.markdown(f"*Text:* `{doc['text']}`")
            st.markdown("---")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Comprehensive Chroma Query Suite", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates:
1. Querying by **Text** (`query_texts`).
2. Querying by **Pre-Computed Vector** (`query_embeddings`).
3. Point Lookup (`collection.get`).
4. The **"More Like This"** item recommendation pattern:
""")

code_query_suite = '''import chromadb
from chromadb.config import Settings
import numpy as np

# Initialize client
client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
col = client.create_collection("tech_docs", metadata={"hnsw:space": "cosine"})

# Ingest 4 documents
col.add(
    documents=[
        "Configure SAML 2.0 Single Sign-On with Okta.",
        "Azure Active Directory enterprise SSO setup.",
        "PostgreSQL connection pooling with PgBouncer.",
        "Kubernetes Horizontal Pod Autoscaler deployment."
    ],
    metadatas=[
        {"dept": "Security", "tech": "Okta"},
        {"dept": "Security", "tech": "Azure"},
        {"dept": "Database", "tech": "Postgres"},
        {"dept": "DevOps", "tech": "K8s"}
    ],
    ids=["doc_okta", "doc_azure", "doc_pg", "doc_k8s"]
)

# 1. Query by Text
print("=== 1. Query by Text ===")
res_text = col.query(query_texts=["How to setup Okta SSO?"], n_results=1)
print(f"Top Text Match ID: {res_text['ids'][0][0]} | Distance: {res_text['distances'][0][0]:.4f}")

# 2. Point Lookup by ID
print("\\n=== 2. Exact Point Lookup by ID ===")
point = col.get(ids=["doc_pg"], include=["documents", "metadatas", "embeddings"])
print(f"Point Lookup Document: '{point['documents'][0]}' | Tech: {point['metadatas'][0]['tech']}")

# 3. "More Like This" Item Recommendation Pattern
print("\\n=== 3. 'More Like This' Recommender Pattern ===")
# Use doc_okta's vector to find related articles (excluding doc_okta itself)
seed_vector = point["embeddings"][0]
res_similar = col.query(query_embeddings=[seed_vector], n_results=2)
# Exclude self
similar_ids = [rid for rid in res_similar['ids'][0] if rid != "doc_pg"]
print(f"Items Most Similar to 'doc_pg': {similar_ids}")
'''

st.code(code_query_suite, language="python")

if st.button("🚀 Execute Chroma Query Suite Demo", type="primary"):
    exec_scope = {}
    exec(code_query_suite, globals(), exec_scope)
    
    top_txt = exec_scope["res_text"]["ids"][0][0]
    p_tech = exec_scope["point"]["metadatas"][0]["tech"]
    sim_recs = exec_scope["similar_ids"]
    
    st.success("Chroma Query Suite executed successfully!")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"**Query by Text Match:** `{top_txt}`")
        st.markdown(f"**Point Lookup Tech:** `{p_tech}`")
    with col_r2:
        st.markdown(f"**'More Like This' Recommended IDs:** `{sim_recs}`")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the fundamental operational difference between `collection.get()` and `collection.query()` in ChromaDB?"""
a1 = """* **`collection.get()` (Point / Filtered Retrieval)**:
  * Executes a deterministic lookup against SQLite by ID (`ids=["doc_1"]`) or by metadata filter (`where={"dept": "HR"}`).
  * Does **NOT** execute vector distance calculations or traverse the HNSW graph.
  * Runs in sub-millisecond `O(1)` or `O(log N)` time.
* **`collection.query()` (Approximate Nearest Neighbor Search)**:
  * Requires query text or query vectors.
  * Traverses the C++ HNSWlib graph index to compute vector distances (Cosine, L2) across nearest neighbors.
  * Returns ranked top-k items sorted in ascending order of geometric distance."""

q2 = """2. How do you implement a 'More Like This' item recommendation engine using ChromaDB?"""
a2 = """* **The 3-Step Pattern**:
  1. Retrieve the target item's pre-computed embedding using `collection.get(ids=[target_id], include=["embeddings"])`.
  2. Query the collection using `collection.query(query_embeddings=item["embeddings"], n_results=K + 1)`.
  3. Filter out the seed `target_id` from the returned IDs (since an item's distance to itself is 0.0).
  4. The remaining K items represent the most geometrically similar items in the catalog."""

q3 = """3. Why is `query_embeddings` mandatory for Multimodal (Image-to-Text or Text-to-Image) retrieval?"""
a3 = """* ChromaDB's default embedding functions (like `all-MiniLM-L6-v2`) only accept text strings.
* If a user uploads an image photo to search an e-commerce catalog:
  * The image must be passed through a Vision Transformer (like CLIP ViT-B/32 or SigLIP) to produce a 512-dimensional vector.
  * You cannot pass an image into `query_texts`.
  * Passing the resulting visual vector into `collection.query(query_embeddings=[image_vec])` allows visual embeddings to search text documents in joint embedding space."""

q4 = """4. What happens if you pass a single string instead of a list of strings to `query_texts`?"""
a4 = """* In ChromaDB, `query_texts` expects a **`List[str]`**: `query_texts=["My query"]`.
* If a developer mistakenly passes a raw string: `query_texts="My query"`, Python treats strings as iterables of characters: `['M', 'y', ' ', 'q', ...]`.
* Chroma will attempt to execute a batch query over 8 individual single-character queries, resulting in bizarre multi-batch outputs or type errors!
* Always wrap query strings in square brackets: `query_texts=[query_str]`."""

q5 = """5. How does the choice of `n_results` (Top-K) impact Large Language Model generation quality in RAG?"""
a5 = """* **Precision vs. Recall Trade-off**:
  * Setting `k = 2 - 4` minimizes prompt token costs and focuses the LLM on high-confidence context.
  * Setting `k = 10 - 20` increases context recall, but triggers the **"Lost in the Middle"** phenomenon, where LLMs fail to attend to facts located in the middle of long prompts.
* **Production Best Practice**: Fetch `k = 20` candidates from ChromaDB, pass them through a Cross-Encoder Reranker (e.g. `bge-reranker-large`), and inject only the top 3-4 reranked chunks into the final LLM prompt."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Query Execution Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The "More Like This" Self-Match Leak**  
> When querying a collection using an existing document's vector, developers often request `n_results=3` 
> and display all 3 results. Rank 1 will **ALWAYS be the seed document itself** (distance = 0.0000)! 
> Always request `n_results = k + 1` and explicitly filter out the seed ID from recommendations.

> [!WARNING]
> **Trap 2: Dimension Mismatch on Manual `query_embeddings`**  
> If a collection was built with 768-dimensional embeddings, but a query script mistakenly passes 
> a 1536-dimensional vector from OpenAI `text-embedding-3-small`, the query immediately crashes in C++ HNSWlib. 
> Always validate that `len(query_vector) == collection.get()['embeddings'][0].shape[0]`.

> [!WARNING]
> **Trap 3: Forgetting that Chroma Returns Nested Lists**  
> ChromaDB query results return **lists of lists** (to support batch multi-querying): 
> `results["ids"] = [["doc_1", "doc_2"]]`. 
> Accessing `results["ids"][0]` returns the first query's hits, NOT the single hit string. 
> Writing `for doc in results["documents"]:` instead of `results["documents"][0]` causes list-of-list iteration bugs!
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: `query_texts` vs `query_embeddings`?"):
        st.markdown("""
        **Query Modalities**:  
        `query_texts` auto-embeds strings using registered model; `query_embeddings` passes raw float vectors directly to HNSW.
        """)
    with st.expander("🗂️ Flashcard: What is `collection.get()`?"):
        st.markdown("""
        **Point Retrieval**:  
        Fetches records by ID or metadata filter in O(1) time without executing vector distance search.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to do 'More Like This'?"):
        st.markdown("""
        **Item Recommendations**:  
        Fetch seed item's embedding via `get(include=['embeddings'])`, query using that vector, and exclude seed ID from results.
        """)
    with st.expander("🗂️ Flashcard: Why is Result Structure Nested?"):
        st.markdown("""
        **Nested Results**:  
        Chroma returns `results["ids"] = [batch_1_ids, batch_2_ids]` to support multi-query batching in a single call.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to execute queries by text using `query_texts`.", value=True)
chk2 = st.checkbox("I understand when and why to execute queries by vector using `query_embeddings`.")
chk3 = st.checkbox("I can differentiate between `collection.get()` and `collection.query()`.")
chk4 = st.checkbox("I know how to implement the 'More Like This' item recommendation pattern.")
chk5 = st.checkbox("I understand the trade-offs of tuning `n_results` for RAG pipelines.")
chk6 = st.checkbox("I can implement a complete multi-modality query suite in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Basic Queries and are ready for Chapter 8.2: Query Parameters!")
