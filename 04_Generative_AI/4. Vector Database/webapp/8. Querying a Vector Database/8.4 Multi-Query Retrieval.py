import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

st.set_page_config(
    page_title="8.4 Multi-Query Retrieval - Vector DB Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("8.4 Multi-Query Retrieval & Practical Query Patterns")
st.caption("Chapter 8: Querying a Vector Database | Domain: 4. Vector Database")
st.markdown("""
In real-world generative AI and search applications, a single user prompt is frequently **ambiguous, undertuned, or multi-faceted**. 
Executing a single vector query (`query_texts=[user_query]`) often leads to **retrieval blind spots**—missing crucial context chunks 
that use alternative phrasing, specialized vocabulary, or sub-topic facets.

**Multi-Query Retrieval** solves this by:
1. **Batching Multiple Vector Queries**: Passing a list of query texts (`query_texts=["Q1", "Q2", "Q3"]`) in a single network round-trip.
2. **Query Expansion & Decomposition**: Generating synonyms, sub-questions, and hypotheticals (HyDE) to cover distinct semantic angles.
3. **Reciprocal Rank Fusion (RRF)**: Merging the multi-list search results without requiring normalized distance score calibration.
4. **Targeted Search Domains**: Applying vector querying to similar documents, questions (FAQs), relevant context (RAG), related products, and user personas.

This lesson explores how multi-query vector retrieval works under the hood, examines practical enterprise patterns, and provides an 
interactive Reciprocal Rank Fusion workbench.
""")

st.divider()

# Section 1: The Multi-Query Vector Retrieval Architecture
st.header("1. The Multi-Query Vector Retrieval Architecture", divider="rainbow")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    ### Why Single-Vector Queries Fail in Production
    * **Lexical and Semantic Mismatch**: A user asking *"How do I fix a leaky pipeline?"* might miss chunks titled *"Mitigating water line micro-fractures"* or *"Resolving CI/CD runner timeout in GitHub Actions"*.
    * **Multi-Intent Questions**: *"Compare PostgreSQL vector performance against ChromaDB and calculate RAM footprint"* contains two orthogonal sub-questions requiring distinct document clusters.
    * **High Distance Variance**: Different embedding models produce widely fluctuating distance distributions across distinct semantic domains.

    ### The ChromaDB Batch Query Interface
    ChromaDB natively supports querying multiple targets simultaneously:
    ```python
    results = collection.query(
        query_texts=[
            "What is vector indexing?",
            "How does HNSW graph navigation work?",
            "Approximate nearest neighbor graph algorithms"
        ],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    # results['ids'] has length 3, each containing top-3 IDs
    ```
    """)

with col_arch2:
    st.markdown("### Multi-Query & Rank Fusion Pipeline")
    arch_dot = """
    digraph MultiQueryFlow {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        User [label="User Query\\n'Optimize vector latency'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Expander [label="LLM Query Expander\\nGenerate Q1, Q2, Q3", fillcolor="#FEF3C7", color="#D97706"];

        Q1 [label="Q1: 'HNSW M & efSearch tuning'", fillcolor="#E0E7FF"];
        Q2 [label="Q2: 'Quantization & RAM reduction'", fillcolor="#E0E7FF"];
        Q3 [label="Q3: 'Vector index latency benchmarks'", fillcolor="#E0E7FF"];

        Engine [label="Vector Database Engine\\nBatch HNSW Traversal", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        R1 [label="Top-K for Q1", fillcolor="#F3F4F6"];
        R2 [label="Top-K for Q2", fillcolor="#F3F4F6"];
        R3 [label="Top-K for Q3", fillcolor="#F3F4F6"];

        RRF [label="Reciprocal Rank Fusion\\nRRF Score Aggregation", fillcolor="#FCE7F3", color="#DB2777", style="filled,bold"];
        Final [label="Fused Top-K Chunks\\nPassed to Generator", fillcolor="#C7D2FE", color="#4338CA"];

        User -> Expander;
        Expander -> Q1;
        Expander -> Q2;
        Expander -> Q3;
        Q1 -> Engine;
        Q2 -> Engine;
        Q3 -> Engine;
        Engine -> R1;
        Engine -> R2;
        Engine -> R3;
        R1 -> RRF;
        R2 -> RRF;
        R3 -> RRF;
        RRF -> Final;
    }
    """
    st.graphviz_chart(arch_dot, use_container_width=False)

st.divider()

# Section 2: Reciprocal Rank Fusion (RRF) Mathematical Formulation
st.header("2. Reciprocal Rank Fusion (RRF) Formulation", divider="rainbow")

st.markdown("""
When combining results from multiple distinct queries (or hybrid sparse and dense queries), raw distance scores cannot be directly averaged:
1. Query 1 might retrieve documents at distance 0.12 (dense tight cluster).
2. Query 2 might retrieve documents at distance 0.45 (broader conceptual cluster).
3. Directly averaging or sorting by raw distance introduces severe bias toward the narrower query.

### The Reciprocal Rank Fusion (RRF) Formula
RRF circumvents score incompatibility by discarding raw distance magnitudes and utilizing **ordinal ranks**:

```text
RRF_Score(d) = Sum over all queries q: [ 1 / (k + rank_q(d)) ]
```

Where:
* `d` is a retrieved document chunk.
* `rank_q(d)` is the 1-based rank position of document `d` in the result list for query `q` (e.g., 1 for 1st place, 2 for 2nd place).
* `k` is a smoothing constant (standard industry default: **k = 60**). The smoothing constant prevents top-ranked documents in a single list from overwhelmingly dominating documents that appear consistently across multiple lists.
* If document `d` does not appear in query `q`'s top-K list, its contribution for that query is 0.
""")

col_rrf_ex1, col_rrf_ex2 = st.columns([1, 1])

with col_rrf_ex1:
    st.markdown("""
    #### Why k = 60 is the Standard Default
    * At `k = 60`, rank 1 yields score `1 / (60 + 1) = 0.01639`.
    * Rank 2 yields `1 / (60 + 2) = 0.01613`.
    * Rank 10 yields `1 / (60 + 10) = 0.01429`.
    * **Impact**: A document appearing at rank 5 across all 3 query variations achieves score `3 * (1 / 65) = 0.04615`, easily outscoring a document that ranked #1 in only one query and completely missed the other two (`0.01639`).
    * **Consensus-Driven Ranking**: RRF naturally rewards multi-angle consensus and semantic relevance stability.
    """)

with col_rrf_ex2:
    st.markdown("""
    #### Python Implementation of RRF
    ```python
    def reciprocal_rank_fusion(
        query_results: List[List[str]], 
        k: int = 60
    ) -> List[Tuple[str, float]]:
        rrf_scores = defaultdict(float)
        
        for q_list in query_results:
            for rank_0_idx, doc_id in enumerate(q_list):
                rank = rank_0_idx + 1  # 1-based rank
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
        # Sort descending by fused score
        sorted_docs = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_docs
    ```
    """)

st.divider()

# Section 3: Five Practical Vector Query Patterns
st.header("3. Five Practical Vector Query Patterns", divider="rainbow")

st.markdown("""
Vector querying serves distinct architectural archetypes across enterprise generative AI and search applications:
""")

tab_pat1, tab_pat2, tab_pat3, tab_pat4, tab_pat5 = st.tabs([
    "1. Similar Documents",
    "2. Similar Questions (FAQ)",
    "3. Relevant Context (RAG)",
    "4. Related Products",
    "5. Similar Users / Personas"
])

with tab_pat1:
    st.markdown("""
    ### Pattern 1: Search for Similar Documents (Document-to-Document)
    * **Objective**: Given an existing document (e.g., research paper, legal contract, bug ticket), retrieve other documents covering related subject matter.
    * **Vector Strategy**: Pass the **document vector itself** (`query_embeddings=[doc_embedding]`) rather than a text query.
    * **Filtering Rule**: Always apply a metadata filter or post-query filter to **exclude the source document's own ID** (`where={"id": {"$ne": source_doc_id}}` or python list exclusion); otherwise distance 0.0 will return the document itself!
    * **Code Pattern**:
    ```python
    # Find legal contracts similar to contract_088
    source_emb = collection.get(ids=["contract_088"], include=["embeddings"])["embeddings"][0]
    
    similar_contracts = collection.query(
        query_embeddings=[source_emb],
        n_results=5,
        where={"jurisdiction": "California"}
    )
    # Filter out contract_088 if present
    ```
    """)

with tab_pat2:
    st.markdown("""
    ### Pattern 2: Search for Similar Questions (FAQ / Query-to-Query Matching)
    * **Objective**: Intercept customer support tickets before they reach a human agent or LLM by matching against verified FAQ answers.
    * **Vector Strategy**: Embed the **question field**, not the answer field. User queries are grammatically and structurally questions; matching `question -> question` yields substantially higher cosine similarity than matching `question -> 10-paragraph technical resolution`.
    * **Metadata Storage**: Store the verified solution text in the metadata payload (`metadata={"resolution": "Reboot router and clear cache"}`).
    * **Code Pattern**:
    ```python
    results = faq_collection.query(
        query_texts=["My screen flickers when waking from sleep mode"],
        n_results=1
    )
    best_dist = results["distances"][0][0]
    if best_dist < 0.15:  # High confidence match
        return results["metadatas"][0][0]["resolution"]
    else:
        # Fallback to general LLM RAG or human agent routing
        pass
    ```
    """)

with tab_pat3:
    st.markdown("""
    ### Pattern 3: Search for Relevant Context (RAG Context Injection)
    * **Objective**: Retrieve precise factual knowledge chunks to inject into an LLM prompt context window.
    * **Vector Strategy**: Asymmetric retrieval (`query text -> passage text`). Uses bi-encoder embedding models specifically trained on question-answer retrieval (e.g., `text-embedding-3-small`, `bge-base-en-v1.5`).
    * **Multi-Query Expansion**: Break down complex user inquiries into 3 focused context searches, retrieve top-4 chunks per query, fuse with RRF, and feed top-5 unique chunks to the prompt.
    * **Code Pattern**:
    ```python
    sub_queries = [
        "Kubernetes HorizontalPodAutoscaler CPU target formula",
        "HPA scaling algorithm tolerance threshold",
        "Kubernetes metrics server sync period default"
    ]
    raw_results = collection.query(query_texts=sub_queries, n_results=4)
    # Deduplicate and rank with RRF before building prompt
    ```
    """)

with tab_pat4:
    st.markdown("""
    ### Pattern 4: Search for Related Products (Vector Recommendation)
    * **Objective**: E-commerce visual or semantic item recommendation (e.g., "Customers who viewed this vintage leather jacket also browsed...").
    * **Vector Strategy**: Multimodal or joint text-attribute embeddings combining title, category, brand, and feature tags.
    * **Hard Metadata Guardrails**: Filter on availability, catalog active status, and price bands while ranking by vector affinity.
    * **Code Pattern**:
    ```python
    results = catalog_collection.query(
        query_embeddings=[active_product_embedding],
        n_results=8,
        where={
            "$and": [
                {"in_stock": True},
                {"price": {"$lte": 150.00}}
            ]
        }
    )
    ```
    """)

with tab_pat5:
    st.markdown("""
    ### Pattern 5: Search for Similar Users / Personas (Behavioral Embeddings)
    * **Objective**: User segmentation, collaborative filtering, or lookalike audience discovery in marketing and social networks.
    * **Vector Strategy**: Embed user interaction histories, purchased categories, or click sequences into fixed-dimensional user preference vectors.
    * **Clustering & K-NN**: Retrieve nearest neighbor user profiles to recommend content consumed by peer users sharing identical taste clusters.
    * **Code Pattern**:
    ```python
    user_cluster = user_collection.query(
        query_embeddings=[target_user_vector],
        n_results=20,
        where={"country": "DE"}
    )
    # Aggregate content IDs popular among these 20 peer users
    ```
    """)

st.divider()

# Section 4: Interactive Workbench: Multi-Query Simulator & RRF Ranker
st.header("4. Interactive Workbench: Multi-Query Simulator & RRF Ranker", divider="rainbow")

st.markdown("""
Experiment with multi-query retrieval, query expansion, and Reciprocal Rank Fusion on a curated mock corpus.
""")

# Mock Knowledge Base Documents
MOCK_CORPUS = [
    {
        "id": "DOC-HNSW-01",
        "title": "HNSW Graph Indexing Internals",
        "text": "Hierarchical Navigable Small World graphs provide O(log N) vector search via multi-layer skip lists. The M parameter governs connectivity.",
        "category": "Architecture",
        "relevance_vector": {"indexing": 0.95, "speed": 0.88, "ram": 0.40, "pricing": 0.05}
    },
    {
        "id": "DOC-IVF-02",
        "title": "Inverted File Index (IVF) Clustering",
        "text": "IVF partitions vector space into Voronoi cells using k-means centroids. At query time, only nprobe cells are traversed.",
        "category": "Architecture",
        "relevance_vector": {"indexing": 0.85, "speed": 0.70, "ram": 0.65, "pricing": 0.05}
    },
    {
        "id": "DOC-RAM-03",
        "title": "Vector Memory Optimization & Product Quantization",
        "text": "PQ compresses float32 vectors by 8x-16x using sub-vector codebooks, dramatically cutting server RAM requirements.",
        "category": "Optimization",
        "relevance_vector": {"indexing": 0.60, "speed": 0.50, "ram": 0.98, "pricing": 0.30}
    },
    {
        "id": "DOC-BENCH-04",
        "title": "Latency vs Recall Trade-offs in ChromaDB",
        "text": "Benchmarking efSearch shows that increasing search depth improves recall at the cost of queries per second (QPS).",
        "category": "Performance",
        "relevance_vector": {"indexing": 0.70, "speed": 0.95, "ram": 0.30, "pricing": 0.10}
    },
    {
        "id": "DOC-COST-05",
        "title": "Cloud Vector DB Pricing Models",
        "text": "Evaluating serverless vs dedicated pods across Pinecone, Qdrant, and ChromaDB based on active vector dimensions and QPS.",
        "category": "Pricing",
        "relevance_vector": {"indexing": 0.10, "speed": 0.30, "ram": 0.40, "pricing": 0.95}
    },
    {
        "id": "DOC-HYBRID-06",
        "title": "Hybrid Search: Combining BM25 with Dense Embeddings",
        "text": "Sparse lexical indices excel at exact keyword lookups while dense embeddings handle synonyms and semantic concepts.",
        "category": "Search",
        "relevance_vector": {"indexing": 0.65, "speed": 0.60, "ram": 0.50, "pricing": 0.15}
    }
]

col_wb_in, col_wb_cfg = st.columns([1.2, 0.8])

with col_wb_in:
    st.subheader("1. Enter Sub-Queries (Batch Input)")
    preset_choice = st.selectbox(
        "Select Domain Preset Scenario:",
        [
            "Custom Multi-Query",
            "Preset 1: Vector Search Performance & Indexing",
            "Preset 2: RAM Footprint & Compression",
            "Preset 3: Hybrid Lexical vs Dense Retrieval"
        ]
    )

    if preset_choice == "Preset 1: Vector Search Performance & Indexing":
        q1_default = "How to maximize vector search speed in HNSW?"
        q2_default = "Low latency approximate nearest neighbor graphs"
        q3_default = "Benchmark queries per second and recall"
    elif preset_choice == "Preset 2: RAM Footprint & Compression":
        q1_default = "Reduce vector database memory consumption"
        q2_default = "Product quantization compression ratio"
        q3_default = "RAM sizing calculation for 10 million vectors"
    elif preset_choice == "Preset 3: Hybrid Lexical vs Dense Retrieval":
        q1_default = "Combine sparse BM25 keyword matching with embeddings"
        q2_default = "Fix vector retrieval keyword blindness"
        q3_default = "Hybrid rank fusion in RAG architectures"
    else:
        q1_default = "What is HNSW vector indexing?"
        q2_default = "How to optimize memory in vector databases?"
        q3_default = "Fast vector search latency benchmarks"

    q1_text = st.text_input("Sub-Query 1:", value=q1_default)
    q2_text = st.text_input("Sub-Query 2:", value=q2_default)
    q3_text = st.text_input("Sub-Query 3 (Optional):", value=q3_default)

with col_wb_cfg:
    st.subheader("2. RRF Parameters")
    rrf_k = st.slider("RRF Constant (k):", min_value=1, max_value=120, value=60, step=1, 
                      help="Higher k smooths out rank disparities; lower k heavily prioritizes top-ranked items.")
    top_k_fetch = st.slider("Retrieval Depth per Query (n_results):", min_value=2, max_value=6, value=4, step=1)
    
    st.info("""
    **Formula in Effect**:  
    `RRF_Score = Sum[ 1 / (k + rank) ]`  
    Single query top-1 score with k={0}: `{1:.5f}`
    """.format(rrf_k, 1.0 / (rrf_k + 1)))

# Simulation scoring engine
def compute_mock_query_ranks(query_str: str, top_n: int) -> List[Dict[str, Any]]:
    q_lower = query_str.lower()
    scores = []
    for doc in MOCK_CORPUS:
        # Heuristic scoring based on semantic facets
        score = 0.1
        if any(w in q_lower for w in ["hnsw", "index", "indexing", "graph", "ivf"]):
            score += doc["relevance_vector"]["indexing"] * 0.9
        if any(w in q_lower for w in ["speed", "fast", "latency", "benchmark", "qps"]):
            score += doc["relevance_vector"]["speed"] * 0.85
        if any(w in q_lower for w in ["ram", "memory", "quantization", "compress", "size"]):
            score += doc["relevance_vector"]["ram"] * 0.95
        if any(w in q_lower for w in ["cost", "price", "pricing", "dollar"]):
            score += doc["relevance_vector"]["pricing"] * 0.9
        if any(w in q_lower for w in ["hybrid", "bm25", "sparse", "keyword"]):
            score += 0.8 if doc["id"] == "DOC-HYBRID-06" else 0.2
        
        # Add slight deterministic variance based on doc ID
        score += (hash(doc["id"] + query_str) % 100) / 1000.0
        # Convert to distance: distance = 1.0 - normalized similarity
        dist = max(0.02, 1.0 - min(0.98, score / 1.5))
        scores.append({"id": doc["id"], "title": doc["title"], "distance": round(dist, 4), "text": doc["text"]})
    
    scores.sort(key=lambda x: x["distance"])
    return scores[:top_n]

active_queries = [q for q in [q1_text, q2_text, q3_text] if q.strip()]

if active_queries:
    st.subheader("3. Individual Query Retrieval Results (Parallel Lists)")
    cols_q_out = st.columns(len(active_queries))
    
    all_query_rankings = []
    for idx, (col_q, q_str) in enumerate(zip(cols_q_out, active_queries)):
        with col_q:
            st.markdown(f"**Query {idx+1}**: *'{q_str[:35]}...'*")
            res_list = compute_mock_query_ranks(q_str, top_k_fetch)
            all_query_rankings.append([r["id"] for r in res_list])
            
            df_q = pd.DataFrame([
                {"Rank": r_idx + 1, "Doc ID": r["id"], "Distance": r["distance"]}
                for r_idx, r in enumerate(res_list)
            ])
            st.dataframe(df_q, use_container_width=True, hide_index=True)

    st.subheader("4. Reciprocal Rank Fusion (RRF) Unified Leaderboard")
    
    # Calculate RRF
    rrf_dict = defaultdict(lambda: {"score": 0.0, "ranks": {}})
    for q_idx, doc_ids in enumerate(all_query_rankings):
        for rank_0, doc_id in enumerate(doc_ids):
            rank = rank_0 + 1
            reciprocal = 1.0 / (rrf_k + rank)
            rrf_dict[doc_id]["score"] += reciprocal
            rrf_dict[doc_id]["ranks"][f"Q{q_idx+1}"] = rank
    
    fused_results = []
    for doc_id, data in rrf_dict.items():
        doc_meta = next(d for d in MOCK_CORPUS if d["id"] == doc_id)
        fused_results.append({
            "Doc ID": doc_id,
            "RRF Score": round(data["score"], 5),
            "Title": doc_meta["title"],
            "Category": doc_meta["category"],
            "Rank breakdown": ", ".join(f"{k}: #{v}" for k, v in data["ranks"].items()),
            "Appearances": len(data["ranks"])
        })
    
    fused_results.sort(key=lambda x: x["RRF Score"], reverse=True)
    df_fused = pd.DataFrame(fused_results)
    
    col_fuse_table, col_fuse_callout = st.columns([1.3, 0.7])
    
    with col_fuse_table:
        st.dataframe(
            df_fused.style.background_gradient(subset=["RRF Score"], cmap="Purples"),
            use_container_width=True,
            hide_index=True
        )
    
    with col_fuse_callout:
        if fused_results:
            winner = fused_results[0]
            st.success(f"""
            🏆 **Top Fused Candidate**: `{winner['Doc ID']}`  
            **RRF Score**: `{winner['RRF Score']}`  
            **Found in**: `{winner['Appearances']} of {len(active_queries)} queries`  
            **Breakdown**: {winner['Rank breakdown']}
            """)
            st.caption("Notice how consensus documents that rank well across multiple query variations rise to the top over single-query outliers.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the fundamental difference between Batch Querying and Multi-Query Retrieval with RRF?"):
    st.markdown("""
    **Answer**:
    * **Batch Querying** is an infrastructure capability in ChromaDB (`collection.query(query_texts=[...])`) that allows submitting multiple independent query vectors in one request to amortize network round-trips and maximize CPU/GPU matrix multiplication throughput. Each query returns its own independent result list.
    * **Multi-Query Retrieval with RRF** is an information retrieval pattern where multiple variations or facets of a single user prompt are queried simultaneously, and their independent result lists are mathematically merged and deduplicated using rank reciprocals into a single unified context list.
    """)

with st.expander("Q2: Why is Reciprocal Rank Fusion preferred over min-max distance score normalization?"):
    st.markdown("""
    **Answer**:
    Distance score normalization (e.g., `(dist - min) / (max - min)`) fails in multi-query and hybrid retrieval for three reasons:
    1. **Dynamic Range Distortion**: An outlier document with very high distance can compress all relevant documents into a narrow fraction of the scale.
    2. **Distribution Mismatch**: Different query variations map to regions of different local density in the vector space; a distance of 0.25 might be in the 99th percentile of tightness for Query A, but only the 50th percentile for Query B.
    3. **Metric Incompatibility**: In hybrid search combining BM25 (unbounded positive score) with cosine distance (`[0, 2]`), there is no mathematically sound linear transform to normalize them without extensive training data. RRF uses purely ordinal ranks (`1, 2, 3...`), bypassing score calibration entirely.
    """)

with st.expander("Q3: When searching for similar documents using an existing document's vector, what critical bug must be avoided?"):
    st.markdown("""
    **Answer**:
    If you query a vector database using an existing document's embedding (`query_embeddings=[doc.embedding]`), the nearest neighbor in the database will almost always be **the document itself at distance 0.0**. 
    
    To avoid recommending a document to itself:
    1. Filter out the source document's ID in the query filter (`where={"id": {"$ne": source_id}}` if supported), or
    2. Request `n_results = K + 1` and programmatically discard the first element if its ID matches the source document ID.
    """)

with st.expander("Q4: Why should FAQ similarity search embed questions rather than answers?"):
    st.markdown("""
    **Answer**:
    Embedding models capture grammatical structure, intent, and stylistic phrasing. User incoming queries are phrased as questions or trouble descriptions (*"Why is my laptop fan buzzing?"*). 
    
    * **Question-to-Question matching**: Computes similarity between two questions, which share linguistic intent, interrogative keywords, and length, resulting in cosine similarity often exceeding 0.85-0.95.
    * **Question-to-Answer matching**: The answer is typically an imperative or expository paragraph (*"Disassemble the chassis, clean the intake vent..."*), which occupies a different semantic subspace and produces lower similarity scores with higher false-negative rates.
    """)

with st.expander("Q5: How does Hypothetical Document Embeddings (HyDE) enhance multi-query retrieval?"):
    st.markdown("""
    **Answer**:
    In HyDE, an LLM generates a **hypothetical ideal answer** to the user's question before vector search. Even if the hypothetical answer contains hallucinations or factual errors, its vocabulary, tone, and technical terminology closely mirror real passage chunks in the knowledge base. 
    Embedding this hypothetical passage bridges the semantic gap between short user questions and long knowledge base documents. Multi-query retrieval often pairs the raw user question vector with the HyDE passage vector and fuses results with RRF.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: The Out-of-Bounds K Request in Multi-Query**  
    Requesting `n_results=100` across 5 sub-queries when the collection only contains 80 documents triggers an exception in ChromaDB.  
    **Fix**: Always dynamically cap `n_results = min(desired_k, collection.count())`.
    """)
    st.warning("""
    **Trap 2: Ignoring Deduplication in Multi-Query Aggregation**  
    Simply concatenating lists (`all_docs = res1 + res2`) without deduplication sends duplicate chunks into the LLM prompt, wasting context window tokens and biasing the LLM toward repeating identical sentences.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Very Small RRF k (e.g., k=1)**  
    Setting `k=1` makes rank 1 worth `0.50` and rank 2 worth `0.33`. A single anomalous #1 hit in an irrelevant sub-query will overpower a document ranked #2 across four other sub-queries. Maintain `k=60` as the robust baseline.
    """)
    st.warning("""
    **Trap 4: Sub-Query Drift in Autonomous LLM Generation**  
    Allowing an unconstrained LLM to generate 10 sub-queries often produces queries that drift off-topic, polluting the candidate pool with noisy chunks. Constrain the generator to 3-4 structured angles with few-shot examples.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the RRF formula?"):
        st.markdown("""
        **RRF Formula**:  
        `Score(d) = Sum[ 1 / (60 + rank(d)) ]` across all query result sets where document `d` appears.
        """)
    with st.expander("🗂️ Flashcard: What does `k=60` achieve in RRF?"):
        st.markdown("""
        **Smoothing**:  
        Prevents top-ranked hits in a single list from overwhelmingly crushing documents that consistently appear across multiple lists.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to do Document-to-Document search?"):
        st.markdown("""
        **Doc-to-Doc**:  
        Retrieve the source document's embedding with `.get(ids=[...], include=['embeddings'])` and pass it to `query_embeddings=[source_emb]`, filtering out the source ID.
        """)
    with st.expander("🗂️ Flashcard: What is HyDE?"):
        st.markdown("""
        **Hypothetical Document Embeddings**:  
        Generating a hypothetical answer via LLM and querying with its vector to match document-to-document style rather than question-to-document.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the difference between ChromaDB batch queries and Multi-Query RRF retrieval.", value=True)
chk2 = st.checkbox("I can implement Reciprocal Rank Fusion (RRF) in Python using standard rank reciprocals.", value=True)
chk3 = st.checkbox("I know why the smoothing constant k=60 prevents single-list dominance.", value=True)
chk4 = st.checkbox("I understand why Question-to-Question matching in FAQs outperforms Question-to-Answer matching.", value=True)
chk5 = st.checkbox("I know how to avoid the self-match trap when performing Document-to-Document similarity search.", value=True)
chk6 = st.checkbox("I can design a production multi-query pipeline that expands, queries, and fuses RAG context.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 8: Querying a Vector Database and are ready for Chapter 9: Metadata Filtering in Depth!")
