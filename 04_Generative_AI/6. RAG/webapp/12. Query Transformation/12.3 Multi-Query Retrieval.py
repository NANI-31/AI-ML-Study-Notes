import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class MultiQueryRetrievalEngine:
    """
    Simulates Multi-Query Retrieval (LangChain MultiQueryRetriever pattern):
    1. Generates multiple synthetic query perspectives using an LLM.
    2. Dispatches concurrent retrieval against the database for each perspective.
    3. Combines candidate pools via union and deduplication.
    4. Ranks final candidates via Reciprocal Rank Fusion (RRF).
    """
    CORPUS = {
        "doc_01": {
            "title": "HNSW Graph Indexing Internals",
            "text": "Hierarchical Navigable Small World (HNSW) constructs multi-layer proximity graphs with logarithmic search complexity.",
            "category": "Vector DB"
        },
        "doc_02": {
            "title": "Vector Memory Optimization & Quantization",
            "text": "Product Quantization (PQ) and Scalar Quantization (SQ8) compress Float32 embeddings into 8-bit integers, reducing RAM footprint by 75%.",
            "category": "Vector DB"
        },
        "doc_03": {
            "title": "Exact Flat Search & Brute Force ANN",
            "text": "IndexFlatIP computes exact Euclidean or inner product distances across all vectors, guaranteeing 100% recall at higher latency.",
            "category": "Vector DB"
        },
        "doc_04": {
            "title": "IVF Inverted File Index Clustering",
            "text": "Inverted File (IVF) partitions high-dimensional vector spaces into Voronoi cells via k-means clustering to bound search space.",
            "category": "Vector DB"
        },
        "doc_05": {
            "title": "Disk-Backed Vector Search Architecture",
            "text": "DiskANN and SPANN store graph structures on NVMe SSDs, caching compressed centroids in RAM for billion-scale retrieval.",
            "category": "Infrastructure"
        }
    }

    PRESET_SCENARIOS = {
        "Scenario 1: Vector Index Trade-offs": {
            "base_query": "How do vector databases achieve low latency with low memory?",
            "generated_queries": [
                "How do vector databases achieve low latency with low memory?",
                "HNSW graph memory overhead and quantization compression trade-offs",
                "Product quantization and IVF indexing techniques for scalable vector search"
            ],
            "mock_hits": {
                0: ["doc_01", "doc_02", "doc_04"],
                1: ["doc_02", "doc_01", "doc_05"],
                2: ["doc_02", "doc_04", "doc_03"]
            }
        },
        "Scenario 2: Hardware Resource Scaling": {
            "base_query": "Billion-scale embedding storage on disk vs RAM",
            "generated_queries": [
                "Billion-scale embedding storage on disk vs RAM",
                "DiskANN NVMe SSD vector search versus in-memory HNSW",
                "Memory consumption of uncompressed float32 vector indices at scale"
            ],
            "mock_hits": {
                0: ["doc_05", "doc_02", "doc_01"],
                1: ["doc_05", "doc_01", "doc_04"],
                2: ["doc_02", "doc_05", "doc_03"]
            }
        }
    }

    @classmethod
    def execute_multi_query(
        cls,
        scenario_key: str,
        k_rrf: int = 60,
        top_k: int = 4
    ) -> Dict[str, Any]:
        scenario = cls.PRESET_SCENARIOS[scenario_key]
        gen_queries = scenario["generated_queries"]
        mock_hits = scenario["mock_hits"]

        # Track appearances and calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        appearances: Dict[str, List[int]] = defaultdict(list)
        all_retrieved_ids = set()

        for q_idx, doc_ids in mock_hits.items():
            for rank, d_id in enumerate(doc_ids, start=1):
                all_retrieved_ids.add(d_id)
                appearances[d_id].append(rank)
                rrf_scores[d_id] += 1.0 / (k_rrf + rank)

        # Sort descending by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        results = []
        for d_id in sorted_ids[:top_k]:
            meta = cls.CORPUS[d_id]
            results.append({
                "id": d_id,
                "title": meta["title"],
                "category": meta["category"],
                "rrf_score": round(rrf_scores[d_id], 5),
                "frequency": len(appearances[d_id]),
                "ranks_per_query": appearances[d_id],
                "text": meta["text"]
            })

        return {
            "base_query": scenario["base_query"],
            "generated_queries": gen_queries,
            "total_candidates_fetched": sum(len(v) for v in mock_hits.values()),
            "unique_candidates_deduped": len(all_retrieved_ids),
            "ranked_results": results
        }

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="12.3 Multi-Query Retrieval - RAG Master Curriculum",
    page_icon="🔀",
    layout="wide"
)

st.title("12.3 Multi-Query Retrieval: Perspective Generation, Concurrent Fan-Out, & RRF Consensus")
st.caption("Chapter 12: Query Transformation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A single prompt from a user represents only one arbitrary phrasing of their underlying information need. 
Because dense embedding models project sentences into sensitive geometric coordinates, slightly different 
word choices can land in different geometric neighborhoods, missing crucial documents.

**Multi-Query Retrieval** overcomes this perspective bias. 
Using an instruction-tuned LLM, it generates multiple synthetic variations (perspectives, synonyms, sub-facets) 
of the user's question, executes concurrent retrieval for every variation, unions and deduplicates 
the candidate sets, and ranks the results using **Reciprocal Rank Fusion (RRF)**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Multi-Query Retrieval and Fusion Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    mq_bp_dot = """
    digraph MultiQueryBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_generation {
            label = "1. Perspective Generation Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_q [label="User Base Query\\ne.g. 'How do vector DBs save memory at low latency?'", fillcolor="#0F172A", color="#38BDF8"];
            prompt_gen [label="Perspective Generator Prompt\\nInstructs LLM to generate 3 alternative angles", fillcolor="#334155", color="#10B981"];
            fast_llm [label="Lightweight LLM (e.g. Gemini Flash)\\nOutputs JSON Array of 3 Queries", fillcolor="#0F172A", color="#8B5CF6"];

            user_q -> prompt_gen -> fast_llm;
        }

        subgraph cluster_dispatch {
            label = "2. Concurrent Multi-Query Retrieval Fan-Out Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_var1 [label="Query Variation 1\\n(Original Perspective)", fillcolor="#0F172A", color="#38BDF8"];
            q_var2 [label="Query Variation 2\\n(Technical Indexing Focus)", fillcolor="#0F172A", color="#38BDF8"];
            q_var3 [label="Query Variation 3\\n(Quantization Compression Focus)", fillcolor="#0F172A", color="#38BDF8"];

            vdb [label="Vector Store / Hybrid Index\\nAsync Concurrent Proximity Walk", fillcolor="#334155", color="#EC4899"];

            fast_llm -> q_var1;
            fast_llm -> q_var2;
            fast_llm -> q_var3;

            q_var1 -> vdb [label="async task 1"];
            q_var2 -> vdb [label="async task 2"];
            q_var3 -> vdb [label="async task 3"];
        }

        subgraph cluster_aggregation {
            label = "3. Aggregation, Deduplication & RRF Ranking Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            dedup_layer [label="Union & Deduplication Gate\\nUnique Doc ID Keying", fillcolor="#334155", color="#F59E0B"];
            rrf_fusion [label="Reciprocal Rank Fusion Layer\\nRRF(doc) = Sum(1 / (k + rank))", fillcolor="#0F172A", color="#10B981"];
            final_topk [label="Consensus Top-K Context Heap\\nPassed to Downstream Synthesis LLM", fillcolor="#334155", color="#38BDF8"];

            vdb -> dedup_layer [label="Combine Hit Pools"];
            dedup_layer -> rrf_fusion [label="Compute Consensuses"];
            rrf_fusion -> final_topk [label="Top Candidates"];
        }
    }
    """
    st.graphviz_chart(mq_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production frameworks (like **LangChain's `MultiQueryRetriever`**), 
the user's query is intercepted by a Runnable sequence:
`retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=chat_model)`.
The LLM generates 3–5 query perspectives in a single streaming generation pass. 
The retriever executes all queries asynchronously across the vector index using `asyncio.gather`, 
combines the returned candidate sets into a unique document set, and re-ranks candidates by frequency and position.
""")

st.divider()

# ==============================================================================
# SECTION 2: GENERATING MULTIPLE QUERIES
# ==============================================================================

st.header("2. Generating Multiple Queries: Perspective Diversification and Prompt Patterns", divider="rainbow")

st.markdown("""
How does generating multiple formulations overcome the fundamental limitations of single vector queries?
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Value of Perspective Diversification")
    st.markdown("""
    * **Overcoming Distance Sensitivity**:
      * In dense vector spaces, tiny grammatical shifts alter cosine angles. 
      * A user typing *"How do vector stores reduce memory?"* might land just outside the nearest neighbor cluster of a document titled *"Quantization and Memory Reduction in HNSW"*.
    * **Simulating Multiple Reader Personas**:
      * An effective multi-query prompt generates queries from three distinct angles:
        1. **Technical / Algorithmic Perspective**: *"HNSW vector memory compression trade-offs"*
        2. **Operational / Architectural Perspective**: *"Vector database RAM footprint optimization"*
        3. **High-Level Conceptual Perspective**: *"Reducing embedding storage costs"*
    * This triples the surface area of retrieval coverage across heterogeneous document collections.
    """)

with col2:
    st.subheader("Multi-Query Generation Prompt Template")
    st.code('''# Standard Production Multi-Query Prompt Template
MULTI_QUERY_PROMPT = """You are an AI language model assistant. 
Your task is to generate 3 different versions of the given user query 
to retrieve relevant documents from a vector database. 

By generating multiple perspectives on the user question, 
your goal is to help the user overcome some of the limitations 
of distance-based similarity search.

Provide these alternative questions separated by newlines.
Original question: {question}"""
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: RETRIEVING DOCUMENTS FOR EACH QUERY
# ==============================================================================

st.header("3. Retrieving Documents for Each Query: Asynchronous Fan-Out and Latency Budgeting", divider="rainbow")

st.markdown("""
Executing multiple queries must not scale wall-clock latency linearly. 
Understanding asynchronous fan-out patterns is essential for production SLAs.
""")

col_ret1, col_ret2 = st.columns([1, 1])

with col_ret1:
    st.subheader("Asynchronous Fan-Out Architecture")
    st.markdown("""
    * **Sequential vs. Concurrent Execution**:
      * *Sequential Execution (Anti-Pattern)*: 
        Running 3 queries sequentially: `Latency = T(Q1) + T(Q2) + T(Q3) = 25ms + 25ms + 25ms = 75ms`.
      * *Concurrent Execution (Production Standard)*: 
        Dispatching all 3 queries concurrently via `asyncio.gather`:
        `Latency = Max(T(Q1), T(Q2), T(Q3)) + Overhead = 28ms`.
    * **Connection Pool Sizing**:
      * Multi-query fan-out triples the instantaneous query volume against your vector database. 
      * Vector DB client connection pools (e.g., Qdrant or Pinecone gRPC client pools) must be sized 
        with at least 3x capacity to prevent connection queue stalling.
    """)

with col_ret2:
    st.subheader("Latency Budget Comparison")
    latency_comp_df = pd.DataFrame([
        {"Pipeline Stage", "Sequential Multi-Query", "Async Concurrent Multi-Query"},
        {"LLM Perspective Generation", "140 ms", "140 ms (single call for 3 queries)"},
        {"Vector DB Searches (3 queries)", "75 ms (3 x 25ms)", "28 ms (Max parallel execution)"},
        {"In-Memory Deduplication & RRF", "2 ms", "2 ms"},
        {"Total Retrieval Phase Latency", "217 ms", "170 ms (22% faster)"}
    ])
    st.dataframe(latency_comp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: COMBINING RETRIEVED DOCUMENTS
# ==============================================================================

st.header("4. Combining Retrieved Documents: Union vs. Intersection Strategies", divider="rainbow")

st.markdown("""
When multiple candidate pools are returned, how should the system combine them into a single candidate set?
""")

comb_cols = st.columns(2)

with comb_cols[0]:
    st.subheader("Union Strategy (Standard Practice)")
    st.markdown("""
    * **Definition**: Retains every document that appeared in **at least one** query candidate list:
      `Candidate_Pool = Pool_1 UNION Pool_2 UNION Pool_3`.
    * **Pros**: Maximizes recall. If a crucial document was only discovered by one specific technical rephrasing, 
      it is successfully preserved in the candidate pool.
    * **Cons**: Expands candidate pool size (up to 3x initial K), requiring downstream deduplication and ranking.
    * **Production Recommendation**: Standard for multi-query RAG where recall is the top priority.
    """)

with comb_cols[1]:
    st.subheader("Intersection Strategy (Conservative Filtering)")
    st.markdown("""
    * **Definition**: Retains only documents that appeared in **all three** candidate lists:
      `Candidate_Pool = Pool_1 INTERSECT Pool_2 INTERSECT Pool_3`.
    * **Pros**: Ultra-high precision. Guarantees that only documents relevant across all perspectives survive.
    * **Cons**: Extremely brittle. If any single query variation fails to retrieve a document due to distance thresholding, 
      the document is discarded entirely.
    * **Production Recommendation**: Rarely used in RAG; only applicable for strict compliance filtering.
    """)

st.divider()

# ==============================================================================
# SECTION 5: REMOVING DUPLICATES
# ==============================================================================

st.header("5. Removing Duplicates: Hash-Based and Content-Based Deduplication", divider="rainbow")

st.markdown("""
Because the generated query variations target the same core information need, there is heavy document overlap 
between candidate lists. In-memory deduplication is mandatory.
""")

col_dedup1, col_dedup2 = st.columns([1, 1])

with col_dedup1:
    st.subheader("Two-Tier Deduplication Mechanics")
    st.markdown("""
    1. **Primary Key / Document ID Deduplication**:
       * The fastest approach (`O(1)` hash lookup). Every indexed chunk has a deterministic unique ID (`doc_id` or UUID).
       * A Python dictionary tracks seen IDs and aggregates rank occurrences:
         ```python
         unique_docs = {}
         for item in candidate_stream:
             if item.id not in unique_docs:
                 unique_docs[item.id] = item
         ```
    2. **Content Hash Deduplication**:
       * Catches identical text chunks that were indexed under different doc IDs across different collections.
       * Uses SHA-256 or MD5 hashing on normalized text strings.
    """)

with col_dedup2:
    st.subheader("Tracking Cross-Query Consensus")
    st.markdown("""
    Deduplication must not simply discard duplicate occurrences! 
    The fact that a document was retrieved by **3 out of 3** queries is a powerful positive relevance signal.
    
    * In production, the deduplicator maintains an occurrence list:
      `doc_appearances["doc_02"] = [Rank 1, Rank 1, Rank 2]`
    * This occurrence list serves as the input to the final Reciprocal Rank Fusion stage.
    """)

st.divider()

# ==============================================================================
# SECTION 6: RANKING RESULTS
# ==============================================================================

st.header("6. Ranking Results: Reciprocal Rank Fusion across Multi-Query Streams", divider="rainbow")

st.markdown("""
Once duplicate documents are unified, the final candidate list must be re-ranked. 
**Reciprocal Rank Fusion (RRF)** is the gold standard for multi-query consensus ranking.
""")

col_rank1, col_rank2 = st.columns([1, 1])

with col_rank1:
    st.subheader("The Multi-Query RRF Formula")
    st.markdown("""
    For each unique document, its consensus score across all N generated queries is:
    
    `RRF_Score(doc) = Sum over queries of (1.0 / (k_rrf + rank(doc, query)))`
    
    * **k_rrf**: Standard empirical constant (**60**).
    * **rank(doc, query)**: The 1-based rank position of the document in that query's result list.
    * **The Consensus Advantage**:
      * A document retrieved as **#1** by Query 1, **#1** by Query 2, and **#2** by Query 3 earns:
        `1/61 + 1/61 + 1/62 = 0.01639 + 0.01639 + 0.01613 = 0.04891`.
      * A document retrieved only once as #1 earns: `1/61 = 0.01639`.
      * Multi-query consensus dramatically outranks single-query noise!
    """)

with col_rank2:
    st.subheader("Multi-Query Ranking Outcome Comparison")
    rank_comp_df = pd.DataFrame([
        {
            "Document Title": "Doc 2: Vector Memory & Quantization",
            "Appearances": "3 of 3 queries",
            "Ranks": "Q1: #2 | Q2: #1 | Q3: #1",
            "Fused RRF Score": "0.04865 (Rank 1 - Champion)",
            "Verdict": "Universal multi-query consensus"
        },
        {
            "Document Title": "Doc 1: HNSW Graph Internals",
            "Appearances": "2 of 3 queries",
            "Ranks": "Q1: #1 | Q2: #2 | Q3: Missed",
            "Fused RRF Score": "0.03252 (Rank 2)",
            "Verdict": "Strong secondary support"
        },
        {
            "Document Title": "Doc 3: Exact Flat Search",
            "Appearances": "1 of 3 queries",
            "Ranks": "Q1: Missed | Q2: Missed | Q3: #3",
            "Fused RRF Score": "0.01587 (Rank 5)",
            "Verdict": "Isolated single-query hit"
        }
    ])
    st.dataframe(rank_comp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Multi-Query Retrieval Engine", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module demonstrating asynchronous multi-query retrieval, 
deduplication, and Reciprocal Rank Fusion scoring.
""")

st.code('''
from typing import List, Dict, Any, Optional
from collections import defaultdict

class EnterpriseMultiQueryRetriever:
    """
    Production-grade multi-query retrieval engine with deduplication
    and Reciprocal Rank Fusion (RRF) scoring.
    """
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    def fuse_query_results(
        self,
        query_results: List[List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate lists from multiple query runs, deduplicates by ID,
        and scores via Reciprocal Rank Fusion.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict[str, Any]] = {}
        appearances: Dict[str, List[int]] = defaultdict(list)

        for query_idx, candidates in enumerate(query_results):
            for rank, doc in enumerate(candidates, start=1):
                doc_id = doc["id"]
                doc_store[doc_id] = doc
                appearances[doc_id].append(rank)
                # Apply RRF formulation
                rrf_scores[doc_id] += 1.0 / (self.k_rrf + rank)

        # Sort descending by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        final_candidates = []
        for d_id in sorted_ids[:top_k]:
            item = dict(doc_store[d_id])
            item["rrf_score"] = float(round(rrf_scores[d_id], 5))
            item["query_appearances"] = len(appearances[d_id])
            item["individual_ranks"] = appearances[d_id]
            final_candidates.append(item)

        return final_candidates
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Multi-Query Generator & Aggregation Simulator", divider="rainbow")

st.markdown("""
Select a scenario below to observe how **Multi-Query Retrieval** generates distinct perspectives, 
executes concurrent candidate lookups, and applies **Reciprocal Rank Fusion** to surface consensus documents.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    scenario_choice = st.selectbox(
        "Select Test Scenario:",
        list(MultiQueryRetrievalEngine.PRESET_SCENARIOS.keys())
    )
    k_param = st.slider("RRF Smoothing Parameter (k_rrf):", min_value=1, max_value=100, value=60)
    top_k_slider = st.slider("Top-K Candidates to Output:", min_value=1, max_value=5, value=3)

    mq_data = MultiQueryRetrievalEngine.execute_multi_query(
        scenario_key=scenario_choice,
        k_rrf=k_param,
        top_k=top_k_slider
    )

    st.markdown("#### 📝 Generated Query Perspectives")
    for idx, q_text in enumerate(mq_data["generated_queries"], start=1):
        st.markdown(f"**Query {idx}**: *\"{q_text}\"*")

with wb_col2:
    st.markdown("### 📊 Multi-Query Execution Metrics")
    m_cols = st.columns(3)
    m_cols[0].metric("Generated Queries", len(mq_data["generated_queries"]))
    m_cols[1].metric("Raw Candidates", mq_data["total_candidates_fetched"])
    m_cols[2].metric("Unique Deduplicated", mq_data["unique_candidates_deduped"])

    st.markdown("#### 🏆 Consensus Fused Results")
    res_df = pd.DataFrame(mq_data["ranked_results"])
    st.dataframe(res_df[["id", "title", "category", "rrf_score", "frequency", "ranks_per_query", "text"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does Multi-Query Retrieval resolve distance sensitivity in vector embeddings?"):
    st.markdown("""
    **Answer**: In high-dimensional vector spaces, sentences that express the exact same question in different words 
    often yield divergent embedding coordinates. If an indexing pipeline relied on specific terminology (e.g., *"quantization"*), 
    a user asking about *"memory optimization"* might miss the nearest neighbor cluster. 
    Multi-Query Retrieval overcomes this by generating 3–5 distinct prompt perspectives, casting a wider geometric net 
    across the vector space and retrieving documents that match any of the plausible phrasing styles.
    """)

with st.expander("Q2: Why is Reciprocal Rank Fusion (RRF) particularly effective for combining multi-query results?"):
    st.markdown("""
    **Answer**: RRF combines candidate lists based strictly on ordinal rank positions rather than uncalibrated raw distance floats. 
    In multi-query search, different query variations can yield different raw cosine distributions (e.g., Query 1 has top score 0.82; 
    Query 2 has top score 0.91). Attempting to average raw scores introduces scale distortion. 
    RRF eliminates scale dependencies entirely and inherently rewards documents that appear across multiple query candidate lists.
    """)

with st.expander("Q3: What are the database resource trade-offs of deploying Multi-Query Retrieval in production?"):
    st.markdown("""
    **Answer**: Generating 3 queries multiplies the database read load by a factor of 3x. 
    Every user query now triggers 3 concurrent HNSW graph traversals or inverted index searches. 
    If a system processes 1,000 queries per second, the vector database must sustain 3,000 QPS. 
    Mitigations include:
    1. Caching multi-query expansions for frequent questions.
    2. Only triggering multi-query retrieval for complex analytical queries (as classified by Query Understanding).
    3. Tuning `top_k` per query downwards to reduce overall candidate payload sizes.
    """)

with st.expander("Q4: How should a production system handle cases where one generated query is completely off-topic?"):
    st.markdown("""
    **Answer**: LLM generation can occasionally produce an irrelevant or drifting query variation (e.g., interpreting "Apple" as fruit 
    rather than company). RRF naturally protects against this failure mode: 
    documents retrieved solely by the hallucinated query will appear in only one candidate list with low individual rank, 
    receiving an RRF score of only `1 / (60 + rank)`. Documents retrieved by the other valid queries will earn consensus scores 
    (`1/61 + 1/62 = 0.032`), easily outranking the single noisy query's results.
    """)

with st.expander("Q5: What is the difference between Multi-Query Retrieval and Query Decomposition?"):
    st.markdown("""
    **Answer**:
    * **Multi-Query Retrieval** generates multiple alternative perspectives for a **single atomic question** to improve recall 
      (e.g., *"How do I reset my password?"* -> *"Account password recovery instructions"*, *"Credential reset steps"*).
    * **Query Decomposition** divides a **complex multi-part question** into distinct, orthogonal sub-questions that each address 
      different aspects of the problem (e.g., *"Compare Q1 revenue with Q2 expenses"* -> Sub-query 1: *"Q1 revenue"*, Sub-query 2: *"Q2 expenses"*).
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Serial Execution of Generated Query Searches**
    * **Symptom**: Query latency triples from 30ms to 90ms upon deploying multi-query retrieval.
    * **Root Cause**: Executing each generated query sequentially in a `for` loop rather than concurrently.
    * **Remedy**: Always dispatch all query variations concurrently using `asyncio.gather` so total latency equals the maximum of the parallel queries.
    """)

    st.error("""
    **Trap 2: Generating Excessive Query Variations (N > 5)**
    * **Symptom**: High database CPU utilization and diminishing recall improvements.
    * **Root Cause**: Generating 7 to 10 query variations per turn.
    * **Remedy**: Keep the number of generated queries between 3 and 5. Research demonstrates that 3 variations capture 90% of the possible recall gain.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Averaging Raw Cosine Scores Across Queries Instead of Using RRF**
    * **Symptom**: Queries with high cosine baselines dominate the final ranking, overriding valid hits from other queries.
    * **Root Cause**: Score distributions vary significantly across phrasing styles; adding raw floats distorts rankings.
    * **Remedy**: Always merge candidate lists using Reciprocal Rank Fusion (RRF) rather than raw score addition or averaging.
    """)

    st.warning("""
    **Trap 4: Discarding Duplicate Document Appearance Counts**
    * **Symptom**: Documents that match 3 out of 3 queries are ranked equally with documents that matched only 1 query.
    * **Root Cause**: Deduplicating into a naive Python `set` without recording frequency counts.
    * **Remedy**: Store occurrence lists per unique document ID and feed rank positions into RRF.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Multi-Query Retrieval?")
        st.markdown("**Back**: An information retrieval technique that generates multiple perspectives of a user query using an LLM, executes parallel searches for each, and fuses candidates to eliminate distance sensitivity.")

    with st.container(border=True):
        st.markdown("**Front**: Why is RRF preferred over score averaging in Multi-Query Retrieval?")
        st.markdown("**Back**: Because different query formulations produce disparate raw similarity score distributions. RRF relies strictly on ordinal ranks, naturally rewarding documents supported by multi-query consensus.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: How does asynchronous fan-out preserve search latency in Multi-Query Retrieval?")
        st.markdown("**Back**: By dispatching all generated queries concurrently via `asyncio.gather`, ensuring total database retrieval latency equals the duration of the single slowest query rather than their sum.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What is the consensus effect in Multi-Query RRF?")
            st.markdown("**Back**: A document retrieved across multiple query variations earns multiple reciprocal terms (`1/(60+r1) + 1/(60+r2)`), easily outranking documents retrieved by only one query variation.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Multi-Query Retrieval, Perspective Generation, and Consensus Ranking:
""")

chk1 = st.checkbox("I understand why single vector queries suffer from distance sensitivity and phrasing bias.", value=True)
chk2 = st.checkbox("I can articulate how perspective diversification expands retrieval coverage across heterogeneous corpora.", value=True)
chk3 = st.checkbox("I know how to execute multi-query retrieval concurrently using asynchronous fan-out patterns.", value=True)
chk4 = st.checkbox("I understand the distinction between Union (high recall) and Intersection (conservative filtering) strategies.", value=True)
chk5 = st.checkbox("I know why Reciprocal Rank Fusion is the gold standard for scoring multi-query candidate pools.", value=True)
chk6 = st.checkbox("I recognize why 3 to 5 query variations represent the optimal balance between recall gain and database load.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Magnificent! You have mastered Multi-Query Retrieval and Consensus Ranking architectures.")
