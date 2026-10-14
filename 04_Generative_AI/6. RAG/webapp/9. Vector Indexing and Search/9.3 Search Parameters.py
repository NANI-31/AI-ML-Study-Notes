import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class ParameterizedSearchController:
    """
    Simulates parameterized vector search with dynamic Top-K, similarity thresholds,
    search depth exploration, metadata filtering, and hybrid dense+sparse fusion (RRF).
    """
    CORPUS = [
        {"id": "doc_1", "text": "HNSW graphs provide sub-2ms approximate nearest neighbor search via skip-layers.", "vector": [0.92, 0.25, 0.10, 0.05], "category": "Algorithms", "tokens": ["hnsw", "graph", "ann", "skip", "search"]},
        {"id": "doc_2", "text": "PostgreSQL pgvector extension enables relational SQL joins with vector distance queries.", "vector": [0.45, 0.85, 0.15, 0.10], "category": "Databases", "tokens": ["postgresql", "pgvector", "sql", "relational", "vector"]},
        {"id": "doc_3", "text": "Product Quantization compresses dense 1536-dimensional embeddings into compact byte codes.", "vector": [0.75, 0.15, 0.60, 0.15], "category": "Algorithms", "tokens": ["product", "quantization", "compression", "embeddings", "byte"]},
        {"id": "doc_4", "text": "Hybrid search combines BM25 lexical keyword matching with dense vector semantics using RRF.", "vector": [0.88, 0.30, 0.20, 0.35], "category": "Retrieval", "tokens": ["hybrid", "search", "bm25", "lexical", "dense", "rrf"]},
        {"id": "doc_5", "text": "Single-stage filtered search evaluates metadata during graph traversal to preserve recall.", "vector": [0.80, 0.35, 0.15, 0.40], "category": "Retrieval", "tokens": ["filtered", "search", "metadata", "graph", "recall"]},
        {"id": "doc_6", "text": "Inverted File Indexes (IVF) partition vector space into Voronoi cells via k-means clustering.", "vector": [0.70, 0.20, 0.45, 0.10], "category": "Algorithms", "tokens": ["ivf", "voronoi", "clustering", "kmeans", "index"]}
    ]

    @classmethod
    def execute_search(
        cls,
        query_vector: List[float],
        query_terms: List[str],
        top_k: int = 3,
        threshold: float = 0.5,
        search_depth_ef: int = 16,
        category_filter: Optional[str] = None,
        enable_hybrid: bool = False,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qv)
        qv = qv / norm if norm > 0 else qv

        # 1. Dense vector scoring
        dense_results = []
        for doc in cls.CORPUS:
            if category_filter and doc["category"] != category_filter:
                continue
            dv = np.array(doc["vector"], dtype=np.float32)
            d_norm = np.linalg.norm(dv)
            sim = float(np.dot(qv, dv / d_norm if d_norm > 0 else dv))
            
            # Simulate search depth impact: lower efSearch introduces minor noise
            if search_depth_ef < 8:
                sim *= 0.95
                
            if sim >= threshold:
                dense_results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "dense_score": round(sim, 4)
                })

        dense_results.sort(key=lambda x: x["dense_score"], reverse=True)

        if not enable_hybrid:
            for rank, item in enumerate(dense_results):
                item["final_score"] = item["dense_score"]
                item["method"] = "Dense Vector Only"
            return dense_results[:top_k]

        # 2. Sparse lexical scoring (simulated BM25 term overlap)
        sparse_results = []
        q_tokens = set([t.lower() for t in query_terms])
        for doc in cls.CORPUS:
            if category_filter and doc["category"] != category_filter:
                continue
            overlap = len(q_tokens.intersection(set(doc["tokens"])))
            if overlap > 0:
                sparse_results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "sparse_score": overlap
                })
        sparse_results.sort(key=lambda x: x["sparse_score"], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map = {d["id"]: d for d in cls.CORPUS}

        for rank, d in enumerate(dense_results):
            rrf_scores[d["id"]] = rrf_scores.get(d["id"], 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, d in enumerate(sparse_results):
            rrf_scores[d["id"]] = rrf_scores.get(d["id"], 0.0) + (1.0 / (rrf_k + rank + 1))

        fused = []
        for doc_id, score in rrf_scores.items():
            fused.append({
                "id": doc_id,
                "text": doc_map[doc_id]["text"],
                "category": doc_map[doc_id]["category"],
                "final_score": round(score, 5),
                "method": "Hybrid Dense + BM25 Fusion (RRF)"
            })
        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return fused[:top_k]


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="9.3 Search Parameters - RAG Master Curriculum",
    page_icon="🎛️",
    layout="wide"
)

st.title("9.3 Search Parameters: Top-K, Similarity Thresholds, Search Depth, Metadata Filters, and Hybrid RRF")
st.caption("Chapter 9: Vector Indexing and Search | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Once vectors are indexed into HNSW graphs or IVF partitions, query execution is governed by a set 
of runtime parameters. How these parameters are configured dictates whether your RAG pipeline returns 
crisp, authoritative context or floods downstream LLMs with hallucination-inducing noise.

Mastering query parameters requires configuring **Top-K candidate limits**, **similarity thresholds**, 
**ANN search depth (`efSearch` / `nprobe`)**, **metadata filter predicates**, and **hybrid lexical+dense fusion**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Query-Time Search Parameter Pipeline & Routing", divider="rainbow")
st.markdown("""
The architecture below illustrates how query parameters govern candidate pruning, graph depth traversal, 
metadata filter application, and hybrid rank fusion.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    param_bp_dot = """
    digraph SearchParametersPipeline {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_input {
            label = "1. Query & Hyperparameter Input";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_vec [label="Query Vector\\nDense Float32 Embedding", fillcolor="#0F172A", color="#38BDF8"];
            q_txt [label="Query Text Tokens\\nBM25 Sparse Lexical Terms", fillcolor="#0F172A", color="#38BDF8"];
            params_node [label="Runtime Parameters\\nTop-K | Threshold | efSearch | Filter AST", fillcolor="#0F172A", color="#F59E0B"];
        }

        subgraph cluster_traversal {
            label = "2. Parameterized Traversal & Pruning";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            depth_eval [label="Search Depth Controller\\n(efSearch / nprobe Queue Size)", fillcolor="#334155", color="#8B5CF6"];
            filter_eval [label="Metadata Filter Evaluator\\nSingle-Stage ACORN / Inverted Index", fillcolor="#334155", color="#10B981"];
            thresh_gate [label="Similarity Threshold Cutoff\\nDiscard if Score < min_threshold", fillcolor="#334155", color="#EF4444"];

            params_node -> depth_eval;
            params_node -> filter_eval;
            params_node -> thresh_gate;
        }

        subgraph cluster_fusion {
            label = "3. Multi-Channel Ranking & Fusion";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            dense_rank [label="Dense Candidates List\\nRanked by Cosine Proximity", fillcolor="#0F172A", color="#38BDF8"];
            sparse_rank [label="Sparse BM25 Candidates\\nRanked by Lexical Token Overlap", fillcolor="#0F172A", color="#38BDF8"];
            rrf_node [label="Reciprocal Rank Fusion (RRF)\\nScore = 1/(k + Rank_dense) + 1/(k + Rank_sparse)", fillcolor="#0F172A", color="#EC4899"];
            topk_cutoff [label="Final Top-K Context Window\\nPruned for LLM Prompt", fillcolor="#0F172A", color="#10B981"];

            q_vec -> depth_eval -> dense_rank;
            filter_eval -> dense_rank;
            filter_eval -> sparse_rank;
            q_txt -> sparse_rank;
            dense_rank -> rrf_node;
            sparse_rank -> rrf_node;
            rrf_node -> thresh_gate -> topk_cutoff;
        }
    }
    """
    st.graphviz_chart(param_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern enterprise RAG pipelines, production queries almost never rely on bare `top_k=5` vector lookup. 
They operate through multi-stage parameterized controllers:
1. **Dynamic Beam Exploration**: Search depth (`efSearch`) is tuned dynamically—higher for critical analytical queries and lower for high-throughput autocomplete.
2. **Hard Distance Thresholds**: Low-similarity chunks are dropped before reaching LLM prompts to prevent hallucinations.
3. **Hybrid Search with RRF**: Systems like **Elasticsearch**, **Qdrant**, and **Pinecone** combine dense vector scores with BM25 keyword matching via Reciprocal Rank Fusion, ensuring exact keyword matches (e.g., error codes or product IDs) are never missed.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOP-K RETRIEVAL
# ==============================================================================

st.header("2. Top-K Retrieval", divider="rainbow")
st.markdown("""
**Top-K** specifies the maximum number of nearest-neighbor document chunks returned to the caller.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Configuring the Top-K Tradeoff")
    st.markdown("""
    * **Setting K Too Low (K = 1 to 2)**:
      * Risk of missing vital complementary context spread across multiple document sections.
      * Brittle answers if the highest-ranking chunk lacks specific answers.
    * **Setting K Too High (K = 20 to 50)**:
      * Injects irrelevant distractor chunks into LLM prompts.
      * Increases inference latency and token costs.
      * Triggers the **"Lost in the Middle"** phenomenon, where LLMs overlook information placed in the center of lengthy prompts.
    * **Production Standard**:
      Most RAG pipelines retrieve K = 10 to 25 candidates from the vector store, 
      then apply a Cross-Encoder Reranker to select the top 3 to 5 most relevant chunks for the LLM context.
    """)

with col2:
    st.subheader("Top-K vs. LLM Context Window Impact")
    k_df = pd.DataFrame([
        {"Top-K Value", "Context Size (~500 tokens/chunk)", "Prompt Latency Impact", "Hallucination Risk"},
        {"Top-3", "~1,500 tokens", "Minimal (< 500 ms)", "Low (Only top context)"},
        {"Top-10", "~5,000 tokens", "Moderate (1 - 2 s)", "Moderate (Some distractor noise)"},
        {"Top-30", "~15,000 tokens", "High (3 - 5 s)", "High ('Lost in the middle' risk)"}
    ])
    st.dataframe(k_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: SIMILARITY THRESHOLDS
# ==============================================================================

st.header("3. Similarity Thresholds & Cutoffs", divider="rainbow")
st.markdown("""
A **Similarity Threshold** establishes a minimum proximity score required for a retrieved candidate 
to be included in the result set.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why Thresholds are Essential")
    st.markdown("""
    * **Preventing Irrelevant Retrieval on Out-of-Domain Queries**:
      Without a threshold, querying `"How do I configure Kubernetes on Mars?"` against a corporate HR database 
      will still return K documents, even if their similarity score is only 0.21!
    * **Enabling Confident Fallback Behaviors**:
      If zero documents meet the threshold (e.g., `similarity < 0.70`), the system can gracefully reply:
      *"I do not have sufficient information in the knowledge base to answer this question,"* 
      completely eliminating hallucinations.
    """)

with col2:
    st.subheader("Calibrating Metric Thresholds")
    st.markdown("""
    Threshold values depend heavily on the underlying embedding model:
    * **OpenAI `text-embedding-3-small` (Cosine)**:
      * `> 0.82`: High confidence match.
      * `0.70 - 0.82`: Moderate topical relevance.
      * `< 0.65`: Low relevance / unrelated noise.
    * **Cohere Embed v3 (Cosine)**:
      * Typically exhibits lower baseline score ranges (`0.50 - 0.75` for high relevance).
    * **Rule of Thumb**: Always calibrate similarity thresholds empirically across a domain validation benchmark before enforcing them in production.
    """)

st.divider()

# ==============================================================================
# SECTION 4: SEARCH DEPTH
# ==============================================================================

st.header("4. Search Depth: efSearch, nprobe, & Dynamic Beam Widths", divider="rainbow")
st.markdown("""
**Search Depth** dictates how deeply the ANN algorithm explores candidate paths within its index structure.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Parameters Controlling Search Depth")
    st.markdown("""
    * **`efSearch` (HNSW)**:
      * The size of the dynamic priority queue maintained during graph traversal.
      * Must satisfy `efSearch >= top_k`.
      * Higher `efSearch` explores more branches, recovering candidates that would otherwise be missed, but increases search latency.
    * **`nprobe` (IVF)**:
      * The number of Voronoi cluster centroids to inspect.
      * Higher `nprobe` evaluates more inverted buckets, reducing boundary clipping errors.
    """)

with col2:
    st.subheader("Dynamic Search Depth Allocation")
    st.markdown("""
    Enterprise search routers dynamically adjust search depth based on query intent:
    * **Quick Autocomplete / Typeahead**: `efSearch = 8` (ultra-low latency, acceptable ~90% recall).
    * **Standard Conversational Chatbot**: `efSearch = 32` (balanced 2-3ms latency, ~97% recall).
    * **Analytical / Legal Fact Discovery**: `efSearch = 128` (deep graph walk, ~99.5% recall).
    """)

st.divider()

# ==============================================================================
# SECTION 5: FILTERING EXECUTION
# ==============================================================================

st.header("5. Filtering Execution: Pre-filtering, Post-filtering, and In-flight Pruning", divider="rainbow")
st.markdown("""
Applying scalar filters during vector retrieval requires choosing the correct execution strategy.
""")

filter_comp_df = pd.DataFrame([
    {"Strategy", "Execution Sequence", "Recall Behavior", "Latency Behavior", "Best For"},
    {"Pre-Filtering", "Filter IDs via scalar index -> Scan vector subset", "100% metadata compliance; graph can disconnect", "Slow if subset is large; degrades to brute force", "Broad filters matching >20% of documents"},
    {"Post-Filtering", "Top-K vector search -> Discard non-matching payloads", "Catastrophic recall drop if filter is selective", "Fastest vector traversal", "Non-critical tags or very loose filters"},
    {"In-Flight (ACORN / Filtered HNSW)", "Inspect metadata bitsets during every graph hop", "High recall + high metadata compliance", "Sub-5ms deterministic latency", "Enterprise multi-tenant RAG (Production standard)"}
])

st.dataframe(filter_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: METADATA-BASED SEARCH
# ==============================================================================

st.header("6. Metadata-Based Search & Boolean Predicate Evaluation", divider="rainbow")
st.markdown("""
Enterprise search queries combine geometric vector coordinates with structured SQL/JSON boolean predicates.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Complex Predicate Construction")
    st.markdown("""
    Modern vector databases parse structured query abstract syntax trees (ASTs):
    * **Conjunctions (`AND`)**: All conditions must be satisfied.
    * **Disjunctions (`OR`)**: Any condition may be satisfied.
    * **Negations (`NOT`)**: Exclude specific document tags.
    * **Range Filters**: Numerical or timestamp intervals (`date >= 2024-01-01`).
    * **Array Intersections**: Tag containment (`tags CONTAINS 'enterprise'`).
    """)

with col2:
    st.subheader("Code Example: Complex Filter Construction")
    st.code('''# Complex Filter Predicate in Qdrant
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

query_filter = Filter(
    must=[
        FieldCondition(key="tenant_id", match=MatchValue(value="corp_42")),
        FieldCondition(key="year", range=Range(gte=2024))
    ],
    must_not=[
        FieldCondition(key="archived", match=MatchValue(value=True))
    ]
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 7: HYBRID SEARCH
# ==============================================================================

st.header("7. Hybrid Search: Combining Sparse BM25 and Dense Vectors via RRF", divider="rainbow")
st.markdown("""
Neither dense semantic search nor sparse keyword search is sufficient on its own. 
**Hybrid Search** combines the strengths of both modalities using **Reciprocal Rank Fusion (RRF)**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why Combine Dense and Sparse?")
    st.markdown("""
    * **Where Dense Vectors Struggle**:
      Dense embeddings represent conceptual semantics. They struggle with exact keyword matches, such as:
      * Error codes (`ERR_404_NULL_POINTER`)
      * Product SKUs (`SKU-9921-X`)
      * Rare legal names, acronyms, and chemical formulas.
    * **Where Sparse BM25 Struggles**:
      BM25 relies on exact term overlap. It fails when queries use synonyms or cross-lingual conceptual descriptions without shared words.
    * **The Hybrid Solution**:
      Dense search captures semantic concepts, while BM25 captures exact token matches.
    """)

with col2:
    st.subheader("Reciprocal Rank Fusion (RRF) Formula")
    st.markdown("""
    RRF combines rankings without requiring score normalization:
    
    `RRF_Score(d) = (1 / (k + Rank_dense(d))) + (1 / (k + Rank_sparse(d)))`
    
    * `Rank_dense(d)` is the document's 1-indexed position in the dense vector result list.
    * `Rank_sparse(d)` is its position in the BM25 keyword result list.
    * k is a smoothing constant (typically k = 60).
    * Documents appearing near the top of **both** lists receive the highest fused priority.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("8. Production Implementation: Enterprise Parameterized Query Controller with RRF Fusion", divider="rainbow")
st.markdown("""
Below is an enterprise query controller demonstrating how Top-K limits, similarity thresholds, 
search depth, and Reciprocal Rank Fusion operate in production Python.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Optional

class HybridSearchController:
    """
    Coordinates parameterized vector retrieval and BM25 sparse keyword matching
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse_rankings(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Combines dense and sparse ranked lists using Reciprocal Rank Fusion."""
        fused_scores: Dict[str, float] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense ranked list
        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            metadata_map[doc_id] = item
            score = 1.0 / (self.rrf_k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score

        # 2. Process sparse ranked list
        for rank, item in enumerate(sparse_results):
            doc_id = item["id"]
            if doc_id not in metadata_map:
                metadata_map[doc_id] = item
            score = 1.0 / (self.rrf_k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score

        # 3. Sort by fused RRF score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        final_candidates = []
        for doc_id, rrf_score in sorted_docs:
            doc_data = metadata_map[doc_id]
            # Enforce similarity threshold if dense score available
            if "dense_score" in doc_data and doc_data["dense_score"] < threshold:
                continue
            final_candidates.append({
                "id": doc_id,
                "rrf_score": round(rrf_score, 5),
                "dense_score": doc_data.get("dense_score"),
                "text": doc_data.get("text", "")
            })
            if len(final_candidates) >= top_k:
                break

        return final_candidates
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("9. Interactive Workbench: Search Tuning & Dynamic Hyperparameter Explorer", divider="rainbow")
st.markdown("""
Experiment with query parameters in real time. Adjust Top-K, similarity thresholds, 
search depth (`efSearch`), and compare standalone dense search against hybrid Reciprocal Rank Fusion.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query & Hyperparameter Tuning")
    user_query = st.text_input("User Search Query", value="HNSW graph skip-layer search")
    
    top_k_slider = st.slider("Top-K Candidates", 1, 6, 3)
    thresh_slider = st.slider("Similarity Threshold Cutoff", 0.0, 1.0, 0.50, step=0.05)
    ef_slider = st.select_slider("Search Depth (efSearch)", options=[4, 8, 16, 32, 64], value=16)
    
    cat_select = st.selectbox("Metadata Filter", ["All", "Algorithms", "Databases", "Retrieval"])
    cat_arg = None if cat_select == "All" else cat_select

    enable_rrf = st.toggle("Enable Hybrid Search (Dense + BM25 Fusion)", value=True)

with col2:
    st.subheader("Retrieved Candidate Documents")
    q_vec_sim = [0.90, 0.25, 0.15, 0.10]
    q_tokens_sim = user_query.lower().split()

    results = ParameterizedSearchController.execute_search(
        query_vector=q_vec_sim,
        query_terms=q_tokens_sim,
        top_k=top_k_slider,
        threshold=thresh_slider,
        search_depth_ef=ef_slider,
        category_filter=cat_arg,
        enable_hybrid=enable_rrf
    )

    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df[["id", "category", "final_score", "text"]], use_container_width=True, hide_index=True)
        st.success(f"Retrieved {len(results)} chunks passing threshold >= {thresh_slider}.")
    else:
        st.warning(f"Zero documents passed the similarity threshold of {thresh_slider}! Fallback triggered.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: What is the 'Lost in the Middle' problem, and how does tuning Top-K mitigate it?"):
    st.markdown("""
    **Answer**:
    Research shows that Large Language Models recall information best when it is placed at the very beginning 
    or the very end of their input context window. When K is set too high (e.g., K = 20), critical answers 
    positioned in the middle of long prompts are often ignored, leading to hallucination. 
    Tuning Top-K to a concise window (K = 3 to 5) or using a Cross-Encoder Reranker to place the most relevant chunk 
    at the prompt boundary mitigates this failure mode.
    """)

with st.expander("Q2: Why is Reciprocal Rank Fusion (RRF) preferred over linear score combination in hybrid search?"):
    st.markdown("""
    **Answer**:
    Dense vector scores (cosine similarity bounded between -1.0 and 1.0 or 0.0 and 1.0) and sparse lexical scores 
    (BM25 unbounded positive scores like 8.42) operate on fundamentally incompatible scales with different distributions. 
    Linear combination (`alpha * dense + (1 - alpha) * sparse`) requires fragile score normalization that breaks as corpus sizes shift. 
    RRF depends strictly on document **ordinal rank positions** (1 / (k + rank)), making it robust, parameter-stable, 
    and distribution-agnostic across varying query domains.
    """)

with st.expander("Q3: How does the similarity threshold parameter protect enterprise RAG applications from hallucination?"):
    st.markdown("""
    **Answer**:
    A fixed Top-K search unconditionally returns K documents even when the query is completely outside the knowledge base. 
    The downstream LLM attempts to answer using these irrelevant chunks, producing hallucinations. 
    A calibrated similarity threshold acts as an automated relevance gate. 
    If all retrieved candidates score below the threshold, the application aborts generation and executes a deterministic fallback, 
    informing the user that no supporting documentation was found.
    """)

with st.expander("Q4: What happens if `efSearch` is set smaller than `top_k` in an HNSW query?"):
    st.markdown("""
    **Answer**:
    `efSearch` defines the size of the dynamic candidate priority queue maintained during graph traversal. 
    If `efSearch < top_k`, the search algorithm cannot track enough candidate nodes to return K items, 
    resulting in either an exception or an artificially truncated result set. 
    Engines enforce `efSearch >= top_k`, with standard production settings typically 2x to 4x higher than K for optimal recall.
    """)

with st.expander("Q5: What is the difference between single-stage filtered search and post-filtering?"):
    st.markdown("""
    **Answer**:
    * **Post-Filtering**: Executes standard vector search to find top-K candidates globally, then filters out non-matching records. If the target filter matches only a tiny subset of documents (e.g., 1%), post-filtering often returns an empty result set.
    * **Single-Stage Filtering (ACORN / Filtered HNSW)**: Evaluates metadata bitsets dynamically during graph traversal. Only valid nodes are added to the candidate queue, guaranteeing both 100% metadata compliance and high recall regardless of filter selectivity.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying Pure Dense Search for Code, Part Numbers, or Serial Codes**  
    Embedding models compress tokens into dense vectors, often losing exact string precision. 
    Queries searching for exact error codes (`ERR_404_TIMEOUT`) or product numbers fail unless paired with BM25 hybrid search.
    """)
    st.error("""
    **Trap 2: Using Top-K Retrieval Without a Minimum Similarity Threshold**  
    Allowing low-confidence documents with similarity < 0.50 to enter the LLM prompt forces the model 
    to hallucinate connections between irrelevant text and the user query. Always enforce a similarity cutoff.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Normalizing BM25 and Dense Scores Linearly Instead of Using RRF**  
    Attempting manual linear weight fusion (`0.7 * cosine + 0.3 * bm25`) creates fragile pipelines that fail 
    when document lengths change or BM25 scores spike unexpectedly. Prefer Reciprocal Rank Fusion (RRF).
    """)
    st.warning("""
    **Trap 4: Setting `efSearch` Too High for High-QPS Endpoints**  
    Setting `efSearch = 256` improves recall by less than 1% over `efSearch = 32`, but increases query latency 
    and CPU utilization by 4x. Keep `efSearch` calibrated to the minimum level meeting your recall target.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Reciprocal Rank Fusion (RRF)")
        st.caption("How does RRF combine rankings?")
        st.markdown("**Back**: Combines dense and sparse lists via `1 / (k + rank)`. Scale-agnostic, parameter-stable, and does not require score normalization.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Lost in the Middle")
        st.caption("Why limit Top-K?")
        st.markdown("**Back**: LLMs overlook information placed in the middle of long contexts. Keep Top-K concise (3-5 chunks) or rerank before prompting.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: efSearch Parameter")
        st.caption("What does efSearch govern?")
        st.markdown("**Back**: HNSW query exploration depth. Controls priority queue size during graph traversal; higher values improve recall at the cost of latency.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Similarity Threshold")
        st.caption("Why is it a security gate?")
        st.markdown("**Back**: Drops out-of-domain chunks before reaching the LLM, enabling safe fallback responses and preventing hallucinations.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Search Parameters:")

c1 = st.checkbox("I know how to tune Top-K to balance context completeness against the 'Lost in the Middle' problem.", value=True)
c2 = st.checkbox("I understand how similarity thresholds prevent hallucinations on out-of-domain queries.", value=True)
c3 = st.checkbox("I know how search depth parameters (`efSearch` and `nprobe`) trade off query latency against recall.", value=True)
c4 = st.checkbox("I understand the mechanics of hybrid search combining dense vectors and BM25 via Reciprocal Rank Fusion (RRF).", value=True)
c5 = st.checkbox("I can differentiate between pre-filtering, post-filtering, and single-stage filtered search.", value=True)
c6 = st.checkbox("I can implement a parameterized search controller with RRF fusion in production Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Search Parameters and understand how to fine-tune retrieval precision, depth, and hybrid fusion for production RAG pipelines.")
