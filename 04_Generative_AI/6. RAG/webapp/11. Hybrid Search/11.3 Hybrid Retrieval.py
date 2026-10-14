import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class HybridFusionEngine:
    """
    Simulates production dual-engine hybrid retrieval merging lexical (BM25)
    and dense neural (embedding) candidate lists using Reciprocal Rank Fusion (RRF)
    and Min-Max Normalized Linear Score Weighting.
    """
    CORPUS = {
        "doc_01": {
            "title": "Postgres Connection Pool Saturation",
            "text": "PostgreSQL connection pool exhausted. Troubleshoot PgBouncer max client connections and backend wait timeouts.",
            "category": "Database DevOps"
        },
        "doc_02": {
            "title": "Application Fatal Crash ERR_409",
            "text": "Production service crashed with fatal state ERR_409 conflict during concurrent database lock acquisition.",
            "category": "Error Runbook"
        },
        "doc_03": {
            "title": "Database Locking and Transaction Isolation",
            "text": "Understanding row-level locks, deadlocks, and serializable transaction isolation anomalies in relational databases.",
            "category": "Architecture"
        },
        "doc_04": {
            "title": "PgBouncer Connection Pooling Guide",
            "text": "Architecture overview of session, transaction, and statement pooling modes for high-scale SQL workloads.",
            "category": "Database DevOps"
        },
        "doc_05": {
            "title": "Microservice Distributed Tracing",
            "text": "Configuring OpenTelemetry spans and trace context propagation across distributed HTTP and gRPC gateways.",
            "category": "Observability"
        }
    }

    PRESET_SCENARIOS = {
        "Scenario A (Exact Error Code — Lexical Advantage)": {
            "query": "Troubleshoot ERR_409 database lock crash",
            "lexical": [
                {"id": "doc_02", "raw_score": 18.45},
                {"id": "doc_03", "raw_score": 6.20},
                {"id": "doc_01", "raw_score": 2.10}
            ],
            "dense": [
                {"id": "doc_03", "raw_score": 0.86},
                {"id": "doc_01", "raw_score": 0.81},
                {"id": "doc_02", "raw_score": 0.74}
            ]
        },
        "Scenario B (Conceptual Paraphrase — Dense Advantage)": {
            "query": "How do I prevent clients from overwhelming my relational database?",
            "lexical": [
                {"id": "doc_03", "raw_score": 4.10},
                {"id": "doc_05", "raw_score": 2.50}
            ],
            "dense": [
                {"id": "doc_01", "raw_score": 0.91},
                {"id": "doc_04", "raw_score": 0.88},
                {"id": "doc_03", "raw_score": 0.79}
            ]
        },
        "Scenario C (Consensus Ranking Demonstration)": {
            "query": "PgBouncer configuration for high concurrency connection timeouts",
            "lexical": [
                {"id": "doc_04", "raw_score": 14.80},
                {"id": "doc_01", "raw_score": 11.20},
                {"id": "doc_02", "raw_score": 3.40}
            ],
            "dense": [
                {"id": "doc_03", "raw_score": 0.89},
                {"id": "doc_01", "raw_score": 0.87},
                {"id": "doc_04", "raw_score": 0.82}
            ]
        }
    }

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        lexical_candidates: List[Dict[str, Any]],
        dense_candidates: List[Dict[str, Any]],
        k_rrf: int = 60,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score: Sum over retrievers of (1.0 / (k_rrf + rank))
        """
        rrf_scores: Dict[str, float] = {}
        lexical_ranks: Dict[str, int] = {}
        dense_ranks: Dict[str, int] = {}

        for rank, item in enumerate(lexical_candidates, start=1):
            d_id = item["id"]
            lexical_ranks[d_id] = rank
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k_rrf + rank))

        for rank, item in enumerate(dense_candidates, start=1):
            d_id = item["id"]
            dense_ranks[d_id] = rank
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k_rrf + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for d_id in sorted_ids[:top_k]:
            doc_meta = cls.CORPUS.get(d_id, {"title": "Unknown", "text": "", "category": "N/A"})
            results.append({
                "id": d_id,
                "title": doc_meta["title"],
                "category": doc_meta["category"],
                "rrf_score": round(rrf_scores[d_id], 5),
                "lexical_rank": lexical_ranks.get(d_id, "Missed"),
                "dense_rank": dense_ranks.get(d_id, "Missed"),
                "text": doc_meta["text"]
            })
        return results

    @classmethod
    def linear_weighted_fusion(
        cls,
        lexical_candidates: List[Dict[str, Any]],
        dense_candidates: List[Dict[str, Any]],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Normalizes scores using min-max scaling and calculates:
        Final_Score = (1 - alpha) * Norm_Lexical + alpha * Norm_Dense
        """
        all_ids = set([x["id"] for x in lexical_candidates] + [x["id"] for x in dense_candidates])

        # Normalize lexical scores
        lex_dict = {x["id"]: x["raw_score"] for x in lexical_candidates}
        dense_dict = {x["id"]: x["raw_score"] for x in dense_candidates}

        min_lex = min(lex_dict.values()) if lex_dict else 0.0
        max_lex = max(lex_dict.values()) if lex_dict else 1.0
        lex_span = max_lex - min_lex if max_lex > min_lex else 1.0

        min_dense = min(dense_dict.values()) if dense_dict else 0.0
        max_dense = max(dense_dict.values()) if dense_dict else 1.0
        dense_span = max_dense - min_dense if max_dense > min_dense else 1.0

        fused_scores = {}
        for d_id in all_ids:
            norm_lex = (lex_dict[d_id] - min_lex) / lex_span if d_id in lex_dict else 0.0
            norm_dense = (dense_dict[d_id] - min_dense) / dense_span if d_id in dense_dict else 0.0
            final_score = ((1.0 - alpha) * norm_lex) + (alpha * norm_dense)
            fused_scores[d_id] = {
                "final_score": round(final_score, 4),
                "norm_lexical": round(norm_lex, 4),
                "norm_dense": round(norm_dense, 4)
            }

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x]["final_score"], reverse=True)

        results = []
        for d_id in sorted_ids[:top_k]:
            doc_meta = cls.CORPUS.get(d_id, {"title": "Unknown", "text": "", "category": "N/A"})
            data = fused_scores[d_id]
            results.append({
                "id": d_id,
                "title": doc_meta["title"],
                "category": doc_meta["category"],
                "score": data["final_score"],
                "norm_lexical": data["norm_lexical"],
                "norm_dense": data["norm_dense"],
                "text": doc_meta["text"]
            })
        return results

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="11.3 Hybrid Retrieval - RAG Master Curriculum",
    page_icon="⚡",
    layout="wide"
)

st.title("11.3 Hybrid Retrieval: Combining BM25 & Dense Vectors, Weighted Fusion, & Reciprocal Rank Fusion")
st.caption("Chapter 11: Hybrid Search | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In production RAG systems, choosing between lexical keyword search and dense semantic search is a false dilemma. 
Both paradigms have distinct, complementary operational strengths and fatal blind spots:
* **BM25 Lexical Search** provides exact term precision, SKU lookups, and error code resolution, but suffers from synonym blindness.
* **Dense Semantic Search** excels at intent extraction, multilingual reasoning, and conceptual paraphrasing, but struggles with exact alphanumeric identifiers and negation.

**Hybrid Retrieval** executes both search engines concurrently and intelligently merges their candidate result lists. 
This lesson examines the algorithms that bridge these disparate engines—specifically **Score-Based Linear Weighting** 
and the industry standard **Reciprocal Rank Fusion (RRF)**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Dual-Engine Hybrid Retrieval Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    hybrid_bp_dot = """
    digraph HybridRetrievalBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_dispatch {
            label = "1. Query Ingress & Concurrent Dispatch Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="User Query\\ne.g. 'Troubleshoot ERR_409 database lock crash'", fillcolor="#0F172A", color="#38BDF8"];
            query_router [label="Async Parallel Dispatcher\\nExecutes Concurrent Coroutines", fillcolor="#334155", color="#10B981"];

            user_query -> query_router [label="Incoming Request"];
        }

        subgraph cluster_engines {
            label = "2. Dual Search Engines Tier (Concurrent Execution)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            bm25_engine [label="Engine A: Lexical BM25\\nInverted Index | TF-IDF Frequencies\\nOutput: Raw Scores [0.0 - 35.0+]", fillcolor="#0F172A", color="#F59E0B"];
            dense_engine [label="Engine B: Dense Vector Index\\nHNSW Graph | Embedding Dot Product\\nOutput: Cosine Scores [0.60 - 0.95]", fillcolor="#0F172A", color="#8B5CF6"];

            query_router -> bm25_engine [label="Tokenized Query Words"];
            query_router -> dense_engine [label="Dense Query Vector (f_q)"];
        }

        subgraph cluster_fusion {
            label = "3. Fusion & Normalization Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            fusion_layer [label="Fusion Engine\\nAlgorithm: RRF (k=60) OR Linear Weighted (alpha)", fillcolor="#334155", color="#EC4899"];
            dedup_layer [label="Deduplication & Union Gate\\nCombines Distinct Document Keys", fillcolor="#0F172A", color="#06B6D4"];
            topk_selector [label="Top-K Final Candidate Heap\\nSelects Top N Highest Fused Scores", fillcolor="#334155", color="#10B981"];

            bm25_engine -> fusion_layer [label="Top-M Lexical Hits"];
            dense_engine -> fusion_layer [label="Top-M Dense Hits"];
            fusion_layer -> dedup_layer [label="Score Calculation"];
            dedup_layer -> topk_selector [label="Merged Candidates"];
        }

        subgraph cluster_downstream {
            label = "4. Downstream RAG Synthesis Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            reranker [label="Optional Cross-Encoder Reranker\\nHigh-Precision Neural Scoring", fillcolor="#0F172A", color="#F59E0B"];
            llm_prompt [label="Context Augmentation & LLM Generation\\nGround Truth Context for Generator", fillcolor="#0F172A", color="#10B981"];

            topk_selector -> reranker [label="Top-25 Fused Items"];
            reranker -> llm_prompt [label="Top-5 Reranked Context"];
        }
    }
    """
    st.graphviz_chart(hybrid_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern enterprise architectures (e.g., **Elasticsearch v8+**, **Azure AI Search**, **Weaviate**, and **Qdrant**), 
hybrid retrieval runs as a single unified query API. 
The database engine dispatches concurrent workers across both inverted token postings and HNSW vector graphs, 
applying Reciprocal Rank Fusion natively in memory before returning candidates to the application tier.
""")

st.divider()

# ==============================================================================
# SECTION 2: COMBINING BM25 AND VECTOR SEARCH
# ==============================================================================

st.header("2. Combining BM25 and Vector Search: The Incompatible Scale Problem", divider="rainbow")

st.markdown("""
Why can systems not simply sum raw search scores via `Final_Score = BM25_Score + Dense_Score`?
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Apples vs. Oranges Dilemma")
    st.markdown("""
    * **Dense Cosine Scores**: Bounded strictly between -1.0 and +1.0 (clustering between 0.65 and 0.90 for relevant matches).
    * **BM25 Lexical Scores**: Unbounded positive floating-point values dependent on document length and inverse document frequency. 
      A highly specific query with multiple rare tokens can easily yield BM25 scores of **18.5**, **24.2**, or higher.
    * **The Naive Addition Trap**: Adding uncalibrated raw scores causes BM25 values to dominate dense scores 
      by an order of magnitude (20x or more), rendering the dense semantic vector search virtually irrelevant!
    * **System Resolution**: Hybrid systems must either:
      1. Calibrate and normalize raw scores onto a standard unit interval `[0.0, 1.0]`.
      2. Eliminate raw score dependencies entirely by utilizing positional ranks via **Reciprocal Rank Fusion**.
    """)

with col2:
    st.subheader("Complementary Strengths Comparison")
    comp_df = pd.DataFrame([
        {
            "Capability": "Exact Code & Identifier Matching",
            "BM25 Lexical": "Superior (100% precision on tokens)",
            "Dense Vectors": "Poor (fragmented subword tokens)"
        },
        {
            "Capability": "Synonym & Paraphrase Tolerance",
            "BM25 Lexical": "Zero (fails on vocabulary mismatch)",
            "Dense Vectors": "Superior (geometric proximity)"
        },
        {
            "Capability": "Polysemy Disambiguation",
            "BM25 Lexical": "Poor (matches token regardless of meaning)",
            "Dense Vectors": "Superior (contextual self-attention)"
        },
        {
            "Capability": "Out-of-Vocabulary / Rare Words",
            "BM25 Lexical": "Superior (high IDF weight boost)",
            "Dense Vectors": "Moderate to Poor"
        },
        {
            "Capability": "Infrastructure Complexity",
            "BM25 Lexical": "Low CPU, low RAM inverted index",
            "Dense Vectors": "High GPU/CPU, large vector memory"
        }
    ])
    st.dataframe(comp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: WEIGHTED RETRIEVAL
# ==============================================================================

st.header("3. Weighted Retrieval: Alpha Parameter Tuning and Convex Combinations", divider="rainbow")

st.markdown("""
In **Weighted Retrieval**, scores from each engine are normalized and blended using a tunable scalar balance parameter, 
commonly referred to as **Alpha** (`alpha`).
""")

col_alpha1, col_alpha2 = st.columns(2)

with col_alpha1:
    st.markdown("""
    **The Convex Combination Formula (Plain Text)**:
    `Score(doc) = ((1.0 - alpha) * Normalized_BM25(doc)) + (alpha * Normalized_Dense(doc))`
    
    * **When alpha = 0.0**: The system behaves as 100% pure BM25 keyword search.
    * **When alpha = 1.0**: The system behaves as 100% pure dense vector search.
    * **When alpha = 0.5**: Both engines carry equal weight.
    * **Production Default (0.60 to 0.70)**: Most production RAG systems favor dense semantics slightly 
      (`alpha = 0.65`) while retaining a 35% lexical backbone for exact keyword stability.
    """)

with col_alpha2:
    st.markdown("""
    **Dynamic Query Routing Based on Intent**:
    Rather than hardcoding a global alpha parameter across an entire enterprise application, advanced systems 
    adjust alpha adaptively:
    * **SKU / Identifier / Error Pattern Detected**: Set `alpha = 0.20` (heavily favors BM25).
    * **Short Question / Conceptual Paraphrase**: Set `alpha = 0.80` (heavily favors dense vectors).
    * **Standard Natural Language Search**: Set `alpha = 0.60` (balanced hybrid).
    """)

st.divider()

# ==============================================================================
# SECTION 4: RECIPROCAL RANK FUSION
# ==============================================================================

st.header("4. Reciprocal Rank Fusion (RRF): Robust Rank-Based Consensus", divider="rainbow")

st.markdown("""
**Reciprocal Rank Fusion (RRF)** is the industry-standard algorithm for combining candidate rankings from disparate 
search systems. Introduced by Cormack, Clarke, and Büttcher (2009), RRF discards raw score magnitudes entirely 
and merges documents based strictly on their **ordinal rank positions**.
""")

col_rrf1, col_rrf2 = st.columns([1, 1])

with col_rrf1:
    st.subheader("The RRF Formula Explained in Plain Text")
    st.markdown("""
    For each candidate document, its RRF score is computed as:
    
    `RRF_Score(doc) = Sum over all retrievers of (1.0 / (k_rrf + rank(doc)))`
    
    * **rank(doc)**: The 1-based integer position of the document in a specific retriever's result list (1 for 1st place, 2 for 2nd place, etc.).
    * **k_rrf**: A constant smoothing parameter (standard production empirical default is **60**).
    * **Missed Documents**: If a retriever does not return the document in its top candidate list, its reciprocal term for that retriever is 0.0.
    """)

with col_rrf2:
    st.subheader("Why the k_rrf = 60 Constant Matters")
    st.markdown("""
    The smoothing parameter `k_rrf = 60` prevents outlier dominance:
    * Without smoothing (`k = 0`), a rank 1 document gets score `1/1 = 1.0`, while rank 2 gets `1/2 = 0.5`. 
      A single top rank would overpower any document appearing in rank 2 or 3 across multiple systems.
    * With `k = 60`:
      * Rank 1 score: `1 / (60 + 1) = 0.01639`
      * Rank 2 score: `1 / (60 + 2) = 0.01613`
      * Rank 3 score: `1 / (60 + 3) = 0.01587`
    * **The Consensus Effect**: A document ranked #2 by BM25 and #2 by Dense vectors earns:
      `0.01613 + 0.01613 = 0.03226`. 
      This completely beats a document ranked #1 by BM25 but missed entirely by Dense search (`0.01639`)!
    """)

st.divider()

# ==============================================================================
# SECTION 5: COMBINING SEARCH SCORES
# ==============================================================================

st.header("5. Combining Search Scores: Min-Max, Softmax, and Rank Transformations", divider="rainbow")

st.markdown("""
When systems choose score-based fusion over rank-based fusion, raw scores must undergo rigorous transformation 
to ensure mathematical compatibility.
""")

norm_df = pd.DataFrame([
    {
        "Transformation Strategy": "1. Min-Max Normalization",
        "Plain Text Formula": "Norm_Score = (Score - Min_Score) / (Max_Score - Min_Score)",
        "Pros": "Strictly bounds all scores to [0.0, 1.0]. Simple to implement in CPU memory.",
        "Cons": "Highly sensitive to outlier scores in small Top-K windows.",
        "Production Engines": "Weaviate, Pinecone Hybrid"
    },
    {
        "Transformation Strategy": "2. Z-Score Standardization",
        "Plain Text Formula": "Norm_Score = (Score - Mean_Score) / Std_Deviation",
        "Pros": "Centers distributions around zero; robust to variance shifts.",
        "Cons": "Does not produce bounded values; negative values require sigmoid mapping.",
        "Production Engines": "Statistical Information Retrieval toolkits"
    },
    {
        "Transformation Strategy": "3. Softmax / Logistic Sigmoid",
        "Plain Text Formula": "Norm_Score = 1.0 / (1.0 + exp(-Scale * (Score - Offset)))",
        "Pros": "Smooth non-linear calibration; suppresses extreme tails.",
        "Cons": "Requires tuning scale and offset hyperparameters per corpus.",
        "Production Engines": "Neural search rerankers"
    },
    {
        "Transformation Strategy": "4. Reciprocal Rank Fusion (RRF)",
        "Plain Text Formula": "RRF_Score = Sum(1.0 / (60 + Rank))",
        "Pros": "Zero normalization required; immune to scale differences; robust across all corpora.",
        "Cons": "Loses magnitude information (cannot distinguish a landslide victory from a narrow win).",
        "Production Engines": "Elasticsearch, Azure AI Search, LangChain EnsembleRetriever"
    }
])
st.dataframe(norm_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: HYBRID SEARCH ARCHITECTURE
# ==============================================================================

st.header("6. Hybrid Search Architecture: Concurrency, Sharding, and Distributed Aggregation", divider="rainbow")

st.markdown("""
Designing a hybrid retrieval tier capable of sustaining sub-50ms latency across millions of documents requires 
careful architectural orchestration:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.subheader("Key Architectural Pillars")
    st.markdown("""
    1. **Asynchronous Parallel Dispatch**:
       * The ingress API gateway receives the user query and forks two non-blocking coroutines concurrently: 
         one to the BM25 cluster and one to the Vector DB cluster.
    2. **Over-Fetching Strategy (Candidate Multiplier)**:
       * If the downstream RAG synthesis pipeline needs `Top-K = 5` documents, the hybrid engine fetches 
         `M = 25` to `50` candidates from *each* independent retriever. 
         This guarantees that candidates achieving consensus across both systems are discovered even if ranked lower initially.
    3. **In-Memory Union and Deduplication**:
       * Results from both streams are merged using unique document IDs.
    4. **Second-Stage Cross-Encoder Reranking**:
       * The top 25 merged hybrid candidates are passed to a Cross-Encoder reranker, selecting the definitive Top-5 
         for generator LLM prompt assembly.
    """)

with col_arch2:
    st.subheader("Distributed Dataflow Blueprint")
    dist_arch_dot = """
    digraph DistributedHybrid {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.35, ranksep=0.45];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=8, fontcolor="#94A3B8"];

        query [label="Client Query", fillcolor="#0F172A", color="#38BDF8"];
        gateway [label="Hybrid Gateway Orchestrator", fillcolor="#334155", color="#10B981"];

        bm25_node [label="BM25 Shard Nodes\\n(Lucene / Tantivy)", fillcolor="#0F172A", color="#F59E0B"];
        vdb_node [label="Vector Shard Nodes\\n(HNSW Qdrant / Milvus)", fillcolor="#0F172A", color="#8B5CF6"];

        aggregator [label="Fusion & RRF Aggregator Node", fillcolor="#334155", color="#EC4899"];
        final_topk [label="Top-K Fused Results", fillcolor="#0F172A", color="#10B981"];

        query -> gateway;
        gateway -> bm25_node [label="async fork"];
        gateway -> vdb_node [label="async fork"];
        bm25_node -> aggregator [label="Top-50 hits"];
        vdb_node -> aggregator [label="Top-50 hits"];
        aggregator -> final_topk [label="RRF consensus"];
    }
    """
    st.graphviz_chart(dist_arch_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Hybrid Fusion Engine", divider="rainbow")

st.markdown("""
Below is a battle-tested Python class implementing concurrent hybrid retrieval with both 
**Reciprocal Rank Fusion (RRF)** and **Min-Max Normalized Convex Weighting**.
""")

st.code('''
from typing import List, Dict, Any, Optional
from collections import defaultdict

class EnterpriseHybridRetriever:
    """
    Enterprise-grade hybrid retrieval combiner supporting both
    Reciprocal Rank Fusion (RRF) and Min-Max Linear Score Fusion.
    """
    def __init__(self, k_rrf: int = 60, default_alpha: float = 0.60):
        self.k_rrf = k_rrf
        self.default_alpha = default_alpha

    def fuse_rrf(
        self,
        retriever_candidate_lists: List[List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Merges an arbitrary number of ranked candidate lists using Reciprocal Rank Fusion.
        Each candidate is expected to contain at least {"id": str, ...}.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict[str, Any]] = {}
        appearances: Dict[str, List[int]] = defaultdict(list)

        for engine_idx, candidate_list in enumerate(retriever_candidate_lists):
            for rank_idx, candidate in enumerate(candidate_list, start=1):
                doc_id = candidate["id"]
                doc_store[doc_id] = candidate
                appearances[doc_id].append(rank_idx)
                # Reciprocal Rank Fusion formula
                rrf_scores[doc_id] += 1.0 / (self.k_rrf + rank_idx)

        # Sort descending by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        results = []
        for d_id in sorted_ids[:top_k]:
            item = dict(doc_store[d_id])
            item["rrf_score"] = float(round(rrf_scores[d_id], 5))
            item["engine_ranks"] = appearances[d_id]
            results.append(item)
        return results

    def fuse_linear_weighted(
        self,
        lexical_hits: List[Dict[str, Any]],
        dense_hits: List[Dict[str, Any]],
        alpha: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Merges lexical and dense candidate lists using Min-Max Normalized Convex Weighting:
        Score = (1 - alpha) * Norm_Lexical + alpha * Norm_Dense
        """
        active_alpha = self.default_alpha if alpha is None else alpha
        all_ids = set([x["id"] for x in lexical_hits] + [x["id"] for x in dense_hits])
        doc_store: Dict[str, Dict[str, Any]] = {}

        # Extract raw scores
        lex_map = {x["id"]: float(x["score"]) for x in lexical_hits}
        dense_map = {x["id"]: float(x["score"]) for x in dense_hits}
        for x in lexical_hits + dense_hits:
            doc_store[x["id"]] = x

        # Min-max normalization for lexical
        lex_vals = list(lex_map.values())
        min_lex = min(lex_vals) if lex_vals else 0.0
        max_lex = max(lex_vals) if lex_vals else 1.0
        span_lex = max_lex - min_lex if max_lex > min_lex else 1.0

        # Min-max normalization for dense
        dense_vals = list(dense_map.values())
        min_dense = min(dense_vals) if dense_vals else 0.0
        max_dense = max(dense_vals) if dense_vals else 1.0
        span_dense = max_dense - min_dense if max_dense > min_dense else 1.0

        fused_scores = {}
        for d_id in all_ids:
            norm_lex = (lex_map[d_id] - min_lex) / span_lex if d_id in lex_map else 0.0
            norm_dense = (dense_map[d_id] - min_dense) / span_dense if d_id in dense_map else 0.0
            fused_score = ((1.0 - active_alpha) * norm_lex) + (active_alpha * norm_dense)
            fused_scores[d_id] = fused_score

        sorted_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)

        results = []
        for d_id in sorted_ids[:top_k]:
            item = dict(doc_store[d_id])
            item["fused_score"] = float(round(fused_scores[d_id], 4))
            results.append(item)
        return results
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Hybrid Fusion Simulator (RRF vs. Weighted Linear)", divider="rainbow")

st.markdown("""
Experiment with candidate lists from lexical and dense retrieval engines. Observe how **Reciprocal Rank Fusion** 
and **Linear Convex Weighting** resolve rank conflicts and reward engine consensus.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    scenario_label = st.selectbox(
        "Select Test Scenario:",
        list(HybridFusionEngine.PRESET_SCENARIOS.keys())
    )
    fusion_algorithm = st.selectbox(
        "Fusion Algorithm:",
        ["Reciprocal Rank Fusion (RRF)", "Linear Weighted Score Fusion (Alpha)"]
    )

    scenario_data = HybridFusionEngine.PRESET_SCENARIOS[scenario_label]
    st.info(f"**Query**: *\"{scenario_data['query']}\"*")

    if fusion_algorithm == "Reciprocal Rank Fusion (RRF)":
        k_param = st.slider("Smoothing Parameter (k_rrf):", min_value=1, max_value=100, value=60)
        alpha_param = 0.5
    else:
        alpha_param = st.slider("Dense Weight (Alpha): 0.0 = Pure Lexical, 1.0 = Pure Dense", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
        k_param = 60

    top_k_select = st.slider("Top-K Candidates to Return:", min_value=1, max_value=5, value=3)

with wb_col2:
    st.markdown("### 📥 Raw Candidate Stream Inputs")
    col_raw_lex, col_raw_dense = st.columns(2)
    with col_raw_lex:
        st.markdown("**Lexical BM25 Stream (Raw Scores)**")
        raw_lex_df = pd.DataFrame([
            {"Doc ID": x["id"], "BM25 Score": x["raw_score"], "Title": HybridFusionEngine.CORPUS[x["id"]]["title"]}
            for x in scenario_data["lexical"]
        ])
        st.dataframe(raw_lex_df, use_container_width=True)

    with col_raw_dense:
        st.markdown("**Dense Vector Stream (Cosine Scores)**")
        raw_dense_df = pd.DataFrame([
            {"Doc ID": x["id"], "Dense Score": x["raw_score"], "Title": HybridFusionEngine.CORPUS[x["id"]]["title"]}
            for x in scenario_data["dense"]
        ])
        st.dataframe(raw_dense_df, use_container_width=True)

st.markdown("### 🏆 Unified Fusion Output")

if fusion_algorithm == "Reciprocal Rank Fusion (RRF)":
    results = HybridFusionEngine.reciprocal_rank_fusion(
        lexical_candidates=scenario_data["lexical"],
        dense_candidates=scenario_data["dense"],
        k_rrf=k_param,
        top_k=top_k_select
    )
    res_df = pd.DataFrame(results)
    st.dataframe(res_df[["id", "title", "category", "rrf_score", "lexical_rank", "dense_rank", "text"]], use_container_width=True)
else:
    results = HybridFusionEngine.linear_weighted_fusion(
        lexical_candidates=scenario_data["lexical"],
        dense_candidates=scenario_data["dense"],
        alpha=alpha_param,
        top_k=top_k_select
    )
    res_df = pd.DataFrame(results)
    st.dataframe(res_df[["id", "title", "category", "score", "norm_lexical", "norm_dense", "text"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is Reciprocal Rank Fusion preferred over Linear Score Weighting in production multi-engine systems?"):
    st.markdown("""
    **Answer**: Linear Score Weighting requires normalizing scores from each retriever onto a common scale (e.g., via min-max scaling). 
    However, min-max scaling is highly vulnerable to outliers within small Top-K result windows, and the score distributions of BM25 
    and vector embeddings shift drastically across different query intents. 
    Reciprocal Rank Fusion (RRF) bypasses score normalization entirely by relying solely on ordinal candidate ranks. 
    It is mathematically immune to disparate score distributions, requires zero hyperparameter tuning across datasets, 
    and naturally promotes documents that achieve consensus across multiple retrieval engines.
    """)

with st.expander("Q2: In Reciprocal Rank Fusion, what is the role of the smoothing constant k_rrf = 60?"):
    st.markdown("""
    **Answer**: The constant `k_rrf = 60` was empirically established by Cormack et al. to solve the "top-rank monopoly" problem. 
    Without smoothing (e.g., `k = 0`), a document ranked #1 by one engine receives an overwhelming score of `1.0`, 
    completely crushing a document ranked #2 across three separate engines (`0.5 + 0.5 + 0.5 = 1.5` vs `1.0`). 
    Setting `k = 60` compresses the reciprocal differences between early ranks:
    Rank 1 earns `1/61 = 0.01639`, while Rank 2 earns `1/62 = 0.01613`. 
    This gentle gradient ensures that broad consensus across multiple retrievers reliably outweighs an isolated #1 spike from a single noisy retriever.
    """)

with st.expander("Q3: How does over-fetching candidate documents impact hybrid retrieval recall?"):
    st.markdown("""
    **Answer**: If a RAG pipeline requires `Top-K = 5` final context documents, fetching only 5 candidates from BM25 and 5 from Dense search 
    results in poor hybrid recall. Often, a highly relevant consensus document might be ranked #7 in BM25 and #9 in Vector search. 
    If both retrievers truncate at K=5, neither retriever passes this consensus document to the fusion layer. 
    In production, hybrid systems employ an over-fetching multiplier (e.g., fetching 25 to 50 candidates per engine), 
    allowing consensus candidates to surface to the top of the fused ranking before truncating to the final Top-5.
    """)

with st.expander("Q4: When is Linear Weighted Hybrid Search superior to Reciprocal Rank Fusion?"):
    st.markdown("""
    **Answer**: Linear Weighted Hybrid Search is superior when domain experts have high confidence in one engine's calibrated score magnitudes 
    or when business logic requires dynamic alpha tuning based on query classification. 
    For example, in an e-commerce search engine where the system detects an exact product model number (e.g., *"RTX 4090 24GB"*), 
    setting `alpha = 0.1` forcefully privileges lexical matching. 
    RRF cannot enforce strong asymmetric preference as cleanly because it only considers rank order, not confidence magnitude.
    """)

with st.expander("Q5: What are the infrastructure and latency trade-offs of executing hybrid search concurrently?"):
    st.markdown("""
    **Answer**: Executing hybrid search concurrently incurs the wall-clock latency of the slower engine:
    `Latency_Hybrid = Max(Latency_BM25, Latency_Dense) + Latency_Fusion`. 
    Because inverted index lookups in Lucene/BM25 are typically 2–8ms and vector HNSW traversals are 10–25ms, total latency remains under 30ms 
    when invoked asynchronously. However, infrastructure resource consumption doubles: every user query triggers CPU/RAM loads 
    across both inverted index clusters and vector index clusters simultaneously, requiring independent autoscaling policies.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Naive Score Summation Without Normalization**
    * **Symptom**: Adding BM25 and dense scores directly causes search results to mirror 100% lexical search, ignoring embeddings.
    * **Root Cause**: Dense cosine scores cluster between 0.60 and 0.90, while BM25 scores are unbounded and routinely reach 20.0+. The raw BM25 score dwarfs the dense score by 25x.
    * **Remedy**: Always use Reciprocal Rank Fusion (RRF) or apply min-max scaling to both candidate lists before linear summation.
    """)

    st.error("""
    **Trap 2: Truncating Candidates Before Fusion (The Under-Fetching Trap)**
    * **Symptom**: Hybrid search recall is barely higher than either single engine alone.
    * **Root Cause**: Fetching only Top-K candidates (e.g., K=5) from each engine before fusing. High-consensus documents ranked #6 to #15 in both engines are discarded before fusion occurs.
    * **Remedy**: Over-fetch candidate pools (e.g., fetch Top-25 to Top-50 from each engine), merge and fuse across the union, and truncate to Top-5 only after fusion.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Outlier Distortions in Min-Max Scaling on Small Result Lists**
    * **Symptom**: In linear weighting, a single document with an unusually high BM25 score compresses all other lexical candidates to near-zero scores.
    * **Root Cause**: Min-max normalization depends entirely on the minimum and maximum scores present in the returned window. An outlier score skews the denominator `(max - min)`.
    * **Remedy**: Default to RRF in production, or apply robust z-score standardization with outlier clipping before linear combination.
    """)

    st.warning("""
    **Trap 4: Sequential Execution of Lexical and Vector Queries**
    * **Symptom**: Search API latency doubles from 18ms to 38ms upon deploying hybrid search.
    * **Root Cause**: Executing BM25 search first, waiting for results, and then executing vector search synchronously (`Latency = BM25 + Vector`).
    * **Remedy**: Dispatch both queries concurrently using `asyncio.gather` or multi-threaded executors so latency equals the maximum of the two engines plus sub-millisecond fusion.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Reciprocal Rank Fusion (RRF) and what is its primary mathematical benefit?")
        st.markdown("**Back**: An algorithm that combines candidate rankings using `Sum(1.0 / (k + rank))`. Its primary benefit is that it requires zero score normalization, making it immune to disparate scale differences between BM25 and dense vector models.")

    with st.container(border=True):
        st.markdown("**Front**: Why is the smoothing constant k_rrf set to 60 by standard convention?")
        st.markdown("**Back**: Setting k=60 prevents a document ranked #1 by an isolated noisy retriever from dominating the list, ensuring that multi-engine consensus across ranks 2 to 5 reliably wins.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the Convex Combination Formula in Weighted Hybrid Retrieval?")
        st.markdown("**Back**: `Final_Score = (1 - alpha) * Normalized_BM25 + alpha * Normalized_Dense`, where alpha tunes the relative balance between lexical keyword precision and dense semantic similarity.")

    with st.container(border=True):
        st.markdown("**Front**: Why is over-fetching mandatory in production hybrid retrieval?")
        st.markdown("**Back**: Because documents that represent the best overall consensus often rank in positions 6–25 across individual engines. Fetching only Top-5 from each engine causes these mutual consensus candidates to be lost before fusion.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Hybrid Retrieval, Reciprocal Rank Fusion, and Weighted Score Normalization:
""")

chk1 = st.checkbox("I understand why raw BM25 scores and dense cosine scores cannot be summed directly without normalization.", value=True)
chk2 = st.checkbox("I can articulate how Reciprocal Rank Fusion (RRF) operates strictly on ordinal ranks without score dependencies.", value=True)
chk3 = st.checkbox("I know why the smoothing constant k=60 prevents outlier dominance and rewards multi-engine consensus.", value=True)
chk4 = st.checkbox("I understand the necessity of candidate over-fetching (e.g. fetching Top-25 to Top-50) before rank fusion.", value=True)
chk5 = st.checkbox("I know how to tune the alpha parameter in linear weighted retrieval for lexical vs. semantic dominance.", value=True)
chk6 = st.checkbox("I understand how asynchronous concurrent dispatch prevents hybrid search latency from becoming additive.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Masterful! You have conquered Hybrid Retrieval and Enterprise Fusion architectures.")
