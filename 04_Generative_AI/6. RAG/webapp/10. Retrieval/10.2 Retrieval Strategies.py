import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class RetrievalStrategySimulator:
    """
    Simulates similarity search, score thresholding, Maximum Marginal Relevance (MMR),
    metadata filtering, and keyword vs semantic retrieval.
    """
    DOCUMENTS = [
        {"id": "doc_1", "text": "HNSW graphs provide fast approximate nearest neighbor search via multi-layer skip connections.", "vector": [0.90, 0.20, 0.10, 0.05], "category": "Algorithms", "tokens": ["hnsw", "graph", "ann", "skip"]},
        {"id": "doc_2", "text": "HNSW index construction builds multi-tier graph hierarchies for low-latency vector proximity.", "vector": [0.89, 0.22, 0.08, 0.06], "category": "Algorithms", "tokens": ["hnsw", "index", "graph", "proximity"]},
        {"id": "doc_3", "text": "PostgreSQL pgvector extension enables relational SQL queries combined with cosine vector distance.", "vector": [0.40, 0.85, 0.15, 0.10], "category": "Databases", "tokens": ["postgresql", "pgvector", "sql", "cosine"]},
        {"id": "doc_4", "text": "Product Quantization compresses dense vector coordinates into compact 8-bit byte codes.", "vector": [0.72, 0.15, 0.60, 0.15], "category": "Algorithms", "tokens": ["product", "quantization", "byte", "compression"]},
        {"id": "doc_5", "text": "Single-stage filtered search evaluates metadata during graph traversal to preserve recall.", "vector": [0.78, 0.30, 0.15, 0.40], "category": "Retrieval", "tokens": ["filtered", "search", "metadata", "recall"]},
        {"id": "doc_6", "text": "BM25 keyword search ranks documents based on term frequency and inverted document frequency.", "vector": [0.30, 0.25, 0.10, 0.85], "category": "Retrieval", "tokens": ["bm25", "keyword", "tf", "idf"]}
    ]

    @classmethod
    def compute_mmr(
        cls,
        query_vector: List[float],
        lambda_mult: float = 0.5,
        top_k: int = 3,
        threshold: float = 0.0,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure NumPy Maximum Marginal Relevance (MMR) implementation:
        MMR = ArgMax [ lambda * Sim(Doc, Query) - (1 - lambda) * Max(Sim(Doc, Selected)) ]
        """
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        qv = qv / q_norm if q_norm > 0 else qv

        # 1. Filter candidates
        candidates = []
        for doc in cls.CORPUS if hasattr(cls, "CORPUS") else cls.DOCUMENTS:
            if category_filter and doc["category"] != category_filter:
                continue
            dv = np.array(doc["vector"], dtype=np.float32)
            d_norm = np.linalg.norm(dv)
            sim = float(np.dot(qv, dv / d_norm if d_norm > 0 else dv))
            if sim >= threshold:
                candidates.append((doc, sim, dv / d_norm if d_norm > 0 else dv))

        if not candidates:
            return []

        selected: List[Dict[str, Any]] = []
        selected_vectors: List[np.ndarray] = []

        while len(selected) < min(top_k, len(candidates)):
            best_mmr_score = -1e9
            best_idx = -1
            best_doc = None
            best_sim_q = 0.0
            best_vec = None

            for i, (doc, sim_q, d_vec) in enumerate(candidates):
                if any(doc["id"] == s["id"] for s in selected):
                    continue

                # Diversity penalty: max similarity to already selected documents
                if selected_vectors:
                    max_sim_selected = max(float(np.dot(d_vec, sv)) for sv in selected_vectors)
                else:
                    max_sim_selected = 0.0

                mmr_score = (lambda_mult * sim_q) - ((1.0 - lambda_mult) * max_sim_selected)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
                    best_doc = doc
                    best_sim_q = sim_q
                    best_vec = d_vec

            if best_doc is None:
                break

            selected.append({
                "id": best_doc["id"],
                "text": best_doc["text"],
                "category": best_doc["category"],
                "relevance_score": round(best_sim_q, 4),
                "mmr_score": round(best_mmr_score, 4)
            })
            selected_vectors.append(best_vec)

        return selected


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="10.2 Retrieval Strategies - RAG Master Curriculum",
    page_icon="🎯",
    layout="wide"
)

st.title("10.2 Retrieval Strategies: Similarity Search, Score Thresholds, MMR Diversity, and Multi-Modal Retrieval")
st.caption("Chapter 10: Retrieval | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Modern RAG systems rarely rely on vanilla nearest-neighbor search alone. 
Simple vector similarity queries suffer from **semantic redundancy**—retrieving three nearly identical document chunks 
that waste context tokens without providing additional information.

Production RAG architectures implement diverse retrieval strategies: **Similarity Threshold Gating**, 
**Maximum Marginal Relevance (MMR)** for diversity, **Metadata Predicate Filtering**, and **Hybrid Keyword/Semantic routing**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Multi-Strategy Retrieval Pipeline", divider="rainbow")
st.markdown("""
The blueprint below illustrates how different retrieval strategies branch from a single query 
and apply distinct filtering, diversity re-ranking, and candidate selection rules.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    strat_bp_dot = """
    digraph RetrievalStrategies {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_q {
            label = "1. Query Intent & Dispatcher";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_in [label="Incoming User Query\\n+ Strategy Configuration", fillcolor="#0F172A", color="#38BDF8"];
            strat_router [label="Strategy Router & Candidate Fetcher\\nFetches Coarse Candidate Pool (K = 20)", fillcolor="#0F172A", color="#38BDF8"];
            q_in -> strat_router;
        }

        subgraph cluster_strategies {
            label = "2. Active Retrieval Strategies";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            s_sim [label="A. Similarity Search\\nPure Cosine/Dot Product Proximity", fillcolor="#334155", color="#38BDF8"];
            s_thresh [label="B. Similarity Threshold\\nRelevance Gate: Discard if Score < cutoff", fillcolor="#334155", color="#EF4444"];
            s_mmr [label="C. Maximum Marginal Relevance (MMR)\\nBalances Relevance vs Novelty (Lambda)", fillcolor="#334155", color="#10B981"];
            s_meta [label="D. Metadata Filter\\nScalar Predicates (Tenant, Dept, Date)", fillcolor="#334155", color="#F59E0B"];
            s_kw [label="E. Keyword Retrieval\\nBM25 Exact Lexical Term Overlap", fillcolor="#334155", color="#8B5CF6"];

            strat_router -> s_sim;
            strat_router -> s_thresh;
            strat_router -> s_mmr;
            strat_router -> s_meta;
            strat_router -> s_kw;
        }

        subgraph cluster_out {
            label = "3. Final Candidate Selection";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            final_pool [label="De-duplicated & Calibrated Top-K Context\\nOptimal Balance of Coverage and Diversity", fillcolor="#0F172A", color="#10B981"];

            s_sim -> final_pool;
            s_thresh -> final_pool;
            s_mmr -> final_pool;
            s_meta -> final_pool;
            s_kw -> final_pool;
        }
    }
    """
    st.graphviz_chart(strat_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise production (e.g., **LangChain**, **LlamaIndex**, and **Haystack**), retrievers are configured via strategy parameters:
`retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6})`. 
Instead of returning 5 paragraphs all explaining the same introductory definition of HNSW, 
MMR fetches 20 candidates and algorithmically picks 5 distinct, non-redundant passages covering definitions, tradeoffs, memory bounds, and code examples.
""")

st.divider()

# ==============================================================================
# SECTION 2: SIMILARITY SEARCH
# ==============================================================================

st.header("2. Similarity Search", divider="rainbow")
st.markdown("""
**Similarity Search** is the baseline retrieval strategy where documents are ranked strictly by geometric proximity 
between query and document vectors.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Mechanics and Strengths")
    st.markdown("""
    * **Direct Vector Ranking**:
      Calculates cosine similarity or inner product across indexed vectors and returns the top K highest-scoring records.
    * **Strengths**:
      Fastest retrieval execution; requires no secondary deduplication passes or inter-document cross-similarity calculations.
    * **Primary Vulnerability: Redundancy Clustering**:
      If a 100-page PDF mentions a specific topic across 10 consecutive paragraphs, vanilla similarity search will return 
      5 slight variations of the exact same paragraph, crowding out other relevant sections.
    """)

with col2:
    st.subheader("When to Use Pure Similarity Search")
    st.markdown("""
    * **Narrow Fact Retrieval**: Lookups where a single specific paragraph contains the entire required answer.
    * **First-Stage Retrieval in Two-Stage Pipelines**: Pulling coarse candidates (K = 50) that will subsequently be pruned by a downstream Cross-Encoder Reranker.
    * **High-Throughput Micro-Services**: Workloads requiring the lowest possible latency (< 3 ms) where MMR compute overhead cannot be tolerated.
    """)

st.divider()

# ==============================================================================
# SECTION 3: SIMILARITY SCORE THRESHOLD
# ==============================================================================

st.header("3. Similarity Score Threshold", divider="rainbow")
st.markdown("""
**Similarity Score Thresholding** introduces an automated relevance gate, returning only documents whose similarity 
score meets or exceeds a pre-configured cutoff value.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dynamic Result Sizing")
    st.markdown("""
    Unlike fixed Top-K retrieval which always returns exactly K items regardless of quality, 
    threshold-gated retrieval dynamically adjusts the result set size:
    * **Highly Relevant Query**: Returns all 5 matching documents.
    * **Partially Relevant Query**: Returns only the 2 high-confidence documents.
    * **Irrelevant / Out-of-Domain Query**: Returns 0 documents, preventing hallucinations.
    """)

with col2:
    st.subheader("Code Example: Threshold Retriever in LangChain")
    st.code('''# Score Threshold Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.75,
        "k": 5
    }
)

docs = retriever.invoke("What are HNSW memory tradeoffs?")
if not docs:
    print("No relevant context found above confidence threshold.")
''', language="python")

st.divider()

# ==============================================================================
# SECTION 4: MAXIMUM MARGINAL RELEVANCE (MMR)
# ==============================================================================

st.header("4. Maximum Marginal Relevance (MMR & Diversity Optimization)", divider="rainbow")
st.markdown("""
**Maximum Marginal Relevance (MMR)** algorithmically eliminates semantic redundancy by balancing 
**query relevance** against **document novelty**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The MMR Mathematical Objective")
    st.markdown("""
    At each selection step, MMR iteratively picks the next document d_i that maximizes:
    
    `MMR_Score = (lambda * Sim(d_i, Query)) - ((1 - lambda) * Max_Sim(d_i, Selected_Docs))`
    
    * **Lambda Mult (lambda) Tuning**:
      * lambda = 1.0: Pure similarity search (zero diversity consideration).
      * lambda = 0.0: Maximum diversity (picks documents maximally distant from each other).
      * lambda = 0.5 to 0.7: Optimal production balance between topical relevance and content diversity.
    * **Algorithm Execution**:
      Pulls a broad candidate pool of `fetch_k` vectors (e.g., 20), selects the top hit, and iteratively selects remaining items by penalizing items similar to already selected documents.
    """)

with col2:
    st.subheader("MMR Diversity Progression")
    mmr_flow_dot = """
    digraph MMRFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        pool [label="Candidate Pool (fetch_k = 20)", fillcolor="#0F172A"];
        first_doc [label="Pick Top Hit (Highest Sim to Query)"];
        eval_rem [label="Score Remaining Candidates\\nRelevance vs Distance to Selected"];
        pick_diverse [label="Select Candidate with Max MMR Score"];
        loop [label="Repeat until Top-K Reached", fillcolor="#0F172A", color="#10B981"];

        pool -> first_doc -> eval_rem -> pick_diverse -> loop;
    }
    """
    st.graphviz_chart(mmr_flow_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: METADATA FILTERING
# ==============================================================================

st.header("5. Metadata Filtering", divider="rainbow")
st.markdown("""
**Metadata Filtering** constrains retrieval using structured scalar criteria (such as tenant ID, author, date, or security level) 
alongside dense semantic vectors.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Filtering Modalities")
    st.markdown("""
    * **Pre-Filtering**: Evaluates scalar index first; runs vector search on matching subset.
    * **Post-Filtering**: Runs vector search first; discards non-matching payloads.
    * **Single-Stage (Filtered HNSW / ACORN)**: Inspects metadata bitsets during each graph traversal step, preserving 100% metadata compliance and high recall.
    """)

with col2:
    st.subheader("Code Example: Parameterized Filter Retriever")
    st.code('''# Filtered Retriever Configuration
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {
            "department": "Engineering",
            "access_tier": {"$lte": 2}
        },
        "k": 4
    }
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 6: KEYWORD RETRIEVAL
# ==============================================================================

st.header("6. Keyword Retrieval", divider="rainbow")
st.markdown("""
**Keyword Retrieval** relies on classical lexical term matching (such as BM25 or TF-IDF) over inverted indices.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why Keyword Retrieval Remains Crucial")
    st.markdown("""
    Dense embedding models compress sentences into continuous coordinates, often losing precision for exact token sequences. 
    Keyword retrieval excels where vector retrieval fails:
    * **Unique Identifiers**: Part numbers (`SKU-4902-X`), UUIDs, and API tokens.
    * **Exact Code Tokens**: Method names (`def calculate_p95_latency()`), error codes (`ERR_CONNECTION_REFUSED`).
    * **Domain-Specific Jargon**: Highly specialized legal, medical, or chemical nomenclature not well-represented in general embedding training corpora.
    """)

with col2:
    st.subheader("Lexical Scoring Profile")
    st.markdown("""
    * **Term Frequency (TF)**: Rewards passages mentioning query terms frequently.
    * **Inverse Document Frequency (IDF)**: Penalizes common words (e.g., 'the', 'is') and heavily weights rare distinguishing terms.
    * **Document Length Normalization**: Prevents lengthy documents from unfairly dominating matches simply due to high word counts.
    """)

st.divider()

# ==============================================================================
# SECTION 7: SEMANTIC RETRIEVAL
# ==============================================================================

st.header("7. Semantic Retrieval", divider="rainbow")
st.markdown("""
**Semantic Retrieval** uses dense neural embeddings to match queries with documents based on **underlying conceptual meaning** 
rather than literal token overlap.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Core Capabilities")
    st.markdown("""
    * **Synonym and Paraphrase Matching**:
      A query for `"automobile repair cost"` effortlessly matches documents stating `"pricing for fixing passenger vehicles"`.
    * **Cross-Lingual Retrieval**:
      Multilingual models (e.g., Cohere Multilingual v3) map concepts across 100+ languages to identical vector neighborhoods.
    * **Intent Generalization**:
      Captures the high-level intent behind exploratory questions without requiring users to guess exact document keywords.
    """)

with col2:
    st.subheader("Semantic vs. Keyword Comparison")
    sk_df = pd.DataFrame([
        {"Query Type", "Keyword (BM25)", "Semantic (Dense Vector)"},
        {"Conceptual ('how to speed up database')", "Weak (Misses synonyms like 'accelerate query')", "Strong (Matches latency optimization concepts)"},
        {"Exact SKU ('SKU-1082-A')", "Strong (Instant exact match)", "Weak (Embeds to generic product vector)"},
        {"Typo in Query ('postgreSql indexx')", "Weak (Tokens fail to match inverted list)", "Strong (Embedding robust to character noise)"},
        {"Rare Acronym ('ITAR compliance')", "Strong (Matches exact acronym)", "Weak (May lack representation in training data)"}
    ])
    st.dataframe(sk_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("8. Production Implementation: Multi-Strategy Retriever with MMR and Threshold Gating", divider="rainbow")
st.markdown("""
Below is an enterprise-grade Retriever implementing pure NumPy Maximum Marginal Relevance (MMR), 
configurable similarity thresholds, metadata filtering, and candidate hydration.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Optional

class MultiStrategyRetriever:
    """
    Enterprise retriever supporting Similarity Search, Score Thresholding,
    and Maximum Marginal Relevance (MMR) diversity optimization.
    """
    def __init__(
        self,
        dimension: int = 4,
        search_type: str = "mmr",
        k: int = 3,
        fetch_k: int = 20,
        score_threshold: float = 0.5,
        lambda_mult: float = 0.5
    ):
        self.dimension = dimension
        self.search_type = search_type
        self.k = k
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.lambda_mult = lambda_mult
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}

    def insert(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        self.vectors[doc_id] = v / norm if norm > 0 else v
        self.payloads[doc_id] = {"text": text, **metadata}

    def retrieve(
        self,
        query_vector: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        qv = qv / q_norm if q_norm > 0 else qv

        # 1. Filter candidates by metadata and compute query similarity
        candidate_pool = []
        for d_id, vec in self.vectors.items():
            payload = self.payloads[d_id]
            if filter_metadata:
                if not all(payload.get(k) == v for k, v in filter_metadata.items()):
                    continue
            sim = float(np.dot(qv, vec))
            if self.search_type == "similarity_score_threshold" and sim < self.score_threshold:
                continue
            candidate_pool.append((d_id, sim, vec))

        # Sort by similarity descending
        candidate_pool.sort(key=lambda x: x[1], reverse=True)

        if self.search_type == "similarity" or self.search_type == "similarity_score_threshold":
            top_hits = candidate_pool[:self.k]
            return [
                {"id": d_id, "score": round(sim, 4), "text": self.payloads[d_id]["text"]}
                for d_id, sim, _ in top_hits
            ]

        # 2. MMR Diversity Algorithm
        coarse_pool = candidate_pool[:self.fetch_k]
        selected_ids: List[str] = []
        selected_vecs: List[np.ndarray] = []
        results = []

        while len(results) < min(self.k, len(coarse_pool)):
            best_mmr = -1e9
            best_choice = None

            for d_id, sim_q, d_vec in coarse_pool:
                if d_id in selected_ids:
                    continue

                max_sim_selected = 0.0
                if selected_vecs:
                    max_sim_selected = max(float(np.dot(d_vec, sv)) for sv in selected_vecs)

                mmr_val = (self.lambda_mult * sim_q) - ((1.0 - self.lambda_mult) * max_sim_selected)

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_choice = (d_id, sim_q, d_vec)

            if best_choice is None:
                break

            c_id, c_sim, c_vec = best_choice
            selected_ids.append(c_id)
            selected_vecs.append(c_vec)
            results.append({
                "id": c_id,
                "score": round(c_sim, 4),
                "mmr_val": round(best_mmr, 4),
                "text": self.payloads[c_id]["text"]
            })

        return results
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("9. Interactive Workbench: MMR Lambda Diversity & Threshold Tuning Console", divider="rainbow")
st.markdown("""
Test how Maximum Marginal Relevance (MMR) balances relevance and diversity. 
Adjust Lambda (lambda) from 1.0 (pure similarity) down to 0.1 (maximum diversity) to observe how redundant passages are replaced with novel information.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("MMR Hyperparameters")
    lambda_val = st.slider("Lambda Multiplier (λ: 1.0 = Pure Sim, 0.0 = Pure Diversity)", 0.0, 1.0, 0.5, step=0.05)
    k_val = st.slider("Target Document Count (K)", 1, 5, 3)
    thresh_val = st.slider("Minimum Relevance Threshold", 0.0, 1.0, 0.20, step=0.05)
    cat_filter = st.selectbox("Metadata Category Filter", ["All", "Algorithms", "Databases", "Retrieval"])
    cat_arg = None if cat_filter == "All" else cat_filter

with col2:
    st.subheader("MMR Selection Output")
    # Query vector heavily focused on HNSW / Indexing
    q_vec_test = [0.90, 0.22, 0.10, 0.05]
    mmr_res = RetrievalStrategySimulator.compute_mmr(
        query_vector=q_vec_test,
        lambda_mult=lambda_val,
        top_k=k_val,
        threshold=thresh_val,
        category_filter=cat_arg
    )

    if mmr_res:
        st.dataframe(pd.DataFrame(mmr_res), use_container_width=True, hide_index=True)
        st.success(f"Retrieved {len(mmr_res)} documents under MMR Lambda={lambda_val}.")
    else:
        st.warning("No documents matched the criteria!")

    st.markdown("""
    **Observation**:
    * At **lambda = 1.0**, doc_1 and doc_2 (both describing HNSW graph construction) both enter the top results despite being semantically redundant.
    * At **lambda = 0.5**, MMR penalizes doc_2's similarity to doc_1, promoting diverse documents (e.g., Product Quantization or Single-Stage Filtered Search).
    """)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: What is the primary limitation of pure similarity search that Maximum Marginal Relevance (MMR) solves?"):
    st.markdown("""
    **Answer**:
    Pure similarity search suffers from **semantic redundancy**. If a long document contains five paragraphs explaining the same concept 
    with slightly different phrasing, all five paragraphs will receive high similarity scores and fill the entire Top-K context window. 
    This deprives the downstream LLM of diverse perspectives and alternative facts. 
    MMR solves this by iteratively balancing query relevance with novelty, penalizing candidates that are overly similar to already selected documents.
    """)

with st.expander("Q2: How does the Lambda parameter in MMR dictate the trade-off between relevance and diversity?"):
    st.markdown("""
    **Answer**:
    In MMR, the scoring formula is: `Score = lambda * Sim(Doc, Query) - (1 - lambda) * Max_Sim(Doc, Selected)`.
    * When lambda = 1.0, the diversity penalty is completely disabled, behaving identically to vanilla cosine similarity search.
    * When lambda = 0.0, relevance is ignored, and the algorithm greedily maximizes diversity relative to previously selected documents.
    * In production, setting lambda between 0.5 and 0.7 provides an optimal balance, ensuring retrieved context remains relevant to the query while introducing diverse informational coverage.
    """)

with st.expander("Q3: When should an enterprise RAG system choose keyword search (BM25) over semantic dense vector search?"):
    st.markdown("""
    **Answer**:
    Keyword search (BM25) should be prioritized when:
    1. Queries target exact alphanumeric tokens, such as error codes (`ERR_404_TIMEOUT`), hardware SKUs, or customer IDs.
    2. The domain involves rare specialized terminology (legal statute numbers, chemical compound names) not well-represented in general neural embedding training sets.
    3. Low computational latency is paramount, as BM25 inverted index lookups operate in under 1 ms with zero GPU embedding inference overhead.
    """)

with st.expander("Q4: How does dynamic similarity thresholding differ from fixed Top-K retrieval?"):
    st.markdown("""
    **Answer**:
    Fixed Top-K retrieval unconditionally returns K documents regardless of whether their relevance scores are 0.95 or 0.12. 
    Dynamic thresholding acts as an automated quality gate: it returns a variable number of documents, including zero documents 
    when no relevant information exists in the database. This allows the system to trigger safe fallback behaviors rather than forcing the LLM to hallucinate on irrelevant context.
    """)

with st.expander("Q5: What is the computational cost of running MMR compared to standard similarity search?"):
    st.markdown("""
    **Answer**:
    Standard similarity search requires finding the top K vectors from an index. 
    MMR requires first retrieving a broader candidate pool of `fetch_k` vectors (typically 20 to 50), 
    and then performing O(K * fetch_k) pairwise dot products in Python memory to compute the diversity penalty matrix. 
    While slightly more CPU-intensive than raw similarity search, this overhead is typically negligible (< 2 ms) on pools of 20-50 candidates.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Setting MMR Lambda Too Low (e.g., Lambda < 0.2) in Factoid Q&A**  
    Setting lambda too low prioritizes diversity over relevance, causing the retriever to pull completely unrelated documents 
    just because they are geometrically distant from the first hit.
    """)
    st.error("""
    **Trap 2: Using Similarity Thresholds Without Cross-Model Calibration**  
    A threshold of 0.75 on OpenAI `text-embedding-3-small` cannot be transferred to HuggingFace BGE or Cohere models. 
    Thresholds must be calibrated against an empirical test set whenever the embedding model changes.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Setting MMR `fetch_k` Equal to `k`**  
    Setting `fetch_k = k` gives MMR zero candidate pool from which to select diverse alternatives, 
    completely neutralizing the algorithm and producing identical results to vanilla similarity search.
    """)
    st.warning("""
    **Trap 4: Relying Exclusively on Vector Search for Technical Documentation**  
    Dense vectors often blur differences between related code functions (e.g., `create_user` vs `update_user`). 
    Always pair semantic search with BM25 keyword matching for technical codebases.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: MMR Objective")
        st.caption("What does MMR balance?")
        st.markdown("**Back**: Maximizes `lambda * Sim(Doc, Query) - (1 - lambda) * Max_Sim(Doc, Selected)`. Prevents redundant repetitive chunks.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Optimal MMR Lambda")
        st.caption("What is the production standard?")
        st.markdown("**Back**: Lambda between 0.5 and 0.7 balances topical query relevance with informational novelty.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Score Threshold Benefit")
        st.caption("Why use dynamic thresholds?")
        st.markdown("**Back**: Automatically adjusts result set size and returns zero documents on out-of-domain queries, preventing hallucinations.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: BM25 vs. Dense Retrieval")
        st.caption("When does BM25 win?")
        st.markdown("**Back**: BM25 excels at exact keyword sequences, alphanumeric IDs, error codes, and rare technical jargon.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Retrieval Strategies:")

c1 = st.checkbox("I understand the mechanics and semantic redundancy vulnerability of pure similarity search.", value=True)
c2 = st.checkbox("I know how similarity score thresholds act as relevance gates to prevent RAG hallucinations.", value=True)
c3 = st.checkbox("I understand the Maximum Marginal Relevance (MMR) formula and how to tune the Lambda multiplier.", value=True)
c4 = st.checkbox("I can differentiate between pre-filtering, post-filtering, and single-stage metadata filtering.", value=True)
c5 = st.checkbox("I know when keyword retrieval (BM25) outperforms dense semantic vector retrieval.", value=True)
c6 = st.checkbox("I can implement a multi-strategy retriever supporting MMR diversity and threshold gating in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Retrieval Strategies and understand how to engineer diverse, non-redundant context pipelines.")
