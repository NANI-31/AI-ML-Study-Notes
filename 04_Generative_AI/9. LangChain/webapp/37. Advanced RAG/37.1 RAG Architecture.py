import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.1 RAG Architecture - LangChain Curriculum",
    page_icon="📚",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .rag-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-pass {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class RAGArchitectureEngine:
    """
    Simulates Naive, Advanced, Modular, Agentic, and Hybrid RAG pipelines,
    scoring precision, recall, latency, and context window efficiency.
    """
    def __init__(self):
        self.architectures = {
            "Basic / Naive RAG": {
                "stages": ["Fixed Chunking (500 tokens)", "Cosine Dense Retrieval (top-4)", "Direct Generation"],
                "precision": 0.62,
                "recall": 0.58,
                "latency_ms": 750,
                "hallucination_risk": "High (unfiltered retrieval)",
                "context_efficiency": "Low (contains chunk noise)"
            },
            "Advanced RAG": {
                "stages": ["Hierarchical Chunking", "HyDE / Query Expansion", "Dense Retrieval (top-20)", "Cross-Encoder Reranker (top-4)", "Contextual Compression"],
                "precision": 0.88,
                "recall": 0.84,
                "latency_ms": 1250,
                "hallucination_risk": "Low (reranked and compressed)",
                "context_efficiency": "High (compact relevant snippets)"
            },
            "Modular RAG": {
                "stages": ["Routing Classifier", "Parallel Specialized Stores", "Reciprocal Rank Fusion", "Dynamic Extraction"],
                "precision": 0.91,
                "recall": 0.89,
                "latency_ms": 1400,
                "hallucination_risk": "Very Low (domain-specific routing)",
                "context_efficiency": "Very High (adaptive extraction)"
            },
            "Agentic RAG": {
                "stages": ["Query Intent Analysis", "Conditional Retrieval", "Document Grading", "Self-Correction Loop", "Iterative Refinement"],
                "precision": 0.96,
                "recall": 0.94,
                "latency_ms": 2800,
                "hallucination_risk": "Near Zero (graded and verified)",
                "context_efficiency": "Optimal (self-correcting synthesis)"
            },
            "Hybrid RAG": {
                "stages": ["Sparse Lexical BM25 (top-20)", "Dense Semantic Vector (top-20)", "Reciprocal Rank Fusion (RRF)", "Cross-Encoder Filter"],
                "precision": 0.93,
                "recall": 0.95,
                "latency_ms": 1100,
                "hallucination_risk": "Low (best lexical and semantic match)",
                "context_efficiency": "High (dual-index coverage)"
            }
        }

    def simulate_rrf_scoring(self, bm25_rank: int, dense_rank: int, k: int = 60) -> float:
        """Calculates Reciprocal Rank Fusion score: 1/(k + rank_bm25) + 1/(k + rank_dense)."""
        score = (1.0 / (k + bm25_rank)) + (1.0 / (k + dense_rank))
        return round(score, 5)

# Instantiate engine
rag_engine = RAGArchitectureEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Evolution of Retrieval-Augmented Generation Paradigms", divider="rainbow")

st.markdown("""
Retrieval-Augmented Generation (RAG) has transformed from a simple vector lookup mechanism into a sophisticated, 
multi-tiered orchestration discipline. As enterprise production demands rigorous factual groundedness, minimal token costs, 
and low latency, modern architectures have evolved across five distinct generations: **Basic RAG**, **Advanced RAG**, 
**Modular RAG**, **Agentic RAG**, and **Hybrid RAG**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAG_Evolution_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. Query Ingress & Adaptive Transformation Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            user_query [label="Incoming User Query\\n(Ambiguous or multi-part prompt)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_transform [label="Query Transformation\\n(Rewriting, HyDE, or Sub-query decomposition)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_retrieval {
            label = "2. Hybrid Dual-Index Retrieval Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            sparse_index [label="Sparse Index (BM25)\\n(Exact keywords, acronyms, product IDs)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            dense_index [label="Dense Index (HNSW / VectorDB)\\n(Semantic meaning, conceptual synonyms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            rrf_fusion [label="Reciprocal Rank Fusion (RRF)\\n(Rank aggregation score = sum(1/(60 + rank)))", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_post_retrieval {
            label = "3. Post-Retrieval Refinement & Context Pruning";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            reranker [label="Cross-Encoder Reranker\\n(Joint attention over Query + Chunk pairs)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            compressor [label="Contextual Compression & Parent-Child\\n(Extracts precise sentences, recovers parent context)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_agentic_eval {
            label = "4. Agentic Verification & Synthesis Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            doc_eval [label="Document Relevance Evaluator\\n(LLM grader: Are retrieved chunks sufficient?)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            generator [label="Frontier LLM Synthesizer\\n(Produces grounded, citation-backed response)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            feedback_loop [label="Self-Correction Trigger\\n(If docs irrelevant -> rewrite query & retry)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
        }

        user_query -> query_transform;
        query_transform -> sparse_index [label="keyword tokens"];
        query_transform -> dense_index [label="dense embedding vector"];

        sparse_index -> rrf_fusion [label="top-20 sparse"];
        dense_index -> rrf_fusion [label="top-20 dense"];

        rrf_fusion -> reranker [label="fused candidate pool"];
        reranker -> compressor [label="top-5 reranked"];
        compressor -> doc_eval [label="pruned context"];

        doc_eval -> generator [label="grade = RELEVANT"];
        doc_eval -> feedback_loop [label="grade = IRRELEVANT"];
        feedback_loop -> query_transform [label="reformulated search"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Architectural Insight:</strong> Basic RAG assumes that similarity in vector space equates to factual relevance. 
    In production enterprise systems, this assumption fails due to vocabulary mismatch, outdated embeddings, and noisy chunks. 
    Modern RAG combines dual-index hybrid search, cross-encoder reranking, and agentic reflection loops to guarantee factual groundedness.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: BASIC RAG
# ==============================================================================
st.header("2. Basic RAG: Naive Vector Search, Fixed Chunking, and Single-Pass Synthesis", divider="rainbow")

st.markdown("""
**Basic (Naive) RAG** represents the first-generation approach popularized in early LLM tutorials:
1. **Ingestion**: Raw documents are split into uniform character or token chunks (e.g., 500 characters with 50-character overlap) without regard for document structure.
2. **Embedding**: Each chunk is converted into a dense vector via an embedding model and stored in an in-memory or cloud vector store.
3. **Retrieval**: The raw user question is embedded, and a K-Nearest Neighbors (KNN) cosine similarity search retrieves the top-K chunks (typically K = 3 to 5).
4. **Generation**: The retrieved chunks are concatenated into a prompt template alongside the user question, and an LLM produces a single-pass answer.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Structural Failure Modes of Basic RAG
    - **Context Window Flooding**: Irrelevant sentences in fixed-size chunks dilute the prompt, causing LLM attention drift.
    - **Lost in the Middle**: LLMs struggle to recall critical details placed in the middle of long concatenated contexts.
    - **Keyword Blindness**: Acronyms, product serial numbers, and exact error codes often have poor dense vector cosine proximity.
    """)

with col2:
    st.markdown("""
    ### Performance Profile
    - **Precision**: Low (0.55 to 0.65)
    - **Recall**: Moderate (0.50 to 0.60)
    - **Latency**: Fast (500ms to 800ms)
    - **Cost**: Low compute overhead, but high downstream token costs due to noisy context.
    """)

# ==============================================================================
# SECTION 3: ADVANCED RAG
# ==============================================================================
st.header("3. Advanced RAG: Pre-Retrieval Optimization, Post-Retrieval Reranking, and Context Pruning", divider="rainbow")

st.markdown("""
**Advanced RAG** introduces targeted pre-retrieval and post-retrieval processing stages to overcome the limitations of naive vector lookup:
""")

st.markdown("""
### 1. Pre-Retrieval Optimization
- **Query Rewriting**: Expanding user acronyms, stripping conversational conversational filler, and resolving multi-turn conversational coreferences.
- **HyDE (Hypothetical Document Embeddings)**: Prompting an LLM to hallucinate a hypothetical ideal answer first, then embedding that answer to retrieve actual documents that share semantic structure.

### 2. Post-Retrieval Optimization
- **Cross-Encoder Reranking**: Vector search provides high-recall candidate generation, but cross-encoders perform full bidirectional attention between `(Query, Document)` pairs, reordering candidates with pinpoint precision.
- **Contextual Compression**: Extracting only the specific 2-3 sentences answering the prompt from retrieved chunks, dropping irrelevant filler text before LLM synthesis.
""")

# ==============================================================================
# SECTION 4: MODULAR RAG
# ==============================================================================
st.header("4. Modular RAG: Composable Retrieval Services, Dynamic Routing, and Pluggable Operators", divider="rainbow")

st.markdown("""
**Modular RAG** decouples retrieval into independent, reusable microservices and operators. 
Rather than enforcing a fixed linear pipeline, an orchestrator dynamically configures components based on query metadata:
""")

st.markdown("""
- **Dynamic Routing**: An intent classifier routes code queries to GitHub repositories, policy questions to vector search, and transactional balance queries directly to SQL databases.
- **Pluggable Retrievers**: Systems can plug in GraphRAG for complex entity relationships, Web Search for real-time temporal queries, and dense Vector Stores for conceptual manuals.
- **Iterative Interleaving**: Modules can be composed into arbitrary Directed Acyclic Graphs (DAGs) where retrieval steps feed extraction and summarization modules before final output generation.
""")

# ==============================================================================
# SECTION 5: AGENTIC RAG
# ==============================================================================
st.header("5. Agentic RAG: Autonomous Decision Loops, Self-Correction, and Multi-Step Tool Orchestration", divider="rainbow")

st.markdown("""
**Agentic RAG** transforms RAG from a passive query-response pipeline into an active **autonomous reasoning agent**. 
Using cyclic state machines (such as LangGraph), the model controls the retrieval lifecycle:
""")

st.markdown("""
1. **Autonomous Triggering**: The agent decides *whether* retrieval is even necessary. If the user asks a simple greeting or general logic puzzle, it bypasses retrieval entirely.
2. **Iterative Multi-Hop Search**: For complex comparative questions ("Compare the revenue growth of Apple and Microsoft in Q3 2024"), the agent executes sequential retrieval passes, gathering facts incrementally.
3. **Retrieval Validation & Grading**: A dedicated evaluation node evaluates retrieved documents. If the retrieved texts lack sufficient answers, the agent triggers query reformulation and searches alternate sources.
4. **Self-Correction & Hallucination Suppression**: Before showing the answer to the user, an evaluator compares the generated output against retrieved passages to ensure 100% factual groundedness.
""")

# ==============================================================================
# SECTION 6: HYBRID RAG
# ==============================================================================
st.header("6. Hybrid RAG: Fusing Dense Semantic Embeddings and Sparse Lexical BM25 via Reciprocal Rank Fusion", divider="rainbow")

st.markdown("""
Neither dense semantic search nor sparse keyword search is sufficient on its own. **Hybrid RAG** combines both to achieve optimal recall and precision:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### The Hybrid Advantage
    - **Sparse Search (BM25)**: Exceptional at exact keyword matching, technical identifiers (`CVE-2024-38077`), part numbers, and rare domain acronyms.
    - **Dense Search (Embeddings)**: Exceptional at capturing semantic synonyms, paraphrase variation, and conceptual themes.
    """)

with col2:
    st.markdown("""
    ### Reciprocal Rank Fusion (RRF)
    RRF combines ranked lists from both retrievers without requiring score normalization:
    """)
    st.code('''def compute_rrf_score(rank_bm25: int, rank_dense: int, k: int = 60) -> float:
    # rank is 1-indexed position in respective search results
    return (1.0 / (k + rank_bm25)) + (1.0 / (k + rank_dense))
''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Stage RAG Pipeline with Hybrid Search and Cross-Encoder Reranking", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**Hybrid RAG engine with BM25, dense semantic mock scoring, Reciprocal Rank Fusion, and cross-encoder reranking**.
""")

st.code('''from typing import List, Dict, Any

class ProductionHybridRAGEngine:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulates sparse lexical keyword retrieval."""
        query_tokens = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            doc_tokens = set(doc["text"].lower().split())
            overlap = len(query_tokens.intersection(doc_tokens))
            scored.append({"id": doc["id"], "text": doc["text"], "sparse_score": overlap})
        scored.sort(key=lambda x: x["sparse_score"], reverse=True)
        return scored[:top_k]

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulates dense embedding vector cosine search."""
        # Simulated semantic score
        scored = []
        for idx, doc in enumerate(self.corpus):
            sim_score = 0.5 + (0.4 / (idx + 1))
            scored.append({"id": doc["id"], "text": doc["text"], "dense_score": round(sim_score, 4)})
        scored.sort(key=lambda x: x["dense_score"], reverse=True)
        return scored[:top_k]

    def reciprocal_rank_fusion(self, sparse_results: List[dict], dense_results: List[dict], k: int = 60) -> List[dict]:
        """Merges sparse and dense results using Reciprocal Rank Fusion."""
        rrf_scores = {}
        doc_map = {}

        # Process Sparse Ranks
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc["text"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Process Dense Ranks
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc["text"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Sort aggregated scores
        fused = [
            {"id": doc_id, "text": doc_map[doc_id], "rrf_score": round(score, 5)}
            for doc_id, score in rrf_scores.items()
        ]
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def cross_encoder_rerank(self, query: str, candidates: List[dict], top_n: int = 2) -> List[dict]:
        """Simulates cross-encoder high-precision reranking."""
        for c in candidates:
            # Score bonus if query keywords are in text
            bonus = 0.3 if any(w in c["text"].lower() for w in query.lower().split()) else 0.0
            c["rerank_score"] = round(c["rrf_score"] * 10 + bonus, 4)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

# Test Execution
corpus_data = [
    {"id": "doc_1", "text": "LangChain v0.3 introduces native support for stream_mode updates and values."},
    {"id": "doc_2", "text": "Kubernetes cluster pod scaling policies and deployment configurations."},
    {"id": "doc_3", "text": "Hybrid RAG combines BM25 lexical search with dense vector embeddings via RRF."}
]

rag_system = ProductionHybridRAGEngine(corpus_data)
s_res = rag_system.bm25_search("Hybrid RAG BM25")
d_res = rag_system.dense_search("Hybrid RAG BM25")
fused_res = rag_system.reciprocal_rank_fusion(s_res, d_res)
final_top = rag_system.cross_encoder_rerank("Hybrid RAG BM25", fused_res)

print("Top Reranked Context:", final_top)
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: RAG Architecture Comparison Matrix & Hybrid Search Simulator", divider="rainbow")

st.markdown("""
Compare the trade-offs across all 5 RAG architectures and test 
how Reciprocal Rank Fusion (RRF) calculates joint rankings between sparse and dense retrievers.
""")

wb_view = st.selectbox(
    "Select Workbench Mode:",
    ["1. RAG Architecture Evaluation Matrix", "2. Reciprocal Rank Fusion (RRF) Calculator", "3. Multi-Stage Pipeline Latency Simulator"]
)

if wb_view == "1. RAG Architecture Evaluation Matrix":
    st.subheader("Architectural Paradigm Trade-Off Comparison")
    
    table_rows = []
    for arch_name, stats in rag_engine.architectures.items():
        table_rows.append({
            "Architecture": arch_name,
            "Precision": f"{stats['precision'] * 100:.0f}%",
            "Recall": f"{stats['recall'] * 100:.0f}%",
            "p95 Latency": f"{stats['latency_ms']} ms",
            "Hallucination Risk": stats["hallucination_risk"],
            "Context Efficiency": stats["context_efficiency"]
        })
    df_compare = pd.DataFrame(table_rows)
    st.dataframe(df_compare, use_container_width=True)

elif wb_view == "2. Reciprocal Rank Fusion (RRF) Calculator":
    st.subheader("Interactive Reciprocal Rank Fusion Calculator")
    st.caption("Score formula: RRF = 1 / (60 + BM25_Rank) + 1 / (60 + Dense_Rank)")

    col_a, col_b = st.columns(2)
    with col_a:
        bm25_pos = st.slider("BM25 Keyword Rank Position:", 1, 50, 1)
    with col_b:
        dense_pos = st.slider("Dense Semantic Vector Rank Position:", 1, 50, 4)

    rrf_val = rag_engine.simulate_rrf_scoring(bm25_pos, dense_pos)
    st.metric("Combined RRF Fusion Score", f"{rrf_val:.5f}")

    if bm25_pos <= 3 and dense_pos <= 3:
        st.success("Strong Consensus: Both sparse and dense retrievers agree this document is top-tier!")
    elif bm25_pos > 30 and dense_pos > 30:
        st.error("Low Consensus: Document is ranked low by both retrieval algorithms.")
    else:
        st.info("Cross-Disciplinary Match: Document rescued by one strong retriever despite lower rank in the other.")

else:
    st.subheader("Pipeline Stage Latency Breakdown")
    arch_choice = st.selectbox("Select Architecture to Profile:", list(rag_engine.architectures.keys()))
    arch_info = rag_engine.architectures[arch_choice]

    st.markdown(f"**Target Architecture:** `{arch_choice}`")
    st.markdown(f"**Total End-to-End Latency:** `{arch_info['latency_ms']} ms`")
    st.markdown("#### Sequential Pipeline Stages:")
    for step_idx, step_name in enumerate(arch_info["stages"], 1):
        st.markdown(f"{step_idx}. `{step_name}`")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why does Naive RAG suffer from the 'Lost in the Middle' problem?"):
    st.markdown("""
    **Answer:**  
    Research by Liu et al. demonstrated that transformer attention mechanisms preferentially weight tokens at the extreme 
    beginning and end of long input sequences. In Naive RAG, retrieving 5-10 fixed chunks and concatenating them creates a 
    long context where crucial evidence buried in chunk #4 or #5 receives lower attention weight, leading to hallucination 
    or omissions.
    """)

with st.expander("Q2: What is Reciprocal Rank Fusion (RRF), and why is it preferred over raw score linear combination?"):
    st.markdown("""
    **Answer:**  
    BM25 produces unbounded lexical scores based on term frequency and document length, while dense embedding models produce 
    cosine similarities bounded between -1.0 and 1.0 (or normalized dot products). Combining these raw scores linearly 
    requires fragile, corpus-dependent calibration. RRF bypasses score calibration by operating strictly on **ordinal rank positions** 
    `1 / (k + rank)`, providing robust, parameter-insensitive rank fusion.
    """)

with st.expander("Q3: When should an enterprise architecture transition from Advanced RAG to Agentic RAG?"):
    st.markdown("""
    **Answer:**  
    Advanced RAG is ideal for single-turn, factual lookup tasks where a single retrieval pass provides sufficient information. 
    Agentic RAG becomes necessary when queries require:
    1. Multi-hop comparative analysis across disparate documents.
    2. Document relevance verification and automatic retry on missing facts.
    3. Tool execution (SQL queries, APIs) interleaved with textual document synthesis.
    """)

with st.expander("Q4: How does Cross-Encoder Reranking differ architecturally from Bi-Encoder embedding search?"):
    st.markdown("""
    **Answer:**  
    - **Bi-Encoders** embed the query and document independently into fixed-dimensional vectors, enabling fast approximate nearest neighbor (ANN) search via dot products. However, they cannot capture word-level cross-attentions.
    - **Cross-Encoders** concatenate the query and candidate chunk into a single sequence `[CLS] Query [SEP] Chunk [SEP]` and feed it through a transformer. Full self-attention across every query and document token enables far higher precision at the cost of higher latency, making it suitable only as a second-stage reranker.
    """)

with st.expander("Q5: What role does Contextual Compression play in mitigating downstream LLM generation costs?"):
    st.markdown("""
    **Answer:**  
    A retrieved 500-token chunk often contains only 40 tokens of actual relevant information. Concatenating 5 such chunks passes 
    2,500 tokens to the generation LLM. Contextual compression uses small extraction models to extract only the relevant 
    sentences, reducing prompt token volume by 70% to 80%, directly cutting latency and cost.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Relying Solely on Dense Vector Search for Enterprise Technical Docs**  
Dense embeddings frequently fail on specific error codes (`ERR_CONNECTION_TIMED_OUT`), product SKUs, or function names (`stream_mode='updates'`). 
Always deploy a Hybrid Search (BM25 + Dense) configuration for technical documentation.
""")

st.error("""
**Trap 2: Passing Entire 1000-Token Chunks Unreranked to Generation LLMs**  
Retrieving 10 uncompressed chunks injects massive filler noise into the prompt, triggering prompt attention dilution and 
inflating monthly API spend. Always apply a reranking and compression step.
""")

st.warning("""
**Trap 3: Normalizing BM25 Scores with Fixed Min-Max Scaling**  
BM25 scores vary wildly depending on query length and term uniqueness. Attempting to linearly blend normalized BM25 with cosine similarity 
breaks whenever query length changes. Use Reciprocal Rank Fusion instead.
""")

st.warning("""
**Trap 4: Deploying Agentic Multi-Hop Loops Without Recursion Limits**  
An agent tasked with self-correcting retrieval can loop indefinitely if the required information does not exist in the corpus. 
Always enforce a maximum recursion depth (e.g., 3 loops).
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Reciprocal Rank Fusion (RRF)</h4>
        <p>A rank aggregation algorithm scoring items as sum(1 / (k + rank)) across multiple disparate search engines without score calibration.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Cross-Encoder Reranker</h4>
        <p>A transformer model applying bidirectional attention over query-document pairs to produce high-precision relevance scores.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Contextual Compression</h4>
        <p>A post-retrieval stage that prunes irrelevant sentences from retrieved chunks before injecting them into the generation prompt.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Agentic RAG</h4>
        <p>A cyclic retrieval paradigm where an autonomous agent plans, retrieves, grades, and self-corrects context dynamically.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architectural evolution from Basic RAG to Advanced, Modular, Agentic, and Hybrid RAG.", value=True)
chk_2 = st.checkbox("I know the failure modes of Basic RAG (context dilution, Lost in the Middle, keyword blindness).", value=True)
chk_3 = st.checkbox("I can explain how Cross-Encoder Rerankers differ from Bi-Encoder vector retrievers.", value=True)
chk_4 = st.checkbox("I know how Reciprocal Rank Fusion (RRF) combines BM25 and dense embeddings without score calibration.", value=True)
chk_5 = st.checkbox("I understand how Agentic RAG implements self-correction and document grading loops.", value=True)
chk_6 = st.checkbox("I know how contextual compression reduces prompt token count while preserving answer recall.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("RAG Architecture competencies verified! Ready for Advanced Retrieval.")
