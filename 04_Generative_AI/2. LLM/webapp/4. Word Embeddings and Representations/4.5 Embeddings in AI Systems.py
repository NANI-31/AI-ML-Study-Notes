import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.5 Embeddings in AI Systems (Semantic Search, RecSys, RAG, Deduplication, Clustering, Knowledge Retrieval)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .takeaway-box {
        background-color: #0B132B;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #1C1917;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .modern-callout {
        background-color: #042F2E;
        border-left: 4px solid #14B8A6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-title">4.5 Embeddings in AI Systems: Semantic Search, RecSys, RAG, and Knowledge Retrieval</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Theoretical Hyperspheres to Production Enterprise Architectures — How Vector Representations Power Modern Search Engines, Recommenders, and Foundation Model RAG.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Dense Vector Topology Across Production AI Stacks", divider="rainbow")

st.markdown("""
In enterprise production environments, vector embeddings serve as the universal lingua franca connecting 
unstructured text, user behavior logs, knowledge graphs, and generative foundation models.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Ingestion & Encoders
    arch_graph.node("CORPUS", "Enterprise Raw Data\nDocs, FAQs, Catalog, Ticket Logs", shape="cylinder", fillcolor="#1E293B", color="#38BDF8")
    arch_graph.node("ENCODER", "Bi-Encoder Model\n(BGE-M3 / OpenAI text-embedding-3)\nOutput: Vectors in R^D (L2 normalized)", shape="box", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("VDB", "Production Vector Database\n(Qdrant / Milvus / Pinecone / pgvector)\nHNSW Index + Metadata Filtering", shape="cylinder", fillcolor="#042F2E", color="#14B8A6")

    with arch_graph.subgraph(name="cluster_apps") as c:
        c.attr(bgcolor="#0F172A", label="Production Enterprise Workloads", fontcolor="#38BDF8", color="#38BDF8")
        c.node("SEARCH", "Semantic & Hybrid Search\nDense + BM25 Lexical (RRF Fusion)", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("RECSYS", "Recommendation Systems\nTwo-Tower User & Item Matching", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("RAG", "Retrieval-Augmented Gen (RAG)\nTop-K Chunk Injection into LLM Context", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("DEDUP", "Deduplication & Clustering\nCosine Thresholding & Topic Partitioning", shape="box", fillcolor="#1E293B", color="#38BDF8")

    # Edges
    arch_graph.edge("CORPUS", "ENCODER")
    arch_graph.edge("ENCODER", "VDB", label="Batch Ingestion")
    arch_graph.edge("VDB", "SEARCH", label="k-NN Query")
    arch_graph.edge("VDB", "RECSYS", label="Candidate Gen")
    arch_graph.edge("VDB", "RAG", label="Context Retrieval")
    arch_graph.edge("VDB", "DEDUP", label="Density Grouping")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### The Six Pillars of Enterprise Embedding Applications
    1. **Semantic Search**: Resolves vocabulary mismatch by matching user queries to documents based on semantic meaning rather than exact keywords.
    2. **Recommendation Systems**: Two-Tower architectures scale personalization to hundreds of millions of items with sub-10ms candidate retrieval.
    3. **Retrieval-Augmented Generation (RAG)**: Mitigates hallucinations and overcomes static knowledge cutoffs by grounding generative LLMs with relevant external passages.
    4. **Duplicate Detection**: Identifies near-duplicate tickets, scraped documents, or customer reviews using calibrated cosine distance thresholds.
    5. **Document Clustering**: Unsupervised topic discovery across large corpuses for market intelligence, taxonomy generation, and pretraining data curation.
    6. **Knowledge Retrieval**: Unifies structured relational databases, entity graphs, and unstructured prose into a coherent search layer.
    """)

# ==============================================================================
# SECTION 2: 1. SEMANTIC SEARCH
# ==============================================================================
st.header("2. Semantic Search: Beyond BM25 Keyword Matching", divider="rainbow")

st.markdown("""
Classical information retrieval relied on lexical frequency metrics like **TF-IDF** and **BM25**. 
While effective for exact keyword matches, BM25 fails completely when queries and documents use synonyms 
(the **Vocabulary Mismatch Problem**).
""")

search_col1, search_col2 = st.columns([1, 1])

with search_col1:
    st.markdown("""
    ### The Vocabulary Mismatch Dilemma
    * **User Query**: `"automobile repair manual"`
    * **Indexed Passage**: `"Official service handbook for passenger car maintenance and motor overhaul."`
    * **BM25 Score**: **0.00** (Zero shared lexical terms; completely missed).
    * **Dense Semantic Search**: **0.88 Cosine Match** (The bi-encoder maps `"automobile"` and `"car"`, `"repair"` and `"maintenance"`, `"manual"` and `"handbook"` to adjacent coordinates).

    ### The Modern Standard: Hybrid Search with Reciprocal Rank Fusion (RRF)
    Production enterprise systems never deploy dense retrieval alone. Dense search occasionally struggles with:
    * Exact part numbers, serial codes, SKU identifiers (`"SKU-8492-X"`).
    * Acronyms and rare proper nouns.

    **Hybrid Search** combines BM25 lexical ranking and Dense Vector ranking using **Reciprocal Rank Fusion (RRF)**:
    ```
    RRF_Score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
    ```
    Where $k \approx 60$ is a smoothing constant preventing top-ranked items from dominating.
    """)

with search_col2:
    st.markdown("""
    ### BM25 vs. Dense vs. Hybrid Search Comparison
    """)
    df_search_comp = pd.DataFrame({
        "Feature": ["Mechanism", "Synonym Handling", "Exact Match / SKU", "Index Latency", "Storage Footprint", "Cold-Start Handling"],
        "Sparse BM25": ["Inverted index term frequency", "Fails (Zero match)", "Exceptional (Exact token)", "Instantaneous (CPU)", "Low (RAM / disk)", "Excellent (No training)"],
        "Dense Embedding": ["Bi-Encoder neural projection", "Exceptional (Latent semantics)", "Poor (Blurs unique IDs)", "Requires GPU inference", "High (Vector DB RAM)", "Requires domain training"],
        "Hybrid RRF": ["Linear / Reciprocal Rank Fusion", "Exceptional (Via dense)", "Exceptional (Via BM25)", "Composite pipeline", "Medium-High", "Balanced"]
    })
    st.dataframe(df_search_comp, use_container_width=True, hide_index=True)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Modern retrieval architectures (Cohere Rerank, BGE-M3) unify sparse lexical token matching and dense semantic vectors inside a single multi-stage pipeline, ensuring both semantic understanding and exact keyword precision.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: 2. RECOMMENDATION SYSTEMS
# ==============================================================================
st.header("3. Recommendation Systems: Two-Tower Vector Architectures", divider="rainbow")

st.markdown("""
Modern streaming, e-commerce, and social media platforms (YouTube, Netflix, Amazon, Pinterest) 
scale candidate retrieval across billions of items using **Two-Tower Neural Networks**.
""")

rec_col1, rec_col2 = st.columns([1, 1])

with rec_col1:
    tt_graph = graphviz.Digraph(format="svg")
    tt_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    tt_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    tt_graph.attr("edge", color="#64748B", penwidth="1.2")

    # Towers
    with tt_graph.subgraph(name="cluster_user") as c:
        c.attr(bgcolor="#0F172A", label="User Tower (Query Tower)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("U_FEAT", "User Features\nAge, History, Location, Device", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("U_DNN", "Deep MLP / Transformer\nComputes User Vector u in R^D", shape="box", fillcolor="#312E81", color="#818CF8")

    with tt_graph.subgraph(name="cluster_item") as c:
        c.attr(bgcolor="#0F172A", label="Item Tower (Candidate Tower)", fontcolor="#10B981", color="#10B981")
        c.node("I_FEAT", "Item Features\nTitle, Category, Tags, Author", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("I_DNN", "Deep MLP / Transformer\nComputes Item Vector v in R^D", shape="box", fillcolor="#042F2E", color="#14B8A6")

    tt_graph.node("DOT_PROD", "Inner Product / Cosine Scoring\nScore(u, v) = u . v\nEvaluated via ANN (FAISS / ScaNN)", shape="ellipse", fillcolor="#1E293B", color="#F59E0B")

    tt_graph.edge("U_FEAT", "U_DNN")
    tt_graph.edge("I_FEAT", "I_DNN")
    tt_graph.edge("U_DNN", "DOT_PROD", label="Vector u")
    tt_graph.edge("I_DNN", "DOT_PROD", label="Vector v")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(tt_graph, use_container_width=True)

with rec_col2:
    st.markdown("""
    ### Why Two Towers Decouple Latency
    * **Asynchronous Item Precomputation**: The billion items in an e-commerce catalog do not change every millisecond. All item vectors $v_i$ are precomputed offline and indexed in an Approximate Nearest Neighbor (ANN) index.
    * **Real-Time User Inference**: At query time, only the **User Tower** runs inference once on the user's latest interaction session ($~2\text{ms}$).
    * **Sub-5ms Candidate Retrieval**: The user vector $u$ queries the ANN index, returning the top 500 candidates via inner product in milliseconds.
    * **Second-Stage Ranking**: A heavy cross-attention ranker or GBDT scores the top 500 candidates with full contextual features to output the final top 10 feed.
    """)

# ==============================================================================
# SECTION 4: 3. RETRIEVAL-AUGMENTED GENERATION (RAG)
# ==============================================================================
st.header("4. Retrieval-Augmented Generation (RAG): Grounding Generative LLMs", divider="rainbow")

st.markdown("""
**Retrieval-Augmented Generation (RAG)** combines the generative reasoning power of foundation LLMs 
with non-parametric external knowledge retrieved from dense vector indices.
""")

rag_col1, rag_col2 = st.columns([1, 1])

with rag_col1:
    st.markdown("""
    ### Why Pure LLMs Fail in Enterprise Environments
    1. **Knowledge Cutoffs**: LLM training weights cannot answer questions about internal company updates or real-time events.
    2. **Hallucination Risk**: When uncertain, generative models produce syntactically convincing but factually incorrect assertions.
    3. **Data Privacy & Access Control**: Fine-tuning proprietary company documents into model weights leaks sensitive data across employee permission tiers.

    ### The RAG Triad Architecture
    1. **Ingest & Chunk**: Clean documents, split into overlapping chunks (e.g., 500 tokens), generate dense embeddings, index in Vector DB.
    2. **Retrieve**: Embed user prompt, retrieve top-$K$ passages ($K=3 \text{ to } 7$) using bi-encoder + metadata filters (e.g., `user_department == 'Engineering'`).
    3. **Augment & Generate**: Format a grounded prompt:
    ```
    Context Information:
    [Passage 1]: ...
    [Passage 2]: ...
    
    User Query: ...
    Instruction: Answer the query based ONLY on the context above.
    ```
    """)

with rag_col2:
    st.markdown("""
    ### Key RAG Failure Modes and Solutions
    """)
    df_rag_traps = pd.DataFrame({
        "Failure Mode": ["Chunk Truncation", "Lost in the Middle", "Outdated Passage Retrieval", "Irrelevant Retrieval"],
        "Symptom": [
            "Facts split across chunk boundaries lose context",
            "LLM ignores relevant facts placed in the center of long contexts",
            "Vector DB retrieves superseded 2021 policy instead of 2024 update",
            "Retrieved chunks match query keywords but lack answer"
        ],
        "Production Solution": [
            "Parent-Document Retrieval / Sentence-Window context expansion",
            "Re-rank top chunks; place highest scoring passages at top & bottom",
            "Hybrid metadata filtering on timestamp (`created_at > 2024-01-01`)",
            "Cross-Encoder reranking + Self-Correction / Corrective RAG (CRAG)"
        ]
    })
    st.dataframe(df_rag_traps, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 5: 4. DUPLICATE DETECTION
# ==============================================================================
st.header("5. Duplicate Detection: Near-Duplicate Text Identification & Thresholds", divider="rainbow")

st.markdown("""
Identifying identical or near-identical texts is critical for customer support ticket routing, 
scraped web corpus cleaning, and online review fraud detection.
""")

dup_col1, dup_col2 = st.columns([1, 1])

with dup_col1:
    st.markdown("""
    ### Exact vs. Near-Duplicate Matching
    * **Exact Duplicates**: Easily detected via cryptographic hash collisions (`SHA-256`, `MD5`) in $O(1)$ time. Fails if a single punctuation mark or whitespace differs.
    * **Lexical Near-Duplicates**: Detected via **MinHash / SimHash** with Jaccard similarity over n-gram sets. Fails when sentences use paraphrased vocabulary.
    * **Semantic Duplicates**: Detected via **Dense Embedding Cosine Proximity**. Recognizes that:
      * Query A: *"I forgot my account password and cannot log in."*
      * Query B: *"Unable to access my profile; login credentials not recognized."*
      Share 0 lexical words, but yield **0.93 Cosine Similarity**!
    """)

with dup_col2:
    st.markdown("""
    ### Establishing Calibrated Decision Thresholds
    For production ticket deduplication:
    * **Cosine > 0.94**: Auto-Merge Tickets. Route to the existing open agent thread automatically without human intervention.
    * **Cosine 0.85 - 0.94**: Suggest Related Ticket. Surface the candidate to the support agent as a recommended link.
    * **Cosine < 0.85**: Treat as Novel Ticket. Dispatch through standard queue triage.
    """)
    st.warning("Always conduct empirical precision-recall threshold sweeps on a labeled golden dataset before deploying auto-merge thresholds into production.")

# ==============================================================================
# SECTION 6: 5. DOCUMENT CLUSTERING
# ==============================================================================
st.header("6. Document Clustering: Topic Discovery and Corpus Partitioning", divider="rainbow")

st.markdown("""
Clustering embeddings reveals natural topic groupings across unlabeled text corpuses without requiring predefined taxonomies.
""")

clust_col1, clust_col2 = st.columns([1, 1])

with clust_col1:
    st.markdown("""
    ### The BERTopic Pipeline Standard
    The modern gold standard for embedding-based topic modeling is the **BERTopic architecture**:
    1. **Embed Documents**: Encode texts using high-quality sentence transformers (`all-MiniLM-L6-v2` or `BGE-Small`).
    2. **Dimensionality Reduction**: Reduce vectors from $D=384$ to $D=5$ using **UMAP** (retains non-linear manifold structure).
    3. **Density Clustering**: Cluster low-dimensional points using **HDBSCAN** (isolates organic clusters and discards noise).
    4. **c-TF-IDF Topic Extraction**: Compute class-based TF-IDF to find the top representative words characterizing each discovered cluster:
       ```
       c-TF-IDF_i = (t_i / w_i) * log(1 + (A / f_i))
       ```
    5. **LLM Representation**: Prompt a small LLM with the top c-TF-IDF words to generate a human-readable topic title.
    """)

with clust_col2:
    st.markdown("""
    ### Enterprise Applications of Embedding Clustering
    * **Customer Feedback Mining**: Group 500,000 app store reviews to discover sudden product bugs or feature requests.
    * **LLM Pretraining Dataset Filtering**: Cluster billions of Common Crawl web pages to enforce balanced domain distributions (e.g., 20% STEM, 15% Humanities, 15% Code).
    * **Automated Knowledge Base Taxonomy**: Organize internal corporate wikis into structured hierarchical catalogs.
    """)

# ==============================================================================
# SECTION 7: 6. KNOWLEDGE RETRIEVAL
# ==============================================================================
st.header("7. Knowledge Retrieval: Graph-Vector Fusion & Enterprise Q&A", divider="rainbow")

st.markdown("""
Enterprise knowledge retrieval expands beyond simple text retrieval to integrate **Knowledge Graphs**, 
structured SQL databases, and multi-modal entity catalogs.
""")

know_col1, know_col2 = st.columns([1, 1])

with know_col1:
    st.markdown("""
    ### GraphRAG: Combining Dense Vectors with Entity Graphs
    While pure vector search excels at retrieving isolated semantic text chunks, it fails at **Global Synthesis Questions** such as:
    * *"What are the top 5 macroeconomic risks facing our global supply chain across all subsidiaries?"*
    
    **GraphRAG (Edge & Community Summarization)**:
    1. Extract entities and relationships from text using an LLM.
    2. Build a Knowledge Graph and detect community clusters (Leiden algorithm).
    3. Generate hierarchical summaries for each community cluster and embed these summaries in a Vector DB.
    4. At query time, perform vector search across community summaries, navigating structural graph relations.
    """)

with know_col2:
    st.markdown("""
    ### Enterprise Search Modalities
    """)
    df_know = pd.DataFrame({
        "System Architecture": ["Vector-Only Retrieval", "Knowledge Graph (SPARQL/Cypher)", "GraphRAG (Hybrid)", "Self-Querying Vector Store"],
        "Strength": ["Fast, fuzzy matching over natural language", "Precise multi-hop relationship reasoning", "Global corpus synthesis & thematic reasoning", "Combines natural language with structured SQL filters"],
        "Weakness": ["Fails at multi-hop joins and global aggregation", "Brittle against ambiguous natural language", "High indexing cost (LLM graph extraction)", "Requires accurate query translation"],
        "Best Use Case": ["Basic FAQ / customer support", "Fraud rings, molecular pathways", "Complex research synthesis & intelligence analysis", "E-commerce product search with price/category constraints"]
    })
    st.dataframe(df_know, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Pure Python Hybrid Search & Reciprocal Rank Fusion (RRF)", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. A **Pure Python BM25 Sparse Search Engine**.
2. A **Dense Vector Semantic Search Engine**.
3. A **Reciprocal Rank Fusion (RRF) Hybrid Search Combiner** integrating both modalities.
""")

st.code('''
import numpy as np
import collections
from typing import List, Dict, Tuple, Optional

class HybridSearchEngine:
    """
    Production-grade educational engine demonstrating:
    1. Lexical BM25 Sparse Inverted Index.
    2. Dense Vector Semantic Search.
    3. Reciprocal Rank Fusion (RRF) for robust Hybrid Retrieval.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.dense_vectors: np.ndarray = np.empty((0, 0))
        
        # BM25 statistics
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = collections.defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list) # term -> [(doc_id, freq)]
        
    def index_corpus(self, docs: List[str], vectors: np.ndarray):
        """Indexes text corpus for both BM25 and Dense Search."""
        self.documents = docs
        self.dense_vectors = vectors.astype(np.float32)
        # Normalize dense vectors to unit length
        norms = np.linalg.norm(self.dense_vectors, axis=1, keepdims=True)
        self.dense_vectors = self.dense_vectors / np.clip(norms, a_min=1e-9, a_max=None)
        
        num_docs = len(docs)
        self.doc_len = []
        
        for doc_id, text in enumerate(docs):
            tokens = text.lower().split()
            self.doc_len.append(len(tokens))
            term_counts = collections.Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term].append((doc_id, count))
                self.doc_freqs[term] += 1
                
        self.avg_dl = float(np.mean(self.doc_len)) if self.doc_len else 1.0

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Scores documents using the BM25 retrieval formula."""
        query_terms = query.lower().split()
        scores = collections.defaultdict(float)
        num_docs = len(self.documents)
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            df = self.doc_freqs[term]
            # IDF formula with smoothing
            idf = np.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            
            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_len[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                scores[doc_id] += idf * (numerator / denominator)
                
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_dense(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Scores documents using dot product over normalized dense vectors."""
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        similarities = np.dot(self.dense_vectors, q_norm.flatten())
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in ranked_indices]

    def search_hybrid_rrf(self, query_text: str, query_vector: np.ndarray, top_k: int = 5, rrf_k: int = 60) -> List[Tuple[int, float]]:
        """
        Combines BM25 and Dense rankings using Reciprocal Rank Fusion (RRF).
        RRF_Score(d) = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
        """
        bm25_results = self.search_bm25(query_text, top_k=50)
        dense_results = self.search_dense(query_vector, top_k=50)
        
        rrf_scores = collections.defaultdict(float)
        
        # Add BM25 rank contributions
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)
            
        # Add Dense rank contributions
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)
            
        ranked_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_rrf[:top_k]
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Enterprise Dense Retrieval & Hybrid Search Fusion Simulator", divider="rainbow")

st.markdown("""
Observe how **Sparse Lexical Search (BM25)** and **Dense Semantic Retrieval** succeed in complementary ways, 
and inspect how **Reciprocal Rank Fusion (RRF)** seamlessly merges both rankings.
""")

# Build sample enterprise catalog
sample_corpus = [
    {"id": 0, "title": "Hardware Spec SKU-9901-X", "text": "Hardware specification and repair guide for model SKU-9901-X server power supplies.", "topic": "Exact SKU Hardware"},
    {"id": 1, "title": "Car Maintenance Handbook", "text": "Official service handbook for passenger car maintenance, motor diagnostics, and automobile repair.", "topic": "Automotive"},
    {"id": 2, "title": "Account Billing and Invoicing", "text": "How to update corporate billing credit cards, download PDF invoices, and review payment history.", "topic": "Finance"},
    {"id": 3, "title": "Automobile Engine Diagnostics", "text": "Detailed handbook for troubleshooting automotive engine faults and mechanical breakdowns.", "topic": "Automotive"},
    {"id": 4, "title": "Password Reset Procedures", "text": "Troubleshooting employee account access, two-factor authentication, and password resets.", "topic": "IT Security"}
]

# Synthetic dense embedding space
np.random.seed(42)
dim = 4
v_hardware = np.array([2.0, 0.0, 0.0, 0.0])
v_auto = np.array([0.0, 2.5, 0.0, 0.0])
v_finance = np.array([0.0, 0.0, 2.0, 0.0])
v_it = np.array([0.0, 0.0, 0.0, 2.0])

doc_vecs = np.array([
    v_hardware + np.random.randn(4)*0.1,
    v_auto + np.random.randn(4)*0.1,
    v_finance + np.random.randn(4)*0.1,
    v_auto + np.random.randn(4)*0.1,
    v_it + np.random.randn(4)*0.1
])
# Normalize
doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

wb_mode = st.radio("Select Interactive Workbench Mode", ["Hybrid RRF Retrieval Tester", "Enterprise Architecture Sizing Calculator"], horizontal=True)

if wb_mode == "Hybrid RRF Retrieval Tester":
    st.markdown("#### Test Queries Demonstrating Complementary Retrieval Strengths")
    test_scenario = st.selectbox(
        "Select Test Scenario Query",
        [
            "Scenario A (Vocabulary Mismatch): 'car mechanical repair handbook'",
            "Scenario B (Exact SKU / Keyword Search): 'SKU-9901-X power supply'",
            "Scenario C (Mixed Semantics and Entity): 'automobile invoice billing payment'"
        ]
    )
    
    if "Scenario A" in test_scenario:
        q_text = "car mechanical repair handbook"
        q_vec = v_auto # Aligns with automotive cluster
    elif "Scenario B" in test_scenario:
        q_text = "SKU-9901-X power supply"
        q_vec = v_hardware * 0.5 + np.random.randn(4)*0.2 # Noisy semantic match
    else:
        q_text = "automobile invoice billing payment"
        q_vec = v_auto * 0.5 + v_finance * 0.5
        
    # Run BM25 simulation
    q_tokens = set(q_text.lower().split())
    bm25_sim_scores = []
    for doc in sample_corpus:
        d_tokens = set(doc["text"].lower().split())
        match_count = len(q_tokens.intersection(d_tokens))
        score = match_count * 2.5 + (5.0 if "sku-9901-x" in doc["text"].lower() and "sku-9901-x" in q_text.lower() else 0.0)
        bm25_sim_scores.append(score)
        
    bm25_ranks = np.argsort(bm25_sim_scores)[::-1]
    
    # Run Dense simulation
    dense_sim_scores = np.dot(doc_vecs, q_vec / np.linalg.norm(q_vec))
    dense_ranks = np.argsort(dense_sim_scores)[::-1]
    
    # Run RRF
    rrf_scores = np.zeros(len(sample_corpus))
    for rank_idx, doc_id in enumerate(bm25_ranks, 1):
        rrf_scores[doc_id] += 1.0 / (60 + rank_idx)
    for rank_idx, doc_id in enumerate(dense_ranks, 1):
        rrf_scores[doc_id] += 1.0 / (60 + rank_idx)
        
    final_ranks = np.argsort(rrf_scores)[::-1]
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("**1. Sparse BM25 Ranks**")
        for i in bm25_ranks[:3]:
            st.markdown(f"• **{sample_corpus[i]['title']}** (Score: {bm25_sim_scores[i]:.2f})")
    with col_r2:
        st.markdown("**2. Dense Semantic Ranks**")
        for i in dense_ranks[:3]:
            st.markdown(f"• **{sample_corpus[i]['title']}** (Score: {dense_sim_scores[i]:.4f})")
    with col_r3:
        st.markdown("**3. Hybrid RRF Ranks (Winner)**")
        for i in final_ranks[:3]:
            st.markdown(f"• **{sample_corpus[i]['title']}** (RRF: {rrf_scores[i]:.4f})")
            
    st.info("Key Insight: RRF consistently brings the ground-truth document to rank #1 regardless of whether the query was dominated by exact keyword codes or semantic synonyms!")

else:
    st.markdown("#### Enterprise Vector Database Hardware & Memory Calculator")
    
    num_docs = st.number_input("Total Corpus Documents / Chunks", min_value=10000, max_value=500000000, value=10000000, step=1000000)
    embed_dim = st.selectbox("Embedding Dimension (D)", [384, 768, 1024, 1536, 3072], index=2)
    index_type = st.selectbox("Index Type", ["Flat (Raw float32)", "HNSW (Graph Links, 1.5x RAM)", "IVF-PQ (8x Compression)", "Scalar Quantization INT8 (4x Compression)"])
    
    raw_bytes = num_docs * embed_dim * 4 # float32
    raw_gb = raw_bytes / (1024**3)
    
    if "Flat" in index_type:
        ram_gb = raw_gb
    elif "HNSW" in index_type:
        ram_gb = raw_gb * 1.6
    elif "IVF-PQ" in index_type:
        ram_gb = raw_gb / 8.0
    else:
        ram_gb = raw_gb / 4.0
        
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Raw Vector Data", f"{raw_gb:.2f} GB")
    with col_m2:
        st.metric("Estimated Index RAM", f"{ram_gb:.2f} GB")
    with col_m3:
        st.metric("Recommended Nodes", f"{int(np.ceil(ram_gb / 32))} x 64GB RAM")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is Reciprocal Rank Fusion (RRF) preferred over weighted linear score combination (alpha * dense + (1-alpha) * bm25)?",
        "a": "Dense cosine similarities (typically in range 0.5 to 0.95) and BM25 scores (unbounded positive scores from 0 to 45+) exist on completely incomparable numerical scales. Linear combinations require continuous score normalization (min-max or z-score), which is sensitive to outlier queries and corpus size drift. RRF operates purely on ordinal ranks (position 1, position 2, ...) rather than raw scores: RRF(d) = 1/(k + rank_dense) + 1/(k + rank_bm25). This makes RRF completely immune to scale divergence and hyperparameter drift."
    },
    {
        "q": "How does a Two-Tower architecture maintain sub-10ms latency when recommending items from a catalog of 500 million products?",
        "a": "The Two-Tower architecture separates the query and item pipelines. The billions of item vectors are computed offline and indexed into an Approximate Nearest Neighbor (ANN) structure such as HNSW or ScaNN. At inference time, only the User Tower is evaluated to project the user's real-time context into vector u in ~2ms. The system then performs an inner product search across the precomputed ANN index in 3-5ms. This decouples inference latency from catalog size."
    },
    {
        "q": "What is the 'Lost in the Middle' phenomenon in RAG systems, and how do you engineer around it?",
        "a": "Liu et al. (2023) demonstrated that large language models attend effectively to information placed at the very beginning or the very end of their context window, but frequently overlook facts situated in the middle of long contexts. To mitigate this, production RAG pipelines apply a reranking step (Cross-Encoder) and strategically order the top retrieved passages: placing the highest scoring chunk at the very beginning (position 0), the second highest at the very end (position -1), and lower-ranking chunks in the interior."
    },
    {
        "q": "What is the difference between Parent-Document Retrieval and standard sliding window chunking in RAG?",
        "a": "Standard chunking creates fixed chunks (e.g., 500 tokens) and indexes them directly. However, small chunks lack broad context, while large chunks dilute semantic density. Parent-Document Retrieval splits documents into small child chunks (e.g., 100 tokens) for dense vector indexing. When a child chunk is retrieved during search, the system retrieves its larger parent chunk (e.g., 1000 tokens) or full document section to inject into the LLM context. This provides high precision in vector matching while preserving full narrative context for generation."
    },
    {
        "q": "Why is BERTopic's pipeline (Embeddings -> UMAP -> HDBSCAN -> c-TF-IDF) superior to traditional LDA for topic modeling?",
        "a": "Latent Dirichlet Allocation (LDA) relies on bag-of-words word co-occurrences, completely ignoring word order, syntax, and semantics. BERTopic uses sentence transformers to capture semantic meaning, UMAP to unroll non-linear high-dimensional manifolds, and HDBSCAN to find natural clusters without forcing every outlier into a category. Finally, c-TF-IDF extracts keywords at the cluster level rather than the document level, producing coherent, human-interpretable topic descriptions."
    }
]

for i, qa in enumerate(qas, 1):
    with st.expander(f"Q{i}: {qa['q']}"):
        st.markdown(f"**Answer**: {qa['a']}")

st.markdown("""
### Production Traps & Failure Modes
""")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1 (Fatal): Naive Linear Combination of BM25 and Cosine Scores**
    * *Failure*: Executing `final_score = 0.5 * bm25_score + 0.5 * cosine_sim`. Because BM25 ranges from 0 to 40 while Cosine is 0 to 1, BM25 completely obliterates the dense signal.
    * *Fix*: Always use rank-based fusion (Reciprocal Rank Fusion - RRF) or properly calibrated sigmoid-transformed score normalization.
    """)
    st.error("""
    **Trap 2 (Fatal): Over-Filtering in Vector Search Pre-Filtering**
    * *Failure*: Applying heavy metadata boolean filters directly to an HNSW graph, causing the search to get trapped in disconnected graph components and returning 0 results.
    * *Fix*: Use Vector DBs with native **ACID-compliant iterative filtering** (e.g., Qdrant, Milvus) that dynamically switch between graph search and payload index filtering.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding Negative Sampling in Two-Tower Training**
    * *Failure*: Training Two-Tower recommenders with only positive user clicks and random unweighted negatives.
    * *Fix*: Use In-Batch Negatives with cross-entropy loss, and explicitly sample **hard negatives** (items displayed to the user but not clicked) to prevent the model from learning trivial popularity shortcuts.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Over-Chunking into Millions of 50-Token Chunks**
    * *Failure*: Splitting enterprise documents into micro-chunks (50 tokens), causing 20x index bloat, severe fragmentation, and high RAM costs in the Vector DB.
    * *Fix*: Maintain chunk sizes between 300 and 600 tokens with 10-15% overlap, or use Hierarchical / Parent-Document Retrieval.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Reciprocal Rank Fusion**\n\nCombines dense and sparse rankings by summing 1 / (60 + rank). Scale-invariant and robust to score drift.")
with fc_col2:
    st.info("**Two-Tower Decoupling**\n\nItem vectors are precomputed offline; only User Tower runs at inference time, enabling sub-5ms candidate retrieval over billions of items.")
with fc_col3:
    st.info("**Parent-Document Retrieval**\n\nSearches on small child chunks for vector precision, returns large parent passage to LLM for full narrative context.")
with fc_col4:
    st.info("**GraphRAG Advantage**\n\nCombines dense vector search with community graph summaries to answer global, thematic corpus synthesis queries.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of enterprise embedding architectures across search, recommenders, RAG, and knowledge retrieval:
""")

c1 = st.checkbox("I can explain why BM25 and Dense Retrieval are complementary and how RRF unifies them.")
c2 = st.checkbox("I understand the architectural latency decoupling in Two-Tower recommendation systems.")
c3 = st.checkbox("I know how the RAG triad operates and how to resolve the 'Lost in the Middle' failure mode.")
c4 = st.checkbox("I can calibrate cosine similarity thresholds for automated near-duplicate ticket detection.")
c5 = st.checkbox("I understand the BERTopic pipeline (Embeddings -> UMAP -> HDBSCAN -> c-TF-IDF).")
c6 = st.checkbox("I can articulate the architectural difference between vector-only RAG and GraphRAG for global synthesis.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered enterprise embedding applications, hybrid search pipelines, and foundation model RAG architectures!")
