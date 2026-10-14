import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class BaseRetrieverEngine:
    """
    Simulates the abstraction contract between a high-level Retriever
    and a low-level Vector Store, handling query vectorization, Top-K gating,
    and similarity score normalization.
    """
    DOCUMENTS = [
        {"id": "doc_1", "text": "Retrievers decouple document retrieval logic from low-level storage primitives.", "vector": [0.88, 0.22, 0.15, 0.05], "source": "rag_handbook.pdf"},
        {"id": "doc_2", "text": "Vector databases provide persistent index storage, WAL durability, and CRUD APIs.", "vector": [0.35, 0.85, 0.20, 0.10], "source": "db_internals.pdf"},
        {"id": "doc_3", "text": "Query embedding encodes user natural language strings into dense coordinate points.", "vector": [0.82, 0.30, 0.25, 0.10], "source": "nlp_foundations.pdf"},
        {"id": "doc_4", "text": "Top-K retrieval controls candidate density to prevent LLM context dilution.", "vector": [0.75, 0.25, 0.10, 0.40], "source": "prompt_engineering.pdf"},
        {"id": "doc_5", "text": "Similarity scoring normalizes distance metrics into intuitive confidence thresholds.", "vector": [0.78, 0.20, 0.35, 0.30], "source": "retrieval_metrics.pdf"}
    ]

    @classmethod
    def retrieve(
        cls,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: float = 0.5,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        qv_norm = qv / q_norm if q_norm > 0 else qv

        results = []
        for doc in cls.DOCUMENTS:
            dv = np.array(doc["vector"], dtype=np.float32)
            d_norm = np.linalg.norm(dv)
            dv_norm = dv / d_norm if d_norm > 0 else dv

            if metric == "cosine":
                sim = float(np.dot(qv_norm, dv_norm))
            elif metric == "dot_product":
                sim = float(np.dot(qv, dv))
            else: # euclidean distance inverted
                dist = float(np.linalg.norm(qv - dv))
                sim = 1.0 / (1.0 + dist)

            if sim >= score_threshold:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "source": doc["source"],
                    "score": round(sim, 4),
                    "raw_distance": round(1.0 - sim if metric == "cosine" else sim, 4)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="10.1 Retriever Basics - RAG Master Curriculum",
    page_icon="🔍",
    layout="wide"
)

st.title("10.1 Retriever Basics: The Retrieval Abstraction Layer, Query Vectorization, and Scoring Mechanics")
st.caption("Chapter 10: Retrieval | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
In enterprise RAG architectures, your application logic should **never directly execute low-level vector database queries**. 
Instead, high-level orchestration frameworks (like LangChain and LlamaIndex) introduce the **Retriever** abstraction.

A **Retriever** is a runnable interface that accepts an unstructured natural language string as input and returns a list 
of relevant `Document` objects. It encapsulates query vectorization, connection pooling, similarity computation, 
top-K cutoff gating, and score calibration behind a clean, deterministic contract.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: The Retriever Layer in Modern RAG Pipelines", divider="rainbow")
st.markdown("""
The architecture below illustrates how the Retriever acts as the mediator between the user's natural language question, 
embedding model inference, and the underlying vector database.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    retriever_bp_dot = """
    digraph RetrieverArchitecture {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_app {
            label = "1. Application & Orchestration Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="Natural Language Query\\ne.g. 'How do retrievers differ from vector databases?'", fillcolor="#0F172A", color="#38BDF8"];
            rag_chain [label="RAG Orchestration Chain / Agent\\n(LangChain LCEL / LlamaIndex Engine)", fillcolor="#0F172A", color="#38BDF8"];
            user_query -> rag_chain [label="Invokes Pipeline"];
        }

        subgraph cluster_retriever {
            label = "2. The Retriever Abstraction Layer (Runnable)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            retriever_entry [label="Retriever.invoke(query_str)\\nStandard Unified Contract", fillcolor="#334155", color="#10B981"];
            embed_call [label="Query Embedding Model\\nGenerates 1536-dim Float32 Vector", fillcolor="#334155", color="#8B5CF6"];
            score_filter [label="Score Calibration & Top-K Gating\\nFilter: score >= threshold | Limit: K", fillcolor="#334155", color="#F59E0B"];
            doc_formatter [label="Document Object Hydration\\nList[Document(page_content, metadata)]", fillcolor="#334155", color="#10B981"];

            retriever_entry -> embed_call [label="Text-to-Vector"];
            embed_call -> score_filter [label="Execute Proximity"];
            score_filter -> doc_formatter [label="Hydrate Top-K"];
        }

        subgraph cluster_vdb {
            label = "3. Low-Level Vector Storage Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            vdb_engine [label="Vector Database Engine\\n(Qdrant / Chroma / Pinecone / Milvus)\\nHNSW Graph Walk & Inverted Metadata Bitset", fillcolor="#0F172A", color="#06B6D4"];
        }

        rag_chain -> retriever_entry [label="query: str"];
        embed_call -> vdb_engine [label="Vector Proximity Query"];
        vdb_engine -> score_filter [label="Raw Candidate Hits"];
        doc_formatter -> rag_chain [label="List[Document]"];
    }
    """
    st.graphviz_chart(retriever_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production frameworks (like **LangChain v0.3+** and **LlamaIndex**), the retriever is designed as a standardized **Runnable** 
(`RunnableSerializable[str, List[Document]]`). It exposes synchronous (`invoke`), asynchronous (`ainvoke`), 
and streaming (`astream`) interfaces. 
This decoupling means your downstream synthesis pipeline does not know whether documents are sourced from Qdrant, Pinecone, 
Elasticsearch, or an in-memory test list—enabling seamless engine migration without touching prompt construction or LLM logic.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHAT IS A RETRIEVER?
# ==============================================================================

st.header("2. What is a Retriever?", divider="rainbow")
st.markdown("""
A **Retriever** is a computational interface that defines how an information retrieval subsystem accepts 
a query and returns relevant context documents.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Core Responsibilities of a Retriever")
    st.markdown("""
    * **Contract Standardization**:
      Defines a universal signature: `get_relevant_documents(query: str) -> List[Document]`.
    * **Automated Query Vectorization**:
      Transparently calls the configured embedding model to convert raw query strings into high-dimensional vectors.
    * **Result Transformation & Metadata Hydration**:
      Translates database-specific payloads (e.g., Qdrant `ScoredPoint` or Pinecone `Match`) into standardized 
      `Document` objects containing `page_content` and `metadata` dictionaries.
    * **Lifecycle Management**:
      Manages connection timeouts, retry backoffs, and distributed cluster fallbacks.
    """)

with col2:
    st.subheader("LangChain LCEL Retriever Interface")
    st.code('''from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class ProductionRetriever(BaseRetriever):
    """Minimal runnable retriever implementation."""
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Embed query
        # 2. Query Vector DB
        # 3. Format into standardized Document objects
        return [
            Document(
                page_content="Retriever documentation content...",
                metadata={"source": "docs.pdf", "score": 0.92}
            )
        ]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: RETRIEVER VS VECTOR DATABASE
# ==============================================================================

st.header("3. Retriever vs. Vector Database", divider="rainbow")
st.markdown("""
A common point of confusion in RAG engineering is conflating the **Vector Database** with the **Retriever**. 
They operate at fundamentally different architectural tiers.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Comparison")
    comp_df = pd.DataFrame([
        {"Dimension", "Vector Database", "Retriever"},
        {"Primary Responsibility", "Persistent storage, indexing (HNSW/IVF), and vector CRUD", "Query orchestration, embedding calling, and context formatting"},
        {"Input Data Type", "Dense float vectors (e.g., float[1536]) + JSON filters", "Unstructured natural language text strings"},
        {"Output Data Type", "Raw point IDs, distance scores, and stored payload bytes", "Standardized Document objects ready for LLM prompt insertion"},
        {"Infrastructure Role", "Stateful database server or managed cloud cluster", "Stateless application logic / runnable orchestration component"},
        {"Scope", "Stores and searches coordinate embeddings", "Can combine Vector DB, BM25, Web APIs, and Graph DBs"}
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Key Takeaway")
    st.markdown("""
    * **A Vector Database is a storage engine**: It manages disk segments, memory buffers, and geometric index calculations.
    * **A Retriever is an access pattern**: It defines how the application extracts relevant knowledge.
    * **Not All Retrievers Use Vector Databases**: A retriever can query an external search API (Google/Bing), a relational SQL database, a Lucene BM25 index, or an ensemble of multiple sources.
    """)

st.divider()

# ==============================================================================
# SECTION 4: QUERY EMBEDDING
# ==============================================================================

st.header("4. Query Embedding Pipeline", divider="rainbow")
st.markdown("""
Before a vector search can occur, the incoming query text must be transformed into a dense vector 
in the exact metric space occupied by the stored document chunks.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query Embedding Mechanics")
    st.markdown("""
    * **Strict Model Parity**:
      The query **must** be embedded using the identical model and dimensionality used during document ingestion. 
      Querying an OpenAI `text-embedding-3-small` index with a Cohere v3 vector produces completely random results.
    * **Asymmetric Query Prefixes**:
      Modern embedding models (e.g., BGE, E5, Cohere v3) distinguish between document passages and user queries:
      * Documents are ingested with: `"passage: <text>"`
      * Queries are encoded with: `"query: <text>"`
      Failing to prepend the query instruction prefix causes a 10%-25% drop in retrieval recall.
    * **Latency Considerations**:
      Query embedding adds 15ms to 80ms of network latency. Enterprise retrievers cache frequent query embeddings in Redis.
    """)

with col2:
    st.subheader("Embedding Transformation Flowchart")
    embed_dot = """
    digraph QueryEmbedFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        raw_q [label="User Query: 'cancel subscription'", fillcolor="#0F172A"];
        prefix_q [label="Prefix Injection: 'query: cancel subscription'"];
        api_call [label="Inference API / Local Model Forward Pass"];
        norm_step [label="L2 Normalization (Unit Vector)"];
        dense_out [label="1536-dim Float32 Coordinate Array", fillcolor="#0F172A", color="#10B981"];

        raw_q -> prefix_q -> api_call -> norm_step -> dense_out;
    }
    """
    st.graphviz_chart(embed_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: TOP-K RETRIEVAL
# ==============================================================================

st.header("5. Top-K Retrieval Mechanics", divider="rainbow")
st.markdown("""
The **Top-K** parameter specifies the maximum number of nearest-neighbor candidates the retriever 
fetches from the database.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Candidate Density Tradeoffs")
    st.markdown("""
    * **Low Top-K (K = 1 to 2)**:
      * Pros: Minimal LLM token consumption and fastest prompt processing.
      * Cons: High risk of missing complementary facts spread across multiple chunks.
    * **High Top-K (K = 15 to 30)**:
      * Pros: High recall coverage.
      * Cons: Floods prompt with distractor text, increases token costs, and induces the **Lost in the Middle** hallucination effect.
    * **Production Standard (Two-Stage Retrieval)**:
      Retrieve K = 20 coarse candidates from the vector store, then pass them through a Cross-Encoder Reranker to select the top 3 to 5 most relevant chunks.
    """)

with col2:
    st.subheader("Top-K Configuration Rules of Thumb")
    rule_df = pd.DataFrame([
        {"Use Case", "Recommended K", "Rationale"},
        {"Factoid Q&A (Single Fact)", "K = 3", "Concise, precise answer needed; avoids distracting noise."},
        {"Analytical Summary", "K = 8 - 12", "Synthesizes multiple perspectives across longer documents."},
        {"Two-Stage Reranking Pipeline", "K = 20 - 50", "Initial coarse filter; reranker prunes down to top-4."},
        {"Agent Tool Retrieval", "K = 3 - 5", "Prevents overflowing small agent reasoning scratchpads."}
    ])
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: SIMILARITY SCORES
# ==============================================================================

st.header("6. Similarity Scores and Scoring Normalization", divider="rainbow")
st.markdown("""
Vector databases return a numerical proximity score alongside each retrieved document. 
Interpreting and normalizing these scores is essential for applying confidence cutoffs.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Distance Metrics vs. Similarity Scores")
    st.markdown("""
    * **Cosine Similarity**:
      Measures the cosine of the angle between vectors. Bounded between -1.0 and 1.0 (or 0.0 and 1.0 for non-negative embeddings). 1.0 indicates identical directional semantics.
    * **Cosine Distance**:
      `Cosine Distance = 1.0 - Cosine Similarity`. 0.0 means identical, 1.0 means orthogonal.
    * **Dot Product**:
      Equals cosine similarity **if and only if** vectors are L2-unit normalized (`||v|| = 1.0`). If unnormalized, dot product scales with vector magnitude.
    * **Euclidean (L2) Distance**:
      Measures physical spatial distance. Always positive. Must be transformed (e.g., `1 / (1 + distance)`) to obtain an intuitive 0-to-1 similarity score.
    """)

with col2:
    st.subheader("Score Interpretation Guidelines")
    score_df = pd.DataFrame([
        {"Normalized Score Range", "Semantic Meaning", "Recommended Action"},
        {"> 0.82", "High Confidence Match", "Directly include in LLM context prompt."},
        {"0.70 - 0.82", "Moderate Topical Relevance", "Include if higher-confidence matches are unavailable."},
        {"0.55 - 0.70", "Weak Semantic Overlap", "Discard or pass through cross-encoder reranker."},
        {"< 0.55", "Unrelated Noise / Out-of-Domain", "Discard immediately to prevent hallucination."}
    ])
    st.dataframe(score_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Base Retriever with Query Vectorization", divider="rainbow")
st.markdown("""
Below is an enterprise-grade Retriever class implementing strict query vector normalization, 
configurable Top-K candidate gating, similarity threshold filtering, and document object hydration.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseDocument:
    """Standardized Document payload for downstream LLM generation."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(score={self.metadata.get('score')}, source={self.metadata.get('source')})"

class EnterpriseRetriever:
    """
    Decoupled retriever component encapsulating query vectorization,
    threshold gating, and document hydration.
    """
    def __init__(
        self,
        dimension: int = 4,
        top_k: int = 3,
        similarity_threshold: float = 0.65
    ):
        self.dimension = dimension
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        # Simulated vector store storage
        self.store_vectors: Dict[str, np.ndarray] = {}
        self.store_payloads: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc_id: str, text: str, vector: List[float], source: str) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        self.store_vectors[doc_id] = v / norm if norm > 0 else v
        self.store_payloads[doc_id] = {"text": text, "source": source}

    def _embed_query(self, query: str) -> np.ndarray:
        """
        In production: Calls OpenAI or HuggingFace embedding client.
        Here: Generates a deterministic unit vector for demonstration.
        """
        # Simulated deterministic vector
        seed = sum(ord(c) for c in query) % 100
        raw = np.array([0.8, 0.2 + (seed % 10) * 0.02, 0.1, 0.05], dtype=np.float32)
        return raw / np.linalg.norm(raw)

    def get_relevant_documents(self, query: str) -> List[EnterpriseDocument]:
        """Core Retriever interface: query string -> List[Document]."""
        query_vec = self._embed_query(query)
        
        candidates = []
        for doc_id, vec in self.store_vectors.items():
            sim = float(np.dot(query_vec, vec))
            if sim >= self.similarity_threshold:
                candidates.append((doc_id, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:self.top_k]

        return [
            EnterpriseDocument(
                page_content=self.store_payloads[d_id]["text"],
                metadata={
                    "doc_id": d_id,
                    "source": self.store_payloads[d_id]["source"],
                    "score": round(sim, 4)
                }
            )
            for d_id, sim in top_candidates
        ]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Retriever Query Simulator & Top-K Cutoff Explorer", divider="rainbow")
st.markdown("""
Experiment with the Retriever layer. Adjust query vector parameters, similarity thresholds, 
and Top-K cutoffs to inspect retrieved documents and analyze how threshold gating suppresses out-of-domain noise.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Retriever Query Configuration")
    q_x = st.slider("Query Coordinate X (Retrievers / Architecture)", 0.0, 1.0, 0.85, step=0.05)
    q_y = st.slider("Query Coordinate Y (Databases / Infrastructure)", 0.0, 1.0, 0.25, step=0.05)
    q_z = st.slider("Query Coordinate Z (Query Embeddings / NLP)", 0.0, 1.0, 0.15, step=0.05)
    q_w = st.slider("Query Coordinate W (Scoring / Thresholds)", 0.0, 1.0, 0.10, step=0.05)

    metric_choice = st.selectbox("Distance Metric", ["cosine", "dot_product", "euclidean"])
    top_k_input = st.slider("Top-K Candidates", 1, 5, 3)
    threshold_input = st.slider("Similarity Threshold Cutoff", 0.0, 1.0, 0.60, step=0.05)

with col2:
    st.subheader("Retriever Output (Hydrated Documents)")
    q_vec = [q_x, q_y, q_z, q_w]
    hits = BaseRetrieverEngine.retrieve(
        query_vector=q_vec,
        top_k=top_k_input,
        score_threshold=threshold_input,
        metric=metric_choice
    )

    if hits:
        hit_df = pd.DataFrame(hits)
        st.dataframe(hit_df[["id", "score", "source", "text"]], use_container_width=True, hide_index=True)
        st.success(f"Retriever returned {len(hits)} documents passing threshold >= {threshold_input}.")
    else:
        st.warning(f"Zero documents passed threshold >= {threshold_input}! Safe fallback triggered.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: Why should production RAG systems decouple the Retriever layer from the Vector Database?"):
    st.markdown("""
    **Answer**:
    Decoupling the Retriever layer provides three architectural advantages:
    1. **Storage Agnosticism**: Application code depends on the generic `List[Document]` contract rather than proprietary vendor SDKs, allowing database migration (e.g., from Chroma to Qdrant) with zero application code changes.
    2. **Multi-Source Orchestration**: A retriever can query multiple underlying stores (e.g., combining pgvector with Elasticsearch BM25) and synthesize results before passing them to the LLM.
    3. **Pre- and Post-Processing Encapsulation**: Query rewriting, embedding prefixing, and threshold calibration live inside the retriever, keeping LLM generation logic clean and maintainable.
    """)

with st.expander("Q2: What is the risk of using unnormalized vectors when calculating dot product similarity?"):
    st.markdown("""
    **Answer**:
    Dot product between two vectors equals cosine similarity multiplied by the product of their Euclidean lengths:
    `DotProduct(A, B) = ||A|| * ||B|| * Cosine(A, B)`.
    If vectors are not L2-normalized during ingestion, longer document chunks that yield larger vector magnitudes 
    will produce artificially high dot product scores regardless of topical relevance, unfairly dominating retrieval rankings.
    """)

with st.expander("Q3: What causes query-document embedding misalignment in asymmetric retrieval models?"):
    st.markdown("""
    **Answer**:
    Asymmetric models (e.g., BGE, E5, Instructor) are trained with dual objectives: queries are short informational prompts, 
    while documents are long declarative passages. These models require specific prompt prefixes (such as `"query: "` vs `"passage: "`). 
    If an application embeds queries without the required prefix, the resulting vector maps to an incorrect region of metric space, 
    degrading retrieval recall by 15% to 30%.
    """)

with st.expander("Q4: How does setting a strict similarity threshold prevent catastrophic hallucinations in enterprise RAG?"):
    st.markdown("""
    **Answer**:
    Without a threshold, Top-K retrieval unconditionally returns K documents even when a user asks completely out-of-domain questions 
    (e.g., asking for internal salary data against a public marketing knowledge base). 
    The downstream LLM is forced to attempt an answer using irrelevant distractor text, leading to severe hallucinations. 
    A strict threshold acts as a circuit breaker: if no document exceeds the threshold, the retriever signals an empty result set, 
    allowing the application to gracefully reply: *"I do not have sufficient information to answer this question."*
    """)

with st.expander("Q5: What is the 'Two-Stage Retrieval' pattern and why is it preferred over large Top-K lookups?"):
    st.markdown("""
    **Answer**:
    In Two-Stage Retrieval:
    1. **Stage 1 (Coarse Retrieval)**: An inexpensive bi-encoder retriever pulls a broad candidate pool (K = 20 to 50) using fast ANN vector search.
    2. **Stage 2 (Fine Reranking)**: A computationally heavy Cross-Encoder model scores the query-document pairs jointly, re-sorting candidates and keeping only the top 3 to 5.
    This provides the high recall coverage of a large Top-K search while keeping LLM context prompts small, concise, and free of middle-context distractors.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Hardcoding Database Queries Directly Inside Agent Chains**  
    Embedding vendor-specific database SDK calls directly in LLM application chains couples your prompt logic to database infrastructure. 
    Always wrap retrieval logic in a standardized `BaseRetriever` interface.
    """)
    st.error("""
    **Trap 2: Omitting Query Asymmetry Prefixes on Fine-Tuned Models**  
    Failing to prepend `"query: "` when using models like E5 or BGE causes query vectors to miss document clusters, 
    leading to silent retrieval quality degradation.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Passing Raw Top-20 Chunks Directly to LLM Prompts**  
    Feeding 20 raw chunks into an LLM context window inflates token costs and triggers the 'Lost in the Middle' problem. 
    Keep Top-K concise (3 to 5) or apply a reranker.
    """)
    st.warning("""
    **Trap 4: Comparing Similarity Scores Across Different Embedding Models**  
    A cosine similarity of 0.75 on OpenAI `text-embedding-3-small` represents strong relevance, 
    while 0.75 on a different model might represent weak overlap. Calibrate thresholds per model.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Retriever vs. Vector DB")
        st.caption("What separates them?")
        st.markdown("**Back**: A Vector DB is a stateful storage engine; a Retriever is a stateless runnable contract taking string queries and returning Document objects.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Asymmetric Query Prefix")
        st.caption("Why is it needed?")
        st.markdown("**Back**: Models like E5/BGE require 'query: ' and 'passage: ' prefixes to bridge the geometric gap between short questions and long passages.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Threshold Circuit Breaker")
        st.caption("How does it stop hallucination?")
        st.markdown("**Back**: Drops low-confidence chunks (score < threshold) before prompting the LLM, enabling safe fallback responses on out-of-domain queries.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Two-Stage Retrieval")
        st.caption("How does it operate?")
        st.markdown("**Back**: Stage 1 uses fast ANN search to fetch 20-50 candidates; Stage 2 uses a Cross-Encoder reranker to extract the top 3-5 high-relevance chunks.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Retriever Basics:")

c1 = st.checkbox("I understand the architectural role of the Retriever abstraction layer in decoupling RAG chains from storage engines.", value=True)
c2 = st.checkbox("I can differentiate between Vector Database storage operations and stateless Retriever query contracts.", value=True)
c3 = st.checkbox("I understand query embedding pipelines, model parity requirements, and asymmetric query prefixing.", value=True)
c4 = st.checkbox("I know how to tune Top-K candidate density to balance recall against the 'Lost in the Middle' hallucination trap.", value=True)
c5 = st.checkbox("I understand similarity score calibration and how threshold gating acts as a relevance circuit breaker.", value=True)
c6 = st.checkbox("I can implement an enterprise-grade BaseRetriever with query vectorization and threshold filtering in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Retriever Basics and understand how to build clean, decoupled retrieval abstraction layers.")
