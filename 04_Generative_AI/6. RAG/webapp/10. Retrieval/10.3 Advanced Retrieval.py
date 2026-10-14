import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class AdvancedRetrievalSimulator:
    """
    Simulates advanced retrieval patterns: Multi-Query Retrieval, Query Expansion,
    Parent-Document (Small-to-Big) Retrieval, and Ensemble Retrieval.
    """
    # Parent documents (Full context)
    PARENTS = {
        "p_1": {
            "title": "Enterprise Vector Database Architecture Whitepaper",
            "full_text": "Production vector databases combine in-memory HNSW graphs with persistent write-ahead logs (WAL) on NVMe storage. When queries arrive, an asynchronous query coordinator fans out requests across multiple immutable segments. For metadata filtering, modern engines leverage single-stage filtered graph traversal (such as ACORN or Filtered HNSW), ensuring that candidate nodes satisfy boolean predicates during each hop without fragmenting graph connectivity or dropping search recall."
        },
        "p_2": {
            "title": "Hybrid Search and Rank Fusion Engineering Guide",
            "full_text": "Hybrid search bridges the gap between lexical keyword matching and dense conceptual vectors. While dense neural embeddings capture conceptual synonyms and cross-lingual intent, they struggle with exact alphanumeric tokens like error codes and product SKUs. Reciprocal Rank Fusion (RRF) evaluates the ordinal positions of candidates across both channels using the formula 1 / (60 + rank), combining results without fragile score normalization."
        }
    }

    # Child chunks (Small, focused vectors for retrieval)
    CHILDREN = [
        {"child_id": "c_1a", "parent_id": "p_1", "text": "HNSW graphs with persistent WAL on NVMe storage.", "vector": [0.88, 0.22, 0.15, 0.05]},
        {"child_id": "c_1b", "parent_id": "p_1", "text": "Single-stage filtered graph traversal (ACORN) preserves recall.", "vector": [0.82, 0.30, 0.20, 0.10]},
        {"child_id": "c_2a", "parent_id": "p_2", "text": "Dense neural embeddings struggle with exact alphanumeric error codes.", "vector": [0.35, 0.85, 0.20, 0.15]},
        {"child_id": "c_2b", "parent_id": "p_2", "text": "Reciprocal Rank Fusion (RRF) evaluates ordinal ranks without normalization.", "vector": [0.75, 0.25, 0.30, 0.40]}
    ]

    @classmethod
    def simulate_parent_child_retrieval(cls, query_vector: List[float], top_k_children: int = 2) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        qv = qv / q_norm if q_norm > 0 else qv

        # 1. Search child vectors
        scored_children = []
        for child in cls.CHILDREN:
            cv = np.array(child["vector"], dtype=np.float32)
            c_norm = np.linalg.norm(cv)
            sim = float(np.dot(qv, cv / c_norm if c_norm > 0 else cv))
            scored_children.append((child, sim))

        scored_children.sort(key=lambda x: x[1], reverse=True)
        top_children = scored_children[:top_k_children]

        # 2. Resolve to distinct parent documents
        retrieved_parents = []
        seen_parents = set()

        for child, sim in top_children:
            p_id = child["parent_id"]
            if p_id not in seen_parents:
                seen_parents.add(p_id)
                parent_info = cls.PARENTS[p_id]
                retrieved_parents.append({
                    "Parent ID": p_id,
                    "Parent Document Title": parent_info["title"],
                    "Matching Child Chunk": child["text"],
                    "Child Vector Score": round(sim, 4),
                    "Full Parent Context Extracted": parent_info["full_text"]
                })
        return retrieved_parents

    @classmethod
    def simulate_multi_query(cls, base_query: str) -> List[str]:
        """Generates multi-query expansions simulating LLM query diversification."""
        return [
            base_query,
            f"Technical architecture and internal mechanics of {base_query}",
            f"Tradeoffs, failure modes, and best practices for {base_query}",
            f"How enterprise production systems implement {base_query}"
        ]


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="10.3 Advanced Retrieval - RAG Master Curriculum",
    page_icon="🚀",
    layout="wide"
)

st.title("10.3 Advanced Retrieval: Multi-Query, Query Expansion, Parent-Document (Small-to-Big), and Self-Querying")
st.caption("Chapter 10: Retrieval | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Standard naive RAG assumes that user queries are perfectly phrased and that document chunks match both 
the optimal size for vector search and the optimal size for LLM generation. In production, **both assumptions fail**.

Users submit ambiguous or poorly phrased questions, and small chunks (100 tokens) yield precise vector matches 
but lack sufficient context for comprehensive LLM answers.

**Advanced Retrieval Patterns** solve these fundamental bottlenecks using **Multi-Query Generation**, 
**Parent-Document (Small-to-Big) Retrieval**, **Contextual Compression**, **Self-Querying Metadata Extraction**, 
and **Ensemble Routing**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Advanced Retrieval Workflows and Query Routers", divider="rainbow")
st.markdown("""
The architecture below outlines the flow of advanced retrieval patterns: decomposing user queries, 
executing multi-channel retrieval, and mapping small index chunks back to parent documents.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    adv_bp_dot = """
    digraph AdvancedRetrievalFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_input {
            label = "1. User Input & Query Transformation";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_q [label="User Query\\n'Explain vector storage persistence'", fillcolor="#0F172A", color="#38BDF8"];
            llm_router [label="LLM Query Transformer\\nMulti-Query / Self-Query Metadata Filter / Expansion", fillcolor="#0F172A", color="#8B5CF6"];
            raw_q -> llm_router;
        }

        subgraph cluster_retrieval_patterns {
            label = "2. Advanced Retrieval Strategies";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            mq_exec [label="A. Multi-Query Execution\\nRun N query variants in parallel & merge via RRF", fillcolor="#334155", color="#EC4899"];
            sq_exec [label="B. Self-Query Engine\\nExtracts structured JSON filter from text", fillcolor="#334155", color="#F59E0B"];
            child_search [label="C. Small-Chunk Vector Search\\nQuery fine-grained 100-token child vectors", fillcolor="#334155", color="#10B981"];

            llm_router -> mq_exec;
            llm_router -> sq_exec;
            llm_router -> child_search;
        }

        subgraph cluster_context_hydration {
            label = "3. Context Expansion & Compression";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            parent_lookup [label="Parent Document Store (Key-Value)\\nFetch rich 1000-token parent document", fillcolor="#0F172A", color="#10B981"];
            compressor [label="Contextual Compressor / Extractive Reranker\\nPrunes irrelevant sentences from parent", fillcolor="#0F172A", color="#06B6D4"];
            final_prompt [label="Final Enriched Context for LLM\\nHigh relevance + zero contextual loss", fillcolor="#0F172A", color="#10B981"];

            child_search -> parent_lookup [label="child.parent_id"];
            mq_exec -> parent_lookup;
            sq_exec -> parent_lookup;
            parent_lookup -> compressor;
            compressor -> final_prompt;
        }
    }
    """
    st.graphviz_chart(adv_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG frameworks (such as **LangChain**, **LlamaIndex**, and **Haystack**), **Parent-Document Retrieval** 
(also known as Small-to-Big retrieval) has become the gold standard for dense document corpuses. 
Instead of embedding a bulky 1,000-token chunk (which dilutes vector semantics), the system embeds small 100-token child chunks. 
When a child vector matches a query, the retriever uses an in-memory or Redis key-value store to retrieve the entire parent document. 
This delivers the best of both worlds: **high vector search precision** paired with **comprehensive generation context**.
""")

st.divider()

# ==============================================================================
# SECTION 2: MULTI-QUERY RETRIEVAL
# ==============================================================================

st.header("2. Multi-Query Retrieval", divider="rainbow")
st.markdown("""
**Multi-Query Retrieval** uses a Large Language Model to generate multiple diverse reformulations of a single user prompt, 
executing searches across all variants and merging the resulting documents.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Overcoming Semantic Brittleness")
    st.markdown("""
    * **The Problem**:
      User prompts are often short, incomplete, or use phrasing that differs from the indexed text. 
      A single query embedding might land just outside the target document cluster in vector space.
    * **The Multi-Query Solution**:
      An LLM generates 3 to 5 alternative perspectives (e.g., technical, high-level, question-form, keyword-oriented).
    * **Merging via RRF**:
      Documents retrieved across all query variants are deduplicated and fused using Reciprocal Rank Fusion (RRF), 
      promoting documents that appeared consistently across multiple queries.
    """)

with col2:
    st.subheader("Multi-Query Generation Example")
    st.code('''# Multi-Query Prompt Template
from langchain_community.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# User Query: "HNSW memory issues"
# Generated Queries:
# 1. "How to optimize RAM consumption in HNSW graph indexes?"
# 2. "Why does HNSW cause out of memory errors in production?"
# 3. "Techniques for vector index memory reduction: Quantization vs HNSW"
docs = retriever.invoke("HNSW memory issues")
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: QUERY EXPANSION
# ==============================================================================

st.header("3. Query Expansion", divider="rainbow")
st.markdown("""
**Query Expansion** enriches the user's initial query by appending related domain terms, synonyms, and contextual qualifiers.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Mechanisms for Expansion")
    st.markdown("""
    * **Lexical Expansion**:
      Appends synonyms using WordNet or specialized domain thesauri (e.g., medical MeSH terms or legal glossaries).
    * **Pseudo-Relevance Feedback (PRF)**:
      Performs an initial search, extracts the top terms from the highest-ranked documents, and appends them to the query for a second search pass.
    * **Hypothetical Document Embeddings (HyDE)**:
      Prompts an LLM to generate a hypothetical answer to the query, then embeds the **hypothetical passage** instead of the query. 
      Passage-to-passage vector similarity is dramatically higher than query-to-passage similarity.
    """)

with col2:
    st.subheader("HyDE Flowchart")
    hyde_dot = """
    digraph HyDEFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        q_in [label="User Question: 'How does pgvector scale?'", fillcolor="#0F172A"];
        llm_gen [label="LLM generates hypothetical passage:\\n'pgvector scales by utilizing HNSW and IVFFlat indexes...'"];
        embed_hyp [label="Embed hypothetical passage as vector"];
        search_vdb [label="Search Vector DB (Passage-to-Passage proximity)"];
        real_docs [label="Real authoritative documentation retrieved", fillcolor="#0F172A", color="#10B981"];

        q_in -> llm_gen -> embed_hyp -> search_vdb -> real_docs;
    }
    """
    st.graphviz_chart(hyde_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: QUERY REWRITING
# ==============================================================================

st.header("4. Query Rewriting", divider="rainbow")
st.markdown("""
**Query Rewriting** transforms conversational, pronoun-heavy, or ambiguous chat turns into self-contained search queries.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Resolving Conversational Context")
    st.markdown("""
    * **The Pronoun Trap in Multi-Turn RAG**:
      * Turn 1: *"Tell me about Qdrant vector database."*
      * Turn 2: *"How does it handle persistence?"*
      If Turn 2 is embedded directly, the query vector represents *"How does it handle persistence?"*, completely omitting the word **Qdrant**!
    * **De-contextualization**:
      An LLM inspects chat history and rewrites Turn 2 into:
      `"How does the Qdrant vector database implement disk persistence and WAL?"`
    """)

with col2:
    st.subheader("Query Rewriting Demonstration")
    rewrite_df = pd.DataFrame([
        {"Turn / Input", "Raw Query", "Rewritten Standalone Query"},
        {"Turn 1", "What is HNSW?", "What is Hierarchical Navigable Small World (HNSW)?"},
        {"Turn 2", "What are its main tradeoffs?", "What are the primary performance, memory, and build-time tradeoffs of HNSW?"},
        {"Turn 3", "How does IVF compare to it?", "How does the Inverted File Index (IVF) algorithm compare to HNSW in vector databases?"}
    ])
    st.dataframe(rewrite_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 5: PARENT-DOCUMENT RETRIEVAL
# ==============================================================================

st.header("5. Parent-Document Retrieval (Hierarchical Small-to-Big Chunking)", divider="rainbow")
st.markdown("""
**Parent-Document Retrieval** decouples the chunk size used for vector indexing from the chunk size provided to the LLM.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Fundamental Chunk Size Dilemma")
    st.markdown("""
    * **Small Chunks (100 tokens)**:
      * Excellent for embedding models: sharp semantic focus without averaging multiple topics.
      * Poor for LLM synthesis: lacks surrounding paragraphs, definitions, and broader context.
    * **Large Chunks (1,000 tokens)**:
      * Excellent for LLM synthesis: full narrative context and complete explanations.
      * Poor for embedding models: multiple topics get blended into an average vector, lowering similarity precision.
    * **The Small-to-Big Architecture**:
      1. Documents are split into large parent chunks (e.g., 1,000 tokens).
      2. Each parent chunk is subdivided into small child chunks (e.g., 100 tokens).
      3. **Only child chunks are embedded** in the vector database.
      4. When a child chunk matches a query, the retriever fetches the **entire parent chunk** from a Key-Value store.
    """)

with col2:
    st.subheader("Parent-Child Architecture")
    pc_dot = """
    digraph ParentChild {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        subgraph cluster_parent {
            label = "DocStore (Key-Value / Redis)";
            color = "#334155";
            style = "rounded";
            parent_doc [label="Parent Document #42\\n(Full 1,200 token section)", fillcolor="#0F172A", color="#10B981"];
        }

        subgraph cluster_children {
            label = "Vector Database Index";
            color = "#334155";
            style = "rounded";
            c1 [label="Child 1 (150 tokens)\\nvector_1"];
            c2 [label="Child 2 (150 tokens)\\nvector_2 (Query Hit!)", color="#EC4899", penwidth=2.0];
            c3 [label="Child 3 (150 tokens)\\nvector_3"];
        }

        q [label="User Query Vector", fillcolor="#0F172A"];
        q -> c2 [label="Closest Vector Hit"];
        c2 -> parent_doc [label="Retrieve Parent via ID"];
    }
    """
    st.graphviz_chart(pc_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: CONTEXTUAL COMPRESSION
# ==============================================================================

st.header("6. Contextual Compression", divider="rainbow")
st.markdown("""
**Contextual Compression** extracts and condenses retrieved documents to pass only the specific sentences 
relevant to the user's prompt into the LLM context window.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("How Contextual Compressors Operate")
    st.markdown("""
    Retrieved documents often contain 80% irrelevant boilerplate and only 20% actionable facts.
    * **Extractive Compression**: Uses a fast local model (or LLM) to score sentences within retrieved documents, discarding irrelevant paragraphs.
    * **Embeddings Filter**: Splits retrieved documents into sentences, recalculates similarity against the query vector, and keeps only sentences scoring above a threshold.
    * **Benefits**: Slashes LLM token costs by 50%-75% and eliminates distractors that cause hallucinations.
    """)

with col2:
    st.subheader("Compression Pipeline")
    comp_dot = """
    digraph CompressionFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        raw_retrieved [label="Retrieved Document (1,500 tokens)\\nContains legal disclaimer, intro, and 2 key facts", fillcolor="#0F172A"];
        compressor_step [label="Contextual Compressor\\nEvaluates relevance per sentence"];
        clean_context [label="Compressed Output (250 tokens)\\nPreserves only the 2 actionable facts", fillcolor="#0F172A", color="#10B981"];

        raw_retrieved -> compressor_step -> clean_context;
    }
    """
    st.graphviz_chart(comp_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: SELF-QUERY RETRIEVAL
# ==============================================================================

st.header("7. Self-Query Retrieval (Metadata Extractor from Natural Language)", divider="rainbow")
st.markdown("""
**Self-Query Retrieval** uses an LLM to parse a natural language query into a **semantic search query string** 
plus a structured **boolean metadata filter predicate**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("How Self-Query Operates")
    st.markdown("""
    When a user asks:
    `"Show me technical whitepapers on vector databases published after 2023 by the architecture team."`
    A standard retriever tries to embed the whole sentence, including dates and author names.
    A **Self-Query Retriever** prompts an LLM with the collection schema:
    * **Semantic Query**: `"technical vector databases"`
    * **Extracted Filter**:
      `category == 'whitepaper' AND year > 2023 AND department == 'architecture'`
    The vector database executes a single-stage filtered search with 100% precision!
    """)

with col2:
    st.subheader("Schema Definition in Self-Query")
    st.code('''from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="category", description="Document type: whitepaper, blog, manual", type="string"),
    AttributeInfo(name="year", description="Publication year", type="integer"),
    AttributeInfo(name="department", description="Author department", type="string")
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(temperature=0),
    vectorstore=vectorstore,
    document_contents="Technical RAG documentation",
    metadata_field_info=metadata_field_info
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: ENSEMBLE RETRIEVAL
# ==============================================================================

st.header("8. Ensemble Retrieval", divider="rainbow")
st.markdown("""
**Ensemble Retrieval** executes multiple independent retrievers (e.g., BM25 keyword search, dense HNSW vector search, 
and parent-document retrievers) in parallel, fusing their result lists using Reciprocal Rank Fusion (RRF).
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Architectural Advantages")
    st.markdown("""
    * **Modal Complementarity**:
      Blends the high keyword precision of BM25 with the conceptual semantic depth of dense embeddings.
    * **Fault Tolerance**:
      If the dense embedding model fails to capture a rare acronym, BM25 catches it. If BM25 fails due to synonyms, the dense retriever catches it.
    * **Configurable Weighting**:
      Allows adjusting relative channel weights:
      `weights = [0.4, 0.6]` for BM25 and dense retrieval respectively.
    """)

with col2:
    st.subheader("Ensemble Retrieval Architecture")
    ens_dot = """
    digraph EnsembleFlow {
        rankdir=TB;
        graph [bgcolor="#0B1120", nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        q [label="User Query", fillcolor="#0F172A"];
        r_dense [label="Dense Vector Retriever (HNSW)"];
        r_bm25 [label="Sparse Lexical Retriever (BM25)"];
        r_parent [label="Parent-Document Retriever"];
        rrf_merger [label="Reciprocal Rank Fusion (RRF)\\nEnsemble Merger"];
        out [label="Unified Ranked Results", fillcolor="#0F172A", color="#10B981"];

        q -> r_dense;
        q -> r_bm25;
        q -> r_parent;
        r_dense -> rrf_merger;
        r_bm25 -> rrf_merger;
        r_parent -> rrf_merger;
        rrf_merger -> out;
    }
    """
    st.graphviz_chart(ens_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("9. Production Implementation: Advanced Orchestrated Retrieval Pipeline (Parent-Document & Multi-Query)", divider="rainbow")
st.markdown("""
Below is an enterprise implementation of the Parent-Document (Small-to-Big) retrieval pattern 
paired with multi-query deduplication and key-value document expansion.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Set

class ProductionParentDocumentRetriever:
    """
    Implements Small-to-Big retrieval: indexes small child chunks in vector memory
    while hydrating full rich parent documents from a Key-Value docstore.
    """
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        # Primary Key-Value Document Store: parent_id -> full_text
        self.docstore: Dict[str, Dict[str, Any]] = {}
        # Vector index: child_id -> normalized_vector
        self.child_vectors: Dict[str, np.ndarray] = {}
        # Mapping: child_id -> (parent_id, child_text)
        self.child_metadata: Dict[str, Dict[str, str]] = {}

    def add_parent_document(self, parent_id: str, title: str, full_text: str, child_chunks: List[Dict[str, Any]]) -> None:
        """Stores parent document in docstore and indexes child vectors."""
        self.docstore[parent_id] = {"title": title, "text": full_text}
        
        for c in child_chunks:
            c_id = c["id"]
            vec = np.array(c["vector"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            self.child_vectors[c_id] = vec / norm if norm > 0 else vec
            self.child_metadata[c_id] = {"parent_id": parent_id, "child_text": c["text"]}

    def retrieve(self, query_vector: List[float], top_k_children: int = 3) -> List[Dict[str, Any]]:
        """Queries child vectors, resolves unique parent IDs, and returns parent documents."""
        qv = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qv)
        qv = qv / norm if norm > 0 else qv

        # 1. Search child vectors
        scores = []
        for c_id, cv in self.child_vectors.items():
            sim = float(np.dot(qv, cv))
            scores.append((c_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_k_children]

        # 2. De-duplicate and hydrate parent documents
        resolved_parents = []
        seen_parents: Set[str] = set()

        for c_id, sim in top_matches:
            p_id = self.child_metadata[c_id]["parent_id"]
            if p_id not in seen_parents:
                seen_parents.add(p_id)
                resolved_parents.append({
                    "parent_id": p_id,
                    "title": self.docstore[p_id]["title"],
                    "matched_child_snippet": self.child_metadata[c_id]["child_text"],
                    "similarity_score": round(sim, 4),
                    "full_context": self.docstore[p_id]["text"]
                })

        return resolved_parents
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("10. Interactive Workbench: Advanced Retrieval Strategy Comparison Simulator", divider="rainbow")
st.markdown("""
Test how Parent-Document (Small-to-Big) retrieval resolves fine-grained child vector hits 
into comprehensive parent documents, and inspect Multi-Query prompt expansions.
""")

tab_pc, tab_mq = st.columns([1, 1])

with tab_pc:
    st.subheader("Parent-Document (Small-to-Big) Simulator")
    q_x = st.slider("Query Dimension X (HNSW / Persistence)", 0.0, 1.0, 0.85, step=0.05)
    q_y = st.slider("Query Dimension Y (Hybrid / RRF)", 0.0, 1.0, 0.20, step=0.05)
    top_k_children = st.slider("Top-K Child Hits to Inspect", 1, 4, 2)

    if st.button("Run Small-to-Big Retrieval"):
        q_vec = [q_x, q_y, 0.15, 0.05]
        parent_results = AdvancedRetrievalSimulator.simulate_parent_child_retrieval(q_vec, top_k_children)
        if parent_results:
            st.dataframe(pd.DataFrame(parent_results)[["Parent ID", "Parent Document Title", "Child Vector Score", "Matching Child Chunk"]], use_container_width=True, hide_index=True)
            for res in parent_results:
                with st.expander(f"Full Extracted Context: {res['Parent Document Title']}"):
                    st.write(res["Full Parent Context Extracted"])

with tab_mq:
    st.subheader("Multi-Query Expansion Generator")
    sample_query = st.text_input("Enter User Prompt", value="vector database persistence")
    if st.button("Generate Multi-Query Perspectives"):
        variants = AdvancedRetrievalSimulator.simulate_multi_query(sample_query)
        st.write("**LLM-Generated Diverse Query Perspectives:**")
        for i, v in enumerate(variants, 1):
            st.info(f"Query {i}: {v}")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: Why is Parent-Document (Small-to-Big) retrieval superior to naive single-size chunking?"):
    st.markdown("""
    **Answer**:
    Naive chunking forces an unavoidable compromise: small chunks (100 tokens) yield high vector search precision 
    because their embeddings have sharp semantic focus, but they lack surrounding narrative context for the LLM. 
    Large chunks (1,000 tokens) contain rich context, but their vectors average multiple sub-topics together, degrading search accuracy. 
    Parent-Document retrieval decouples indexing from generation: fine-grained child chunks are embedded in the vector index for high-precision search, 
    and upon matching, the retriever fetches the entire parent document from a key-value store, providing rich context to the LLM.
    """)

with st.expander("Q2: How does Multi-Query Retrieval resolve user prompt ambiguity?"):
    st.markdown("""
    **Answer**:
    A user's initial query is often biased toward specific phrasing that may not align with the terminology used in indexed documents. 
    Multi-Query retrieval uses an LLM to generate 3 to 5 semantically diverse reformulations from different angles (technical terms, question form, high-level summary). 
    Executing all queries in parallel and merging their candidate sets via Reciprocal Rank Fusion (RRF) ensures documents matching any valid interpretation 
    are captured, boosting overall retrieval recall.
    """)

with st.expander("Q3: What is Hypothetical Document Embeddings (HyDE) and when should it be avoided?"):
    st.markdown("""
    **Answer**:
    HyDE prompts an LLM to generate a hypothetical answer to a user's question, and then embeds that **hypothetical passage** 
    to search the vector database. Because passage-to-passage similarity is much higher than query-to-passage similarity, 
    HyDE significantly improves recall on conceptual questions. 
    However, HyDE should be **avoided** in domains requiring exact factual precision (legal statutes, medical dosages, financial numbers) 
    because if the LLM hallucinates incorrect facts in its hypothetical passage, the retriever will fetch real documents that match the hallucination!
    """)

with st.expander("Q4: How does Contextual Compression reduce inference costs in enterprise RAG?"):
    st.markdown("""
    **Answer**:
    Standard retrieval returns full chunks that often contain 70%-80% irrelevant introductory or boilerplate text. 
    Contextual Compression uses a lightweight model or extractive cross-encoder to evaluate the sentences within retrieved documents, 
    filtering out text that does not directly answer the user query. Passing only the condensed relevant sentences into the final prompt 
    reduces prompt token volume by 50%-75%, directly cutting LLM inference latency and API costs.
    """)

with st.expander("Q5: What is Self-Query Retrieval and how does it convert unstructured text into structured filters?"):
    st.markdown("""
    **Answer**:
    Self-Query Retrieval uses an LLM provided with the database schema metadata (field names, types, descriptions). 
    The LLM parses user prompts (e.g., *"Show me PDF whitepapers from 2024 about Qdrant"*) and extracts two separate elements: 
    1) a pure semantic query string (`"Qdrant"`), and 2) a structured JSON filter AST (`format == 'pdf' AND year == 2024 AND category == 'whitepaper'`). 
    The vector database then applies single-stage metadata filtering during vector search, guaranteeing exact metadata compliance.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying HyDE in Fact-Critical Financial or Medical Workloads**  
    If the LLM hallucinates incorrect facts in the generated hypothetical passage, the vector store will retrieve documents 
    matching the hallucination, reinforcing false information in the final answer.
    """)
    st.error("""
    **Trap 2: Multiplying LLM API Costs and Latency with Naive Multi-Query**  
    Calling a large LLM to generate 5 query variants and embedding all 5 queries adds 500ms-1500ms of latency and 5x embedding calls. 
    Use small, fast models (e.g., GPT-4o-mini or Claude 3.5 Haiku) and cache popular query expansions.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Forgetting Key-Value DocStore Backups in Parent-Document RAG**  
    If child vectors are preserved in the vector database but the underlying Redis/Postgres parent docstore is corrupted or lost, 
    the retriever cannot hydrate parent text, rendering the vector index useless.
    """)
    st.warning("""
    **Trap 4: Missing Deduplication Across Multi-Query Result Lists**  
    Merging multiple query result lists without deduplicating document IDs will pack the prompt with multiple copies of the same passage. 
    Always use Reciprocal Rank Fusion (RRF) with distinct ID set deduplication.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Parent-Document RAG")
        st.caption("How does Small-to-Big operate?")
        st.markdown("**Back**: Embeds small child chunks for high vector search precision, and fetches full parent documents from a Key-Value docstore for rich LLM context.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Multi-Query Retrieval")
        st.caption("What problem does it fix?")
        st.markdown("**Back**: Overcomes query phrasing brittleness by generating multiple prompt variants, searching all in parallel, and fusing hits via RRF.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: HyDE Mechanism")
        st.caption("What does HyDE embed?")
        st.markdown("**Back**: Embeds an LLM-generated hypothetical answer rather than the raw user query, leveraging passage-to-passage semantic proximity.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Self-Query Retrieval")
        st.caption("How does it work?")
        st.markdown("**Back**: Uses an LLM to parse natural language queries into a semantic search string plus a structured JSON boolean metadata filter.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Advanced Retrieval:")

c1 = st.checkbox("I understand how Multi-Query retrieval overcomes prompt phrasing brittleness and how to fuse candidates via RRF.", value=True)
c2 = st.checkbox("I understand the mechanics and risks of Hypothetical Document Embeddings (HyDE).", value=True)
c3 = st.checkbox("I know how Query Rewriting de-contextualizes multi-turn conversational chat histories.", value=True)
c4 = st.checkbox("I understand the Parent-Document (Small-to-Big) architecture and why it solves the chunk size dilemma.", value=True)
c5 = st.checkbox("I know how Contextual Compression and Self-Querying metadata extraction operate in production.", value=True)
c6 = st.checkbox("I can implement an enterprise Small-to-Big Parent Document Retriever with Key-Value docstore hydration in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Advanced Retrieval Patterns and understand how to build resilient, production-grade RAG retrieval pipelines.")
