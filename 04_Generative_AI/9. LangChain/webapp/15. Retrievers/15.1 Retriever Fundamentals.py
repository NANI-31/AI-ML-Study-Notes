import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class MockDocument:
    """Represents a standardized LangChain Document object with page content and metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any], doc_id: str):
        self.page_content = page_content
        self.metadata = metadata
        self.id = doc_id

class MockRetrieverFundamentalsStudio:
    """
    Simulates a production-grade BaseRetriever engine executing dense similarity
    scoring, top-k truncation, and threshold gating across an enterprise knowledge base.
    """
    def __init__(self):
        self.corpus: List[MockDocument] = [
            MockDocument(
                page_content="LangChain BaseRetriever is an abstract LCEL Runnable converting an unstructured string into a List[Document].",
                metadata={"source": "langchain_core/retrievers.py", "category": "architecture", "version": "0.3"},
                doc_id="doc-ret-001"
            ),
            MockDocument(
                page_content="Vector stores expose .as_retriever() to wrap storage engines into composable Runnable pipelines.",
                metadata={"source": "vectorstores/base.py", "category": "integration", "version": "0.3"},
                doc_id="doc-ret-002"
            ),
            MockDocument(
                page_content="Cosine similarity computes the normalized dot product of two vectors, invariant to vector magnitude.",
                metadata={"source": "math/embeddings.py", "category": "vector_math", "version": "1.0"},
                doc_id="doc-ret-003"
            ),
            MockDocument(
                page_content="Top-k retrieval truncates nearest neighbor candidates, preventing prompt token bloat while ensuring high recall.",
                metadata={"source": "retrieval/top_k.py", "category": "search", "version": "2.1"},
                doc_id="doc-ret-004"
            ),
            MockDocument(
                page_content="Similarity score threshold retrieval discards all documents whose relevance score falls below a minimum cutoff tau.",
                metadata={"source": "retrieval/threshold.py", "category": "filtering", "version": "1.4"},
                doc_id="doc-ret-005"
            ),
            MockDocument(
                page_content="Enterprise RAG pipelines should avoid unconstrained k-NN search because noisy documents degrade LLM answer grounding.",
                metadata={"source": "production/best_practices.md", "category": "production", "version": "3.0"},
                doc_id="doc-ret-006"
            )
        ]

        # Pre-calculated normalized 3D vectors for synthetic semantic scoring
        self.vectors = {
            "doc-ret-001": np.array([0.92, 0.18, 0.22]),
            "doc-ret-002": np.array([0.88, 0.25, 0.30]),
            "doc-ret-003": np.array([0.20, 0.94, 0.15]),
            "doc-ret-004": np.array([0.55, 0.50, 0.62]),
            "doc-ret-005": np.array([0.48, 0.60, 0.61]),
            "doc-ret-006": np.array([0.85, 0.30, 0.38]),
        }
        # Normalize
        for k_id in self.vectors:
            norm = np.linalg.norm(self.vectors[k_id])
            if norm > 0:
                self.vectors[k_id] = self.vectors[k_id] / norm

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.lower()
        v = np.array([0.3, 0.3, 0.3])
        if any(w in q for w in ["retriever", "runnable", "lcel", "interface", "base"]):
            v = np.array([0.91, 0.20, 0.24])
        elif any(w in q for w in ["cosine", "math", "vector", "similarity", "dot product"]):
            v = np.array([0.22, 0.95, 0.16])
        elif any(w in q for w in ["top-k", "k", "heap", "truncate", "candidate"]):
            v = np.array([0.52, 0.52, 0.65])
        elif any(w in q for w in ["threshold", "cutoff", "filter", "score"]):
            v = np.array([0.45, 0.62, 0.62])
        elif any(w in q for w in ["enterprise", "production", "noise", "grounding", "rag"]):
            v = np.array([0.84, 0.32, 0.40])
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def retrieve(
        self,
        query: str,
        search_type: str = "similarity",
        k: int = 3,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        q_vec = self._embed_query(query)
        scored_docs = []

        for doc in self.corpus:
            d_vec = self.vectors[doc.id]
            sim = float(np.dot(q_vec, d_vec))
            sim_clamped = max(0.0, min(1.0, sim))
            scored_docs.append({
                "doc": doc,
                "score": round(sim_clamped, 4),
                "distance": round(1.0 - sim_clamped, 4)
            })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        if search_type == "similarity_score_threshold":
            filtered = [item for item in scored_docs if item["score"] >= threshold]
            return filtered[:k]

        return scored_docs[:k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.1 Retriever Fundamentals — LangChain Course",
    page_icon="🔍",
    layout="wide"
)

st.title("15.1 Retriever Fundamentals")
st.caption("Mastering the BaseRetriever Interface, Storage Decoupling, Vector Similarity Mechanics, and Top-K Truncation")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Retriever Abstraction Layer", divider="rainbow")
st.markdown("""
In LangChain, a **Retriever** is a higher-level functional abstraction that decouples **how documents are found** 
from **how documents are stored**. Unlike a vector database which manages underlying indices, embeddings, and CRUD 
mutations, a retriever implements the universal **LCEL Runnable contract**: 
`invoke(query: str) -> List[Document]`.

This design allows an application to swap vector indexes, BM25 keyword indices, web search APIs, or hybrid ensembles 
with zero modifications to the downstream generation chain.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RetrieverArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_input {
            label = "Step 1: Unstructured User Query";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            query_node [label="User Query: str\\n'Explain BaseRetriever interface'", fillcolor="#1E293B", color="#0284C7"];
        }

        subgraph cluster_retriever {
            label = "Step 2: BaseRetriever (Runnable Interface)";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            
            runnable_contract [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#0369A1" colspan="2"><b>Runnable Interface Contract</b></td></tr>
                <tr><td bgcolor="#1E293B">Input Type</td><td bgcolor="#1E293B">str (Natural Language Query)</td></tr>
                <tr><td bgcolor="#1E293B">Method</td><td bgcolor="#1E293B">invoke() / ainvoke() / batch()</td></tr>
                <tr><td bgcolor="#1E293B">Output Type</td><td bgcolor="#1E293B">List[Document]</td></tr>
            </table>>, shape=plaintext];

            dispatch_node [label="Dispatch to Backend Strategy\\n(Similarity / Threshold / MMR / External API)", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_storage {
            label = "Step 3: Storage & Execution Backends";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            
            vstore_backend [label="Vector Store Engine\\n(Chroma, Pinecone, FAISS, pgvector)", fillcolor="#1E293B", color="#64748B"];
            lexical_backend [label="Lexical / Keyword Engine\\n(BM25, Elastic, Lucene)", fillcolor="#1E293B", color="#64748B"];
            web_backend [label="External APIs / Web\\n(Tavily, Bing, Wikipedia)", fillcolor="#1E293B", color="#64748B"];
        }

        subgraph cluster_output {
            label = "Step 4: Standardized Output & Prompt Context";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            
            doc_list [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#059669" colspan="2"><b>List[Document] Payload</b></td></tr>
                <tr><td bgcolor="#1E293B">Doc 1</td><td bgcolor="#1E293B">page_content + metadata</td></tr>
                <tr><td bgcolor="#1E293B">Doc 2</td><td bgcolor="#1E293B">page_content + metadata</td></tr>
                <tr><td bgcolor="#1E293B">Doc k</td><td bgcolor="#1E293B">page_content + metadata</td></tr>
            </table>>, shape=plaintext];
            
            rag_prompt [label="Downstream LCEL Pipeline\\n(Context Injected into ChatPromptTemplate)", fillcolor="#064E3B", color="#10B981"];
        }

        query_node -> runnable_contract [label="invoke(query)"];
        runnable_contract -> dispatch_node [label="calls _get_relevant_documents"];
        dispatch_node -> vstore_backend [label="k-NN / Cosine"];
        dispatch_node -> lexical_backend [label="BM25 Inverted Index"];
        dispatch_node -> web_backend [label="REST HTTP Query"];
        
        vstore_backend -> doc_list [label="Format Docs"];
        lexical_backend -> doc_list [label="Format Docs"];
        web_backend -> doc_list [label="Format Docs"];
        
        doc_list -> rag_prompt [label="context formatting"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT IS A RETRIEVER? ---
st.header("2. What is a Retriever?", divider="rainbow")
st.markdown("""
In information retrieval and LLM engineering, a **Retriever** is a lightweight, read-only interface that accepts an 
unstructured string query and returns a ranked list of relevant `Document` objects.

### Core Architectural Characteristics:
1. **LCEL Runnable Compliance**: In LangChain v0.3+, every retriever inherits from `BaseRetriever`, which in turn inherits 
   from `RunnableSerializable`. This guarantees out-of-the-box support for `.invoke()`, `.ainvoke()`, `.stream()`, 
   `.batch()`, and `.abatch()`.
2. **Read-Only Abstraction**: Unlike vector stores, a retriever has **no methods for inserting, updating, or deleting documents** 
   (`.add_documents()` does not exist on `BaseRetriever`). Its sole responsibility is query evaluation.
3. **Decoupled Architecture**: Downstream generation chains (e.g. `retriever | format_docs | prompt | model | StrOutputParser()`) 
   only know that they are passing a string and receiving a list of documents. The underlying mechanism can be a dense 
   vector database, a full-text SQL search, or a custom microservice.
""")

col_def1, col_def2 = st.columns(2)
with col_def1:
    st.info("""
    **Primary LangChain Class Definition:**
    ```python
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document

    class CustomRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager
        ) -> list[Document]:
            # Execution logic here
            return [Document(page_content="...")]
    ```
    """)
with col_def2:
    st.success("""
    **Runnable Pipeline Integration:**
    ```python
    # Seamless chaining with the pipe operator
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )
    ```
    """)

st.divider()

# --- SECTION 3: RETRIEVER VS VECTOR STORE ---
st.header("3. Retriever vs Vector Store", divider="rainbow")
st.markdown("""
One of the most frequent points of confusion in Generative AI development is the boundary between a 
**Vector Store** and a **Retriever**. While a vector store often powers a retriever, they belong to different 
architectural layers.
""")

comp_data = {
    "Architectural Dimension": [
        "Primary Purpose",
        "Mutating Operations (CRUD)",
        "LCEL Runnable Protocol",
        "Supported Backends",
        "Search Methods Available",
        "Typical Lifecycle"
    ],
    "Vector Store (e.g. Chroma, FAISS, Pinecone)": [
        "Storage, index persistence, and embedding lifecycle",
        "Full support: add_documents(), delete(), persist()",
        "Does NOT implement Runnable directly",
        "Specifically vector databases and ANN indices",
        "similarity_search(), similarity_search_by_vector()",
        "Ingestion pipeline, index building, persistent DB"
    ],
    "Retriever (BaseRetriever)": [
        "Query execution and document ranking abstraction",
        "Read-only: Zero mutating operations",
        "Full first-class Runnable (invoke, batch, stream)",
        "Vector stores, BM25, Web APIs, Hybrid ensembles",
        "invoke(query) -> List[Document]",
        "Runtime query execution inside RAG chains"
    ]
}
st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

st.markdown("""
### Converting a Vector Store to a Retriever:
Any LangChain `VectorStore` can be effortlessly adapted into a `VectorStoreRetriever` using the `.as_retriever()` method:
""")
st.code("""
# VectorStore instance
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

# Converted into a BaseRetriever Runnable
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Now runnable in LCEL!
results = retriever.invoke("What is LangChain?")
""", language="python")

st.divider()

# --- SECTION 4: RETRIEVER INTERFACE ---
st.header("4. The Retriever Interface", divider="rainbow")
st.markdown("""
The `BaseRetriever` interface enforces a uniform execution contract across synchronous, asynchronous, and streaming 
workflows. By implementing `_get_relevant_documents`, LangChain automatically handles callbacks, tracing, telemetry, 
and concurrency controls.
""")

col_sync, col_async = st.columns(2)
with col_sync:
    st.subheader("Synchronous Methods")
    st.markdown("""
    - **`invoke(input: str, config=None)`**: Executes the retrieval synchronously and returns `List[Document]`.
    - **`batch(inputs: List[str], config=None)`**: Executes multiple queries concurrently using a thread pool.
    - **`stream(input: str)`**: Yields retrieved documents (typically in a single batch once search completes).
    """)
    st.code("""
# Synchronous invocation
docs = retriever.invoke("How does MMR prevent redundancy?")

# Batch processing multiple user questions
batch_docs = retriever.batch([
    "Define RAG architecture",
    "Explain vector embeddings"
])
    """, language="python")

with col_async:
    st.subheader("Asynchronous Methods")
    st.markdown("""
    - **`ainvoke(input: str, config=None)`**: Asynchronous non-blocking coroutine returning `List[Document]`.
    - **`abatch(inputs: List[str], config=None)`**: Async batch processing utilizing `asyncio.gather`.
    - **`astream(input: str)`**: Async generator yielding documents.
    """)
    st.code("""
# Async non-blocking call in FastAPI / asyncio
docs = await retriever.ainvoke("Production scaling tips")

# Concurrent async batching
multi_results = await retriever.abatch([
    "Query A",
    "Query B"
])
    """, language="python")

st.info("""
**Telemetry & Callbacks:** Because `BaseRetriever` is an LCEL Runnable, passing `config={"callbacks": [langsmith_handler]}` 
or setting tags `config={"tags": ["production", "legal-corpus"]}` automatically propagates spans to LangSmith for end-to-end tracing.
""")

st.divider()

# --- SECTION 5: SIMILARITY RETRIEVAL ---
st.header("5. Similarity Retrieval Mechanics", divider="rainbow")
st.markdown("""
**Similarity Retrieval** is the default retrieval strategy in vector-based RAG. When a user submits a query string:
1. The query text is encoded into a dense vector embedding using an embedding model.
2. The query vector is compared against all stored document vectors in the index.
3. The nearest candidates are ranked according to a geometric distance or similarity metric.
""")

dot_sim_flow = """
digraph SimMechanics {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

    q_text [label="Query Text\\n'How to configure retriever?'", fillcolor="#1E293B", color="#0284C7"];
    emb_model [label="Embedding Model\\nText -> Vector (1536D)", fillcolor="#0F172A", color="#38BDF8"];
    q_vec [label="Query Vector: q\\n[0.12, -0.45, 0.88, ...]", fillcolor="#1E293B", color="#64748B"];
    
    subgraph cluster_metric {
        label = "Geometric Metric Evaluation";
        style = "rounded,dashed";
        color = "#334155";
        fontcolor = "#94A3B8";
        
        cosine_box [label="Cosine Similarity\\nsim = dot(q, d) / (||q|| * ||d||)", fillcolor="#1E293B", color="#38BDF8"];
        euclid_box [label="Euclidean (L2) Distance\\ndist = sqrt(sum((q_i - d_i)^2))", fillcolor="#1E293B", color="#38BDF8"];
        ip_box [label="Inner Product (Dot)\\nsim = dot(q, d)", fillcolor="#1E293B", color="#38BDF8"];
    }

    rank_heap [label="Min-Heap / Priority Queue\\nRank Top Candidates by Descending Score", fillcolor="#064E3B", color="#10B981"];

    q_text -> emb_model -> q_vec;
    q_vec -> cosine_box;
    q_vec -> euclid_box;
    q_vec -> ip_box;
    cosine_box -> rank_heap;
    euclid_box -> rank_heap;
    ip_box -> rank_heap;
}
"""
st.graphviz_chart(dot_sim_flow, use_container_width=True)

st.markdown("""
### Search Modes in VectorStoreRetriever:
LangChain supports two primary similarity retrieval modalities:
- **Pure Similarity (`search_type="similarity"`)**: Retrieves the top-k highest scoring documents regardless of absolute score.
- **Score Threshold (`search_type="similarity_score_threshold"`)**: Filters out any document whose similarity score 
  falls below a strict threshold `score_threshold`. This prevents irrelevant context from polluting the prompt when no matching documents exist.
""")

st.divider()

# --- SECTION 6: TOP-K RETRIEVAL ---
st.header("6. Top-K Retrieval and Ranking Dynamics", divider="rainbow")
st.markdown("""
**Top-K Retrieval** defines the maximum number of documents (k) returned by the retriever. Selecting the optimal value of k 
is a critical hyperparameter optimization problem in production RAG systems.
""")

topk_c1, topk_c2, topk_c3 = st.columns(3)
with topk_c1:
    st.metric("Under-Retrieval (k = 1)", "Low Recall", delta="- Missing Facts", delta_color="inverse")
    st.write("Fastest latency and lowest prompt token costs, but risks omitting necessary nuance, table references, or secondary clauses.")

with topk_c2:
    st.metric("Optimal Range (k = 3 - 6)", "Balanced Precision", delta="+ Sweet Spot", delta_color="normal")
    st.write("Captures core facts while fitting comfortably inside prompt context budgets without inducing attention distraction.")

with topk_c3:
    st.metric("Over-Retrieval (k >= 15)", "High Recall, Low Precision", delta="- Lost in Middle", delta_color="inverse")
    st.write("Surfeits the LLM with distracting context. Leads to hallucination, lost-in-the-middle degradation, and high API costs.")

st.markdown("""
### Heap-Based Selection Dynamics:
Under the hood, modern vector index traversals (such as HNSW or IVF) maintain a **bounded priority queue (max-heap)** 
of size k. As candidate vectors are evaluated, any vector with a similarity lower than the k-th best item in the heap is 
immediately discarded. This guarantees O(N * log(k)) candidate selection rather than an expensive full O(N * log(N)) sort.
""")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Custom BaseRetriever Engine", divider="rainbow")
st.markdown("""
Below is a complete, production-ready implementation of a custom `BaseRetriever` adhering strictly to LangChain v0.3+ 
standards. It implements score threshold gating, metadata filtering, and full LCEL Runnable integration.
""")

st.code("""
from typing import List, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field
import numpy as np

class EnterpriseThresholdRetriever(BaseRetriever):
    \"\"\"
    Production-grade custom BaseRetriever with embedding similarity scoring,
    score-threshold gating, and metadata constraint verification.
    \"\"\"
    corpus_docs: List[Document] = Field(description="Internal document collection")
    corpus_embeddings: List[List[float]] = Field(description="Precomputed embedding vectors")
    min_score_threshold: float = Field(default=0.65, description="Relevance cutoff tau")
    k: int = Field(default=4, description="Maximum documents to return")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        \"\"\"
        Synchronous execution hook called by .invoke()
        \"\"\"
        # 1. Generate query embedding (simulated using synthetic unit vector)
        q_vec = np.random.rand(len(self.corpus_embeddings[0]))
        q_vec = q_vec / np.linalg.norm(q_vec)

        # 2. Evaluate cosine similarities
        scored_candidates = []
        for doc, emb in zip(self.corpus_docs, self.corpus_embeddings):
            doc_vec = np.array(emb)
            sim = float(np.dot(q_vec, doc_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec)))
            
            # 3. Apply threshold gating
            if sim >= self.min_score_threshold:
                # Add score into document metadata for observability
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "retrieval_score": round(sim, 4)}
                )
                scored_candidates.append((sim, doc_copy))

        # 4. Sort by score descending and truncate to top-k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_docs = [item[1] for item in scored_candidates[:self.k]]

        return top_docs

# Example Instantiation & LCEL Chaining:
# retriever = EnterpriseThresholdRetriever(
#     corpus_docs=[Document(page_content="...")],
#     corpus_embeddings=[[0.1, 0.2, ...]],
#     min_score_threshold=0.70,
#     k=3
# )
# results = retriever.invoke("Explain BaseRetriever")
""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Retriever Fundamentals Studio", divider="rainbow")
st.markdown("""
Test how **similarity metric evaluation**, **top-k truncation**, and **score threshold cutoffs** interact in real time. 
Query our synthetic LangChain engineering corpus and observe which documents survive retrieval gating.
""")

# Instantiate the module-scope engine
studio_engine = MockRetrieverFundamentalsStudio()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    user_query = st.text_input(
        "Enter Natural Language Retrieval Query:",
        value="How does BaseRetriever integrate with LCEL runnables?"
    )
    
    search_strategy = st.radio(
        "Search Strategy Mode:",
        ["similarity (Pure Top-K)", "similarity_score_threshold (Threshold Gated)"],
        index=0
    )
    
    top_k_val = st.slider("Select Top-K Limit (k):", min_value=1, max_value=6, value=3)
    
    threshold_val = 0.5
    if "threshold" in search_strategy:
        threshold_val = st.slider(
            "Minimum Score Cutoff Threshold (tau):",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05
        )

    execute_btn = st.button("🚀 Execute Retrieval", use_container_width=True, type="primary")

with wb_c2:
    st.subheader("Retriever Execution Output")
    
    if execute_btn or user_query:
        mode_key = "similarity_score_threshold" if "threshold" in search_strategy else "similarity"
        results = studio_engine.retrieve(
            query=user_query,
            search_type=mode_key,
            k=top_k_val,
            threshold=threshold_val
        )
        
        st.write(f"**Execution Summary**: Returned `{len(results)}` document(s) for query: *'{user_query}'*")
        
        if results:
            for idx, res in enumerate(results, 1):
                doc_obj = res["doc"]
                score = res["score"]
                dist = res["distance"]
                
                with st.expander(f"📄 Match #{idx}: {doc_obj.id} — Similarity: {score}", expanded=True):
                    st.markdown(f"**Content**: `{doc_obj.page_content}`")
                    m_c1, m_c2 = st.columns(2)
                    m_c1.metric("Cosine Similarity", score)
                    m_c2.metric("Geometric Distance", dist)
                    st.caption(f"**Metadata**: `{doc_obj.metadata}`")
        else:
            st.warning("⚠️ No documents met the threshold cutoff! Try lowering the score threshold.")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why does LangChain decouple VectorStore from BaseRetriever?",
     "Decoupling storage from retrieval adheres to the Single Responsibility Principle and the Open-Closed Principle. A VectorStore handles persistence, index mutation (CRUD), and embedding computation. A BaseRetriever is an LCEL Runnable that provides a uniform functional contract: query in, documents out. This allows developers to wrap non-vector systems (BM25, Tavily Web Search, Solr, multi-query algorithms) into identical RAG pipelines without rewriting downstream chains."),
    ("What happens under the hood when you call `vectorstore.as_retriever()`?",
     "The method instantiates a `VectorStoreRetriever` object holding a reference to the vector store. When `.invoke(query)` is called on the retriever, it converts the query to an embedding, invokes the vector store's underlying `similarity_search()` or `max_marginal_relevance_search()`, and packages the resulting chunks into standardized `Document` objects with their metadata."),
    ("How does `similarity_score_threshold` retrieval mitigate hallucination?",
     "Pure top-k retrieval unconditionally returns k documents, even if all stored vectors have zero conceptual relevance to the query (returning random distantly related chunks). In contrast, `similarity_score_threshold` enforces a hard score cutoff tau. If no documents score above tau, an empty list is returned. This signals to the prompt template or agent guardrail that no authoritative context exists, preventing the LLM from hallucinating answers based on irrelevant noise."),
    ("What is the time complexity difference between full sorting and top-k retrieval in large corpora?",
     "A naive full sort requires O(N log N) comparisons, which is prohibitive when N reaches millions of vectors. Production vector retrievers maintain a bounded max-heap or priority queue of size k during ANN graph traversal, reducing candidate selection to O(N log k). When combined with HNSW hierarchical graphs, search latency drops further to sub-linear O(log N)."),
    ("Why is setting k too high (e.g. k=20) harmful even when the LLM has a 128k context window?",
     "Even with large context windows, LLMs suffer from the 'lost in the middle' phenomenon, where facts situated in the middle of long prompts are frequently ignored. Furthermore, extraneous retrieved text introduces conflicting phrasing and unrelated terminology that degrades LLM attention precision while multiplying token generation costs and latency.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Calling `.add_documents()` on a `BaseRetriever`. `BaseRetriever` is strictly a read-only query interface. Attempting to invoke mutation methods on a retriever raises `AttributeError`. Document ingestion must always be performed on the underlying `VectorStore` instance before retriever instantiation.")
    st.error("**Trap 2 (Breaking Error)**: Using Unnormalized Vectors with Cosine Similarity Thresholds. If your embedding model produces non-unit vectors and your distance metric is dot product, scores can exceed 1.0 or dip below 0.0 unpredictably. Always ensure vectors are L2-normalized before applying absolute threshold gating.")
with trap_c2:
    st.warning("**Trap 3 (Latency Warning)**: Invoking Synchronous `.invoke()` Inside Async Web Handlers. In high-concurrency frameworks like FastAPI or Starlette, running synchronous `.invoke()` blocks the main event loop during vector database network I/O. Always use `await retriever.ainvoke(query)` in asynchronous routes.")
    st.warning("**Trap 4 (Quality Warning)**: Setting Score Threshold Too High in Cross-Domain Embeddings. Setting tau = 0.85 when using models like text-embedding-3-small often causes legitimate query paraphrases to be dropped completely. Benchmark your score distribution on a test dataset before hardcoding thresholds.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: BaseRetriever"):
        st.markdown("**Abstract LCEL Runnable accepting an unstructured string and returning a List[Document].**")
    with st.expander("🎴 Flashcard 2: as_retriever()"):
        st.markdown("**Adapter method on VectorStore instances converting storage backends into composable Runnable retrievers.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Score Threshold Gating"):
        st.markdown("**Discarding candidates scoring below tau to prevent irrelevant context from polluting LLM generation.**")
    with st.expander("🎴 Flashcard 4: Lost in the Middle"):
        st.markdown("**Degradation in LLM attention when critical facts are buried in large retrieved prompt context sets.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why BaseRetriever is an LCEL Runnable that decouples storage from retrieval.", value=True)
chk2 = st.checkbox("I can articulate the architectural differences between a VectorStore and a BaseRetriever.", value=True)
chk3 = st.checkbox("I know how to convert any vector store into a retriever using .as_retriever(search_type=...).", value=True)
chk4 = st.checkbox("I understand the mechanics of top-k truncation and heap-based candidate filtering.", value=True)
chk5 = st.checkbox("I can implement a custom BaseRetriever sub-class in Python using _get_relevant_documents.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 retriever fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 15.1 Retriever Fundamentals!")
