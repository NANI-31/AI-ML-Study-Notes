import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class RetrieverStudioDoc:
    """Represents a document chunk with raw content, metadata, and parent context."""
    def __init__(
        self,
        doc_id: str,
        content: str,
        parent_content: str,
        metadata: Dict[str, Any],
        keywords: List[str]
    ):
        self.doc_id = doc_id
        self.content = content
        self.parent_content = parent_content
        self.metadata = metadata
        self.keywords = [k.lower() for k in keywords]

class MockRetrieverTypesStudio:
    """
    Simulates advanced retriever types: VectorStoreRetriever, MultiQueryRetriever,
    ContextualCompressionRetriever, ParentDocumentRetriever, and EnsembleRetriever.
    """
    def __init__(self):
        self.docs = [
            RetrieverStudioDoc(
                doc_id="sec-01",
                content="OAuth 2.0 PKCE flow is mandatory for mobile and Single Page Applications.",
                parent_content="Section 4: Enterprise Identity Architecture. All public clients without client secrets must enforce OAuth 2.0 Proof Key for Code Exchange (PKCE) flow with SHA-256 code challenge verification. Legacy implicit grant flows are strictly deprecated to prevent auth code interception attacks across browser storage.",
                metadata={"category": "Security", "year": 2026, "author": "AppSec Team"},
                keywords=["oauth", "pkce", "security", "token", "auth", "spa"]
            ),
            RetrieverStudioDoc(
                doc_id="sec-02",
                content="API gateway rate limits are set to 10,000 requests per minute with Redis sliding window.",
                parent_content="Section 7: Ingress Traffic Management. To prevent volumetric Denial of Service and API exhaustion, ingress edge proxies enforce a distributed Redis-backed sliding window counter rate limiter. The standard tier receives 10,000 req/min, while authenticated internal microservices bypass this constraint.",
                metadata={"category": "Infrastructure", "year": 2025, "author": "CloudOps"},
                keywords=["rate limit", "redis", "gateway", "traffic", "sliding window", "infrastructure"]
            ),
            RetrieverStudioDoc(
                doc_id="sec-03",
                content="Zero Trust mTLS authentication verifies cryptographic x509 certs on every inter-pod call.",
                parent_content="Section 9: Service Mesh Zero Trust. Perimeter firewalls are insufficient for modern microservices. Every inter-pod communication within the Kubernetes cluster is encrypted and authenticated via SPIFFE/SPIRE x509 certificates with mandatory mutual TLS (mTLS) rotated every 24 hours.",
                metadata={"category": "Security", "year": 2026, "author": "SecOps"},
                keywords=["zero trust", "mtls", "x509", "certificates", "spiffe", "security", "pods"]
            ),
            RetrieverStudioDoc(
                doc_id="sec-04",
                content="PostgreSQL read replicas scale horizontally up to 15 nodes using streaming replication.",
                parent_content="Section 12: Database Tier Performance. OLTP query spikes are mitigated by offloading analytical and read-heavy workloads to an asynchronous PostgreSQL streaming replication pool. The cluster automatically scales read replicas up to 15 nodes based on CPU and buffer cache pressure.",
                metadata={"category": "Database", "year": 2026, "author": "DataPlatform"},
                keywords=["postgresql", "database", "replicas", "streaming", "scale", "sql"]
            )
        ]

    def _simulate_dense_similarity(self, query: str, doc: RetrieverStudioDoc) -> float:
        """Simulates dense vector similarity based on keyword overlap and semantic intent."""
        q_tokens = set(query.lower().split())
        doc_tokens = set(doc.content.lower().split())
        overlap = len(q_tokens.intersection(doc_tokens))
        kw_overlap = sum(1 for kw in doc.keywords if kw in query.lower())
        base_score = 0.35 + (0.15 * overlap) + (0.25 * kw_overlap)
        return min(0.98, max(0.20, base_score))

    def _simulate_bm25_score(self, query: str, doc: RetrieverStudioDoc) -> float:
        """Simulates lexical BM25 keyword matching."""
        q_terms = query.lower().split()
        matches = sum(1 for t in q_terms if t in doc.content.lower())
        return min(1.0, matches / max(1, len(q_terms)))

    def run_vector_store(self, query: str, k: int) -> List[Dict[str, Any]]:
        scored = []
        for d in self.docs:
            sim = self._simulate_dense_similarity(query, d)
            scored.append({"doc": d, "score": round(sim, 4), "type": "Dense Vector (k-NN)"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def run_multi_query(self, query: str, k: int) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Generate 3 simulated query variations
        variations = [
            query,
            f"Alternative phrasing: What protocols govern {query}?",
            f"Engineering perspective: Architectural specifications for {query}"
        ]
        doc_scores = {}
        for q_var in variations:
            for d in self.docs:
                sim = self._simulate_dense_similarity(q_var, d)
                if d.doc_id not in doc_scores or sim > doc_scores[d.doc_id]["score"]:
                    doc_scores[d.doc_id] = {"doc": d, "score": round(sim, 4), "type": "Multi-Query Max Fusion"}
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return variations, ranked[:k]

    def run_contextual_compression(self, query: str, k: int) -> List[Dict[str, Any]]:
        base_results = self.run_vector_store(query, k)
        compressed_results = []
        for item in base_results:
            # Simulate extractive compression: keep only the core relevant sentence
            extracted = item["doc"].content
            compression_ratio = round((1.0 - (len(extracted) / len(item["doc"].parent_content))) * 100, 1)
            compressed_results.append({
                "doc_id": item["doc"].doc_id,
                "compressed_content": extracted,
                "original_parent": item["doc"].parent_content,
                "compression_ratio": f"{compression_ratio}% tokens pruned",
                "score": item["score"]
            })
        return compressed_results

    def run_parent_document(self, query: str, k: int) -> List[Dict[str, Any]]:
        base_results = self.run_vector_store(query, k)
        parent_results = []
        for item in base_results:
            parent_results.append({
                "matched_child_chunk": item["doc"].content,
                "returned_parent_doc": item["doc"].parent_content,
                "doc_id": item["doc"].doc_id,
                "score": item["score"],
                "category": item["doc"].metadata["category"]
            })
        return parent_results

    def run_ensemble(self, query: str, k: int) -> List[Dict[str, Any]]:
        # Dense rank & BM25 rank combined via RRF
        dense_results = self.run_vector_store(query, len(self.docs))
        bm25_scored = []
        for d in self.docs:
            bm25 = self._simulate_bm25_score(query, d)
            bm25_scored.append({"doc": d, "score": bm25})
        bm25_scored.sort(key=lambda x: x["score"], reverse=True)

        # RRF formula: Score = sum( 1 / (60 + rank) )
        rrf_dict = {}
        for rank, item in enumerate(dense_results, 1):
            did = item["doc"].doc_id
            rrf_dict[did] = rrf_dict.get(did, 0.0) + (1.0 / (60 + rank))
        for rank, item in enumerate(bm25_scored, 1):
            did = item["doc"].doc_id
            rrf_dict[did] = rrf_dict.get(did, 0.0) + (1.0 / (60 + rank))

        final_ranked = []
        for d in self.docs:
            final_ranked.append({
                "doc": d,
                "rrf_score": round(rrf_dict[d.doc_id], 5),
                "type": "Ensemble (BM25 + Dense RRF)"
            })
        final_ranked.sort(key=lambda x: x["rrf_score"], reverse=True)
        return final_ranked[:k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.2 Retriever Types — LangChain Course",
    page_icon="📚",
    layout="wide"
)

st.title("15.2 Retriever Types")
st.caption("Comprehensive Guide to Vector Store, Multi-Query, Contextual Compression, Parent Document, Self-Query, and Ensemble Retrievers")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Modern LangChain Retriever Zoo", divider="rainbow")
st.markdown("""
Real-world RAG applications face severe retrieval bottlenecks when relying solely on naive top-k vector search:
vocabulary mismatch, loss of surrounding context, prompt token bloat, and inability to filter by structured attributes.

LangChain provides a specialized taxonomy of **advanced retriever classes** designed to solve each bottleneck.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RetrieverZooArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_problem {
            label = "Step 1: Production Retrieval Failure Modes";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            p_phrasing [label="Vocabulary & Phrasing Mismatch\\n(User phrased query poorly)", fillcolor="#1E293B", color="#EF4444"];
            p_chunk [label="Chunk Size Dilemma\\n(Small chunk = good vector, bad context)", fillcolor="#1E293B", color="#EF4444"];
            p_bloat [label="Prompt Token Bloat\\n(Retrieved chunks full of irrelevant fluff)", fillcolor="#1E293B", color="#EF4444"];
            p_meta [label="Unstructured Filter Queries\\n('Find security papers after 2025')", fillcolor="#1E293B", color="#EF4444"];
            p_rare [label="Rare Keywords & Acronyms\\n(Dense vector misses exact serial numbers)", fillcolor="#1E293B", color="#EF4444"];
        }

        subgraph cluster_solutions {
            label = "Step 2: Specialized LangChain Retriever Architectures";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            
            mq_node [label="MultiQueryRetriever\\nLLM generates N query variations -> Union", fillcolor="#0F172A", color="#38BDF8"];
            parent_node [label="ParentDocumentRetriever\\nSearch small child vectors -> Return full parent chunk", fillcolor="#0F172A", color="#38BDF8"];
            comp_node [label="ContextualCompressionRetriever\\nCompressor strips fluff -> Returns only relevant lines", fillcolor="#0F172A", color="#38BDF8"];
            self_q_node [label="SelfQueryRetriever\\nLLM parses query text + metadata filters", fillcolor="#0F172A", color="#38BDF8"];
            ens_node [label="EnsembleRetriever\\nBM25 Keyword + Dense Vector via RRF", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_downstream {
            label = "Step 3: High-Precision Context Output";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            
            clean_context [label="Synthesized High-Relevance Context\\nClean, bounded, context-rich chunks", fillcolor="#064E3B", color="#10B981"];
            llm_gen [label="LLM Generation Phase\\nAccurate grounding without hallucination", fillcolor="#022C22", color="#059669"];
        }

        p_phrasing -> mq_node;
        p_chunk -> parent_node;
        p_bloat -> comp_node;
        p_meta -> self_q_node;
        p_rare -> ens_node;

        mq_node -> clean_context;
        parent_node -> clean_context;
        comp_node -> clean_context;
        self_q_node -> clean_context;
        ens_node -> clean_context;

        clean_context -> llm_gen;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: VECTOR STORE RETRIEVER ---
st.header("2. Vector Store Retriever", divider="rainbow")
st.markdown("""
The `VectorStoreRetriever` is the baseline retriever in LangChain. It wraps any underlying vector database 
(`Chroma`, `FAISS`, `Pinecone`, `Qdrant`) and converts its similarity query routines into a standardized `BaseRetriever` runnable.
""")

col_vs1, col_vs2 = st.columns(2)
with col_vs1:
    st.markdown("""
    ### Search Configuration Options:
    1. **`search_type="similarity"`**: Default k-NN search returning the top-k highest cosine or Euclidean similarity vectors.
    2. **`search_type="similarity_score_threshold"`**: Rejects any candidate whose similarity score falls below `score_threshold`.
    3. **`search_type="mmr"`**: Maximal Marginal Relevance. Optimizes for a balance of high query similarity and diversity among returned chunks.
    """)
with col_vs2:
    st.code("""
# Creating a VectorStoreRetriever with MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,       # Initial pool of candidates
        "lambda_mult": 0.7   # 1.0 = pure similarity, 0.0 = max diversity
    }
)

# Invoking in LCEL
docs = retriever.invoke("Explain zero trust mTLS")
    """, language="python")

st.divider()

# --- SECTION 3: MULTI-QUERY RETRIEVER ---
st.header("3. Multi-Query Retriever", divider="rainbow")
st.markdown("""
Vector distance search is acutely sensitive to subtle phrasing choices. A user might ask: *"How do we protect web tokens?"*, 
while the corporate document is titled *"OAuth 2.0 PKCE Specification for Single Page Apps"*. A single embedding search might 
miss the document due to vocabulary divergence.

The **`MultiQueryRetriever`** solves this by using an LLM to generate multiple alternative formulations of the user's query 
from different perspectives, issuing vector searches for each generated query, and computing the unique union of all retrieved documents.
""")

dot_mq = """
digraph MultiQueryFlow {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

    uq [label="User Query: 'How to protect web tokens?'", fillcolor="#1E293B", color="#0284C7"];
    llm_gen [label="LLM Query Generator\\nPrompt: 'Generate 3 alternative search queries'", fillcolor="#0F172A", color="#38BDF8"];
    
    q1 [label="Query 1: 'Token storage best practices'", fillcolor="#1E293B", color="#64748B"];
    q2 [label="Query 2: 'OAuth PKCE security architecture'", fillcolor="#1E293B", color="#64748B"];
    q3 [label="Query 3: 'Preventing JWT interception attacks'", fillcolor="#1E293B", color="#64748B"];
    
    union_box [label="Unique Union & Deduplication Engine", fillcolor="#064E3B", color="#10B981"];

    uq -> llm_gen;
    llm_gen -> q1;
    llm_gen -> q2;
    llm_gen -> q3;
    q1 -> union_box [label="k-NN search"];
    q2 -> union_box [label="k-NN search"];
    q3 -> union_box [label="k-NN search"];
}
"""
st.graphviz_chart(dot_mq, use_container_width=True)

st.code("""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

# Executes 3 queries in parallel and deduplicates results
unique_docs = retriever.invoke("How do we protect web tokens?")
""", language="python")

st.divider()

# --- SECTION 4: CONTEXTUAL COMPRESSION ---
st.header("4. Contextual Compression", divider="rainbow")
st.markdown("""
When documents are chunked into 500 or 1,000 token segments, only a single sentence or paragraph within each chunk is 
typically relevant to the user's specific query. Dumping entire chunks into the prompt wastes expensive context tokens and 
distracts the LLM.

**`ContextualCompressionRetriever`** wraps a base retriever with a **Document Compressor**. The base retriever fetches 
coarse chunks, and the compressor either extracts only relevant sentences (`LLMChainExtractor`) or drops documents 
failing semantic criteria (`EmbeddingsFilter`).
""")

col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    st.info("""
    **`LLMChainExtractor`**:
    Uses an LLM to read each retrieved document in the context of the user query and extract only the sentences containing answers.
    Irrelevant filler paragraphs, headers, and footers are discarded.
    """)
with col_comp2:
    st.success("""
    **`EmbeddingsFilter`**:
    A lightweight, low-cost alternative to LLM extraction. Computes vector similarity between the query and smaller sub-segments of each chunk, 
    dropping sub-segments whose similarity falls below a threshold without making LLM API calls.
    """)

st.code("""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Returns only the exact relevant sentences, pruning 70%+ of prompt tokens!
compressed_docs = compression_retriever.invoke("What is the Redis sliding window limit?")
""", language="python")

st.divider()

# --- SECTION 5: PARENT DOCUMENT RETRIEVAL ---
st.header("5. Parent Document Retrieval", divider="rainbow")
st.markdown("""
In naive RAG, developers face an irreconcilable trade-off known as the **Chunk Size Dilemma**:
- **Small Chunks (e.g. 100 tokens)**: Yield tight, highly specific embedding vectors that accurately match user queries, 
  but lack sufficient surrounding narrative context for the LLM to formulate complete answers.
- **Large Chunks (e.g. 1,500 tokens)**: Contain abundant narrative context, but produce diluted vector embeddings that fail 
  similarity searches for narrow, specific queries.

The **`ParentDocumentRetriever`** completely eliminates this dilemma by decoupling the **search index** from the **context payload**.
""")

dot_parent = """
digraph ParentDocFlow {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

    p_doc [label="Full Parent Document (1,500 tokens)\\nStored in InMemoryDocstore / Redis", fillcolor="#1E293B", color="#38BDF8"];
    
    subgraph cluster_children {
        label = "Child Chunks (100 tokens each) with 'doc_id' pointing to Parent";
        style = "rounded,dashed";
        color = "#334155";
        fontcolor = "#94A3B8";
        c1 [label="Child 1 Vector\\n(In VectorStore)", fillcolor="#0F172A", color="#64748B"];
        c2 [label="Child 2 Vector\\n(In VectorStore)", fillcolor="#0F172A", color="#64748B"];
        c3 [label="Child 3 Vector\\n(In VectorStore)", fillcolor="#0F172A", color="#64748B"];
    }

    q_user [label="User Query: 'PKCE code challenge'", fillcolor="#1E293B", color="#0284C7"];
    match_node [label="Vector Match: Child 2\\n(Score: 0.94)", fillcolor="#064E3B", color="#10B981"];
    return_parent [label="Retriever returns FULL PARENT DOCUMENT to LLM!\\nRich narrative context preserved", fillcolor="#022C22", color="#059669"];

    p_doc -> c1 [label="split"];
    p_doc -> c2 [label="split"];
    p_doc -> c3 [label="split"];
    
    q_user -> match_node [label="similarity search"];
    match_node -> c2;
    c2 -> return_parent [label="lookup parent_id in Docstore"];
}
"""
st.graphviz_chart(dot_parent, use_container_width=True)

st.code("""
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Child splitter: small chunks for high vector precision
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# 2. Parent splitter: full document or large sections for rich context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

store = InMemoryByteStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add full documents: children are embedded into vectorstore, parents saved to byte_store
retriever.add_documents(raw_documents)

# Query matches a child chunk, but returns the full 2,000-token parent document!
parent_docs = retriever.invoke("What is PKCE code challenge?")
""", language="python")

st.divider()

# --- SECTION 6: SELF-QUERY RETRIEVAL ---
st.header("6. Self-Query Retrieval", divider="rainbow")
st.markdown("""
Standard semantic vector search is incapable of handling queries that combine **semantic intent** with **structured metadata constraints**.

For example, when a user asks:
> *"Find security incident reviews written after 2025 by AppSec Team with rating above 4."*

A naive vector search embeds the entire sentence, including "after 2025" and "AppSec Team". Because numbers and dates embed poorly, 
the search frequently retrieves 2021 reports or documents from CloudOps.

The **`SelfQueryRetriever`** uses an LLM to deconstruct the query into two distinct components:
1. **Semantic Query String**: `"security incident reviews"`
2. **Structured Metadata Filter**: `{"year": {">=": 2025}, "author": "AppSec Team", "rating": {">": 4}}`
""")

st.code("""
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(name="category", description="Document category, e.g. Security, Infrastructure, Database", type="string"),
    AttributeInfo(name="year", description="Year the policy was published", type="integer"),
    AttributeInfo(name="author", description="Authoring team or engineer", type="string"),
]

document_content_description = "Enterprise engineering architectural specifications and policies"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# LLM translates prompt into:
# query='security incident reviews' filter=Comparison(comparator=gte, attribute='year', value=2025)
filtered_docs = self_query_retriever.invoke("Find security incident reviews written after 2025")
""", language="python")

st.divider()

# --- SECTION 7: ENSEMBLE RETRIEVAL ---
st.header("7. Ensemble Retrieval", divider="rainbow")
st.markdown("""
Dense vector search excels at conceptual matching but struggles with rare technical jargon, product codes, or exact serial numbers. 
Conversely, lexical search (BM25) excels at exact keyword matching but fails at semantic synonymy.

The **`EnsembleRetriever`** executes multiple retrievers in parallel (such as a BM25 retriever and a Dense vector retriever) 
and combines their ranked outputs using **Reciprocal Rank Fusion (RRF)**.
""")

st.info("""
**Reciprocal Rank Fusion (RRF) Formula:**
For any document d in candidate sets R_1, ..., R_m:
`RRF_score(d) = sum( w_i / (k_const + rank_i(d)) )`
Where `k_const` is typically 60, and `w_i` is the weighting assigned to retriever i. Documents that score well across both 
lexical and semantic indices receive high composite scores.
""")

st.code("""
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 1. Lexical BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4

# 2. Semantic Dense Vector Retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Ensemble Fusion with equal weighting
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# Hybrid search combining exact keyword hits with dense semantic understanding
hybrid_results = ensemble_retriever.invoke("OAuth 2.0 PKCE flow")
""", language="python")

st.divider()

# --- SECTION 8: METADATA-BASED RETRIEVAL ---
st.header("8. Metadata-Based Retrieval", divider="rainbow")
st.markdown("""
In multi-tenant SaaS and regulated enterprise architectures, **unfiltered retrieval is a catastrophic security vulnerability**. 
Metadata-based retrieval enforces tenant boundaries, role-based access control (RBAC), and temporal validity constraints.
""")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.subheader("Hard Pre-Filtering (Mandatory)")
    st.markdown("""
    Pre-filtering applies metadata constraints at the vector database index level before any similarity calculations occur:
    ```python
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {
                "tenant_id": "org_enterprise_99",
                "clearance_level": "secret"
            }
        }
    )
    ```
    Guarantees zero data leakage across tenant boundaries and ensures returned items always satisfy security policies.
    """)

with col_meta2:
    st.subheader("Dynamic Runtime Filter Injection")
    st.markdown("""
    In LCEL, filters can be dynamically injected at runtime per-request using `configurable_fields`:
    ```python
    from langchain_core.runnables import ConfigurableField

    retriever = vectorstore.as_retriever().configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Kwargs",
            description="Dynamic runtime filter parameters"
        )
    )

    # Injected dynamically from JWT bearer token
    docs = retriever.with_config(
        configurable={"search_kwargs": {"filter": {"user_id": current_user.id}}}
    ).invoke("My past orders")
    ```
    """)

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Multi-Retriever Composition Engine", divider="rainbow")
st.markdown("""
The production script below demonstrates how to construct a robust, production-grade retrieval pipeline that composes 
an `EnsembleRetriever` with a `ContextualCompressionRetriever`.
""")

st.code("""
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def build_production_hybrid_retriever(
    documents: List[Document],
    openai_api_key: str
) -> ContextualCompressionRetriever:
    \"\"\"
    Builds a two-stage hybrid retrieval pipeline:
    Stage 1: Parallel Ensemble (BM25 Lexical + FAISS Semantic Dense) combined via RRF.
    Stage 2: Contextual Compression using LLMChainExtractor to prune irrelevant sentences.
    \"\"\"
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

    # 1. Lexical BM25
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 6

    # 2. Semantic FAISS VectorStore
    vectorstore = FAISS.from_documents(documents, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 3. Stage 1: Ensemble Fusion
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6]  # 40% lexical, 60% semantic
    )

    # 4. Stage 2: Contextual Compression
    compressor = LLMChainExtractor.from_llm(llm)
    pipeline_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )

    return pipeline_retriever
""", language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Retriever Types Studio", divider="rainbow")
st.markdown("""
Compare how different retriever architectures process the exact same user query across our enterprise architecture corpus.
Observe how **Parent Document Retrieval** resolves the context boundary, how **Multi-Query** expands vocabulary, and how 
**Contextual Compression** trims tokens.
""")

# Instantiate the module-scope engine
types_studio = MockRetrieverTypesStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_type = st.selectbox(
        "Select Retriever Architecture to Test:",
        [
            "VectorStoreRetriever (Standard Dense k-NN)",
            "MultiQueryRetriever (Query Expansion & Union)",
            "ContextualCompressionRetriever (Sentence Pruning)",
            "ParentDocumentRetriever (Child Vector -> Parent Return)",
            "EnsembleRetriever (BM25 + Dense RRF Fusion)"
        ]
    )

    test_query = st.text_input(
        "Enter Query:",
        value="How does OAuth PKCE prevent auth code attacks?"
    )

    top_k_select = st.slider("Select Output k Limit:", min_value=1, max_value=4, value=2)

    run_bench_btn = st.button("⚡ Execute Retriever Evaluation", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Retriever Output Inspection")

    if run_bench_btn or test_query:
        if "VectorStore" in selected_type:
            results = types_studio.run_vector_store(test_query, top_k_select)
            for idx, item in enumerate(results, 1):
                with st.expander(f"Match #{idx}: {item['doc'].doc_id} (Score: {item['score']})", expanded=True):
                    st.write(f"**Content**: `{item['doc'].content}`")
                    st.caption(f"**Metadata**: `{item['doc'].metadata}`")

        elif "MultiQuery" in selected_type:
            variations, results = types_studio.run_multi_query(test_query, top_k_select)
            st.info(f"**Generated Query Variations:**\\n- " + "\\n- ".join(variations))
            for idx, item in enumerate(results, 1):
                with st.expander(f"Union Match #{idx}: {item['doc'].doc_id} (Max Score: {item['score']})", expanded=True):
                    st.write(f"**Content**: `{item['doc'].content}`")

        elif "Compression" in selected_type:
            comp_results = types_studio.run_contextual_compression(test_query, top_k_select)
            for idx, item in enumerate(comp_results, 1):
                with st.expander(f"Compressed Match #{idx}: {item['doc_id']} ({item['compression_ratio']})", expanded=True):
                    st.markdown(f"**Compressed Snippet Sent to LLM**:\\n`{item['compressed_content']}`")
                    st.caption(f"**Original Raw Parent Document**:\\n`{item['original_parent']}`")

        elif "ParentDocument" in selected_type:
            parent_results = types_studio.run_parent_document(test_query, top_k_select)
            for idx, item in enumerate(parent_results, 1):
                with st.expander(f"Parent Match #{idx}: {item['doc_id']} — Category: {item['category']}", expanded=True):
                    st.write(f"**Child Vector Matched**: `{item['matched_child_chunk']}`")
                    st.success(f"**Full Parent Context Returned to LLM**:\\n`{item['returned_parent_doc']}`")

        elif "Ensemble" in selected_type:
            ens_results = types_studio.run_ensemble(test_query, top_k_select)
            for idx, item in enumerate(ens_results, 1):
                with st.expander(f"Hybrid Fusion #{idx}: {item['doc'].doc_id} (RRF Score: {item['rrf_score']})", expanded=True):
                    st.write(f"**Content**: `{item['doc'].content}`")
                    st.caption(f"**Keywords**: `{item['doc'].keywords}`")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the Chunk Size Dilemma, and how does ParentDocumentRetriever solve it?",
     "The Chunk Size Dilemma states that small chunks produce accurate, concentrated embedding vectors for semantic search but lack sufficient narrative context for LLM generation, whereas large chunks contain complete context but produce diluted embeddings. ParentDocumentRetriever solves this by decoupling the search unit from the retrieval unit: it splits documents into small child chunks for vector indexing, but when a child chunk matches, it looks up and returns the full parent document from a docstore."),
    ("How does Reciprocal Rank Fusion (RRF) calculate composite rankings in EnsembleRetriever?",
     "RRF combines ranked candidate lists from disparate retrieval algorithms (e.g. BM25 and dense vectors) without needing to calibrate raw score distributions. For each document d, its score is sum( w_i / (k_const + rank_i(d)) ), where rank_i is its 1-based position in list i and k_const (typically 60) dampens the impact of top-ranked outliers. Documents ranked highly by multiple retrievers float to the top."),
    ("When should you choose ContextualCompressionRetriever over standard vector search?",
     "Choose ContextualCompressionRetriever when documents have low information density (e.g. legal contracts, regulatory filings, financial reports where answers are buried inside boilerplate paragraphs). The compressor strips out irrelevant sentences before context is injected into the LLM prompt, reducing token consumption by 50-80% and preventing the model from losing focus."),
    ("How does SelfQueryRetriever translate natural language into structured metadata filters?",
     "SelfQueryRetriever provides an LLM with metadata schema definitions (AttributeInfo objects describing field names, descriptions, and data types) alongside the user query. The LLM parses the query into a structured AST containing a query string and comparison/logical filter expressions (e.g. AND, OR, >, ==). LangChain then compiles this AST into the native query syntax of the specific vector database (e.g. Pinecone metadata syntax or Chroma where clause)."),
    ("What is the primary latency drawback of MultiQueryRetriever in production?",
     "MultiQueryRetriever introduces an initial LLM generation step to create alternative queries, followed by N separate vector searches. This adds substantial round-trip latency (typically 500ms to 1.5 seconds) and triples the embedding query load. In latency-critical customer-facing applications, this trade-off must be weighed against recall benefits.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using ContextualCompression with High-Latency LLMs in High-QPS Endpoints. Invoking an LLM extractor across 10 retrieved chunks requires 10 extraction LLM calls or a heavy multi-chunk prompt. At high traffic, this will exhaust provider rate limits and skyrocket costs. Use EmbeddingsFilter or a local cross-encoder for low-latency compression.")
    st.error("**Trap 2 (Breaking Error)**: Forgetting to Persist the ByteStore in ParentDocumentRetriever. In `ParentDocumentRetriever`, child vectors live in the vector database, but parent documents live in an external `DocStore` (e.g. `InMemoryByteStore`). If your container restarts, the in-memory docstore is wiped, and child vector matches will fail to resolve their parents. Always back the byte store with persistent Redis or disk.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running SelfQueryRetriever on Unstructured Noisy Schemas. If your metadata attributes are not strictly defined with clean types and enumerated values, the LLM will generate invalid query filters that cause vector store queries to fail with 400 Bad Request.")
    st.warning("**Trap 4 (Quality Warning)**: Setting EnsembleRetriever BM25 Weights to 1.0 Without Tokenizer Tuning. BM25 requires tokenization matched to your domain. If technical jargon or code identifiers are split naively, BM25 returns zero matches, degrading hybrid search performance.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: ParentDocumentRetriever"):
        st.markdown("**Embeds small child chunks for vector precision, but returns full parent documents for rich narrative context.**")
    with st.expander("🎴 Flashcard 2: Contextual Compression"):
        st.markdown("**Prunes irrelevant sentences from retrieved chunks using an LLM or embeddings filter to minimize prompt token bloat.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: SelfQueryRetriever"):
        st.markdown("**Uses an LLM to deconstruct a query into natural language text plus structured metadata filter ASTs.**")
    with st.expander("🎴 Flashcard 4: EnsembleRetriever"):
        st.markdown("**Combines lexical (BM25) and semantic (Dense vector) retrievers using Reciprocal Rank Fusion (RRF).**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the mechanics of VectorStoreRetriever with similarity, threshold, and MMR modes.", value=True)
chk2 = st.checkbox("I know how MultiQueryRetriever uses query rewriting to overcome vocabulary mismatch.", value=True)
chk3 = st.checkbox("I can articulate how ParentDocumentRetriever resolves the Chunk Size Dilemma.", value=True)
chk4 = st.checkbox("I understand how ContextualCompressionRetriever prunes prompt token bloat.", value=True)
chk5 = st.checkbox("I understand how SelfQueryRetriever extracts metadata filters from natural language.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 retriever types interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 15.2 Retriever Types!")
