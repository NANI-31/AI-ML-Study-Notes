import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class AdvancedRAGDoc:
    """Represents an enterprise knowledge base record with rich metadata and citation info."""
    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        source: str,
        page: int,
        department: str,
        year: int,
        keywords: List[str]
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.source = source
        self.page = page
        self.department = department
        self.year = year
        self.keywords = [k.lower() for k in keywords]

class MockAdvancedRAGStudio:
    """
    Simulates modern Advanced RAG:
    Query Rewriting -> Hybrid BM25+Dense Retrieval -> Cross-Encoder Reranking -> 
    Metadata Filtering -> Citation-Backed Grounded Generation.
    """
    def __init__(self):
        self.docs: List[AdvancedRAGDoc] = [
            AdvancedRAGDoc(
                doc_id="doc-sec-101",
                title="Zero Trust Network Architecture & mTLS",
                content="Mutual TLS (mTLS) with SPIFFE/SPIRE certificates is strictly mandatory across all internal service meshes. Certificates are rotated every 24 hours automatically by Vault.",
                source="sec_architecture_2026.pdf",
                page=14,
                department="Security",
                year=2026,
                keywords=["mtls", "spiffe", "spire", "zero trust", "certificates", "vault", "security"]
            ),
            AdvancedRAGDoc(
                doc_id="doc-sec-102",
                title="Legacy Perimeter Firewall Rules",
                content="Perimeter firewalls filter ingress HTTP traffic on ports 80 and 443. Internal subnet communications rely on standard VPC routing tables without host-to-host cryptographic validation.",
                source="legacy_network_guide_2020.pdf",
                page=3,
                department="Security",
                year=2020,
                keywords=["firewall", "perimeter", "legacy", "ports", "vpc", "security"]
            ),
            AdvancedRAGDoc(
                doc_id="doc-cloud-201",
                title="Kubernetes Ingress Rate Limiting with Redis",
                content="Edge proxies enforce a distributed sliding window rate limit of 10,000 requests per minute using clustered Redis. Excess requests return HTTP 429 Too Many Requests.",
                source="ingress_traffic_sop_2026.pdf",
                page=8,
                department="CloudOps",
                year=2026,
                keywords=["rate limit", "redis", "sliding window", "ingress", "429", "kubernetes"]
            ),
            AdvancedRAGDoc(
                doc_id="doc-db-301",
                title="PostgreSQL Read Replica Replication",
                content="Asynchronous streaming replication powers up to 15 read replicas per primary cluster. Replication lag threshold is capped at 100 milliseconds before alerting triggers.",
                source="database_tier_runbook_2026.pdf",
                page=22,
                department="DataPlatform",
                year=2026,
                keywords=["postgresql", "streaming replication", "replicas", "lag", "database"]
            )
        ]

    def rewrite_query(self, query: str) -> str:
        q_lower = query.lower()
        if "mtls" in q_lower or "zero trust" in q_lower or "certificate" in q_lower:
            return "What are the cryptographic specifications and rotation periods for mTLS and Zero Trust mesh architecture?"
        elif "rate limit" in q_lower or "redis" in q_lower:
            return "What distributed rate limiting algorithms and quotas govern ingress edge proxy traffic?"
        elif "replica" in q_lower or "postgres" in q_lower:
            return "What are the configuration parameters and lag thresholds for PostgreSQL streaming read replicas?"
        return query

    def execute_advanced_rag(
        self,
        user_query: str,
        enable_rewriting: bool,
        enable_reranking: bool,
        dept_filter: str,
        year_filter: int
    ) -> Dict[str, Any]:
        # 1. Query Rewriting
        effective_q = self.rewrite_query(user_query) if enable_rewriting else user_query

        # 2. Metadata Filtering & Candidate Scoring
        candidates = []
        for d in self.docs:
            if dept_filter != "All" and d.department != dept_filter:
                continue
            if d.year < year_filter:
                continue

            # Hybrid score simulation
            q_tokens = effective_q.lower().split()
            kw_hits = sum(1 for kw in d.keywords if any(kw in t or t in kw for t in q_tokens))
            dense_sim = 0.35 + (0.15 * kw_hits)
            if "sec" in d.doc_id and ("mtls" in effective_q.lower() or "certificate" in effective_q.lower()):
                dense_sim += 0.35

            sim_clamped = min(0.98, max(0.20, dense_sim))
            candidates.append({"doc": d, "score": round(sim_clamped, 4)})

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 3. Cross-Encoder Reranking
        if enable_reranking and candidates:
            for c in candidates:
                # Cross encoder boosts modern 2026 Zero Trust documents over legacy
                if c["doc"].year == 2026 and ("mtls" in effective_q.lower() or "rate" in effective_q.lower() or "postgres" in effective_q.lower()):
                    c["score"] = round(min(0.99, c["score"] + 0.15), 4)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        top_candidates = candidates[:2]

        # 4. Synthesize Answer with Citations
        if not top_candidates:
            ans = "No authoritative documents met the active metadata and relevance filters."
            citations = []
        else:
            primary_doc = top_candidates[0]["doc"]
            ans = (
                f"{primary_doc.content} "
                f"[Source: {primary_doc.source}, Page: {primary_doc.page}, Team: {primary_doc.department}]"
            )
            citations = [
                {
                    "title": c["doc"].title,
                    "source": c["doc"].source,
                    "page": c["doc"].page,
                    "score": c["score"]
                }
                for c in top_candidates
            ]

        return {
            "effective_query": effective_q,
            "top_candidates": top_candidates,
            "citations": citations,
            "generated_answer": ans
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.2 Advanced RAG — LangChain Course",
    page_icon="⚡",
    layout="wide"
)

st.title("16.2 Advanced RAG")
st.caption("Mastering Multi-Query Expansion, Hybrid Ensembles, Neural Reranking, Metadata Filtering, and Citation Grounding")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Multi-Stage Advanced RAG Architecture", divider="rainbow")
st.markdown("""
While Basic RAG suffers from vocabulary mismatch, context loss, and hallucinated attributions, **Advanced RAG** introduces 
pre-retrieval optimization, multi-channel hybrid retrieval, post-retrieval reranking, and citation-grounded generation.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedRAGArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_pre {
            label = "Stage 1: Pre-Retrieval Optimization";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            user_in [label="User Query: 'How often are certs rotated?'", fillcolor="#1E293B", color="#F59E0B"];
            rewrite_step [label="Query Rewriting & Multi-Query Expansion\\nGenerates dense search queries & sub-queries", fillcolor="#0F172A", color="#38BDF8"];
            meta_filter [label="Metadata Filter Extraction\\nExtracts WHERE year >= 2026 AND dept = 'Security'", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_retrieval {
            label = "Stage 2: Hybrid Retrieval & Fusion";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            bm25_channel [label="BM25 Lexical Channel\\nExact acronyms & serial codes", fillcolor="#1E293B", color="#64748B"];
            dense_channel [label="Dense Vector Channel\\nSemantic embeddings", fillcolor="#1E293B", color="#64748B"];
            rrf_fusion [label="Reciprocal Rank Fusion (RRF)\\nCombines sparse + dense rankings", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_post {
            label = "Stage 3: Post-Retrieval Processing & Compression";
            style = "rounded";
            color = "#D97706";
            fontcolor = "#FBBF24";
            rerank_node [label="Cross-Encoder Reranker (Cohere / BGE)\\nRe-scores top 25 candidates via cross-attention", fillcolor="#78350F", color="#F59E0B"];
            comp_node [label="Contextual Compression\\nStrips non-essential sentences", fillcolor="#1E293B", color="#EF4444"];
        }

        subgraph cluster_gen {
            label = "Stage 4: Grounded Generation & Citation Verification";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            ground_prompt [label="Strict Grounding Prompt\\nInjects formatted context with document IDs & page numbers", fillcolor="#064E3B", color="#10B981"];
            cite_out [label="Verified Answer with Structured Citations\\n[Source: sec_arch.pdf, Page 14]", fillcolor="#022C22", color="#059669"];
        }

        user_in -> rewrite_step;
        user_in -> meta_filter;
        rewrite_step -> bm25_channel [label="text query"];
        rewrite_step -> dense_channel [label="vector query"];
        meta_filter -> bm25_channel [label="apply constraints"];
        meta_filter -> dense_channel [label="apply constraints"];

        bm25_channel -> rrf_fusion;
        dense_channel -> rrf_fusion;
        
        rrf_fusion -> rerank_node [label="top-25 broad pool"];
        rerank_node -> comp_node [label="top-5 high precision"];
        
        comp_node -> ground_prompt [label="clean context"];
        ground_prompt -> cite_out [label="LLM synthesis"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: QUERY REWRITING ---
st.header("2. Query Rewriting", divider="rainbow")
st.markdown("""
Real-world users write colloquial, abbreviated, or poorly structured questions. **Query Rewriting** uses an LLM pre-step 
to rephrase the input into an information-dense query optimized for index matching.
""")

col_rw1, col_rw2 = st.columns(2)
with col_rw1:
    st.markdown("""
    ### Key Rewriting Patterns:
    1. **Conversational De-contextualization**: Resolving pronouns (*"What about its SLA?"* -> *"What is Aurora PostgreSQL's uptime SLA?"*).
    2. **Hypothetical Document Embeddings (HyDE)**: Synthesizing a hypothetical passage to land near real answer clusters.
    3. **Step-Back Prompting**: Extracting high-level concepts for foundational retrieval.
    """)
with col_rw2:
    st.code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a user inquiry, reformulate it into an authoritative search query for enterprise technical documentation."),
    ("human", "{question}")
])

rewrite_chain = rephrase_prompt | chat_model | StrOutputParser()
    """, language="python")

st.divider()

# --- SECTION 3: MULTI-QUERY RETRIEVAL ---
st.header("3. Multi-Query Retrieval", divider="rainbow")
st.markdown("""
A single query perspective often fails to span domain vocabulary divergence. The **Multi-Query Retriever** generates 
multiple alternative phrasings and queries the vector index for each variation, taking the unique union.
""")

st.code("""
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=chat_model
)

# Generates 3-5 alternative queries, runs searches in parallel, and deduplicates chunks
docs = multi_query_retriever.invoke("How are certificates refreshed?")
""", language="python")

st.divider()

# --- SECTION 4: HYBRID RETRIEVAL ---
st.header("4. Hybrid Retrieval (BM25 + Dense Vectors)", divider="rainbow")
st.markdown("""
Hybrid retrieval merges sparse lexical matching (BM25 for exact terms and IDs) with dense vector representations 
(bi-encoders for conceptual intent) using **Reciprocal Rank Fusion (RRF)**.
""")

st.code("""
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Sparse Lexical Retriever
bm25 = BM25Retriever.from_documents(corpus_docs)
bm25.k = 10

# Dense Vector Retriever
dense = vectorstore.as_retriever(search_kwargs={"k": 10})

# Ensemble Fusion (RRF with 50/50 weighting)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.5, 0.5]
)
""", language="python")

st.divider()

# --- SECTION 5: RERANKING ---
st.header("5. Cross-Encoder Reranking", divider="rainbow")
st.markdown("""
Bi-encoders independently compress queries and documents into isolated vectors, discarding word-to-word cross-attention. 
A **Cross-Encoder Reranker** processes candidate `(query, document)` pairs through full cross-attention layers, 
dramatically improving precision at the top of the context window.
""")

st.code("""
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-v3.5",
    top_n=3,
    score_threshold=0.60
)

reranked_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)
""", language="python")

st.divider()

# --- SECTION 6: CONTEXT COMPRESSION ---
st.header("6. Context Compression", divider="rainbow")
st.markdown("""
Retrieved chunks often contain large blocks of irrelevant text surrounding the target fact. 
**Context Compression** extracts only relevant sentences before prompt assembly, reducing prompt token bloat by 50-80%.
""")

st.code("""
from langchain.retrievers.document_compressors import LLMChainExtractor

extractor = LLMChainExtractor.from_llm(chat_model)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=extractor,
    base_retriever=reranked_retriever
)
""", language="python")

st.divider()

# --- SECTION 7: METADATA FILTERING ---
st.header("7. Metadata Filtering", divider="rainbow")
st.markdown("""
In multi-tenant or regulated environments, security policies require hard metadata constraints. Pre-filtering ensures 
candidates from unauthorized departments or deprecated document years are rejected before similarity scoring occurs.
""")

st.code("""
# Hard pre-filtering via search_kwargs
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {
            "department": "Security",
            "year": 2026
        }
    }
)
""", language="python")

st.divider()

# --- SECTION 8: SOURCE CITATIONS ---
st.header("8. Source Citations & Provenance Tracking", divider="rainbow")
st.markdown("""
Trustworthy enterprise RAG must provide **verifiable citations** linking every assertion back to its authoritative document, 
page number, and authoring team.
""")

st.code("""
def format_docs_with_citations(docs):
    formatted = []
    for d in docs:
        source_tag = f"[{d.metadata.get('source', 'Unknown')}, Page {d.metadata.get('page', 'N/A')}]"
        formatted.append(f"{source_tag}:\\n{d.page_content}")
    return "\\n\\n".join(formatted)
""", language="python")

st.divider()

# --- SECTION 9: ANSWER GROUNDING ---
st.header("9. Answer Grounding & Hallucination Suppression", divider="rainbow")
st.markdown("""
**Answer Grounding** commands the LLM to write answers whose claims are directly verifiable from the provided context chunks.
""")

st.code("""
system_grounding_prompt = (
    "You are an enterprise AI assistant. Your responses must be 100% grounded in the "
    "provided context. For every factual statement, append the source citation tag in brackets, "
    "e.g. [source.pdf, Page 12]. If the context does not contain sufficient facts to answer, "
    "explicitly refuse to answer. Never speculate."
)
""", language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Advanced RAG Pipeline", divider="rainbow")
st.markdown("""
Below is a full production-ready implementation assembling query rewriting, hybrid search, cross-encoder reranking, 
and citation-backed generation into an integrated LCEL chain.
""")

st.code("""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def build_advanced_enterprise_rag(
    documents: List[Document],
    openai_key: str,
    cohere_key: str
):
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)

    # 1. First Stage Hybrid
    vectorstore = FAISS.from_documents(documents, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 20

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 2. Second Stage Cross-Encoder Reranker
    reranker = CohereRerank(
        cohere_api_key=cohere_key,
        model="rerank-v3.5",
        top_n=3,
        score_threshold=0.60
    )
    pipeline_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble
    )

    # 3. Citation-Aware Context Formatter
    def format_with_citations(docs):
        return "\\n\\n".join([
            f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]: {d.page_content}"
            for d in docs
        ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly using context. Append inline citations [Source, Page] for every claim:\\n\\n{context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": pipeline_retriever | format_with_citations, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Advanced RAG System Simulator", divider="rainbow")
st.markdown("""
Test how **Query Rewriting**, **Cross-Encoder Reranking**, and **Metadata Pre-Filtering** impact answer grounding 
and citation provenance in real time.
""")

# Module-scope simulator instantiation
adv_rag_studio = MockAdvancedRAGStudio()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    user_q_input = st.text_input(
        "Enter Inquiry Query:",
        value="How often are mTLS certificates rotated across the mesh?"
    )

    enable_rw = st.toggle("Enable Query Rewriting (Pre-retrieval)", value=True)
    enable_rr = st.toggle("Enable Cross-Encoder Reranker (Post-retrieval)", value=True)

    dept_filter_choice = st.selectbox("Metadata Filter: Department", ["All", "Security", "CloudOps", "DataPlatform"], index=0)
    year_filter_choice = st.slider("Metadata Filter: Minimum Document Year", min_value=2020, max_value=2026, value=2024)

    run_adv_rag_btn = st.button("⚡ Execute Advanced RAG Funnel", type="primary", use_container_width=True)

with wb_c2:
    st.subheader("Grounded Output & Provenance")

    if run_adv_rag_btn or user_q_input:
        res = adv_rag_studio.execute_advanced_rag(
            user_query=user_q_input,
            enable_rewriting=enable_rw,
            enable_reranking=enable_rr,
            dept_filter=dept_filter_choice,
            year_filter=year_filter_choice
        )

        st.info(f"**Effective Processed Query:**\\n`{res['effective_query']}`")

        st.markdown("### 🤖 Synthesized Grounded Answer")
        st.success(res["generated_answer"])

        st.markdown("### 📚 Authoritative Citations")
        if res["citations"]:
            for idx, c in enumerate(res["citations"], 1):
                st.markdown(f"**Citation #{idx}**: `{c['title']}` — *{c['source']} (Page {c['page']})* [Relevance Score: `{c['score']}`]")
        else:
            st.warning("No citations available.")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is two-stage retrieval (Hybrid + Reranker) necessary in Advanced RAG?",
     "Single-stage vector search suffers from a trade-off: broad candidate pools capture more relevant chunks but dilute precision, while narrow top-k misses nuanced answers. Two-stage retrieval resolves this: stage 1 casts a wide net (top 20-50 candidates) using hybrid BM25 and dense search, while stage 2 uses an expensive cross-encoder to evaluate full cross-attention over query-document pairs, boosting true relevant chunks to top-3 positions."),
    ("How does metadata pre-filtering prevent data leakage in multi-tenant RAG?",
     "Pre-filtering applies boolean constraints directly within the vector index before ANN distance computation. If tenant_id = 'tenant-A' is enforced as a pre-filter, vectors belonging to other tenants are completely excluded from graph traversal, guaranteeing that no unauthorized data can be retrieved or leaked into prompts."),
    ("What is the difference between Contextual Compression and standard chunking?",
     "Standard chunking chops documents into static token blocks (e.g. 500 tokens). Contextual compression operates dynamically at query time: it takes retrieved 500-token chunks and uses an extractor model to strip away boilerplate sentences, returning only the specific sentences that directly answer the query, reducing prompt token costs and model distraction."),
    ("Why is Reciprocal Rank Fusion (RRF) preferred over linear score combination in hybrid search?",
     "BM25 produces unbounded scores (0 to 30+) while dense cosine similarity produces scores bounded between -1 and 1. Normalizing these disparate scales with min-max formulas is brittle and sensitive to outliers. RRF operates purely on relative rank order: score = sum( 1 / (60 + rank) ), making it completely invariant to score magnitudes."),
    ("How do source citations help detect and mitigate hallucinations in production?",
     "By instructing the model to provide explicit citation anchors [Source, Page] for every factual assertion, human reviewers and automated evaluation agents can verify whether each claim is supported by the cited chunk. If an assertion lacks a valid citation or cites an irrelevant document, the response can be flagged as ungrounded.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Post-Filtering for Multi-Tenant Data Isolation. Performing vector similarity search first and filtering by tenant metadata afterward will leak tenant info or return zero results if the top-k vectors belong to other tenants. Always enforce hard pre-filtering.")
    st.error("**Trap 2 (Breaking Error)**: Cross-Encoder Token Truncation Errors. Cross-encoders have strict context window limits (typically 512 tokens for query + document combined). Passing 2,000-token chunks to a cross-encoder results in silent truncation or API exceptions. Always compress or slice chunks before reranking.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running Query Rewriting on Trivial Queries. Invoking an LLM to rephrase every single user query adds 500ms of latency and API costs. Use lightweight intent classifiers to skip rewriting on well-formed queries.")
    st.warning("**Trap 4 (Quality Warning)**: Hallucinated Citations in Weak LLMs. Smaller models often invent fictional page numbers or cite documents not present in the prompt context. Enforce strict JSON output parsing schemas for citations.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Advanced RAG Funnel"):
        st.markdown("**Pre-retrieval rewriting -> Hybrid retrieval -> Cross-encoder reranking -> Grounded generation.**")
    with st.expander("🎴 Flashcard 2: Cross-Encoder Reranker"):
        st.markdown("**Evaluates joint query-document cross-attention to elevate high-precision ground truth to the top.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Metadata Pre-Filtering"):
        st.markdown("**Applies metadata constraints before ANN search to guarantee multi-tenant security and compliance.**")
    with st.expander("🎴 Flashcard 4: Answer Grounding"):
        st.markdown("**Enforcing that every assertion in the LLM response is supported by an explicit source citation.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between Basic RAG and Advanced RAG.", value=True)
chk2 = st.checkbox("I know how Query Rewriting and Multi-Query Expansion resolve vocabulary divergence.", value=True)
chk3 = st.checkbox("I can explain how Hybrid Search and RRF merge sparse and dense candidate pools.", value=True)
chk4 = st.checkbox("I understand the mechanics of Cross-Encoder Reranking and Contextual Compression.", value=True)
chk5 = st.checkbox("I can implement citation provenance tracking and answer grounding in an LCEL pipeline.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 advanced RAG interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 16.2 Advanced RAG!")
