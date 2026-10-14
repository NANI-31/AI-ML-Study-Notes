import streamlit as st
import json

st.set_page_config(
    page_title="25.2 Supporting Technologies - Prompt Engineering",
    page_icon="🧩",
    layout="wide"
)

# Header Section
st.title("25.2 Supporting Technologies: Vector DBs, Embeddings, Rerankers, & Observability")
st.caption("Chapter 25: Prompt Engineering Frameworks and Ecosystem | Domain: 1. Prompt Engineering")
st.markdown("""
A prompt is only as effective as the surrounding infrastructure feeding it context and observing its outputs. 
Production generative AI architectures rely on an interconnected ecosystem of **Vector Databases, Dense Embedding Models, 
Cross-Encoder Rerankers, Automated Evaluation Frameworks, and Open Observability Platforms**.
""")

st.divider()

# Section 1: The Supporting Stack Overview
st.header("1. The Enterprise Generative AI Supporting Stack", divider="rainbow")
st.markdown("""
An enterprise LLM application integrates five distinct supporting technology layers around the core prompt:
""")

st.graphviz_chart('''
digraph SupportingStack {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Obs [label="5. Observability & Tracing Layer\\n(Langfuse, LangSmith, Arize Phoenix, Helicone)", fillcolor="#FFCDD2"];
    Eval [label="4. Evaluation & Benchmarking Layer\\n(Ragas, TruLens, DeepEval, Promptfoo)", fillcolor="#FFE0B2"];
    Rerank [label="3. Neural Reranking Layer\\n(Cohere Rerank 3, BGE-Reranker-Large)", fillcolor="#FFF9C4"];
    VDB [label="2. Vector Storage Layer\\n(Pinecone, Qdrant, ChromaDB, Weaviate, pgvector)", fillcolor="#C8E6C9"];
    Embed [label="1. Dense Embeddings Layer\\n(OpenAI text-embedding-3, Voyage-3, BGE-M3)", fillcolor="#BBDEFB"];
    
    Obs -> Eval -> Rerank -> VDB -> Embed;
}
''', use_container_width=False)

st.divider()

# Section 2: Deep-Dive into Stack Layers
st.header("2. Detailed Layer Architecture & Trade-Offs", divider="rainbow")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Vector Databases")
    st.markdown("""
    Specialized indices designed for sub-10ms Approximate Nearest Neighbor (ANN) search across high-dimensional vector spaces (HNSW, IVF):
    * **ChromaDB**: Lightweight, local in-memory/embedded vector store. Perfect for local prototyping and rapid dev.
    * **Pinecone**: Fully managed, serverless cloud vector database. High scalability with zero cluster maintenance.
    * **Qdrant**: High-performance Rust-based vector database with exceptional payload filtering and hybrid search.
    * **pgvector**: PostgreSQL extension adding vector indexing. Ideal for teams wanting to retain existing Postgres infrastructure.
    """)

    st.subheader("2. Embedding & Reranking Models")
    st.markdown("""
    * **Embedding Models (Bi-Encoders)**: Compress text into dense numerical representations (e.g., 1536-dim or 3072-dim vectors). Fast for searching millions of items, but loses subtle cross-query interactions.
    * **Rerankers (Cross-Encoders)**: Feed query and candidate chunk jointly into full multi-head attention. High precision; reranks top 50 retrieved chunks down to top 5 with exceptional accuracy.
    """)

with col2:
    st.subheader("3. Evaluation Frameworks")
    st.markdown("""
    Automate quality regression testing in CI/CD pipelines:
    * **Ragas**: Standard for RAG metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall).
    * **DeepEval**: Unit testing framework for LLMs with G-Eval and custom rule assert statements.
    * **Promptfoo**: Command-line CI/CD tool to run side-by-side prompt diffs against hundreds of test cases.
    """)

    st.subheader("4. Observability & Tracing Platforms")
    st.markdown("""
    Provide real-time visibility into production inference clusters:
    * **Langfuse**: Open-source, self-hostable LLM observability platform tracking traces, latency, costs, and user feedback.
    * **LangSmith**: Native observability suite tightly coupled with LangChain and LangGraph deployments.
    * **Arize Phoenix**: OpenTelemetry-based tracing with automated clustering of user drift and failure modes.
    """)

st.divider()

# Section 3: Two-Stage Retrieval Pattern
st.header("3. Two-Stage Retrieval Architecture Pattern (Bi-Encoder + Cross-Encoder)", divider="rainbow")
st.markdown("""
The gold standard for production RAG pipelines combines fast vector search with high-precision neural reranking:
""")

st.code('''# Two-Stage Retrieval Implementation
from typing import List

def two_stage_rag_retrieve(query: str, vector_store, reranker, top_k=5) -> List[str]:
    # Stage 1: Fast ANN Search (Bi-Encoder)
    # Retrieve broad candidate pool of 30 candidates
    candidates = vector_store.similarity_search(query, k=30)
    
    # Stage 2: Deep Cross-Attention Reranking (Cross-Encoder)
    # Computes joint attention between query and each candidate chunk
    reranked_docs = reranker.rerank(
        query=query,
        documents=[doc.page_content for doc in candidates],
        top_n=top_k
    )
    
    # Return highest quality top-k chunks
    return [doc.document for doc in reranked_docs]''', language="python")

st.divider()

# Section 4: Interactive Stack Configurator
st.header("4. Interactive Supporting Stack Configurator", divider="rainbow")
st.markdown("Configure your application constraints to receive a tailored supporting tech stack.")

col_c1, col_c2 = st.columns(2)

with col_c1:
    infra_env = st.selectbox("Infrastructure Environment:", ["Cloud Serverless (AWS/GCP)", "On-Premises / Air-Gapped", "Existing Enterprise Postgres"])
    scale_docs = st.selectbox("Document Index Scale:", ["Small (< 100k chunks)", "Medium (100k - 5M chunks)", "Massive (10M+ chunks)"])
    budget_priority = st.selectbox("Engineering Priority:", ["Maximum Retrieval Precision", "Lowest Latency (< 100ms)", "Minimal Cost & Complexity"])

with col_c2:
    st.subheader("Recommended Stack Specification")
    if "Postgres" in infra_env:
        rec_vdb = "PostgreSQL + pgvector"
    elif "On-Premises" in infra_env:
        rec_vdb = "Qdrant (Self-Hosted Docker) / ChromaDB"
    else:
        rec_vdb = "Pinecone Serverless / Qdrant Cloud"
        
    rec_embed = "text-embedding-3-small" if "Cost" in budget_priority else "Cohere embed-v3 / text-embedding-3-large"
    rec_rerank = "Cohere Rerank 3.5 / BGE-Reranker-Large" if "Precision" in budget_priority else "None (Single-Stage to preserve latency)"
    rec_obs = "Langfuse (Open-Source)" if "On-Premises" in infra_env else "Langfuse / LangSmith Cloud"
    
    st.markdown(f"""
    * **Vector Storage**: `{rec_vdb}`
    * **Embedding Model**: `{rec_embed}`
    * **Reranking Layer**: `{rec_rerank}`
    * **Observability Suite**: `{rec_obs}`
    """)
    st.success("Configuration optimized for your infrastructure and latency requirements.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: Why is a two-stage retrieval (Vector DB + Cross-Encoder) superior to pure vector search?"):
        st.write("""
        * **Bi-Encoders (Embeddings)** encode the query and document independently into fixed vectors. 
        While ultra-fast ($O(1)$ lookup via ANN indexes), they lose token-level cross-interaction nuances, leading to false positives.
        * **Cross-Encoders (Rerankers)** pass the query and candidate chunk jointly through full multi-head attention heads, 
        evaluating every token interaction directly. Running a cross-encoder on millions of docs is too slow ($O(N)$), 
        but reranking the top 30 candidates takes only ~20ms, providing near-human relevance filtering.
        """)

    with st.expander("Q: What is the primary advantage of OpenTelemetry-based tracing in LLM observability?"):
        st.write("""
        OpenTelemetry (OTel) provides vendor-neutral instrumentation standards. 
        By using OTel spans, your application is not locked into a single proprietary dashboard. 
        You can emit spans from your prompt chains and export them interchangeably to Datadog, Langfuse, Arize Phoenix, 
        or Honeycomb without changing a single line of business logic.
        """)

with qa2:
    with st.expander("Trap to Avoid: Reranking Unfiltered Haystacks"):
        st.write("""
        Attempting to pass 200+ chunks to a reranker causes severe latency degradation (adding 500ms+ to request time) 
        and high API costs. Always configure your vector database with strict metadata filters (e.g. `tenant_id == X` and `doc_type == 'policy'`) 
        to narrow candidate pools down to 20-40 chunks before invoking the neural reranker.
        """)

    with st.expander("Flashcards: Supporting Technologies"):
        fc_sel = st.selectbox(
            "Select Technology Flashcard:",
            ["HNSW (Hierarchical Navigable Small World)", "Cross-Encoder", "OpenTelemetry Spans", "RAGAS Faithfulness"]
        )
        if fc_sel == "HNSW (Hierarchical Navigable Small World)":
            st.info("The premier graph-based index algorithm used in vector DBs for logarithmic-time nearest neighbor search.")
        elif fc_sel == "Cross-Encoder":
            st.info("A neural model that computes joint attention between query and document text for high-precision reranking.")
        elif fc_sel == "OpenTelemetry Spans":
            st.info("Standardized telemetry packets capturing execution start, finish, latency, and attributes across pipeline nodes.")
        else:
            st.info("An automated evaluation metric measuring what fraction of generated claims are grounded in retrieved context.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know when to select ChromaDB vs Pinecone vs pgvector.")
    st.checkbox("I understand the architectural distinction between Bi-Encoders and Cross-Encoders.")
with c2:
    st.checkbox("I can implement a two-stage retrieval pipeline with neural reranking.")
    st.checkbox("I understand the role of observability tools (Langfuse/LangSmith) in tracking latency and costs.")
