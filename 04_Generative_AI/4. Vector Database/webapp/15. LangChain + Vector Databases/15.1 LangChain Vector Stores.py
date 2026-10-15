import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="15.1 LangChain Vector Stores - Vector DB Curriculum",
    page_icon="🦜",
    layout="wide"
)

# Title & Overview
st.title("15.1 LangChain Vector Stores: The Universal VectorStore Interface & Chroma Integration")
st.caption("Chapter 15: LangChain + Vector Databases | Domain: 4. Vector Database")
st.markdown("""
In modern enterprise AI stacks, developers rarely interact with raw vector database client drivers directly. 
Instead, they build upon framework abstraction layers like **LangChain** to achieve modularity, vendor independence, 
and seamless integration with LangChain Expression Language (LCEL) chains and autonomous agents.

At the center of LangChain's data layer sits the **`VectorStore`** abstract base class:
* A standardized interface implemented across 50+ vector backends (Chroma, Pinecone, Qdrant, Milvus, FAISS).
* Provides unified methods: `.from_documents()`, `.add_texts()`, `.similarity_search()`, and **`.as_retriever()`**.
* Decouples retrieval logic from underlying database drivers, allowing teams to migrate from local embedded ChromaDB 
  to distributed cloud clusters with zero application code changes.

In this lesson, we dissect LangChain's `VectorStore` architecture, master production `langchain-chroma` integration patterns, 
understand how to convert stores into LCEL-ready **Retrievers**, and test an interactive LangChain Pipeline Simulator.
""")

st.divider()

# Section 1: The LangChain VectorStore Abstraction Layer
st.header("1. The LangChain VectorStore Abstraction Layer", divider="rainbow")

col_abs1, col_abs2 = col_abs1, col_abs2 = st.columns([1.1, 0.9])

with col_abs1:
    st.markdown("""
    ### Why Framework Abstractions Matter
    Writing custom glue code directly against individual vector database APIs leads to vendor lock-in. 
    LangChain solves this with the `VectorStore` protocol:

    1. **Universal Ingestion Interface**:
       * `.add_documents(documents: List[Document], ids: List[str])`: Standardized ingestion taking `Document(page_content=..., metadata=...)` objects.
       * `.from_documents(...)`: Factory constructor that handles embedding model instantiation and collection creation in a single call.
    2. **Universal Search Interface**:
       * `.similarity_search(query, k)`: Returns raw `List[Document]`.
       * `.similarity_search_with_score(query, k)`: Returns tuples of `(Document, distance_float)`.
    3. **The `.as_retriever()` Bridge**:
       * Converts any vector store into a first-class **`Runnable`**, enabling native piping in LCEL chains:
         `retriever | prompt | llm | StrOutputParser()`.
    """)

with col_abs2:
    st.markdown("### LangChain VectorStore Architecture")
    lc_dot = """
    digraph LangChainVectorStore {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Base [label="langchain_core.vectorstores.VectorStore\\n(Abstract Base Class)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_backends {
            label="Unified Backend Implementations"; style="dashed"; color="#059669";
            ChromaImpl [label="langchain_chroma.Chroma\\n(Embedded / Persistent)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            PineconeImpl [label="langchain_pinecone.Pinecone\\n(Managed Cloud)", fillcolor="#FEF3C7"];
            QdrantImpl [label="langchain_qdrant.Qdrant\\n(Hybrid Filtered)", fillcolor="#E0E7FF"];
        }

        Retriever [label="VectorStoreRetriever\\n(Runnable Interface for LCEL)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Base -> ChromaImpl;
        Base -> PineconeImpl;
        Base -> QdrantImpl;
        ChromaImpl -> Retriever [label=".as_retriever()"];
    }
    """
    st.graphviz_chart(lc_dot, use_container_width=False)

st.divider()

# Section 2: Production Chroma Integration Code Reference
st.header("2. Production Chroma Integration Code Reference", divider="rainbow")

st.markdown("""
In modern LangChain, Chroma is maintained in the dedicated package **`langchain-chroma`** (replacing the legacy `langchain-community` integration):
""")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. Ingestion via `from_documents`")
    st.code("""from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Sample LangChain Documents
docs = [
    Document(
        page_content="Envoy mTLS requires mutual TLS certs in /etc/certs.",
        metadata={"source": "envoy.md", "dept": "Security"}
    ),
    Document(
        page_content="PostgreSQL WAL archives flush to S3 every 5 minutes.",
        metadata={"source": "db_sla.pdf", "dept": "Infra"}
    )
]

# 3. Create persistent Chroma VectorStore
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="production_kb",
    persist_directory="./langchain_chroma_db"
)
""", language="python")

with col_code2:
    st.markdown("### 2. Attaching to Existing Store & Querying")
    st.code("""# Connect to existing on-disk Chroma collection
vectorstore = Chroma(
    collection_name="production_kb",
    embedding_function=embeddings,
    persist_directory="./langchain_chroma_db"
)

# Search with metadata filtering
results = vectorstore.similarity_search_with_score(
    query="How does Envoy authenticate?",
    k=2,
    filter={"dept": "Security"}
)

for doc, score in results:
    print(f"Score: {score:.4f} | Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}\\n")
""", language="python")

st.divider()

# Section 3: The `.as_retriever()` API & LCEL Chaining
st.header("3. The `.as_retriever()` API & LCEL Chaining", divider="rainbow")

st.markdown("""
The true power of LangChain vector stores emerges when converting them into a **`VectorStoreRetriever`**. 
This allows piping the store directly into the **LangChain Expression Language (LCEL)**:
""")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### Configuring Retriever Search Types
    * **Standard Similarity (`search_type="similarity"`)**:
      ```python
      retriever = vectorstore.as_retriever(
          search_kwargs={"k": 3}
      )
      ```
    * **Maximal Marginal Relevance (`search_type="mmr"`)**:
      ```python
      # Balances relevance with topic diversity
      retriever = vectorstore.as_retriever(
          search_type="mmr",
          search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
      )
      ```
    * **Similarity Score Threshold (`search_type="similarity_score_threshold"`)**:
      ```python
      # Drops any chunk whose similarity is below threshold
      retriever = vectorstore.as_retriever(
          search_type="similarity_score_threshold",
          search_kwargs={"score_threshold": 0.75, "k": 5}
      )
      ```
    """)

with col_pipe2:
    st.markdown("### The Modern LCEL RAG Pipeline")
    st.code("""from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(\"\"\"
Answer the question based strictly on the context below:
<context>
{context}
</context>

Question: {question}
\"\"\")

def format_docs(docs):
    return "\\n\\n".join(d.page_content for d in docs)

# Declarative LCEL RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Execute Chain
response = rag_chain.invoke("What are our mTLS requirements?")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: LangChain Chroma Explorer & LCEL Simulator
st.header("4. Interactive Workbench: LangChain Chroma Explorer & LCEL Simulator", divider="rainbow")

st.markdown("""
Configure retriever parameters dynamically and observe how LangChain executes similarity search, 
scores candidates, and constructs the augmented context payload.
""")

col_lc_cfg, col_lc_query = st.columns([1, 1.2])

with col_lc_cfg:
    st.subheader("1. Configure Retriever Parameters")
    
    ret_type = st.selectbox(
        "Retriever Search Type (`search_type`):",
        ["similarity", "mmr (Maximal Marginal Relevance)", "similarity_score_threshold"]
    )
    
    ret_k = st.slider("Number of Chunks to Retrieve (`k`):", min_value=1, max_value=5, value=2)
    
    if "mmr" in ret_type:
        lambda_val = st.slider("MMR Diversity Penalty (`lambda_mult`):", min_value=0.1, max_value=0.9, value=0.7, step=0.1)
    elif "threshold" in ret_type:
        min_thresh = st.slider("Minimum Similarity Threshold (`score_threshold`):", min_value=0.5, max_value=0.95, value=0.75, step=0.05)

    apply_filter = st.checkbox("Apply Metadata Filter (`dept == 'Security'`)", value=True)

with col_lc_query:
    st.subheader("2. Test User Query")
    
    query_text = st.selectbox(
        "User Question:",
        [
            "How is mutual TLS configured in Envoy proxy?",
            "What is our PostgreSQL WAL archive frequency?",
            "What are our general corporate expense reimbursement limits?"
        ]
    )

# Mock LangChain Knowledge Corpus
MOCK_DOCS = [
    {
        "content": "Envoy proxy mTLS requires mounting client certificates in /etc/envoy/certs and enforcing require_client_certificate: true.",
        "meta": {"source": "envoy_security_v2.md", "dept": "Security", "author": "Alice"},
        "sim_score": 0.92
    },
    {
        "content": "Automated certificate rotation for Envoy listeners uses SPIFFE/SPIRE agents communicating over local UNIX domain sockets.",
        "meta": {"source": "spiffe_mesh.pdf", "dept": "Security", "author": "Bob"},
        "sim_score": 0.84
    },
    {
        "content": "PostgreSQL WAL archiving streams transaction logs continuously to S3 with Point-In-Time Recovery (PITR) enabled for 35 days.",
        "meta": {"source": "database_runbook.md", "dept": "Infra", "author": "Charlie"},
        "sim_score": 0.88
    },
    {
        "content": "Daily business travel meal stipends are capped at $85 per day. Alcoholic beverages are non-reimbursable.",
        "meta": {"source": "hr_travel_2024.docx", "dept": "HR", "author": "Dana"},
        "sim_score": 0.79
    }
]

# Filtering logic
filtered_candidates = [
    d for d in MOCK_DOCS
    if not apply_filter or d["meta"]["dept"] == "Security"
]

# Adjust scores based on query
if "WAL" in query_text:
    for d in filtered_candidates:
        if d["meta"]["dept"] == "Infra": d["sim_score"] = 0.95
        else: d["sim_score"] = 0.35
elif "expense" in query_text:
    for d in filtered_candidates:
        if d["meta"]["dept"] == "HR": d["sim_score"] = 0.94
        else: d["sim_score"] = 0.30

# Threshold filter
if "threshold" in ret_type:
    filtered_candidates = [d for d in filtered_candidates if d["sim_score"] >= min_thresh]

# Sort by score
filtered_candidates.sort(key=lambda x: x["sim_score"], reverse=True)
final_docs = filtered_candidates[:ret_k]

st.divider()

col_out_docs, col_out_lcel = st.columns([1.2, 0.8])

with col_out_docs:
    st.subheader(f"Retriever Output (`List[Document]`, {len(final_docs)} returned)")
    
    if final_docs:
        for idx, doc in enumerate(final_docs):
            with st.expander(f"📄 Document {idx+1}: {doc['meta']['source']} (Sim Score: {doc['sim_score']:.2f})", expanded=True):
                st.markdown(f"**page_content**: *\"{doc['content']}\"*")
                st.caption(f"metadata: `{doc['meta']}`")
    else:
        st.warning("⚠️ No documents matched the active query and metadata filter criteria.")

with col_out_lcel:
    st.subheader("Generated LCEL Execution Code")
    st.code(f"""# Configured in LangChain
retriever = vectorstore.as_retriever(
    search_type="{ret_type.split()[0]}",
    search_kwargs={{
        "k": {ret_k},
        {"'filter': {'dept': 'Security'}," if apply_filter else ""}
    }}
)

# Run LCEL Invocation
retrieved_docs = retriever.invoke("{query_text[:35]}...")
""", language="python")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the architectural difference between a VectorStore and a Retriever in LangChain?"):
    st.markdown("""
    **Answer**:
    * **VectorStore**: A low-level storage abstraction wrapping an indexed vector database. It directly manages vector embeddings, persistence, and CRUD mutations (`add_documents`, `delete`). It requires passing embedding arrays or text strings directly.
    * **Retriever**: A higher-level, lightweight interface defined in `langchain_core.retrievers`. It implements the **`Runnable`** protocol (`invoke(query: str) -> List[Document]`). 
    A Retriever does NOT store documents; it simply defines *how* to fetch them. A VectorStore can be converted into a Retriever via `.as_retriever()`, but Retrievers can also wrap web search engines, BM25 indices, or hybrid multi-hop graphs.
    """)

with st.expander("Q2: Why was `langchain-community` split into partner packages like `langchain-chroma`?"):
    st.markdown("""
    **Answer**:
    In older versions of LangChain (prior to v0.1), all third-party integrations lived in the monolithic `langchain-community` repository. 
    This caused severe dependency bloat, slow installation times, and version conflicts (e.g. updating ChromaDB broke Pinecone dependencies).
    
    **The Partner Package Split**:
    LangChain extracted core database providers into lightweight, independently versioned packages (e.g., `langchain-chroma`, `langchain-openai`, `langchain-pinecone`). 
    This ensures that breaking changes in ChromaDB only impact `langchain-chroma` without affecting the rest of the application ecosystem.
    """)

with st.expander("Q3: How does LangChain Expression Language (LCEL) handle streaming from a Retriever-based chain?"):
    st.markdown("""
    **Answer**:
    In an LCEL chain (`rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)`):
    1. The **Retrieval stage** runs synchronously or asynchronously to completion, fetching the top-K documents and formatting the prompt.
    2. Once the prompt is assembled, the **LLM stage** natively streams tokens to the client using `rag_chain.stream()`.
    3. LCEL automatically manages the asynchronous streaming yield, delivering the first generated token in milliseconds without waiting for the entire LLM response to complete.
    """)

with st.expander("Q4: What is the difference between `.similarity_search()` and `.similarity_search_with_score()` in LangChain?"):
    st.markdown("""
    **Answer**:
    * `.similarity_search(query, k)` returns only a `List[Document]`, discarding the raw mathematical distance values returned by the underlying vector engine.
    * `.similarity_search_with_score(query, k)` returns a list of tuples: `List[Tuple[Document, float]]`.
    * **Crucial Caveat**: The score metric depends entirely on the underlying database! In Chroma with Cosine distance, lower scores indicate higher similarity (`distance = 0.0` is an exact match). In other vector stores using inner product, higher scores indicate higher similarity. Always inspect database documentation before filtering by score.
    """)

with st.expander("Q5: How do you implement dynamic metadata filtering in an LCEL chain where the filter depends on the logged-in user?"):
    st.markdown("""
    **Answer**:
    In production multi-tenant applications, the metadata filter (`tenant_id`) cannot be hardcoded in `.as_retriever()`.
    
    Use a **Configurable Runnable** or dynamic input dictionary:
    ```python
    from langchain_core.runnables import chain

    @chain
    def custom_retriever(input_dict):
        question = input_dict["question"]
        user_tenant = input_dict["tenant_id"]
        
        return vectorstore.similarity_search(
            query=question,
            k=3,
            filter={"tenant_id": user_tenant}
        )

    # Integrated LCEL Pipeline:
    chain = custom_retriever | prompt | llm
    chain.invoke({"question": "Show my docs", "tenant_id": "tenant_42"})
    ```
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Importing from `langchain.vectorstores.chroma`**  
    Importing Chroma from legacy paths (`langchain.vectorstores`) is deprecated and imports unmaintained code. Always import from `langchain_chroma import Chroma`.
    """)
    st.warning("""
    **Trap 2: Assuming Similarity Scores Are Uniform Across Backends**  
    Assuming `score > 0.8` means a good match across all vector stores breaks when migrating from FAISS (where score is L2 distance) to Chroma. Calibrate thresholds per backend.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Re-instantiating `Chroma.from_documents` on Every Web Request**  
    Calling `from_documents` inside an API endpoint re-chunks and re-embeds all documents on every user request. Use `from_documents` once during ingestion, and instantiate with `Chroma(...)` for queries.
    """)
    st.warning("""
    **Trap 4: Missing the `persist_directory` Parameter**  
    Omitting `persist_directory` in `langchain_chroma.Chroma` creates an ephemeral in-memory store that vanishes when the Python process exits. Always specify a persistence path.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the LangChain VectorStore ABC?"):
        st.markdown("""
        **VectorStore Interface**:  
        Standardized abstraction implemented by 50+ backends, providing `.add_documents()`, `.similarity_search()`, and `.as_retriever()`.
        """)
    with st.expander("🗂️ Flashcard: What package hosts Chroma in modern LangChain?"):
        st.markdown("""
        **Partner Package**:  
        `langchain-chroma` (`from langchain_chroma import Chroma`).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How does `.as_retriever()` enable LCEL?"):
        st.markdown("""
        **Runnable Bridge**:  
        Wraps the vector store into a `Runnable` that takes a string query and outputs `List[Document]`, enabling pipe chaining (`|`).
        """)
    with st.expander("🗂️ Flashcard: What are the 3 search types in as_retriever?"):
        st.markdown("""
        **Search Types**:  
        1. `similarity` (Standard Top-K)  
        2. `mmr` (Maximal Marginal Relevance diversity)  
        3. `similarity_score_threshold` (Cutoff filter)
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the purpose of the LangChain VectorStore abstraction layer.", value=True)
chk2 = st.checkbox("I know how to instantiate persistent Chroma stores using `langchain-chroma`.", value=True)
chk3 = st.checkbox("I can convert a VectorStore into an LCEL-ready Retriever via `.as_retriever()`.", value=True)
chk4 = st.checkbox("I know how to configure `similarity`, `mmr`, and `similarity_score_threshold` search types.", value=True)
chk5 = st.checkbox("I understand how to construct declarative RAG chains using LangChain Expression Language (LCEL).", value=True)
chk6 = st.checkbox("I can implement dynamic tenant-filtered retrieval pipelines in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered LangChain Vector Stores and are ready for Chapter 15.2: Document Loaders!")
