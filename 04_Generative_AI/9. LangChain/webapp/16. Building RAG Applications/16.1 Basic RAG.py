import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class RAGDocumentChunk:
    """Represents a chunked document item with metadata and vector representation."""
    def __init__(self, chunk_id: str, text: str, source: str, vector: np.ndarray):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.vector = vector

class MockBasicRAGSimulator:
    """
    Simulates the end-to-end 7-stage basic RAG lifecycle:
    Load -> Split -> Embed -> Store -> Retrieve -> Build Prompt -> Generate Answer.
    """
    def __init__(self):
        # Raw documents
        self.raw_corpus = [
            {
                "id": "doc-cloud-sla",
                "source": "infrastructure/cloud_sla_2026.pdf",
                "text": "Enterprise cloud infrastructure guarantees 99.995% uptime for multi-region active-active deployments. Automated failover triggers within 12 seconds of health-check timeout, with zero data loss on synchronous RDS PostgreSQL clusters."
            },
            {
                "id": "doc-iam-policy",
                "source": "security/iam_governance_2026.md",
                "text": "IAM role credentials expire every 60 minutes. Mandatory multi-factor authentication (MFA) with FIDO2 hardware keys is enforced for all administrative API access, and cross-account access requires external ID validation."
            },
            {
                "id": "doc-incident-sop",
                "source": "devops/incident_response_sop.txt",
                "text": "Severity-1 outages mandate incident commander assignment within 5 minutes. PagerDuty automated escalation alerts on-call engineering leads, and customer status page updates must be broadcast every 15 minutes."
            }
        ]

        # Pre-calculated chunk vectors
        self.chunks: List[RAGDocumentChunk] = [
            RAGDocumentChunk(
                "chunk-01",
                "Enterprise cloud infrastructure guarantees 99.995% uptime for multi-region active-active deployments.",
                "infrastructure/cloud_sla_2026.pdf",
                np.array([0.92, 0.15, 0.22])
            ),
            RAGDocumentChunk(
                "chunk-02",
                "Automated failover triggers within 12 seconds of health-check timeout, with zero data loss on synchronous RDS PostgreSQL clusters.",
                "infrastructure/cloud_sla_2026.pdf",
                np.array([0.88, 0.20, 0.28])
            ),
            RAGDocumentChunk(
                "chunk-03",
                "IAM role credentials expire every 60 minutes. Mandatory MFA with FIDO2 keys is enforced for administrative API access.",
                "security/iam_governance_2026.md",
                np.array([0.18, 0.94, 0.16])
            ),
            RAGDocumentChunk(
                "chunk-04",
                "Cross-account role assumptions require explicit external ID validation and STS session tagging.",
                "security/iam_governance_2026.md",
                np.array([0.22, 0.89, 0.24])
            ),
            RAGDocumentChunk(
                "chunk-05",
                "Severity-1 outages mandate incident commander assignment within 5 minutes and automated PagerDuty escalation.",
                "devops/incident_response_sop.txt",
                np.array([0.35, 0.25, 0.91])
            ),
            RAGDocumentChunk(
                "chunk-06",
                "Customer status page updates must be broadcast every 15 minutes during active Sev-1 triage operations.",
                "devops/incident_response_sop.txt",
                np.array([0.28, 0.30, 0.89])
            )
        ]
        # Normalize vectors
        for c in self.chunks:
            norm = np.linalg.norm(c.vector)
            if norm > 0:
                c.vector = c.vector / norm

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.lower()
        v = np.array([0.3, 0.3, 0.3])
        if any(w in q for w in ["uptime", "sla", "failover", "cloud", "rds", "postgres"]):
            v = np.array([0.91, 0.18, 0.24])
        elif any(w in q for w in ["iam", "mfa", "fido2", "credential", "security", "role"]):
            v = np.array([0.19, 0.93, 0.18])
        elif any(w in q for w in ["incident", "outage", "sev-1", "pagerduty", "status", "commander"]):
            v = np.array([0.32, 0.26, 0.92])
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def run_pipeline(self, question: str, k: int = 2) -> Dict[str, Any]:
        # 1. Embed query
        q_vec = self._embed_query(question)

        # 2. Retrieve
        scored = []
        for c in self.chunks:
            sim = float(np.dot(q_vec, c.vector))
            scored.append({"chunk": c, "score": round(max(0.0, min(1.0, sim)), 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        retrieved = scored[:k]

        # 3. Build Prompt
        context_str = "\n".join([f"- [{item['chunk'].source}]: {item['chunk'].text}" for item in retrieved])
        prompt_str = (
            f"System: You are an enterprise assistant. Answer based strictly on context.\n"
            f"Context:\n{context_str}\n\n"
            f"User Question: {question}\nAnswer:"
        )

        # 4. Generate Answer (Simulated synthesis)
        if "uptime" in question.lower() or "sla" in question.lower():
            answer = f"Based on {retrieved[0]['chunk'].source}, our enterprise cloud infrastructure guarantees a 99.995% uptime SLA with automated 12-second failover on synchronous RDS clusters."
        elif "iam" in question.lower() or "credential" in question.lower() or "mfa" in question.lower():
            answer = f"According to {retrieved[0]['chunk'].source}, IAM credentials expire every 60 minutes, and administrative API access strictly requires FIDO2 hardware MFA."
        elif "incident" in question.lower() or "outage" in question.lower() or "sev-1" in question.lower():
            answer = f"Per {retrieved[0]['chunk'].source}, Severity-1 incidents require assigning an incident commander within 5 minutes and broadcasting status page updates every 15 minutes."
        else:
            answer = f"Synthesizing retrieved context: The policy documents indicate strict compliance controls regarding operational resilience ({retrieved[0]['chunk'].text})."

        return {
            "retrieved_chunks": retrieved,
            "assembled_context": context_str,
            "final_prompt": prompt_str,
            "generated_answer": answer
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.1 Basic RAG — LangChain Course",
    page_icon="🏗️",
    layout="wide"
)

st.title("16.1 Basic RAG")
st.caption("Step-by-Step Construction of the 7-Stage RAG Pipeline: Load, Split, Embed, Store, Retrieve, Prompt, and Generate")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The 7-Stage Foundational RAG Pipeline", divider="rainbow")
st.markdown("""
Retrieval-Augmented Generation (RAG) is the foundational design pattern for giving Large Language Models access to 
private, real-time, or domain-specific enterprise knowledge without modifying model weights.

The basic RAG architecture executes sequentially across **7 discrete engineering stages**, divided into an 
**Ingestion Pipeline** (offline indexing) and a **Query Pipeline** (real-time LCEL execution).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph BasicRAGArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_ingestion {
            label = "Phase A: Offline Knowledge Ingestion Pipeline";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            
            s1_load [label="Stage 1: Load Documents\\nDocumentLoader (PDF, Markdown, HTML, CSV)", fillcolor="#1E293B", color="#0284C7"];
            s2_split [label="Stage 2: Split Documents\\nTextSplitter (chunk_size=500, chunk_overlap=50)", fillcolor="#1E293B", color="#0284C7"];
            s3_embed [label="Stage 3: Generate Embeddings\\nEmbeddings Model (e.g. text-embedding-3-small)", fillcolor="#1E293B", color="#0284C7"];
            s4_store [label="Stage 4: Store Vectors & Index\\nVectorStore (Chroma, FAISS, Pinecone)", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_query {
            label = "Phase B: Real-Time LCEL Generation Pipeline";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            
            user_q [label="User Natural Language Query\\n'What is our cloud uptime SLA?'", fillcolor="#1E293B", color="#F59E0B"];
            s5_retrieve [label="Stage 5: Retrieve Top-k Chunks\\nvectorstore.as_retriever(search_kwargs={'k': 3})", fillcolor="#0F172A", color="#38BDF8"];
            s6_prompt [label="Stage 6: Assemble Prompt Template\\nChatPromptTemplate injecting {context} + {question}", fillcolor="#064E3B", color="#10B981"];
            s7_generate [label="Stage 7: Generate Final Answer\\nChatModel + StrOutputParser()", fillcolor="#022C22", color="#059669"];
            final_out [label="Verified Factual Answer with Grounding", fillcolor="#1E293B", color="#10B981"];
        }

        s1_load -> s2_split [label="raw documents"];
        s2_split -> s3_embed [label="text chunks"];
        s3_embed -> s4_store [label="dense vectors + metadata"];
        
        user_q -> s5_retrieve [label="query text"];
        s4_store -> s5_retrieve [label="k-NN candidate search"];
        s5_retrieve -> s6_prompt [label="format_docs()"];
        user_q -> s6_prompt [label="passthrough question"];
        s6_prompt -> s7_generate [label="assembled messages"];
        s7_generate -> final_out [label="stream / invoke"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: LOAD DOCUMENTS ---
st.header("2. Stage 1: Load Documents", divider="rainbow")
st.markdown("""
The first stage extracts text from source formats into standardized LangChain `Document` objects containing 
`page_content: str` and `metadata: dict`.
""")

col_l1, col_l2 = st.columns(2)
with col_l1:
    st.markdown("""
    ### Best Practices for Ingestion:
    1. **Preserve Metadata**: Always capture `source`, `page_number`, `created_at`, and `department` to enable 
       downstream citations and tenant filtering.
    2. **Lazy Loading**: Use `.lazy_load()` rather than `.load()` when processing gigabyte-scale datasets to avoid 
       exhausting worker memory.
    """)
with col_l2:
    st.code("""
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Loading a single file
loader = PyPDFLoader("data/cloud_sla_2026.pdf")
docs = loader.load()

# Each doc has page_content and metadata:
# docs[0].metadata -> {'source': 'data/cloud_sla_2026.pdf', 'page': 0}
    """, language="python")

st.divider()

# --- SECTION 3: SPLIT DOCUMENTS ---
st.header("3. Stage 2: Split Documents", divider="rainbow")
st.markdown("""
Embedding whole multi-page documents yields diluted vectors that fail similarity search. Splitting segments raw documents 
into semantic chunks.
""")

col_sp1, col_sp2 = st.columns(2)
with col_sp1:
    st.info("""
    **`RecursiveCharacterTextSplitter`**:
    The industry-standard splitter for generic text. Attempts to split sequentially by:
    `["\\n\\n", "\\n", " ", ""]` to preserve paragraphs and sentences intact.
    """)
with col_sp2:
    st.success("""
    **Chunk Overlap Mechanics**:
    Setting `chunk_overlap=50` repeats the trailing 50 characters of chunk N at the start of chunk N+1, preventing 
    critical context from being cut in half at sentence boundaries.
    """)

st.code("""
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Maximum characters per chunk
    chunk_overlap=50,       # Sliding window overlap
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(docs)
""", language="python")

st.divider()

# --- SECTION 4: GENERATE EMBEDDINGS ---
st.header("4. Stage 3: Generate Embeddings", divider="rainbow")
st.markdown("""
The embedding model transforms unstructured string chunks into dense geometric vectors (e.g., 1536 dimensions for OpenAI). 
Vectors positioned closely in geometric space represent semantically similar thoughts.
""")

st.code("""
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# Embeddings interface provides two core methods:
# 1. embed_documents(List[str]) -> List[List[float]] (Batch ingestion)
# 2. embed_query(str) -> List[float] (Single query)
""", language="python")

st.divider()

# --- SECTION 5: STORE VECTORS ---
st.header("5. Stage 4: Store Vectors & Index", divider="rainbow")
st.markdown("""
Vector stores store embedding vectors alongside original text chunks and metadata payloads, building approximate nearest neighbor 
(ANN) graph indices (such as HNSW) for sub-millisecond retrieval.
""")

st.code("""
from langchain_community.vectorstores import Chroma

# Build index and persist locally
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="enterprise_knowledge"
)
""", language="python")

st.divider()

# --- SECTION 6: RETRIEVE DOCUMENTS ---
st.header("6. Stage 5: Retrieve Documents", divider="rainbow")
st.markdown("""
When the user poses a question, the vector store is converted into an LCEL `BaseRetriever` runnable via `.as_retriever()`. 
The retriever encodes the question and queries the index for the top-k most similar chunks.
""")

st.code("""
# Convert vector store to retriever runnable
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Execute retrieval
retrieved_docs = retriever.invoke("What is our cloud uptime SLA?")
""", language="python")

st.divider()

# --- SECTION 7: BUILD PROMPT ---
st.header("7. Stage 6: Build Prompt Template", divider="rainbow")
st.markdown("""
Retrieved document chunks are formatted into a single context string and injected into a `ChatPromptTemplate` 
alongside explicit instructions commanding the model to ground its response exclusively in provided facts.
""")

st.code("""
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a helpful enterprise assistant. Answer the user question based ONLY on the "
    "provided context below. If you cannot find the answer in the context, state that you do "
    "not know. Do not extrapolate or fabricate facts.\\n\\n"
    "Context:\\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])
""", language="python")

st.divider()

# --- SECTION 8: GENERATE ANSWER ---
st.header("8. Stage 7: Generate Final Answer with LCEL", divider="rainbow")
st.markdown("""
Modern LangChain assembles the complete RAG pipeline declaratively using the pipe operator (`|`). 
`RunnablePassthrough` allows the user question to flow concurrently into both the retriever and the prompt template.
""")

st.code("""
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# Composable, type-safe LCEL RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Execution
response = rag_chain.invoke("What is our cloud uptime SLA?")
""", language="python")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: End-to-End Basic RAG Pipeline", divider="rainbow")
st.markdown("""
Here is the clean, self-contained implementation of a production-grade basic RAG pipeline wrapping all 7 stages 
into an operational utility function.
""")

st.code("""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_basic_rag_chain(raw_texts: List[str], openai_api_key: str):
    \"\"\"
    Builds an end-to-end 7-stage basic RAG pipeline using FAISS and GPT-4o-mini.
    \"\"\"
    # Stage 1: Load (wrap raw texts into Documents)
    raw_docs = [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(raw_texts)]

    # Stage 2: Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    split_docs = splitter.split_documents(raw_docs)

    # Stage 3 & 4: Embed & Store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Stage 5: Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Stage 6: Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based strictly on context:\\n\\n{context}"),
        ("human", "{question}")
    ])

    # Stage 7: Generate
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    
    def format_docs(docs):
        return "\\n\\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
""", language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Basic RAG Lifecycle Simulator", divider="rainbow")
st.markdown("""
Test the full **7-stage execution lifecycle** interactively. Choose a question, adjust the top-k retrieval budget, 
and inspect the exact retrieved chunks, prompt context injection, and generated answer.
""")

# Instantiate simulator at module scope
rag_simulator = MockBasicRAGSimulator()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    test_question = st.selectbox(
        "Select Inquiry Question:",
        [
            "What is our enterprise cloud uptime SLA and failover policy?",
            "How frequently do IAM credentials expire and what MFA is required?",
            "What is our response timeline for Severity-1 outages?",
            "Explain the database replication architecture."
        ]
    )

    k_choice = st.slider("Select Retrieval Top-k Chunks (k):", min_value=1, max_value=4, value=2)
    exec_btn = st.button("🚀 Run 7-Stage RAG Pipeline", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Pipeline Stage Output")
    
    if exec_btn or test_question:
        pipeline_run = rag_simulator.run_pipeline(test_question, k=k_choice)

        st.markdown("### 📄 Stage 5: Retrieved Chunks")
        for idx, item in enumerate(pipeline_run["retrieved_chunks"], 1):
            with st.expander(f"Chunk #{idx}: {item['chunk'].chunk_id} (Score: {item['score']})", expanded=True):
                st.write(f"**Text**: `{item['chunk'].text}`")
                st.caption(f"**Source**: `{item['chunk'].source}`")

        st.markdown("### 📝 Stage 6: Formatted Context Injected")
        st.code(pipeline_run["assembled_context"], language="markdown")

        st.markdown("### 🤖 Stage 7: Generated Synthesized Answer")
        st.success(pipeline_run["generated_answer"])

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is `RunnablePassthrough()` necessary in an LCEL RAG chain?",
     "`RunnablePassthrough()` takes the user's input query and passes it unchanged to downstream components. In a RAG chain `{'context': retriever | format_docs, 'question': RunnablePassthrough()}`, the input query must be sent to the retriever to fetch relevant documents while simultaneously being forwarded to the prompt template under the 'question' key. Without RunnablePassthrough, the prompt template would not receive the original question string."),
    ("What is the difference between `.load()` and `.lazy_load()` in LangChain DocumentLoaders?",
     "`.load()` eagerly reads and parses all documents into memory as a Python list. If a directory contains thousands of large PDFs or CSV files, this can cause Out-Of-Memory (OOM) crashes. `.lazy_load()` returns a Python generator that yields Document objects one by one, enabling streaming chunk ingestion and bounded memory usage."),
    ("Why is chunk overlap necessary when splitting documents?",
     "Text splitters split text based on character or token counts. If a critical semantic statement (e.g. 'The server timeout is 30 seconds') happens to straddle the boundary between chunk N and chunk N+1, splitting without overlap will cut the sentence in half. Chunk overlap preserves the trailing context of the previous chunk, ensuring boundary statements remain intact."),
    ("What happens if the retriever returns zero relevant chunks in basic RAG?",
     "In basic RAG without guardrails, if the retriever returns irrelevant or empty chunks, the prompt still instructs the model to answer. Weak models will hallucinate plausible-sounding answers based on internal pre-training data. Production prompts must include an explicit fallback constraint: 'If the answer cannot be found in the provided context, state that you do not know.'"),
    ("Why should temperature be set to 0.0 in enterprise RAG applications?",
     "Temperature controls the randomness of next-token sampling. In creative generation, higher temperatures introduce variety. In factual enterprise RAG, the goal is determinism and adherence to retrieved source documents. Setting temperature to 0.0 maximizes greedy token selection, drastically reducing hallucinations.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Ommitting `format_docs` Function in LCEL Chain. Retrievers return a `List[Document]`, but prompt templates expect a string for `{context}`. Passing the raw list directly without formatting will stringify Document objects including memory pointers, degrading model response quality. Always pass `retriever | format_docs`.")
    st.error("**Trap 2 (Breaking Error)**: Not Normalizing Vectors Before Cosine Search. If using dot product indices (e.g. FAISS IndexFlatIP) without L2-normalizing vectors beforehand, documents with longer texts or larger norms will dominate rankings regardless of true semantic similarity.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Splitting by Raw Characters Instead of Tokens. Characters do not map 1:1 to tokens. Splitting with `chunk_size=1000` on code or JSON can result in 500+ tokens per chunk, unexpectedly exhausting context windows. Use token-aware splitters for precise token budgeting.")
    st.warning("**Trap 4 (Quality Warning)**: Setting Top-k Too High in Basic RAG. Passing k=10 chunks in basic RAG triggers the 'lost-in-the-middle' phenomenon where LLMs overlook facts buried in intermediate chunks. Keep k between 2 and 4 unless using a reranker.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: 7 RAG Stages"):
        st.markdown("**Load, Split, Embed, Store, Retrieve, Build Prompt, Generate Answer.**")
    with st.expander("🎴 Flashcard 2: RunnablePassthrough"):
        st.markdown("**LCEL primitive that forwards the raw query into prompt template branches without alteration.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Recursive Character Splitting"):
        st.markdown("**Hierarchical text splitting respecting paragraphs, sentences, and words to preserve semantic unity.**")
    with st.expander("🎴 Flashcard 4: Grounding Instruction"):
        st.markdown("**Prompt directive commanding the model to answer strictly from retrieved context and admit ignorance.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can name and explain all 7 stages of the foundational RAG pipeline.", value=True)
chk2 = st.checkbox("I understand the difference between DocumentLoader .load() and .lazy_load().", value=True)
chk3 = st.checkbox("I know why chunk overlap is necessary to prevent boundary truncation.", value=True)
chk4 = st.checkbox("I can construct a complete declarative LCEL RAG chain using the pipe operator.", value=True)
chk5 = st.checkbox("I understand why temperature=0.0 and grounding directives mitigate hallucinations.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 basic RAG technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 16.1 Basic RAG!")
