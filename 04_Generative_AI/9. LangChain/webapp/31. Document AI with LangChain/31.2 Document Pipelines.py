import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="31.2 Document Pipelines - LangChain Architecture",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for 8-stage pipeline visualization
st.markdown("""
<style>
    .stage-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-stage {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DocumentPipelineSimulator:
    """
    Simulation engine tracking an incoming document through all 8 stages of the Document AI lifecycle:
    Loading -> Cleaning -> Parsing -> Chunking -> Embedding -> Indexing -> Retrieval -> Generation.
    """
    def run_pipeline(self, doc_text: str, chunk_size: int, overlap: int) -> Dict[str, Any]:
        # Stage 1: Loading
        raw_len = len(doc_text)
        
        # Stage 2: Cleaning
        cleaned = " ".join(doc_text.split()) # Normalize whitespace
        
        # Stage 3: Parsing
        has_table = "|" in cleaned
        
        # Stage 4: Chunking
        words = cleaned.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            ch = " ".join(words[i:i + chunk_size])
            if ch:
                chunks.append(ch)

        # Stage 5 & 6: Embedding & Indexing
        vectors_generated = len(chunks)
        dim = 1536

        # Stage 7 & 8: Retrieval & Generation
        top_k = min(2, len(chunks))
        retrieved_context = chunks[:top_k] if chunks else []
        generation = f"Synthesized answer based on {len(retrieved_context)} retrieved chunks."

        return {
            "stages": [
                {"stage": "1. Loading", "metric": f"{raw_len} raw characters acquired"},
                {"stage": "2. Cleaning", "metric": "Whitespace normalized, control codes stripped"},
                {"stage": "3. Parsing", "metric": f"Structural elements identified (Table: {has_table})"},
                {"stage": "4. Chunking", "metric": f"Produced {len(chunks)} chunks (size={chunk_size}, overlap={overlap})"},
                {"stage": "5. Embedding", "metric": f"{vectors_generated} vectors computed (dim={dim})"},
                {"stage": "6. Indexing", "metric": "Vectors persisted with hash deduplication"},
                {"stage": "7. Retrieval", "metric": f"Top-{top_k} most relevant chunks retrieved"},
                {"stage": "8. Generation", "metric": "Final grounded answer produced by ChatModel"}
            ],
            "chunks": chunks,
            "final_answer": generation
        }


# --- LESSON HEADER ---
st.title("31.2 The 8-Stage Document AI Pipeline Lifecycle")
st.caption("Chapter 31: Document AI with LangChain | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Transforming enterprise documents into high-performance, accurate Generative AI systems requires a rigorous,
multi-stage data engineering pipeline. A naive approach of loading raw text and feeding it into an embedding model
results in noisy retrievals, severe hallucination, and degraded answers.

In this lesson, you will master the complete **8-Stage Document AI Lifecycle**:
1. **Loading**: Ingesting heterogeneous blobs across local filesystems, cloud object stores, and web APIs.
2. **Cleaning**: Normalizing whitespace, fixing character encodings, and stripping boilerplate noise.
3. **Parsing**: Performing layout analysis, extracting tables, and enriching structural metadata.
4. **Chunking**: Balancing fixed-size, recursive character, and semantic boundary splitting strategies.
5. **Embedding**: Synthesizing dense vector representations and managing API rate limits.
6. **Indexing**: Persisting vectors with record managers, deduplication hashes, and namespace scoping.
7. **Retrieval**: Combining dense vector search, BM25 keyword matching, and cross-encoder rerankers.
8. **Generation**: Injecting grounded context into parameterized prompts for streaming token egress.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: The 8-Stage Document AI Pipeline", divider="rainbow")
st.markdown("""
The complete Document AI pipeline operates across two asynchronous workflows:
the **Offline Ingestion Pipeline** (Stages 1 through 6) and the **Online Query Pipeline** (Stages 7 and 8).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    pipe_dot = """
    digraph DocumentPipelineBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_offline_ingestion {
            label="Offline Ingestion & Indexing Pipeline (Stages 1-6)";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            s1_load [label="Stage 1: Loading\\nBlob loaders (S3, Disk, GCS)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s2_clean [label="Stage 2: Cleaning\\nUnicode NFC & boilerplate filter", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s3_parse [label="Stage 3: Parsing\\nLayout analysis & table extraction", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s4_chunk [label="Stage 4: Chunking\\nRecursive splitter & overlap", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            s5_embed [label="Stage 5: Embedding\\nDense embeddings (text-embedding-3)", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            s6_index [label="Stage 6: Indexing\\nVectorStore Record Manager (Deduplication)", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_online_query {
            label="Online Retrieval & Generation Pipeline (Stages 7-8)";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            user_q [label="User Query\\n'What is the refund policy?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s7_retrieval [label="Stage 7: Retrieval\\nHybrid Search (BM25 + Vector) + Rerank", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            s8_gen [label="Stage 8: Generation\\nContext injection & ChatModel stream", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        s1_load -> s2_clean [label="Raw Text"];
        s2_clean -> s3_parse [label="Sanitized"];
        s3_parse -> s4_chunk [label="Sections"];
        s4_chunk -> s5_embed [label="Chunks"];
        s5_embed -> s6_index [label="Vectors"];
        user_q -> s7_retrieval [label="Query Vector"];
        s6_index -> s7_retrieval [label="Vector DB Match"];
        s7_retrieval -> s8_gen [label="Top Chunks"];
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=True)

# --- SECTION 2: LOADING ---
st.header("2. Loading", divider="rainbow")
st.markdown("""
**Stage 1: Loading** is the initial data acquisition step. A production pipeline must handle diverse storage backends:
local disk directories, AWS S3 buckets, Google Cloud Storage, Microsoft SharePoint, and Confluence spaces.
""")

col_ld1, col_ld2 = st.columns(2)
with col_ld1:
    st.markdown("""
    #### Core Architectural Best Practices:
    - **Streaming Blob Loading**: Avoid reading hundreds of files into RAM simultaneously. Use LangChain's `FileSystemBlobLoader` or `lazy_load()` to stream documents page-by-page.
    - **Incremental Synchronization**: Check file modification timestamps (`mtime`) and content hashes to load only newly created or modified files.
    - **Metadata Enrichment**: Attach file origin, author, creation timestamp, and tenant ownership immediately upon loading.
    """)

with col_ld2:
    with st.container(border=True):
        st.markdown("💾 **Lazy Loading with FileSystemBlobLoader**")
        st.code("""
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader

loader = FileSystemBlobLoader(
    path="./knowledge_base/",
    glob="**/*.pdf"
)

# Stream blobs iteratively to conserve RAM
for blob in loader.yield_blobs():
    print(f"Loading blob: {blob.source}")
        """, language="python")

# --- SECTION 3: CLEANING ---
st.header("3. Cleaning", divider="rainbow")
st.markdown("""
**Stage 2: Cleaning** purges noise before structural parsing. Raw extracted text often contains corrupted encoding,
redundant whitespace, non-printable control characters, and repetitive boilerplate.
""")

st.markdown("""
#### Five Essential Cleaning Passes:
1. **Unicode NFC Normalization**: Converts composite character sequences into standard single-code-point representations.
2. **Whitespace Collapsing**: Replaces multiple consecutive tabs and spaces with single space characters, reducing token waste.
3. **Null & Control Byte Removal**: Strips ASCII `0x00` and unprintable characters that crash downstream databases.
4. **Header & Footer Stripping**: Eliminates repeating page numbers and copyright banners that confuse semantic search.
5. **PII Anonymization (Optional)**: Scrubs Social Security Numbers, emails, and phone numbers before embedding.
""")

# --- SECTION 4: PARSING ---
st.header("4. Parsing", divider="rainbow")
st.markdown("""
**Stage 3: Parsing** bridges plain text and structural semantics. In modern Document AI, parsing entails layout analysis,
table recognition, and reading order reconstruction.
""")

col_prs1, col_prs2 = st.columns(2)
with col_prs1:
    st.markdown("""
    #### Layout-Aware Parsing:
    - **Table-to-Markdown Extraction**: Converts 2D cell grids into markdown tables (`| Column 1 | Column 2 |`) so downstream models understand column-row relationships.
    - **Header Hierarchy Tagging**: Identifies document titles, section headers, and subheaders to maintain logical document trees.
    - **OCR Layout Analysis**: Models like Donut, LayoutLMv3, and PaddleOCR segment scanned images into titles, captions, and text blocks.
    """)

with col_prs2:
    with st.container(border=True):
        st.markdown("📊 **Document Element Segmentation**")
        st.caption("Unstructured element classification")
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────┐
        │ Title: "Q3 Financial Earnings Report"                  │
        ├────────────────────────────────────────────────────────┤
        │ Narrative: "Operating margin expanded to 32%..."       │
        ├────────────────────────────────────────────────────────┤
        │ Table: | Metric | Q2 | Q3 | -> Serialized Markdown     │
        ├────────────────────────────────────────────────────────┤
        │ Caption: "Figure 1.2: YoY regional growth rates"       │
        └────────────────────────────────────────────────────────┘
        ```
        """)

# --- SECTION 5: CHUNKING ---
st.header("5. Chunking", divider="rainbow")
st.markdown("""
**Stage 4: Chunking** segments parsed documents into optimal context windows.
If chunks are too small, they lose critical context. If chunks are too large, they dilute embedding specificity.
""")

st.subheader("Comparison of Chunking Strategies")
chunk_comp = pd.DataFrame({
    "Chunking Strategy": ["Fixed-Size Character Splitter", "Recursive Character Splitter", "Semantic Boundary Splitter", "Parent-Document Splitter"],
    "How It Works": ["Splits strictly every N characters (e.g. 500 chars)", "Tries splitting by `\\n\\n`, then `\\n`, then spaces", "Splits when embedding cosine distance shifts between sentences", "Indexes small 150-token chunks but retrieves 1,000-token parent"],
    "Semantic Coherence": ["Lowest (Cuts words and sentences in half)", "High (Preserves paragraphs and sentences)", "Highest (Respects topic transitions)", "Superior (High precision search + broad context)"],
    "Computational Cost": ["Near Zero", "Very Low", "High (Requires embedding every sentence)", "Moderate (Dual-tier indexing)"]
})
st.dataframe(chunk_comp, use_container_width=True)

# --- SECTION 6: EMBEDDING ---
st.header("6. Embedding", divider="rainbow")
st.markdown("""
**Stage 5: Embedding** translates text chunks into dense numerical vectors representing semantic meaning in high-dimensional space.
""")

col_emb1, col_emb2 = st.columns(2)
with col_emb1:
    st.markdown("""
    #### Production Considerations:
    - **Dimensional Trade-offs**: Models like OpenAI `text-embedding-3-small` (1536 dim) vs `large` (3072 dim). Higher dimensions offer greater retrieval precision at higher storage cost.
    - **Embedding Caching**: Using `CacheBackedEmbeddings` to cache vector calculations in Redis or SQLite, preventing expensive re-computation of identical chunks.
    - **Rate Limit Batching**: Grouping chunks into batches of 100 to 500 chunks per API invocation to respect provider TPM (Tokens Per Minute) ceilings.
    """)

with col_emb2:
    with st.container(border=True):
        st.markdown("⚡ **CacheBackedEmbeddings in LangChain**")
        st.code("""
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("./cache_embeddings/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
        """, language="python")

# --- SECTION 7: INDEXING ---
st.header("7. Indexing", divider="rainbow")
st.markdown("""
**Stage 6: Indexing** commits generated vectors and metadata into a vector database (Pinecone, Chroma, Milvus, Qdrant).
A production system must handle **Incremental Updates and Deduplication**; re-indexing an entire repository
every night wastes massive embedding budgets and duplicates vectors.
""")

col_idx1, col_idx2 = st.columns(2)
with col_idx1:
    st.markdown("""
    #### The LangChain Indexing API (`RecordManager`):
    - **Hash Deduplication**: Computes sha256 hashes of chunk contents and metadata.
    - **Cleanup Modes**:
      - `cleanup="incremental"`: Only indexes modified/new chunks; leaves unchanged vectors untouched.
      - `cleanup="full"`: Deletes vectors for documents that no longer exist in the source repository.
    """)

with col_idx2:
    with st.container(border=True):
        st.markdown("🔄 **LangChain Indexing API**")
        st.code("""
from langchain.indexes import SQLRecordManager, index

record_manager = SQLRecordManager(
    namespace="acme_kb", db_url="sqlite:///record_manager.db"
)
record_manager.create_schema()

# Performs automatic deduplication & cleanup!
res = index(docs, record_manager, vectorstore, cleanup="incremental", source_id_key="source")
print(res) # {'num_added': 5, 'num_updated': 0, 'num_deleted': 1, 'num_skipped': 42}
        """, language="python")

# --- SECTION 8: RETRIEVAL ---
st.header("8. Retrieval", divider="rainbow")
st.markdown("""
**Stage 7: Retrieval** is the online phase where a user's query searches the indexed corpus.
Modern production pipelines abandon pure dense vector lookup in favor of **Hybrid Search with Reranking**.
""")

ret_dot = """
digraph HybridRetrieval {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    query [label="User Query", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    dense_branch [label="Dense Vector Search\\nCosine similarity", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    sparse_branch [label="Sparse BM25 Search\\nExact keyword match", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    rrf_fusion [label="Reciprocal Rank Fusion\\nRanks combined candidates", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    reranker [label="Cross-Encoder Reranker\\n(Cohere Rerank / BGE-Reranker)", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];

    query -> dense_branch [label="Embed Query"];
    query -> sparse_branch [label="Tokenize Query"];
    dense_branch -> rrf_fusion [label="Top 20 Chunks"];
    sparse_branch -> rrf_fusion [label="Top 20 Chunks"];
    rrf_fusion -> reranker [label="Top 25 Merged"];
    reranker -> reranker [label="Cross-Attention Score"];
}
"""
st.graphviz_chart(ret_dot, use_container_width=True)

# --- SECTION 9: GENERATION ---
st.header("9. Generation", divider="rainbow")
st.markdown("""
**Stage 8: Generation** synthesizes retrieved chunks into a coherent, cited answer.
The prompt template injects context chunks inside strict XML delimiters and instructs the model to state
`"I do not have sufficient information"` if the answer is absent from the provided context.
""")

col_gen1, col_gen2 = st.columns(2)
with col_gen1:
    st.markdown("""
    #### Production Best Practices for Generation:
    - **Citation Anchoring**: Force the model to cite chunk indices (`[Doc 1, p. 4]`) for every factual statement.
    - **Streaming Output**: Stream tokens in real time to the frontend using `chain.stream()` or `astream_events()`.
    - **Hallucination Detection**: Implement post-generation evaluation asserting that all claims are grounded in context.
    """)

with col_gen2:
    with st.container(border=True):
        st.markdown("💬 **Grounded Generation Prompt**")
        st.code("""
PROMPT = \"\"\"
You are an enterprise knowledge assistant.
Answer the question strictly based on the provided context.
If the answer is not contained in the context, say 'I cannot find that in the documents.'

<context>
{context}
</context>

Question: {question}
Grounded Answer (with citations):
\"\"\"
        """, language="python")

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Full 8-Stage Pipeline in LCEL", divider="rainbow")
st.markdown("""
Below is a runnable LangChain LCEL pipeline assembling the stages into a unified, end-to-end RAG architecture.
""")

prod_pipe_code = '''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Ingestion Stages (Simulated)
raw_text = "Acme Corp Policy: Standard annual vacation entitlement is 25 days for full-time employees."
docs = [Document(page_content=raw_text, metadata={"source": "hr_policy.pdf", "page": 1})]

# 2. Chunking Stage
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = splitter.split_documents(docs)

# 3. Simulated Retriever Function
def mock_retriever(query: str) -> str:
    # Formats retrieved chunks with metadata citations
    return "\\n".join([f"[{d.metadata['source']} p.{d.metadata['page']}]: {d.page_content}" for d in splits])

# 4. Grounded Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user question strictly using the provided context:\\n\\n{context}"),
    ("human", "{question}")
])

# 5. Production LCEL Retrieval-Generation Chain
rag_chain = (
    {"context": RunnableLambda(lambda x: mock_retriever(x["question"])), "question": lambda x: x["question"]}
    | prompt
    # | chat_model (In production: ChatOpenAI / ChatAnthropic)
    | StrOutputParser()
)
'''
st.code(prod_pipe_code, language="python")

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: 8-Stage Document AI Pipeline Simulator", divider="rainbow")
st.markdown("""
Simulate how an enterprise document flows through all 8 stages of the Document AI lifecycle.
Adjust chunk parameters to observe real-time chunking, embedding estimation, and final generation synthesis.
""")

pipe_engine = DocumentPipelineSimulator()

col_pw1, col_pw2 = st.columns([1, 1])

with col_pw1:
    st.subheader("⚙️ Pipeline Input & Chunking Controls")
    sample_corpus = st.text_area(
        "Enter Raw Document Text to Ingest:",
        value="Cloud Infrastructure Policy 2026. All production Kubernetes clusters must enforce mutual TLS encryption. Databases require daily automated snapshots retained for 30 days. Contact DevOps lead at ops@acme.com for break-glass credentials.",
        height=120
    )
    chunk_size_input = st.slider("Chunk Size (Words):", min_value=10, max_value=50, value=20)
    overlap_input = st.slider("Chunk Overlap (Words):", min_value=0, max_value=15, value=5)
    run_pipe_btn = st.button("🚀 Execute 8-Stage Pipeline", type="primary", use_container_width=True)

with col_pw2:
    st.subheader("📊 8-Stage Lifecycle Telemetry")
    if run_pipe_btn:
        pipe_res = pipe_engine.run_pipeline(sample_corpus, chunk_size_input, overlap_input)

        st.markdown("##### 🔄 Stage-by-Stage Progression")
        stage_df = pd.DataFrame(pipe_res["stages"])
        st.dataframe(stage_df, use_container_width=True)

        st.markdown("##### 📑 Produced Chunks (Stage 4 Output)")
        for idx, ch in enumerate(pipe_res["chunks"]):
            st.caption(f"Chunk #{idx+1} ({len(ch.split())} words):")
            st.code(ch, language="text")

        st.success(f"**Stage 8 Generation:** {pipe_res['final_answer']}")
    else:
        st.info("Click 'Execute 8-Stage Pipeline' to trace a document through the lifecycle.")

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is Incremental Indexing mandatory for enterprise document repositories?"):
    st.markdown("""
    **Answer**: In an enterprise repository with 500,000 documents, re-indexing the entire corpus on every update would consume
    thousands of dollars in embedding API fees and require hours of compute time. The LangChain Indexing API with a `RecordManager`
    computes deterministic content hashes of every chunk. When synchronization runs, only new or modified documents are embedded
    and written to the vector store (`num_added`), while unchanged chunks are skipped (`num_skipped`), and deleted documents
    have their vectors automatically purged (`num_deleted`).
    """)

with st.expander("Q2: What is the Parent-Document Retrieval pattern, and why is it superior to naive chunking?"):
    st.markdown("""
    **Answer**: Small chunks (e.g. 100 tokens) produce superior, highly specific vector embeddings because they focus on a single concept,
    whereas large chunks (e.g. 1,000 tokens) dilute semantic similarity. However, an LLM generating an answer needs broad context
    to avoid hallucination. The Parent-Document Retriever embeds small child chunks for high-precision search, but when a match occurs,
    it retrieves and injects the larger parent document into the prompt context, providing the best of both worlds.
    """)

with st.expander("Q3: How does Hybrid Search (BM25 + Dense Vector) mitigate the weaknesses of pure vector search?"):
    st.markdown("""
    **Answer**: Dense vector search operates on conceptual semantic similarity, but frequently fails on exact keyword lookups:
    part numbers (`XYZ-9941`), UUIDs, error codes (`ERR_404_TIMEOUT`), or rare medical terms. BM25 (sparse keyword search) excels
    at exact term frequency matching. Hybrid search queries both algorithms in parallel and merges their rankings using
    Reciprocal Rank Fusion (RRF), ensuring that both semantic intent and exact keyword matches are surfaced.
    """)

with st.expander("Q4: What is the purpose of a Cross-Encoder Reranker in Stage 7 Retrieval?"):
    st.markdown("""
    **Answer**: Bi-encoder vector models compute query and document embeddings independently, enabling fast index lookup via cosine distance,
    but they cannot model fine-grained token-level cross-attention between the query and the document. A Cross-Encoder Reranker
    (e.g., Cohere Rerank or BGE-Reranker) feeds the query and candidate chunk together into a full transformer model, evaluating
    deep token-level cross-attention. Running a reranker over the top 25 candidates boosts retrieval precision by 20% to 35%.
    """)

with st.expander("Q5: Why is Unicode NFC Normalization essential in Stage 2 Cleaning?"):
    st.markdown("""
    **Answer**: Unicode supports multiple encodings for visually identical characters (e.g., composite characters like 'é' can be stored
    as a single code point `U+00E9` [NFC] or as 'e' followed by an acute accent `U+0065 U+0301` [NFD]). If an indexed document uses NFD
    and a user query uses NFC, exact keyword matchers and tokenizers will produce different token IDs, causing missed matches.
    NFC normalization standardizes all strings into identical code point sequences before embedding.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Zero-Overlap Chunking)**: Chunking text with `chunk_overlap=0`. A critical sentence spanning across the boundary between chunk 1 and chunk 2 is sliced in half, corrupting the semantic meaning in both chunks.")
    st.warning("**Trap 2 (Re-Embedding Stale Documents)**: Running nightly batch scripts that call `add_documents()` without deduplication. Your vector store accumulates 30 duplicate copies of the same company handbook in 30 days.")
with col_trap2:
    st.error("**Trap 3 (Omission of Citation Grounding)**: Prompting the LLM without strict instructions to cite specific document chunks. When the model hallucinates an answer, users have no way to verify claim validity.")
    st.warning("**Trap 4 (Embedding API Rate Limit Spikes)**: Sending 10,000 chunks to OpenAI's embedding API in a single unthrottled loop, triggering HTTP 429 rate limit exceptions and crashing the ingestion job.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 8 stages of the Document AI lifecycle?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Loading, 2. Cleaning, 3. Parsing, 4. Chunking, 5. Embedding, 6. Indexing, 7. Retrieval, 8. Generation.")
    with st.container(border=True):
        st.markdown("**Front**: What does the LangChain RecordManager accomplish?")
        st.caption("Flip for answer:")
        st.info("**Back**: It tracks content hashes of indexed chunks, enabling incremental deduplication and automatic deletion of stale vectors.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Hybrid Search (BM25 + Vector) superior to vector-only search?")
        st.caption("Flip for answer:")
        st.info("**Back**: It combines semantic conceptual understanding (vectors) with exact keyword and error-code matching (BM25).")
    with st.container(border=True):
        st.markdown("**Front**: What is the Parent-Document Retriever pattern?")
        st.caption("Flip for answer:")
        st.info("**Back**: It embeds small child chunks for precise vector matching, but injects the larger parent document into the prompt for rich context.")

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand all 8 sequential stages of the Document AI lifecycle.", value=True)
c2 = st.checkbox("I know how to implement streaming lazy loading with FileSystemBlobLoader.", value=True)
c3 = st.checkbox("I understand why incremental indexing and content hash deduplication are mandatory.", value=True)
c4 = st.checkbox("I know how to design hybrid search pipelines combining BM25 and dense vectors.", value=True)
c5 = st.checkbox("I understand the architectural benefits of cross-encoder rerankers in Stage 7.", value=True)
c6 = st.checkbox("I recognize how citation grounding in Stage 8 eliminates unverified hallucinations.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 31.3: Document Applications.")
