import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="18.3 Retrieval-Augmented Generation",
    page_icon="📚",
    layout="wide"
)

st.title("18.3 Retrieval-Augmented Generation (RAG) in Streamlit")
st.caption("Architecting production RAG systems: document extraction, text chunking strategies, dense embeddings, vector indexing, semantic search, grounded generation, and source citations.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Retrieval-Augmented Generation (RAG) Pipeline

RAG connects parametric LLM capabilities with non-parametric external knowledge bases.
Instead of relying solely on static weights, RAG retrieves relevant document chunks dynamically at query time:

""")

st.graphviz_chart(
    r"""
    digraph RAG_Architecture {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        ingest [label="1. Ingestion & Indexing Phase\n• Document Upload (PDF/TXT)\n• Text Chunking & Dense Embeddings\n• Vector Store (HNSW / FAISS / Chroma)", fillcolor="#DBEAFE", color="#1D4ED8"];
        retrieve [label="2. Query & Retrieval Phase\n• User Query Embedding\n• Cosine Similarity Vector Search\n• Top-K Semantic Chunks Extracted", fillcolor="#FEF3C7", color="#D97706"];
        generate [label="3. Grounded Generation Phase\n• Augmented Context Prompt\n• LLM Streaming Synthesis\n• Streamlit Source Citations", fillcolor="#D1FAE5", color="#059669"];

        ingest -> retrieve [label="Indexed Vector Space"];
        retrieve -> generate [label="Augmented Context Prompt"];
    }
    """
)


tabs = st.tabs([
    "1. What is RAG?",
    "2. Document Ingestion",
    "3. Text Extraction",
    "4. Text Chunking",
    "5. Dense Embeddings",
    "6. Vector Databases",
    "7. Similarity Search",
    "8. Context Retrieval",
    "9. Grounded Answer Gen",
    "10. Source Citations"
])

# =============================================================================
# 1. WHAT IS RAG?
# =============================================================================
with tabs[0]:
    st.header("1. What is Retrieval-Augmented Generation (RAG)?")
    st.markdown(r"""
    Standard LLMs suffer from two critical limitations:
    1. **Knowledge Cutoffs**: Inability to know private, proprietary, or post-training information.
    2. **Hallucinations**: Confidently fabricating facts when uncertain.
    
    RAG solves this by injecting authoritative excerpts directly into the model's context window.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Parametric vs Non-Parametric Memory**")
        st.markdown(r"""
        | Dimension | Parametric (Pure LLM) | Non-Parametric (RAG) |
        | :--- | :--- | :--- |
        | **Knowledge Source** | Frozen neural network weights. | External vector databases & documents. |
        | **Update Frequency** | Requires costly re-training ($M). | Instantaneous (insert new document chunk). |
        | **Hallucination Rate**| Moderate to High on niche domains. | Low (Strictly grounded in provided excerpts). |
        | **Traceability** | Black box (cannot cite exact page). | Full traceability with page & paragraph citations. |
        | **Access Control** | Uniform (everyone sees all weights). | Granular (documents filtered by user ACLs). |
        """)

    with col2:
        st.markdown("**🖥️ RAG Mode Simulator**")
        with st.container(border=True):
            rag_mode = st.toggle("Enable RAG Grounding", value=True, key="rag_toggle_demo")
            st.caption("Query: *'What is our Q3 cloud infrastructure spend budget?'*")

            if rag_mode:
                st.success("✅ RAG Enabled: Grounded in `Financial_Report_Q3.pdf` (Page 14)")
                st.write("**Answer**: The approved Q3 cloud infrastructure budget is **$1.42M**, allocated across AWS (65%) and GCP (35%).")
            else:
                st.warning("⚠️ Unassisted Base LLM (Parametric Hallucination)")
                st.write("**Answer**: Typically, cloud infrastructure budgets for enterprise tech companies range from $500k to $2M depending on scale...")

# =============================================================================
# 2. DOCUMENT INGESTION
# =============================================================================
with tabs[1]:
    st.header("2. Document Upload & File Handling")
    st.markdown(r"""
    Using `st.file_uploader` to accept single or multi-file document corpora (`pdf`, `txt`, `md`, `docx`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

uploaded_files = st.file_uploader(
    "Upload Knowledge Base Documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    total_bytes = sum(f.size for f in uploaded_files)
    st.success(f"Ingested {len(uploaded_files)} files ({total_bytes / 1024:.1f} KB total).")
    for f in uploaded_files:
        st.write(f"- `{f.name}` ({f.size} bytes)")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Document Ingestion Console:")
            sample_docs = [
                {"name": "Security_Policies_2026.pdf", "size_kb": 245.2, "status": "Ready"},
                {"name": "Architecture_Overview.md", "size_kb": 38.4, "status": "Ready"},
                {"name": "SOC2_Compliance_Audit.pdf", "size_kb": 890.1, "status": "Ready"}
            ]
            st.dataframe(pd.DataFrame(sample_docs), width="stretch", hide_index=True)
            st.caption("Total Ingested: **3 files** (1,173.7 KB)")

# =============================================================================
# 3. TEXT EXTRACTION
# =============================================================================
with tabs[2]:
    st.header("3. PDF & Text Extraction Mechanics")
    st.markdown(r"""
    Extracting clean plaintext from binary file formats (PDFs, Word documents, Markdown) 
    using tools like `pypdf`, `pdfplumber`, or `PyMuPDF`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import io

def extract_pdf_text_sample(file_stream: io.BytesIO) -> list[dict]:
    # In production:
    # reader = pypdf.PdfReader(file_stream)
    # pages = [{"page": i + 1, "text": p.extract_text()} for i, p in enumerate(reader.pages)]
    
    # Mock extracted pages
    return [
        {"page": 1, "char_count": 1420, "text_snippet": "Enterprise AI Policy Overview..."},
        {"page": 2, "char_count": 2180, "text_snippet": "Section 4.1: Encryption Standards..."}
    ]
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Text Extraction Pipeline Best Practices

            1. **Header/Footer Stripping**: Discard recurring page headers and pagination numbers to prevent polluting embedding context.
            2. **Table Extraction**: Standard PDF extractors mangle multi-column tables. Use structured table parsers (e.g., `pdfplumber` or OCR layouts).
            3. **Encoding Normalization**: Clean unicode replacement characters and collapse extraneous whitespace.
            """)

# =============================================================================
# 4. TEXT CHUNKING
# =============================================================================
with tabs[3]:
    st.header("4. Text Chunking Strategies (Size vs Overlap)")
    st.markdown(r"""
    Documents are broken down into manageable chunks to fit into embedding models and context windows:
    - **Chunk Size ($C$)**: Target length (e.g. 500 characters or 256 tokens).
    - **Chunk Overlap ($O$)**: Shared boundary tokens (e.g. 50 tokens) to preserve contextual meaning across split boundaries.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

sample = "Large Language Models require systematic chunking to preserve context across document boundaries."
chunks = chunk_text(sample, chunk_size=35, overlap=10)
for i, c in enumerate(chunks):
    st.write(f"Chunk {i+1}: `{c}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            c_size = st.slider("Chunk Size (chars)", 50, 300, 120, step=10, key="chunk_size_sl")
            c_overlap = st.slider("Chunk Overlap (chars)", 0, 50, 20, step=5, key="chunk_over_sl")

            raw_sample = (
                "Retrieval-Augmented Generation optimizes LLM outputs by referencing an authoritative "
                "knowledge base outside of its training data sources before generating a response. "
                "Chunking ensures that semantic boundaries are respected without truncating meaning."
            )

            chunks = []
            st_idx = 0
            while st_idx < len(raw_sample):
                chunks.append(raw_sample[st_idx:st_idx + c_size])
                st_idx += max(1, (c_size - c_overlap))

            st.caption(f"Decomposed into **{len(chunks)}** overlapping chunks:")
            for i, chunk in enumerate(chunks[:3]):
                st.code(f"Chunk {i+1} [{len(chunk)} chars]:\n{chunk}")

# =============================================================================
# 5. DENSE EMBEDDINGS
# =============================================================================
with tabs[4]:
    st.header("5. Dense Semantic Vector Embeddings")
    st.markdown(r"""
    Embedding models map textual strings into dense continuous vectors in high-dimensional space:
    $$\mathbf{e} = \text{Embed}(\text{text}) \in \mathbb{R}^d$$
    Semantically similar sentences map to proximal vectors in space regardless of vocabulary overlap.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import numpy as np

# Dense vector representation (d=1536 for text-embedding-3-small)
def mock_embedding(text: str, dim: int = 4) -> np.ndarray:
    np.random.seed(abs(hash(text)) % 10000)
    vec = np.random.normal(0, 1, dim)
    return vec / np.linalg.norm(vec)  # L2 Normalized

v1 = mock_embedding("Artificial Intelligence and neural networks")
v2 = mock_embedding("Deep learning and machine learning models")
cosine_sim = np.dot(v1, v2)
st.write(f"Cosine Similarity: {cosine_sim:.3f}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Popular Embedding Models

            | Model | Dimension ($d$) | MTEB Benchmark | Cost / 1M Tokens |
            | :--- | :---: | :---: | :--- |
            | `text-embedding-3-small` | 1536 | 62.3 | $0.02 |
            | `text-embedding-3-large` | 3072 | 64.6 | $0.13 |
            | `bge-large-en-v1.5` | 1024 | 64.1 | Free (Open Source) |
            | `all-MiniLM-L6-v2` | 384 | 56.3 | Ultra-Fast Edge / CPU |
            """)

# =============================================================================
# 6. VECTOR DATABASES
# =============================================================================
with tabs[5]:
    st.header("6. Vector Databases & Approximate Nearest Neighbors")
    st.markdown(r"""
    Vector databases index high-dimensional embeddings for sub-millisecond retrieval across millions of chunks.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Vector Database Matrix**")
        st.markdown(r"""
        | Database | Type | Index Algorithm | Streamlit Integration |
        | :--- | :--- | :--- | :--- |
        | **Chroma** | Embedded / In-Memory | HNSW | `chromadb.PersistentClient()` |
        | **FAISS** | Facebook AI Library | IVF-PQ / Flat | Pure Python in `@st.cache_resource` |
        | **Qdrant** | Rust Engine / Cloud | HNSW + Filtering | `qdrant_client.QdrantClient()` |
        | **pgvector** | PostgreSQL Extension | HNSW / IVFFlat | Native SQL via `st.connection("postgresql")` |
        | **Pinecone** | Managed Cloud SaaS | Proprietary Graph | Serverless API client |
        """)

    with col2:
        st.markdown("**🖥️ In-Memory Index Status**")
        with st.container(border=True):
            st.markdown("### 🗄️ Active Vector Index HUD")
            c1, c2 = st.columns(2)
            c1.metric("Indexed Chunks", "1,420")
            c2.metric("Index Architecture", "HNSW (M=16, ef=64)")
            st.success("🟢 Vector store loaded and connected via memory reference.")

# =============================================================================
# 7. SIMILARITY SEARCH
# =============================================================================
with tabs[6]:
    st.header("7. Cosine Similarity Search ($K$-Nearest Neighbors)")
    st.markdown(r"""
    Given a user query vector $\mathbf{q}$ and document chunk vectors $\mathbf{d}_i$, 
    similarity is computed via the cosine of the angle between them:
    $$\text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

corpus_chunks = [
    {"doc": "Policy.pdf", "text": "Annual PTO entitlement is 25 days.", "score": 0.892},
    {"doc": "Policy.pdf", "text": "Health insurance covers dental and vision.", "score": 0.741},
    {"doc": "Travel.pdf", "text": "Business travel expenses require pre-approval.", "score": 0.412}
]

top_k = st.slider("Top-K Retrieved Chunks", 1, 3, 2)
retrieved = corpus_chunks[:top_k]

st.dataframe(pd.DataFrame(retrieved), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            top_k_val = st.slider("Top-K Retrieved Chunks", 1, 3, 2, key="sim_top_k_sl")
            chunks_data = [
                {"Source": "Policy.pdf", "Relevance": 0.892, "Excerpt": "Annual PTO entitlement is 25 days per calendar year."},
                {"Source": "Policy.pdf", "Relevance": 0.741, "Excerpt": "Health insurance covers comprehensive dental and vision."},
                {"Source": "Travel.pdf", "Relevance": 0.412, "Excerpt": "Business travel expenses require manager pre-approval."}
            ]
            st.caption(f"Showing Top-{top_k_val} nearest neighbor matches:")
            st.dataframe(
                pd.DataFrame(chunks_data[:top_k_val]).style.format({"Relevance": "{:.1%}"}),
                width="stretch",
                hide_index=True
            )

# =============================================================================
# 8. CONTEXT RETRIEVAL
# =============================================================================
with tabs[7]:
    st.header("8. Augmented Context Prompt Synthesis")
    st.markdown(r"""
    Formatting the retrieved chunks into a structured prompt context block 
    injected between the system instructions and the user question.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_str = "\n\n".join([
        f"--- CONTEXT CHUNK {i+1} (Source: {c['Source']}) ---\n{c['Excerpt']}"
        for i, c in enumerate(retrieved_chunks)
    ])
    
    prompt = f"""You are an enterprise knowledge assistant.

INSTRUCTIONS:
1. Answer the question STRICTLY using the context excerpts provided below.
2. If the context does not contain the answer, reply: "I cannot find this information in the provided documentation."
3. Include bracketed citations like [Source 1] matching the context sources.

CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""
    return prompt
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Synthesized Augmented Prompt Preview:")
            st.code(
                "You are an enterprise knowledge assistant.\n"
                "INSTRUCTIONS: Answer STRICTLY using provided context.\n\n"
                "--- CONTEXT CHUNK 1 (Source: Policy.pdf) ---\n"
                "Annual PTO entitlement is 25 days per calendar year.\n\n"
                "USER QUESTION: How many vacation days do I get?\n\n"
                "ANSWER:",
                language="markdown"
            )

# =============================================================================
# 9. GROUNDED ANSWER GEN
# =============================================================================
with tabs[8]:
    st.header("9. Grounded Answer Generation & Hallucination Defense")
    st.markdown(r"""
    Executing the final LLM completion with strict grounding to eliminate hallucinations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

user_query = st.text_input("Ask Question", "What is the paid leave policy?")

if user_query:
    with st.chat_message("assistant"):
        st.markdown("""
        According to the company policy [1], employees are entitled to **25 days of Paid Time Off (PTO)** 
        per calendar year. Unused days roll over up to a maximum cap of 5 days.
        """)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            u_q = st.text_input("Ask Question", "What is the paid leave policy?", key="rag_q_in")
            if u_q:
                with st.chat_message("assistant"):
                    st.markdown("""
                    According to company policy [1], employees are entitled to **25 days of Paid Time Off (PTO)** 
                    per calendar year. Unused days roll over up to a maximum cap of 5 days.
                    """)
                    st.caption("🔒 Grounded answer verified against active vector context.")

# =============================================================================
# 10. SOURCE CITATIONS
# =============================================================================
with tabs[9]:
    st.header("10. Interactive Source Citations & Document Provenance")
    st.markdown(r"""
    Enterprise credibility requires displaying collapsible provenance citation cards 
    so users can verify the exact document, page number, and similarity score behind every claim.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

citations = [
    {
        "citation_id": 1,
        "doc_name": "Employee_Handbook_2026.pdf",
        "page": 14,
        "relevance": 0.892,
        "text": "Full-time personnel accrue 2.08 days of paid vacation per month worked (25 days annual total)."
    },
    {
        "citation_id": 2,
        "doc_name": "Benefits_Summary.pdf",
        "page": 3,
        "relevance": 0.741,
        "text": "Medical, dental, and vision insurance coverage takes effect on the first calendar day of employment."
    }
]

with st.expander("📚 View 2 Source Citations & References"):
    for c in citations:
        st.markdown(f"**[{c['citation_id']}] {c['doc_name']} (Page {c['page']})** • Score: `{c['relevance']:.1%}`")
        st.caption(f'"{c["text"]}"')
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            citations = [
                {
                    "citation_id": 1,
                    "doc_name": "Employee_Handbook_2026.pdf",
                    "page": 14,
                    "relevance": 0.892,
                    "text": "Full-time personnel accrue 2.08 days of paid vacation per month worked (25 days annual total)."
                },
                {
                    "citation_id": 2,
                    "doc_name": "Benefits_Summary.pdf",
                    "page": 3,
                    "relevance": 0.741,
                    "text": "Medical, dental, and vision insurance coverage takes effect on the first day of employment."
                }
            ]

            with st.expander("📚 View 2 Source Citations & Provenance References", expanded=True):
                for c in citations:
                    st.markdown(f"**[{c['citation_id']}] `{c['doc_name']}` (Page {c['page']})** • Match: `{c['relevance']:.1%}`")
                    st.caption(f'"{c["text"]}"')

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### RAG Best Practices

| Phase | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Chunking** | Splitting purely on character counts | Use `RecursiveCharacterTextSplitter` splitting on paragraph and sentence boundaries. |
| **Embeddings** | Computing embeddings synchronously on every question | Pre-compute document embeddings once and cache in a Vector DB. |
| **Prompting** | Letting the LLM assume unmentioned facts | Enforce strict negative guardrails: *"If not in context, state 'Unknown'"*. |
| **Citations** | Giving answers without provenance | Always render collapsible citation cards with page numbers and relevance scores. |
""")
