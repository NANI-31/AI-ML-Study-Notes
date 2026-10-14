import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Optional

st.set_page_config(
    page_title="16.2 Document Embeddings - Embeddings Curriculum",
    page_icon="📄",
    layout="wide"
)

# Title & Overview
st.title("16.2 LangChain Document Embeddings: Loaders, Text Splitters, & Vector Stores")
st.caption("Chapter 16: Embeddings with LangChain | Domain: 3. Vector Embeddings")
st.markdown("""
In LangChain, unstructured data moves through a standardized ingestion quadrant before it can be queried:
1. **Document Loaders**: Read raw files (PDFs, Markdown, Webpages, CSVs) and produce standardized **`Document(page_content, metadata)`** objects.
2. **Text Splitters**: Break large documents into semantically coherent passages while preserving paragraph, sentence, and word boundaries.
3. **Embedding Models**: Vectorize text passages into dense arrays.
4. **Vector Stores**: Index vectors and document payloads into nearest-neighbor engines (`Chroma`, `FAISS`, `Qdrant`, `Pinecone`).

This lesson explores how LangChain coordinates these four primitives into an automated, production-grade ingestion pipeline.
""")

st.divider()

# Section 1: The LangChain Ingestion Quadrant Architecture
st.header("1. The LangChain Ingestion Quadrant", divider="rainbow")

col_quad1, col_quad2 = st.columns([1, 1])

with col_quad1:
    st.markdown("""
    ### The Four Ingestion Primitives
    * **1. Document Loaders (`BaseLoader`)**:
      * Parses external file formats into Python objects.
      * Produces `Document` instances containing `page_content` (the text string) and `metadata` (a dictionary tracking `source`, `page_number`, `created_at`, etc.).
    * **2. Text Splitters (`TextSplitter`)**:
      * Implements intelligent boundary-aware chunking.
      * **`RecursiveCharacterTextSplitter`**: Iteratively attempts splits along natural markdown/language boundaries: paragraphs (`\n\n`), then sentences (`\n`), then spaces (` `), and finally characters (`""`).
      * Propagates parent metadata down to all child chunks automatically.
    * **3. Embedding Models (`Embeddings`)**:
      * Calls `embed_documents()` on the extracted chunk strings.
    * **4. Vector Stores (`VectorStore`)**:
      * Exposes factory methods like `.from_documents(docs, embedding)` to ingest, index, and serialize data with a single line of code.
    """)

with col_quad2:
    st.markdown("### Ingestion Quadrant Workflow")
    ingest_dot = """
    digraph LangChainIngestion {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawFiles [label="Raw Files\\n(PDF, Markdown, HTML, Notion)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Loader [label="1. Document Loader\\nPyPDFLoader / TextLoader", fillcolor="#DBEAFE"];
        Docs [label="Document Objects\\nDocument(page_content, metadata)", fillcolor="#BFDBFE"];
        Splitter [label="2. Recursive Text Splitter\\n(chunk_size=500, overlap=50)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Chunks [label="Chunked Documents\\nPreserved metadata + chunk_id", fillcolor="#FDE68A"];
        EmbedModel [label="3. Embedding Model\\nembed_documents(chunks)", fillcolor="#ECFDF5", color="#10B981"];
        VStore [label="4. Vector Store\\nVectorStore.from_documents(...)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        RawFiles -> Loader -> Docs -> Splitter -> Chunks;
        Chunks -> EmbedModel [label="Extract texts"];
        EmbedModel -> VStore [label="Dense vectors"];
        Chunks -> VStore [label="Payload & metadata"];
    }
    """
    st.graphviz_chart(ingest_dot, use_container_width=False)

st.markdown("### Common LangChain Document Loaders")
st.markdown("""
| Loader Class | Package | Input Source | Metadata Extracted |
| :--- | :--- | :--- | :--- |
| **`TextLoader`** | `langchain-community` | Plain text (`.txt`, `.py`) | `source` (file path) |
| **`PyPDFLoader`** | `langchain-community` | PDF documents | `source`, `page` (0-indexed page number) |
| **`UnstructuredMarkdownLoader`** | `langchain-community` | Markdown (`.md`) | `source`, heading structure |
| **`DirectoryLoader`** | `langchain-community` | Entire folders of mixed files | Batch loads with glob patterns (`**/*.pdf`) |
| **`WebBaseLoader`** | `langchain-community` | Webpages / URLs | `source` (URL), title, description |
""")

st.divider()

# Section 2: Production Code: End-to-End Document Ingestion
st.header("2. Production Code: The Canonical `.from_documents` Ingestion Loop", divider="rainbow")

st.markdown("""
The production snippet below demonstrates how LangChain bundles loaders, splitters, embeddings, and vector stores into a clean 4-step pipeline.
""")

st.code(r'''
# In production environment:
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma

from typing import List, Dict, Any

class Document:
    """Standard LangChain Document representation."""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def run_document_ingestion_pipeline():
    # Step 1: Simulate loading a raw document
    raw_text = """# Service Level Agreement (SLA)
    
Enterprise customers are guaranteed 99.95% monthly uptime for all production API clusters.
Uptime calculations exclude scheduled maintenance windows announced 7 days in advance.

## Credit Schedule
If monthly uptime drops below 99.95% but remains above 99.0%, a 10% billing credit is issued.
If monthly uptime drops below 99.0%, a 25% billing credit is issued upon request."""

    loaded_doc = Document(
        page_content=raw_text,
        metadata={"source": "sla_policy_2024.md", "department": "legal", "author": "compliance_team"}
    )

    # Step 2: Split document with RecursiveCharacterTextSplitter
    # separator priority: ["\n\n", "\n", " ", ""]
    chunk_size = 200
    chunk_overlap = 40
    
    # In real LangChain:
    # splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    # chunk_docs = splitter.split_documents([loaded_doc])
    
    # Step 3 & 4: Ingest into Vector Store in a single line
    # vectorstore = Chroma.from_documents(
    #     documents=chunk_docs,
    #     embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    #     persist_directory="./chroma_sla_db"
    # )
    
    print(f"Loaded document from {loaded_doc.metadata['source']}")
    print("Ready for downstream retrieval chains!")

if __name__ == "__main__":
    run_document_ingestion_pipeline()
''', language="python")

st.divider()

# Section 3: Interactive Ingestion & Chunking Workbench
st.header("3. Interactive Workbench: Recursive Splitter & VectorStore Simulator", divider="rainbow")

st.markdown("""
Test how LangChain splits documents and populates vector stores. Customize the document source, chunk size, and overlap to see how boundaries and metadata are preserved.
""")

class RecursiveSplitterDemo:
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        # Simple recursive splitting simulation
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph itself exceeds chunk_size, split by sentences
                if len(p) > self.chunk_size:
                    sentences = p.split(". ")
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) + 2 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk}. {s}".strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = s
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.subheader("1. Ingestion Document Setup")
    doc_preset = st.radio(
        "Choose Knowledge Document:",
        ["Cloud Architecture Guide", "Human Resources Code of Conduct", "Custom Input"],
        horizontal=True
    )

    if doc_preset == "Cloud Architecture Guide":
        sample_body = (
            "# Cloud High-Availability Architecture\n\n"
            "High availability requires deploying services across at least three distinct Availability Zones. "
            "Stateless API web tiers should be hosted behind Network Load Balancers with automatic health probes.\n\n"
            "## Database Replication Tier\n\n"
            "PostgreSQL databases should utilize synchronous streaming replication for the primary standby instance. "
            "Read-only replicas can utilize asynchronous replication across secondary regions to offload analytics traffic.\n\n"
            "## Disaster Recovery Strategy\n\n"
            "Recovery Point Objective (RPO) is configured for 5 minutes via continuous WAL archiving to S3 buckets. "
            "Recovery Time Objective (RTO) is capped at 15 minutes utilizing automated Terraform failover workflows."
        )
        meta_source = "cloud_arch_manual.md"
        meta_category = "infrastructure"
    elif doc_preset == "Human Resources Code of Conduct":
        sample_body = (
            "# Enterprise Code of Conduct\n\n"
            "All employees are expected to maintain professional integrity and respect towards colleagues and partners. "
            "Harassment, discrimination, or retaliation of any form will result in immediate disciplinary review.\n\n"
            "## Conflict of Interest\n\n"
            "Employees must formally disclose outside business interests, board memberships, and external consulting agreements. "
            "Gifts from vendors exceeding $100 in market value must be reported to the corporate compliance team.\n\n"
            "## Confidentiality Obligations\n\n"
            "Proprietary customer data, trade secrets, and internal source code must never be shared on public forums or unvetted external AI tools."
        )
        meta_source = "hr_conduct_handbook.pdf"
        meta_category = "compliance"
    else:
        sample_body = "# Custom Documentation\n\nVector databases index embeddings for low-latency nearest neighbor retrieval."
        meta_source = "custom_note.txt"
        meta_category = "general"

    edit_text = st.text_area("Document Content:", value=sample_body, height=140)

    st.subheader("2. TextSplitter Configuration")
    c_size = st.slider("Chunk Size (Characters):", min_value=100, max_value=400, value=200, step=25)
    c_overlap = st.slider("Chunk Overlap (Characters):", min_value=0, max_value=60, value=30, step=5)

with col_w2:
    st.subheader("3. Split Documents & VectorStore Preview")
    
    splitter = RecursiveSplitterDemo(chunk_size=c_size, chunk_overlap=c_overlap)
    split_chunks = splitter.split_text(edit_text)

    st.success(f"Successfully generated **{len(split_chunks)}** child Document chunks!")

    doc_records = []
    for idx, c_str in enumerate(split_chunks):
        doc_records.append({
            "Chunk ID": f"chunk_{idx+1:02d}",
            "Chars": len(c_str),
            "Metadata": f"source='{meta_source}', cat='{meta_category}'",
            "Content Preview": c_str[:70] + "..."
        })

    st.dataframe(pd.DataFrame(doc_records), use_container_width=True)

    st.markdown("#### Inspect Individual Document Chunk")
    sel_chunk_idx = st.selectbox("Select chunk to view full content:", [f"chunk_{i+1:02d}" for i in range(len(split_chunks))])
    chunk_num = int(sel_chunk_idx.split("_")[1]) - 1
    
    st.markdown(f"**Full Content of `{sel_chunk_idx}`:**")
    st.code(split_chunks[chunk_num])

    st.markdown("#### Simulated `VectorStore.from_documents()` Storage")
    st.info(f"Vector Store instantiated with {len(split_chunks)} vectors. Metadata catalog initialized.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Ingestion Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is `RecursiveCharacterTextSplitter` preferred over naive `CharacterTextSplitter`?**
    * **Answer**: `CharacterTextSplitter` splits strictly on a single separator (e.g. `\n\n`). If a single paragraph is longer than `chunk_size`, it either creates an oversized chunk or cuts text abruptly in the middle of a sentence.
    `RecursiveCharacterTextSplitter` attempts a hierarchy of separators: `["\n\n", "\n", " ", ""]`. It only splits along sentence boundaries if paragraphs are too large, and only splits along word boundaries if sentences are too large, maximizing semantic integrity within the token limit.

    **Q2: How does LangChain ensure metadata is not lost when splitting large documents?**
    * **Answer**: When calling `splitter.split_documents(documents)`, the splitter extracts the `metadata` dictionary from each parent `Document` and copies it into every child `Document` generated from that parent. It often also appends chunk-specific metadata like `chunk_index` or start/end character offsets.

    **Q3: What is the difference between `VectorStore.from_documents()` and `VectorStore.from_texts()`?**
    * **Answer**:
      * `from_documents()`: Takes a `List[Document]`, automatically extracting `.page_content` for vectorization and `.metadata` for database storage.
      * `from_texts()`: Takes a raw `List[str]` and an optional parallel `List[dict]` of metadata dictionaries.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Splitting by Character Count Instead of Token Count**:
      * *Issue*: Setting `chunk_size=1000` assuming 1 character = 1 token.
      * *Consequence*: In non-English languages or code, 1000 characters can easily exceed the embedding model's 512-token context limit, resulting in silent truncation.
      * *Fix*: Configure `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` to measure chunk length strictly in model tokens rather than characters.

    * **Trap 2: Losing Document Lineage During Splitting**:
      * *Issue*: Splitting raw text without preserving the original file path or page number.
      * *Consequence*: When the LLM answers with citations, it cannot tell the user which PDF or page number the fact came from.
      * *Fix*: Always use Document Loaders that inject `source` and `page` metadata into every chunk.

    * **Trap 3: Oversized Chunk Overlap Leading to Duplicate Chunks in Top-K**:
      * *Issue*: Setting `chunk_size=200` with `chunk_overlap=150` (75% overlap).
      * *Consequence*: Adjacent chunks share almost identical text; a Top-3 vector query retrieves 3 almost identical copies of the same sentence, wasting prompt context.
      * *Fix*: Keep chunk overlap between 10% and 20% of total chunk size.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `Document(page_content, metadata)`**
      * *Definition*: LangChain's universal data class encapsulating extracted text alongside arbitrary key-value metadata attributes.
    * **Flashcard 2: `RecursiveCharacterTextSplitter`**
      * *Definition*: The industry-standard text splitter that recursively attempts splitting along paragraphs, sentences, and words to preserve semantic structure.
    * **Flashcard 3: `VectorStore.from_documents()`**
      * *Definition*: A factory method that extracts text, generates embeddings, and indexes records into a vector database in a single call.
    * **Flashcard 4: Token-Aware Splitting**
      * *Definition*: Measuring chunk size using a tokenizer (e.g. tiktoken) rather than raw string length to guarantee strict compliance with model context windows.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of LangChain Document Embeddings:

- [ ] I can trace the four primitives of the LangChain Ingestion Quadrant.
- [ ] I understand how `Document(page_content, metadata)` preserves provenance.
- [ ] I know why `RecursiveCharacterTextSplitter` is preferred over naive character splitting.
- [ ] I understand how metadata propagates from parent documents to child chunks.
- [ ] I know how to avoid duplicate context by keeping chunk overlap between 10% and 20%.
""")
