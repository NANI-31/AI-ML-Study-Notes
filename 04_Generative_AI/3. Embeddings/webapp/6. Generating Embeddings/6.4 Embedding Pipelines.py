import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.4 Embedding Pipelines - Embeddings Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("6.4 Embedding Pipelines: Ingestion, Sanitization, Chunking, Storage, & Retrieval Verification")
st.caption("Chapter 6: Generating Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In isolation, an embedding model is simply a mathematical function that converts strings to vectors.
In a production enterprise architecture, however, the model sits at the center of an **End-to-End Embedding Pipeline**.

A robust embedding pipeline coordinates **6 core sequential stages**:
1. **Data Ingestion**: Loading raw documents from PDFs, APIs, SQL databases, and Markdown feeds.
2. **Sanitization & Normalization**: Stripping HTML tags, fixing broken Unicode, removing boilerplate headers/footers, and filtering low-information noise.
3. **Chunking & Metadata Binding**: Partitioning text into token-bounded semantic blocks and attaching lineage metadata (source, author, timestamps).
4. **Batch Embedding Generation**: Parallelizing model inference with smart bucketing and unit L2 normalization.
5. **Vector Storage & Indexing**: Persisting vectors and payloads into dedicated vector databases (ChromaDB, Qdrant, Pinecone, FAISS).
6. **Retrieval & Sanity Verification**: Executing query projection and validating retrieval recall with assertion tests.

This lesson explores the architecture and engineering best practices of end-to-end embedding pipelines.
""")

st.divider()

# Section 1: The 6 Stages of an Enterprise Embedding Pipeline
st.header("1. The 6 Stages of an Enterprise Embedding Pipeline", divider="rainbow")

col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### Pipeline Stage Breakdown:
    * **Stage 1 (Load)**: Extract raw text from heterogeneous sources. Handling encoding quirks (UTF-8 vs Latin-1) and corrupted characters.
    * **Stage 2 (Clean)**: Remove noisy artifacts that degrade vector quality:
      * Strip HTML tags (`<p>`, `<div>`) and markdown tables that confuse language models.
      * Collapse multiple spaces, non-breaking spaces (`\\xa0`), and excessive newlines.
      * Filter out repetitive copyright disclaimers and cookie banners.
    * **Stage 3 (Chunk)**: Split documents into 100-300 token passages with 10-20% overlap. Crucially: keep chunk ID, parent document ID, and page numbers as metadata!
    * **Stage 4 (Embed)**: Generate normalized vectors using batching. Attach instruction prefixes (`passage: `) if required by the model.
    * **Stage 5 (Store)**: Write vectors, chunk text, and metadata payloads to the vector store with an appropriate distance metric (Cosine / Inner Product).
    * **Stage 6 (Retrieve & Verify)**: Run synthetic test queries against the index to assert that critical documents are retrievable with expected similarity thresholds.
    """)

with col_pipe2:
    st.markdown("### End-to-End Pipeline Architecture")
    pipe_dot = """
    digraph EndToEndPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawDoc [label="Stage 1: Raw Document Ingestion\\nPDFs, HTML, Markdown, SQL", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Cleaner [label="Stage 2: Sanitization & Clean\\nRegex strip, Unicode normalize", fillcolor="#FEF3C7"];
        
        Chunker [label="Stage 3: Token-Aware Chunking\\n200 tokens + metadata tags", fillcolor="#FDE68A"];
        
        BatchEmbed [label="Stage 4: Batch Embedding\\nSmart bucketing + L2 norm", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        
        VectorStore [label="Stage 5: Vector DB Storage\\nIndex (HNSW / Flat) + Payload", fillcolor="#EDE9FE", color="#8B5CF6"];
        
        VerifyEngine [label="Stage 6: Retrieval & Sanity Verification\\nTop-k test queries + recall assertions", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        RawDoc -> Cleaner -> Chunker -> BatchEmbed -> VectorStore -> VerifyEngine;
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=False)

st.divider()

# Section 2: Production Code: Complete End-to-End Pipeline Engine
st.header("2. Production Implementation: Full Pipeline Orchestrator", divider="rainbow")

st.markdown("""
The production Python class below orchestrates the entire 6-stage lifecycle: sanitizing raw messy text, splitting into chunks, generating unit-normalized vectors, storing in a mock vector index, and retrieving relevant passages for user queries.
""")

st.code(r'''
import re
import numpy as np
from typing import List, Dict, Any, Optional

class ProductionEmbeddingPipeline:
    """
    Orchestrates the 6 stages of the production embedding lifecycle:
    Load -> Clean -> Chunk -> Embed -> Store -> Retrieve.
    """
    def __init__(self, embedding_dim: int = 384, chunk_size_words: int = 40, chunk_overlap_words: int = 10):
        self.dim = embedding_dim
        self.chunk_size = chunk_size_words
        self.chunk_overlap = chunk_overlap_words
        self.vector_store: List[Dict[str, Any]] = []

    def clean_text(self, raw_text: str) -> str:
        """Stage 2: Strips HTML, excessive whitespace, and special noise."""
        text = re.sub(r'<[^>]+>', ' ', raw_text)  # Remove HTML tags
        text = re.sub(r'[\r\n\t]+', ' ', text)    # Normalize newlines
        text = re.sub(r'\s{2,}', ' ', text)       # Collapse spaces
        return text.strip()

    def chunk_text(self, clean_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Stage 3: Splits text into overlapping word chunks with lineage metadata."""
        words = clean_text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{doc_id}_c{len(chunks)}",
                "doc_id": doc_id,
                "text": chunk_str,
                "word_count": len(chunk_words)
            })
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Generates unit-normalized vectors for all chunks."""
        for item in chunks:
            # Deterministic pseudo-vector simulation
            rng = np.random.default_rng(abs(hash(item["text"])) % (2**31))
            raw_v = rng.standard_normal(self.dim).astype(np.float32)
            norm_v = raw_v / np.linalg.norm(raw_v)
            item["vector"] = norm_v
        return chunks

    def store_chunks(self, embedded_chunks: List[Dict[str, Any]]):
        """Stage 5: Persists chunks and vectors into the vector store."""
        self.vector_store.extend(embedded_chunks)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Stage 6: Projects query and retrieves nearest chunks via dot product."""
        if not self.vector_store:
            return []
        
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        q_v = rng.standard_normal(self.dim).astype(np.float32)
        norm_q = q_v / np.linalg.norm(q_v)
        
        results = []
        for item in self.vector_store:
            sim = float(np.dot(norm_q, item["vector"]))
            results.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": sim
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractivePipelineOrchestrator:
    def __init__(self, embedding_dim: int = 6):
        self.dim = embedding_dim
        self.vector_store: List[Dict[str, Any]] = []

    def clean_text(self, raw_text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', raw_text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, doc_id: str, chunk_size_words: int = 25, overlap_words: int = 5) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        step = max(1, chunk_size_words - overlap_words)
        for i in range(0, len(words), step):
            chunk_slice = words[i : i + chunk_size_words]
            chunk_str = " ".join(chunk_slice)
            chunks.append({
                "chunk_id": f"{doc_id}_c{len(chunks)}",
                "doc_id": doc_id,
                "text": chunk_str,
                "word_count": len(chunk_slice)
            })
            if i + chunk_size_words >= len(words):
                break
        return chunks

    def embed_and_store(self, chunks: List[Dict[str, Any]]):
        self.vector_store = []
        for item in chunks:
            # Deterministic vector based on words
            rng = np.random.default_rng(abs(hash(item["text"])) % (2**31))
            raw_v = rng.standard_normal(self.dim).astype(np.float32)
            norm_v = raw_v / np.linalg.norm(raw_v)
            item["vector"] = norm_v
            self.vector_store.append(item)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        q_v = rng.standard_normal(self.dim).astype(np.float32)
        norm_q = q_v / np.linalg.norm(q_v)
        
        results = []
        for item in self.vector_store:
            sim = float(np.dot(norm_q, item["vector"]))
            results.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": sim
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

st.divider()

# Section 3: Interactive Workbench: 6-Stage Pipeline Simulator
st.header("3. Interactive Workbench: 6-Stage Embedding Pipeline Simulator", divider="rainbow")

st.markdown("""
Run a complete embedding pipeline step-by-step! Start with raw, messy, uncleaned input text containing HTML and redundant whitespace. Watch how the text is sanitized, chunked, converted to unit vectors, indexed, and retrieved.
""")

raw_default_doc = """<div class="article-content">
<h1>Production Vector Search Architecture</h1>
<p>In modern enterprise RAG systems,    <b>vector embeddings</b> convert textual tokens into dense mathematical coordinates on a unit hypersphere.</p>
<p>Data preprocessing is essential. Raw documents frequently contain <i>noisy HTML markup</i>, broken line breaks, and irrelevant cookie banners that corrupt vector geometry.</p>
<div>After sanitization, texts are partitioned into 200-token chunks with 10% overlap to preserve semantic context across chunk boundaries.</div>
<p>Finally, embeddings are persisted in a vector database like ChromaDB, Qdrant, or Pinecone for low-latency cosine similarity retrieval.</p>
</div>"""

orch = InteractivePipelineOrchestrator(embedding_dim=6)

col_pipe_in, col_pipe_out = st.columns([1, 1])

with col_pipe_in:
    st.subheader("1. Ingestion & Preprocessing Controls")
    doc_input = st.text_area("Stage 1: Raw Ingested Document (with HTML & Noise):", value=raw_default_doc, height=180)
    
    chunk_sz = st.slider("Stage 3 Chunk Size (Words):", min_value=15, max_value=60, value=25, step=5)
    overlap_sz = st.slider("Stage 3 Chunk Overlap (Words):", min_value=0, max_value=15, value=5, step=1)
    
    clean_txt = orch.clean_text(doc_input)
    chunks_gen = orch.chunk_text(clean_txt, doc_id="DOC_001", chunk_size_words=chunk_sz, overlap_words=overlap_sz)
    orch.embed_and_store(chunks_gen)

    with st.expander("View Stage 2: Sanitized Clean Text"):
        st.write(clean_txt)

    st.markdown(f"#### Stage 3: Generated Chunks ({len(chunks_gen)} chunks)")
    df_chunks = pd.DataFrame([{"Chunk ID": c["chunk_id"], "Word Count": c["word_count"], "Snippet": c["text"][:60] + "..."} for c in chunks_gen])
    st.dataframe(df_chunks, use_container_width=True)

with col_pipe_out:
    st.subheader("2. Vector Storage & Retrieval Verification")
    st.markdown("#### Stage 5: Vector Store Catalog")
    df_store = pd.DataFrame([
        {
            "Chunk ID": item["chunk_id"],
            "Dim": len(item["vector"]),
            "L2 Norm": f"{np.linalg.norm(item['vector']):.4f}",
            "Preview Vector": f"[{item['vector'][0]:.2f}, {item['vector'][1]:.2f}, ...]"
        }
        for item in orch.vector_store
    ])
    st.dataframe(df_store, use_container_width=True)

    st.markdown("#### Stage 6: Retrieval Verification Test")
    test_q = st.selectbox(
        "Select Verification Query:",
        [
            "How do vector embeddings represent text on a hypersphere?",
            "Why is HTML sanitization necessary before chunking?",
            "What vector databases store chunks for low-latency retrieval?"
        ]
    )
    
    hits = orch.retrieve(test_q, top_k=2)
    df_hits = pd.DataFrame(hits)
    df_hits.rename(columns={"chunk_id": "Chunk ID", "text": "Retrieved Text", "score": "Cosine Similarity"}, inplace=True)
    
    st.dataframe(
        df_hits.style.format({"Cosine Similarity": "{:.4f}"}).background_gradient(subset=["Cosine Similarity"], cmap="Blues"),
        use_container_width=True
    )
    st.success(f"✅ **Pipeline Executed Successfully**: Retrieved top match `{hits[0]['chunk_id']}` with cosine score **{hits[0]['score']:.4f}**.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Pipeline Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is text sanitization (stripping HTML, boilerplate, unicode normalization) vital before embedding generation?**
    * **Answer**:
      * Embedding models are sensitive to token sequences.
      * If documents contain repetitive HTML tags (`<div class="header">`), CSS snippets, or legal boilerplate disclaimers, the tokenizer spends a large portion of its positional context encoding syntactic noise.
      * In mean pooling, the noise tokens contribute directly to the final vector, pulling all documents toward an artificial "HTML boilerplate" cluster and degrading semantic retrieval precision.

    **Q2: How does chunk overlap prevent information loss at boundary splits?**
    * **Answer**:
      * In hard non-overlapping chunking, a crucial sentence or proposition might be split right down the middle across Chunk A and Chunk B (e.g. subject in Chunk A, verb and object in Chunk B).
      * Neither chunk has the complete proposition, causing both chunks to fail vector retrieval for queries targeting that fact.
      * Introducing a 10% to 20% overlap (e.g. 20-40 tokens) guarantees that boundary sentences appear in full within at least one chunk.

    **Q3: What metadata should always be saved alongside embedding vectors in a production vector database?**
    * **Answer**:
      * 1. **Document Lineage**: `doc_id`, `parent_doc_id`, file path, URL, or database primary key.
      * 2. **Positional Tracking**: `chunk_index`, page number, start/end character offsets (enables highlighting in UI).
      * 3. **Temporal Tracking**: `created_at`, `updated_at`, document version (enables time-based filtering and TTL invalidation).
      * 4. **Access Control**: User permissions, tenant ID, security clearance tags (enables pre-filtering for multi-tenant systems).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Storing Embeddings without Raw Text Payloads**:
      * *Issue*: Saving only the vector and chunk ID in the database, without storing the raw text chunk.
      * *Consequence*: After retrieving nearest neighbor vector IDs, you must make a secondary slow SQL/S3 roundtrip to fetch the text, doubling retrieval latency.
      * *Fix*: Store text payloads directly inside the vector database collection.

    * **Trap 2: Ignoring Document Updates (Zombie Vectors)**:
      * *Issue*: Re-ingesting an updated document without deleting old chunks belonging to the previous version.
      * *Consequence*: Vector database contains stale, duplicate, or contradictory vectors that corrupt LLM responses.
      * *Fix*: Enforce transactional upserts or delete all previous chunks matching `parent_doc_id` before ingesting new versions.

    * **Trap 3: Inconsistent Normalization Between Ingestion and Querying**:
      * *Issue*: Sanitizing and lowercasing text during document ingestion, but passing raw uncleaned user queries with trailing punctuation to the model.
      * *Consequence*: Causes unnecessary semantic drift between query and passage vectors.
      * *Fix*: Route queries through the identical text cleaning function used during ingestion.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Ingestion Pipeline**
      * *Definition*: The multi-stage software workflow that loads, cleans, chunks, embeds, and indexes textual data into a searchable vector database.
    * **Flashcard 2: Chunk Overlap**
      * *Definition*: A rolling window buffer (typically 10-20% of chunk size) that prevents semantic truncation of sentences that straddle chunk boundaries.
    * **Flashcard 3: Payload Metadata**
      * *Definition*: Structured non-vector attributes (doc ID, chunk text, timestamps, author) stored alongside vectors in vector database records.
    * **Flashcard 4: Zombie Vectors**
      * *Definition*: Obsolete embeddings left orphaned in a vector index after a document is updated or deleted, polluting retrieval results.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embedding Pipelines:

- [ ] I can describe the 6 sequential stages of an enterprise embedding pipeline.
- [ ] I understand why HTML and noise sanitization is essential before tokenization.
- [ ] I know how chunk overlap protects against boundary proposition truncation.
- [ ] I know which critical metadata fields (lineage, timestamps, permissions) must accompany vectors.
- [ ] I can articulate strategies to prevent zombie vectors and handle document updates in vector stores.
""")
