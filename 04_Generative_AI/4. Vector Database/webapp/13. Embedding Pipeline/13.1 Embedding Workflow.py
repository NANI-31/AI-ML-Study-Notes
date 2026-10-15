import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="13.1 Embedding Workflow - Vector DB Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("13.1 Embedding Workflow: The 7-Stage End-to-End Ingestion Pipeline")
st.caption("Chapter 13: Embedding Pipeline | Domain: 4. Vector Database")
st.markdown("""
An enterprise vector database is not filled by running ad-hoc scripts; it relies on an automated, resilient, and 
observable **7-Stage Ingestion ETL (Extract, Transform, Load) Pipeline**.

From raw file ingestion to real-time search availability, every document must pass through a strict sequence of transformations:
1. **Load Documents**: Extract raw text and structural metadata across formats (PDF, DOCX, CSV, SQL).
2. **Clean Documents**: Strip boilerplate, fix mojibake encoding corruption, and normalize Unicode.
3. **Split Documents**: Chunk text hierarchically using recursive splitting with contextual overlap.
4. **Generate Embeddings**: Pass text chunks through bi-encoder transformer models (e.g., OpenAI, Cohere, BGE).
5. **Store Embeddings & Documents**: Persist raw vectors and hydrated text to disk.
6. **Store Metadata**: Record structured provenance attributes in SQLite.
7. **Index Vectors**: Insert vectors into the multi-layer HNSW graph index for immediate sub-millisecond retrieval.

In this lesson, we break down each pipeline stage, architect an enterprise Python ingestion orchestrator, 
and test an interactive end-to-end Ingestion Pipeline Simulator.
""")

st.divider()

# Section 1: The Seven-Stage Ingestion Pipeline Architecture
st.header("1. The Seven-Stage Ingestion Pipeline Architecture", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1.1, 0.9])

with col_pipe1:
    st.markdown("""
    ### The 7 Discrete Ingestion Phases
    
    1. **Stage 1: Ingestion & Extraction (Load)**:
       * Reads multi-format binary streams and extracts textual content and layout metadata.
    2. **Stage 2: Sanitization (Clean)**:
       * Normalizes typography via `NFKC`, removes running headers/footers, and strips script tags.
    3. **Stage 3: Segmentation (Split)**:
       * Splits sanitized text into 400-600 token chunks with 15% sliding overlap to preserve cross-boundary context.
    4. **Stage 4: Vectorization (Embed)**:
       * Batches text chunks (e.g. 100 at a time) and transmits them to embedding inference engines.
    5. **Stage 5 & 6: Dual Storage (Vectors + Metadata)**:
       * Writes records to SQLite with ACID WAL durability, storing IDs, text payloads, and scalar metadata.
    6. **Stage 7: Graph Compilation (Index)**:
       * Connects new vectors into the HNSW skip-graph at layers `0 .. L_max` using `efConstruction`.
    """)

with col_pipe2:
    st.markdown("### End-to-End Ingestion Flow")
    flow_dot = """
    digraph SevenStagePipeline {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        S1 [label="1. Load Document\\nExtract raw text + format", fillcolor="#EFF6FF", color="#3B82F6"];
        S2 [label="2. Clean Text\\nNFKC, mojibake, whitespace", fillcolor="#FEF3C7", color="#D97706"];
        S3 [label="3. Split Chunks\\nRecursive 500 tokens + 75 overlap", fillcolor="#FEF3C7", color="#D97706"];
        S4 [label="4. Generate Embeddings\\nBatch Inference (OpenAI / BGE)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        S5 [label="5 & 6. Persist Store\\nSQLite WAL + Segment binaries", fillcolor="#E0E7FF", color="#4338CA"];
        S7 [label="7. HNSW Graph Index\\nWire M edges into skip graph", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        S1 -> S2;
        S2 -> S3;
        S3 -> S4;
        S4 -> S5;
        S5 -> S7;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# Section 2: Production Ingestion Pipeline Orchestrator in Python
st.header("2. Production Ingestion Pipeline Orchestrator in Python", divider="rainbow")

st.markdown("""
In enterprise software, an ingestion script must incorporate **batching, retry logic with exponential backoff, and progress tracking**:
""")

st.code("""import time
import chromadb
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class ProductionIngestionPipeline:
    \"\"\"Orchestrates document loading, sanitization, chunking, and ChromaDB insertion.\"\"\"
    
    def __init__(self, collection, batch_size: int = 100):
        self.collection = collection
        self.batch_size = batch_size

    def clean_text(self, text: str) -> str:
        # Standardized normalization
        import unicodedata, re
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        # Recursive splitting logic
        paragraphs = text.split("\\n\\n")
        chunks, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) <= chunk_size:
                current = (current + "\\n\\n" + p).strip()
            else:
                if current: chunks.append(current)
                current = p
        if current: chunks.append(current)
        return chunks

    def ingest_document(self, doc_id: str, raw_text: str, source_meta: Dict[str, Any]):
        # 1. Clean
        cleaned = self.clean_text(raw_text)
        # 2. Chunk
        chunks = self.chunk_document(cleaned)
        
        # 3. Prepare Batch Payloads
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metas = []
        for i in range(len(chunks)):
            m = source_meta.copy()
            m["chunk_index"] = i
            m["total_chunks"] = len(chunks)
            m["ingested_epoch"] = int(time.time())
            metas.append(m)
            
        # 4. Atomic Batch Upsert into ChromaDB
        for i in range(0, len(chunks), self.batch_size):
            b_ids = ids[i : i + self.batch_size]
            b_docs = chunks[i : i + self.batch_size]
            b_metas = metas[i : i + self.batch_size]
            
            # Chroma automatically handles embedding generation and HNSW indexing
            self.collection.upsert(
                ids=b_ids,
                documents=b_docs,
                metadatas=b_metas
            )
            print(f"Upserted batch {i // self.batch_size + 1} ({len(b_ids)} chunks)")
""", language="python")

st.divider()

# Section 3: ChromaDB's Native Ingestion Mechanics
st.header("3. ChromaDB's Native Ingestion Mechanics", divider="rainbow")

col_mech1, col_mech2 = col_mech1, col_mech2 = st.columns([1, 1])

with col_mech1:
    st.markdown("""
    ### What Happens When You Pass `documents=[...]`
    When you call `collection.add(documents=[...])` without supplying pre-computed `embeddings`:
    1. **Embedding Function Hook**: Chroma automatically invokes its configured `EmbeddingFunction` (e.g. `DefaultEmbeddingFunction` or `OpenAIEmbeddingFunction`).
    2. **Local Tokenization**: Batches the texts, converts them to tokens, and passes them to the model weights.
    3. **Vector Normalization**: Automatically normalizes vectors if the distance metric is `cosine`.
    4. **SQLite Insertion**: Documents and metadata are inserted into SQLite tables with primary keys matching `ids`.
    5. **HNSW Adjacency Insertion**: Calls C++ hnswlib routines to wire each vector into the graph.
    """)

with col_mech2:
    st.markdown("""
    ### Passing Pre-Computed Embeddings
    If you have a dedicated embedding cluster (e.g., high-throughput GPU Triton Server or vLLM), you can pass pre-computed float arrays:
    ```python
    collection.add(
        ids=["doc_1", "doc_2"],
        embeddings=[
            [0.082, -0.145, ..., 0.091],
            [0.104, -0.099, ..., 0.052]
        ],
        documents=["Raw text 1", "Raw text 2"],
        metadatas=[{"dept": "AI"}, {"dept": "Security"}]
    )
    ```
    * **Advantage**: Bypasses Chroma's local embedding engine, cutting CPU load by 95% on the Chroma server.
    """)

st.divider()

# Section 4: Interactive Workbench: End-to-End Ingestion Pipeline Simulator
st.header("4. Interactive Workbench: End-to-End Ingestion Pipeline Simulator", divider="rainbow")

st.markdown("""
Execute an interactive step-by-step ingestion run on a simulated enterprise policy document. 
Observe the transformation from raw file input to ChromaDB index registration.
""")

col_sim_ctl, col_sim_view = st.columns([1, 1.2])

with col_sim_ctl:
    st.subheader("1. Configure Ingestion Task")
    
    doc_title = st.text_input("Document Title:", value="SOC2 Security Incident Response Handbook")
    doc_dept = st.selectbox("Department:", ["Security Operations", "DevOps", "HR", "Legal"])
    clearance_level = st.slider("Security Clearance Level:", min_value=1, max_value=4, value=2)
    
    raw_content = st.text_area(
        "Source Document Body:",
        value="""CONFIDENTIAL - INTERNAL AUDIT ONLY
Page 1 of 8

1.0 Incident Severity Classification
Severity 1 incidents are defined as unauthorized exfiltration of customer PII, database corruption, or complete loss of primary production availability.

In the event of a Severity 1 incident, the Incident Commander must assemble the triage squad within 15 minutes of detection and establish a dedicated Slack war room.

External communications to affected enterprise clients must be vetted by Legal and dispatched within 72 hours under GDPR Article 33 compliance mandates.""",
        height=180
    )
    
    trigger_ingest = st.button("🚀 Run Ingestion Pipeline", use_container_width=True)

with col_sim_view:
    st.subheader("2. Step-by-Step Pipeline Execution")
    
    if trigger_ingest:
        with st.status("Executing 7-Stage Ingestion Pipeline...", expanded=True) as status:
            st.write("🟢 **Stage 1 (Load)**: Extracted raw text (384 chars) from virtual PDF source.")
            time.sleep(0.3)
            
            # Cleaning
            cleaned_text = raw_content.replace("CONFIDENTIAL - INTERNAL AUDIT ONLY", "").replace("Page 1 of 8", "").strip()
            st.write("🟢 **Stage 2 (Clean)**: Stripped running headers & page numbers (-42 chars).")
            time.sleep(0.3)
            
            # Chunking
            paras = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
            st.write(f"🟢 **Stage 3 (Split)**: Generated **{len(paras)} semantic chunks** with preserved paragraph boundaries.")
            time.sleep(0.3)
            
            st.write("🟢 **Stage 4 (Embed)**: Generated 1,536-dim embeddings via `text-embedding-3-small` (Latency: 28ms).")
            time.sleep(0.3)
            
            st.write("🟢 **Stage 5 & 6 (Store)**: Committed 3 records to SQLite WAL with clearance metadata.")
            time.sleep(0.3)
            
            st.write("🟢 **Stage 7 (Index)**: Added 3 nodes to HNSW Layer 0 with M=16 edges.")
            status.update(label="✅ Ingestion Complete: All 7 Stages Succeeded!", state="complete")
        
        st.subheader("3. Registered ChromaDB Collection Payloads")
        for i, p in enumerate(paras):
            st.markdown(f"**Chunk ID**: `soc2_doc_chunk_{i}`")
            st.code(f"""Text: "{p[:80]}..."
Metadata: {{
    "title": "{doc_title}",
    "department": "{doc_dept}",
    "clearance": {clearance_level},
    "chunk_index": {i},
    "ingested_epoch": {int(time.time())}
}}""", language="json")
    else:
        st.info("Click 'Run Ingestion Pipeline' to trigger the 7-stage simulation.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the optimal batch size when ingesting documents into ChromaDB?"):
    st.markdown("""
    **Answer**:
    * **Too Small (1 - 10 records)**: High per-request overhead, excessive SQLite transaction commits, and thread synchronization bottlenecks.
    * **Too Large (10,000+ records)**: Exceeds network timeout limits, triggers embedding API rate limits (e.g. OpenAI 8,192 token batch limit), and creates massive memory spikes during vector serialization.
    * **Optimal Production Batch Size**: **100 to 500 records** (or ~50,000 to 100,000 tokens per batch). This balances CPU memory consumption, amortizes SQLite WAL commit locks, and maximizes throughput.
    """)

with st.expander("Q2: Why should document IDs be deterministic hashes rather than random UUIDs?"):
    st.markdown("""
    **Answer**:
    Using deterministic IDs (e.g., `id = sha256(file_path + chunk_index)`) provides native **idempotency**:
    1. **Safe Retries**: If an ingestion pipeline crashes halfway through a 10,000-page job and restarts, calling `upsert()` with deterministic IDs safely overwrites existing chunks rather than creating duplicate vector records.
    2. **Update Tracking**: When source documents are modified, re-running ingestion cleanly updates the existing vector in-place.
    3. Random UUIDs create duplicate vectors every time a file is re-ingested, resulting in index bloat and duplicate search hits.
    """)

with st.expander("Q3: How do you handle embedding API rate limits (HTTP 429) during high-throughput ingestion?"):
    st.markdown("""
    **Answer**:
    1. **Exponential Backoff with Jitter**: When an API returns HTTP 429 (Too Many Requests), pause execution for `2^attempt + random_jitter_seconds` before retrying (using libraries like `tenacity`).
    2. **Token Bucket Rate Limiting**: Track token consumption locally using an in-memory token bucket to throttle requests *before* hitting the cloud provider's RPM (Requests Per Minute) or TPM (Tokens Per Minute) threshold.
    3. **Asynchronous Worker Pools**: Regulate the number of concurrent worker threads (`ThreadPoolExecutor(max_workers=4)`) based on provider tier limits.
    """)

with st.expander("Q4: What happens if an embedding generation call succeeds for half a batch and fails for the rest?"):
    st.markdown("""
    **Answer**:
    In a poorly designed pipeline, partial batch failure causes orphaned records or skipped documents. 
    **Production Best Practice**: 
    * Decouple embedding generation from database insertion.
    * Collect embeddings for the entire batch into memory.
    * Only when the entire batch of embeddings has verified shapes (`len(embeddings) == len(ids)` and `shape == (1536,)`), execute an atomic `collection.upsert()` transaction.
    * If embedding fails, reject the entire batch and route it to a Dead-Letter Queue (DLQ) for retry.
    """)

with st.expander("Q5: Why is pre-computing embeddings on a separate worker service preferred over ChromaDB's in-process EmbeddingFunction?"):
    st.markdown("""
    **Answer**:
    ChromaDB's native `EmbeddingFunction` runs synchronously inside the database process. 
    * For local models (e.g. Sentence-Transformers), running PyTorch inference inside the Chroma server consumes 100% of the CPU/GPU, starving the HNSW search engine and causing latency spikes for ongoing user search queries.
    * In production, **embedding generation should be offloaded to worker pods** (e.g., Celery / Ray workers). The workers generate vectors and pass ready-made float arrays to Chroma via `collection.add(embeddings=[...])`.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Random UUIDs for Repeated Ingestions**  
    Generating `id = str(uuid.uuid4())` causes re-ingested documents to duplicate in the vector database. Use deterministic composite IDs: `f"{doc_hash}_{chunk_idx}"`.
    """)
    st.warning("""
    **Trap 2: Ignoring Embedding Model Dimension Mismatches**  
    Ingesting 768-dim embeddings into a collection initialized for 1,536 dims crashes ChromaDB with `InvalidDimensionException`. Always lock collection dimension metadata.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unbounded Concurrent API Ingestion**  
    Spawning 100 threads to call OpenAI embedding APIs simultaneously triggers immediate HTTP 429 rate limit blocks. Cap concurrency to 4-8 threads with retry backoff.
    """)
    st.warning("""
    **Trap 4: Ingesting Without Text Backing**  
    Saving embeddings with `collection.add(ids=..., embeddings=...)` without passing `documents=[...]` means Chroma cannot return the text payload during retrieval. Always store the hydrated text.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What are the 7 stages of ingestion?"):
        st.markdown("""
        **The 7 Stages**:  
        1. Load -> 2. Clean -> 3. Split -> 4. Embed -> 5. Store Vectors -> 6. Store Metadata -> 7. Index Graph.
        """)
    with st.expander("🗂️ Flashcard: Why use deterministic IDs?"):
        st.markdown("""
        **Idempotency**:  
        Prevents duplicate records when re-ingesting modified documents or retrying failed pipeline runs.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the optimal ingestion batch size?"):
        st.markdown("""
        **Optimal Batch**:  
        100 to 500 chunks (~50k to 100k tokens), balancing memory, API limits, and SQLite commit locks.
        """)
    with st.expander("🗂️ Flashcard: How to prevent 429 rate limit crashes?"):
        st.markdown("""
        **Backoff & Jitter**:  
        Use exponential backoff with jitter and a token bucket rate limiter to throttle requests.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know all 7 stages of the enterprise vector database ingestion workflow.", value=True)
chk2 = st.checkbox("I can implement a resilient Python ingestion orchestrator with batching and error handling.", value=True)
chk3 = st.checkbox("I understand why deterministic IDs guarantee idempotency and prevent duplicate vector records.", value=True)
chk4 = st.checkbox("I know how ChromaDB handles embedding generation internally vs pre-computed vectors.", value=True)
chk5 = st.checkbox("I can configure exponential backoff and token bucket rate limits for cloud embedding APIs.", value=True)
chk6 = st.checkbox("I can design a production ingestion pipeline that decouples vectorization from storage.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Embedding Workflow and are ready for Chapter 13.2: Query Pipeline!")
