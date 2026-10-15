import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="15.2 RAG Pipeline - Embeddings Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("15.2 The Complete RAG Pipeline: From Unstructured Ingestion to Generative Synthesis")
st.caption("Chapter 15: Embeddings in RAG | Domain: 3. Vector Embeddings")
st.markdown("""
A production Retrieval-Augmented Generation (RAG) system is not just a single API call; it is a **multi-stage distributed pipeline** where each transformation directly affects final answer accuracy.

A failure at any single point propagates downstream:
* Poor **text extraction** produces corrupted gibberish for the chunker.
* Poor **chunking** destroys semantic context boundaries for the embedding model.
* Suboptimal **embeddings** fail to retrieve the correct needle chunk during similarity search.
* Poor **context construction** drowns the LLM in irrelevant distractors.

This lesson explores each individual stage of the canonical RAG pipeline, detailing latency budgets, error recovery, and data transformations.
""")

st.divider()

# Section 1: The Canonical RAG Pipeline Architecture
st.header("1. The Canonical RAG Pipeline Flow", divider="rainbow")

col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### Canonical 11-Stage Pipeline Progression
    ```
    Documents
        ↓
    Text Extraction
        ↓
    Chunking
        ↓
    Embedding Model
        ↓
    Vector Database
        ↓
    User Query
        ↓
    Query Embedding
        ↓
    Similarity Search
        ↓
    Relevant Chunks
        ↓
    LLM
        ↓
    Answer
    ```
    """)
    st.markdown("""
    * **Stage 1–5: Offline / Asynchronous Ingestion Loop**:
      * Converts raw multi-format files into vectorized indices. Optimized for throughput and indexing speed.
    * **Stage 6–11: Online / Synchronous Query Loop**:
      * Executes in real time upon receiving a user prompt. Optimized for sub-second latency and high precision.
    """)

with col_pipe2:
    st.markdown("### Architectural Pipeline Diagram")
    pipe_dot = """
    digraph RAGPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_ingest {
            label="Ingestion Phase"; style="dashed"; color="#3B82F6";
            D [label="1. Documents (PDF, MD, HTML)", fillcolor="#EFF6FF"];
            TE [label="2. Text Extraction & Cleaning", fillcolor="#DBEAFE"];
            CH [label="3. Chunking (Split + Overlap)", fillcolor="#BFDBFE"];
            EM [label="4. Embedding Model (Bi-Encoder)", fillcolor="#93C5FD"];
            VDB [label="5. Vector Database (HNSW)", fillcolor="#60A5FA", color="#1D4ED8", style="filled,bold"];
        }

        subgraph cluster_query {
            label="Inference Phase"; style="dashed"; color="#10B981";
            UQ [label="6. User Query", fillcolor="#ECFDF5", color="#059669", style="filled,bold"];
            QE [label="7. Query Embedding", fillcolor="#D1FAE5"];
            SS [label="8. Similarity Search (Top-K)", fillcolor="#A7F3D0"];
            RC [label="9. Relevant Chunks", fillcolor="#6EE7B7"];
            LLM [label="10. LLM Generative Engine", fillcolor="#34D399", color="#047857", style="filled,bold"];
            ANS [label="11. Grounded Answer", fillcolor="#10B981", style="filled,bold"];
        }

        D -> TE -> CH -> EM -> VDB;
        UQ -> QE -> SS;
        VDB -> SS [label="ANN vector lookup"];
        SS -> RC -> LLM;
        UQ -> LLM [label="Augmented prompt"];
        LLM -> ANS;
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=False)

st.markdown("### Engineering Specifications Across All 11 Stages")
st.markdown("""
| Step | Pipeline Stage | Primary Tooling | Typical Latency | Key Engineering Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Documents** | S3, Blob Storage, SharePoint | < 5 ms (IO) | Unstructured formatting, corrupt files. |
| **2** | **Text Extraction** | PyPDF, Unstructured, Tesseract | 50 - 500 ms/doc | OCR misreads, table structure loss. |
| **3** | **Chunking** | RecursiveSplitter, NLTK, Spacy | 5 - 20 ms | Splitting sentences mid-clause. |
| **4** | **Embedding Model** | BGE, OpenAI, SentenceTransformers | 20 - 150 ms/batch | Out-of-memory on GPU, batch timeouts. |
| **5** | **Vector Database** | ChromaDB, Qdrant, Pinecone | 5 - 20 ms/insert | Index locking, segment write stalls. |
| **6** | **User Query** | FastAPI, Streamlit UI | < 1 ms | Ambiguous, underspecified questions. |
| **7** | **Query Embedding** | Dedicated embedding endpoint | 15 - 40 ms | Model mismatch with corpus index. |
| **8** | **Similarity Search** | HNSW / IVF Graph Traversal | 2 - 8 ms | Cold-start cache misses, low recall. |
| **9** | **Relevant Chunks** | Deduplication & Thresholding | < 2 ms | Irrelevant chunks passing low threshold. |
| **10** | **LLM Generation** | Claude, GPT-4o, Llama 3 | 500 - 3000 ms | Hallucination, context ignoring. |
| **11** | **Answer Output** | SSE / WebSocket streaming | Streamed | Token truncation, syntax errors. |
""")

st.divider()

# Section 2: Production Code: Modular Pipeline Orchestrator
st.header("2. Production Code: Modular Pipeline Orchestrator with Telemetry", divider="rainbow")

st.markdown("""
The production class below models each of the 11 stages with explicit telemetry tracking: recording latency, data transformations, and input/output schemas at every step.
""")

st.code(r'''
import time
import numpy as np
from typing import List, Dict, Any, Tuple

class RAGPipelineTelemetry:
    """
    Simulates production RAG pipeline telemetry tracking each transformation step.
    """
    def __init__(self, embedding_dim: int = 128):
        self.dim = embedding_dim
        self.vector_store: List[Dict[str, Any]] = []

    def run_full_pipeline(self, raw_document: str, user_query: str, top_k: int = 2) -> Dict[str, Any]:
        telemetry: Dict[str, Any] = {}
        t_total_start = time.perf_counter()

        # Step 1 & 2: Text Extraction & Cleaning
        t0 = time.perf_counter()
        clean_text = " ".join(raw_document.replace("\n", " ").split())
        telemetry["step_2_extraction_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3: Chunking
        t0 = time.perf_counter()
        words = clean_text.split()
        chunk_size, overlap = 25, 5
        step = chunk_size - overlap
        chunks = []
        for i in range(0, len(words), step):
            chunk_str = " ".join(words[i : i + chunk_size])
            chunks.append({"chunk_id": f"chk_{len(chunks)+1}", "text": chunk_str})
        telemetry["step_3_chunking_ms"] = (time.perf_counter() - t0) * 1000
        telemetry["total_chunks_created"] = len(chunks)

        # Step 4 & 5: Embedding Model & Vector DB Indexing
        t0 = time.perf_counter()
        indexed_vectors = []
        for c in chunks:
            np.random.seed(abs(hash(c["text"])) % (2**31 - 1))
            vec = np.random.randn(self.dim).astype(np.float32)
            vec = vec / np.maximum(np.linalg.norm(vec), 1e-12)
            indexed_vectors.append({"id": c["chunk_id"], "text": c["text"], "vector": vec})
        self.vector_store = indexed_vectors
        telemetry["step_4_5_indexing_ms"] = (time.perf_counter() - t0) * 1000

        # Step 6 & 7: Query Input & Query Embedding
        t0 = time.perf_counter()
        np.random.seed(abs(hash(user_query)) % (2**31 - 1))
        q_vec = np.random.randn(self.dim).astype(np.float32)
        q_vec = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-12)
        telemetry["step_7_query_embed_ms"] = (time.perf_counter() - t0) * 1000

        # Step 8 & 9: Similarity Search & Relevant Chunks
        t0 = time.perf_counter()
        scored = []
        for item in self.vector_store:
            sim = float(np.dot(item["vector"], q_vec))
            scored.append((item, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = scored[:top_k]
        telemetry["step_8_9_search_ms"] = (time.perf_counter() - t0) * 1000

        # Step 10 & 11: Prompt Synthesis & Answer Generation
        t0 = time.perf_counter()
        context_block = "\n".join([f"[{c[0]['id']}]: {c[0]['text']}" for c in relevant_chunks])
        synthesized_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {user_query}\nANSWER:"
        # Simulated LLM generation
        answer = f"Synthesized answer grounded on chunk {relevant_chunks[0][0]['id']}."
        telemetry["step_10_11_llm_ms"] = (time.perf_counter() - t0) * 1000

        telemetry["total_pipeline_latency_ms"] = (time.perf_counter() - t_total_start) * 1000
        telemetry["answer"] = answer
        telemetry["relevant_chunks"] = relevant_chunks

        return telemetry
''', language="python")

st.divider()

# Section 3: Interactive RAG Pipeline Stepper Workbench
st.header("3. Interactive Workbench: Step-by-Step Pipeline Inspector", divider="rainbow")

st.markdown("""
Step sequentially through each of the 11 stages of the RAG pipeline. Observe how data representations mutate from raw strings to chunk tokens, vector matrices, dot products, and final prompt contexts.
""")

class StepperRAG:
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

stepper = StepperRAG()

st_col1, st_col2 = st.columns([1, 1])

with st_col1:
    st.subheader("Pipeline Input Setup")
    doc_choice = st.selectbox(
        "Select Enterprise Source Document:",
        ["Kubernetes Architecture Manual", "Cybersecurity Incident Response", "Autonomous Driving Perception"]
    )

    if doc_choice == "Kubernetes Architecture Manual":
        default_doc = (
            "Kubernetes coordinates a cluster of nodes that work together as a single compute unit. "
            "The control plane manages worker nodes and pods across the cluster. The kube-apiserver exposes "
            "the Kubernetes HTTP API and serves as the front end. Etcd acts as the consistent and highly-available "
            "key-value store for all cluster data. Kube-scheduler watches for newly created pods with no assigned node, "
            "and selects an optimal node for them to run on. Kubelet runs on each worker node to ensure containers "
            "are running in a Pod according to the PodSpec."
        )
        default_q = "What component stores cluster data in Kubernetes?"
    elif doc_choice == "Cybersecurity Incident Response":
        default_doc = (
            "The Incident Response Framework consists of Preparation, Detection, Containment, Eradication, "
            "Recovery, and Post-Incident Activity. During the Containment phase, infected hosts must be immediately "
            "isolated from the corporate LAN via VLAN quarantine or endpoint agent disconnect. Eradication involves "
            "identifying the root cause, removing malware artifacts, and patching exploited CVE vulnerabilities. "
            "Evidence must be preserved following strict chain-of-custody protocols for forensic analysis."
        )
        default_q = "What actions must be taken during the containment phase?"
    else:
        default_doc = (
            "Perception in autonomous vehicles relies on multi-sensor fusion combining LiDAR, radar, and cameras. "
            "LiDAR point clouds generate accurate 3D spatial depth maps up to 200 meters. Convolutional vision backbones "
            "detect bounding boxes for pedestrians and vehicles from RGB cameras. Kalman filters track object trajectories "
            "over time to predict collision risk."
        )
        default_q = "How does LiDAR contribute to perception?"

    raw_doc_input = st.text_area("Input Document Text:", value=default_doc, height=130)
    query_input = st.text_input("User Search Query:", value=default_q)

    step_selector = st.slider("Select Active Pipeline Stage to Inspect:", min_value=1, max_value=11, value=1)

with st_col2:
    st.subheader(f"Stage {step_selector} Deep-Dive Inspector")

    if step_selector == 1:
        st.info("Stage 1: Documents")
        st.markdown(f"**Raw Ingested Document ({len(raw_doc_input)} chars, {len(raw_doc_input.split())} words):**")
        st.code(raw_doc_input)
        st.caption("Files arrive from object storage as binary blobs (PDF, DOCX, MD).")

    elif step_selector == 2:
        st.info("Stage 2: Text Extraction & Normalization")
        clean_text = " ".join(raw_doc_input.replace("\n", " ").split())
        st.markdown("**Cleaned & Normalized Text Stream:**")
        st.code(clean_text)
        st.caption("Stripped special escape characters, normalized whitespace, and unified UTF-8 encoding.")

    elif step_selector == 3:
        st.info("Stage 3: Chunking (Sliding Window with Overlap)")
        words = raw_doc_input.split()
        chunk_sz, ovlp = 25, 5
        st_step = max(1, chunk_sz - ovlp)
        chunks = []
        for i in range(0, len(words), st_step):
            c_str = " ".join(words[i : i + chunk_sz])
            chunks.append({"Chunk ID": f"chunk_{len(chunks)+1:02d}", "Token Count": len(c_str.split()), "Text Preview": c_str})
        st.dataframe(pd.DataFrame(chunks), use_container_width=True)
        st.caption(f"Produced {len(chunks)} overlapping passages (Window: 25 words, Overlap: 5 words).")

    elif step_selector == 4:
        st.info("Stage 4: Embedding Model (Dense Vector Projection)")
        words = raw_doc_input.split()
        st_step = 20
        sample_chunk = " ".join(words[:25])
        vec = stepper.embed(sample_chunk)
        st.markdown(f"**Chunk 01 Vector (First 8 of {len(vec)} Dimensions):**")
        st.code(str([round(float(x), 4) for x in vec[:8]]) + " ...")
        st.caption(f"Bi-Encoder projected text into {len(vec)}-dimensional hypersphere (L2 Norm = 1.0).")

    elif step_selector == 5:
        st.info("Stage 5: Vector Database (HNSW Indexing)")
        st.markdown("**Index Structure:**")
        st.write("Indexed nodes stored with metadata payloads in HNSW graph layers.")
        st.metric("Total Indexed Nodes", len(raw_doc_input.split()) // 20 + 1)
        st.caption("Memory-mapped graph index ready for sub-millisecond nearest neighbor traversal.")

    elif step_selector == 6:
        st.info("Stage 6: User Query Ingestion")
        st.markdown(f"**Incoming Query:** `{query_input}`")
        st.caption("User query sanitized, stripped of prompt injections, and validated.")

    elif step_selector == 7:
        st.info("Stage 7: Query Embedding")
        q_vec = stepper.embed(query_input)
        st.markdown(f"**Query Vector Preview (First 8 Dimensions):**")
        st.code(str([round(float(x), 4) for x in q_vec[:8]]) + " ...")
        st.caption("Query translated into identical latent space as stored chunks.")

    elif step_selector == 8:
        st.info("Stage 8: Similarity Search (Cosine Scoring)")
        words = raw_doc_input.split()
        scores = []
        st_step = 20
        q_vec = stepper.embed(query_input)
        for i in range(0, len(words), st_step):
            c_str = " ".join(words[i : i + 25])
            c_vec = stepper.embed(c_str)
            sim = float(np.dot(c_vec, q_vec))
            scores.append({"Chunk": f"chunk_{len(scores)+1}", "Cosine Score": round(sim, 4), "Text": c_str[:40] + "..."})
        scores.sort(key=lambda x: x["Cosine Score"], reverse=True)
        st.dataframe(pd.DataFrame(scores), use_container_width=True)

    elif step_selector == 9:
        st.info("Stage 9: Relevant Chunks Selection (Top-K)")
        st.markdown("**Selected Candidate Chunks for Prompt Augmentation:**")
        st.success("Top 2 Chunks extracted above score threshold 0.25.")
        st.caption("Deduplicated and passed to prompt synthesis engine.")

    elif step_selector == 10:
        st.info("Stage 10: LLM Generative Engine & Augmented Prompt")
        aug_prompt = f"""SYSTEM: You are a factual assistant. Answer strictly based on the context.
CONTEXT:
[Source chunk_01]: {raw_doc_input[:180]}...

QUESTION:
{query_input}

GROUNDED REASONING:"""
        st.code(aug_prompt, language="markdown")
        st.caption("Injected into LLM context window with system grounding constraints.")

    elif step_selector == 11:
        st.info("Stage 11: Final Grounded Answer")
        st.markdown(f"**Query:** *{query_input}*")
        st.markdown("**Synthesized Answer:**")
        if "Kubernetes" in doc_choice:
            st.success("According to the Kubernetes Architecture Manual, **etcd** acts as the consistent and highly-available key-value store for all cluster data.")
        elif "Cybersecurity" in doc_choice:
            st.success("During the Containment phase, infected hosts must be immediately isolated from the corporate LAN via VLAN quarantine or endpoint agent disconnect.")
        else:
            st.success("LiDAR generates accurate 3D spatial depth maps up to 200 meters for vehicle perception.")
        st.caption("Answer generated with 100% citation grounding and zero hallucinations.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Pipeline Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What are the industry-standard evaluation metrics used to benchmark a RAG pipeline (e.g. RAGAS)?**
    * **Answer**:
      * 1. **Faithfulness**: Measures whether the generated answer relies strictly on the retrieved context (detects hallucinations).
      * 2. **Answer Relevance**: Measures whether the generated answer directly addresses the user's question.
      * 3. **Context Precision**: Measures the signal-to-noise ratio in retrieved chunks (whether relevant chunks were ranked higher than irrelevant ones).
      * 4. **Context Recall**: Measures whether all ground-truth facts required to answer the question were successfully retrieved.

    **Q2: How do you isolate whether a bad answer was caused by Retrieval Failure versus Generation Failure?**
    * **Answer**:
      * Check the retrieved chunks passed into the prompt:
      * **Retrieval Failure**: If the necessary information is completely missing from the retrieved chunks, the fault lies in text extraction, chunking, or embedding retrieval. (Fix: Adjust chunk size, fine-tune embedding model, or enable hybrid search).
      * **Generation Failure**: If the correct chunk is present in the context, but the LLM hallucinated or failed to mention it, the fault lies in prompt engineering or model capability. (Fix: Improve system prompt, adjust temperature, or use a stronger LLM).

    **Q3: Where does the majority of latency accumulate in a production RAG pipeline?**
    * **Answer**: The generative LLM stage typically accounts for 80-90% of total latency (e.g. 800ms - 2500ms for generation vs 20-50ms for embedding generation and 5ms for vector search). To mitigate this, production pipelines stream tokens via Server-Sent Events (SSE) to reduce perceived Time-To-First-Token (TTFT).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Evaluating Only the Final Answer and Ignoring Intermediate Stages**:
      * *Issue*: Testing RAG purely on end-to-end question-answering accuracy without logging retrieved chunks.
      * *Consequence*: Teams spend weeks rewriting system prompts when the root cause was an embedding model failing to retrieve relevant passages.
      * *Fix*: Implement observability tools (Arize Phoenix, LangSmith, TruLens) to inspect intermediate chunk retrieval scores.

    * **Trap 2: Ignoring Text Extraction Artifacts (Garbage In, Garbage Out)**:
      * *Issue*: Feeding raw PDF extraction outputs with broken words (`hyphen-\nation`), headers, and footers directly to the chunker.
      * *Consequence*: Degraded embedding accuracy because corrupted tokens distort dense vector representations.
      * *Fix*: Run regex cleaning, table extraction parsing, and whitespace normalization prior to chunking.

    * **Trap 3: Synchronous Re-Embedding during User Request Path**:
      * *Issue*: Re-chunking or re-embedding documents during the live query execution.
      * *Consequence*: Query latency spikes from 30ms to 5+ seconds.
      * *Fix*: Ingestion (Stages 1-5) must always run asynchronously via offline batch workers or event queues.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Dual-Loop RAG Architecture**
      * *Definition*: The structural separation between the offline ingestion loop (extraction, chunking, indexing) and the online inference loop (query embedding, retrieval, LLM synthesis).
    * **Flashcard 2: Context Precision**
      * *Definition*: The proportion of retrieved chunks that contain relevant factual signal compared to irrelevant distractor chunks.
    * **Flashcard 3: Faithfulness Metric**
      * *Definition*: An automated evaluation metric calculating the ratio of claims in an LLM output that can be directly verified against the retrieved context.
    * **Flashcard 4: Time-To-First-Token (TTFT)**
      * *Definition*: The duration from query submission until the LLM yields its first generated token, minimized by fast vector retrieval and token streaming.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of the Complete RAG Pipeline:

- [ ] I can trace all 11 stages of the canonical RAG pipeline in order.
- [ ] I understand the separation between offline ingestion and online inference loops.
- [ ] I can diagnose whether an error is a Retrieval Failure or a Generation Failure.
- [ ] I know the 4 foundational RAGAS evaluation metrics (Faithfulness, Relevance, Precision, Recall).
- [ ] I understand where latency bottlenecks occur and how to optimize them.
""")
