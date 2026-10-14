import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="6.2 Batch Operations - ChromaDB Curriculum",
    page_icon="📦",
    layout="wide"
)

# Title & Overview
st.title("6.2 Batch Operations: Batch Sizing, Ingestion Engines, Error Handling & Checkpointing")
st.caption("Chapter 6: Adding Data to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
Ingesting a small toy dataset of 50 documents into ChromaDB takes less than a second. 
However, ingesting an enterprise repository of **500,000 PDF pages, technical manuals, and database records** 
is an entirely different engineering challenge.

Naively looping through documents one at a time will cause ingestion to crawl at 20 docs/second, 
requiring over **7 hours** to complete! Conversely, attempting to pass all 500,000 documents in a single 
massive call will crash the system with Out-of-Memory (OOM) errors or HTTP payload limits.

Mastering production ingestion requires designing a **Resilient Batch Ingestion Engine**:
* **Optimal Batch Sizing**: Amortizing SQLite transaction locks and unlocking SIMD tensor parallelization without RAM exhaustion.
* **Incremental Checkpointing**: Persisting ingestion state so that network blips or pod restarts resume instantly rather than starting over.
* **Dead-Letter Queue (DLQ) Isolation**: Quarantining malformed or corrupted documents without terminating the entire ingestion batch.

In this lesson, we break down batch performance mechanics, design a crash-resilient ingestion engine in Python, 
explore checkpoint architectures, and test an interactive Batch Sizing & Ingestion Simulator.
""")

st.divider()

# Section 1: The Physics of Batch Ingestion Performance
st.header("1. The Physics of Batch Ingestion Performance", divider="rainbow")

col_phys1, col_phys2 = col_phys1, col_phys2 = st.columns([1.1, 0.9])

with col_phys1:
    st.markdown("""
    ### Why Batching Multiplies Throughput by 100x
    
    1. **SQLite Transaction Locking**:
       * Every `collection.add()` call triggers a disk write and WAL commit (`fsync`).
       * Writing 10,000 documents individually incurs 10,000 sequential disk syncs.
       * Batching into 10 groups of 1,000 documents requires only **10 atomic commits**, slashing I/O wait by 99%.
    
    2. **Vectorized Neural Network Forward Passes**:
       * Transformer embedding models (ONNX / PyTorch) execute matrix multiplications.
       * Modern CPUs (AVX-512) and GPUs (Tensor Cores) process matrices of shape `(BatchSize, SeqLen)` in parallel.
       * Passing a batch of 128 chunks takes almost the same time as passing 4 chunks!
    
    3. **HTTP Overhead in Client/Server Mode**:
       * Sending 10,000 individual HTTP requests introduces TCP handshake, TLS encryption, and HTTP header latency on every document.
    """)

with col_phys2:
    st.markdown("### The Batch Sizing Sweet Spot Curve")
    size_dot = """
    digraph BatchSizing {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        SuboptimalLow [label="Batch Size: 1 – 50 Docs\\nHigh Disk I/O Wait\\nThroughput: ~25 docs/sec\\nToo Slow!", fillcolor="#FEE2E2", color="#DC2626"];

        OptimalZone [label="SWEET SPOT: 500 – 2,000 Docs\\nMax SIMD Tensor Parallelism\\nAmortized SQLite Commits\\nThroughput: ~2,500 docs/sec", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        SuboptimalHigh [label="Batch Size: > 10,000 Docs\\nRisk: OOM RAM Spikes\\nRisk: HTTP 413 Payload Too Large\\nRisk: Unrecoverable Batch Timeouts", fillcolor="#FEE2E2", color="#DC2626"];

        SuboptimalLow -> OptimalZone [label="Increase Batch"];
        OptimalZone -> SuboptimalHigh [label="Exceeds Limits"];
    }
    """
    st.graphviz_chart(size_dot, use_container_width=False)

st.divider()

# Section 2: Resilient Ingestion with Checkpointing & Dead-Letter Queues
st.header("2. Resilient Ingestion with Checkpointing & Dead-Letter Queues", divider="rainbow")

col_ckpt1, col_ckpt2 = col_ckpt1, col_ckpt2 = st.columns([1, 1])

with col_ckpt1:
    st.markdown("""
    ### Checkpointed Resumption Pattern
    * In multi-hour ETL jobs, server reboots, network timeouts, or spot instance terminations are inevitable.
    * **The Anti-Pattern**: Restarting the entire ingestion job from Document #1 after a failure at Document #450,000.
    * **The Production Solution**:
      * Maintain a persistent state cursor in a local JSON file or Redis key (`checkpoint.json`).
      * After every batch commits successfully, record the highest completed chunk index.
      * On worker startup, query the cursor and seek directly to that offset!
    
    ### Dead-Letter Queue (DLQ) Isolation
    * If Document #14,291 contains malformed UTF-8 characters or exceeds token length, the batch will fail.
    * Catch batch exceptions, log failed items to a `dead_letter_queue.json` file for asynchronous review, and continue processing remaining batches.
    """)

with col_ckpt2:
    st.markdown("### Resilient Ingestion Architecture")
    ckpt_dot = """
    digraph ResilientETL {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        SourceData [label="Unstructured Document Corpus\\n500,000 Chunks to Ingest", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Checkpointer [label="Checkpoint Inspector\\nResume from index: 142,000", fillcolor="#FEF3C7", color="#D97706"];

        BatchSlicer [label="Batch Slicer (Size=1,000)\\nYields parallel arrays", fillcolor="#EDE9FE", color="#7C3AED"];

        ChromaCommit [label="collection.upsert()\\nCommit to ChromaDB", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        UpdateCKPT [label="Update Checkpoint Cursor\\nPersist to disk", fillcolor="#A7F3D0", color="#059669"];

        DLQ [label="Dead-Letter Queue (DLQ)\\nIsolate Poison Documents", fillcolor="#FEE2E2", color="#DC2626"];

        SourceData -> Checkpointer -> BatchSlicer;
        BatchSlicer -> ChromaCommit;
        ChromaCommit -> UpdateCKPT [label="Success"];
        ChromaCommit -> DLQ [label="Error Caught"];
        UpdateCKPT -> BatchSlicer [label="Next Batch"];
    }
    """
    st.graphviz_chart(ckpt_dot, use_container_width=False)

st.divider()

# Section 3: Clean Python Resilient Batch Ingestion Engine
st.header("3. Clean Python Resilient Batch Ingestion Engine", divider="rainbow")

st.code("""import chromadb
import json
import os
from typing import List, Dict, Any

def resilient_batch_ingestion(
    collection,
    all_chunks: List[Dict[str, Any]],
    batch_size: int = 1000,
    checkpoint_file: str = "./ingest_checkpoint.json"
):
    \"\"\"Ingests a large corpus in batches with checkpointing and error isolation.\"\"\"
    
    # 1. Load last completed checkpoint offset
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            start_index = data.get("last_processed_index", 0)
            print(f"Resuming ingestion from chunk offset: {start_index}")

    total_chunks = len(all_chunks)
    
    # 2. Iterate through batches starting from checkpoint
    for i in range(start_index, total_chunks, batch_size):
        batch = all_chunks[i : i + batch_size]
        
        batch_ids = [item["id"] for item in batch]
        batch_docs = [item["text"] for item in batch]
        batch_metas = [item["metadata"] for item in batch]
        
        try:
            # Use upsert for idempotency
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            
            # 3. Update checkpoint on successful commit
            with open(checkpoint_file, "w") as f:
                json.dump({"last_processed_index": i + len(batch)}, f)
                
            print(f"Committed batch: {i + len(batch)} / {total_chunks}")
            
        except Exception as exc:
            print(f"Error in batch {i}-{i+batch_size}: {exc}")
            # Isolate failed batch to Dead-Letter Queue
            with open("./failed_batches.jsonl", "a") as err_log:
                err_log.write(json.dumps({"batch_offset": i, "error": str(exc)}) + "\\n")
            continue

    print("Corpus ingestion complete!")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Batch Sizing & Ingestion Simulator
st.header("4. Interactive Workbench: Batch Sizing & Ingestion Simulator", divider="rainbow")

st.markdown("""
Simulate ingesting large-scale corpora into ChromaDB. Adjust corpus volume and batch size 
to analyze throughput (docs/sec), total disk transactions, peak RAM usage, and estimated completion time.
""")

col_sim_corpus, col_sim_params = col_sim_corpus, col_sim_params = st.columns(2)

with col_sim_corpus:
    st.subheader("1. Ingestion Scale Demographics")
    sim_corpus_size = st.select_slider(
        "Total Corpus Documents (Chunks):",
        options=[1000, 10000, 50000, 100000, 500000, 1000000],
        value=100000
    )
    dim_size = st.selectbox("Embedding Model:", ["all-MiniLM-L6-v2 (384d / CPU)", "text-embedding-3-small (1536d / Cloud)", "bge-base-en (768d / Local)"])

with col_sim_params:
    st.subheader("2. Batch Configuration")
    sim_batch_size = st.select_slider(
        "Ingestion Batch Size (`batch_size`):",
        options=[1, 10, 50, 250, 500, 1000, 2000, 5000, 10000, 50000],
        value=1000
    )
    enable_checkpointing = st.checkbox("Enable Incremental Checkpointing (Disk Save)", value=True)

# Performance calculation heuristics
num_batches = int(np.ceil(sim_corpus_size / sim_batch_size))

if sim_batch_size == 1:
    docs_per_sec = 25.0
    mem_spike_mb = 120.0
elif sim_batch_size < 50:
    docs_per_sec = 180.0
    mem_spike_mb = 180.0
elif sim_batch_size <= 2000:
    docs_per_sec = 2600.0  # Sweet spot
    mem_spike_mb = 450.0 + (sim_batch_size * 0.15)
elif sim_batch_size <= 10000:
    docs_per_sec = 2200.0  # Diminishing returns
    mem_spike_mb = 1200.0 + (sim_batch_size * 0.25)
else:
    docs_per_sec = 900.0  # GC pressure & memory thrashing
    mem_spike_mb = 4500.0

total_time_seconds = sim_corpus_size / docs_per_sec

st.divider()

col_met_a, col_met_b, col_met_c, col_met_d = st.columns(4)
col_met_a.metric(label="Estimated Throughput", value=f"{int(docs_per_sec):,} docs/sec")
col_met_b.metric(label="Total SQLite Transactions", value=f"{num_batches:,} commits")
col_met_c.metric(label="Peak Ingestion RAM", value=f"{mem_spike_mb:.0f} MB")
col_met_d.metric(label="Est. Completion Time", value=f"{total_time_seconds:.1f} s ({total_time_seconds/60:.1f} min)")

if 500 <= sim_batch_size <= 2000:
    st.success(f"🏆 **OPTIMAL BATCH CONFIGURATION**: Batch size `{sim_batch_size}` delivers maximum throughput ({int(docs_per_sec):,} docs/sec) while keeping RAM footprint safe ({mem_spike_mb:.0f} MB).")
elif sim_batch_size <= 10:
    st.error(f"🔴 **CRITICAL BOTTLENECK**: Single-document or tiny batch calls incur {num_batches:,} disk transactions, wasting 99% of time in I/O wait!")
else:
    st.warning(f"⚠️ **HIGH MEMORY RISK**: Batch size `{sim_batch_size}` allocates large tensor arrays in RAM ({mem_spike_mb:.0f} MB). High risk of HTTP payload timeouts or OOM container restarts.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the optimal batch size when adding documents to ChromaDB, and why?"):
    st.markdown("""
    **Answer**:
    * **Recommended Batch Size**: **500 to 2,000 documents**.
    * **Why Not 1?**: Adding documents individually incurs separate disk fsyncs and SQLite transactions for each item, limiting throughput to ~20–30 docs/sec.
    * **Why Not 50,000?**: Excessive batches cause tensor memory allocation spikes, trigger reverse proxy HTTP 413 (Payload Too Large) errors, and lead to unrecoverable job failures if a single record fails.
    * Slicing into 1,000 chunks achieves the ideal sweet spot of **2,000–3,000 docs/sec**.
    """)

with st.expander("Q2: How does checkpointing protect long-running vector ETL pipelines?"):
    st.markdown("""
    **Answer**:
    * A vector ingestion pipeline for millions of records can take hours. If the process crashes at 90% completion, restarting from the beginning wastes compute, API tokens, and developer time.
    * **Checkpointing**:
      1. Commits data to ChromaDB in discrete batches.
      2. Atomically records the completed batch index to persistent storage (e.g. SQLite, JSON, or Redis).
      3. On restart, the worker queries the checkpoint and resumes immediately from the failed boundary.
    """)

with st.expander("Q3: What is a Dead-Letter Queue (DLQ) in vector ingestion architectures?"):
    st.markdown("""
    **Answer**:
    * In a batch of 1,000 documents, 999 might be valid while 1 document contains corrupted binary characters or invalid metadata.
    * Without isolation, the entire batch throws an exception and crashes the ETL worker.
    * A **Dead-Letter Queue (DLQ)** intercepts the failing batch, logs the bad records and their stack traces to an isolated queue/file, and allows the worker to continue ingesting remaining valid documents without human intervention.
    """)

with st.expander("Q4: Why is `collection.upsert()` preferred over `collection.add()` in batch pipelines?"):
    st.markdown("""
    **Answer**:
    * Batch pipelines frequently re-run due to network retries, scheduled cron triggers, or worker restarts.
    * With `add()`, any retry crashes immediately on the first existing record (`DuplicateIDError`).
    * `upsert()` is **mathematically idempotent**: running it 1 time or 100 times results in the exact same deterministic state without errors.
    """)

with st.expander("Q5: How does embedding model batching differ from ChromaDB database batching?"):
    st.markdown("""
    **Answer**:
    * **Embedding Model Batching**: Determines how many text chunks pass through the transformer neural network simultaneously on GPU/CPU (typically **64 to 256 chunks** to prevent GPU VRAM out-of-memory).
    * **ChromaDB Database Batching**: Determines how many vectors and documents are committed in a single SQLite transaction and HNSW update (typically **500 to 2,000 chunks**).
    * An ingestion engine often nests these: generating embeddings in sub-batches of 128, and committing to ChromaDB in database batches of 1,000.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Passing Entire Massive Corpora in a Single Function Call**  
    Passing 200,000 documents in one `collection.add()` call triggers multi-gigabyte RAM spikes and HTTP body timeout errors. Slice into batches of 1,000.
    """)
    st.warning("""
    **Trap 2: Zero Checkpoint Tracking in Long ETL Jobs**  
    Failing to save progress cursors means a network blip at 95% completion forces the entire pipeline to re-run from chunk zero, wasting hours of compute.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Terminating the Entire Pipeline on a Single Corrupt Record**  
    Letting one malformed string crash an entire 500k-doc ingestion job wastes developer time. Catch exceptions and route poison records to a DLQ.
    """)
    st.warning("""
    **Trap 4: Sub-Optimal Single-Document Ingestion Loops**  
    Running `for doc in dataset: collection.add(doc)` creates 50,000 separate disk sync locks. Group into chunks before committing.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the optimal batch size?"):
        st.markdown("""
        **500 to 2,000 Documents**:  
        Sweet spot maximizing SIMD tensor parallelism and amortizing disk commits.
        """)
    with st.expander("🗂️ Flashcard: What is Checkpointing?"):
        st.markdown("""
        **Cursor Persistence**:  
        Saves last completed chunk offset so crashed ETL pipelines resume instantly.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is a Dead-Letter Queue?"):
        st.markdown("""
        **Error Isolation**:  
        Quarantines poisoned documents without terminating the broader ingestion pipeline.
        """)
    with st.expander("🗂️ Flashcard: Why is upsert() essential for batching?"):
        st.markdown("""
        **Idempotent Retries**:  
        Permits safe pipeline retries without crashing on duplicate ID errors.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why batching documents multiplies ingestion throughput by 100x.", value=True)
chk2 = st.checkbox("I understand the batch sizing sweet spot (500–2,000 docs) and the risks of oversized payloads.", value=True)
chk3 = st.checkbox("I know how to implement checkpointed resumption to protect long-running ETL pipelines.", value=True)
chk4 = st.checkbox("I understand how Dead-Letter Queues (DLQ) isolate corrupt documents without terminating jobs.", value=True)
chk5 = st.checkbox("I know how embedding model tensor batching coordinates with database transaction batching.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 6.3: Upsert Operations in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 6.2: Batch Operations and are ready for Chapter 6.3: Upsert Operations!")
