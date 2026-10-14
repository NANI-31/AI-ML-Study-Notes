import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import heapq
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.3 Similarity Calculation - Vector DB Curriculum",
    page_icon="🧮",
    layout="wide"
)

# Title & Overview
st.title("3.3 Similarity Calculation: Matrix-Vector SIMD, Ranking Algorithms & Dynamic Thresholds")
st.caption("Chapter 3: Similarity Search | Domain: 4. Vector Database")
st.markdown("""
Calculating similarity is not simply an abstract mathematical formula—in a production vector database, it is 
a **high-throughput linear algebra pipeline** executed over millions of vectors in single-digit milliseconds.

Understanding how single-pair vector comparisons scale into **matrix-vector multiplications (`M @ q`)**, 
how **partial sorting (`argpartition`)** outperforms naive full-array sorts, how **bounded priority queues** 
collect the Top-K candidates, and how to calibrate **similarity score thresholds** to filter out irrelevant hallucinations 
is fundamental to vector search engineering.
""")

st.divider()

# Section 1: From Pairwise Vectors to Matrix-Vector Batch Processing
st.header("1. From Pairwise Vectors to Matrix-Vector Batch Compute", divider="rainbow")

col_calc1, col_calc2 = col_calc1, col_calc2 = st.columns([1, 1])

with col_calc1:
    st.markdown("""
    ### 1. Pairwise vs. Matrix-Vector Compute
    * **Pairwise Vector Comparison**:
      * Calculating similarity between two vectors `u` and `v`:
        `sim = sum(u_i * v_i)` for `i = 1 to d`.
      * Requires `d` multiplications and `d - 1` additions (2d floating-point operations).
    * **Single-Query vs. Matrix Ingestion (`M @ q`)**:
      * A collection of N documents of dimension d forms a matrix `M in R^(N x d)`.
      * A user query vector is `q in R^d`.
      * The dot product across ALL documents is expressed as a single matrix-vector multiplication:
        `S = M @ q in R^N`.
      * Modern hardware libraries (BLAS, MKL, OpenBLAS, CUDA cuBLAS) execute `M @ q` using SIMD registers (AVX-512) and GPU Tensor Cores at teraflop speeds.
    * **Batch Query Ingestion (`Q @ M^T`)**:
      * If B queries arrive concurrently: `Q in R^(B x d)`.
      * Batch similarity computation becomes matrix-matrix multiplication (GEMM):
        `S_batch = Q @ M.T in R^(B x N)`.
      * Hardware throughput increases by up to **10x** due to cache locality and tensor pipeline saturation.
    """)

with col_calc2:
    st.markdown("### The Matrix-Vector Similarity Pipeline")
    mat_dot = """
    digraph MatrixVectorPipeline {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        QueryVec [label="Query Vector q\\nShape: (d, 1)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        DocMatrix [label="Document Matrix M\\nShape: (N, d)\\nNormalized Chunks", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];

        BLAS [label="Hardware BLAS GEMV / GEMM\\nAVX-512 / CUDA Tensor Cores", fillcolor="#DBEAFE"];
        RawScores [label="Raw Score Vector S\\nShape: (N, 1)\\nS = M @ q", fillcolor="#D1FAE5"];
        
        ArgPart [label="Top-K Partial Sort\\nargpartition(S, -K)", fillcolor="#FEF3C7"];
        ThresholdFilter [label="Similarity Threshold\\nDrop if Score < tau", fillcolor="#FDE68A"];
        HydratedResults [label="Final Top-K Results\\nPass to Reranker / LLM", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];

        QueryVec -> BLAS;
        DocMatrix -> BLAS;
        BLAS -> RawScores -> ArgPart -> ThresholdFilter -> HydratedResults;
    }
    """
    st.graphviz_chart(mat_dot, use_container_width=False)

st.divider()

# Section 2: Ranking Algorithms: Argpartition vs. Bounded Heap vs. Full Sort
st.header("2. Ranking Algorithms: Sorting Complexity at Scale", divider="rainbow")

st.markdown("""
Once raw similarity scores `S in R^N` are calculated, retrieving the top K items efficiently is a critical bottleneck:
""")

col_rank1, col_rank2, col_rank3 = st.columns(3)

with col_rank1:
    st.markdown("""
    #### 1. Naive Full Sort (`np.argsort`)
    * **Mechanism**: Sorts all N scores in descending order: `np.argsort(scores)[::-1][:k]`.
    * **Complexity**: `O(N * log N)`.
    * **The Trap**:
      * For 1,000,000 documents, sorting all 1,000,000 items when you only need the top 5 is wildly inefficient!
      * Wastes 99.9% of sorting cycles on items that will be immediately discarded.
    """)

with col_rank2:
    st.markdown("""
    #### 2. Partial Quickselect (`np.argpartition`)
    * **Mechanism**: Partitions the array around the k-th largest element using Introselect / Quickselect:
      `np.argpartition(scores, -k)[-k:]`.
    * **Complexity**: `O(N)`.
    * **Why It Wins**:
      * Guarantees that the top-K items are partitioned in linear time without sorting the remaining N - k items!
      * You only sort the tiny k-sized subset at cost `O(k * log k)`. Total: `O(N + k * log k)`.
    """)

with col_rank3:
    st.markdown("""
    #### 3. Bounded Max-Heap (`heapq`)
    * **Mechanism**: Streaming priority queue of fixed size k.
    * **Complexity**: `O(N * log k)`.
    * **When to Use**:
      * Streaming workloads where vectors arrive sequentially or when index structures visit nodes incrementally (e.g. during HNSW graph traversal).
    """)

st.divider()

# Section 3: Similarity Thresholds: Static, Dynamic & Knee-Method
st.header("3. Similarity Thresholds: Preventing Hallucinations & Noise", divider="rainbow")

st.markdown("""
Top-K retrieval always returns k documents—even if the user query is completely nonsensical or unrelated to the database! 
Without similarity thresholding, your LLM will be fed irrelevant garbage, triggering hallucinations.
""")

col_thresh1, col_thresh2 = col_thresh1, col_thresh2 = st.columns([1, 1])

with col_thresh1:
    st.markdown("""
    ### 1. Static Threshold Cutoffs
    * **How it works**: Discard any candidate whose cosine similarity is below a fixed cutoff `tau`:
      `results = [doc for doc in top_k if doc.score >= tau]`.
    * **Typical Ranges (Cosine Similarity)**:
      * `0.85 - 1.00`: Extremely strong match (near paraphrases, direct answers).
      * `0.70 - 0.85`: Moderately strong semantic match (related domain concepts).
      * `0.50 - 0.70`: Weak topical overlap (often noisy in specific RAG tasks).
      * `< 0.50`: Semantic noise (typically discard).
    * **Limitation**:
      * Highly sensitive to query phrasing and model family. A threshold of 0.75 on OpenAI might correspond to 0.62 on BGE.
    """)

with col_thresh2:
    st.markdown("""
    ### 2. Dynamic Adaptive Thresholds (The Knee Method)
    * **How it works**: Examine the **score drop-off curve** between consecutive ranks:
      `Delta_i = Score_i - Score_(i+1)`.
    * **The Cliff Phenomenon**:
      * Often, Rank 1 has score `0.89`, Rank 2 has `0.87`, Rank 3 has `0.85`... and Rank 4 abruptly collapses to `0.52`!
      * Rank 4 represents a topic cliff.
    * **Adaptive Rule**:
      * Truncate retrieval if `Score_i < (alpha * Score_1)` (e.g. discard any candidate with less than 80% of Rank 1's score) or when `Delta_i > 0.20`.
    """)

st.divider()

# Section 4: Interactive Matrix-Vector Calculation & Threshold Workbench
st.header("4. Interactive Similarity Calculation & Threshold Simulator", divider="rainbow")

st.markdown("""
Simulate matrix-vector dot product scoring over a synthetic document corpus. 
Tune corpus size, dimensionality, Top-K, and similarity thresholds to observe runtime and ranking behavior:
""")

col_wb_in, col_wb_out = st.columns([1, 1])

with col_wb_in:
    st.subheader("Collection & Query Controls")
    wb_num_docs = st.select_slider(
        "Corpus Size (N Documents)",
        options=[100, 500, 1000, 5000, 10000, 50000],
        value=1000,
        format_func=lambda x: f"{x:,} docs"
    )
    wb_dim = st.selectbox("Embedding Dimension (d)", [64, 128, 384, 768, 1536], index=2)
    wb_top_k = st.slider("Top-K Candidates", 1, 20, 5)
    wb_threshold = st.slider("Similarity Threshold Cutoff (tau)", 0.0, 1.0, 0.65, 0.05)
    wb_use_knee = st.checkbox("Enable Dynamic Relative Drop Cutoff (Drop if score < 80% of Rank 1)", value=True)

with col_wb_out:
    st.subheader("Matrix Computation & Retrieval Pipeline")
    
    np.random.seed(42)
    # Generate mock document matrix (N, d)
    raw_matrix = np.random.randn(wb_num_docs, wb_dim)
    norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
    matrix = raw_matrix / np.where(norms > 0, norms, 1.0)
    
    # Generate synthetic query aligned with 2 specific documents
    query = matrix[42].copy() + np.random.randn(wb_dim) * 0.15 # Strong match with doc 42
    query = query / np.linalg.norm(query)
    
    # 1. Matrix-Vector Multiplication: M @ q
    scores = np.dot(matrix, query) # Shape: (N,)
    
    # 2. Fast partial sort via argpartition
    k = min(wb_top_k * 2, wb_num_docs) # Over-fetch for thresholding
    partition_indices = np.argpartition(scores, -k)[-k:]
    sorted_top_indices = partition_indices[np.argsort(scores[partition_indices])[::-1]]
    
    # 3. Apply thresholds
    retrieved_results = []
    top_score = scores[sorted_top_indices[0]]
    
    for idx in sorted_top_indices:
        s = float(scores[idx])
        # Absolute threshold
        if s < wb_threshold:
            continue
        # Relative knee threshold
        if wb_use_knee and s < (0.80 * top_score):
            continue
        retrieved_results.append({
            "Document ID": f"DOC-{idx:05d}",
            "Cosine Similarity": s,
            "Relative to Top Rank": f"{(s / top_score) * 100:.1f}%",
            "Verdict": "Accepted (Hydrated)"
        })
        if len(retrieved_results) == wb_top_k:
            break
            
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.metric("Total Documents Scanned", f"{wb_num_docs:,}", f"Matrix: ({wb_num_docs}, {wb_dim})")
    with col_st2:
        st.metric("Final Documents Retained", f"{len(retrieved_results)} / {wb_top_k}", f"Threshold: >= {wb_threshold:.2f}")
        
    if retrieved_results:
        res_df = pd.DataFrame(retrieved_results)
        st.dataframe(
            res_df.style.format({"Cosine Similarity": "{:.4f}"}).background_gradient(subset=["Cosine Similarity"], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Zero candidates met the threshold cutoff! Upstream fallback or 'I don't know' response triggered.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: High-Performance Matrix Similarity Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements **Matrix-Vector Similarity Computation** with 
**`np.argpartition` linear-time sorting**, **adaptive score thresholding**, and benchmarking:
""")

code_matrix_engine = '''import numpy as np
from typing import List, Dict, Any, Optional

class MatrixSimilarityEngine:
    """High-throughput matrix-vector similarity calculation and ranking engine."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_collection(self, doc_ids: List[str], vectors: np.ndarray):
        """Ingests and L2-normalizes document vectors."""
        assert vectors.shape[1] == self.dimension
        # Unit normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vecs = vectors / np.where(norms > 0, norms, 1.0)
        
        self.doc_ids.extend(doc_ids)
        if len(self.matrix) == 0:
            self.matrix = unit_vecs.astype(np.float32)
        else:
            self.matrix = np.vstack([self.matrix, unit_vecs.astype(np.float32)])

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        abs_threshold: float = 0.60,
        relative_drop_ratio: float = 0.80
    ) -> List[Dict[str, Any]]:
        """Executes M @ q, partial partition, and adaptive threshold filtering."""
        assert len(query_vector) == self.dimension
        N = len(self.doc_ids)
        if N == 0:
            return []

        # 1. Normalize query
        q_norm = np.linalg.norm(query_vector)
        q_unit = query_vector / (q_norm if q_norm > 0 else 1.0)

        # 2. Matrix-vector dot product: (N, d) @ (d,) -> (N,)
        scores = np.dot(self.matrix, q_unit)

        # 3. Linear-time partial sort via argpartition
        fetch_k = min(top_k * 3, N) # Overfetch for threshold headroom
        partition_idx = np.argpartition(scores, -fetch_k)[-fetch_k:]
        
        # Sort only the partitioned candidates
        sorted_candidates = partition_idx[np.argsort(scores[partition_idx])[::-1]]

        # 4. Adaptive thresholding
        results = []
        top_score = float(scores[sorted_candidates[0]]) if len(sorted_candidates) > 0 else 0.0
        
        for idx in sorted_candidates:
            sc = float(scores[idx])
            # Absolute threshold cutoff
            if sc < abs_threshold:
                break
            # Relative drop cutoff (knee detection)
            if sc < (top_score * relative_drop_ratio):
                break

            results.append({
                "doc_id": self.doc_ids[idx],
                "score": sc
            })
            if len(results) == top_k:
                break

        return results

# Demonstration
dim = 128
num_docs = 5000
np.random.seed(42)

engine = MatrixSimilarityEngine(dimension=dim)

# Ingest 5000 random docs
doc_ids = [f"chunk_{i}" for i in range(num_docs)]
vecs = np.random.randn(num_docs, dim).astype(np.float32)
engine.add_collection(doc_ids, vecs)

# Construct a query close to chunk_99
q = vecs[99].copy() + np.random.randn(dim) * 0.10

hits = engine.query(q, top_k=3, abs_threshold=0.60)

print(f"Top-K Retained Hits (Found {len(hits)}):")
for rank, h in enumerate(hits, 1):
    print(f"Rank {rank}: ID={h['doc_id']} | Cosine Similarity={h['score']:.4f}")
'''

st.code(code_matrix_engine, language="python")

if st.button("🚀 Execute Matrix Similarity Engine", type="primary"):
    exec_scope = {}
    exec(code_matrix_engine, globals(), exec_scope)
    hits_out = exec_scope["hits"]
    
    st.success("Matrix Similarity Engine executed successfully!")
    for rank, h in enumerate(hits_out, 1):
        st.markdown(f"**Rank {rank}** | `ID: {h['doc_id']}` | **Score:** `{h['score']:.4f}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why is `np.argpartition` preferred over `np.argsort` for Top-K candidate extraction?"""
a1 = """* `np.argsort` sorts the entire array of N scores, running in `O(N log N)` time. For 10 million documents, sorting 10M numbers when only top 10 are needed wastes millions of CPU cycles.
* `np.argpartition` utilizes the **Introselect (Quickselect)** algorithm, which partitions the array around the k-th largest element in linear `O(N)` time.
* It guarantees that the k largest elements reside in the last k positions of the partition without sorting them internally.
* You then sort only those k elements in `O(k log k)` time.
* Total complexity is reduced from `O(N log N)` to `O(N + k log k)`, accelerating Top-K extraction by up to 15x on large collections."""

q2 = """2. What is the danger of returning Top-K results without a similarity score threshold in RAG systems?"""
a2 = """* **The Forced Hallucination Trap**:
  * Top-K retrieval is unconditional: if asked a completely out-of-domain question (e.g. `'What is the recipe for chocolate cake?'` against a database of corporate tax law), Top-K will STILL return the top 5 documents!
  * Those 5 documents might have a dismal cosine similarity of `0.22`, but the downstream LLM assumes they are authoritative context.
  * The LLM then attempts to hallucinate a connection between tax law and chocolate cake.
* **The Solution**: Enforce an absolute score threshold (e.g., `tau = 0.70`). If all retrieved candidates fall below `tau`, abort LLM generation and return: `'I do not have documentation to answer this question.'`"""

q3 = """3. How does batch query processing (`Q @ M.T`) achieve higher throughput than sequential single-query execution?"""
a3 = """* A single query `M @ q` is a **Matrix-Vector multiplication (GEMV)**. GEMV is memory-bandwidth bound: memory buses must stream the entire matrix M from RAM to CPU/GPU caches for every single query.
* A batch of B queries `Q @ M.T` is a **General Matrix-Matrix multiplication (GEMM)**.
* In GEMM, each tile of the document matrix M loaded into high-speed L1/L2 cache is reused across B query vectors simultaneously!
* This shifts the operation from memory-bandwidth limited to compute-limited, increasing hardware throughput by **5x to 10x** on modern GPUs and AVX-512 processors."""

q4 = """4. Explain the difference between Static Thresholds and Adaptive Knee/Cliff Thresholds."""
a4 = """* **Static Threshold**: A hard numerical cutoff (e.g., `score >= 0.75`). Simple to implement, but brittle: different query formulations yield different baseline scores, and models updated via fine-tuning shift their score distributions.
* **Adaptive Knee/Cliff Threshold**: Analyzes the score gradient across ranked results:
  * If Rank 1 is `0.92`, Rank 2 is `0.90`, but Rank 3 drops to `0.54`, the adaptive algorithm recognizes a semantic cliff (`Delta > 0.30`) and truncates retrieval to the top 2 items, even if `k = 5`.
  * Protects LLM context windows from noisy trailing candidates."""

q5 = """5. How does score normalization differ between Cosine Similarity and L2 Distance when defining thresholds?"""
a5 = """* **Cosine Similarity**: Naturally bounded in `[-1.0, 1.0]`. For text embeddings, typically `[0.0, 1.0]`. Setting a threshold like `0.75` is intuitive because 1.0 is identical and 0.0 is orthogonal.
* **Euclidean Distance (L2)**: Bounded in `[0.0, infinity)`. Smaller is closer (0.0 is identical). An L2 threshold must be an upper bound (`dist <= delta`), and `delta` varies depending on vector dimensionality `d` unless normalized.
* To convert Euclidean distance on unit vectors to cosine similarity: `cos_sim = 1.0 - (L2^2 / 2.0)`."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Similarity Calculation Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The OOM Crash on Unbounded Batch Query Matrices**  
> Writing `Q @ M.T` when `Q` has 10,000 queries and `M` has 1,000,000 documents creates an intermediate 
> score matrix of shape `(10000, 1000000)` float32 values = **40 GB of RAM**! 
> The system will immediately crash with an Out-of-Memory (OOM) error. Always chunk batch queries into 
> mini-batches of 64 or 128 queries.

> [!WARNING]
> **Trap 2: Fixed-K Context Poisoning in RAG**  
> Setting `k = 10` unconditionally means simple, single-fact queries receive 1 useful chunk and 9 useless 
> distractors. Complex queries might need all 10, but simple queries are degraded by the irrelevant context. 
> Combine Top-K with dynamic knee thresholding to return only high-confidence matches.

> [!WARNING]
> **Trap 3: Model Version Calibration Shift**  
> If you migrate an embedding model from `text-embedding-ada-002` to `text-embedding-3-small`, your hardcoded 
> static threshold of `0.80` will break. The newer model may have tighter cluster distributions where a genuine match 
> scores `0.72`. Always re-calibrate similarity thresholds against an evaluation test suite when changing models.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Matrix-Vector Compute?"):
        st.markdown("""
        **Matrix-Vector GEMV (`M @ q`)**:  
        Computing dot products between all N database vectors and a query vector in a single SIMD hardware instruction call.
        """)
    with st.expander("🗂️ Flashcard: Why Use `np.argpartition`?"):
        st.markdown("""
        **Linear-Time Top-K**:  
        Partitions the top-k elements in `O(N)` time using Quickselect, avoiding the expensive `O(N log N)` full-array sort.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Knee/Cliff Method?"):
        st.markdown("""
        **Adaptive Cliff Detection**:  
        Truncating retrieved chunks when the similarity drop between consecutive ranks exceeds a threshold (e.g. `Delta > 0.20`), eliminating low-quality trailing results.
        """)
    with st.expander("🗂️ Flashcard: Why Enforce Similarity Thresholds?"):
        st.markdown("""
        **Hallucination Prevention**:  
        Prevents vector databases from feeding irrelevant context to LLMs when user queries have zero true matches in the database.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can explain how pairwise dot products scale into matrix-vector (`M @ q`) operations.", value=True)
chk2 = st.checkbox("I understand why `np.argpartition` (O(N)) is faster than `np.argsort` (O(N log N)) for Top-K extraction.")
chk3 = st.checkbox("I know why Top-K retrieval without similarity thresholds causes LLM hallucinations.")
chk4 = st.checkbox("I understand the difference between static thresholds and dynamic knee/cliff detection.")
chk5 = st.checkbox("I know how batching queries into GEMM (`Q @ M.T`) improves hardware cache locality.")
chk6 = st.checkbox("I can implement a complete matrix similarity ranking engine with thresholding in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Similarity Calculation and are ready for Chapter 3.4: Practical Examples!")
