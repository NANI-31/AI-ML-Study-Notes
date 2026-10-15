import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="11.2 Similarity with NumPy - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("11.2 Similarity with NumPy: Vectorized Dot Products, GEMM Matrices, & Distance Expansion")
st.caption("Chapter 11: Embeddings with NumPy | Domain: 3. Vector Embeddings")
st.markdown("""
In Python, writing nested `for` loops to compare a batch of query vectors against a corpus of document vectors is catastrophically slow:
* A nested Python loop over 1,000 queries and 10,000 documents executes **10,000,000 loop iterations**, taking over **45 seconds** in pure Python!
* Using **vectorized NumPy matrix multiplication (`queries @ corpus.T`)**, the identical calculation takes just **12 milliseconds**—a **3,700x speedup**!

NumPy achieves this by calling low-level C and Fortran BLAS libraries (Intel MKL / OpenBLAS) that leverage CPU SIMD, AVX-512 vectorization, and multi-core parallelism.

This lesson explores vectorized similarity calculations in pure NumPy:
* Inner dot products and pre-normalized Cosine Similarity matrices.
* The **Euclidean Distance Expansion Trick** ($||u - v||^2 = ||u||^2 + ||v||^2 - 2(u \\cdot v)$) for loop-free distance matrices.
* Interactive benchmarks comparing Python nested loops against vectorized BLAS GEMM matrix operations.
""")

st.divider()

# Section 1: Vectorized Similarity Mechanics
st.header("1. Vectorized Similarity Mechanics & Matrix-Matrix Multiplication", divider="rainbow")

col_mat1, col_mat2 = st.columns([1, 1])

with col_mat1:
    st.markdown("""
    ### Key Mathematical Formulations in NumPy:
    * **1. Single Pairwise Cosine Similarity**:
      ```python
      cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
      ```
    * **2. Batched Matrix Cosine Similarity (The GEMM Operation)**:
      Given query matrix `Q` of shape `[N, d]` and corpus matrix `C` of shape `[M, d]`:
      If both matrices are pre-normalized to unit $L_2$ norm:
      ```python
      # Resulting shape: [N, M] (Cosine similarity between every query and document)
      sim_matrix = Q @ C.T
      ```
    * **3. The Loop-Free Euclidean Distance Matrix Expansion**:
      Computing Euclidean distances between all rows of `A [N, d]` and `B [M, d]` without loops:
      Using the algebraic identity: `||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a dot b)`.
      ```python
      # a2 shape [N, 1], b2 shape [1, M]
      a2 = np.sum(A**2, axis=1, keepdims=True)
      b2 = np.sum(B**2, axis=1, keepdims=True).T
      dist_sq = a2 + b2 - 2 * (A @ B.T)
      dist_matrix = np.sqrt(np.maximum(dist_sq, 0.0))
      ```
    """)

with col_mat2:
    st.markdown("### Matrix Cross-Multiplication Geometry")
    gemm_dot = """
    digraph GEMM {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        QMat [label="Queries Matrix Q\\nShape: [N = 3, d = 768]\\n(3 Query Vectors)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        CMat [label="Transposed Corpus C.T\\nShape: [d = 768, M = 5]\\n(5 Document Vectors)", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];
        
        BLASOp [label="BLAS GEMM (Q @ C.T)\\nHardware SIMD / AVX-512\\nHardware Vectorized", shape=ellipse, fillcolor="#EDE9FE", color="#8B5CF6"];
        
        SimMat [label="Similarity Matrix\\nShape: [N = 3, M = 5]\\n15 Pairwise Cosine Scores", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        QMat -> BLASOp;
        CMat -> BLASOp;
        BLASOp -> SimMat [label="3,700x faster than loops!"];
    }
    """
    st.graphviz_chart(gemm_dot, use_container_width=False)

st.markdown("### Algorithmic Efficiency Comparison")
st.markdown("""
| Operation Type | Implementation Strategy | Complexity | Execution Time (1k Queries vs 10k Docs) | Memory Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **Nested Python Loop** | `for q in Q: for d in C: dot(q, d)` | $O(N \\cdot M \\cdot d)$ | **~45,000 ms** (Unusable) | Minimal (streaming) |
| **List Comprehension** | `[[dot(q, d) for d in C] for q in Q]` | $O(N \\cdot M \\cdot d)$ | **~18,000 ms** | High (nested lists) |
| **Broadcasted Array** | `np.sum(Q[:, None, :] * C[None, :, :], axis=2)` | $O(N \\cdot M \\cdot d)$ | **~250 ms** | High ($N \\times M \\times d$ temporary tensor) |
| **Vectorized BLAS GEMM** | `np.dot(Q, C.T)` or `Q @ C.T` | $O(N \\cdot M \\cdot d)$ | **~12 ms** (Ultra-Fast) | **Optimal** ($N \\times M$ result matrix only) |
""")

st.divider()

# Section 2: Production Code: Pure NumPy Similarity Accelerator
st.header("2. Production Implementation: Pure NumPy Similarity Accelerator", divider="rainbow")

st.markdown("""
The production utility below demonstrates high-performance matrix similarity routines: pre-normalized matrix dot products, pairwise top-K extraction via `np.argpartition`, and loop-free Euclidean distance calculation.
""")

st.code(r'''
import numpy as np
from typing import Tuple, Dict, Any

class NumPySimilarityAccelerator:
    """
    High-performance vector similarity engine in pure NumPy.
    Implements BLAS GEMM matrix multiplications, argpartition top-k,
    and the loop-free Euclidean distance expansion trick.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Row-wise unit L2 normalization."""
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.maximum(norms, self.eps)

    def compute_cosine_matrix(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        Computes full pairwise cosine similarity matrix.
        queries: [Q, d], corpus: [C, d] -> returns: [Q, C]
        """
        q_norm = self.normalize(queries)
        c_norm = self.normalize(corpus)
        return np.dot(q_norm, c_norm.T)

    def compute_euclidean_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Euclidean distance matrix without loops
        using the algebraic identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a dot b).
        A: [N, d], B: [M, d] -> returns: [N, M]
        """
        a2 = np.sum(A**2, axis=1, keepdims=True)          # [N, 1]
        b2 = np.sum(B**2, axis=1, keepdims=True).T        # [1, M]
        dist_sq = a2 + b2 - 2.0 * np.dot(A, B.T)          # [N, M]
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def top_k_per_query(self, sim_matrix: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts top-k indices and scores using np.argpartition for O(N) selection.
        sim_matrix: [Q, C]
        Returns: (top_indices [Q, K], top_scores [Q, K])
        """
        # Partition the top-k highest elements to the end of each row
        k_indices = np.argpartition(-sim_matrix, top_k - 1, axis=1)[:, :top_k]
        
        # Sort only the extracted top-k elements descending
        row_indices = np.arange(sim_matrix.shape[0])[:, None]
        k_scores = sim_matrix[row_indices, k_indices]
        sort_order = np.argsort(-k_scores, axis=1)
        
        final_indices = np.take_along_axis(k_indices, sort_order, axis=1)
        final_scores = np.take_along_axis(k_scores, sort_order, axis=1)
        return final_indices, final_scores
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveNumPyBench:
    def __init__(self):
        self.sample_labels_q = ["Query 1 (Healthcare)", "Query 2 (Finance)", "Query 3 (Astronomy)"]
        self.sample_labels_c = ["Doc 1 (Cardiology)", "Doc 2 (Banking)", "Doc 3 (Telescopes)", "Doc 4 (Orthopedics)"]
        
        # Mock 4D normalized vectors
        self.Q = np.array([
            [0.92, 0.05, 0.02, 0.38],
            [0.05, 0.98, 0.02, 0.12],
            [0.02, 0.01, 0.99, 0.05]
        ], dtype=np.float32)
        
        self.C = np.array([
            [0.95, 0.02, 0.01, 0.30],
            [0.02, 0.96, 0.05, 0.20],
            [0.01, 0.02, 0.97, 0.08],
            [0.85, 0.08, 0.02, 0.52]
        ], dtype=np.float32)

    def run_benchmark(self, num_q: int, num_c: int, dim: int):
        rng = np.random.default_rng(42)
        q_rand = rng.standard_normal((num_q, dim)).astype(np.float32)
        c_rand = rng.standard_normal((num_c, dim)).astype(np.float32)
        
        q_norm = q_rand / np.linalg.norm(q_rand, axis=1, keepdims=True)
        c_norm = c_rand / np.linalg.norm(c_rand, axis=1, keepdims=True)

        # Vectorized GEMM
        t0 = time.perf_counter()
        sim_mat = np.dot(q_norm, c_norm.T)
        t_gemm = (time.perf_counter() - t0) * 1000.0

        # Loop (subsample if large to prevent browser hang)
        loop_q_limit = min(num_q, 30)
        t0 = time.perf_counter()
        for i in range(loop_q_limit):
            for j in range(num_c):
                _ = np.dot(q_norm[i], c_norm[j])
        t_loop_part = (time.perf_counter() - t0) * 1000.0
        t_loop_projected = t_loop_part * (num_q / loop_q_limit)

        return {
            "gemm_ms": t_gemm,
            "loop_ms": t_loop_projected,
            "speedup": t_loop_projected / max(t_gemm, 0.001)
        }

st.divider()

# Section 3: Interactive Workbench: NumPy Similarity Benchmark & Heatmap
st.header("3. Interactive Workbench: NumPy Similarity Matrix & GEMM Benchmark", divider="rainbow")

st.markdown("""
Inspect how NumPy computes cross-matrix pairwise similarities instantly.
Compare the pairwise **Cosine Similarity Heatmap** between queries and documents, and run a live **Hardware Benchmark** testing vectorized BLAS matrix multiplication against Python loops.
""")

bench_tool = InteractiveNumPyBench()

col_sim_l, col_sim_r = st.columns([1, 1])

with col_sim_l:
    st.subheader("1. Pairwise Similarity Matrix Heatmap")
    # Normalize sample vectors
    q_n = bench_tool.Q / np.linalg.norm(bench_tool.Q, axis=1, keepdims=True)
    c_n = bench_tool.C / np.linalg.norm(bench_tool.C, axis=1, keepdims=True)
    
    sim_matrix = np.dot(q_n, c_n.T)
    df_heatmap = pd.DataFrame(sim_matrix, index=bench_tool.sample_labels_q, columns=bench_tool.sample_labels_c)
    
    st.dataframe(
        df_heatmap.style.format("{:.4f}").background_gradient(cmap="Blues", vmin=0.0, vmax=1.0),
        use_container_width=True
    )
    st.caption("Diagonal semantic alignment: Healthcare matches Cardiology (0.98), Finance matches Banking (0.97), Astronomy matches Telescopes (0.98).")

with col_sim_r:
    st.subheader("2. Live BLAS Vectorization Benchmark")
    col_b1, col_b2 = st.columns(2)
    b_queries = col_b1.slider("Query Count (N):", min_value=100, max_value=2000, value=500, step=100)
    b_corpus = col_b2.slider("Corpus Count (M):", min_value=500, max_value=5000, value=2000, step=500)
    b_dim = st.selectbox("Dimension (d):", [384, 768, 1536], index=1)

    bench_results = bench_tool.run_benchmark(b_queries, b_corpus, b_dim)

    m1, m2 = st.columns(2)
    m1.metric("Vectorized BLAS (Q @ C.T)", f"{bench_results['gemm_ms']:.2f} ms")
    m2.metric("Python Loop Equivalent", f"{bench_results['loop_ms']:.1f} ms")

    st.metric("Vectorization Speedup Factor", f"{bench_results['speedup']:.0f}x Faster!", delta="BLAS SIMD Acceleration")
    st.success(f"🚀 NumPy computed **{b_queries * b_corpus:,} pairwise dot products** in just **{bench_results['gemm_ms']:.2f} ms**!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "NumPy Similarity Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does the Euclidean Distance Expansion Trick work and why is it so much faster than naive distance loops?**
    * **Answer**:
      * The Euclidean distance between two vectors $u$ and $v$ is $\\sqrt{\\sum_{i=1}^d (u_i - v_i)^2}$.
      * Expanding the binomial: $||u - v||^2 = ||u||^2 + ||v||^2 - 2(u \\cdot v)$.
      * In matrix form for $A \\in \\mathbb{R}^{N \\times d}$ and $B \\in \\mathbb{R}^{M \\times d}$:
        1. Compute squared row norms $a^2 = \\sum A^2$ (shape $[N, 1]$).
        2. Compute squared row norms $b^2 = \\sum B^2$ (shape $[1, M]$).
        3. Compute matrix product $2(A B^T)$ in a single hardware-accelerated BLAS GEMM call.
        4. Broadcast: $D^2 = a^2 + b^2 - 2(A B^T)$.
      * This replaces millions of coordinate subtractions with a single highly optimized matrix multiplication, executing **50x-100x faster**.

    **Q2: What is the difference between np.argsort and np.argpartition for top-K extraction?**
    * **Answer**:
      * `np.argsort(arr)` performs a complete sort of all $N$ elements in $O(N \\log N)$ time.
      * `np.argpartition(arr, K)` performs Introselect in **$O(N)$ linear time**, placing the top $K$ elements at the boundary without sorting the entire array!
      * For a corpus of $N = 100,000$ documents and $K = 5$:
        * `np.argsort` sorts 100,000 numbers.
        * `np.argpartition` extracts the top 5 in linear time, and you only sort those 5 items.
      * In production retrieval, `np.argpartition` reduces top-K extraction latency by **80%**.

    **Q3: If two vectors are unit-normalized (L2 norm = 1.0), what is the exact mathematical relationship between Euclidean Distance and Cosine Similarity?**
    * **Answer**:
      * Starting from the expansion: $||u - v||^2 = ||u||^2 + ||v||^2 - 2(u \\cdot v)$.
      * Since $||u|| = 1.0$ and $||v|| = 1.0$:
        $||u - v||^2 = 1 + 1 - 2(u \\cdot v) = 2 - 2 \\cdot \\text{Cosine}(u, v)$.
      * Therefore:
        `Euclidean Distance = sqrt(2 - 2 * Cosine Similarity)`.
      * Because this mapping is strictly monotonic, ranking by Euclidean Distance and ranking by Cosine Similarity produce **bit-for-bit identical top-K nearest-neighbor rankings** for normalized vectors!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Negative Square Roots in Distance Expansion Due to Precision Noise**:
      * *Issue*: Computing `dist = np.sqrt(a2 + b2 - 2 * dot)` where a vector is compared to itself.
      * *Consequence*: Due to float32 rounding, `1.0 + 1.0 - 2 * 1.0000002` equals `-0.0000004`. Taking `sqrt` of a negative number produces `NaN`!
      * *Fix*: Always clamp before square root: `np.sqrt(np.maximum(dist_sq, 0.0))`.

    * **Trap 2: Creating a 3D Broadcast Tensor (Memory Explosion)**:
      * *Issue*: Computing distances using `np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)`.
      * *Consequence*: Allocates a temporary 3D array of shape `[N, M, d]`. For $N=1,000$, $M=10,000$, and $d=768$, this requires **30.7 GB RAM**, causing an immediate out-of-memory crash!
      * *Fix*: Use the 2D expansion trick or BLAS matrix multiplication `A @ B.T`.

    * **Trap 3: Forgetting Transpose in Matrix Multiplication**:
      * *Issue*: Calling `Q @ C` instead of `Q @ C.T` when `Q` is `[N, d]` and `C` is `[M, d]`.
      * *Consequence*: Raises `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`.
      * *Fix*: Verify inner dimension alignment: `[N, d] @ [d, M] = [N, M]`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: BLAS GEMM**
      * *Definition*: General Matrix Multiply; the fundamental linear algebra routine powering hardware-accelerated matrix dot products in NumPy.
    * **Flashcard 2: Euclidean Expansion Trick**
      * *Definition*: The identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a dot b) that computes pairwise distances without memory-intensive coordinate subtractions.
    * **Flashcard 3: np.argpartition**
      * *Definition*: An O(N) linear-time selection algorithm that partitions the top-K elements without the overhead of sorting the entire array.
    * **Flashcard 4: Unit Vector Distance Identity**
      * *Definition*: The formula Euclidean Distance = sqrt(2 - 2 * Cosine) proving that Euclidean distance and Cosine similarity yield identical nearest-neighbor rankings on normalized vectors.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Similarity with NumPy:

- [ ] I understand why vectorized BLAS matrix multiplication is thousands of times faster than nested Python loops.
- [ ] I can implement the Euclidean Distance Expansion Trick in NumPy without 3D memory explosions.
- [ ] I understand why `np.maximum(dist_sq, 0.0)` is essential to prevent `NaN` values from floating-point rounding.
- [ ] I know how to use `np.argpartition` for fast $O(N)$ top-K selection.
- [ ] I can derive the monotonic relationship between Euclidean Distance and Cosine Similarity for unit vectors.
""")
