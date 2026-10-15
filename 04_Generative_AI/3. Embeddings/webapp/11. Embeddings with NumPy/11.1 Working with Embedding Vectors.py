import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="11.1 Working with Embedding Vectors - Embeddings Curriculum",
    page_icon="🔢",
    layout="wide"
)

# Title & Overview
st.title("11.1 Working with Embedding Vectors in NumPy: Memory Layouts, L2 Normalization, & BLAS Operations")
st.caption("Chapter 11: Embeddings with NumPy | Domain: 3. Vector Embeddings")
st.markdown("""
While machine learning frameworks like PyTorch and Hugging Face are used to generate embeddings, **NumPy is the universal standard for manipulating, storing, and indexing embeddings in production**.
Whether persisting vectors to disk, computing pairwise similarity matrices, or preparing payloads for vector databases, an engineer must master low-level NumPy vector mechanics:
* **Memory Layout & Precision**: Understanding why `float32` is the industry standard (halving RAM compared to default Python `float64`).
* **Array Shapes**: Navigating 1D vectors `(d,)`, row vectors `(1, d)`, and 2D batch matrices `(N, d)`.
* **Vectorized L2 Normalization**: Executing safe, broadcasted Euclidean normalization across millions of rows with zero-division safeguards.
* **BLAS-Accelerated Matrix Multiplication**: Harnessing hardware SIMD and AVX vector instructions for high-throughput math.

This lesson explores foundational NumPy array operations for high-dimensional embedding engineering.
""")

st.divider()

# Section 1: NumPy Memory Layouts, Precision, & Array Shapes
st.header("1. NumPy Memory Layouts, Precision, & Array Shapes", divider="rainbow")

col_np1, col_np2 = st.columns([1, 1])

with col_np1:
    st.markdown("""
    ### Why NumPy Dominates Embedding Workflows:
    * **Contiguous C-Memory Layout**: Python lists store pointers to heap objects; NumPy stores raw contiguous float buffers in cache-friendly memory.
    * **The Float32 Rule**:
      * Python defaults to 64-bit double precision (`float64`, 8 bytes per number).
      * Neural network weights and embeddings use **32-bit single precision** (`float32`, 4 bytes per number).
      * For 1,000,000 vectors of dimension 1,536:
        * `float64`: **12.28 GB RAM**
        * `float32`: **6.14 GB RAM** (50% memory savings with ZERO loss of retrieval precision!)
    * **Array Shape Conventions**:
      * Single query vector: shape `(d,)` or `(1, d)`.
      * Corpus embedding matrix: shape `(N, d)` where $N$ is document count and $d$ is dimension.
    """)

with col_np2:
    st.markdown("### Matrix Shape & Normalization Flow")
    np_dot = """
    digraph NumPyArray {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        PyList [label="Python Raw List\\n[[0.12, -0.45, ...], ...]", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        CastStep [label="np.asarray(..., dtype=np.float32)\\nContiguous C-Order Buffer\\nShape: [N, d]", fillcolor="#FEF3C7"];
        
        NormStep [label="np.linalg.norm(arr, axis=1, keepdims=True)\\nComputes Euclidean lengths: [N, 1]", fillcolor="#FDE68A"];
        
        BroadcastDiv [label="arr / np.maximum(norms, 1e-12)\\nBroadcasted in-place division", fillcolor="#D1FAE5"];
        
        UnitMatrix [label="Unit L2 Matrix\\nAll rows have norm = 1.0000\\nReady for BLAS GEMM Dot Product", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        PyList -> CastStep -> NormStep -> BroadcastDiv -> UnitMatrix;
    }
    """
    st.graphviz_chart(np_dot, use_container_width=False)

st.markdown("### Numeric Precision & Memory Footprints for 100,000 Vectors (d = 1,536)")
st.markdown("""
| Precision Type | Bytes per Float | Storage per Vector | Total RAM Footprint (100k Vectors) | Retrieval Impact | Recommended Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Float64 (double)** | 8 bytes | 12,288 bytes | **1.23 GB** | Identical | Overkill (wastes memory) |
| **Float32 (single)** | 4 bytes | 6,144 bytes | **614.4 MB** | **Reference Standard** | **Production Vector DB Default** |
| **Float16 (half)** | 2 bytes | 3,072 bytes | **307.2 MB** | <0.1% loss | GPU VRAM optimization |
| **Int8 (quantized)** | 1 byte | 1,536 bytes | **153.6 MB** | ~1.5% loss | Memory-constrained edge deployment |
""")

st.divider()

# Section 2: Production Code: High-Performance NumPy Vector Toolset
st.header("2. Production Implementation: NumPy Vector Toolkit", divider="rainbow")

st.markdown("""
The production utility below demonstrates essential NumPy vector routines: precision casting, row-wise norm calculation, zero-safe broadcasting normalization, and array stacking.
""")

st.code(r'''
import numpy as np
from typing import List, Tuple, Dict, Any

class NumPyVectorToolkit:
    """
    Production-grade NumPy utility for high-dimensional embedding manipulations.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def to_float32_matrix(self, vector_list: List[List[float]]) -> np.ndarray:
        """Converts arbitrary Python list of lists to contiguous float32 matrix."""
        return np.ascontiguousarray(vector_list, dtype=np.float32)

    def compute_magnitudes(self, matrix: np.ndarray) -> np.ndarray:
        """
        Computes L2 Euclidean norm for each row vector.
        matrix: [N, d] -> returns: [N, 1]
        """
        return np.linalg.norm(matrix, axis=-1, keepdims=True)

    def normalize_matrix(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Safely normalizes matrix rows to unit length using broadcasting.
        Guards against zero-division using eps clamp.
        """
        norms = self.compute_magnitudes(matrix)
        normalized = matrix / np.maximum(norms, self.eps)
        return normalized, norms

    def batch_dot_product(self, query_matrix: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
        """
        BLAS-accelerated matrix multiplication for cosine similarity.
        queries: [Q, d], corpus: [C, d] -> returns: [Q, C]
        """
        # Assumes both matrices are pre-normalized to unit length
        return np.dot(query_matrix, corpus_matrix.T)
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveNumPyWorkbench:
    def __init__(self):
        self.eps = 1e-12

    def calculate_memory_and_stats(self, num_vectors: int, dim: int, precision: str) -> Dict[str, Any]:
        bytes_map = {"Float64 (8 bytes)": 8, "Float32 (4 bytes)": 4, "Float16 (2 bytes)": 2, "Int8 (1 byte)": 1}
        bytes_per_coord = bytes_map[precision]
        total_bytes = num_vectors * dim * bytes_per_coord
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_mb / 1024

        # Generate sample vectors to test normalization
        rng = np.random.default_rng(42)
        raw_mat = rng.standard_normal((3, dim)) * 3.5 + 1.2
        raw_norms = np.linalg.norm(raw_mat, axis=-1, keepdims=True)
        norm_mat = raw_mat / np.maximum(raw_norms, self.eps)
        verified_norms = np.linalg.norm(norm_mat, axis=-1)

        return {
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "total_gb": total_gb,
            "bytes_per_vec": dim * bytes_per_coord,
            "sample_raw_norms": raw_norms.flatten().tolist(),
            "sample_verified_norms": verified_norms.tolist(),
            "sample_coords": norm_mat[0, :5].tolist()
        }

st.divider()

# Section 3: Interactive Workbench: NumPy Vector Laboratory
st.header("3. Interactive Workbench: NumPy Vector Dimension & Memory Laboratory", divider="rainbow")

st.markdown("""
Configure embedding collection parameters below. Observe the exact **Memory Footprints (RAM)**, calculate row-wise Euclidean norms, and verify **Broadcasted L2 Normalization** across tensor coordinates.
""")

np_tool = InteractiveNumPyWorkbench()

col_lab_l, col_lab_r = st.columns([1, 1])

with col_lab_l:
    st.subheader("1. Ingestion Sizing Parameters")
    num_vecs = st.number_input("Number of Vectors in Collection (N):", min_value=1_000, max_value=50_000_000, value=250_000, step=50_000)
    
    dim_sel = st.selectbox(
        "Vector Dimensionality (d):",
        [
            "384 (e.g. all-MiniLM-L6-v2, bge-small)",
            "768 (e.g. all-mpnet-base-v2, nomic-embed)",
            "1536 (e.g. text-embedding-3-small, ada-002)",
            "3072 (e.g. text-embedding-3-large)"
        ]
    )
    dim_val = int(dim_sel.split()[0])

    prec_sel = st.radio("Numeric Precision Format:", ["Float32 (4 bytes) — Standard", "Float64 (8 bytes)", "Float16 (2 bytes)", "Int8 (1 byte)"], horizontal=True)
    clean_prec = prec_sel.split(" — ")[0]

with col_lab_r:
    st.subheader("2. Memory Footprint & Tensor Statistics")
    stats = np_tool.calculate_memory_and_stats(num_vecs, dim_val, clean_prec)

    m1, m2 = st.columns(2)
    m1.metric("Memory Footprint", f"{stats['total_mb']:.1f} MB ({stats['total_gb']:.2f} GB)")
    m2.metric("Per-Vector Size", f"{stats['bytes_per_vec']:,} bytes")

    st.markdown("#### Normalization Verification Diagnostic")
    df_diag = pd.DataFrame({
        "Sample Row": ["Row 1", "Row 2", "Row 3"],
        "Raw Unnormalized Norm": [f"{v:.4f}" for v in stats["sample_raw_norms"]],
        "Post-L2 Normalized Norm": [f"{v:.6f}" for v in stats["sample_verified_norms"]]
    })
    st.dataframe(df_diag, use_container_width=True)

    st.markdown("#### Sample Normalized Coordinates (First 5 Dimensions):")
    st.code(f"[{', '.join([f'{v:.6f}' for v in stats['sample_coords']])}, ...]", language="python")

    if "Float32" in clean_prec:
        st.success("✅ **Optimal Precision**: Float32 provides optimal numeric stability with 50% RAM savings over standard Python float64.")
    elif "Float64" in clean_prec:
        st.warning("⚠️ **Memory Inefficiency**: Float64 consumes double the RAM of Float32 without any measurable improvement in cosine similarity ranking!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "NumPy Vector Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why should text embeddings always be stored as float32 rather than float64 in NumPy?**
    * **Answer**:
      * In Python, numbers default to 64-bit IEEE double precision (`float64`, 8 bytes).
      * Neural network weights and GPU forward passes are trained and computed in 32-bit single precision (`float32`, 4 bytes) or half precision (16-bit).
      * Storing vectors as `float64` doubles RAM consumption and network serialization bandwidth without adding a single bit of useful semantic information.
      * For a corpus of 1 million 1536-d vectors, `float32` takes **6.14 GB**, whereas `float64` takes **12.28 GB**.
      * Always specify `dtype=np.float32` when initializing or converting embedding arrays.

    **Q2: How do you normalize a 2D NumPy array of embeddings safely in a single vectorized line?**
    * **Answer**:
      * The correct vectorized expression is:
        ```python
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / np.maximum(norms, 1e-12)
        ```
      * `axis=1` ensures the norm is computed independently across each row vector.
      * `keepdims=True` preserves the 2D shape `[N, 1]` so NumPy can broadcast division across the `[N, d]` matrix.
      * `np.maximum(norms, 1e-12)` clamps zero vectors to a tiny epsilon, preventing `RuntimeWarning: invalid value encountered in divide` and `NaN` propagation.

    **Q3: What is the difference between C-contiguous and Fortran-contiguous memory layouts for embedding matrices?**
    * **Answer**:
      * **C-Contiguous (Row-Major, Default in NumPy)**: Array elements along the same row are stored adjacent in physical RAM.
        * Ideal for embeddings because operations access all coordinates of a single vector together (`matrix[i, :]`).
      * **Fortran-Contiguous (Column-Major)**: Elements along the same column are stored adjacent.
      * If a matrix becomes non-contiguous (e.g. through strided slicing), BLAS operations like `np.dot()` copy memory, causing latency spikes.
      * Ensure contiguity with `np.ascontiguousarray(matrix)`.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting keepdims=True in Normalization**:
      * *Issue*: Calling `norms = np.linalg.norm(matrix, axis=1)` without `keepdims=True`.
      * *Consequence*: The shape collapses to 1D `(N,)`. Trying to divide `matrix / norms` triggers an improper broadcast error or incorrectly divides columns instead of rows!
      * *Fix*: Always pass `keepdims=True` to maintain `(N, 1)` shape.

    * **Trap 2: Creating Arrays from Lists with Mixed Precision**:
      * *Issue*: Converting a list containing a single Python `float` without setting `dtype`.
      * *Consequence*: NumPy auto-promotes the entire array to `float64`, silently doubling RAM usage.
      * *Fix*: Explicitly enforce `np.array(vectors, dtype=np.float32)`.

    * **Trap 3: Zero-Vector Division Producing NaNs**:
      * *Issue*: Dividing by `np.linalg.norm()` directly when an empty text produced an all-zero vector.
      * *Consequence*: Zero division produces `NaN`. A single `NaN` in a matrix corrupts dot-product calculations across all queries!
      * *Fix*: Always clamp with `np.maximum(norms, 1e-12)`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Float32 Precision**
      * *Definition*: The 32-bit floating-point standard (4 bytes) used for deep learning embeddings, offering optimal balance of accuracy and memory economy.
    * **Flashcard 2: keepdims=True**
      * *Definition*: A NumPy parameter that preserves the reduced axis as dimension 1, allowing seamless broadcasting across matrix operations.
    * **Flashcard 3: C-Contiguous Array**
      * *Definition*: Memory arrangement where row elements are adjacent in RAM, maximizing CPU cache line efficiency for vector dot products.
    * **Flashcard 4: Epsilon Clamping**
      * *Definition*: Clamping vector norms to a small positive constant (1e-12) before division to prevent NaN generation on zero vectors.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Working with Embedding Vectors in NumPy:

- [ ] I understand why `float32` is mandatory over `float64` for storing embeddings.
- [ ] I can write the single-line broadcasted row-normalization formula in NumPy.
- [ ] I know why `keepdims=True` is required when computing norms across matrix rows.
- [ ] I understand how epsilon clamping prevents `NaN` corruption during vector division.
- [ ] I can calculate the RAM footprint for arbitrary vector collections based on dimension and precision.
""")
