import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.1 Basic Embedding Generation - Embeddings Curriculum",
    page_icon="🧪",
    layout="wide"
)

# Title & Overview
st.title("6.1 Basic Embedding Generation: Model Instantiation, Vector Inspection, & Coordinate Diagnostics")
st.caption("Chapter 6: Generating Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
At its core, **embedding generation** is the programmatic transformation of an unstructured string into a dense mathematical vector.
While calling `.encode(text)` or `client.embeddings.create(input=text)` feels like a single operation, a multi-stage compilation occurs under the hood: text normalization, tokenization, contextual forward-passing, pooling, and metric normalization.

Before deploying embeddings into production vector databases, an engineer must understand how to:
* **Instantiate and inspect embedding models** safely across environments.
* **Inspect the raw coordinate values**: floating-point precision, distribution (mean, variance), min/max ranges, and sparsity.
* **Verify vector dimensions**: confirming that the geometry matches downstream index expectations (e.g. 384, 768, 1536, 3072).
* **Perform sanity tests**: validating that identical inputs yield identical vectors (determinism) and that the Euclidean norm conforms to expectations.

This lesson walks through the essential mechanics of basic embedding generation and coordinate diagnostics.
""")

st.divider()

# Section 1: The Anatomy of Vector Generation
st.header("1. The Anatomy of Vector Generation & Coordinate Inspection", divider="rainbow")

col_ana1, col_ana2 = st.columns([1, 1])

with col_ana1:
    st.markdown("""
    ### What Really Happens Inside the Generator:
    * **1. Input Sanitization**: Stripping erratic whitespace, trailing newlines, or invisible control characters.
    * **2. Tokenization**: Converting the input string into a sequence of discrete vocabulary IDs (e.g. WordPiece or BPE tokens) plus a binary attention mask.
    * **3. Model Forward Pass**: Computing high-dimensional contextual hidden activations for each token across multiple transformer layers.
    * **4. Pooling**: Aggregating the sequence of token vectors into a single fixed-width vector (typically via masked mean pooling).
    * **5. Metric Normalization**: Projecting the raw pooled vector onto the unit hypersphere (L2 norm = 1.0) so that cosine similarity can be computed via blazing-fast dot products.

    ### Key Coordinate Diagnostic Metrics:
    * **Dimensionality (d)**: Total number of coordinate axes (e.g. 384 for MiniLM, 1536 for Ada/3-Small, 3072 for 3-Large).
    * **L2 Euclidean Norm**: sqrt(sum(v_i ^ 2)). For normalized vectors, this must equal exactly 1.000000 +/- 1e-6.
    * **Mean Activation (mu)**: The average coordinate value. In an isotropic, zero-centered space, mu is approximately 0.0.
    * **Variance (sigma^2)**: Spread of coordinates. For a normalized vector of dimension d, expected variance is approx 1 / d.
    * **Min / Max Values**: Typical range for normalized 768-d vectors is between -0.15 and +0.15. Extreme outliers may signal numerical instability.
    """)

with col_ana2:
    st.markdown("### Step-by-Step Embedding Pipeline")
    flow_dot = """
    digraph BasicGen {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawText [label="Raw Text String\\n'Machine learning models learn patterns.'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Tokens [label="Tokenization\\nToken IDs: [101, 3698, 4083, ..., 102]\\nAttention Mask: [1, 1, 1, ..., 1]", fillcolor="#FEF3C7"];
        
        Backbone [label="Transformer Encoder\\nHidden States: [1, seq_len, d]", fillcolor="#D1FAE5"];
        
        Pool [label="Pooling Layer\\nAggregates to [1, d]", fillcolor="#EDE9FE", color="#8B5CF6"];
        
        Norm [label="Unit L2 Normalization\\nv = v / ||v||", fillcolor="#FCE7F3"];
        
        VectorInspect [label="Final Embedding Vector\\nd-dimensional Float32 Array\\nNorm = 1.0000", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        RawText -> Tokens -> Backbone -> Pool -> Norm -> VectorInspect;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# Section 2: Production Code: Robust Vector Inspector
st.header("2. Production Implementation: Embedding Diagnostic Inspector", divider="rainbow")

st.markdown("""
A production embedding utility should not only produce vectors, but also run **integrity assertions** to detect silent model degradation, dimension mismatches, or unnormalized vectors before they reach the database.
""")

st.code(r'''
import numpy as np
from typing import Dict, Any, List

class EmbeddingInspector:
    """
    Production-grade vector inspection and diagnostic utility.
    Validates dimensionality, L2 normalization, coordinate statistics, and sparsity.
    """
    def __init__(self, expected_dim: int = 384, tolerance: float = 1e-5):
        self.expected_dim = expected_dim
        self.tolerance = tolerance

    def inspect_vector(self, vector: np.ndarray, label: str = "sample") -> Dict[str, Any]:
        """
        Runs comprehensive statistical diagnostics on an embedding vector.
        """
        vec = np.asarray(vector, dtype=np.float64).flatten()
        actual_dim = len(vec)
        
        # 1. Dimensionality Verification
        dim_match = (actual_dim == self.expected_dim)
        
        # 2. L2 Euclidean Norm Verification
        l2_norm = float(np.linalg.norm(vec))
        is_normalized = abs(l2_norm - 1.0) <= self.tolerance
        
        # 3. Coordinate Statistics
        coord_mean = float(np.mean(vec))
        coord_std = float(np.std(vec))
        coord_min = float(np.min(vec))
        coord_max = float(np.max(vec))
        
        # 4. Sparsity Audit (detect unexpected all-zero coordinates)
        zero_count = int(np.sum(np.abs(vec) < 1e-9))
        sparsity_pct = (zero_count / actual_dim) * 100.0
        
        return {
            "label": label,
            "dimension": actual_dim,
            "dimension_valid": dim_match,
            "l2_norm": l2_norm,
            "is_normalized": is_normalized,
            "mean": coord_mean,
            "std": coord_std,
            "min": coord_min,
            "max": coord_max,
            "zero_coords": zero_count,
            "sparsity_pct": sparsity_pct,
            "sample_coords": vec[:5].tolist()
        }
''', language="python")

# Actual Python implementation for the interactive workbench
class EmbeddingInspector:
    def __init__(self, expected_dim: int = 384, tolerance: float = 1e-4):
        self.expected_dim = expected_dim
        self.tolerance = tolerance

    def inspect_vector(self, vector: np.ndarray, label: str = "sample") -> Dict[str, Any]:
        vec = np.asarray(vector, dtype=np.float64).flatten()
        actual_dim = len(vec)
        dim_match = (actual_dim == self.expected_dim)
        l2_norm = float(np.linalg.norm(vec))
        is_normalized = abs(l2_norm - 1.0) <= self.tolerance
        coord_mean = float(np.mean(vec))
        coord_std = float(np.std(vec))
        coord_min = float(np.min(vec))
        coord_max = float(np.max(vec))
        zero_count = int(np.sum(np.abs(vec) < 1e-9))
        sparsity_pct = (zero_count / actual_dim) * 100.0
        return {
            "label": label,
            "dimension": actual_dim,
            "dimension_valid": dim_match,
            "l2_norm": l2_norm,
            "is_normalized": is_normalized,
            "mean": coord_mean,
            "std": coord_std,
            "min": coord_min,
            "max": coord_max,
            "zero_coords": zero_count,
            "sparsity_pct": sparsity_pct,
            "sample_coords": vec[:6].tolist()
        }

st.divider()

# Section 3: Interactive Workbench: Vector Inspection Laboratory
st.header("3. Interactive Workbench: Vector Inspection & Diagnostics Laboratory", divider="rainbow")

st.markdown("""
Type or select an input text, choose an embedding architecture, and inspect the resulting dense vector coordinates, distribution histograms, and diagnostic integrity checks.
""")

col_in, col_diag = st.columns([1, 1])

with col_in:
    st.subheader("1. Embedding Configuration")
    user_text = st.text_area(
        "Input Text to Embed:",
        value="Dense embeddings represent semantic meaning as coordinates in high-dimensional geometric space.",
        height=100
    )

    model_preset = st.selectbox(
        "Simulated Model Architecture:",
        [
            "all-MiniLM-L6-v2 (384-d, Mean Pooled, Normalized)",
            "bge-base-en-v1.5 (768-d, Normalized)",
            "text-embedding-3-small (1536-d, OpenAI)",
            "text-embedding-3-large (3072-d, OpenAI)",
            "Unnormalized Raw Transformer Output (768-d, Non-Unit Norm)"
        ]
    )

    # Parse dimensions and normalization from selection
    if "384" in model_preset:
        dim = 384
        normalize = True
    elif "1536" in model_preset:
        dim = 1536
        normalize = True
    elif "3072" in model_preset:
        dim = 3072
        normalize = True
    elif "Unnormalized" in model_preset:
        dim = 768
        normalize = False
    else:
        dim = 768
        normalize = True

    # Generate synthetic vector deterministically from text
    rng = np.random.default_rng(abs(hash(user_text)) % (2**31))
    raw_coords = rng.standard_normal(dim)
    if not normalize:
        # Scale up so norm != 1
        raw_coords = raw_coords * 4.5 + 2.0
        generated_vec = raw_coords
    else:
        generated_vec = raw_coords / np.linalg.norm(raw_coords)

with col_diag:
    st.subheader("2. Coordinate Diagnostics")
    inspector = EmbeddingInspector(expected_dim=dim)
    diag = inspector.inspect_vector(generated_vec, label=user_text[:20])

    m1, m2 = st.columns(2)
    m1.metric("Dimension Count", f"{diag['dimension']} dims", delta="Matches Target" if diag['dimension_valid'] else "Mismatch!")
    m2.metric("Euclidean Norm (L2)", f"{diag['l2_norm']:.6f}", delta="Unit Normalized" if diag['is_normalized'] else "Unnormalized!", delta_color="normal" if diag['is_normalized'] else "inverse")

    m3, m4 = st.columns(2)
    m3.metric("Mean Coordinate Value", f"{diag['mean']:.6f}")
    m4.metric("Std Deviation", f"{diag['std']:.6f}")

    st.markdown("#### First 6 Coordinate Values")
    sample_df = pd.DataFrame({
        "Index": [f"coord_{i}" for i in range(len(diag['sample_coords']))],
        "Value (Float32)": [f"{v:.6f}" for v in diag['sample_coords']]
    })
    st.dataframe(sample_df, use_container_width=True)

    if diag['is_normalized']:
        st.success("✅ **Integrity Verified**: Vector is strictly unit-normalized (L2 norm = 1.0) and ready for cosine similarity dot-product indexing.")
    else:
        st.error("❌ **Integrity Alert**: Vector is NOT unit-normalized. Dot products will reflect vector magnitude rather than angular semantic similarity!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Generation Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is inspecting the Euclidean L2 norm of an embedding vector critical before storing it in a vector database?**
    * **Answer**:
      * Most modern vector search engines (ChromaDB, Pinecone, FAISS, Qdrant) use **Inner Product (IP)** distance for maximum query speed.
      * Inner Product is mathematically identical to Cosine Similarity **only if both vectors have an L2 norm of exactly 1.0**:
        `Cosine Similarity = (u dot v) / (||u|| * ||v||) = u dot v (when ||u|| = ||v|| = 1)`
      * If a vector is unnormalized (norm != 1.0), the dot product is heavily biased by vector magnitude: longer sentences or text with extreme weights will dominate search results regardless of semantic relevance.

    **Q2: What is the relationship between embedding dimension d and expected coordinate variance?**
    * **Answer**:
      * For a uniformly distributed vector on the unit sphere in d dimensions:
        The sum of squared coordinates equals 1: `sum(v_i ^ 2) = 1`.
      * Assuming zero mean, the expected value of each squared coordinate is:
        `E[v_i ^ 2] = 1 / d`.
      * Therefore, the expected standard deviation of coordinates is `1 / sqrt(d)`.
      * In a 384-dimensional vector, `std ~ 1 / sqrt(384) ~ 0.051`.
      * In a 1536-dimensional vector, `std ~ 1 / sqrt(1536) ~ 0.025`.
      * Coordinates in higher-dimensional embeddings naturally possess smaller individual magnitudes.

    **Q3: How do you verify that an embedding pipeline is strictly deterministic?**
    * **Answer**:
      * During inference, embedding models should be purely deterministic: embedding the exact same string twice must produce bit-for-bit identical vectors.
      * Verification:
        1. Set `model.eval()` to turn off dropout layers.
        2. Disable any stochastic decoding or non-deterministic GPU kernel execution (`torch.use_deterministic_algorithms(True)`).
        3. Assert `np.allclose(vec_1, vec_2, atol=1e-7)`.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Passing Whitespace-Only or Empty Strings**:
      * *Issue*: Embedding empty strings `""` or `"\n\n\t"`.
      * *Consequence*: The tokenizer generates only special tokens (`[CLS], [SEP]`). The resulting vector represents "empty syntax" rather than meaningful semantics, polluting clustering indexes.
      * *Fix*: Implement pre-filtering: assert `len(text.strip()) > 0` before calling the model.

    * **Trap 2: Floating Point Precision Loss in Serialization**:
      * *Issue*: Saving float32 vectors to CSV or JSON as text strings with truncated decimal precision (e.g. `f"{v:.2f}"`).
      * *Consequence*: Precision drops from 32-bit (7 decimal places) to 2 decimal places, destroying directional angles and reducing retrieval accuracy.
      * *Fix*: Store embeddings in binary formats: NumPy `.npy`, Arrow/Parquet, or native vector database binary blobs.

    * **Trap 3: Dimension Mismatch Across Pipeline Steps**:
      * *Issue*: Initializing an index for 1536 dimensions and accidentally feeding 768-dimensional vectors from a different local model.
      * *Consequence*: Crashes with dimension assertion errors, or corrupts flat binary index buffers.
      * *Fix*: Validate `len(vector) == index.dimension` in ingestion pre-flight checks.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Euclidean L2 Norm**
      * *Definition*: The square root of the sum of squared coordinate values, measuring the vector's geometric length.
    * **Flashcard 2: Unit Normalization**
      * *Definition*: Dividing a vector by its L2 norm to force its length to 1.0, enabling direct dot-product cosine similarity.
    * **Flashcard 3: Embedding Dimensionality (d)**
      * *Definition*: The fixed number of continuous floating-point features produced by an embedding model's final projection head.
    * **Flashcard 4: Determinism**
      * *Definition*: The guarantee that feeding identical input text into an embedding model in evaluation mode always yields identical vector coordinates.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Basic Embedding Generation:

- [ ] I can describe the 5 steps that transform an input string into a dense unit vector.
- [ ] I understand why vector coordinates in higher dimensions naturally have smaller standard deviations.
- [ ] I know why unit L2 normalization is mandatory for fast inner-product vector database indexing.
- [ ] I can implement an automated coordinate diagnostic utility in Python.
- [ ] I know how to check for empty strings, dimension mismatches, and precision loss during vector serialization.
""")
