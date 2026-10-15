import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.2 Understanding Embedding Vectors - Embeddings Curriculum",
    page_icon="🔬",
    layout="wide"
)

# Title & Overview
st.title("3.2 Understanding Embedding Vectors: Dimensions, Float Precision, Distributions, & Latent Semantics")
st.caption("Chapter 3: Text Embeddings Basics | Domain: 3. Vector Embeddings")
st.markdown("""
When you print an embedding vector in Python, you see a long array of decimal numbers:
`[-0.0142, 0.0381, -0.0914, 0.0028, ..., 0.0512]`.

What do these individual numbers represent? Do specific dimensions correspond to human concepts like *"color"*, *"tense"*, or *"price"*?
Understanding the **anatomy and statistical properties of embedding vectors** is essential to engineering high-performance retrieval systems:
1. **Distributed Representations**: Unlike symbolic features, meaning in deep embeddings is **distributed across all dimensions simultaneously**; no single coordinate holds isolated concepts.
2. **Coordinate Distribution**: Well-trained embedding coordinates follow a **Gaussian bell curve centered near 0.0** with a narrow standard deviation (std approximately 0.02 to 0.05).
3. **The Anisotropy Phenomenon**: In many transformer models, raw embeddings occupy a narrow cone in vector space rather than dispersing uniformly, requiring normalization to restore angular contrast.
4. **Precision Economics**: Storing billions of vectors requires choosing the right floating-point format (**Float32 vs. Float16 vs. Int8 quantized**).

This lesson dives deep into the internal structure, coordinate distributions, and precision economics of embedding vectors.
""")

st.divider()

# Section 1: Statistical Anatomy & Precision Economics
st.header("1. Vector Statistical Properties & Precision Architecture", divider="rainbow")

col_anat1, col_anat2 = st.columns([1, 1])

with col_anat1:
    st.markdown("""
    ### Key Architectural Insights:
    * **1. Distributed vs. Localized Features**:
      In classical NLP, dimension 4 might indicate `is_noun`. In transformer embeddings, a concept like *"royalty"* is represented as a complex linear combination across hundreds of dimensions.
    * **2. The Gaussian Coordinate Distribution**:
      For an L2-normalized vector in D dimensions, coordinates typically satisfy:
      `mean(v) ≈ 0.0`, with standard deviation `std(v) ≈ 1 / sqrt(D)`.
    * **3. Embedding Anisotropy**:
      Unnormalized raw embeddings often exhibit high directional bias: random pairs have positive cosine similarity (> 0.6) because vectors share a common dominant direction.
    * **4. Float Precision & Memory Footprint**:
      * *Float32 (Single Precision)*: 4 bytes/dim -> 1M vectors (1536D) = **6.14 GB** RAM.
      * *Float16 (Half Precision)*: 2 bytes/dim -> 1M vectors (1536D) = **3.07 GB** RAM.
      * *Int8 (Quantized SQ8)*: 1 byte/dim -> 1M vectors (1536D) = **1.54 GB** RAM.
    """)

with col_anat2:
    st.markdown("### Float Precision Memory Hierarchy")
    prec_dot = """
    digraph PrecisionTiers {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Float32 [label="Float32 (Full Precision)\\n4 Bytes per Coordinate\\n1M Vectors (1536D) = 6.14 GB RAM", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        Float16 [label="Float16 / BFloat16\\n2 Bytes per Coordinate\\n1M Vectors (1536D) = 3.07 GB RAM (50% Savings)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Int8 [label="Int8 Scalar Quantization (SQ8)\\n1 Byte per Coordinate\\n1M Vectors (1536D) = 1.54 GB RAM (75% Savings)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        Float32 -> Float16 [label="Cast Weights"];
        Float16 -> Int8 [label="Calibrate SQ8"];
    }
    """
    st.graphviz_chart(prec_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Vector Distribution Profiler in Python", divider="rainbow")
st.markdown("""
The following Python class inspects the statistical properties, coordinate distribution, and quantization memory footprint of embedding vectors.
""")

class ProductionVectorDistributionProfiler:
    """
    Analyzes statistical distributions, coordinate histograms,
    and memory footprints of high-dimensional embedding vectors.
    """
    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def generate_synthetic_embedding(self, domain: str = "technical") -> np.ndarray:
        """Generates realistic embedding coordinates centered near zero."""
        np.random.seed(42 if domain == "technical" else (99 if domain == "financial" else 123))
        # Expected standard deviation for unit vector in D dimensions is 1 / sqrt(D)
        expected_std = 1.0 / np.sqrt(self.dimensions)
        raw_coords = np.random.normal(loc=0.002, scale=expected_std, size=self.dimensions)
        # Apply unit normalization
        unit_vec = raw_coords / np.linalg.norm(raw_coords)
        return unit_vec

    @staticmethod
    def calculate_memory_footprint(num_vectors: int, dimensions: int) -> pd.DataFrame:
        """Calculates RAM requirements across precision formats."""
        records = []
        formats = [
            ("Float32 (Standard)", 4.0, "Full Precision (0% loss)"),
            ("Float16 (Half)", 2.0, "Negligible recall drop (< 0.1%)"),
            ("Int8 (Scalar Quantized)", 1.0, "High throughput (~0.5% drop)"),
            ("Int4 (Extreme Quantized)", 0.5, "Memory constrained (~2% drop)")
        ]
        for name, bytes_per_elem, notes in formats:
            total_bytes = num_vectors * dimensions * bytes_per_elem
            gb_size = total_bytes / (1024 ** 3)
            records.append({
                "Format": name,
                "Bytes / Dim": f"{bytes_per_elem} B",
                "RAM Footprint": f"{gb_size:.2f} GB",
                "Compression Ratio": f"{4.0 / bytes_per_elem:.1f}x",
                "Notes": notes
            })
        return pd.DataFrame(records)

code_snippet = r'''
import numpy as np
import pandas as pd

class VectorProfiler:
    @staticmethod
    def profile_vector(vec: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "l2_norm": float(np.linalg.norm(vec)),
            "zero_fraction": float(np.mean(vec == 0.0))
        }

    @staticmethod
    def quantize_to_int8(vec: np.ndarray) -> np.ndarray:
        # Scale float values to [-128, 127]
        scale = max(abs(vec.min()), abs(vec.max()))
        int8_vec = np.clip(np.round((vec / scale) * 127), -128, 127).astype(np.int8)
        return int8_vec
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Vector Anatomy & Distribution Workbench
st.header("3. Interactive Vector Anatomy & Distribution Workbench", divider="rainbow")
st.markdown("""
Inspect the statistical anatomy of a realistic **768-dimensional embedding vector**. Observe how coordinates cluster around 0.0, and test the **RAM Precision Calculator** for enterprise workloads.
""")

profiler = ProductionVectorDistributionProfiler(dimensions=768)

tab_stat1, tab_stat2 = st.tabs(["1. Coordinate Distribution & Histogram", "2. Vector Precision & Memory Sizing"])

with tab_stat1:
    col_d_sel, col_d_info = st.columns([1, 1])
    with col_d_sel:
        domain_choice = st.selectbox(
            "Select text domain for simulated 768-dim embedding:",
            ["technical", "financial", "conversational"]
        )
    with col_d_info:
        st.info("A 768-dimensional unit vector has an expected coordinate standard deviation of approximately `1 / sqrt(768) ≈ 0.0361`.")

    active_vector = profiler.generate_synthetic_embedding(domain=domain_choice)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Coordinate Mean", f"{np.mean(active_vector):.5f}", help="Expected near 0.0")
    col_m2.metric("Standard Deviation", f"{np.std(active_vector):.5f}", help="Expected ~0.036")
    col_m3.metric("Min Coordinate", f"{np.min(active_vector):.4f}")
    col_m4.metric("Max Coordinate", f"{np.max(active_vector):.4f}")

    st.markdown("##### Coordinate Distribution Histogram (768 Coordinates):")
    counts, bin_edges = np.histogram(active_vector, bins=30, range=(-0.15, 0.15))
    hist_df = pd.DataFrame({
        "Coordinate Range": [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(len(counts))],
        "Coordinate Count": counts
    })
    st.bar_chart(hist_df.set_index("Coordinate Range"), use_container_width=True)

    with st.expander("🔍 View First 20 Raw Vector Coordinates"):
        preview_df = pd.DataFrame([active_vector[:20]], columns=[f"D_{i+1}" for i in range(20)])
        st.dataframe(preview_df.style.format("{:.4f}").background_gradient(cmap="coolwarm", axis=1), use_container_width=True)

with tab_stat2:
    st.markdown("### Enterprise Vector Memory & Precision Sizing")
    st.markdown("Model the physical RAM required to host your vector database working set in cluster memory:")

    col_q_in1, col_q_in2 = st.columns([1, 1])
    with col_q_in1:
        vector_count_sel = st.select_slider(
            "Total Vectors in Knowledge Base:",
            options=[100000, 500000, 1000000, 5000000, 10000000, 50000000],
            value=5000000,
            format_func=lambda x: f"{x:,} Vectors"
        )
    with col_q_in2:
        dim_sel = st.selectbox("Embedding Dimensionality:", [384, 768, 1024, 1536, 3072], index=3)

    sizing_table = profiler.calculate_memory_footprint(vector_count_sel, dim_sel)
    st.dataframe(sizing_table, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Anatomy Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is Embedding Anisotropy and why is it a problem in raw language model representations?**
    * **Answer**:
      * **Anisotropy (The 'Cone Effect')**: In unnormalized transformer representations, embeddings tend to cluster tightly in a narrow cone of high-dimensional space rather than dispersing uniformly in all directions.
      * **Consequence**: Two completely unrelated sentences often yield high baseline cosine similarity (e.g. 0.60 to 0.75), compressing dynamic range and reducing retrieval discrimination.
      * **Solutions**:
        1. **Contrastive Fine-Tuning**: Sentence Transformers explicitly train models using contrastive InfoNCE loss to repel unrelated negative pairs, restoring isotropy.
        2. **Whitening / Mean Subtraction**: Subtracting the global corpus mean vector from all embeddings centers the cloud around the origin.

    **Q2: Can individual dimensions in modern dense embeddings be interpreted by humans?**
    * **Answer**:
      * **No**. Unlike classical features (`is_capitalized`, `sentiment_score`), modern neural embeddings are **fully distributed representations**.
      * Information is encoded in the continuous geometric relationship between vectors, not in individual coordinate axes.
      * Rotating the entire vector space by an orthogonal rotation matrix $R$ preserves 100% of cosine similarities and dot products, even though every single coordinate number changes completely.

    **Q3: Why is Int8 Scalar Quantization (SQ8) so widely adopted in enterprise vector databases?**
    * **Answer**:
      * **75% Memory Reduction**: Downsamples 32-bit floats (4 bytes) to 8-bit signed integers (1 byte). A 100GB vector store shrinks to 25GB, enabling it to fit in single-node RAM without disk swapping.
      * **Negligible Accuracy Impact**: Because cosine similarity evaluates angular trends across hundreds of coordinates, rounding small float decimals degrades top-k retrieval recall by less than 0.5% in practice.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Storing Vectors in Float64 (Double Precision)**:
      * *Issue*: Saving embeddings from PyTorch using default 64-bit float arrays.
      * *Consequence*: Consumes 8 bytes per dimension (2x Float32), doubling RAM costs with zero measurable retrieval benefit.
      * *Fix*: Always cast embeddings to `float32` or `float16` before serialization.

    * **Trap 2: Attempting to Manually Edit a Single Vector Coordinate**:
      * *Issue*: Trying to make a document "more positive" by incrementing coordinate 42 by +1.0.
      * *Consequence*: Completely corrupts the representation; dimensions do not map to human linguistic concepts.
      * *Fix*: Use vector delta arithmetic or re-embed modified text.

    * **Trap 3: Quantizing Vectors with Outlier Coordinates without Clipping**:
      * *Issue*: A single coordinate with extreme value (e.g. +50.0) ruins the quantization scale for all other coordinates.
      * *Consequence*: All normal coordinates (-0.1 to +0.1) round to zero, destroying semantic detail.
      * *Fix*: Apply percentile clipping (e.g. 99.9th percentile) before quantizing.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Distributed Representation**
      * *Definition*: A representational scheme where semantic concepts are distributed across continuous activations of all dimensions rather than single localized features.
    * **Flashcard 2: Embedding Anisotropy**
      * *Definition*: The undesirable geometric tendency for embeddings to cluster within a narrow directional cone, compressing cosine similarity dynamic range.
    * **Flashcard 3: Rotational Invariance**
      * *Definition*: The mathematical property that rotating a vector space preserves all pairwise cosine similarities and distances.
    * **Flashcard 4: Scalar Quantization (SQ8)**
      * *Definition*: Mapping 32-bit floating-point coordinates to 8-bit integers to slash vector database RAM footprint by 75%.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Vector Anatomy:

- [ ] I understand why modern embeddings are distributed representations with uninterpretable individual axes.
- [ ] I know why embedding coordinates follow a Gaussian distribution centered near 0.0.
- [ ] I can explain the anisotropy phenomenon and how contrastive learning restores isotropy.
- [ ] I can calculate the RAM footprint of multi-million vector collections across Float32, Float16, and Int8.
- [ ] I understand why Int8 quantization slashes memory by 75% with negligible recall degradation.
""")
