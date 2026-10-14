import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.3 Embedding Dimensions - Embeddings Curriculum",
    page_icon="📊",
    layout="wide"
)

# Title & Overview
st.title("3.3 Embedding Dimensions: Dimensionality Spectrum (128D to 3072D), Performance Trade-Offs, & Matryoshka Pruning")
st.caption("Chapter 3: Text Embeddings Basics | Domain: 3. Vector Embeddings")
st.markdown("""
A defining parameter of every embedding model is its **vector dimensionality ($D$)**.
State-of-the-art embedding models span a wide spectrum:
* **128D – 256D**: Ultra-compact models engineered for on-device mobile NLP and extreme micro-second latency.
* **384D**: Lightweight standard (`all-MiniLM-L6-v2`), delivering 5x faster CPU search with 80MB model footprints.
* **768D**: Enterprise workhorse (`bge-base-en-v1.5`, `e5-base-v2`), balancing semantic depth and memory.
* **1024D – 1536D**: High-capacity frontier encoders (`text-embedding-3-small`, `bge-large-en-v1.5`, Cohere Embed v3).
* **3072D**: Flagship maximum-precision models (`text-embedding-3-large`).

However, higher dimensionality is **not a free lunch**: doubling dimensions doubles vector database RAM consumption, 
slows HNSW graph construction, and increases network serialization latency.

Furthermore, frontier models now support **Matryoshka Representation Learning (MRL)**—allowing developers to 
**truncate a 1536-dimensional vector down to 256 dimensions, slashing storage by 83% while retaining 98% of retrieval accuracy**.

This lesson explores the dimensionality spectrum, hardware trade-offs, and Matryoshka dimension pruning.
""")

st.divider()

# Section 1: The Dimensionality Spectrum & Matryoshka Architecture
st.header("1. The Dimensionality Spectrum & Matryoshka Representation Learning", divider="rainbow")

col_dim1, col_dim2 = st.columns([1, 1])

with col_dim1:
    st.markdown("""
    ### Engineering Levers & Trade-Offs:
    * **1. Capacity vs. Compute Scaling**:
      * Moving from 128D to 768D yields massive accuracy gains (+18% MTEB benchmark score).
      * Moving from 1536D to 3072D yields diminishing returns (+1.2% score) while doubling RAM costs.
    * **2. The Matryoshka Innovation (MRL)**:
      * Like Russian nesting dolls, models trained with MRL (e.g. OpenAI v3, BGE-M3, Nomic) structure latent features so that the **first K coordinates contain the most critical information**.
      * Slicing `vec[:256]` and re-normalizing produces a valid, high-accuracy unit vector!
    * **3. Two-Stage MRL Retrieval**:
      * *Stage 1*: Fast candidate filter over 100k items using truncated 256D vectors (~1ms).
      * *Stage 2*: Re-rank top 50 candidates using full 1536D vectors.
    """)

with col_dim2:
    st.markdown("### Matryoshka Nesting Architecture")
    mrl_dot = """
    digraph MRLDiagram {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        FullVec [label="Full 1536D Embedding\\n(100% Accuracy / 6.14 GB RAM)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        MRL512 [label="Truncate to 512D\\n(99.2% Accuracy / 2.05 GB RAM)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        MRL256 [label="Truncate to 256D\\n(97.8% Accuracy / 1.02 GB RAM)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        MRL64 [label="Truncate to 64D\\n(91.5% Accuracy / 0.25 GB RAM)", fillcolor="#FEE2E2", color="#DC2626"];

        FullVec -> MRL512 [label="Slice [:512]"];
        MRL512 -> MRL256 [label="Slice [:256]"];
        MRL256 -> MRL64 [label="Slice [:64]"];
    }
    """
    st.graphviz_chart(mrl_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Matryoshka Pruning in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates how to safely slice Matryoshka embeddings and re-normalize them to preserve cosine similarity validity.
""")

class ProductionMatryoshkaEngine:
    """
    Implements Matryoshka Representation Learning dimension truncation
    and L2 unit re-normalization.
    """
    @staticmethod
    def truncate_and_normalize(full_vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Safely slices a Matryoshka vector to target dimension and re-normalizes to unit length.
        CRITICAL: Slicing changes the vector magnitude; L2 re-normalization is mandatory!
        """
        if target_dim > len(full_vector):
            raise ValueError(f"Target dim {target_dim} exceeds vector length {len(full_vector)}")
        
        # 1. Slice first K dimensions
        sliced = full_vector[:target_dim]
        
        # 2. Re-normalize to unit length
        norm = np.linalg.norm(sliced)
        if norm > 0:
            return sliced / norm
        return sliced

    @staticmethod
    def simulate_mteb_retention(target_dim: int, max_dim: int = 1536) -> Dict[str, float]:
        """Simulates MTEB retrieval accuracy retention and RAM compression."""
        ratio = target_dim / max_dim
        # Logarithmic retention curve characteristic of MRL training
        accuracy_retention = 0.85 + (0.15 * np.log2(1 + 7 * ratio) / np.log2(8))
        ram_savings = (1.0 - ratio) * 100.0
        return {
            "retention_pct": round(accuracy_retention * 100.0, 1),
            "ram_savings_pct": round(ram_savings, 1),
            "ram_per_million_gb": round((target_dim * 4.0 * 1000000) / (1024 ** 3), 2)
        }

code_snippet = r'''
import numpy as np

class MatryoshkaPruner:
    @staticmethod
    def prune_embedding(vector: np.ndarray, target_dim: int = 256) -> np.ndarray:
        """
        Truncate Matryoshka-trained embedding and re-normalize.
        Mandatory: Re-normalize so dot product remains valid cosine similarity.
        """
        sliced = vector[:target_dim]
        norm = np.linalg.norm(sliced)
        return sliced / norm if norm > 0 else sliced

# Example with OpenAI text-embedding-3-large
# raw_1536 = client.embeddings.create(input=text, model="text-embedding-3-large", dimensions=256)
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Matryoshka & Dimensionality Workbench
st.header("3. Interactive Matryoshka Pruning & Trade-Off Workbench", divider="rainbow")
st.markdown("""
Experiment with dimension truncation on simulated Matryoshka embeddings. Observe how pruning from **1536D down to 256D or 512D** slashes storage RAM 
while retaining over **97% of retrieval accuracy**.
""")

mrl_engine = ProductionMatryoshkaEngine()

col_bench1, col_bench2 = st.columns([1, 1])

with col_bench1:
    base_model = st.selectbox(
        "1. Select Base Model Family:",
        [
            "OpenAI text-embedding-3-large (Max: 3072D)",
            "OpenAI text-embedding-3-small (Max: 1536D)",
            "BAAI BGE-M3 (Max: 1024D)"
        ]
    )
    max_d = 3072 if "3072" in base_model else (1536 if "1536" in base_model else 1024)

with col_bench2:
    target_d = st.select_slider(
        "2. Choose Target Truncated Dimensions:",
        options=[64, 128, 256, 512, 768, 1024, 1536, 3072] if max_d == 3072 else (
            [64, 128, 256, 512, 768, 1024, 1536] if max_d == 1536 else [64, 128, 256, 512, 1024]
        ),
        value=256
    )

stats = mrl_engine.simulate_mteb_retention(target_d, max_dim=max_d)

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Target Dimensions", f"{target_d}D")
col_kpi2.metric("Accuracy Retained", f"{stats['retention_pct']}%", delta=f"{stats['retention_pct'] - 100:.1f}%")
col_kpi3.metric("RAM Storage / 1M Vectors", f"{stats['ram_per_million_gb']} GB", delta=f"-{stats['ram_savings_pct']}%", delta_color="normal")
col_kpi4.metric("Throughput Speedup", f"{max_d / target_d:.1f}x Faster Dot Products")

# Spectrum Comparison Table
st.markdown("### Production Dimensionality Spectrum Comparison Matrix")
spectrum_df = pd.DataFrame([
    {"Tier": "Ultra-Compact", "Dimensions": "128D - 256D", "Example Models": "MobileBERT, All-MiniLM (Pruned)", "RAM / 1M (FP32)": "0.51 GB - 1.02 GB", "Best Fit": "Mobile apps, smart devices, sub-1ms search"},
    {"Tier": "Lightweight", "Dimensions": "384D", "Example Models": "all-MiniLM-L6-v2, BGE-small", "RAM / 1M (FP32)": "1.54 GB", "Best Fit": "CPU-only servers, high QPS chatbots"},
    {"Tier": "Standard", "Dimensions": "768D", "Example Models": "BGE-base-en, E5-base-v2, BERT", "RAM / 1M (FP32)": "3.07 GB", "Best Fit": "Enterprise documentation, standard RAG"},
    {"Tier": "High-Capacity", "Dimensions": "1024D - 1536D", "Example Models": "text-embedding-3-small, BGE-large", "RAM / 1M (FP32)": "4.10 GB - 6.14 GB", "Best Fit": "Large-scale semantic search, legal, compliance"},
    {"Tier": "Frontier Flagship", "Dimensions": "3072D", "Example Models": "text-embedding-3-large", "RAM / 1M (FP32)": "12.29 GB", "Best Fit": "Complex cross-lingual nuance, research QA"}
])
st.dataframe(spectrum_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Dimensionality Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is Matryoshka Representation Learning (MRL) and how does it train models?**
    * **Answer**:
      * In standard embedding training, contrastive loss is computed only on the final full-dimensional vector v in D-dimensional space.
      * **MRL Loss**: Computes contrastive loss simultaneously on nested subsets of dimensions: [v[:64], v[:128], v[:256], v[:512], v[:D]].
      * The loss forces the model to pack the most critical, broad semantic signals into the earliest dimensions (0 to 64), while reserving later dimensions for fine-grained, subtle distinctions.
      * Developers can slice the vector at any of the trained nested cutoffs without retraining.

    **Q2: Why is L2 re-normalization mandatory after slicing a Matryoshka embedding?**
    * **Answer**:
      * A full 1536D vector has unit length: sum(v_i^2) = 1.0.
      * When you slice the first 256 coordinates, the sum of squares is strictly less than 1.0: sum(v_i^2) < 1.0.
      * If you do not re-normalize, the vector magnitude is truncated and dot products will no longer reflect true cosine similarity.
      * You must divide by the sliced norm: `v_256 = v[:256] / norm(v[:256])`.

    **Q3: How does Two-Stage MRL Retrieval optimize latency and cost in large-scale search?**
    * **Answer**:
      * **Stage 1 (Coarse Candidate Filtering)**:
        Store 256D versions of all 10M documents in RAM (slashes memory from 61GB to 10GB).
        Perform fast ANN search with the 256D query vector to retrieve the top 100 candidates in under 3ms.
      * **Stage 2 (Fine Re-Scoring)**:
        Fetch the full 1536D vectors for only those 100 candidates from disk or SSD cache and compute exact dot products to output the final top 10.
      * Yields the accuracy of 1536D with the memory and latency of 256D.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Arbitrarily Slicing Non-Matryoshka Embeddings**:
      * *Issue*: Taking the first 256 dimensions of a standard model like `all-MiniLM-L6-v2`.
      * *Consequence*: Retrieval accuracy collapses catastrophically; standard models distribute features uniformly across all coordinates without nesting.
      * *Fix*: Only truncate models explicitly trained with MRL (OpenAI v3, BGE-M3, Nomic).

    * **Trap 2: Forgetting to Re-Normalize after Slicing**:
      * *Issue*: Slicing `vec[:512]` and directly querying a vector database configured for Dot Product.
      * *Consequence*: The vectors have non-unit norms, producing corrupted nearest-neighbor rankings.
      * *Fix*: Always divide by `np.linalg.norm()` after slicing.

    * **Trap 3: Over-Allocating 3072 Dimensions for Simple FAQ Applications**:
      * *Issue*: Using `text-embedding-3-large` at full 3072D for a 5,000-page internal company HR portal.
      * *Consequence*: Wasting 12GB of RAM and increasing query latency with zero observable benefit over a 384D MiniLM model.
      * *Fix*: Benchmark small 384D models before jumping to 3072D frontier models.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Matryoshka Representation Learning (MRL)**
      * *Definition*: A training technique that optimizes embeddings across multiple nested dimensional subsets, allowing vectors to be sliced to lower dimensions without retraining.
    * **Flashcard 2: Two-Stage MRL Retrieval**
      * *Definition*: Filtering millions of candidates using ultra-compact sliced vectors (e.g. 256D) before re-scoring the top candidates with full-dimensional vectors.
    * **Flashcard 3: Dimensionality vs. RAM Cost**
      * *Definition*: Vector memory consumption scales linearly with dimension (D * 4 bytes per vector in Float32).
    * **Flashcard 4: MRL Re-Normalization**
      * *Definition*: The mandatory step of re-scaling a sliced vector to unit length (L2 norm = 1.0) before computing similarity.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embedding Dimensions:

- [ ] I can describe the dimensionality spectrum from 128D mobile embeddings to 3072D frontier models.
- [ ] I understand how Matryoshka Representation Learning (MRL) enables dynamic dimension pruning.
- [ ] I know why L2 re-normalization is strictly mandatory after slicing an MRL vector.
- [ ] I can design a Two-Stage MRL retrieval pipeline (coarse 256D search + fine 1536D re-scoring).
- [ ] I have completed Chapter 3: Text Embeddings Basics!
""")
