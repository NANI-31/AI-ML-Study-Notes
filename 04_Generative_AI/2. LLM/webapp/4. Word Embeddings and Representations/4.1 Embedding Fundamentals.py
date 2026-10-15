import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="4.1 Embedding Fundamentals - LLM Curriculum",
    page_icon="🌌",
    layout="wide"
)

# ==============================================================================
# Title Block and Executive Overview
# ==============================================================================
st.title("4.1 Embedding Fundamentals: Vector Spaces, Dense Geometry, and Distance Metrics")
st.caption("Chapter 4: Word Embeddings and Representations | Domain: 2. Large Language Models")

st.markdown("""
An embedding is a **continuous, dense mathematical representation** of discrete symbols—such as words, sentences, code, 
or images—projected into a high-dimensional geometric vector space. In this latent space, geometric proximity directly 
reflects semantic relationship: concepts sharing linguistic contexts or ontological meaning are mapped to coordinates 
that cluster tightly together, enabling linear algebra operations on human concepts.

While classical NLP treated vocabulary words as isolated, orthogonal discrete symbols (where the distance between `"doctor"` 
and `"nurse"` was mathematically identical to the distance between `"doctor"` and `"banana"`), modern embeddings create 
smooth, continuous manifolds where semantic attributes, analogies, and syntactic hierarchies naturally emerge.

In this lesson, you will master:
1. The 7 core foundational principles of dense geometric embeddings and vector space representations.
2. The Curse of Orthogonality: why sparse representations collapse semantic reasoning and waste GPU memory.
3. The geometric and computational trade-offs of embedding dimensionality (from 384 to 4,096 dimensions).
4. Mathematical formulations of primary distance metrics: Cosine Similarity, Euclidean (L2) Distance, and Dot Product.
5. A runnable pure Python implementation of an embedding calculation engine with vector normalization and metric comparison.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Geometric Manifold of Continuous Representations
# ==============================================================================
st.header("1. Architectural Blueprint: The Geometric Manifold of Continuous Representations", divider="rainbow")

st.markdown("""
Observe how discrete categorical token IDs are projected into continuous high-dimensional vector spaces, 
forming semantic clusters and relational sub-manifolds:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    manifold_dot = """
    digraph EmbeddingManifold {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_discrete {
            label="Discrete Categorical Domain";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            Token_Doctor [label="Token: 'doctor'\\nID: 8492\\n(Orthogonal Scalar)", fillcolor="#1E293B"];
            Token_Nurse [label="Token: 'nurse'\\nID: 19402\\n(Orthogonal Scalar)", fillcolor="#1E293B"];
            Token_Apple [label="Token: 'apple'\\nID: 2841\\n(Orthogonal Scalar)", fillcolor="#1E293B"];
        }

        subgraph cluster_projection {
            label="Continuous Embedding Matrix W_emb";
            style=filled;
            color="#1E293B";
            fontcolor="#818CF8";
            LookupTable [label="Embedding Table W_emb\\nShape: [Vocab_Size, Dimension]\\nRetrieval: W_emb[Token_ID]", fillcolor="#0F172A", color="#818CF8", shape=cylinder];
        }

        subgraph cluster_latent {
            label="High-Dimensional Latent Manifold (R^D)";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            MedicalCluster [label="Medical Profession Cluster\\nVec('doctor') ~ Vec('nurse')\\nCosine Similarity: 0.89", fillcolor="#064E3B", color="#34D399"];
            FruitCluster [label="Botany / Food Cluster\\nVec('apple') ~ Vec('orange')\\nCosine Similarity: 0.84", fillcolor="#065F46", color="#34D399"];
            DistanceEdge [label="Cross-Cluster Distance\\nDoctor vs Apple\\nCosine Similarity: 0.08 (Divergent)", fillcolor="#78350F", color="#F59E0B"];
        }

        Token_Doctor -> LookupTable;
        Token_Nurse -> LookupTable;
        Token_Apple -> LookupTable;

        LookupTable -> MedicalCluster;
        LookupTable -> FruitCluster;
        MedicalCluster -> DistanceEdge [style=dotted, color="#EF4444"];
        FruitCluster -> DistanceEdge [style=dotted, color="#EF4444"];
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(manifold_dot, use_container_width=True)

with col_diag2:
    st.markdown("#### The 4 Foundational Postulates of Embeddings")
    st.markdown("""
    * **Distributional Hypothesis**: Words that appear in similar linguistic contexts have similar meanings (Harris, 1954; Firth, 1957).
    * **Geometric Encoding**: Syntactic properties, grammatical roles, and factual associations correspond to linear directions in vector space.
    * **Dimensional Density**: Compressing language into 384 to 4,096 dimensions forces the model to discover shared abstract semantic latent factors.
    * **Invertible Distance**: Cosine similarity between normalized vectors bounds semantic proximity from -1.0 (antithetical) to +1.0 (synonymous).
    """)

# Data Structure Comparison Matrix
st.markdown("#### Data Structure Architecture: Sparse Representations vs. Dense Embeddings")

rep_table_dot = """
digraph RepTable {
    bgcolor="#0B1120";
    rankdir=TB;

    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=plaintext];

    TableNode [label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#0F172A" colspan="5"><font color="#38BDF8"><b>Representation Paradigm Comparison</b></font></td>
            </tr>
            <tr bgcolor="#334155">
                <td><b>Dimension Property</b></td>
                <td><b>One-Hot Encoding</b></td>
                <td><b>Bag-of-Words / TF-IDF</b></td>
                <td><b>Static Word2Vec / GloVe</b></td>
                <td><b>Modern Foundation Embedding</b></td>
            </tr>
            <tr>
                <td bgcolor="#0C4A6E"><b>Dimensionality</b></td>
                <td>Vocab Size (100k+)</td>
                <td>Vocab Size (100k+)</td>
                <td>Fixed (300 dims)</td>
                <td>Dense (768 to 4,096 dims)</td>
            </tr>
            <tr>
                <td bgcolor="#1E1B4B"><b>Sparsity</b></td>
                <td>99.999% Zeroes</td>
                <td>99.95% Zeroes</td>
                <td>0.00% (100% Dense)</td>
                <td>0.00% (100% Dense)</td>
            </tr>
            <tr>
                <td bgcolor="#78350F"><b>Semantic Distance</b></td>
                <td>Orthogonal (Dot product = 0)</td>
                <td>Coarse word overlap only</td>
                <td>Cosine reflects static similarity</td>
                <td>Deep contextual manifold geometry</td>
            </tr>
            <tr>
                <td bgcolor="#064E3B"><b>Memory Footprint</b></td>
                <td>Intractable without sparse CSR</td>
                <td>High (Large sparse matrices)</td>
                <td>Moderate (~1 GB table)</td>
                <td>Model weight tensor in GPU VRAM</td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(rep_table_dot, use_container_width=True)

st.divider()

# ==============================================================================
# Section 2: 1. What is an Embedding?
# ==============================================================================
st.header("2. What is an Embedding? From Discrete Tokens to Latent Coordinates", divider="rainbow")

st.markdown("""
In computational linguistics and linear algebra, an embedding is a **parameterized mapping function** from a discrete 
set of objects (words, subwords, users, documents) to a continuous coordinate vector space:

`E: Vocabulary -> R^D`

Where V is the discrete vocabulary size and D is the embedding dimension.
""")

col_emb1, col_emb2 = st.columns(2)

with col_emb1:
    st.markdown("#### The Discrete Representation Dilemma")
    st.markdown("""
    Computers naturally process text as discrete Unicode character strings:
    * `"king"`, `"queen"`, `"apple"`, `"orange"`.
    * If we assign arbitrary integer IDs: `king = 1`, `queen = 2`, `apple = 3`:
      * Does `queen - king == apple - queen`? Obviously not!
      * Scalar integer IDs introduce an artificial, arbitrary linear ordering that poisons neural gradient descent.
    * If we use one-hot vectors of length V:
      `e_king = [1, 0, 0, ...]`, `e_queen = [0, 1, 0, ...]`
      * Every vector is strictly **orthogonal** (e_king . e_queen = 0).
      * The model cannot infer that a queen is related to a king.
    """)

with col_emb2:
    st.markdown("#### Continuous Latent Projections")
    st.markdown("""
    An embedding layer replaces scalar IDs with a learned weight matrix:
    `W_emb in R^{[V, D]}`
    
    * **Indexing as Matrix Multiplication**:
      Retrieving the embedding for token index k is mathematically equivalent to multiplying a one-hot vector e_k by the weight matrix W_{\text{emb}}:
      `v_k = e_k * W_emb = W_emb[k, :]`
    * **Continuous Optimization**:
      Because every dimension is a continuous floating-point number, the coordinates can be tuned via **gradient descent backpropagation**:
      `W_emb <- W_emb - lr * (dL / dW_emb)`
      The network shifts the geometric coordinates of words based on prediction error, naturally learning relational clusters.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
In modern Transformer decoders (Llama 3, GPT-4), the embedding table is often **tied** to the final output projection head (`lm_head = W_emb^T`). 
This weight-tying reduces total model parameters by tens of millions while enforcing a dual mathematical constraint: 
the embedding vector used to ingest a word must equal the logit vector used to generate that word.
""")

emb_prop_df = pd.DataFrame({
    "Property": [
        "Mathematical Nature",
        "Trainability",
        "Relational Topology",
        "Analogy Arithmetic",
        "Hardware Computation"
    ],
    "Discrete Token ID (Scalar)": [
        "Integer index in [0, V-1]",
        "Static index (Non-differentiable)",
        "Zero topological meaning",
        "Impossible",
        "Index pointer / hash key"
    ],
    "Continuous Embedding (Tensor)": [
        "Dense float vector in R^D",
        "Fully differentiable via backpropagation",
        "Clustered by semantic similarity",
        "v(king) - v(man) + v(woman) ~ v(queen)",
        "Dense BLAS matrix multiplication (GEMM)"
    ]
})
st.dataframe(emb_prop_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Dense Vectors
# ==============================================================================
st.header("3. Dense Vectors: Information Concentration vs. Orthogonal Sparsity", divider="rainbow")

st.markdown("""
The transition from sparse discrete matrices to dense continuous vectors is one of the most critical breakthroughs 
in deep learning history. It transformed sequence processing from memory-bound sparse search to compute-bound 
dense linear algebra.
""")

col_den1, col_den2 = st.columns(2)

with col_den1:
    st.markdown("#### The Curse of Sparsity")
    st.markdown("""
    In sparse bag-of-words or one-hot matrices:
    * **Dimensionality**: Set equal to vocabulary size (V >= 100,000).
    * **Zero Dominance**: 99.99% of all matrix entries are literal `0.0`.
    * **Memory Inefficiency**: Storing sparse matrices requires complex Compressed Sparse Row (CSR) data structures that thrash CPU memory caches.
    * **Orthogonality**: In high-dimensional sparse spaces, almost all vectors are mutually perpendicular. There is zero geometric overlap between synonymous words that do not share exact spelling.
    """)

with col_den2:
    st.markdown("#### Dense Information Concentration")
    st.markdown("""
    In dense embedding spaces:
    * **Dimensionality**: Fixed low-to-medium rank (D in [256, 4096]).
    * **100% Value Density**: Every single coordinate is an active 16-bit or 32-bit floating-point number.
    * **Distributed Representation**: Semantic concepts are not isolated to a single coordinate; they are **distributed across all D dimensions** simultaneously.
    * **Hardware Alignment**: Modern GPUs (NVIDIA Tensor Cores) are specifically optimized for dense General Matrix Multiplies (GEMMs). Dense vectors execute at maximum hardware throughput.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern frontier models maintain total density throughout the network. In models like Llama 3-70B, every single token 
is represented by a dense continuous vector of **8,192 floating-point values**, passed continuously through 80 layers 
of Transformer self-attention.
""")

dense_comp_df = pd.DataFrame({
    "Criterion": ["Vector Dimensionality", "Non-Zero Percentage", "Semantic Closeness", "GPU Hardware Fit", "Storage Format"],
    "Sparse Vectors (One-Hot / TF-IDF)": ["100,000 to 1,000,000+", "< 0.01% non-zero", "Dot product is 0 for non-identical words", "Extremely poor (Memory bandwidth bound)", "Sparse CSR / Scipy matrix"],
    "Dense Embeddings (Transformers)": ["256 to 8,192", "100.0% non-zero", "Dot product reflects latent semantic affinity", "Optimal (Dense GPU Tensor Cores)", "Contiguous FP16 / BF16 tensor"]
})
st.dataframe(dense_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Embedding Dimensions
# ==============================================================================
st.header("4. Embedding Dimensions: Precision, Capacity, and Memory Trade-Offs", divider="rainbow")

st.markdown("""
Choosing the hidden embedding dimension (D) is a fundamental architectural hyperparameter. 
It establishes the **information capacity** of the model's latent representation.
""")

col_dim1, col_dim2 = st.columns(2)

with col_dim1:
    st.markdown("#### The Dimensionality Capacity Curve")
    st.markdown("""
    * **Small Dimensions (D = 128 to 384, e.g. MiniLM)**:
      * Fast inference, tiny memory footprint (ideal for edge devices, mobile phones, and ultra-fast vector search).
      * *Limitation*: Suffers from **capacity crowding**: representing millions of world facts in 384 dimensions forces unrelated concepts into overlapping subspace angles.
    * **Medium Dimensions (D = 768 to 1536, e.g. BERT-base, OpenAI text-embedding-3-small)**:
      * The industry standard for enterprise search, RAG retrieval, and classification.
      * Sufficient capacity to separate nuanced semantic domains without excessive GPU memory overhead.
    * **Frontier Foundation Dimensions (D = 4,096 to 8,192, e.g. Llama 3 8B/70B)**:
      * Massive expressivity: captures intricate syntactic dependency trees, mathematical logic, code syntax, and multi-lingual idioms simultaneously.
    """)

with col_dim2:
    st.markdown("#### Memory and Compute Cost Scaling")
    st.markdown("""
    The embedding dimension directly dictates GPU parameter consumption:
    
    1. **Embedding Matrix Memory**:
       `Params = Vocab_Size * Hidden_Dimension`
       * For Llama 3 (V = 128,256, D = 4,096):
         `128,256 * 4,096 * 2 bytes (FP16) = 1.05 GB VRAM`
    2. **Attention Key-Value Memory**:
       Each token's KV cache tensor scales linearly with hidden dimension D. Expanding D increases KV cache VRAM per user session.
    3. **Vector Database Index Size**:
       Storing 10 million document embeddings in a vector database:
       * At D = 384: `10,000,000 * 384 * 4 bytes = 15.36 GB`
       * At D = 1536: `10,000,000 * 1536 * 4 bytes = 61.44 GB`
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern embedding models (such as OpenAI `text-embedding-3-large` and Google Gecko) support **Matryoshka Representation Learning (MRL)**. 
MRL trains the model so that the most critical semantic information is packed into the earliest dimensions. 
Developers can truncate a 3,072-dimensional vector down to **256 or 512 dimensions** with negligible accuracy loss, 
slashing vector database RAM costs by up to 80%.
""")

dim_comp_df = pd.DataFrame({
    "Dimension (D)": [128, 384, 768, 1536, 4096, 8192],
    "Representative Model": ["MobileBERT", "all-MiniLM-L6-v2", "BERT-base / BGE-base", "OpenAI text-embedding-3-small", "Llama 3 8B", "Llama 3 70B"],
    "Primary Use Case": ["Edge microcontrollers", "Low-latency RAG search", "Text classification", "Production enterprise RAG", "Generative Foundation LLM", "Frontier Generative LLM"],
    "Vector DB Size (1M Docs in FP32)": ["512 MB", "1.53 GB", "3.07 GB", "6.14 GB", "16.38 GB", "32.76 GB"],
    "Semantic Expressivity": ["Low", "Moderate", "High", "Very High", "Exceptional", "Superhuman"]
})
st.dataframe(dim_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Semantic Representation
# ==============================================================================
st.header("5. Semantic Representation: Distributional Semantics and Relational Geometry", divider="rainbow")

st.markdown("""
How does pure mathematical linear algebra encode human semantic meaning? 
The foundation of all modern embedding spaces is John Rupert Firth's famous 1957 linguistic axiom: 
**"You shall know a word by the company it keeps."**
""")

col_sem1, col_sem2 = st.columns(2)

with col_sem1:
    st.markdown("#### The Distributional Hypothesis in Action")
    st.markdown("""
    Consider three target words: `"ophthalmologist"`, `"cardiologist"`, and `"tractor"`.
    * Sentences in training text:
      * *"The **ophthalmologist** performed delicate cataract surgery."*
      * *"The **cardiologist** performed delicate arterial surgery."*
      * *"The **tractor** plowed the muddy corn field."*
    * Notice that `"ophthalmologist"` and `"cardiologist"` share nearly identical surrounding context words: 
      `["performed", "delicate", "surgery", "patient", "hospital", "clinic"]`.
    * When a neural network is trained to predict surrounding words from target tokens, the weights for 
      `"ophthalmologist"` and `"cardiologist"` are updated using nearly identical gradient signals.
    * Result: Their coordinate vectors naturally converge to almost identical positions in vector space!
    """)

with col_sem2:
    st.markdown("#### Vector Analogies and Subspace Arithmetic")
    st.markdown("""
    Mikolov et al. (2013) demonstrated that semantic relationships emerge as **linear offset vectors**:
    
    `Vector("king") - Vector("man") + Vector("woman") approx Vector("queen")`
    
    * The vector difference `Vector("king") - Vector("man")` isolates the abstract directional concept of **"Royalty"**.
    * Adding `Vector("woman")` applies the royalty direction to the female gender vector, landing squarely at the coordinate for `"queen"`.
    * Other emergent linear directions in latent space include:
      * **Capital Cities**: `Vector("Paris") - Vector("France") + Vector("Japan") approx Vector("Tokyo")`
      * **Verb Tenses**: `Vector("walked") - Vector("walking") + Vector("swimming") approx Vector("swam")`
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
While classical static embeddings demonstrated simple word analogies, modern foundation models exhibit 
**multi-dimensional concept decomposition**. In Transformer residual streams, attention heads track high-order semantic 
features—such as truthfulness, corporate hierarchy, sentiment polarity, and political affiliation—as linear directions 
identifiable via linear probes (Anthropic Representation Engineering).
""")

sem_analogy_df = pd.DataFrame({
    "Analogy Relation": ["Gender / Royalty", "Capital / Country", "Comparative / Superlative", "Currency / Nation", "Company / Product"],
    "Vector Equation": [
        "v(king) - v(man) + v(woman)",
        "v(Paris) - v(France) + v(Germany)",
        "v(bigger) - v(big) + v(small)",
        "v(Yen) - v(Japan) + v(UK)",
        "v(iPhone) - v(Apple) + v(Google)"
    ],
    "Expected Target Vector": ["v(queen)", "v(Berlin)", "v(smaller)", "v(Pound)", "v(Pixel)"],
    "Latent Dimension Encoding": ["Gender direction vector", "Sovereign capital direction", "Adjective grade direction", "Monetary jurisdiction direction", "Flagship consumer hardware direction"]
})
st.dataframe(sem_analogy_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Similarity Between Embeddings
# ==============================================================================
st.header("6. Similarity Between Embeddings: Linear Algebra Proximity and Directional Alignment", divider="rainbow")

st.markdown("""
Once tokens or documents are projected into continuous vector space, quantifying how similar two concepts are 
reduces to evaluating **linear algebra proximity**.
""")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.markdown("#### Magnitude vs. Angular Direction")
    st.markdown("""
    In high-dimensional spaces, a vector u possesses two distinct mathematical properties:
    1. **Magnitude (Length / Norm)**:
       `||u||_2 = sqrt(Sum_i u_i^2)`
       * Reflects word frequency, information density, or token confidence. Frequently occurring generic words often accumulate larger vector norms.
    2. **Angular Direction**:
       The direction in which the arrow points in high-dimensional space.
       * Reflects the **pure semantic topical identity** of the concept.
    * **The Golden Rule**: When evaluating semantic similarity, **angular direction is far more important than length**. Two articles about quantum physics should be considered highly similar even if Article A is 500 words long and Article B is 5,000 words long!
    """)

with col_sim2:
    st.markdown("#### Unit Normalization (L2 Normalization)")
    st.markdown("""
    To eliminate magnitude distortion, embeddings are almost universally **L2-Normalized** before storage:
    
    `u_norm = u / ||u||_2`
    
    * After L2 normalization, every vector lies directly on the surface of a unit hypersphere (||u_{\text{norm}}||_2 = 1.0).
    * **The Computational Miracle of Unit Normalization**:
      When vectors are unit-normalized, **Cosine Similarity simplifies to a raw Dot Product**:
      `Cosine(u_norm, v_norm) = u_norm . v_norm`
      Vector databases can execute fast inner-product matrix multiplications directly without computing costly square roots at query time!
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
All modern embedding retrieval backends (OpenAI embeddings, Cohere embed-v3, Hugging Face TEI) output strictly 
L2-normalized vectors by default. This enables instant Maximum Inner Product Search (MIPS) using fast GPU BLAS libraries.
""")

norm_comp_df = pd.DataFrame({
    "Vector State": ["Raw Embedding Vector u", "L2-Normalized Vector u_norm"],
    "Norm Length ||u||": ["Variable (e.g. 5.42 or 28.9)", "Strictly 1.0000"],
    "Spatial Position": ["Arbitrary depth in R^D", "Surface of unit hypersphere S^{D-1}"],
    "Cosine Similarity Formula": ["(u . v) / (||u|| * ||v||)", "u_norm . v_norm (Fast dot product)"],
    "Hardware Query Efficiency": ["Slow (Requires norm division)", "Maximum speed (Raw BLAS matrix dot product)"]
})
st.dataframe(norm_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Vector Spaces
# ==============================================================================
st.header("7. Vector Spaces: Continuous Manifolds, Clusters, and High-Dimensional Geometry", divider="rainbow")

st.markdown("""
A vector space is the continuous geometric arena within which all embeddings reside. 
However, high-dimensional vector spaces (D >= 512) behave in ways that strongly defy everyday 3D human intuition.
""")

col_vs1, col_vs2 = st.columns(2)

with col_vs1:
    st.markdown("#### Counter-Intuitive Properties of High-D Space")
    st.markdown("""
    1. **The Volume Concentration at the Surface**:
       In a 1,536-dimensional hypersphere, virtually **100% of the sphere's volume is concentrated in a razor-thin shell** near the outer surface. Almost zero volume exists in the center!
    2. **Near-Universal Orthogonality**:
       If you sample two completely random vectors in 1,536 dimensions, the angle between them is almost guaranteed to be **90 degrees** (Cosine \approx 0.0). Unrelated concepts naturally separate with zero risk of accidental overlap.
    3. **The Hubness Problem**:
       In high dimensions, certain points (called "hubs") become nearest neighbors to thousands of unrelated queries. High-performance retrieval models apply centering and whitening to suppress artificial hubs.
    """)

with col_vs2:
    st.markdown("#### Semantic Subspaces and Clustering")
    st.markdown("""
    Within the global embedding space, concepts organize into distinct geometric subspaces:
    * **Topical Manifolds**: Legal terminology clusters into one subspace; Python programming syntax into another; conversational small talk into a third.
    * **Clustering Algorithms (K-Means, HDBSCAN)**:
      Vector databases group embeddings into Voronoi cells (IVF - Inverted File Indexing). During search, the engine queries only the closest cluster centroid, skipping 95% of database vectors!
    * **Dimensionality Reduction (t-SNE / UMAP)**:
      Projects high-dimensional manifolds down to 2D/3D for human visualization while preserving local neighborhood clusters.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Vector databases like Pinecone, Qdrant, Milvus, and pgvector construct **Hierarchical Navigable Small World (HNSW)** graphs 
across these high-dimensional vector spaces, performing approximate nearest neighbor searches across 100 million documents 
in under 10 milliseconds.
""")

vs_prop_df = pd.DataFrame({
    "Geometric Concept": [
        "Unit Hypersphere",
        "HNSW Proximity Graph",
        "IVF Voronoi Clusters",
        "UMAP Dimensionality Reduction"
    ],
    "Role in AI Systems": [
        "The continuous surface where all normalized embeddings live",
        "Multi-layer graph enabling sub-10ms nearest neighbor search",
        "Partitions vector space into clustered cells for inverted indexing",
        "Non-linear projection compressing high-D vectors to 2D for visualization"
    ],
    "Key Parameter": [
        "Dimension D (e.g. 1536)",
        "M (connections) and efSearch (search beam)",
        "nlist (centroids) and nprobe (clusters queried)",
        "n_neighbors and min_dist"
    ]
})
st.dataframe(vs_prop_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Distance Metrics
# ==============================================================================
st.header("8. Distance Metrics: Cosine Similarity, Euclidean (L2) Distance, and Dot Product", divider="rainbow")

st.markdown("""
Choosing the correct distance metric is vital when configuring vector search, clustering, or fine-tuning loss functions. 
There are three fundamental mathematical metrics:
""")

col_met1, col_met2, col_met3 = st.columns(3)

with col_met1:
    st.markdown("#### 1. Cosine Similarity")
    st.markdown("""
    Measures the **cosine of the angle** between vectors:
    
    `Cosine(u, v) = (u . v) / (||u|| * ||v||)`
    
    * **Range**: `-1.0` (opposite) to `+1.0` (identical).
    * **Invariance**: Completely invariant to vector length!
    * **Use Case**: Best for text embeddings where document length varies significantly.
    """)

with col_met2:
    st.markdown("#### 2. Euclidean Distance (L2)")
    st.markdown("""
    Measures the **straight-line distance** between vector coordinates:
    
    `L2(u, v) = sqrt(Sum_i (u_i - v_i)^2)`
    
    * **Range**: `0.0` (identical) to `+Infinity`.
    * **Sensitivity**: Highly sensitive to vector magnitude!
    * **Use Case**: Clustering algorithms (K-Means), image feature spaces, un-normalized embeddings.
    """)

with col_met3:
    st.markdown("#### 3. Dot Product (Inner Product)")
    st.markdown("""
    Measures the **unnormalized scalar projection**:
    
    `Dot(u, v) = Sum_i (u_i * v_i)`
    
    * **Range**: `-Infinity` to `+Infinity`.
    * **Equivalence**: If vectors are L2-normalized, **Dot Product equals Cosine Similarity**!
    * **Use Case**: Maximum Inner Product Search (MIPS) on GPU accelerators.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
When vectors are L2-normalized, there is an exact mathematical relationship between Euclidean Distance and Cosine Similarity:

`L2_Distance(u_norm, v_norm)^2 = 2 - 2 * Cosine_Similarity(u_norm, v_norm)`

Because the ranking order of documents is **strictly identical** across all three metrics for normalized vectors, 
modern vector databases default exclusively to **Dot Product** for maximum hardware throughput!
""")

metric_summary_df = pd.DataFrame({
    "Distance Metric": ["Cosine Similarity", "Euclidean Distance (L2)", "Dot Product (Inner Product)"],
    "Mathematical Formula": ["(u . v) / (||u|| * ||v||)", "sqrt(sum((u_i - v_i)^2))", "sum(u_i * v_i)"],
    "Metric Range": ["[-1.0, 1.0]", "[0.0, +inf)", "(-inf, +inf)"],
    "Magnitude Sensitivity": ["Completely Invariant", "Highly Sensitive", "Highly Sensitive"],
    "Behavior on L2-Normalized Vectors": ["Identical ranking", "L2^2 = 2 - 2*Cosine", "Identical to Cosine"],
    "Fastest Hardware Evaluation": ["Moderate (norm divisions)", "Slowest (square root)", "Blazing Fast (Raw GEMM dot product)"]
})
st.dataframe(metric_summary_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: Production Implementation: Pure Python Embedding Engine
# ==============================================================================
st.header("9. Production Implementation: Pure Python Embedding Engine & Distance Metrics", divider="rainbow")

st.markdown("""
The self-contained Python implementation below demonstrates the core mathematical operations on embeddings: 
L2-normalization, Cosine Similarity, Euclidean Distance, Dot Product, and vector analogy arithmetic using pure NumPy:
""")

st.code('''import numpy as np
from typing import Dict, List, Tuple, Any

class VectorEmbeddingEngine:
    """
    Demonstrates the linear algebra fundamentals of high-dimensional embeddings:
    - L2 Unit Normalization
    - Cosine Similarity, Euclidean Distance, and Dot Product
    - Vector Analogy Arithmetic (king - man + woman = queen)
    - Nearest-Neighbor Search
    """

    def __init__(self, dimension: int = 128):
        self.d = dimension
        self.vocab: Dict[str, np.ndarray] = {}
        np.random.seed(42)

    def add_vector(self, word: str, vector: np.ndarray, normalize: bool = True) -> None:
        """Adds a named vector to the internal space, optionally L2-normalizing it."""
        if normalize:
            norm = np.linalg.norm(vector)
            vector = vector / (norm if norm > 0 else 1.0)
        self.vocab[word] = vector

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Computes Cosine Similarity between vectors u and v."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Computes Euclidean (L2) Distance between vectors u and v."""
        return float(np.linalg.norm(u - v))

    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> float:
        """Computes raw Inner Product between vectors u and v."""
        return float(np.dot(u, v))

    def find_nearest_neighbors(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Finds top-k nearest neighbors in vector space using Cosine Similarity."""
        scores = []
        for word, vec in self.vocab.items():
            sim = self.cosine_similarity(query_vec, vec)
            scores.append((word, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def solve_analogy(self, word_a: str, word_b: str, word_c: str) -> List[Tuple[str, float]]:
        """
        Solves analogy: word_a is to word_b as word_c is to ???
        Formula: target = v(word_b) - v(word_a) + v(word_c)
        """
        va = self.vocab[word_a]
        vb = self.vocab[word_b]
        vc = self.vocab[word_c]
        
        target_vector = vb - va + vc
        # Find neighbors, excluding input words
        neighbors = self.find_nearest_neighbors(target_vector, top_k=5)
        filtered = [n for n in neighbors if n[0] not in [word_a, word_b, word_c]]
        return filtered

# Verification Demonstration
if __name__ == "__main__":
    engine = VectorEmbeddingEngine(dimension=64)
    
    # 1. Synthesize structured latent vectors with deliberate directional alignment
    base_royalty = np.random.randn(64)
    base_gender = np.random.randn(64)
    base_food = np.random.randn(64)
    
    # king = royalty + male, queen = royalty + female
    engine.add_vector("king", base_royalty + base_gender * 1.5)
    engine.add_vector("man", base_gender * 1.5)
    engine.add_vector("woman", -base_gender * 1.5)
    engine.add_vector("queen", base_royalty - base_gender * 1.5)
    engine.add_vector("apple", base_food)
    
    # 2. Metric Verification
    u = engine.vocab["king"]
    v = engine.vocab["queen"]
    cos_sim = engine.cosine_similarity(u, v)
    l2_dist = engine.euclidean_distance(u, v)
    dot_prod = engine.dot_product(u, v)
    
    print(f"King vs Queen -> Cosine Similarity: {cos_sim:.4f}")
    print(f"King vs Queen -> Euclidean Distance: {l2_dist:.4f}")
    print(f"King vs Queen -> Dot Product (Normalized): {dot_prod:.4f}")
    
    # Check identity: for unit vectors, dot_product == cosine_similarity
    assert np.isclose(cos_sim, dot_prod), "For L2-normalized vectors, Dot Product must equal Cosine Similarity!"
    
    # 3. Solve Analogy: man is to king as woman is to ???
    results = engine.solve_analogy(word_a="man", word_b="king", word_c="woman")
    print("Analogy (king - man + woman) Top Result:", results[0])
    assert results[0][0] == "queen", "Analogy arithmetic must resolve to queen!"
    print("All Linear Algebra Assertions Passed!")
''', language="python")

st.divider()

# ==============================================================================
# Section 10: Interactive Workbench: High-Dimensional Vector Proximity Simulator
# ==============================================================================
st.header("10. Interactive Workbench: High-Dimensional Vector Proximity Simulator", divider="rainbow")

st.markdown("""
Experiment with high-dimensional vector geometry. Compare Cosine Similarity, Euclidean Distance, 
and Dot Product across synthesized semantic vectors, and test how vector magnitude affects metric divergence:
""")

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    concept_a = st.selectbox(
        "Select Concept Vector A:",
        ["Artificial Intelligence", "Machine Learning", "Quantum Computing", "Fresh Red Apple", "Golden Retriever"]
    )
    
    concept_b = st.selectbox(
        "Select Concept Vector B:",
        ["Machine Learning", "Deep Neural Networks", "Database Query", "Fresh Red Apple", "Banana Fruit"]
    )

with col_wb2:
    vector_dim_sim = st.select_slider("Embedding Dimension (D):", options=[64, 128, 384, 768, 1536], value=384)
    apply_normalization = st.checkbox("Apply L2 Unit Normalization to Vectors", value=True)
    scale_magnitude_b = st.slider("Vector B Magnitude Scaling Factor:", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# Deterministic simulation of semantic vectors
def get_simulated_vector(concept: str, dim: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    base = np.random.randn(dim)
    # Add domain clusters
    if "Intelligence" in concept or "Learning" in concept or "Neural" in concept:
        base += np.ones(dim) * 0.8
    elif "Apple" in concept or "Banana" in concept:
        base += np.linspace(-1, 1, dim) * 1.2
    return base

seeds = {
    "Artificial Intelligence": 101,
    "Machine Learning": 102,
    "Quantum Computing": 201,
    "Fresh Red Apple": 301,
    "Golden Retriever": 401,
    "Deep Neural Networks": 103,
    "Database Query": 501,
    "Banana Fruit": 302
}

vec_a = get_simulated_vector(concept_a, vector_dim_sim, seeds.get(concept_a, 42))
vec_b = get_simulated_vector(concept_b, vector_dim_sim, seeds.get(concept_b, 43)) * scale_magnitude_b

if apply_normalization:
    vec_a = vec_a / np.linalg.norm(vec_a)
    vec_b = vec_b / np.linalg.norm(vec_b)

# Compute metrics
sim_cos = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
dist_l2 = float(np.linalg.norm(vec_a - vec_b))
dot_prod_val = float(np.dot(vec_a, vec_b))

st.markdown("#### Distance Metric Comparative Diagnostics")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Cosine Similarity (Angle)", f"{sim_cos:.4f}")
    st.caption("Invariant to magnitude scaling; strictly reflects directional semantic overlap.")

with col_m2:
    st.metric("Euclidean Distance (L2)", f"{dist_l2:.4f}")
    st.caption("Straight-line physical coordinate distance; sensitive to vector norm.")

with col_m3:
    st.metric("Dot Product (Projection)", f"{dot_prod_val:.4f}")
    st.caption("Inner product; matches Cosine Similarity when vectors are L2-normalized.")

# Coordinate distribution visualizer
st.markdown("#### Latent Coordinate Slice Inspection (First 30 Dimensions)")
slice_df = pd.DataFrame({
    "Dimension Index": list(range(30)),
    f"Vector A ({concept_a})": vec_a[:30],
    f"Vector B ({concept_b})": vec_b[:30]
})
st.line_chart(slice_df.set_index("Dimension Index"), height=240)

if apply_normalization:
    st.success("✅ **L2 Normalization Active**: Both vectors have length 1.0000. Notice that Dot Product equals Cosine Similarity, and Euclidean Distance squared equals 2 - 2*(Cosine Similarity).")
else:
    st.warning("⚠️ **Un-Normalized Vectors**: Magnitude scaling alters Euclidean Distance and Dot Product, while Cosine Similarity remains perfectly invariant!")

st.divider()

# ==============================================================================
# Section 11: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering high-dimensional embeddings requires understanding linear algebra mechanics, 
hardware serving performance, and vector database retrieval failure modes:
""")

st.subheader("Principal / Staff-Level Interview Questions")

with st.expander("Q1: Why is Dot Product preferred over Cosine Similarity in high-throughput vector search engines?"):
    st.markdown("""
    **Direct Answer:**
    The preference is driven strictly by **hardware compute complexity**:
    
    * **Cosine Similarity Formula**:
      `Cosine(u, v) = (u . v) / (sqrt(sum(u_i^2)) * sqrt(sum(v_i^2)))`
      Evaluating this query across 10 million vectors requires computing 10 million square roots and floating-point divisions, which are computationally expensive and cannot be easily vectorized across GPU Tensor Cores.
    * **The Production Optimization**:
      By **pre-normalizing all vectors to unit length (||u||_2 = 1.0)** during ingestion, the denominator is guaranteed to equal 1.0.
      Therefore, `Cosine(u, v) == Dot(u, v)`.
    * **Hardware Advantage**:
      The Dot Product reduces to a single contiguous matrix multiplication:
      `Scores = Query_Vector * Database_Matrix^T`
      This executes as a dense General Matrix Multiply (GEMM) running at maximum theoretical memory bandwidth on GPU Tensor Cores or CPU AVX-512 vector units.
    """)

with st.expander("Q2: What is the Hubness Problem in high-dimensional vector spaces and how does it affect k-NN retrieval?"):
    st.markdown("""
    **Direct Answer:**
    As dimensionality (D) scales beyond 500, vector spaces suffer from an intrinsic topological distortion known as the **Hubness Phenomenon**:
    
    1. Due to high-dimensional variance in distance distributions, a tiny fraction of vectors (called **"Hubs"**) end up having exceptionally small average distances to almost all other vectors.
    2. These hub points appear as top-k nearest neighbors to thousands of completely unrelated user queries, creating artificial false-positive retrievals.
    3. **Remediation**:
       * Apply **Centering and Whitening**: subtract the global mean vector mu and divide by covariance eigenvalues.
       * Use **Local Inverted Distance Weighting** or Mutual Proximity algorithms (such as Cosine Anti-Hubness normalizers) to penalize vectors that appear too frequently across disjoint queries.
    """)

with st.expander("Q3: What is Matryoshka Representation Learning (MRL) and why is it revolutionary for RAG systems?"):
    st.markdown("""
    **Direct Answer:**
    Traditional embedding models produce fixed-length vectors (e.g. 1,536 dimensions). If you truncate the vector to save database memory, retrieval accuracy collapses because information is uniformly distributed across all dimensions.
    
    **Matryoshka Representation Learning (MRL)**:
    1. During training, the loss function evaluates multiple nested prefixes of the vector simultaneously: e.g. loss is computed on the first 64 dims, first 128 dims, first 256 dims, first 512 dims, and all 1,536 dims.
    2. This forces the neural network to compress the most critical, high-signal semantic information into the **earliest dimensions** (like nested Russian Matryoshka nesting dolls).
    3. **Production Impact**: A vector database can store only the first **256 dimensions** in RAM for initial fast nearest-neighbor filtering, slashing VRAM and storage costs by **83%** while retaining over 98% of full-dimensional retrieval recall.
    """)

with st.expander("Q4: Why does Euclidean Distance produce identical ranking orders to Cosine Similarity for L2-normalized embeddings?"):
    st.markdown("""
    **Direct Answer:**
    Expand the squared Euclidean distance between two unit vectors u and v (where ||u||^2 = 1 and ||v||^2 = 1):
    
    `||u - v||^2 = (u - v) . (u - v) = ||u||^2 + ||v||^2 - 2 * (u . v)`
    `||u - v||^2 = 1 + 1 - 2 * Cosine(u, v) = 2 - 2 * Cosine(u, v)`
    
    * Notice that ||u - v||^2 is a **strictly monotonic decreasing linear function** of \text{Cosine}(u, v).
    * As Cosine Similarity increases from -1 to 1, Euclidean distance decreases monotonically from 2 to 0.
    * Therefore, the relative ranking order of candidate documents is **100% mathematically identical** regardless of whether an index uses Euclidean Distance or Cosine Similarity, provided the vectors are unit-normalized.
    """)

with st.expander("Q5: What is the Curse of Dimensionality and why does distance contrast diminish as D increases?"):
    st.markdown("""
    **Direct Answer:**
    As dimension D -> infinity, the ratio between the distance to the farthest neighbor and the distance to the nearest neighbor approaches 1.0:
    
    `lim_{D -> inf} (Dist_max - Dist_min) / Dist_min -> 0`
    
    In ultra-high dimensions, almost all pairs of points become equidistant from one another. Random noise can easily flip the top-1 retrieved neighbor. 
    This is why foundation embedding models generally cap dimensionality between **768 and 3,072 dimensions**; scaling beyond 4,096 dimensions provides diminishing semantic returns while exacerbating distance contrast degradation.
    """)

st.subheader("Production Engineering Traps")

st.error("""
🚨 **Trap 1 (Error): Storing Un-Normalized Vectors in Inner-Product (Dot Product) Indexes**
If you configure a vector database (Pinecone, Milvus, pgvector) to use `dot_product` or `ip` index metrics without L2-normalizing the input vectors, documents with artificially large vector norms (e.g. extremely long documents) will dominate search results regardless of topical relevance.
*Remediation*: Always enforce `u = u / np.linalg.norm(u)` before upserting vectors into inner-product indexes.
""")

st.error("""
🚨 **Trap 2 (Error): Dimension Mismatch between Query Encoder and Pre-Indexed Database**
Deploying an updated embedding model (e.g. migrating from `bge-small` with 384 dimensions to `bge-large` with 1,024 dimensions) without re-indexing the underlying database causes queries to crash with dimension mismatch vector errors or produces garbage nearest neighbors.
*Remediation*: Embed model metadata (model name, version, dimension) directly in vector database index schemas and enforce CI/CD schema verification.
""")

st.warning("""
⚠️ **Trap 3 (Warning): Using Euclidean Distance on Variable-Length Document Embeddings**
When document embeddings are un-normalized, Euclidean distance measures length differences alongside topical differences. A 50-word abstract will appear distant from a 5,000-word paper on the exact same topic because of norm disparities.
*Remediation*: Always use Cosine Similarity or pre-normalize vectors to eliminate length bias.
""")

st.warning("""
⚠️ **Trap 4 (Warning): Blindly Scaling Embedding Dimensions Beyond 4,096 Dimensions**
Engineering teams often assume higher dimensionality always yields better retrieval. In practice, scaling from 1,536 to 8,192 dimensions quadruples memory bandwidth and vector DB storage while offering < 1% improvement in MTEB benchmark recall.
*Remediation*: Benchmark task-specific recall across dimensions before committing to massive embedding matrices.
""")

st.subheader("Concept Flashcards")

col_fc1, col_fc2 = st.columns(2)

with col_fc1:
    with st.container():
        st.info("**Dense Embedding vs. One-Hot**")
        st.caption("Dense embeddings compress concepts into continuous low-dimensional coordinates where proximity reflects semantics; One-hot vectors are high-dimensional, sparse, and completely orthogonal.")

    with st.container():
        st.info("**L2 Unit Normalization**")
        st.caption("Rescaling a vector to length 1.0 (u / ||u||), placing it on the unit hypersphere and simplifying Cosine Similarity to a fast hardware dot product.")

with col_fc2:
    with st.container():
        st.info("**Distributional Hypothesis**")
        st.caption("Firth's linguistic principle: words occurring in similar linguistic contexts possess similar semantic meanings, driving neural embedding convergence.")

    with col_fc2:
        with st.container():
            st.info("**Matryoshka Embeddings (MRL)**")
            st.caption("Embeddings trained so that early dimension prefixes (e.g. first 256 of 1536 dims) contain the highest-signal information, allowing 80% storage reduction.")

st.divider()

# ==============================================================================
# Section 12: Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of embedding fundamentals, dense geometry, and vector distance metrics. 
Complete all 6 checklist items:
""")

cb1 = st.checkbox("1. I understand how dense embeddings map discrete categorical tokens into continuous semantic coordinate spaces.", key="chk_41_1")
cb2 = st.checkbox("2. I can explain the Curse of Orthogonality and why sparse representations fail at semantic similarity.", key="chk_41_2")
cb3 = st.checkbox("3. I know the capacity and memory trade-offs of embedding dimensionality (from 384 to 4,096 dimensions).", key="chk_41_3")
cb4 = st.checkbox("4. I understand why Cosine Similarity simplifies to a raw Dot Product when vectors are L2 unit-normalized.", key="chk_41_4")
cb5 = st.checkbox("5. I can formulate the exact mathematical relationship between Euclidean Distance and Cosine Similarity.", key="chk_41_5")
cb6 = st.checkbox("6. I understand how Matryoshka Representation Learning (MRL) enables 80% vector database memory reduction.", key="chk_41_6")

if cb1 and cb2 and cb3 and cb4 and cb5 and cb6:
    st.success("🎉 Outstanding! You have mastered all foundational principles of dense embeddings, vector spaces, and distance metrics!")
