import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="7.1 Embedding Fundamentals - RAG Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class EmbeddingProjectionSimulator:
    """Simulates high-dimensional embedding generation and cosine similarity calculation."""
    def __init__(self):
        self.reference_embeddings = {
            "Cloud Infrastructure": np.array([0.88, 0.22, 0.15, 0.10]),
            "Distributed Kubernetes": np.array([0.82, 0.31, 0.18, 0.08]),
            "Corporate Finance": np.array([0.10, 0.12, 0.92, 0.18]),
            "Accounting & Tax": np.array([0.15, 0.08, 0.89, 0.25]),
            "Employee Healthcare": np.array([0.12, 0.85, 0.10, 0.30])
        }

    def compute_similarity(self, concept_a: str, concept_b: str) -> float:
        vec_a = self.reference_embeddings.get(concept_a)
        vec_b = self.reference_embeddings.get(concept_b)
        if vec_a is None or vec_b is None:
            return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

# Title & Overview
st.title("7.1 Embedding Fundamentals: Vector Spaces, Projections & Semantic Geometry")
st.caption("Chapter 7: Embeddings | Domain: 6. RAG")
st.markdown("""
Vector embeddings are the **mathematical bedrock of modern Artificial Intelligence and Retrieval-Augmented Generation**. 
While computers process binary numbers and human beings communicate in natural language words, embeddings serve as the 
universal translation layer—mapping unstructured phrases, sentences, and documents into continuous, high-dimensional coordinate spaces.

In an embedding space, **geometric proximity mirrors human semantic meaning**. Concepts that share topical affinities 
cluster together tightly, while unrelated concepts are separated by vast multidimensional angles.

In this lesson, you will master:
1. **The Fundamental Definition**: How neural networks project discrete strings into dense coordinate arrays.
2. **The RAG Imperative**: Why semantic embeddings unlock intent-driven search that keyword matching cannot achieve.
3. **The Transformation Pipeline**: Step-by-step tokenization, transformer attention, pooling, and L2 normalization.
4. **Vector Dimensionality**: Analyzing trade-offs across 384-d, 768-d, 1536-d, and 3072-d representation spaces.
5. **Production Failure Traps**: Navigating metric mismatches, unnormalized vectors, and out-of-domain vocabulary degradation.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: High-Dimensional Embedding Projection Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph EmbeddingProjectionBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_input {
            label="Stage 1: Text Tokenization & Subword Encoding";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            raw_str [label="Input Text String\\n'Deploy container cluster to production'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            tokenizer [label="Subword Tokenizer (BPE / WordPiece)\\nMaps string to token integer IDs: [412, 8912, 3105, ...]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            raw_str -> tokenizer;
        }

        subgraph cluster_transformer {
            label="Stage 2: Transformer Encoder & Contextual Representation";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            embed_lookup [label="Token & Position Embedding Lookup\\nInitial static token vectors + sinusoidal position vectors", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            self_attention [label="Multi-Head Self-Attention Layers (12-24 Layers)\\nContextualizes tokens based on surrounding sentence words", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            token_hidden [label="Token Hidden States Matrix\\nMatrix shape: [Sequence Length, Hidden Dim 1536]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            tokenizer -> embed_lookup -> self_attention -> token_hidden;
        }

        subgraph cluster_pooling {
            label="Stage 3: Pooling, Dimensional Reduction & Normalization";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            mean_pool [label="Mean Pooling Layer\\nAverages token vectors weighted by attention mask", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            l2_norm [label="L2 Normalization Engine\\nScales vector to unit magnitude: ||v|| = 1.0", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            final_vector [label="Dense Normalized Embedding\\n1536-d float32 array ready for Vector DB indexing", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            token_hidden -> mean_pool -> l2_norm -> final_vector;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: What are Embeddings?
st.header("2. What are Embeddings?", divider="rainbow")
st.markdown("""
An **embedding** is a dense numerical vector—an ordered array of floating-point numbers—that represents 
the semantic meaning of a piece of data (text, image, audio, or graph) in a continuous mathematical vector space.

* **Dense vs Sparse**: 
  * **Sparse Representations (One-Hot, TF-IDF, BM25)**: Vectors have dimensions equal to vocabulary size (50,000+). 99.9% of the values are zeros. Each dimension corresponds to an exact keyword. There is zero concept of synonyms: *"physician"* and *"doctor"* have completely orthogonal vectors with zero overlap.
  * **Dense Embeddings (Word2Vec, BERT, OpenAI)**: Compact vectors typically consisting of 384 to 3072 continuous floating-point numbers. Every single dimension carries a non-zero value representing a latent semantic feature.
* **Continuous Semantic Space**: In dense embedding space, *"physician"* and *"doctor"* share almost identical coordinate values, yielding a Cosine Similarity score exceeding 0.95.
""")
st.divider()

# Section 3: Why Embeddings are Used in RAG
st.header("3. Why Embeddings are Used in RAG", divider="rainbow")
st.markdown("""
Embeddings form the engine that powers modern Retrieval-Augmented Generation:

1. **Natural Vocabulary Independence**: Users rarely format queries using the exact terminology present in corporate manuals. A user asking *"How do I fix my crashed container?"* needs to retrieve documentation titled *"Troubleshooting Pod Eviction and Out-of-Memory Errors"*. Embeddings bridge this linguistic gap automatically.
2. **Sub-Millisecond Approximate Nearest Neighbor (ANN) Search**: Because embeddings are fixed-length numerical arrays, search engines utilize hardware-accelerated SIMD instructions (AVX-512) and graph indices (HNSW) to search millions of documents in under 5 milliseconds.
3. **Multilingual and Cross-Lingual Matching**: State-of-the-art multilingual embedding models (e.g., Cohere Multilingual v3) project multiple languages into the same vector space. A query submitted in Spanish will retrieve the relevant English document because their conceptual coordinates align.
""")
st.divider()

# Section 4: Text-to-Vector Conversion
st.header("4. Text-to-Vector Conversion: The Internal Pipeline", divider="rainbow")
st.markdown("""
How does a raw English sentence transform into an array of floating-point coordinates? 
The transformation follows four deterministic stages:
""")

st.markdown("""
1. **Subword Tokenization**: The input string is broken into tokens using a tokenizer (e.g., Byte-Pair Encoding). For example, *"unbreakable"* becomes `["un", "break", "able"]`, mapped to integer IDs `[281, 1402, 591]`.
2. **Contextual Transformer Layers**: The token IDs pass through 12 to 24 transformer layers. Multi-Head Self-Attention updates each token's representation based on its surrounding context (e.g., distinguishing the financial *"bank"* from the river *"bank"*).
3. **Pooling Layer**: Transformers output an individual vector for *every token*. To obtain a single vector representing the entire sentence, the model applies a pooling strategy:
   * **Mean Pooling**: Averages all token vectors across the sequence (the standard approach in Sentence Transformers).
   * **[CLS] Token Pooling**: Uses the first special classification token vector.
4. **L2 Unit Normalization**: The pooled vector is divided by its Euclidean length, scaling its magnitude to exactly 1.0. This allows subsequent cosine similarity queries to execute as blazing-fast dot products.
""")
st.divider()

# Section 5: Semantic Representation
st.header("5. Semantic Representation: Manifolds and Conceptual Geometry", divider="rainbow")
st.markdown("""
Embedding models do not store memorized text—they learn a **latent semantic manifold**.

* **Geometric Directions as Semantic Concepts**: In well-trained embedding spaces, specific geometric directions correspond to abstract properties (e.g., formality, sentiment, tense, domain).
* **The Classical Analogy Vector Arithmetic**:
  `Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")`.
* **Clustering by Domain**: In high-dimensional space, technical database manuals group into one cluster, financial audit reports into another, and HR employment policies into a third. Distance between cluster centers reflects high-level conceptual distance.
""")
st.divider()

# Section 6: Vector Dimensions
st.header("6. Vector Dimensions: Representational Capacity vs System Cost", divider="rainbow")
st.markdown("""
Embedding models are characterized by their **dimensionality (`d`)**—the number of floating-point values per vector:
""")

st.markdown("""
| Model Family | Dimensions | Vector Size in RAM (Float32) | Optimal Engineering Use Case |
| :--- | :---: | :---: | :--- |
| **`all-MiniLM-L6-v2`** | 384 | 1.5 KB per vector | Ultra-lightweight, local CPU execution, edge devices |
| **`bge-base-en-v1.5`** | 768 | 3.0 KB per vector | Balanced accuracy and storage for standard enterprise RAG |
| **`text-embedding-3-small`** | 1536 | 6.0 KB per vector | Production cloud RAG; scalable MTEB benchmark accuracy |
| **`text-embedding-3-large`** | 3072 | 12.0 KB per vector | High-precision legal, medical, and complex reasoning |
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
State-of-the-art models support **Matryoshka Representation Learning (MRL)** (such as OpenAI's `text-embedding-3`). 
MRL models are trained so that the most critical information is packed into the *first* dimensions. 
Engineers can truncate a 1536-dimensional vector down to 512 dimensions by simply slicing the array (`v[:512]`) 
and re-normalizing, slashing vector database RAM costs by 66% with less than 2% loss in retrieval recall!
""")
st.divider()

# Section 7: Embedding Similarity
st.header("7. Embedding Similarity: Angular Alignment vs Coordinate Distance", divider="rainbow")
st.markdown("""
Once text is projected into vector space, retrieval engines evaluate relevance using similarity metrics:

* **Cosine Similarity**: Measures the cosine of the angle between two vectors:
  `Cosine Similarity = (u . v) / (||u|| * ||v||)`.
  * Values range from `-1.0` (opposite) to `+1.0` (identical orientation).
  * Scale-invariant: ignores magnitude, ensuring that short queries match long passages accurately.
* **Dot Product (Inner Product)**: Measures `Sum(u_i * v_i)`.
  * When vectors are L2-normalized (`||u|| = 1.0`), Dot Product is mathematically identical to Cosine Similarity.
  * Supported directly by CPU SIMD hardware instructions (AVX-512) for nanosecond execution speeds.
""")
st.divider()

# Section 8: Production Implementation
st.header("8. Production Implementation: Dense Neural Vector Generator Pipeline", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating tokenization, embedding generation, 
mean pooling, and L2 normalization using NumPy.
""")

st.code('''
import numpy as np
from typing import List, Dict, Any

class NeuralEmbeddingPipeline:
    """Simulates internal transformer mean-pooling and L2 normalization."""
    def __init__(self, hidden_dim: int = 1536):
        self.hidden_dim = hidden_dim

    def mean_pool_and_normalize(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Applies mean pooling weighted by attention mask, followed by L2 unit normalization.
        token_embeddings: shape [seq_len, hidden_dim]
        attention_mask: shape [seq_len] (1 for real token, 0 for padding)
        """
        # Expand mask for broadcasting
        expanded_mask = attention_mask[:, np.newaxis]
        weighted_sum = np.sum(token_embeddings * expanded_mask, axis=0)
        total_tokens = np.clip(np.sum(expanded_mask), a_min=1e-9, a_max=None)
        
        # Mean pooled sentence vector
        pooled_vector = weighted_sum / total_tokens
        
        # L2 Unit Normalization
        norm = np.linalg.norm(pooled_vector)
        if norm > 0:
            normalized_vector = pooled_vector / norm
        else:
            normalized_vector = pooled_vector

        return normalized_vector.astype(np.float32)

    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Normalized dot product equals cosine similarity in unit space."""
        return float(np.dot(vec_a, vec_b))
''', language="python")
st.divider()

# Section 9: Interactive Workbench
st.header("9. Interactive Workbench: Latent Vector Projection & Dimensionality Simulator", divider="rainbow")
st.markdown("""
Select pairs of conceptual domains to evaluate their simulated Cosine Similarity in a normalized semantic coordinate space.
""")

sim = EmbeddingProjectionSimulator()
concepts = list(sim.reference_embeddings.keys())

col_c1, col_c2 = st.columns(2)
with col_c1:
    concept_1 = st.selectbox("Select Concept A:", concepts, index=0)
with col_c2:
    concept_2 = st.selectbox("Select Concept B:", concepts, index=1)

similarity_score = sim.compute_similarity(concept_1, concept_2)

st.subheader("Semantic Proximity Analysis")
col_s1, col_s2 = st.columns([1, 1])

with col_s1:
    st.metric(f"Cosine Similarity ({concept_1} vs {concept_2})", f"{similarity_score:.4f}")
    if similarity_score >= 0.80:
        st.success("🟢 **High Semantic Affinity**: Concepts belong to the identical topical manifold.")
    elif similarity_score >= 0.40:
        st.info("🟡 **Moderate Topical Overlap**: Concepts share peripheral technical associations.")
    else:
        st.warning("🔴 **Low Semantic Affinity**: Concepts occupy orthogonal coordinate spaces.")

with col_s2:
    st.markdown("""
    **Understanding the Metric**:
    * **1.000**: Perfect directional alignment (identical concept).
    * **0.80–0.95**: Closely related synonyms and technical domains.
    * **0.00–0.30**: Unrelated domains (e.g., Cloud Infrastructure vs Employee Healthcare).
    """)
st.divider()

# Section 10: Technical Interview Preparation & Production Traps
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is the mathematical difference between Dense Embeddings and Sparse Keyword Representations?", expanded=True):
    st.markdown("""
    **Answer:**
    * **Sparse Representations (One-Hot, TF-IDF, BM25)**: Dimensionality equals vocabulary size (50,000+). Vectors are mostly zeros. Each dimension represents a discrete, exact keyword. Synonyms occupy completely orthogonal axes (`dot product = 0`).
    * **Dense Embeddings (Transformers, OpenAI, BGE)**: Dimensionality is compact and fixed (384 to 3072). Every value is a non-zero continuous float representing a learned latent feature. Synonyms occupy near-identical directional vectors (`cosine similarity > 0.90`).
    """)

with st.expander("Q2: Why must embedding vectors be L2-normalized in production vector search?"):
    st.markdown("""
    **Answer:**
    L2 normalization scales every vector to a unit length (`||v|| = 1.0`). 
    When vectors have unit length, the Cosine Similarity formula simplifies directly to the **Dot Product** (`dot(u, v)`), 
    eliminating the need to compute expensive vector magnitudes and square roots during query execution. 
    Dot product operations execute in nanoseconds using CPU hardware SIMD instructions (AVX-512).
    """)

with st.expander("Q3: What is Matryoshka Representation Learning (MRL), and why is it valuable for RAG?"):
    st.markdown("""
    **Answer:**
    Matryoshka Representation Learning trains embedding models to compress the most critical semantic variance into the 
    first `N` dimensions (like Russian nesting dolls). This allows developers to truncate 1536-dimensional embeddings 
    down to 512 or 256 dimensions simply by slicing the vector (`v[:512]`) and re-normalizing. This reduces vector database 
    RAM consumption and query latency by up to 75% with negligible recall degradation.
    """)

with st.expander("Q4: How does Mean Pooling differ from [CLS] Token Pooling in transformer embedding models?"):
    st.markdown("""
    **Answer:**
    * **Mean Pooling**: Calculates the mathematical average across all token hidden state vectors in the sequence, weighted by the attention mask. Captures broad document-wide topical context; standard in Sentence Transformers and BGE.
    * **[CLS] Token Pooling**: Extracts the hidden state of the first special classification token (`[CLS]`). Popular in BERT classification, but can over-index on sentence beginnings unless specifically trained for pooling.
    """)

with st.expander("Q5: Why can't you mix embeddings from different models in the same vector database collection?"):
    st.markdown("""
    **Answer:**
    Every embedding model learns its own distinct, idiosyncratic high-dimensional coordinate manifold during training. 
    A vector generated by `all-MiniLM-L6-v2` (384-d) shares zero geometric alignment with a vector from `text-embedding-3-small` (1536-d). 
    Even if two different models happen to share the identical dimensionality (e.g., two 768-d models), their semantic axes represent 
    different latent features. Comparing vectors across models yields meaningless random distances.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Querying a Vector Collection with a Different Embedding Model (Silent Scrambling)**
    Embedding the corpus with OpenAI `text-embedding-3-small`, but accidentally using a local Hugging Face model for user queries. The vector database computes distances without throwing errors, but the search results are completely random garbage.
    """)
    st.error("""
    **Trap 2: Mismatched Distance Space Configuration in Vector Database (Inverted Rankings)**
    Configuring a vector database collection to use Euclidean Distance (`l2`) when your application logic assumes Cosine Similarity where higher scores are better. The application treats the farthest, least relevant documents as top hits!
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Omitting Attention Mask During Manual Mean Pooling (Padding Pollution)**
    Averaging token embeddings across the entire sequence length without filtering out zero-padding tokens. Sequences with heavy padding have their vectors pulled towards origin, corrupting semantic distances.
    """)
    st.warning("""
    **Trap 4: Disregarding Text Prefix Requirements on Asymmetric Models (Recall Drop)**
    Using models like BGE or E5 that mandate specific query prefixes (`"query: "` for search, `"passage: "` for documents) without prepending them. Asymmetric retrieval recall drops by over 15% when prefixes are omitted.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the key property of dense vector embeddings?")
        st.markdown("**Back**: Geometric spatial proximity mirrors human semantic conceptual meaning.")
    with st.container(border=True):
        st.markdown("**Front**: Why does L2 normalization accelerate vector similarity search?")
        st.markdown("**Back**: It allows Cosine Similarity to be computed as a simple, fast hardware-accelerated Dot Product.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is Matryoshka Representation Learning (MRL)?")
        st.markdown("**Back**: Training vectors so they can be truncated to smaller dimensions with minimal loss in accuracy.")
    with st.container(border=True):
        st.markdown("**Front**: Can you compare vectors generated by two different models?")
        st.markdown("**Back**: Never. Each model projects into a completely distinct, unaligned coordinate manifold.")

st.divider()

# Section 11: Self-Assessment Checklist
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Embedding Fundamentals before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand what embeddings are and how dense vectors differ from sparse keyword representations.", value=True, key="chk_emb_fund_1")
    chk2 = st.checkbox("I can explain why semantic embeddings are essential for intent-driven search in RAG.", value=True, key="chk_emb_fund_2")
    chk3 = st.checkbox("I understand the transformation pipeline: tokenization, attention, mean pooling, and L2 normalization.", value=True, key="chk_emb_fund_3")
with c2:
    chk4 = st.checkbox("I know the trade-offs between different vector dimensions (384-d to 3072-d).", value=True, key="chk_emb_fund_4")
    chk5 = st.checkbox("I understand how Matryoshka Representation Learning enables vector dimension truncation.", value=True, key="chk_emb_fund_5")
    chk6 = st.checkbox("I can identify production traps such as model mismatch, distance space confusion, and padding pollution.", value=True, key="chk_emb_fund_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Embedding Fundamentals Mastery Demonstrated!** You are ready to advance to Lesson 7.2 in the curriculum.")
