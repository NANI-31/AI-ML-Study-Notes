import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="1.1 What are Embeddings? - Embeddings Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Title & Overview
st.title("1.1 What are Embeddings? Text-to-Vector Foundations, Numerical Representations, & Semantic Spaces")
st.caption("Chapter 1: Introduction to Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Computers, GPUs, and neural networks are fundamentally mathematical engines: they perform addition, dot products, 
and matrix multiplications on numerical tensors. They cannot directly compute similarity or meaning on raw Unicode strings like `"apple"` or `"automobile"`.

**Vector Embeddings** are the universal bridge connecting human language to computational geometry:
* An embedding translates discrete symbols (words, sentences, documents, audio, or images) into **dense continuous vectors of real numbers** in high-dimensional space (typically 384 to 3072 dimensions).
* Crucially, embeddings are **semantic representations**: items with similar meanings are mapped to **geometrically proximate coordinates** in vector space.

This lesson explores the transition from raw text to dense numerical vectors, the failure modes of traditional one-hot and keyword matching, and how geometric distance encodes human conceptual meaning.
""")

st.divider()

# Section 1: Conceptual Architecture: From String to Vector Space
st.header("1. The Text-to-Vector Projection Pipeline", divider="rainbow")

col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    st.markdown("""
    ### Key Foundations:
    * **1. What is a Vector?**:
      An ordered list of floating-point numbers [v1, v2, ..., vd] representing coordinates in a d-dimensional vector space.
    * **2. Sparse vs. Dense Representations**:
      * *One-Hot Encodings (Sparse)*: A vocabulary of 50,000 words produces 50,000-dimensional vectors with 49,999 zeros and a single 1. Orthogonal vectors have zero dot product; `"car"` and `"automobile"` share 0% overlap.
      * *Embeddings (Dense)*: Dense vectors of 768 or 1536 non-zero floats where latent semantic dimensions capture abstract concepts (e.g. animacy, formality, sentiment).
    * **3. Semantic Proximity**:
      Concepts that appear in similar textual contexts occupy nearby geometric regions. In vector space:
      `distance("king", "queen")` is significantly smaller than `distance("king", "banana")`.
    """)

with col_v2:
    st.markdown("### Text-to-Vector Pipeline Architecture")
    emb_dot = """
    digraph TextToVec {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Input [label="Raw Text String\\n'Artificial intelligence transforms healthcare'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Tokenizer [label="Subword Tokenizer\\n(BPE / WordPiece -> Token IDs)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Transformer [label="Transformer Encoder\\n(Self-Attention Contextualization)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
        Pooling [label="Mean / CLS Pooling\\n(Aggregate token vectors into sentence vector)", fillcolor="#FEE2E2", color="#DC2626"];
        Norm [label="L2 Normalization\\n(Unit vector length = 1.0)", fillcolor="#DCFCE7", color="#16A34A"];
        VectorOut [label="Dense 1536-Dim Embedding\\n[0.014, -0.082, 0.311, ..., -0.045]", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        Input -> Tokenizer -> Transformer -> Pooling -> Norm -> VectorOut;
    }
    """
    st.graphviz_chart(emb_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Embedding Representation in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates simulated dense embedding projection, L2 vector normalization, and cosine similarity calculation.
""")

class ProductionEmbeddingConceptEngine:
    """
    Demonstrates fundamental vector operations, L2 normalization,
    and geometric similarity calculations for text representations.
    """
    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        # Seeded dictionary of conceptual vector anchors
        np.random.seed(42)
        self.concept_anchors = {
            "technology": np.array([0.85, 0.72, 0.15, 0.10, -0.30, 0.45, 0.60, 0.20]),
            "finance": np.array([0.78, 0.20, 0.88, 0.05, -0.10, 0.25, 0.40, 0.65]),
            "medical": np.array([0.20, 0.15, 0.10, 0.92, 0.80, -0.10, 0.30, 0.05]),
            "culinary": np.array([-0.50, -0.30, -0.20, 0.10, 0.15, 0.85, -0.40, 0.70])
        }

    def generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generates a normalized semantic vector based on conceptual keywords."""
        vec = np.zeros(self.dimensions, dtype=np.float64)
        text_lower = text.lower()
        matched = False

        for concept, anchor in self.concept_anchors.items():
            if any(term in text_lower for term in [concept, "ai", "computer", "cloud", "code"]) and concept == "technology":
                vec += anchor * 1.5
                matched = True
            elif any(term in text_lower for term in [concept, "money", "stock", "revenue", "bank"]) and concept == "finance":
                vec += anchor * 1.5
                matched = True
            elif any(term in text_lower for term in [concept, "doctor", "health", "hospital", "patient"]) and concept == "medical":
                vec += anchor * 1.5
                matched = True
            elif any(term in text_lower for term in [concept, "food", "recipe", "cake", "cook", "pizza"]) and concept == "culinary":
                vec += anchor * 1.5
                matched = True

        if not matched:
            # Deterministic pseudo-random vector based on string hash
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.uniform(-0.5, 0.5, self.dimensions)

        # Apply L2 Normalization: v / norm(v)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors: dot(v1, v2)."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

code_snippet = r'''
import numpy as np

class VectorEmbeddingOperations:
    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Scale vector to unit length (norm = 1.0)."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity is equivalent to dot product for L2-normalized vectors."""
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        return float(np.dot(v1_norm, v2_norm))
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Embedding Exploration Workbench
st.header("3. Interactive Vector Embedding & Similarity Workbench", divider="rainbow")
st.markdown("""
Compare how different sentences map to dense vector representations. Observe how sentences sharing the same conceptual domain 
yield high cosine similarity, while cross-domain sentences are geometrically distant.
""")

engine = ProductionEmbeddingConceptEngine(dimensions=8)

preset_sentences = [
    "Artificial intelligence transforms modern cloud computing systems.",
    "Machine learning algorithms optimize scalable software infrastructure.",
    "Quarterly financial revenue exceeded consensus earnings forecasts.",
    "Physicians prescribe clinical medications to improve patient health.",
    "A delicious chocolate cake recipe with vanilla frosting."
]

col_in1, col_in2 = st.columns([1, 1])

with col_in1:
    sentence_a = st.selectbox("Sentence A:", preset_sentences, index=0)
with col_in2:
    sentence_b = st.selectbox("Sentence B:", preset_sentences, index=1)

vec_a = engine.generate_mock_embedding(sentence_a)
vec_b = engine.generate_mock_embedding(sentence_b)
similarity_score = engine.cosine_similarity(vec_a, vec_b)

st.markdown("### Dense Vector Coordinate Inspection (8-Dimensional Preview)")

col_v_a, col_v_b = st.columns([1, 1])

with col_v_a:
    st.markdown("**Vector A Coordinates:**")
    df_a = pd.DataFrame([vec_a], columns=[f"Dim_{i+1}" for i in range(8)])
    st.dataframe(df_a.style.format("{:.3f}").background_gradient(cmap="coolwarm", axis=1), use_container_width=True)

with col_v_b:
    st.markdown("**Vector B Coordinates:**")
    df_b = pd.DataFrame([vec_b], columns=[f"Dim_{i+1}" for i in range(8)])
    st.dataframe(df_b.style.format("{:.3f}").background_gradient(cmap="coolwarm", axis=1), use_container_width=True)

st.markdown("### Geometric Alignment & Similarity Metric")
col_sc1, col_sc2 = st.columns([1, 1])
with col_sc1:
    st.metric(
        "Cosine Similarity Score",
        f"{similarity_score:.4f}",
        delta="Strong Semantic Match" if similarity_score > 0.70 else ("Moderate Similarity" if similarity_score > 0.40 else "Orthogonal / Distant")
    )
with col_sc2:
    st.markdown("""
    * **1.0**: Identical semantic direction.
    * **0.0**: Orthogonal (completely unrelated concepts).
    * **-1.0**: Diametrically opposite concepts.
    """)

# Pairwise Matrix across all preset sentences
st.markdown("### All-Pairs Semantic Similarity Heatmap Matrix")
matrix_data = []
for s1 in preset_sentences:
    row = []
    v1 = engine.generate_mock_embedding(s1)
    for s2 in preset_sentences:
        v2 = engine.generate_mock_embedding(s2)
        row.append(engine.cosine_similarity(v1, v2))
    matrix_data.append(row)

labels = ["Tech: AI Cloud", "Tech: ML Infra", "Finance: Earnings", "Medical: Healthcare", "Food: Cake"]
heatmap_df = pd.DataFrame(matrix_data, index=labels, columns=labels)
st.dataframe(heatmap_df.style.format("{:.2f}").background_gradient(cmap="viridis", axis=None), use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Embedding Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the fundamental difference between sparse and dense representations?**
    * **Answer**:
      * **Sparse Vectors (One-Hot, TF-IDF, BM25)**: Dimensionality equals vocabulary size (typically 50,000+). Almost all values are 0.0. Vectors are orthogonal; they fail to recognize that `"physician"` and `"doctor"` represent the same underlying semantic concept.
      * **Dense Vectors (Embeddings)**: Fixed, compact dimensionality (e.g. 384 to 1536). Every dimension is a non-zero floating-point coordinate learned through deep neural network pre-training. Closeness in vector space directly corresponds to semantic relatedness.

    **Q2: Why must embedding vectors be L2-normalized before computing dot products?**
    * **Answer**:
      * The dot product between two vectors u and v is: dot(u, v) = norm(u) * norm(v) * cos(theta).
      * If vectors are unnormalized, longer sentences or words with higher token frequencies may have larger vector norms, artificially inflating the dot product regardless of true angular similarity.
      * By enforcing L2 norm = 1.0, the dot product becomes mathematically identical to **Cosine Similarity**: dot(u, v) = cos(theta). This allows vector databases to execute high-speed cosine search using hardware-accelerated dot products.

    **Q3: How do transformer-based sentence embeddings differ from static word embeddings (Word2Vec/GloVe)?**
    * **Answer**:
      * **Static Word Embeddings (Word2Vec)**: Each word has a single static vector. The word `"bank"` has the exact same vector whether in *"river bank"* or *"investment bank"*, causing catastrophic polysemy confusion.
      * **Contextual Sentence Embeddings (BERT / Sentence Transformers)**: Multi-head self-attention dynamically conditions each token's vector on surrounding words, producing distinct representations for ambiguous words based on context.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Comparing Vectors Generated from Different Embedding Models**:
      * *Issue*: Computing cosine similarity between an OpenAI `text-embedding-3-small` vector and a HuggingFace `all-MiniLM-L6-v2` vector.
      * *Consequence*: Meaningless garbage scores; each embedding model projects into a completely distinct geometric latent coordinate space.
      * *Fix*: Always ensure queries and documents are embedded using the exact same model weights and checkpoint.

    * **Trap 2: Using Euclidean Distance on Unnormalized Embeddings**:
      * *Issue*: Measuring Euclidean distance on vectors whose magnitudes vary due to sentence length.
      * *Consequence*: Two sentences with identical meaning but different lengths appear far apart in Euclidean distance.
      * *Fix*: Always L2-normalize vectors to unit length prior to distance calculation.

    * **Trap 3: Truncating Tokens without Warning**:
      * *Issue*: Passing a 2,000-word document into an embedding model with a 512-token context limit.
      * *Consequence*: The model silently drops the last 1,500 words, losing the document's core conclusions.
      * *Fix*: Implement chunking pipelines to ensure full document coverage.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Vector Embedding**
      * *Definition*: A dense, continuous floating-point vector that captures the semantic meaning, context, and latent relationships of text in high-dimensional space.
    * **Flashcard 2: L2 Normalization**
      * *Definition*: Scaling a vector so that its Euclidean norm (magnitude) equals exactly 1.0, making the dot product equivalent to cosine similarity.
    * **Flashcard 3: Dense vs. Sparse**
      * *Definition*: Sparse representations use high dimensions with mostly zeros; dense representations pack rich semantic features into compact continuous coordinates.
    * **Flashcard 4: Polysemy**
      * *Definition*: The capacity for a single word to have multiple meanings (e.g. "apple" the fruit vs "Apple" the corporation), resolved by contextual transformer embeddings.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embedding Fundamentals:

- [ ] I can define a vector embedding and explain why neural networks require numerical representations.
- [ ] I understand the mathematical differences between sparse one-hot encodings and dense embeddings.
- [ ] I know why L2 normalization makes dot product equivalent to cosine similarity.
- [ ] I can contrast static word embeddings (Word2Vec) with contextual transformer embeddings (BERT/E5).
- [ ] I know how to calculate and interpret pairwise cosine similarity matrices across semantic domains.
""")
