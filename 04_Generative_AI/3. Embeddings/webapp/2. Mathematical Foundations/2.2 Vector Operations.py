import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.2 Vector Operations - Embeddings Curriculum",
    page_icon="➕",
    layout="wide"
)

# Title & Overview
st.title("2.2 Vector Operations: Addition, Subtraction, Dot Products, Matrix Transforms, & Semantic Analogies")
st.caption("Chapter 2: Mathematical Foundations | Domain: 3. Vector Embeddings")
st.markdown("""
In raw text, you cannot perform mathematical arithmetic: subtracting `"man"` from `"king"` produces a syntax error.
In **Vector Embedding Spaces**, however, **words and concepts are geometric coordinates subject to linear algebra**.

By performing fundamental vector operations:
1. **Vector Addition**: Merges semantic concepts. In mean pooling, token vectors are summed to represent an entire paragraph: `v_sentence = mean(token_vectors)`.
2. **Vector Subtraction**: Isolates semantic deltas (direction vectors). The delta `v_woman - v_man` isolates the concept of gender.
3. **Analogy Arithmetic**: Solves conceptual proportional reasoning: `v_king - v_man + v_woman ≈ v_queen`.
4. **Scalar Multiplication**: Amplifies or dampens semantic features (e.g. weighting keyword importance by 2.0x).
5. **Dot Product & Matrix Multiplication**: Projects high-dimensional vectors to compute batch similarity across millions of candidate documents in a single hardware-accelerated BLAS call.

This lesson explores the mathematics of vector operations and their direct applications in neural retrieval and representation learning.
""")

st.divider()

# Section 1: The Linear Algebra of Semantic Operations
st.header("1. Core Linear Algebra Operations in Embedding Systems", divider="rainbow")

col_op1, col_op2 = st.columns([1, 1])

with col_op1:
    st.markdown("""
    ### Five Fundamental Operations:
    * **1. Vector Addition (u + v)**:
      Component-wise sum: [u1 + v1, u2 + v2, ...]. Used in **Mean Pooling** where all token embeddings in a sentence are averaged into a single pooled document embedding.
    * **2. Vector Subtraction (u - v)**:
      Constructs a directional displacement vector pointing from v to u. Encodes abstract semantic relationships (e.g. capital-of, gender, tense).
    * **3. Scalar Multiplication (c * v)**:
      Scales magnitude without changing directional line. Used in BM25-hybrid weighting to boost high-confidence keywords.
    * **4. Dot Product (dot(u, v) = sum(u_i * v_i))**:
      Measures directional alignment and magnitude projection. Foundational to transformer attention layers: `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`.
    * **5. Batch Matrix Multiplication (M * q)**:
      Multiplying a matrix M (N x D document vectors) by a query vector q (D x 1) computes all N similarity scores in parallel.
    """)

with col_op2:
    st.markdown("### Semantic Analogy Arithmetic Vector Diagram")
    analogy_dot = """
    digraph AnalogyMath {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        King [label="v(King)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        MinusMan [label="- v(Man)", fillcolor="#FEE2E2", color="#DC2626"];
        RoyaltyDelta [label="Royalty Concept Vector\\n[King - Man]", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        PlusWoman [label="+ v(Woman)", fillcolor="#DCFCE7", color="#16A34A"];
        Queen [label="Target Result:\\nv(Queen)", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        King -> MinusMan -> RoyaltyDelta -> PlusWoman -> Queen;
    }
    """
    st.graphviz_chart(analogy_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Vector Operations Engine in Python", divider="rainbow")
st.markdown("""
The following Python class implements vector addition, subtraction, scalar scaling, batch matrix similarity, and semantic analogy arithmetic.
""")

class ProductionVectorOperationsEngine:
    """
    Production linear algebra engine for dense embeddings:
    Vector arithmetic, analogy solving, and batch matrix-vector projection.
    """
    def __init__(self):
        # Seeded vocabulary coordinate bank (6 dimensions)
        np.random.seed(101)
        self.vocab = {
            "king": np.array([0.90, 0.85, 0.10, 0.05, 0.20, 0.80]),
            "man": np.array([0.15, 0.82, 0.08, 0.04, 0.18, 0.12]),
            "woman": np.array([0.18, 0.12, 0.85, 0.06, 0.22, 0.15]),
            "queen": np.array([0.92, 0.14, 0.88, 0.04, 0.21, 0.82]),
            "paris": np.array([0.10, 0.20, 0.15, 0.92, 0.85, 0.10]),
            "france": np.array([0.08, 0.18, 0.12, 0.88, 0.40, 0.12]),
            "tokyo": np.array([0.12, 0.22, 0.14, 0.35, 0.90, 0.88]),
            "japan": np.array([0.10, 0.19, 0.11, 0.30, 0.45, 0.89])
        }
        # Normalize all vectors to unit length
        for k in self.vocab:
            self.vocab[k] = self.vocab[k] / np.linalg.norm(self.vocab[k])

    def solve_analogy(self, word_a: str, word_b: str, word_c: str) -> List[Tuple[str, float]]:
        """
        Solves the analogy: A is to B as C is to [Target]
        Formula: Target Vector = Vector(B) - Vector(A) + Vector(C)
        """
        v_a = self.vocab[word_a]
        v_b = self.vocab[word_b]
        v_c = self.vocab[word_c]

        # Target direction calculation
        target_vec = (v_b - v_a) + v_c
        target_norm = target_vec / np.linalg.norm(target_vec)

        # Rank all candidates by cosine similarity
        candidates = []
        for word, vec in self.vocab.items():
            if word in [word_a, word_b, word_c]:
                continue # Exclude input words
            sim = float(np.dot(target_norm, vec))
            candidates.append((word, round(sim, 4)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    @staticmethod
    def batch_matrix_similarity(doc_matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity against N documents in a single matrix-vector operation.
        Matrix shape: (N, D), Query shape: (D,) -> Output shape: (N,)
        """
        q_unit = query_vec / np.linalg.norm(query_vec)
        # Assuming doc_matrix rows are pre-normalized
        scores = np.dot(doc_matrix, q_unit)
        return scores

code_snippet = r'''
import numpy as np

class VectorOperations:
    @staticmethod
    def analogy(v_a: np.ndarray, v_b: np.ndarray, v_c: np.ndarray) -> np.ndarray:
        """Computes target vector for: A is to B as C is to Target."""
        target = (v_b - v_a) + v_c
        return target / np.linalg.norm(target)

    @staticmethod
    def batch_search(corpus_matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        """Matrix-vector product computes all N similarity scores in parallel."""
        q_norm = query_vector / np.linalg.norm(query_vector)
        return np.dot(corpus_matrix, q_norm)
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Analogy & Matrix Multiplication Workbench
st.header("3. Interactive Vector Algebra & Batch Matrix Workbench", divider="rainbow")
st.markdown("""
Experiment with **Semantic Analogy Algebra** (`King - Man + Woman = ?`) or test **Batch Matrix Multiplication** to see how vector databases evaluate thousands of candidates simultaneously.
""")

ops_engine = ProductionVectorOperationsEngine()

tab_alg1, tab_alg2 = st.tabs(["1. Semantic Analogy Algebra", "2. Batch Matrix Dot Product"])

with tab_alg1:
    st.markdown("### Proportional Analogy Calculator")
    st.markdown("Formula: **Target = Vector(B) - Vector(A) + Vector(C)**")

    preset_analogies = {
        "Man is to King as Woman is to [ ? ]": ("man", "king", "woman"),
        "France is to Paris as Japan is to [ ? ]": ("france", "paris", "japan")
    }

    selected_analogy = st.selectbox("Select analogy scenario:", list(preset_analogies.keys()))
    w_a, w_b, w_c = preset_analogies[selected_analogy]

    if st.button("Compute Analogy Vector Arithmetic", type="primary"):
        ranked_targets = ops_engine.solve_analogy(w_a, w_b, w_c)
        top_word, top_score = ranked_targets[0]

        st.success(f"Top Target Prediction: **'{top_word.upper()}'** (Cosine Proximity: {top_score:.4f})")
        
        st.markdown("##### Candidate Ranking Results:")
        res_df = pd.DataFrame([
            {"Rank": i+1, "Candidate Word": word, "Cosine Similarity": f"{score:.4f}"}
            for i, (word, score) in enumerate(ranked_targets)
        ])
        st.dataframe(res_df, use_container_width=True)

with tab_alg2:
    st.markdown("### Batch Matrix Multiplication (BLAS Acceleration)")
    st.markdown("""
    When searching 10,000 documents, looping through vectors sequentially in Python is 100x too slow. 
    Instead, documents are stacked into an (N x D) matrix M, and scores are evaluated in a single matrix-vector dot product M * q.
    """)

    col_mat1, col_mat2 = st.columns([1, 1])
    with col_mat1:
        num_docs = st.slider("Simulated Document Count (N):", min_value=100, max_value=5000, value=1000, step=100)
    with col_mat2:
        vector_dim = st.selectbox("Vector Dimensions (D):", [128, 384, 768, 1536], index=1)

    # Generate synthetic matrix and query
    np.random.seed(42)
    synthetic_matrix = np.random.randn(num_docs, vector_dim)
    # Row normalize
    synthetic_matrix /= np.linalg.norm(synthetic_matrix, axis=1, keepdims=True)
    synthetic_query = np.random.randn(vector_dim)

    scores = ops_engine.batch_matrix_similarity(synthetic_matrix, synthetic_query)
    top_indices = np.argsort(scores)[::-1][:5]

    st.markdown(f"##### Evaluated {num_docs:,} Documents across {vector_dim} Dimensions in a Single Matrix Product:")
    batch_res_df = pd.DataFrame([
        {"Rank": i+1, "Document Row Index": f"Doc_{idx}", "Cosine Score": f"{scores[idx]:.4f}"}
        for i, idx in enumerate(top_indices)
    ])
    st.dataframe(batch_res_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Operations Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does vector arithmetic work for semantic analogies (e.g. King - Man + Woman = Queen)?**
    * **Answer**:
      * In word embedding models (such as GloVe and Word2Vec), co-occurrence statistics are mapped to linear substructures in vector space.
      * The vector difference v(king) - v(man) isolates a directional displacement vector representing the semantic concept of **"Royalty"**.
      * When this royalty vector is added to v(woman), it translates the feminine coordinate along the royalty axis, landing nearest to the coordinate for v(queen).
      * This linear regularity emerges because the objective function encourages differences between word vectors to encode the log odds-ratio of their context words.

    **Q2: What is the computational complexity of brute-force matrix-vector dot product search?**
    * **Answer**:
      * For a corpus of N documents each of dimension D, computing dot products against a single query vector requires N * D multiplications and additions.
      * **Complexity**: O(N * D) time complexity and O(N * D) memory footprint.
      * For N = 10,000,000 and D = 1536, a single query requires 15.36 billion floating-point operations. This is why Approximate Nearest Neighbor (ANN) indexing (HNSW, IVF) is required for billion-scale search.

    **Q3: Why is Mean Pooling generally preferred over Max Pooling for sentence embeddings?**
    * **Answer**:
      * **Mean Pooling**: Averages all token vectors across the sequence length. Every token contributes proportionally to the final representation, capturing the overall topical distribution.
      * **Max Pooling**: Takes the maximum value along each dimension across all tokens. While it captures extreme salient keywords, it discards subtle syntactical qualifiers, negations, and transitional context.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting to Re-Normalize after Vector Addition/Subtraction**:
      * *Issue*: Computing v_target = v_B - v_A + v_C and immediately taking dot products without re-normalizing.
      * *Consequence*: The magnitude of the resulting vector is distorted, invalidating cosine similarity comparisons.
      * *Fix*: Always divide the target vector by its L2 norm: v_target = v_target / norm(v_target).

    * **Trap 2: Assuming Word2Vec-style Analogy Arithmetic Always Works in Transformer Sentence Encoders**:
      * *Issue*: Attempting to perform single-word analogy subtraction on modern sentence-transformer embeddings.
      * *Consequence*: Sentence transformers are optimized for whole-sentence semantic similarity via contrastive loss, not isolated word algebra; results can be noisy.
      * *Fix*: Use sentence transformers for whole-sequence comparison and dedicated word vectors (GloVe/FastText) for subword arithmetic.

    * **Trap 3: Sequential Looping Over Documents in Python**:
      * *Issue*: Writing a `for doc in documents:` loop to compute cosine similarity in pure Python.
      * *Consequence*: Execution takes seconds instead of milliseconds.
      * *Fix*: Stack documents into a single 2D NumPy array and use vectorized `np.dot()`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Mean Pooling**
      * *Definition*: The mathematical process of averaging token embedding vectors along the sequence dimension to create a single fixed-length sentence vector.
    * **Flashcard 2: Semantic Delta Vector**
      * *Definition*: The directional difference vector (v_B - v_A) capturing the conceptual transition or relationship connecting concept A to concept B.
    * **Flashcard 3: Batch Matrix Search**
      * *Definition*: Computing similarity against an entire corpus simultaneously using optimized matrix-vector multiplication (M . q).
    * **Flashcard 4: O(N . D) Complexity**
      * *Definition*: The linear computational complexity of brute-force nearest neighbor search across N documents of dimension D.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Vector Operations:

- [ ] I can describe the mathematical mechanics of vector addition, subtraction, and scalar multiplication.
- [ ] I understand why analogy arithmetic (King - Man + Woman = Queen) emerges in linear vector spaces.
- [ ] I know how Mean Pooling aggregates token embeddings into a single sentence vector.
- [ ] I can implement batch matrix-vector dot product search in Python/NumPy.
- [ ] I understand the O(N * D) time complexity of brute-force search and why ANN indexing is needed at scale.
""")
