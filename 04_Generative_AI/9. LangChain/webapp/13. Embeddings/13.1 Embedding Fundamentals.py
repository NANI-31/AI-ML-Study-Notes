import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class SemanticVectorItem:
    """Represents a simulated text item in dense geometric space."""
    def __init__(self, phrase: str, category: str, vector_coords: np.ndarray):
        self.phrase = phrase
        self.category = category
        self.coords = vector_coords  # 3D representation for visualization

class MockSemanticSpaceSimulator:
    """Simulates semantic vector space geometry, dimensionality, and distance calculations."""
    def __init__(self):
        # 3D coordinates representing semantic clusters: Technology, Finance, Culinary
        self.corpus = [
            SemanticVectorItem("Kubernetes cluster autoscaling", "Technology", np.array([0.92, 0.15, 0.08])),
            SemanticVectorItem("Docker container networking", "Technology", np.array([0.88, 0.20, 0.12])),
            SemanticVectorItem("Python asynchronous event loop", "Technology", np.array([0.84, 0.25, 0.15])),
            SemanticVectorItem("Corporate balance sheet audit", "Finance", np.array([0.18, 0.91, 0.12])),
            SemanticVectorItem("Quarterly EBITDA margin growth", "Finance", np.array([0.12, 0.94, 0.10])),
            SemanticVectorItem("Federal Reserve interest rate hike", "Finance", np.array([0.22, 0.86, 0.18])),
            SemanticVectorItem("Artisanal sourdough bread fermentation", "Culinary", np.array([0.10, 0.14, 0.93])),
            SemanticVectorItem("Truffle infused risotto preparation", "Culinary", np.array([0.15, 0.11, 0.95])),
            SemanticVectorItem("Espresso bean roasting temperature", "Culinary", np.array([0.08, 0.19, 0.89]))
        ]

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def project_query(self, query: str) -> np.ndarray:
        q_lower = query.lower()
        # Heuristic mapping for interactive demonstration
        tech_score = 0.1
        fin_score = 0.1
        food_score = 0.1

        if any(w in q_lower for w in ["k8s", "docker", "server", "code", "devops", "cloud", "api"]):
            tech_score += 0.8
        if any(w in q_lower for w in ["money", "profit", "bank", "invest", "revenue", "ebitda", "tax"]):
            fin_score += 0.8
        if any(w in q_lower for w in ["recipe", "bake", "cook", "coffee", "food", "kitchen", "eat"]):
            food_score += 0.8

        return self._normalize(np.array([tech_score, fin_score, food_score]))

    def search_similar(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        results = []
        for item in self.corpus:
            norm_item = self._normalize(item.coords)
            cos_sim = float(np.dot(query_vector, norm_item))
            euc_dist = float(np.linalg.norm(query_vector - norm_item))
            results.append({
                "phrase": item.phrase,
                "category": item.category,
                "coords": item.coords.tolist(),
                "cosine_similarity": round(cos_sim, 4),
                "euclidean_distance": round(euc_dist, 4)
            })
        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return results[:top_k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="13.1 Embedding Fundamentals",
    page_icon="🧭",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">13.1 Embedding Fundamentals</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Vector spaces, dense geometric representations, embedding dimensions, and the mathematical mechanics of semantic similarity.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Semantic Vector Space & Dense Geometric Representation", divider="rainbow")

st.markdown("""
Machine learning models cannot process words directly as qualitative symbols.
**Embeddings** map discrete, human-readable language (words, sentences, documents) into continuous, 
dense vectors of floating-point numbers in a high-dimensional geometric space.
In this learned geometric manifold, **semantic meaning is encoded as spatial proximity**: 
texts with similar meanings reside close together, while unrelated texts are far apart.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EmbeddingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_unstructured {
            label="1. Unstructured Discrete Text";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            TextA [label="Phrase A: 'Kubernetes cluster deployment'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            TextB [label="Phrase B: 'Docker container rollout'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            TextC [label="Phrase C: 'Baking artisanal sourdough bread'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_transformer {
            label="2. Transformer Neural Encoder";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            Encoder [label="Dense Embedding Neural Model\\n(BERT / text-embedding-3 / BGE / Mistral-Embed)\\nSelf-Attention Token Contextualization + Mean Pooling", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
        }

        subgraph cluster_dense_vectors {
            label="3. Continuous Dense Vectors (d = 1536 Dimensions)";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            VecA [label="Vector A: [0.042, -0.198, 0.812, ..., -0.015]", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            VecB [label="Vector B: [0.045, -0.185, 0.795, ..., -0.012]", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            VecC [label="Vector C: [-0.612, 0.441, -0.210, ..., 0.384]", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_geometric_space {
            label="4. Multi-Dimensional Manifold";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            ClusterTech [label="Cloud DevOps Cluster\\nHigh Cosine Similarity (~0.92)\\nSmall Angular Distance (theta = 23 deg)", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            ClusterFood [label="Culinary Cluster\\nLow Cosine Similarity (~0.12)\\nNear Orthogonal Angle (theta = 83 deg)", fillcolor="#0F172A", fontcolor="#94A3B8", color="#64748B"];
        }

        TextA -> Encoder [label="Tokenize A"];
        TextB -> Encoder [label="Tokenize B"];
        TextC -> Encoder [label="Tokenize C"];

        Encoder -> VecA [label="Forward pass"];
        Encoder -> VecB [label="Forward pass"];
        Encoder -> VecC [label="Forward pass"];

        VecA -> ClusterTech [label="Spatial neighbor"];
        VecB -> ClusterTech [label="Spatial neighbor"];
        VecC -> ClusterFood [label="Distant cluster"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT ARE EMBEDDINGS? ---
st.header("2. What are Embeddings?", divider="rainbow")

st.markdown("""
In traditional computer science, computers compared text via exact keyword matching (e.g. SQL `LIKE '%apple%'` or RegEx).
This approach completely fails when a user searches for *"canine doctor"* and the database contains *"veterinarian clinic"*: 
zero words overlap, yet the semantic intent is nearly identical.

An **embedding** is a translation function that maps discrete symbolic tokens into a continuous geometric vector space:
- Every concept becomes a coordinate in multi-dimensional space.
- Similar concepts point in similar directions.
- Arithmetic relationships emerge naturally (e.g. `Vector("King") - Vector("Man") + Vector("Woman") approx Vector("Queen")`).
""")

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.markdown("#### ❌ Lexical / Keyword Matching (Sparse)")
    st.markdown("""
    - Treats words as independent atomic symbols.
    - Matches only exact spelling and string intersections.
    - Completely blind to synonyms, analogies, and typos.
    - Represented as sparse high-dimensional arrays (e.g. One-Hot, TF-IDF).
    """)
with col_e2:
    st.markdown("#### ✅ Semantic Embeddings (Dense)")
    st.markdown("""
    - Captures deep contextual nuance and topical themes.
    - Matches queries across disparate vocabularies and multilingual boundaries.
    - Tolerates typos, slang, and phrasing variations.
    - Represented as compact dense floating-point arrays (e.g. 768 or 1536 floats).
    """)

st.divider()

# --- SECTION 3: TEXT EMBEDDINGS ---
st.header("3. Text Embeddings", divider="rainbow")

st.markdown("""
Modern text embeddings are generated using pre-trained Transformer encoder backbones (such as BERT, RoBERTa, or modern decoder-based models).

### Generation Workflow:
1. **Tokenization**: Text is parsed into subword tokens (Byte-Pair Encoding / WordPiece).
2. **Contextual Self-Attention**: Each token attends to all neighboring tokens across multiple transformer layers.
3. **Pooling**: Token vectors are condensed into a single fixed-size sentence vector (typically via **Mean Pooling** across non-padding tokens).
4. **L2 Normalization**: The resulting vector is normalized to unit length (`||v|| = 1.0`), allowing rapid cosine similarity computation via simple dot products.
""")

st.code("""# Standard LangChain Text Embedding Interface
from langchain_core.embeddings import Embeddings

class CustomTextEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        \"\"\"Batch vectorization of text corpus for database indexing.\"\"\"
        return [self._compute_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        \"\"\"Vectorization of a single incoming search query.\"\"\"
        return self._compute_vector(text)""", language="python")

st.divider()

# --- SECTION 4: VECTOR REPRESENTATIONS ---
st.header("4. Vector Representations", divider="rainbow")

st.markdown("""
A vector is an ordered array of real numbers representing coordinates along orthogonal geometric axes.
In a 3-dimensional space, a vector is represented as `[x, y, z]`. 
In modern embedding models, vectors reside in 768, 1536, or 3072 dimensions, where each axis represents a latent semantic feature 
(e.g. formality, sentiment, technicality, temporality) learned automatically during pre-training.
""")

vec_df = pd.DataFrame([
    {"Concept Phrase": "Financial dividend payout", "Latent Feature 1 (Tech)": 0.05, "Latent Feature 2 (Finance)": 0.94, "Latent Feature 3 (Culinary)": 0.02, "Dominant Cluster": "Capital Markets"},
    {"Concept Phrase": "Quarterly balance sheet audit", "Latent Feature 1 (Tech)": 0.12, "Latent Feature 2 (Finance)": 0.91, "Latent Feature 3 (Culinary)": 0.04, "Dominant Cluster": "Capital Markets"},
    {"Concept Phrase": "Kubernetes pod orchestration", "Latent Feature 1 (Tech)": 0.96, "Latent Feature 2 (Finance)": 0.08, "Latent Feature 3 (Culinary)": 0.01, "Dominant Cluster": "Software DevOps"},
    {"Concept Phrase": "Baking whole wheat sourdough", "Latent Feature 1 (Tech)": 0.03, "Latent Feature 2 (Finance)": 0.02, "Latent Feature 3 (Culinary)": 0.97, "Dominant Cluster": "Culinary Gastronomy"}
])
st.dataframe(vec_df, use_container_width=True)

st.divider()

# --- SECTION 5: EMBEDDING DIMENSIONS ---
st.header("5. Embedding Dimensions", divider="rainbow")

st.markdown("""
The **dimensionality** (d) of an embedding determines how much semantic fidelity the vector can compress:
""")

dim_c1, dim_c2, dim_c3 = st.columns(3)
with dim_c1:
    st.subheader("🔹 384 - 768 Dimensions")
    st.markdown("""
    - Models: `all-MiniLM-L6-v2`, `bge-small`
    - RAM per 1M Vectors: ~1.5 GB to 3.0 GB
    - Ideal for edge devices, ultra-low-latency real-time apps, and high-concurrency search.
    """)
with dim_c2:
    st.subheader("🔹 1536 Dimensions")
    st.markdown("""
    - Models: OpenAI `text-embedding-3-small`, `bge-large`
    - RAM per 1M Vectors: ~6.1 GB
    - The enterprise sweet spot balancing high semantic capture with cost-effective vector index RAM.
    """)
with dim_c3:
    st.subheader("🔹 3072+ Dimensions")
    st.markdown("""
    - Models: `text-embedding-3-large`, `cohere-embed-v3`
    - RAM per 1M Vectors: ~12.3+ GB
    - Maximum nuance for subtle legal, medical, and multi-lingual technical cross-referencing.
    """)

st.info("""
**Matryoshka Representation Learning (MRL)**: Modern models (like OpenAI `text-embedding-3`) are trained with nested dimensions. 
You can truncate a 3072D vector down to 512D or 256D by keeping only the first k floats, discarding 75% of storage space while retaining 98% of retrieval accuracy!
""")

st.divider()

# --- SECTION 6: SIMILARITY ---
st.header("6. Similarity", divider="rainbow")

st.markdown("""
Once text is converted into vectors, measuring semantic similarity is transformed into calculating **geometric distance**:
- **Cosine Similarity**: Measures the cosine of the angle between two vectors. Values range from -1.0 to +1.0 (for unit-normalized embeddings, values range from 0.0 to 1.0).
- **Euclidean Distance (L2)**: Measures the straight-line distance between vector endpoints in Euclidean space. Smaller values mean closer proximity.
- **Dot Product**: Multiplies corresponding coordinates. If vectors are L2-normalized, the Dot Product equals the Cosine Similarity exactly.
""")

st.divider()

# --- SECTION 7: SEMANTIC SIMILARITY ---
st.header("7. Semantic Similarity", divider="rainbow")

st.markdown("""
Semantic similarity differs from topical relatedness:
- **Lexically Identical, Semantically Opposite**: *"I love this product"* vs *"I do not love this product"*. (Share 80% lexical overlap, but represent opposite sentiment).
- **Lexically Distinct, Semantically Identical**: *"How old are you?"* vs *"What is your age?"*. (Share 0% lexical overlap, but represent the exact same inquiry).

Dense embedding models project queries and candidate documents into shared semantic space, enabling instant resolution of synonyms, intent, and paraphrasing.
""")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Dense Embedding Generation & Geometric Space Analysis", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates how raw text is mapped to unit-normalized vectors, 
and computes pairwise cosine similarity and Euclidean distance matrices using pure NumPy.
""")

st.code("""import numpy as np
from typing import List, Dict, Any

class DenseVectorSpace:
    \"\"\"Production vector geometry utility for semantic indexing.\"\"\"
    def __init__(self):
        pass

    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        \"\"\"Scales vector to unit Euclidean norm (L2 norm = 1.0).\"\"\"
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        \"\"\"Computes cosine similarity between two dense vectors.\"\"\"
        u1 = DenseVectorSpace.normalize_vector(v1)
        u2 = DenseVectorSpace.normalize_vector(v2)
        return float(np.dot(u1, u2))

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        \"\"\"Computes straight-line L2 distance between two points.\"\"\"
        return float(np.linalg.norm(v1 - v2))

    def rank_candidates(self, query_vec: np.ndarray, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        scored = []
        q_norm = self.normalize_vector(query_vec)

        for c in candidates:
            c_vec = np.array(c["vector"])
            c_norm = self.normalize_vector(c_vec)
            sim = float(np.dot(q_norm, c_norm))
            scored.append({
                "text": c["text"],
                "similarity": round(sim, 4),
                "distance": round(float(np.linalg.norm(q_norm - c_norm)), 4)
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]""", language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: 3D/2D Semantic Vector Space & Projection Visualizer", divider="rainbow")

st.markdown("""
Input a query to project it into a simulated 3-dimensional semantic manifold. 
Observe how proximity in geometric space correlates directly with conceptual meaning.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Query Vectorization")
    custom_query = st.text_input(
        "Enter Natural Language Query",
        value="How do I deploy a containerized application to AWS?"
    )

    top_k_val = st.slider("Max Nearest Neighbors (Top-K)", min_value=1, max_value=5, value=3)

    btn_calc = st.button("🚀 Calculate Vector Projections", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Geometric Search Results")

    if btn_calc:
        sim_engine = MockSemanticSpaceSimulator()
        q_vec = sim_engine.project_query(custom_query)
        matches = sim_engine.search_similar(q_vec, top_k=top_k_val)

        st.markdown(f"**Query Vector Coordinates (Normalized)**: `[{q_vec[0]:.2f}, {q_vec[1]:.2f}, {q_vec[2]:.2f}]`")

        for idx, m in enumerate(matches, 1):
            with st.expander(f"Neighbor #{idx}: {m['phrase']} ({m['category']})", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Cosine Similarity", m["cosine_similarity"])
                c2.metric("Euclidean Distance", m["euclidean_distance"])
                st.caption(f"Spatial Coordinates: `{m['coords']}`")
    else:
        st.info("Enter a query and click **Calculate Vector Projections** to compute semantic nearest neighbors.")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why does L2 normalization simplify vector similarity computation?",
     "Cosine similarity is defined as the dot product divided by the product of vector magnitudes: A dot B / (||A|| * ||B||). When all embedding vectors are pre-normalized to unit length (||A|| = 1.0 and ||B|| = 1.0), the denominator becomes exactly 1.0. Consequently, Cosine Similarity simplifies to a bare dot product (A dot B), which can be computed in ultra-fast SIMD hardware matrix operations without costly square roots and divisions."),
    ("Explain the difference between Sparse embeddings (e.g. TF-IDF / BM25) and Dense embeddings.",
     "Sparse embeddings have high dimensionality (often 50,000+ words in vocabulary) where almost all entries are zero. They match exact character tokens and are excellent for specific part numbers or rare names. Dense embeddings have fixed, compact dimensions (e.g. 768 or 1536 floats) where every coordinate is a non-zero real number. Dense vectors capture abstract concepts, synonyms, and semantic context."),
    ("What is Matryoshka Representation Learning (MRL), and why is it important in production?",
     "MRL trains an embedding model so that the most important semantic information is packed into the earliest vector dimensions. This allows engineers to truncate 3072-dimensional vectors down to 512 or 256 dimensions simply by slicing the array. Truncating vectors reduces vector database RAM and indexing costs by up to 80% while retaining over 95% of retrieval precision."),
    ("Why can't we use Euclidean Distance directly on unnormalized text vectors?",
     "Unnormalized vector lengths often correlate with document length (longer documents accumulate higher token magnitudes). If two documents share the exact same topic but one is 50 words and the other is 5,000 words, their Euclidean distance in unnormalized space will be massive, misleading the search. Normalizing to unit length eliminates length bias."),
    ("How does embedding model pooling work (CLS vs Mean Pooling)?",
     "Transformer encoders output a vector for every token in the sequence. To produce a single document-level vector, the model must pool them. CLS pooling takes the vector from the special [CLS] token at the beginning of the sequence. Mean pooling averages the output vectors of all valid tokens (weighted by an attention mask). Mean pooling consistently delivers higher retrieval accuracy for sentence and document search.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Mixing Embeddings from Different Models. Indexing documents with text-embedding-3-small (1536D) and querying with all-MiniLM-L6-v2 (384D) or even a different 1536D model causes garbage results. Different models define completely incompatible geometric coordinates. All vectors in an index must originate from the exact same model revision.")
    st.error("**Trap 2 (Breaking Error)**: Missing Input Text Length Truncation. Passing a 10,000-word chapter into an embedding model with an 8,192 token limit without chunking causes silent truncation or API 400 Bad Request errors. Always validate token length prior to embedding.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Re-embedding Unchanged Documents on Every Application Boot. Re-embedding static knowledge bases on startup wastes thousands of API dollars. Persist embeddings in disk-backed vector stores or cache them with LangChain's CacheBackedEmbeddings.")
    st.warning("**Trap 4 (Quality Warning)**: Embedding Questions and Documents Without Task Prefixes. Asymmetric embedding models (like E5 or BGE) require prefixing queries with 'query: ' and documents with 'passage: '. Omitting prefixes degrades retrieval ranking accuracy.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Dense Embedding"):
        st.markdown("**Continuous floating-point vector mapping semantic meaning into geometric spatial proximity.**")
    with st.expander("🎴 Flashcard 2: Unit Normalization"):
        st.markdown("**Scaling vectors to length 1.0 so cosine similarity simplifies to a fast dot product.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Matryoshka Truncation"):
        st.markdown("**Slicing vectors to fewer dimensions to save up to 80% RAM with minimal accuracy loss.**")
    with st.expander("🎴 Flashcard 4: Model Incompatibility"):
        st.markdown("**Vectors from different models occupy incompatible geometric coordinate spaces.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the mathematical definition and role of dense embeddings in modern AI.", value=True)
chk2 = st.checkbox("I can articulate the architectural difference between sparse lexical search and dense semantic search.", value=True)
chk3 = st.checkbox("I understand the trade-offs between 384, 1536, and 3072 embedding dimensions.", value=True)
chk4 = st.checkbox("I know how Matryoshka Representation Learning enables dimension truncation.", value=True)
chk5 = st.checkbox("I can compute cosine similarity and understand why L2 normalization simplifies dot products.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 embedding fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 13.1 Embedding Fundamentals!")
