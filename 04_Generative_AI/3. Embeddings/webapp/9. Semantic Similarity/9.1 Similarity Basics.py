import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="9.1 Similarity Basics - Embeddings Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("9.1 Semantic Similarity Basics: Lexical vs. Dense Matching, Synonymy, & Polysemy")
st.caption("Chapter 9: Semantic Similarity | Domain: 3. Vector Embeddings")
st.markdown("""
For decades, search systems relied on **Lexical Matching** (exact keyword matching, TF-IDF, BM25).
If a user searched for *"automobile insurance"*, a lexical search engine looked for documents containing the exact tokens `"automobile"` and `"insurance"`.
If an authoritative document instead used the words *"car policy"*, lexical search scored it as **zero**!

**Semantic Similarity** measures similarity in **meaning, conceptual intent, and proposition**, completely independent of the specific surface words chosen.

Semantic similarity resolves the two classical banes of Information Retrieval:
1. **Synonymy (Different words, identical meaning)**:
   *"The physician examined the ailing child"* and *"The doctor treated the sick kid"* share **zero vocabulary overlap**, yet dense embeddings place them within 0.05 angular radians of each other!
2. **Polysemy (Identical words, completely different meanings)**:
   *"Deposit money into the bank"* vs. *"Walk along the river bank"*. Lexical search treats `"bank"` identically; dense contextual embeddings place them in completely orthogonal coordinates!

This lesson explores the foundations of semantic similarity versus lexical overlap.
""")

st.divider()

# Section 1: Lexical vs. Semantic Similarity
st.header("1. Lexical Matching vs. Dense Semantic Geometry", divider="rainbow")

col_sem1, col_sem2 = st.columns([1, 1])

with col_sem1:
    st.markdown("""
    ### Why Lexical Overlap Fails Modern Search:
    * **1. Vocabulary Mismatch (Zero-Keyword Trap)**:
      * Users rarely use the exact vocabulary of the authors.
      * Lexical Jaccard similarity between `"purchase footwear"` and `"buy shoes"` is **0.00** (0 shared words).
      * Dense semantic cosine similarity is **0.95+**!
    * **2. Word Order Insensitivity (Bag-of-Words Trap)**:
      * Lexical search counts word frequency regardless of syntax:
        * Sentence A: *"The dog bit the man."*
        * Sentence B: *"The man bit the dog."*
      * Jaccard similarity is **1.00** (100% lexical match), yet the real-world semantics are opposite!
      * Dense contextual models leverage self-attention to encode grammatical role, distinguishing the subject from the object.
    * **3. Negation Blindness**:
      * Sentence A: *"This software is secure."*
      * Sentence B: *"This software is NOT secure."*
      * Lexical overlap is **80%**, but contextual embeddings capture the critical negation flip.
    """)

with col_sem2:
    st.markdown("### Lexical vs. Semantic Space Topology")
    sim_dot = """
    digraph SemanticSpace {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_lexical {
            label="Lexical Space (Discrete Tokens)"; style="dashed"; color="#EF4444";
            L1 [label="'doctor' (Token ID 451)", fillcolor="#FEE2E2"];
            L2 [label="'physician' (Token ID 8923)", fillcolor="#FEE2E2"];
            L1 -> L2 [label="Orthogonal\\nDistance = Infinity", dir=none, style=dotted];
        }

        subgraph cluster_dense {
            label="Dense Latent Space (Continuous Hypersphere)"; style="dashed"; color="#10B981";
            D1 [label="vec('doctor')\\n[0.82, 0.15, ...]", fillcolor="#ECFDF5"];
            D2 [label="vec('physician')\\n[0.81, 0.17, ...]", fillcolor="#D1FAE5"];
            D1 -> D2 [label="Cosine Sim = 0.96\\nAngular Delta < 5 deg", color="#059669", penwidth=2];
        }

        Query [label="User Query: 'pediatric medical care'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Query -> L1 [label="No token match!"];
        Query -> D1 [label="High Cosine Proximity"];
    }
    """
    st.graphviz_chart(sim_dot, use_container_width=False)

st.markdown("### Feature Matrix: Lexical vs. Dense Semantic Matching")
st.markdown("""
| Dimension | Lexical Matching (BM25 / TF-IDF) | Dense Semantic Embeddings (SBERT / OpenAI) |
| :--- | :--- | :--- |
| **Representation** | High-dimensional sparse vector (Vocab size ~100k+) | Low-dimensional continuous vector (d = 384 to 3072) |
| **Synonym Handling** | Fails (requires manual synonym expansion dictionaries) | **Native** (synonyms cluster to near-identical coordinates) |
| **Polysemy Handling** | Fails (collapses all meanings of `"bank"` into 1 token) | **Native** (contextual attention separates meanings) |
| **Word Order / Grammar** | Ignored (Bag-of-words assumption) | **Preserved** (directional self-attention matrices) |
| **Rare Keyword Matching** | **Superior** (matches exact part numbers, GUIDs, codes) | Vulnerable to hallucinated or generalized proximity |
| **Recommended Architecture** | Hybrid Search (BM25 for keywords + Dense for semantics) | Hybrid Search (Combined via Reciprocal Rank Fusion) |
""")

st.divider()

# Section 2: Production Code: Lexical vs. Semantic Similarity Benchmark
st.header("2. Production Implementation: Dual-Mode Similarity Comparator", divider="rainbow")

st.markdown("""
The production class below computes both **Lexical Jaccard Similarity** and **Dense Cosine Similarity**, allowing engineering teams to audit where lexical search fails and quantify the semantic gap.
""")

st.code(r'''
import re
import numpy as np
from typing import Dict, Any, Set

class DualModeSimilarityComparator:
    """
    Compares lexical token overlap (Jaccard) against dense embedding cosine similarity.
    Exposes vocabulary mismatch and semantic alignment metrics.
    """
    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Computes lexical token set intersection over union."""
        tokens_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        tokens_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection / union)

    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes dot product between two unit-normalized vectors."""
        norm_a = vec_a / np.maximum(np.linalg.norm(vec_a), 1e-12)
        norm_b = vec_b / np.maximum(np.linalg.norm(vec_b), 1e-12)
        return float(np.dot(norm_a, norm_b))
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveSimilaritySimulator:
    def __init__(self):
        # 5 latent concept dimensions: [Medical/Health, Legal/Law, Finance, Canine/Pet, Geography/Nature]
        self.pairs = {
            "Zero Word Overlap Synonyms": {
                "text_a": "The physician examined the ailing child in the clinic.",
                "text_b": "The doctor treated the sick kid at the hospital.",
                "vec_a": np.array([0.95, 0.02, 0.01, 0.01, 0.02]),
                "vec_b": np.array([0.94, 0.01, 0.02, 0.01, 0.03]),
                "explanation": "0% lexical overlap, but virtually identical medical meaning!"
            },
            "High Word Overlap Opposite Meaning": {
                "text_a": "The dog bit the aggressive mailman in the yard.",
                "text_b": "The mailman bit the aggressive dog in the yard.",
                # Subject/object flip alters semantics
                "vec_a": np.array([0.10, 0.40, 0.01, 0.90, 0.05]),
                "vec_b": np.array([0.15, 0.70, 0.01, 0.45, 0.05]),
                "explanation": "88% lexical overlap, but grammar inversion changes the meaning!"
            },
            "Polysemy Disambiguation ('Bank')": {
                "text_a": "Deposit cash and withdraw currency at the downtown bank.",
                "text_b": "Fish for trout and pitch a tent along the river bank.",
                "vec_a": np.array([0.01, 0.05, 0.98, 0.01, 0.02]),
                "vec_b": np.array([0.01, 0.02, 0.05, 0.05, 0.97]),
                "explanation": "Share the word 'bank', but finance vs outdoor nature are completely orthogonal!"
            },
            "Negation Flip": {
                "text_a": "The experimental drug is proven safe for human trials.",
                "text_b": "The experimental drug is NOT safe for human trials.",
                "vec_a": np.array([0.92, 0.15, 0.05, 0.01, 0.01]),
                "vec_b": np.array([0.25, 0.85, 0.02, 0.01, 0.01]),
                "explanation": "High lexical overlap, but 'NOT' changes the clinical decision!"
            }
        }

    def compute_jaccard(self, text_a: str, text_b: str) -> float:
        import re
        tokens_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        tokens_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return float(len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b)))

    def compute_cosine(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = vec_a / np.linalg.norm(vec_a)
        norm_b = vec_b / np.linalg.norm(vec_b)
        return float(np.dot(norm_a, norm_b))

st.divider()

# Section 3: Interactive Workbench: Lexical vs. Semantic Comparator
st.header("3. Interactive Workbench: Lexical vs. Semantic Similarity Comparator", divider="rainbow")

st.markdown("""
Select a benchmark linguistic challenge to inspect the fundamental difference between **Lexical Jaccard Overlap** and **Dense Semantic Cosine Similarity**.
Observe how dense embeddings gracefully handle synonyms and polysemy where keyword matching fails!
""")

sim_bench = InteractiveSimilaritySimulator()

col_pair_l, col_pair_r = st.columns([1, 1])

with col_pair_l:
    st.subheader("1. Select Linguistic Challenge")
    selected_pair_key = st.selectbox(
        "Choose Comparison Scenario:",
        list(sim_bench.pairs.keys())
    )

    pair_data = sim_bench.pairs[selected_pair_key]
    
    st.markdown("#### Sentence Pair:")
    st.info(f"**Sentence A**: {pair_data['text_a']}")
    st.info(f"**Sentence B**: {pair_data['text_b']}")
    st.caption(f"💡 **Analysis**: {pair_data['explanation']}")

with col_pair_r:
    st.subheader("2. Metric Comparison")
    jaccard_score = sim_bench.compute_jaccard(pair_data["text_a"], pair_data["text_b"])
    cosine_score = sim_bench.compute_cosine(pair_data["vec_a"], pair_data["vec_b"])

    m1, m2 = st.columns(2)
    m1.metric("Lexical Jaccard Similarity", f"{jaccard_score:.4f}")
    m2.metric("Dense Cosine Similarity", f"{cosine_score:.4f}")

    divergence = abs(cosine_score - jaccard_score)
    st.metric("Absolute Divergence (|Cosine - Jaccard|)", f"{divergence:.4f}")

    st.markdown("#### Diagnostic Summary")
    df_diag = pd.DataFrame({
        "Metric": ["Lexical Jaccard Overlap", "Dense Semantic Cosine Similarity", "Divergence Magnitude"],
        "Value": [f"{jaccard_score:.4f}", f"{cosine_score:.4f}", f"{divergence:.4f}"],
        "Interpretation": [
            "Exact token set overlap" if jaccard_score > 0 else "Zero common words",
            "High semantic alignment" if cosine_score > 0.8 else "Distinct or opposing concepts",
            "Strong semantic abstraction" if divergence > 0.4 else "Aligned lexical and semantic agreement"
        ]
    })
    st.dataframe(df_diag, use_container_width=True)

    if selected_pair_key == "Zero Word Overlap Synonyms":
        st.success("🎯 **Dense Superpower**: Lexical search is completely blind (Jaccard = 0.00), while dense embeddings correctly score near-perfect alignment (Cosine = 0.99)!")
    elif selected_pair_key == "High Word Overlap Opposite Meaning":
        st.warning("⚠️ **Grammar Sensitivity**: Lexical search gives near-perfect score due to identical words, but dense embeddings recognize the reversal of actor roles!")
    elif selected_pair_key == "Polysemy Disambiguation ('Bank')":
        st.info("🏦 **Disambiguation**: The shared word 'bank' does not fool dense embeddings; contextual attention separates financial banking from river banks.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Similarity Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the fundamental mathematical difference between Lexical Similarity (Jaccard/BM25) and Semantic Similarity (Cosine on Embeddings)?**
    * **Answer**:
      * **Lexical Similarity**: Measures surface token co-occurrence in a discrete vocabulary space.
        Jaccard similarity measures the intersection over union of token sets: |Intersection(A, B)| / |Union(A, B)|.
        If words are synonyms (e.g. `car` and `automobile`), the intersection is 0.
      * **Semantic Similarity**: Measures geometric proximity (angle theta) between continuous vectors on a unit hypersphere:
        `Cosine = (u dot v) / (||u|| * ||v||)`.
        During pre-training, synonyms are forced into adjacent coordinate positions because they appear in interchangeable contexts, yielding high cosine similarity regardless of spelling.

    **Q2: What is the 'Polysemy Problem' in search, and how do Transformer embeddings solve it?**
    * **Answer**:
      * Polysemy occurs when a single word possesses multiple distinct meanings (e.g. *"Apple"* the company vs the fruit; *"Python"* the snake vs programming language).
      * Classical static embeddings (Word2Vec) assigned a **single fixed vector** to each word, forcing all meanings into an awkward average.
      * Modern Transformer encoders (BERT, OpenAI) compute **dynamic contextual embeddings**: the self-attention mechanism updates the word vector based on its neighboring words.
      * In *"Python language syntax"*, the attention heads pull the token vector toward computing concepts.
      * In *"Python swallowed the rodent"*, the attention heads pull it toward zoological concepts.

    **Q3: When does Lexical Search (BM25) outperform Dense Semantic Embeddings?**
    * **Answer**:
      * 1. **Exact Identifier Search**: Part numbers (`Part-TX89-Z`), UUIDs, git commit hashes, error codes (`ERR_404_TIMEOUT`). Dense models often generalize codes into nearby generic categories.
      * 2. **Rare Out-of-Distribution Proper Nouns**: Obscure legal names, new medication brand names, or slang never seen in pretraining.
      * 3. **Keyword-Heavy Auditing**: Legal discovery where lawyers search for an exact contractual phrase.
      * *Production Solution*: Deploy **Hybrid Search** (BM25 + Dense Vectors) fused via Reciprocal Rank Fusion (RRF).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Abandoning Lexical Search Entirely**:
      * *Issue*: Replacing BM25 with pure dense vector search for all enterprise queries.
      * *Consequence*: Users searching for exact product IDs, error codes, or rare acronyms receive completely irrelevant generalized semantic matches.
      * *Fix*: Implement Hybrid Search combining BM25 keyword matching with dense semantic retrieval.

    * **Trap 2: Assuming High Cosine Similarity Always Implies Factual Agreement**:
      * *Issue*: Assuming that a cosine similarity of 0.92 between a query and document means the document answers the query affirmatively.
      * *Consequence*: A query (*"Is medication X approved for children?"*) and a passage (*"Medication X is NOT approved for children"*) share massive topical similarity (0.85+), but have opposing truth values.
      * *Fix*: Use embeddings for retrieval only; leave logical reasoning and factual verification to the generator LLM or cross-encoder reranker.

    * **Trap 3: Comparing Vectors from Different Embedding Models**:
      * *Issue*: Computing cosine similarity between a vector from OpenAI `text-embedding-3-small` and a vector from `all-MiniLM-L6-v2`.
      * *Consequence*: Coordinate systems are completely orthogonal and incompatible; similarity scores are meaningless random noise.
      * *Fix*: Enforce model registry checks to ensure query and document vectors originate from the identical model checkpoint.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Lexical Similarity**
      * *Definition*: The degree of exact token or keyword overlap between two texts, regardless of semantic meaning.
    * **Flashcard 2: Semantic Similarity**
      * *Definition*: The degree of conceptual alignment and shared meaning between two texts in a continuous vector space.
    * **Flashcard 3: Synonymy**
      * *Definition*: Different words conveying the same concept (physician / doctor), easily handled by dense embeddings but missed by lexical search.
    * **Flashcard 4: Polysemy**
      * *Definition*: A single word possessing multiple meanings depending on context (bank / python), resolved dynamically by transformer self-attention.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Semantic Similarity Basics:

- [ ] I can articulate the fundamental differences between Lexical (BM25/Jaccard) and Dense Semantic matching.
- [ ] I understand how dense embeddings solve the classical Synonymy problem.
- [ ] I can explain how transformer self-attention resolves the Polysemy problem dynamically.
- [ ] I know the scenarios where BM25 outperforms dense embeddings (exact IDs, part numbers).
- [ ] I understand why hybrid search (BM25 + Dense) is the industry standard for production retrieval.
""")
