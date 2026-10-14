import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="10.1 What is Vector Search - Embeddings Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("10.1 What is Vector Search: Search Evolution, Nearest Neighbors, & Semantic Geometry")
st.caption("Chapter 10: Vector Search | Domain: 3. Vector Embeddings")
st.markdown("""
For over 50 years, Information Retrieval relied on **lexical inverted indexes**: mapping words to document IDs.
While inverted indexes excel at keyword filtering, they are fundamentally blind to conceptual intent, synonyms, and multilingual semantics.

**Vector Search (also known as Dense Semantic Retrieval or Nearest Neighbor Search)** transforms retrieval from a discrete string-matching problem into a **geometric spatial lookup**:
1. Documents and user queries are embedded into a shared $d$-dimensional continuous vector space.
2. The user query is projected as a coordinate in that space.
3. The search engine finds the $K$ stored vectors that are geometrically closest to the query coordinates.

Whether finding matching customer support tickets, retrieving relevant passages for RAG, or recommending e-commerce items, vector search allows systems to understand **what users mean, rather than just what words they typed**.

This lesson explores the three generations of search technology, the nearest-neighbor formulation, and the architecture of modern vector search.
""")

st.divider()

# Section 1: The Three Generations of Search Technology
st.header("1. The Three Generations of Search Technology", divider="rainbow")

col_gen1, col_gen2 = st.columns([1, 1])

with col_gen1:
    st.markdown("""
    ### The Evolution of Search:
    * **Generation 1: Relational & Boolean Search (SQL / Grep)**:
      * Mechanism: `WHERE content LIKE '%heart attack%'`.
      * Limitation: Binary match (yes or no). No relevance ranking, zero synonym tolerance.
    * **Generation 2: Inverted Indexes & Lexical Scoring (BM25 / Lucene)**:
      * Mechanism: Postings lists mapping words to document frequencies and inverse document frequencies (TF-IDF).
      * Strength: Sub-millisecond keyword lookup; excellent for exact IDs and specific nouns.
      * Limitation: Fails on synonyms (*"myocardial infarction"* vs *"heart attack"*), polysemy, and conversational language.
    * **Generation 3: Dense Vector Search (k-NN / ANN)**:
      * Mechanism: Deep neural networks project text into continuous geometry; nearest-neighbor algorithms find angular proximity.
      * Strength: Understands concepts, handles multilingual queries, and aligns short queries with long passages.
      * Current Best Practice: **Hybrid Search** (combining Gen 2 BM25 with Gen 3 Dense Vectors).
    """)

with col_gen2:
    st.markdown("### The Three Generations of Search Architecture")
    gen_dot = """
    digraph SearchEvolution {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        G1 [label="Gen 1: Boolean SQL\\nLIKE '%term%'\\nBinary True/False | No Ranking", fillcolor="#FEE2E2", color="#EF4444"];
        G2 [label="Gen 2: Inverted Index (BM25)\\nTerm Frequencies | Fast Lexical", fillcolor="#FEF3C7", color="#F59E0B"];
        G3 [label="Gen 3: Dense Vector Search\\nGeometric Embeddings | Semantic Concepts", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        Hybrid [label="Production State-of-the-Art: Hybrid Search\\nBM25 (Keywords) + Vector (Concepts) + Reranker", fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];

        G1 -> G2 [label="Relevance ranking added"];
        G2 -> G3 [label="Semantic understanding added"];
        G2 -> Hybrid;
        G3 -> Hybrid;
    }
    """
    st.graphviz_chart(gen_dot, use_container_width=False)

st.markdown("### Comparative Matrix: Search Paradigms")
st.markdown("""
| Search Paradigm | Core Indexing Data Structure | Query Representation | Strengths | Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **Boolean SQL** | B-Tree / Sequential Scan | Exact substring / Regex | Deterministic, exact filtering | Zero ranking, vocabulary mismatch |
| **Lexical (BM25)** | Inverted Index (Postings List) | Bag of sparse token IDs | Blazing speed, exact part numbers | Synonym blindness, polysemy confusion |
| **Vector Search (Dense)** | Vector Graph (HNSW) / Trees (IVF) | Dense float array (d=384-1536) | Conceptual match, multilingual, paraphrase | Hallucinated proximity on rare serial numbers |
| **Hybrid Search** | Dual Index (Postings List + HNSW) | Sparse Bag + Dense Vector | Combines exact precision with semantic recall | Higher index RAM and compute footprint |
""")

st.divider()

# Section 2: Production Code: Multi-Paradigm Search Engine
st.header("2. Production Implementation: Multi-Paradigm Search Engine", divider="rainbow")

st.markdown("""
The production class below implements all three paradigms—**Boolean Substring**, **Lexical BM25-style Jaccard**, and **Dense Vector Dot Product**—demonstrating how different algorithms score the same corpus.
""")

st.code(r'''
import re
import numpy as np
from typing import List, Dict, Any

class MultiParadigmSearchEngine:
    """
    Demonstrates Boolean, Lexical, and Vector search paradigms over a shared corpus.
    """
    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus

    def search_boolean(self, query: str) -> List[Dict[str, Any]]:
        """Gen 1: Exact case-insensitive substring match."""
        q_clean = query.lower().strip()
        hits = []
        for doc in self.corpus:
            if q_clean in doc["text"].lower():
                hits.append({"id": doc["id"], "score": 1.0, "match_type": "Exact Substring", "text": doc["text"]})
        return hits

    def search_lexical_overlap(self, query: str) -> List[Dict[str, Any]]:
        """Gen 2: Token-overlap Jaccard score."""
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        hits = []
        for doc in self.corpus:
            d_tokens = set(re.findall(r'\b\w+\b', doc["text"].lower()))
            if not q_tokens or not d_tokens:
                score = 0.0
            else:
                score = len(q_tokens.intersection(d_tokens)) / len(q_tokens.union(d_tokens))
            if score > 0.0:
                hits.append({"id": doc["id"], "score": float(score), "match_type": "Lexical Overlap", "text": doc["text"]})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    def search_dense_vector(self, query_vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Gen 3: Normalized dot-product cosine similarity."""
        q_norm = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-12)
        hits = []
        for doc in self.corpus:
            d_norm = doc["vector"] / np.maximum(np.linalg.norm(doc["vector"]), 1e-12)
            score = float(np.dot(q_norm, d_norm))
            hits.append({"id": doc["id"], "score": score, "match_type": "Dense Vector", "text": doc["text"]})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]
''', language="python")

# Actual Python implementation for the interactive workbench
class InteractiveParadigmWorkbench:
    def __init__(self):
        # Sample technical knowledge base corpus with mock 4D vectors
        # Coordinates: [Cardiology/Medical, Database/SQL, Astronomy/Space, Culinary/Food]
        self.corpus = [
            {
                "id": "KB_001",
                "text": "Myocardial infarction occurs when coronary blood flow is abruptly occluded.",
                "vector": np.array([0.96, 0.02, 0.01, 0.01])
            },
            {
                "id": "KB_002",
                "text": "Relational PostgreSQL database tables require primary key constraints for referential integrity.",
                "vector": np.array([0.01, 0.98, 0.01, 0.01])
            },
            {
                "id": "KB_003",
                "text": "Cardiovascular wellness guidelines recommend daily aerobic activity and blood pressure monitoring.",
                "vector": np.array([0.89, 0.03, 0.01, 0.05])
            },
            {
                "id": "KB_004",
                "text": "Telescopes observe redshift in distant galaxies, confirming cosmic universe expansion.",
                "vector": np.array([0.01, 0.01, 0.98, 0.01])
            },
            {
                "id": "KB_005",
                "text": "Fermenting sourdough starter with wild yeast cultures creates aromatic artisanal loaves.",
                "vector": np.array([0.01, 0.01, 0.02, 0.97])
            }
        ]

    def query_shootout(self, query_text: str, query_vec: np.ndarray) -> Dict[str, Any]:
        # Boolean match
        bool_hits = [d for d in self.corpus if any(w in d["text"].lower() for w in query_text.lower().split())]
        
        # Lexical Jaccard
        q_words = set(query_text.lower().split())
        lex_scores = []
        for d in self.corpus:
            d_words = set(d["text"].lower().split())
            jaccard = len(q_words.intersection(d_words)) / max(len(q_words.union(d_words)), 1)
            lex_scores.append((d, jaccard))
        lex_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Dense Cosine
        q_norm = query_vec / np.linalg.norm(query_vec)
        dense_scores = []
        for d in self.corpus:
            d_norm = d["vector"] / np.linalg.norm(d["vector"])
            cos = float(np.dot(q_norm, d_norm))
            dense_scores.append((d, cos))
        dense_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "boolean_count": len(bool_hits),
            "lexical_top": lex_scores[0],
            "dense_top": dense_scores[0],
            "all_dense": dense_scores
        }

st.divider()

# Section 3: Interactive Workbench: Search Paradigm Shootout
st.header("3. Interactive Workbench: Search Paradigm Shootout", divider="rainbow")

st.markdown("""
Test how three generations of search engines evaluate the same user query.
Select a test query and inspect why **Boolean** and **Lexical** search fail on synonym queries like *"How to treat a heart attack"*, while **Dense Vector Search** finds the medical documentation effortlessly!
""")

shootout = InteractiveParadigmWorkbench()

col_sh_l, col_sh_r = st.columns([1, 1])

with col_sh_l:
    st.subheader("1. Select Test Query")
    test_queries = {
        "How to treat a heart attack": {
            "text": "How to treat a heart attack",
            "vec": np.array([0.95, 0.02, 0.01, 0.02]),
            "description": "Zero lexical overlap with 'Myocardial infarction'!"
        },
        "PostgreSQL database constraints": {
            "text": "PostgreSQL database constraints",
            "vec": np.array([0.02, 0.96, 0.01, 0.01]),
            "description": "Exact keyword overlap with KB_002."
        },
        "Deep space galaxies and universe": {
            "text": "Deep space galaxies and universe",
            "vec": np.array([0.01, 0.01, 0.96, 0.02]),
            "description": "Astronomy inquiry."
        }
    }

    q_sel = st.selectbox("Choose Inquiry Scenario:", list(test_queries.keys()))
    q_data = test_queries[q_sel]
    st.info(f"💡 **Scenario**: {q_data['description']}")

with col_sh_r:
    st.subheader("2. Paradigm Shootout Results")
    res = shootout.query_shootout(q_data["text"], q_data["vec"])

    m1, m2 = st.columns(2)
    m1.metric("Lexical Best Match Score", f"{res['lexical_top'][1]:.4f}")
    m2.metric("Dense Vector Best Match Score", f"{res['dense_top'][1]:.4f}")

    st.markdown("#### Dense Vector Nearest Neighbors (Top Hits)")
    df_hits = pd.DataFrame([
        {
            "Doc ID": d["id"],
            "Cosine Similarity": f"{score:.4f}",
            "Document Snippet": d["text"]
        }
        for d, score in res["all_dense"][:3]
    ])
    st.dataframe(df_hits, use_container_width=True)

    top_d, top_score = res["dense_top"]
    if q_sel == "How to treat a heart attack":
        st.success(f"🎯 **Vector Superpower Confirmed**: Query used 'heart attack' (0 shared words), yet Vector Search retrieved `{top_d['id']}` ('Myocardial infarction') with a **{top_score:.4f}** cosine score!")
    else:
        st.success(f"✅ Top Match: `{top_d['id']}` with strong score of **{top_score:.4f}**.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Vector Search Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the computational difference between Exact Nearest Neighbor (k-NN) and Approximate Nearest Neighbor (ANN) search?**
    * **Answer**:
      * **Exact k-NN (Flat Index)**: Computes the dot product between the query vector and every single vector in the database (brute-force linear scan).
        * Time Complexity: O(N * d) where N is corpus size and d is dimension.
        * Scaling: For 10 million 1536-d vectors, one query requires 15.3 billion floating-point multiplications (~500ms on CPU).
      * **Approximate Nearest Neighbor (ANN)**: Uses graph structures (HNSW) or inverted file clustering (IVF) to prune 99% of search space.
        * Time Complexity: O(log N).
        * Trade-off: Trades a tiny margin of recall (retrieving 98% of true nearest neighbors) for a **100x speedup** (sub-5ms retrieval).

    **Q2: What is the fundamental architecture of Hierarchical Navigable Small World (HNSW) graphs in vector search?**
    * **Answer**:
      * HNSW is inspired by the "six degrees of separation" skip-list concept.
      * It builds a multi-layer graph hierarchy:
        * **Top Layers**: Sparse graphs with long-range edges (express highway). The query makes large leaps across vector space quickly.
        * **Bottom Layer (Layer 0)**: Dense graph containing all data points with short-range local connections.
      * The search starts at the top layer, greedily traverses to the nearest node, drops down to the next layer, and repeats until reaching Layer 0.

    **Q3: Why is Hybrid Search (BM25 + Vector) the industry gold standard for enterprise production systems?**
    * **Answer**:
      * Pure Vector Search excels at conceptual queries, synonyms, and conversational intent, but fails on exact part numbers, GUIDs, error codes, and rare proper nouns.
      * Pure BM25 excels at exact keyword filtering, but fails on paraphrases and vocabulary mismatch.
      * **Hybrid Search** runs both in parallel and fuses the result rankings using **Reciprocal Rank Fusion (RRF)**:
        `RRF_Score(d) = 1 / (60 + Rank_BM25(d)) + 1 / (60 + Rank_Vector(d))`
      * This ensures that whether a user searches with exact technical jargon or vague conceptual phrasing, the system retrieves the correct document!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Deploying Brute-Force Flat Search at Scale**:
      * *Issue*: Using `IndexFlatIP` or `sklearn.neighbors.NearestNeighbors` for a production corpus of 2,000,000 vectors.
      * *Consequence*: Query latency increases linearly with database size, spiking from 10ms to 800ms and crashing under multi-user concurrency.
      * *Fix*: Deploy ANN indexes (HNSW, IVF-PQ) via vector databases like ChromaDB, Qdrant, or FAISS.

    * **Trap 2: Ignoring Cold Index Build Times**:
      * *Issue*: Rebuilding an HNSW graph from scratch synchronously during application startup.
      * *Consequence*: Building HNSW graphs with high connectivity ($M=64$) on millions of vectors takes hours and high CPU/RAM.
      * *Fix*: Persist serialized index graphs to disk/cloud storage and load pre-built graph files.

    * **Trap 3: Normalization Inconsistency in Vector Indexes**:
      * *Issue*: Creating a database index configured for `Inner Product` and inserting unnormalized raw vectors.
      * *Consequence*: Long documents or high-magnitude vectors dominate search results, corrupting relevance rankings.
      * *Fix*: Enforce unit $L_2$ normalization on all vectors prior to index insertion.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Vector Search**
      * *Definition*: Information retrieval based on finding nearest neighbors in a continuous geometric embedding space rather than matching keywords.
    * **Flashcard 2: Approximate Nearest Neighbor (ANN)**
      * *Definition*: Algorithmic indexing techniques (HNSW, IVF) that achieve logarithmic retrieval latency by trading a tiny percentage of recall.
    * **Flashcard 3: HNSW (Hierarchical Navigable Small World)**
      * *Definition*: The frontier graph-based ANN algorithm that navigates multi-layer geometric graphs to achieve single-digit millisecond vector search.
    * **Flashcard 4: Reciprocal Rank Fusion (RRF)**
      * *Definition*: A parameter-free rank aggregation method that combines lexical BM25 and dense vector rankings into a single optimal hybrid score.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Vector Search:

- [ ] I can describe the 3 generations of search technology (Boolean, Inverted Index, Vector Search).
- [ ] I understand the difference between exact k-NN (O(N)) and Approximate Nearest Neighbor (O(log N)) search.
- [ ] I understand the high-level architecture of HNSW multi-layer skip-graphs.
- [ ] I know why Hybrid Search (BM25 + Dense) resolves the complementary blind spots of each method.
- [ ] I can explain Reciprocal Rank Fusion (RRF) for combining keyword and vector rankings.
""")
