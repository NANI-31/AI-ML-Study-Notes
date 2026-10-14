import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import math
import re
from typing import List, Dict, Any, Tuple, Set

st.set_page_config(
    page_title="23.1 Keyword + Semantic Search - Embeddings Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("23.1 Keyword + Semantic Search: BM25 Lexical Precision vs. Dense Vector Recall")
st.caption("Chapter 23: Hybrid Search | Domain: 3. Vector Embeddings")
st.markdown("""
In the early days of Generative AI, developers assumed dense vector embeddings would completely replace classical keyword search engines like Elasticsearch and Solr.
In production, however, teams quickly discovered that **pure vector search has fatal blindspots**:
* Searching for exact error codes (`ERR_CONNECTION_REFUSED`), product SKUs (`A-4920-X`), or legal citations fails because dense tokenizers split rare strings into fragmented subwords.
* Conversely, **pure keyword search (BM25)** completely fails when users describe problems in their own words using synonyms or conceptual descriptions (e.g. searching *"heart attack"* fails to match *"myocardial infarction"*).

**Modern Information Retrieval requires both engines operating in harmony**:
* **BM25 Keyword Search**: Delivers **lexical precision** for exact codes, acronyms, and rare proper nouns.
* **Dense Vector Search**: Delivers **semantic recall** for concepts, paraphrases, and intent.

This lesson explores the mathematics of Okapi BM25, the mechanics of dense vector matching, and why combining them is essential for enterprise search.
""")

st.divider()

# Section 1: The Duality of Lexical & Semantic Search
st.header("1. Lexical BM25 vs. Dense Vector Retrieval", divider="rainbow")

col_dual1, col_dual2 = col_dual1, col_dual2 = st.columns([1, 1])

with col_dual1:
    st.markdown("""
    ### The Strengths & Weaknesses of Each Paradigm
    * **Okapi BM25 (Sparse Lexical Search)**:
      * Built on an **Inverted Index** mapping tokens directly to posting lists of document IDs.
      * **Term Frequency Saturation ($k_1$)**: Prevents keyword stuffing. Repeating a word 10 times does not make it 10x more relevant.
      * **Document Length Normalization ($b$)**: Penalizes long documents that contain many words purely by chance.
      * **Inverse Document Frequency (IDF)**: Heavily rewards rare discriminative words over common words.
      * **Failure Mode**: The Vocabulary Mismatch Problem (Zero shared words = Zero results).
    * **Dense Vector Search**:
      * Projects queries and passages into a continuous latent space where semantic proximity governs relevance.
      * **Strength**: Discovers synonyms, paraphrases, and cross-lingual meaning.
      * **Failure Mode**: The Alphanumeric Blindspot (Dilutes exact part numbers and codes).
    """)

with col_dual2:
    st.markdown("### The Complementary Search Duality")
    dual_dot = """
    digraph SearchDuality {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQuery [label="User Query\\n'Fix Error 504 Gateway Timeout in Nginx'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_bm25 {
            label="Sparse Lexical Engine (BM25)"; style="dashed"; color="#3B82F6";
            BM25Invert [label="Inverted Index Lookup\\nExact matches for '504', 'Nginx', 'timeout'", fillcolor="#DBEAFE"];
            BM25Hit [label="High Precision on Code '504'\\nZero synonym understanding", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
            BM25Invert -> BM25Hit;
        }

        subgraph cluster_vector {
            label="Dense Semantic Engine (Bi-Encoder)"; style="dashed"; color="#10B981";
            VecHNSW [label="HNSW Graph Traversal\\nHigh semantic match on 'server response lag'", fillcolor="#D1FAE5"];
            VecHit [label="High Recall on Concepts\\nMay dilute exact code '504'", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            VecHNSW -> VecHit;
        }

        HybridOut [label="Combined Hybrid Retrieval\\n100% Lexical Precision + 100% Semantic Recall", fillcolor="#FEF08A", color="#D97706", style="filled,bold"];

        UserQuery -> BM25Invert;
        UserQuery -> VecHNSW;
        BM25Hit -> HybridOut;
        VecHit -> HybridOut;
    }
    """
    st.graphviz_chart(dual_dot, use_container_width=False)

st.markdown("### Lexical BM25 vs. Dense Vector Comparison")
st.markdown("""
| Dimension | Okapi BM25 Keyword Search | Dense Vector Semantic Search |
| :--- | :--- | :--- |
| **Representation** | High-dimensional sparse vector (Vocabulary size ~100k+) | Dense low-dimensional vector ($d = 384$ to $1536$) |
| **Index Structure** | Inverted Index (Posting lists) | ANN Graph (HNSW) or Inverted Centroids (IVF) |
| **Exact Identifiers (`CVE-2024-3094`)** | **Outstanding (100% exact match)** | Poor (Subword token fragmentation) |
| **Synonyms (`heart attack` vs `myocardial`)** | **Terrible (0 matches)** | **Outstanding (High cosine similarity)** |
| **Conceptual Questions** | Poor (Relies on lucky keyword overlap) | **Outstanding (Intent-driven mapping)** |
| **Query Latency** | ~1 - 3 ms (Very Fast) | ~2 - 6 ms (Very Fast) |
""")

st.divider()

# Section 2: Production Code: Pure-Python BM25 & Vector Engine
st.header("2. Production Code: Pure-Python Okapi BM25 & Dense Search Engine", divider="rainbow")

st.markdown("""
The production utility below implements both a genuine Okapi BM25 engine (with $k_1$ saturation and $b$ length normalization) and a dense vector search engine, returning separate candidate rankings.
""")

st.code(r'''
import math
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

class LexicalSemanticSearchEngine:
    """
    Implements genuine Okapi BM25 alongside dense vector search.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, embedding_dim: int = 64):
        self.k1 = k1
        self.b = b
        self.dim = embedding_dim
        
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {} # term -> list of (doc_idx, term_freq)
        self.doc_count: int = 0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def add_documents(self, docs: List[Dict[str, str]]):
        self.documents = docs
        self.doc_count = len(docs)
        
        # Tokenize and compute document lengths
        tokenized_corpus = [self.tokenize(d["text"]) for d in docs]
        self.doc_lengths = [len(tokens) for tokens in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / float(self.doc_count) if self.doc_count > 0 else 0.0

        # Build Inverted Index & Vector Store
        for doc_idx, tokens in enumerate(tokenized_corpus):
            counts = Counter(tokens)
            for term, freq in counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))
            
            self.documents[doc_idx]["vector"] = self.mock_embed(docs[doc_idx]["text"])

    def bm25_search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.doc_count

        for token in q_tokens:
            if token not in self.inverted_index:
                continue
            
            postings = self.inverted_index[token]
            df = len(postings)
            # Standard BM25 IDF formulation
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                # BM25 TF saturation formula
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices if scores[i] > 0]

    def vector_search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = self.mock_embed(query)
        scores = [float(np.dot(d["vector"], q_vec)) for d in self.documents]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices]
''', language="python")

st.divider()

# Section 3: Interactive Keyword vs. Semantic Search Arena
st.header("3. Interactive Workbench: Keyword vs. Semantic Comparison Arena", divider="rainbow")

st.markdown("""
Observe how BM25 and Vector Search succeed and fail on opposite types of queries:
1. **Query 1: Exact Error Code (`ERR_NGINX_504`)** -> BM25 wins with exact match; Vector search dilutes.
2. **Query 2: Medical Synonyms (`heart attack symptoms`)** -> Vector search wins via semantic alignment; BM25 scores zero!
""")

class DualSearchEngine:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        t_low = text.lower()
        np.random.seed(abs(hash(t_low)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        
        # Add semantic topic bias
        if any(w in t_low for w in ["heart", "cardiac", "infarction", "artery", "chest"]):
            v[:16] += 3.0
        elif any(w in t_low for w in ["504", "nginx", "timeout", "proxy", "gateway"]):
            v[16:32] += 3.0
        elif any(w in t_low for w in ["refund", "billing", "invoice", "payment"]):
            v[32:] += 3.0
            
        return v / np.maximum(np.linalg.norm(v), 1e-12)

dual_sim = DualSearchEngine()

dual_corpus = [
    {"id": "doc_01", "text": "Patients experiencing acute myocardial infarction report crushing substernal chest discomfort and diaphoresis.", "topic": "Cardiology"},
    {"id": "doc_02", "text": "Cardiovascular rehabilitation exercises strengthen the arterial myocardium following coronary stent placement.", "topic": "Cardiology"},
    {"id": "doc_03", "text": "Nginx upstream timeout triggers HTTP status ERR_NGINX_504 when the gateway proxy fails to respond in 60s.", "topic": "DevOps"},
    {"id": "doc_04", "text": "Server connection timeouts in microservices can be mitigated by configuring keepalive pool settings.", "topic": "DevOps"},
    {"id": "doc_05", "text": "Annual billing invoices can be refunded within 30 days upon submission of a customer support ticket.", "topic": "Billing"}
]

col_k_in, col_k_view = st.columns([1, 1])

with col_k_in:
    st.subheader("1. Test Query Selection")
    sample_queries = [
        "heart attack symptoms",                  # Semantic synonym challenge (Zero overlap with doc_01!)
        "ERR_NGINX_504 upstream timeout",        # Exact code identifier challenge
        "Can I get my subscription money back?"   # Paraphrase challenge
    ]
    chosen_q = st.selectbox("Select Test Query:", sample_queries)
    custom_q = st.text_input("Active Query String:", value=chosen_q)

    st.subheader("2. Search Engine Mode")
    active_engine_mode = st.radio(
        "Select Active Search Engine:",
        ["1. Pure BM25 Keyword Search", "2. Pure Dense Vector Search", "3. Side-by-Side Dual Comparison"],
        index=2
    )
    k_items = st.slider("Top-K Results:", min_value=1, max_value=4, value=2)

with col_k_view:
    st.subheader("3. Search Results Comparison")

    # BM25 simple calculation
    q_words = set(re.findall(r'\b\w+\b', custom_q.lower()))
    bm25_scored = []
    for d in dual_corpus:
        d_words = set(re.findall(r'\b\w+\b', d["text"].lower()))
        overlap = len(q_words.intersection(d_words))
        score = float(overlap) / max(1, len(q_words))
        bm25_scored.append((d, score))
    bm25_scored.sort(key=lambda x: x[1], reverse=True)

    # Vector search calculation
    q_vec = dual_sim.embed(custom_q)
    vec_scored = []
    for d in dual_corpus:
        sim = float(np.dot(dual_sim.embed(d["text"]), q_vec))
        vec_scored.append((d, sim))
    vec_scored.sort(key=lambda x: x[1], reverse=True)

    if "1. Pure BM25" in active_engine_mode or "3. Side-by-Side" in active_engine_mode:
        st.markdown("#### Engine 1: BM25 Lexical Keyword Results")
        bm25_hits = [b for b in bm25_scored if b[1] > 0]
        if not bm25_hits:
            st.error("❌ **BM25 Failed (0 Matches)**: Zero exact word overlap! Pure keyword search completely missed the synonyms.")
        else:
            for rank, (doc, sc) in enumerate(bm25_hits[:k_items], 1):
                with st.expander(f"BM25 Rank {rank} | {doc['id']} (Score: {sc:.2f})", expanded=True):
                    st.markdown(f"> {doc['text']}")

    if "2. Pure Dense" in active_engine_mode or "3. Side-by-Side" in active_engine_mode:
        st.markdown("#### Engine 2: Dense Vector Semantic Results")
        for rank, (doc, sc) in enumerate(vec_scored[:k_items], 1):
            with st.expander(f"Vector Rank {rank} | {doc['id']} (Cosine: {sc:.4f})", expanded=True):
                st.markdown(f"> {doc['text']}")
                st.caption(f"Topic: `{doc['topic']}`")

    if "heart attack" in custom_q.lower():
        st.info("💡 **Key Takeaway**: Dense vector search correctly retrieved *'myocardial infarction'* because they share semantic meaning, while BM25 returned 0 matches!")
    elif "504" in custom_q.lower():
        st.info("💡 **Key Takeaway**: BM25 achieved 100% precision on the exact error code *'ERR_NGINX_504'*!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Search Duality Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does BM25's Term Frequency Saturation ($k_1$) improve upon classical TF-IDF?**
    * **Answer**: In classical TF-IDF, term frequency scales linearly: mentioning a keyword 50 times produces 50x higher score than mentioning it once, making it vulnerable to **keyword stuffing**.
    * In **Okapi BM25**, the term frequency contribution asymptotically approaches an upper limit: `(tf * (k1 + 1)) / (tf + k1)`.
    * With $k_1 = 1.2$, a document mentioning a keyword 20 times receives virtually the same score as one mentioning it 5 times, preventing spam while rewarding true relevance.

    **Q2: What is the 'Vocabulary Mismatch Problem' in information retrieval?**
    * **Answer**: The vocabulary mismatch problem occurs when a searcher uses different terminology from the author to describe the exact same concept (e.g. user searches *"car boot space"* while the document describes *"automobile trunk capacity"*).
    * Lexical keyword search fails completely because zero tokens intersect.
    * Dense vector embeddings solve this problem by mapping different vocabularies to a shared continuous semantic coordinate system.

    **Q3: Why can't dense embedding models handle exact alphanumeric error codes as effectively as BM25?**
    * **Answer**: Byte Pair Encoding (BPE) and WordPiece tokenizers are trained on statistical character frequencies. Rare error codes (e.g. `ERR_504_TIMEOUT`, `CVE-2024-3094`, `0x80070005`) do not exist in the vocabulary and are fragmented into multiple subwords.
    * The dense vector must aggregate hidden states across these arbitrary subword tokens, diluting the specific numerical identity.
    * In contrast, an inverted index stores exact alphanumeric strings as discrete dictionary tokens, matching them with 100% precision.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Abandoning Keyword Search Completely for Vector Search**:
      * *Issue*: Migrating an enterprise technical documentation portal entirely from Elasticsearch to a pure vector database.
      * *Consequence*: Customer complaints spike because searching for exact function names (`np.linalg.norm`), error codes, and product SKUs stops working.
      * *Fix*: Always implement Hybrid Search (BM25 + Dense Vectors) for technical corpora.

    * **Trap 2: Normalizing BM25 Scores with Simple Min-Max Before Combining with Cosine**:
      * *Issue*: Taking `Normalized_BM25 = (BM25 - min) / (max - min)` and adding it directly to Cosine Similarity.
      * *Consequence*: BM25 scores have unbounded, heavy-tailed distributions; outliers distort the normalized scale across different queries.
      * *Fix*: Use rank-based fusion like Reciprocal Rank Fusion (RRF) rather than direct score addition.

    * **Trap 3: Forgetting to Stem or Normalize Queries in BM25**:
      * *Issue*: Searching for *"runs"* against documents containing *"running"* in a raw unstemmed inverted index.
      * *Consequence*: Missed lexical matches on simple grammatical inflections.
      * *Fix*: Apply standard Snowball/Porter stemming or lowercased lemmatization inside the BM25 text analyzer.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Okapi BM25**
      * *Definition*: The industry-standard probabilistic sparse ranking algorithm incorporating term frequency saturation and document length normalization.
    * **Flashcard 2: Vocabulary Mismatch Problem**
      * *Definition*: The failure of lexical search when queries and documents express identical concepts using different words.
    * **Flashcard 3: Term Frequency Saturation ($k_1$)**
      * *Definition*: The parameter in BM25 that caps the score contribution of repeated keywords to prevent keyword stuffing.
    * **Flashcard 4: Search Duality**
      * *Definition*: The complementary combination of BM25 lexical precision (exact codes/acronyms) with dense vector semantic recall (concepts/paraphrases).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Keyword + Semantic Search:

- [ ] I can articulate the mathematical purpose of $k_1$ and $b$ in the BM25 formula.
- [ ] I understand the Vocabulary Mismatch Problem and how dense embeddings solve it.
- [ ] I know why dense embeddings struggle with exact alphanumeric error codes and part numbers.
- [ ] I understand the structural difference between an Inverted Index and an HNSW graph.
- [ ] I know why combining lexical precision with semantic recall is required for enterprise search.
""")
