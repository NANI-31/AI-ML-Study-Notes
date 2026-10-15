import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class BM25EngineSimulator:
    """
    Pure Python implementation of the Okapi BM25 ranking algorithm,
    demonstrating term frequency saturation, inverse document frequency (IDF),
    and document length normalization.
    """
    CORPUS = [
        {"id": "doc_1", "text": "BM25 keyword search ranks documents based on term frequency and inverted document frequency.", "tokens": ["bm25", "keyword", "search", "ranks", "documents", "term", "frequency", "inverted", "document", "frequency"]},
        {"id": "doc_2", "text": "Inverted index data structures map unique token terms to posting lists of document identifiers.", "tokens": ["inverted", "index", "data", "structures", "map", "unique", "token", "terms", "posting", "lists", "document", "identifiers"]},
        {"id": "doc_3", "text": "Dense vector search maps text into continuous semantic coordinate spaces using neural embeddings.", "tokens": ["dense", "vector", "search", "maps", "text", "continuous", "semantic", "coordinate", "spaces", "neural", "embeddings"]},
        {"id": "doc_4", "text": "Hybrid search combines BM25 lexical keyword matching with dense vector semantics.", "tokens": ["hybrid", "search", "combines", "bm25", "lexical", "keyword", "matching", "dense", "vector", "semantics"]},
        {"id": "doc_5", "text": "Exact matching fails on synonyms but excels on unique serial codes and technical acronyms.", "tokens": ["exact", "matching", "fails", "synonyms", "excels", "unique", "serial", "codes", "technical", "acronyms"]}
    ]

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.CORPUS)
        self.doc_lengths = {d["id"]: len(d["tokens"]) for d in self.CORPUS}
        self.avg_doc_len = sum(self.doc_lengths.values()) / float(self.doc_count)
        
        # Calculate document frequency (DF) for each term
        self.df: Dict[str, int] = {}
        for d in self.CORPUS:
            unique_terms = set(d["tokens"])
            for t in unique_terms:
                self.df[t] = self.df.get(t, 0) + 1

    def compute_idf(self, term: str) -> float:
        n_qi = self.df.get(term, 0)
        # Standard Lucene/BM25 IDF formula: ln(1 + (N - n + 0.5) / (n + 0.5))
        return float(np.log(1.0 + (self.doc_count - n_qi + 0.5) / (n_qi + 0.5)))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_terms = query.lower().split()
        scores: Dict[str, float] = {d["id"]: 0.0 for d in self.CORPUS}

        for term in query_terms:
            idf = self.compute_idf(term)
            for d in self.CORPUS:
                d_id = d["id"]
                tf = d["tokens"].count(term)
                if tf > 0:
                    d_len = self.doc_lengths[d_id]
                    # BM25 TF saturation with length normalization
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    term_score = idf * (numerator / denominator)
                    scores[d_id] += term_score

        results = []
        for d in self.CORPUS:
            s = scores[d["id"]]
            if s > 0:
                results.append({
                    "id": d["id"],
                    "bm25_score": round(s, 4),
                    "text": d["text"],
                    "doc_len": self.doc_lengths[d["id"]]
                })
        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="11.1 Keyword Search - RAG Master Curriculum",
    page_icon="🔤",
    layout="wide"
)

st.title("11.1 Keyword Search: Inverted Indexes, The BM25 Algorithm, Exact Matching, and Term Relevance")
st.caption("Chapter 11: Hybrid Search | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
While dense semantic vector embeddings have transformed modern information retrieval, 
**classical lexical keyword search remains an indispensable foundation of enterprise RAG**.

Dense embedding models compress language into continuous mathematical coordinates, which often blurs 
the distinction between closely related terms and loses fidelity for exact alphanumeric strings. 
**Keyword Search (BM25)** guarantees exact token matching, inverse document frequency (IDF) term weighting, 
and sub-millisecond retrieval speeds across inverted indices.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Inverted Index & Lexical BM25 Information Retrieval Pipeline", divider="rainbow")
st.markdown("""
The architecture below traces how document collections are tokenized, processed into inverted indices, 
and queried using BM25 term weighting.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    bm25_bp_dot = """
    digraph BM25Pipeline {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_ingest {
            label = "1. Inverted Index Construction Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_docs [label="Raw Text Documents\\nCorpus of N Documents", fillcolor="#0F172A", color="#38BDF8"];
            analyzer [label="Text Analyzer Pipeline\\nTokenize -> Lowercase -> Stopwords -> Stemming", fillcolor="#0F172A", color="#38BDF8"];
            inv_index [label="Inverted Index Store\\nTerm Dictionary -> Posting Lists [DocID, TF]", fillcolor="#0F172A", color="#10B981"];

            raw_docs -> analyzer -> inv_index;
        }

        subgraph cluster_scoring {
            label = "2. Query Execution & Okapi BM25 Scoring";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_in [label="User Query String\\ne.g. 'ERR_404 connection timeout'", fillcolor="#0F172A", color="#EC4899"];
            q_terms [label="Query Tokens Extraction\\n['err_404', 'connect', 'timeout']", fillcolor="#334155", color="#EC4899"];
            idf_calc [label="IDF Weighting Stage\\nln(1 + (N - n + 0.5) / (n + 0.5))", fillcolor="#334155", color="#F59E0B"];
            tf_sat [label="TF Saturation & Length Normalization\\nk1 (saturation limit) & b (length penalty)", fillcolor="#334155", color="#8B5CF6"];

            q_in -> q_terms;
            q_terms -> idf_calc;
            inv_index -> tf_sat [label="Postings [TF, DocLen]"];
            idf_calc -> tf_sat [label="Term Weights"];
        }

        subgraph cluster_out {
            label = "3. Ranked Output";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            top_bm25 [label="Top-K Ranked Document Chunks\\nRanked by Cumulative BM25 Score", fillcolor="#0F172A", color="#10B981"];
            tf_sat -> top_bm25 [label="Aggregate & Sort"];
        }
    }
    """
    st.graphviz_chart(bm25_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production search engines like **Elasticsearch**, **OpenSearch**, and **SQLite FTS5** rely on Apache Lucene's optimized 
BM25 implementation. Unlike neural embedding generation (which requires GPU forward passes or model API round trips), 
Lucene's inverted index lookups take **under 1 millisecond** across millions of documents. 
In enterprise RAG, BM25 serves as the high-precision keyword half of **Hybrid Search pipelines**, ensuring exact model numbers, 
customer IDs, and code snippets are never lost.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHAT IS KEYWORD SEARCH?
# ==============================================================================

st.header("2. What is Keyword Search?", divider="rainbow")
st.markdown("""
**Keyword Search** (also called lexical information retrieval) is a technique that locates documents 
based on the exact appearance and statistical frequency of query tokens within an inverted index.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Core Concepts of Lexical Retrieval")
    st.markdown("""
    * **Inverted Index Data Structure**:
      Instead of scanning documents sequentially, an inverted index maintains a dictionary of all unique vocabulary words. 
      Each word maps to a **posting list** containing document IDs and term frequencies where that word appears.
    * **Exact Token Verification**:
      Every match is deterministic; a document either contains the token or does not.
    * **Zero Model Inference Latency**:
      Executes as memory and bitwise operations without neural network computation.
    """)

with col2:
    st.subheader("Inverted Index Structure Example")
    inv_df = pd.DataFrame([
        {"Token Term", "Document Frequency (DF)", "Posting List (DocID, TermFrequency)"},
        {"bm25", "2", "[doc_1: 1, doc_4: 1]"},
        {"inverted", "2", "[doc_1: 1, doc_2: 1]"},
        {"search", "3", "[doc_1: 1, doc_3: 1, doc_4: 1]"},
        {"vector", "2", "[doc_3: 1, doc_4: 1]"},
        {"acronyms", "1", "[doc_5: 1]"}
    ])
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: BM25
# ==============================================================================

st.header("3. BM25 (Best Matching 25) Algorithmic Formulation", divider="rainbow")
st.markdown("""
**Okapi BM25** is a probabilistic ranking function that improves upon classical TF-IDF 
by incorporating **term frequency saturation** and **document length normalization**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Three Core Components of BM25")
    st.markdown("""
    1. **Inverse Document Frequency (IDF)**:
       Measures how rare a word is across the entire corpus. Common words receive low weight; rare keywords receive high weight:
       `IDF(q) = ln(1 + (N - DF + 0.5) / (DF + 0.5))`
    2. **Term Frequency Saturation (k1)**:
       Controls how quickly the score saturates as a word appears repeatedly in a document.
       * If k1 = 1.2 to 2.0, seeing a keyword 20 times provides only slightly more score than seeing it 4 times.
    3. **Document Length Normalization (b)**:
       Penalizes verbose documents. If a 10-page paper and a 1-paragraph memo both mention a keyword twice, 
       the shorter document receives a higher relevance score.
       * Parameter b typically ranges between 0.6 and 0.8 (default 0.75).
    """)

with col2:
    st.subheader("Why BM25 Outperforms Pure TF-IDF")
    st.markdown("""
    * **Linear TF Trap in TF-IDF**:
      In classical TF-IDF, a spam document repeating the word `"database"` 500 times achieves a score 500x higher than a focused document mentioning it once.
    * **BM25 Diminishing Returns**:
      BM25 caps the score contribution of repeated terms via an asymptote:
      as term frequency approaches infinity, the term weight asymptotically approaches `k1 + 1`.
    """)

st.divider()

# ==============================================================================
# SECTION 4: EXACT MATCHING
# ==============================================================================

st.header("4. Exact Matching & Tokenization", divider="rainbow")
st.markdown("""
The performance of keyword search depends heavily on the **text analysis pipeline** that transforms raw strings into indexable tokens.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Text Analysis Stages")
    st.markdown("""
    * **Tokenization**:
      Splitting text streams into discrete tokens on whitespace, punctuation, or camelCase boundaries.
    * **Case Folding**:
      Converting tokens to lowercase (`"PostgreSQL"` -> `"postgresql"`).
    * **Stopword Removal**:
      Pruning ubiquitous high-frequency words (`"the"`, `"and"`, `"is"`) that dilute search scores.
    * **Stemming / Lemmatization**:
      Reducing words to their root forms (e.g., Porter Stemmer: `"indexing"`, `"indexed"`, `"indexes"` -> `"index"`).
    * **N-gram Shingling**:
      Creating sub-word character sequences to enable fuzzy matching and typo tolerance.
    """)

with col2:
    st.subheader("Where Exact Matching is Mandatory")
    st.markdown("""
    * **Code Search**: Function names (`def authenticate_jwt_token()`), SQL statements, regex strings.
    * **Legal Case Law**: Precise statute citations (`18 U.S.C. Section 1030`), docket numbers.
    * **Financial Records**: Stock tickers (`GOOGL`, `MSFT`), account numbers.
    * **Healthcare & Life Sciences**: Clinical trial IDs (`NCT04280705`), chemical formula strings.
    """)

st.divider()

# ==============================================================================
# SECTION 5: KEYWORD RELEVANCE
# ==============================================================================

st.header("5. Keyword Relevance & Term Frequency Saturation", divider="rainbow")
st.markdown("""
Understanding how BM25 hyperparameters k1 and b balance term saturation and document length.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Hyperparameter Tuning Guide")
    param_guide_df = pd.DataFrame([
        {"Hyperparameter", "Default", "Typical Range", "Engineering Effect"},
        {"k1 (Term Saturation)", "1.5", "1.2 - 2.0", "Higher k1 delays saturation; rewards documents with high repeated keyword density."},
        {"b (Length Normalization)", "0.75", "0.5 - 0.9", "Higher b imposes strict penalties on long documents; b=0 disables length normalization entirely."}
    ])
    st.dataframe(param_guide_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Failure Modes of Pure Lexical Search")
    st.markdown("""
    * **The Vocabulary Mismatch Problem**:
      If a user searches for `"automobile repairs"` and a manual uses `"car maintenance"`, BM25 returns 0 results because there is zero token overlap.
    * **Synonym & Paraphrase Ignorance**:
      Cannot recognize that `"heart attack"` and `"myocardial infarction"` represent the exact same medical condition.
    * **Cross-Lingual Blindness**:
      Cannot match queries across languages without explicit translation.
    """)

st.divider()

# ==============================================================================
# SECTION 6: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("6. Production Implementation: Pure Python BM25 Lexical Search Engine", divider="rainbow")
st.markdown("""
Below is an enterprise-grade pure Python BM25 search engine implementing an inverted index, 
length normalization, and Okapi BM25 scoring.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Set

class ProductionBM25Engine:
    """
    Pure Python Okapi BM25 search engine with inverted indexing,
    length normalization, and IDF caching.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.df: Dict[str, int] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = text.lower().split()
        self.corpus[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        
        # Build inverted index
        seen_terms: Set[str] = set()
        for t in tokens:
            if t not in self.inverted_index:
                self.inverted_index[t] = {}
            self.inverted_index[t][doc_id] = self.inverted_index[t].get(doc_id, 0) + 1
            seen_terms.add(t)

        for t in seen_terms:
            self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths.values()) / float(len(self.doc_lengths))

    def _compute_idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        N = len(self.doc_lengths)
        if n == 0:
            return 0.0
        return float(np.log(1.0 + (N - n + 0.5) / (n + 0.5)))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = query.lower().split()
        scores: Dict[str, float] = {d_id: 0.0 for d_id in self.doc_lengths.keys()}

        for term in query_terms:
            idf = self._compute_idf(term)
            if idf <= 0.0:
                continue

            postings = self.inverted_index.get(term, {})
            for d_id, tf in postings.items():
                d_len = self.doc_lengths[d_id]
                # BM25 TF saturation formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                scores[d_id] += idf * (numerator / denominator)

        ranked = [(d_id, score) for d_id, score in scores.items() if score > 0.0]
        ranked.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "id": d_id,
                "score": round(score, 4),
                "text": self.corpus[d_id]
            }
            for d_id, score in ranked[:top_k]
        ]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 7: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("7. Interactive Workbench: BM25 Hyperparameter Tuning & Token Weight Explorer", divider="rainbow")
st.markdown("""
Experiment with BM25 lexical retrieval. Adjust parameters k1 (term saturation) and b (length penalty) 
to observe how ranking scores and candidate document positions adapt.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("BM25 Tuning Controls")
    search_query = st.text_input("Lexical Query", value="bm25 keyword search frequency")
    k1_param = st.slider("k1 (Term Frequency Saturation)", 0.5, 3.0, 1.5, step=0.1)
    b_param = st.slider("b (Document Length Normalization)", 0.0, 1.0, 0.75, step=0.05)
    top_k_val = st.slider("Top-K Candidates", 1, 5, 3)

with col2:
    st.subheader("BM25 Query Results")
    engine = BM25EngineSimulator(k1=k1_param, b=b_param)
    results = engine.search(search_query, top_k=top_k_val)

    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df[["id", "bm25_score", "doc_len", "text"]], use_container_width=True, hide_index=True)
        st.success(f"Retrieved {len(results)} documents with BM25 matches.")
    else:
        st.warning("No documents matched any query tokens!")

    st.markdown("""
    **Observation**:
    * Adjusting **b to 1.0** increases the penalty on long documents, favoring concise chunks.
    * Adjusting **k1 higher** rewards documents with repeated keyword occurrences.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: What is the primary mathematical improvement of BM25 over classical TF-IDF?"):
    st.markdown("""
    **Answer**:
    Classical TF-IDF assumes that term frequency scales linearly: a document mentioning a keyword 100 times receives 100x the score of a document mentioning it once. 
    BM25 introduces **non-linear term frequency saturation** using hyperparameter k1: as term frequency rises, additional occurrences produce diminishing returns, asymptotically capping the score contribution at `k1 + 1`. 
    Furthermore, BM25 incorporates **document length normalization** via hyperparameter b, penalizing lengthy documents so that concise, focused chunks receive higher relevance scores.
    """)

with st.expander("Q2: Why does pure dense semantic search struggle with serial codes, error IDs, and legal citations?"):
    st.markdown("""
    **Answer**:
    Dense embedding models compress sentences into low-dimensional continuous coordinates (e.g., 768 or 1,536 dimensions). 
    During tokenization and neural encoding, unique alphanumeric tokens (like `ERR_404_NULL_POINTER` or `18 U.S.C. 1030`) 
    are often broken into common sub-words (e.g., Byte-Pair Encoding) or mapped to generic semantic clusters. 
    The resulting vector represents the broad concept of an error or law, losing the precise alphanumeric identity. 
    BM25 inverted indices, in contrast, store exact literal strings, guaranteeing instant deterministic matches.
    """)

with st.expander("Q3: What role does Inverse Document Frequency (IDF) play in BM25 scoring?"):
    st.markdown("""
    **Answer**:
    IDF measures the rarity and information content of each query term across the entire corpus. 
    Common words (like 'system', 'data', 'information') that appear in almost every document receive near-zero IDF weights. 
    Rare distinguishing keywords (like 'HNSW', 'Quantization', 'Tombstone') receive high IDF weights. 
    When computing BM25 relevance, matches on rare terms dominate the final score, ensuring document relevance aligns with the user's specific intent.
    """)

with st.expander("Q4: What is the impact of setting parameter b to 0 in BM25?"):
    st.markdown("""
    **Answer**:
    Parameter b governs document length normalization. When b = 0, length normalization is completely disabled, 
    meaning document length is ignored during scoring. A 100-page document containing two occurrences of a keyword receives 
    the exact same score as a 1-sentence paragraph containing two occurrences. 
    In production, b = 0.75 is standard, ensuring shorter, more focused chunks are prioritized.
    """)

with st.expander("Q5: What is the vocabulary mismatch problem in lexical search and how is it resolved?"):
    st.markdown("""
    **Answer**:
    The vocabulary mismatch problem occurs when the user phrases a query using words that differ from the exact terms in the source document 
    (e.g., searching for `"laptop pricing"` when the document states `"notebook computer cost"`). 
    Because BM25 relies on exact token overlap, it returns zero results. 
    In modern production systems, this is resolved using **Hybrid Search**: combining BM25 keyword matching with dense vector embeddings via Reciprocal Rank Fusion (RRF).
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying BM25 Without Removing Punctuation and Common Stopwords**  
    Failing to strip punctuation or stopwords causes words like `"the"` or commas to enter posting lists, 
    bloating inverted index RAM and slowing query lookups by 3x.
    """)
    st.error("""
    **Trap 2: Normalizing BM25 Scores Linearly with Cosine Similarity**  
    BM25 scores are unbounded positive numbers (e.g., 0.5 to 14.8), while cosine similarity is bounded in [0, 1]. 
    Attempting linear combination (`0.5 * bm25 + 0.5 * cosine`) breaks whenever corpus size shifts. Use Reciprocal Rank Fusion (RRF) instead.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Setting BM25 `b = 0` on Heterogeneous Chunk Lengths**  
    When document chunk sizes vary widely (e.g., 50 tokens to 1,500 tokens), setting b = 0 allows massive chunks 
    to dominate search rankings simply by virtue of having more text. Always retain length normalization (b approx 0.75).
    """)
    st.warning("""
    **Trap 4: Omitting Substring/N-gram Tokenization on Serial Numbers**  
    If users search for partial part numbers (e.g., `"4902"` instead of `"SKU-4902-X"`), standard whitespace tokenization 
    fails to find the record. Configure 3-gram or edge-ngram analyzers for serial fields.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Inverted Index Concept")
        st.caption("How does it work?")
        st.markdown("**Back**: Maps unique vocabulary terms to posting lists of document IDs and term frequencies, enabling sub-millisecond keyword lookup.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: BM25 k1 Parameter")
        st.caption("What does k1 control?")
        st.markdown("**Back**: Term frequency saturation limit. Higher k1 delays saturation; standard default is 1.5.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: BM25 b Parameter")
        st.caption("What does b control?")
        st.markdown("**Back**: Document length normalization penalty. Default 0.75 penalizes verbose documents in favor of concise chunks.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Vocabulary Mismatch")
        st.caption("What is the core flaw of BM25?")
        st.markdown("**Back**: Fails completely when users use synonyms or paraphrases without exact token overlap. Resolved by pairing with dense vector search.")

st.divider()

# ==============================================================================
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("9. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Keyword Search:")

c1 = st.checkbox("I understand the inverted index data structure and why it enables sub-millisecond lexical lookups.", value=True)
c2 = st.checkbox("I understand the Okapi BM25 formula, including IDF weighting, term saturation (k1), and length normalization (b).", value=True)
c3 = st.checkbox("I know why BM25 outperforms classical TF-IDF by capping repeated term scores asymptotically.", value=True)
c4 = st.checkbox("I understand the text analysis pipeline: tokenization, stopword removal, and stemming.", value=True)
c5 = st.checkbox("I can identify use cases where BM25 exact matching is strictly required over dense neural vectors.", value=True)
c6 = st.checkbox("I can implement a pure Python BM25 search engine with an inverted index and length normalization.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Keyword Search and understand the lexical information retrieval foundations of modern hybrid search.")
