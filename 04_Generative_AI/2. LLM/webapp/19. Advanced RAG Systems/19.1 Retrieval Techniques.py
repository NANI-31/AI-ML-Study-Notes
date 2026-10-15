import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.1 Retrieval Techniques - Advanced RAG",
    page_icon="🔎",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class ProductionHybridSearchEngine:
    """
    Self-contained hybrid search engine implementing:
    - Pure BM25 keyword retrieval (Robertson-Spärck Jones formula)
    - Pure Dense vector retrieval (Cosine similarity)
    - Reciprocal Rank Fusion (RRF) rank blending
    """
    
    DOCUMENTS = [
        {
            "id": "DOC-01",
            "title": "Database Connection Error Resolution",
            "text": "When encountering ERR_CONN_403, verify PostgreSQL pg_hba.conf host authorization and password credentials.",
            "tokens": ["when", "encountering", "err_conn_403", "verify", "postgresql", "pg_hba", "conf", "host", "authorization", "and", "password", "credentials"],
            "vector": np.array([0.85, 0.15, 0.40, 0.10, 0.20], dtype=np.float32)
        },
        {
            "id": "DOC-02",
            "title": "General Troubleshooting Guide for Network Outages",
            "text": "If client machines cannot connect to services, check firewall rules, DNS resolution, and router port forwarding.",
            "tokens": ["if", "client", "machines", "cannot", "connect", "to", "services", "check", "firewall", "rules", "dns", "resolution", "and", "router", "port", "forwarding"],
            "vector": np.array([0.80, 0.20, 0.45, 0.12, 0.25], dtype=np.float32)
        },
        {
            "id": "DOC-03",
            "title": "Quarterly Financial Reporting Procedures",
            "text": "Accounting teams must reconcile ledger entries and compile Form 10-Q before the third business day of the quarter.",
            "tokens": ["accounting", "teams", "must", "reconcile", "ledger", "entries", "and", "compile", "form", "10-q", "before", "the", "third", "business", "day", "of", "the", "quarter"],
            "vector": np.array([0.05, 0.92, 0.15, 0.30, 0.12], dtype=np.float32)
        },
        {
            "id": "DOC-04",
            "title": "Hardware Return and Employee Offboarding",
            "text": "Departing personnel must return access keys, laptops, and monitors within 7 business days via prepaid courier.",
            "tokens": ["departing", "personnel", "must", "return", "access", "keys", "laptops", "and", "monitors", "within", "7", "business", "days", "via", "prepaid", "courier"],
            "vector": np.array([0.10, 0.45, 0.12, 0.88, 0.20], dtype=np.float32)
        },
        {
            "id": "DOC-05",
            "title": "Post-Quantum Cryptography Architecture",
            "text": "Deployment of Kyber-768 and Dilithium algorithms will replace legacy RSA-2048 encryption across all gateway routes in Q3 2026.",
            "tokens": ["deployment", "of", "kyber-768", "and", "dilithium", "algorithms", "will", "replace", "legacy", "rsa-2048", "encryption", "across", "all", "gateway", "routes", "in", "q3", "2026"],
            "vector": np.array([0.25, 0.28, 0.95, 0.08, 0.05], dtype=np.float32)
        }
    ]
    
    # Query vectors for simulation
    QUERY_VECTORS = {
        "Fix ERR_CONN_403 database authorization error": np.array([0.84, 0.14, 0.38, 0.11, 0.22], dtype=np.float32),
        "Network connection failed and cannot reach server": np.array([0.82, 0.18, 0.42, 0.10, 0.26], dtype=np.float32),
        "When is Form 10-Q quarterly report due?": np.array([0.06, 0.90, 0.16, 0.28, 0.14], dtype=np.float32),
        "Timeline for Kyber-768 quantum encryption rollout": np.array([0.22, 0.30, 0.94, 0.09, 0.06], dtype=np.float32)
    }

    @classmethod
    def compute_bm25(cls, query_str: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        """Calculates BM25 scores across documents."""
        q_tokens = [t.lower() for t in re.findall(r'\b[\w-]+\b', query_str)]
        N = len(cls.DOCUMENTS)
        avgdl = np.mean([len(d["tokens"]) for d in cls.DOCUMENTS])
        
        scores = []
        for doc in cls.DOCUMENTS:
            doc_len = len(doc["tokens"])
            score = 0.0
            doc_token_counts = {}
            for t in doc["tokens"]:
                doc_token_counts[t] = doc_token_counts.get(t, 0) + 1
                
            for q in q_tokens:
                # Calculate document frequency
                df = sum(1 for d in cls.DOCUMENTS if q in d["tokens"])
                if df == 0:
                    continue
                # Robertson-Spärck Jones IDF
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc_token_counts.get(q, 0)
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                score += idf * (num / den)
                
            scores.append((doc["id"], float(score)))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @classmethod
    def compute_dense(cls, query_str: str) -> List[Tuple[str, float]]:
        """Calculates Dense Cosine similarity scores."""
        q_vec = cls.QUERY_VECTORS.get(query_str, np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        q_norm = q_vec / np.linalg.norm(q_vec)
        
        scores = []
        for doc in cls.DOCUMENTS:
            d_norm = doc["vector"] / np.linalg.norm(doc["vector"])
            sim = float(np.dot(q_norm, d_norm))
            scores.append((doc["id"], sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        bm25_ranked: List[Tuple[str, float]],
        dense_ranked: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Calculates Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum_{m in {bm25, dense}} 1 / (k + rank_m(d))
        """
        rrf_scores = {}
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_ranked)}
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranked)}
        
        for doc in cls.DOCUMENTS:
            did = doc["id"]
            r_bm25 = bm25_ranks.get(did, 999)
            r_dense = dense_ranks.get(did, 999)
            score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
            rrf_scores[did] = {
                "id": did,
                "title": doc["title"],
                "text": doc["text"],
                "bm25_rank": r_bm25,
                "dense_rank": r_dense,
                "rrf_score": score
            }
            
        ranked_list = list(rrf_scores.values())
        ranked_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return ranked_list


# --- HEADER & INTRODUCTION ---
st.title("19.1 Retrieval Techniques: Dense vs Sparse, BM25, & Hybrid Search with RRF")
st.caption("Chapter 19: Advanced RAG Systems | Domain: 2. Large Language Models")

st.markdown("""
Why does a multi-million-dollar vector database often fail to find the exact document when a user searches for an error code, 
a part number, or a legal statute?

Because **Dense Semantic Retrieval** projects text onto general conceptual manifolds: it understands that *"automobile"* relates to *"car"*, 
but it struggles with **exact keyword matching**, rare acronyms, product serial numbers, and specific error codes (`ERR_CONN_403`). 
Conversely, **Sparse Keyword Search (BM25)** matches exact alphanumeric tokens flawlessly, but cannot understand synonyms or conceptual paraphrasing.

The gold standard of modern enterprise information retrieval is **Hybrid Search**: querying dense vector embeddings and 
sparse inverted indexes in parallel, then unifying the candidate lists using **Reciprocal Rank Fusion (RRF)**. 
This lesson explores dense, sparse, BM25, and hybrid retrieval mechanics.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Dual-Engine Hybrid Retrieval Architecture & Rank Fusion Pipeline", divider="rainbow")

st.markdown("""
How modern hybrid retrieval architectures execute dual-branch search and fuse candidate ranks:
""")

hybrid_blueprint_dot = """
digraph HybridBlueprint {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_query {
        label="Query Ingress";
        color="#1E293B";
        fontcolor="#38BDF8";
        UserQ [label="User Query\\n'Fix ERR_CONN_403 authorization'", fillcolor="#1E293B"];
    }

    subgraph cluster_dual_branch {
        label="Parallel Dual-Branch Retrieval";
        color="#1E293B";
        fontcolor="#38BDF8";
        SparseBranch [label="Branch A: Lexical BM25\\nInverted Index / Postings\\nExact Token Match (k1=1.5, b=0.75)", fillcolor="#0F766E"];
        DenseBranch [label="Branch B: Dense Bi-Encoder\\nHNSW Vector Index\\nSemantic Cosine Matching", fillcolor="#0369A1"];
    }

    subgraph cluster_fusion {
        label="Rank Fusion & Normalization";
        color="#1E293B";
        fontcolor="#38BDF8";
        RRF [label="Reciprocal Rank Fusion (RRF)\\nScore(d) = sum 1 / (k + rank)\\nScale-Invariant Rank Blending", fillcolor="#6D28D9"];
    }

    subgraph cluster_output {
        label="Fused Candidates";
        color="#1E293B";
        fontcolor="#38BDF8";
        TopCandidates [label="Blended Top-K Chunks\\nPassed to Cross-Encoder / LLM", fillcolor="#059669"];
    }

    UserQ -> SparseBranch;
    UserQ -> DenseBranch;
    SparseBranch -> RRF [label="Rank List A"];
    DenseBranch -> RRF [label="Rank List B"];
    RRF -> TopCandidates;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(hybrid_blueprint_dot, use_container_width=True)

st.markdown("""
### Retrieval Modalities Comparison Matrix
""")

modality_matrix = [
    {
        "Modality": "Dense Semantic Retrieval",
        "Representation": "Dense continuous float vectors in R^1536 (all coordinates non-zero)",
        "Matching Mechanism": "Cosine similarity / Dot Product in latent space",
        "Primary Strengths": "Captures intent, synonyms, cross-lingual concepts",
        "Primary Weaknesses": "Blind to specific error codes, SKU numbers, rare tokens"
    },
    {
        "Modality": "Sparse Keyword (BM25)",
        "Representation": "High-dimensional sparse vectors in R^V (only matching words non-zero)",
        "Matching Mechanism": "Term frequency saturation & Inverse Document Frequency",
        "Primary Strengths": "Exact token precision; handles serial numbers and function names",
        "Primary Weaknesses": "Fails when user query uses synonyms not present in corpus"
    },
    {
        "Modality": "Learned Sparse (SPLADE / BGE-M3 Sparse)",
        "Representation": "Sparse vector in R^V with ML-predicted vocabulary expansion weights",
        "Matching Mechanism": "Dot product over expanded vocabulary postings",
        "Primary Strengths": "Combines exact keywords with learned synonym expansions",
        "Primary Weaknesses": "Higher indexing latency; larger inverted index storage"
    },
    {
        "Modality": "Hybrid Search (Dense + BM25 via RRF)",
        "Representation": "Dual independent representation (Dense float vector + Sparse inverted postings)",
        "Matching Mechanism": "Parallel execution unified via Reciprocal Rank Fusion",
        "Primary Strengths": "Best of both worlds: catches both conceptual paraphrases AND exact codes",
        "Primary Weaknesses": "Requires maintaining two separate index structures"
    }
]
st.dataframe(pd.DataFrame(modality_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - DENSE RETRIEVAL ---
st.header("2. Dense Retrieval: Continuous Latent Manifolds & Bi-Encoder Embeddings", divider="rainbow")

st.markdown("""
**Dense Retrieval** represents text as dense continuous floating-point vectors in high-dimensional hyperspace 
(typically 768, 1024, or 1536 dimensions).

### How Dense Retrieval Captures Latent Meaning
- Unlike keyword matching, dense embeddings map text based on **contextual distribution**:
  - *"How do I fix a leaking faucet?"* clusters right beside *"Plumbing repair instructions"*.
  - Even though the two sentences share zero common words, their dense vectors have a Cosine Similarity exceeding `0.85`!
- Excellent for conversational queries, multi-lingual matching, and conceptual explorations.

### The Fatal Flaw of Pure Dense Retrieval
When a user searches for an exact technical identifier:
- Query: *"Resolve error ERR_CONN_403 on port 5432"*
- The dense model's subword tokenizer splits `ERR_CONN_403` into `ERR`, `_`, `CONN`, `_`, `403`.
- The embedding model projects the query into a broad *"database connection"* cluster.
- A generic guide titled *"How to connect to databases"* scores higher than the specific troubleshooting manual 
  titled *"ERR_CONN_403 resolution"*, causing the search to fail!
""")

st.divider()

# --- SECTION 3: TOPIC 2 - SPARSE RETRIEVAL ---
st.header("3. Sparse Retrieval: High-Dimensional Vocabulary Spaces & Inverted Indexes", divider="rainbow")

st.markdown("""
**Sparse Retrieval** represents text within a vector space whose dimensionality equals the **entire vocabulary size `|V|`** 
(typically 50,000 to 250,000 unique words).

### The Mathematical Structure
For a document `D`, its sparse vector `v in R^|V|` contains zeros for almost all coordinates:
- Only words actually appearing in the document have non-zero weights.
- Stored as an **Inverted Index**: a mapping from each word token to its list of matching document IDs and frequencies (*postings list*).
- Searching a 10-word query requires scanning only the postings lists for those 10 words, executing in sub-millisecond CPU time!
""")

st.divider()

# --- SECTION 4: TOPIC 3 - BM25 ---
st.header("4. BM25: Best Matching 25 Formula, Term Frequency Saturation, & Length Normalization", divider="rainbow")

st.markdown("""
Pioneered by Stephen Robertson and Karen Spärck Jones in 1994, **BM25 (Best Matching 25)** is the most successful 
probabilistic information retrieval algorithm in computer science history.

### The BM25 Mathematical Formulation
For a document `D` and query `Q` containing tokens `q_1, q_2, ...`:

```text
BM25_Score(D, Q) = Sum_{q in Q} IDF(q) * [ ( f(q, D) * (k1 + 1) ) / ( f(q, D) + k1 * (1 - b + b * (|D| / avgdl)) ) ]
```

Where:
1. **Inverse Document Frequency (IDF)**:
   ```text
   IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
   ```
   Down-weights common words (like *"the"*, *"with"*) and exponentially rewards rare domain words (like *"ERR_CONN_403"* or *"Kyber-768"*).
2. **Term Frequency Saturation (`k1`)**:
   In naive TF-IDF, if a document repeats a word 20 times, its score increases 20-fold. In BM25, parameter `k1` (typically `1.2` to `1.5`) 
   enforces **asymptotic saturation**: repeating a word beyond 3 or 4 times provides diminishing marginal score increases.
3. **Document Length Normalization (`b`)**:
   Parameter `b` (typically `0.75`) normalizes for document length relative to the average document length `avgdl`. 
   Prevents long, rambling 5,000-word documents from winning simply because they contain more total words.
""")

st.divider()

# --- SECTION 5: TOPIC 4 - HYBRID SEARCH ---
st.header("5. Hybrid Search: Fusing Dense and Sparse Engines with Reciprocal Rank Fusion (RRF)", divider="rainbow")

st.markdown("""
In enterprise production, neither pure Dense nor pure BM25 is sufficient. 
**Hybrid Search** queries both engines in parallel and merges the candidate rankings.

### Why Simple Score Normalization Fails
A naive approach would be to linearly combine scores: `Final_Score = alpha * Dense_Score + (1 - alpha) * BM25_Score`.
- **The Scale Incompatibility Trap**: Cosine similarity is bounded in `[-1.0, 1.0]`, while BM25 scores are unbounded positive numbers (`0.0` to `45.0+`).
- Even with min-max normalization, a single query with an extreme BM25 outlier will completely overwhelm the dense score.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reciprocal Rank Fusion (RRF)")
    st.markdown("""
    Introduced by Cormack et al. (SIGIR 2009), **RRF** bypasses raw score magnitudes entirely by operating on **rank positions**:

    ```text
    RRF_Score(d) = Sum_{m in Methods} [ 1.0 / ( k + Rank_m(d) ) ]
    ```

    Where:
    - `Rank_m(d)` is the rank position (1, 2, 3...) of document `d` in system `m`.
    - `k` is a ranking smoothing constant (standard industry default: `k = 60`).
    """)

with col2:
    st.subheader("Why RRF is Robust")
    st.markdown("""
    - If a document is ranked #1 in BM25 and #2 in Dense, it receives a massive consensus boost:
      `1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.03251`.
    - A document that ranks #1 in only one engine and doesn't appear in the other receives only `1/(60+1) = 0.01639`.
    - Completely immune to scale differences, requires zero calibration, and consistently outperforms linear weighting across diverse benchmarks!
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - SEMANTIC SEARCH ---
st.header("6. Semantic Search: Intent Disambiguation, Paraphrasing, & Conceptual Proximity", divider="rainbow")

st.markdown("""
**Semantic Search** understands user intent, grammatical context, and conceptual relationships 
rather than performing literal string matching.

### Key Capabilities of Semantic Search
1. **Synonym & Paraphrase Mapping**: Recognizes that *"annual leave"* means *"vacation days"*, and *"automobile warranty"* means *"car coverage"*.
2. **Intent Disambiguation**: In context, recognizes whether *"bank"* refers to a financial institution or a riverbank.
3. **Cross-Lingual Retrieval**: A user searching in Spanish (*"política de reembolso"*) matches English documentation (*"reimbursement policy"*) 
   because modern multilingual embedding models map both languages to identical geometric regions in latent space.
""")

st.divider()

# --- SECTION 7: TOPIC 6 - KEYWORD SEARCH ---
st.header("7. Keyword Search: Lexical Precision, Exact Token Matches, & Regex Fallbacks", divider="rainbow")

st.markdown("""
While semantic search handles natural language questions, **Keyword Search** remains irreplaceable for deterministic technical queries:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Unmatched Use Cases for Keyword Search
    - **Code Identifiers & APIs**: `kubectl get pods -n kube-system`, `re.compile()`, `SELECT * FROM users`.
    - **Error Codes & Hashes**: `ERR_403_V2`, `0x80070005`, Git commit SHA `a3f91b`.
    - **Exact Product SKUs**: `SKU-9482-TX`, `Part #440-21`.
    - **Legal Statute Citations**: `Article 17, Section 3.b`.
    """)

with col2:
    st.markdown("""
    ### Keyword Search Engineering Challenges
    - **Stemming & Lemmatization**: Reducing words to base forms (e.g., *"running"* -> *"run"*).
    - **Stop-Word Pitfalls**: Naively stripping stop words (like *"to"*, *"be"*, *"or"*) destroys famous queries like *"to be or not to be"* or code syntax like *"in"*.
    - **Fuzzy Search Integration**: Using Levenshtein edit distance to handle typos (*"PotsgreSQL"* -> *"PostgreSQL"*).
    """)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: Complete Hybrid Search Engine with BM25 & RRF in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements a production hybrid search engine. 
It combines Robertson-Spärck Jones BM25, dense Cosine similarity, and Reciprocal Rank Fusion (RRF):
""")

code_block = r'''import re
import numpy as np
from typing import List, Dict, Tuple, Any

class HybridSearchEngine:
    """
    Self-contained hybrid search engine implementing:
    - BM25 keyword scoring with length normalization and IDF
    - Dense vector cosine similarity
    - Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.documents: List[Dict[str, Any]] = []
        
    def add_document(self, doc_id: str, title: str, text: str, vector: np.ndarray):
        tokens = [t.lower() for t in re.findall(r'\b[\w-]+\b', text)]
        norm_v = vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        self.documents.append({
            "id": doc_id,
            "title": title,
            "text": text,
            "tokens": tokens,
            "vector": norm_v
        })
        
    def search_bm25(self, query: str) -> List[Tuple[str, float]]:
        q_tokens = [t.lower() for t in re.findall(r'\b[\w-]+\b', query)]
        N = len(self.documents)
        avgdl = np.mean([len(d["tokens"]) for d in self.documents]) if N > 0 else 1.0
        
        scores = []
        for doc in self.documents:
            score = 0.0
            doc_len = len(doc["tokens"])
            for q in q_tokens:
                df = sum(1 for d in self.documents if q in d["tokens"])
                if df == 0:
                    continue
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc["tokens"].count(q)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                score += idf * (num / den)
            scores.append((doc["id"], float(score)))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search_dense(self, query_vector: np.ndarray) -> List[Tuple[str, float]]:
        q_norm = query_vector / np.linalg.norm(query_vector)
        scores = []
        for doc in self.documents:
            sim = float(np.dot(q_norm, doc["vector"]))
            scores.append((doc["id"], sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search_hybrid_rrf(self, query: str, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        bm25_res = self.search_bm25(query)
        dense_res = self.search_dense(query_vector)
        
        bm25_ranks = {did: rank + 1 for rank, (did, _) in enumerate(bm25_res)}
        dense_ranks = {did: rank + 1 for rank, (did, _) in enumerate(dense_res)}
        
        fused = []
        for doc in self.documents:
            did = doc["id"]
            r_b = bm25_ranks.get(did, 999)
            r_d = dense_ranks.get(did, 999)
            rrf_score = (1.0 / (self.rrf_k + r_b)) + (1.0 / (self.rrf_k + r_d))
            fused.append({
                "id": did,
                "title": doc["title"],
                "text": doc["text"],
                "bm25_rank": r_b,
                "dense_rank": r_d,
                "rrf_score": rrf_score
            })
            
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: Dense vs Sparse vs Hybrid Search Simulator", divider="rainbow")

st.markdown("""
Test how **Dense Semantic Retrieval**, **Sparse BM25 Keyword Search**, and **Reciprocal Rank Fusion (RRF)** 
perform on different query types in real time:
""")

queries = list(ProductionHybridSearchEngine.QUERY_VECTORS.keys())

wb_col1, wb_col2 = st.columns(2)
with wb_col1:
    selected_q = st.selectbox("Select Test Query", queries)
with wb_col2:
    rrf_k_param = st.slider("RRF Constant (k)", min_value=10, max_value=100, value=60, step=5)

# Execute searches
bm25_scores = ProductionHybridSearchEngine.compute_bm25(selected_q)
dense_scores = ProductionHybridSearchEngine.compute_dense(selected_q)
hybrid_scores = ProductionHybridSearchEngine.reciprocal_rank_fusion(bm25_scores, dense_scores, k=rrf_k_param)

# Display comparisons
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1. Sparse BM25 Ranks")
    t_bm25 = []
    for rank, (did, score) in enumerate(bm25_scores[:3], 1):
        t_bm25.append({"Rank": f"#{rank}", "Doc ID": did, "BM25 Score": f"{score:.3f}"})
    st.dataframe(pd.DataFrame(t_bm25), use_container_width=True)

with c2:
    st.subheader("2. Dense Vector Ranks")
    t_dense = []
    for rank, (did, score) in enumerate(dense_scores[:3], 1):
        t_dense.append({"Rank": f"#{rank}", "Doc ID": did, "Cosine Sim": f"{score:.3f}"})
    st.dataframe(pd.DataFrame(t_dense), use_container_width=True)

with c3:
    st.subheader("3. Fused Hybrid (RRF)")
    t_rrf = []
    for rank, item in enumerate(hybrid_scores[:3], 1):
        t_rrf.append({
            "Rank": f"#{rank}",
            "Doc ID": item["id"],
            "BM25": f"#{item['bm25_rank']}",
            "Dense": f"#{item['dense_rank']}",
            "RRF Score": f"{item['rrf_score']:.5f}"
        })
    st.dataframe(pd.DataFrame(t_rrf), use_container_width=True)

st.markdown("### Top Winner Excerpt")
winner = hybrid_scores[0]
st.success(f"**Top Fused Result**: `{winner['id']}` — {winner['title']}\n\n*{winner['text']}*")

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why is Reciprocal Rank Fusion (RRF) preferred over linear score normalization in production hybrid search?",
        "a": "Dense Cosine similarity produces scores bounded within [-1.0, 1.0], whereas sparse BM25 scores are unbounded positive numbers (typically 0 to 45+) with non-linear distributions that vary widely depending on query token rarity. Attempting to combine them via linear interpolation (alpha * Dense + (1 - alpha) * BM25) requires fragile dynamic min-max normalization that breaks when outlier scores occur. RRF evaluates only rank orders (1/(k + rank)), making it completely scale-invariant, robust to score distribution skew, and parameter-free without requiring calibration datasets."
    },
    {
        "q": "In the BM25 formula, what do parameters k1 and b control physically?",
        "a": "Parameter k1 controls term frequency saturation. Unlike linear TF (where 10 mentions score 10x higher than 1 mention), BM25 asymptotes toward an upper ceiling as term frequency rises. Lower k1 (e.g. 1.2) causes the score to saturate rapidly, treating repeated words similarly. Parameter b controls document length normalization: b=1.0 heavily penalizes documents that are longer than the average document length avgdl, while b=0.0 turns off length normalization entirely. The industry standard default is k1=1.5 and b=0.75."
    },
    {
        "q": "Why does pure dense semantic search fail on queries containing error codes or product SKUs?",
        "a": "Dense embedding models are trained to group text based on general semantic themes. When processing unique strings like 'ERR_CONN_403', subword tokenizers split the identifier into common subwords ('ERR', '_', 'CONN', '403'). The resulting dense vector blends into a broad 'database connection' cluster, losing the exact alphanumeric specificity. Sparse keyword search treats the identifier as an exact token in an inverted index with a high IDF weight, matching the precise troubleshooting manual instantly."
    },
    {
        "q": "What is the difference between traditional sparse search (BM25) and learned sparse models like SPLADE?",
        "a": "BM25 matches only exact literal words present in the document. Learned sparse models like SPLADE (SParse Lexical AnD Expansion) use a masked language model (BERT) to predict related vocabulary words during indexing. For example, if a document contains 'faucet repair', SPLADE assigns non-zero weights in the sparse vector not only to 'faucet' and 'repair', but also to 'plumbing', 'sink', and 'leak'. It stores these expanded weights in a standard inverted index, combining exact matching with learned synonym expansion."
    },
    {
        "q": "How does a production search router dynamically adjust between dense and sparse search?",
        "a": "Production systems use query classification routers: 1) If the query contains alphanumeric tokens matching regular expressions for error codes, SKUs, or function calls, the router weights BM25 heavily; 2) If the query is an open-ended conversational question or long conceptual inquiry, the router weights dense retrieval heavily; 3) In general multi-word enterprise queries, it executes hybrid search via RRF to capture both dimensions."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Score Range Distortion in Linear Hybrid Combination**
    Directly adding raw BM25 scores to Cosine similarity scores without normalization. BM25 scores can reach 30.0+ while Cosine 
    tops out at 1.0. The BM25 score completely overpowers the dense score, reducing the system to pure keyword search. 
    Always use Reciprocal Rank Fusion (RRF) instead.
    """)
    st.warning("""
    **Trap 3: Over-Aggressive Stop-Word Removal Stripping Code Tokens**
    Using standard English stop-word filters on programming documentation. Removing words like `in`, `is`, `for`, or `not` 
    corrupts technical queries like *"for loop in Python"* or *"not in SQL"*. Use code-aware tokenization.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Stale Average Document Length (avgdl) in Dynamic Ingestion**
    Updating documents in real time without refreshing the global `avgdl` and document count `N` metrics in the BM25 index. 
    As the corpus grows, stale length statistics cause BM25 scores to drift, distorting length normalization penalties.
    """)
    st.warning("""
    **Trap 4: Setting RRF Smoothing Constant `k` Too Small**
    Configuring `k = 1` or `k = 5` in RRF. A tiny `k` causes rank 1 to receive a massively disproportionate score 
    (`1/(1+1) = 0.5`) compared to rank 2 (`1/(1+2) = 0.33`), preventing consensus rank blending. Keep `k` at the standard default `k = 60`.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: RRF Formula"):
        st.markdown("**What is RRF?**\\n\\n`Score(d) = sum 1 / (k + rank_m(d))`\\n\\nRank-based scale-invariant fusion with standard `k=60`.")
with fc2:
    with st.popover("Flashcard 2: BM25 k1"):
        st.markdown("**What does k1 control?**\\n\\nTerm frequency saturation (`k1 in [1.2, 1.5]`). Prevents score from scaling linearly with word repeats.")
with fc3:
    with st.popover("Flashcard 3: BM25 b"):
        st.markdown("**What does b control?**\\n\\nDocument length normalization (`b=0.75`). Penalizes long documents relative to `avgdl`.")
with fc4:
    with st.popover("Flashcard 4: SPLADE"):
        st.markdown("**What is SPLADE?**\\n\\nLearned sparse embeddings using BERT to predict vocabulary expansion weights in inverted indexes.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why dense retrieval fails on exact alphanumeric identifiers and error codes.",
    "I can derive the BM25 formula and explain the physical roles of parameters k1 and b.",
    "I know why linear score combination fails across disparate metric ranges and why RRF is superior.",
    "I can implement Reciprocal Rank Fusion (1 / (k + rank)) in Python with constant k=60.",
    "I understand the difference between traditional sparse BM25 and learned sparse models like SPLADE.",
    "I can architect a dual-branch hybrid retrieval pipeline that queries dense and sparse engines in parallel."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of advanced retrieval techniques, dense vs sparse representations, BM25, and hybrid rank fusion.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
