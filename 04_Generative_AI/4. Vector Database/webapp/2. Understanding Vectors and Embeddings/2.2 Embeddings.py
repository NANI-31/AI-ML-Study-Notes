import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.2 Embeddings - Vector DB Curriculum",
    page_icon="🧬",
    layout="wide"
)

# Title & Overview
st.title("2.2 Embeddings: Dense vs. Sparse, Transformation Pipelines & Asymmetric Retrieval")
st.caption("Chapter 2: Understanding Vectors and Embeddings | Domain: 4. Vector Database")
st.markdown("""
An **embedding** is the mathematical bridge between human language and vector database storage. 
While traditional databases store text as literal strings of characters or tokens, artificial intelligence models 
transform discrete vocabulary into **continuous geometric coordinates** where spatial distance mirrors semantic meaning.

Understanding the internal mechanics of embeddings—how tokens are pooled into sentence vectors, the architectural 
differences between **Dense** and **Sparse** representations, and why **Query embeddings** must often be handled 
differently from **Document embeddings**—is critical for building production-grade retrieval systems.
""")

st.divider()

# Section 1: What Are Embeddings & How Text Becomes an Embedding
st.header("1. Anatomy of an Embedding: From Text to Geometric Vector", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1, 1])

with col_anat1:
    st.markdown("""
    ### 1. The Transformation Pipeline
    1. **Tokenization**:
       * Raw text is segmented into subword tokens using algorithms like Byte-Pair Encoding (BPE) or WordPiece.
       * *Example*: `"Vector databases are fast"` -> `[12431, 23981, 460, 2911]`.
    2. **Token Embedding Lookup & Positional Encoding**:
       * Each token ID looks up an initial static vector from an embedding matrix `W_e in R^(V x d_model)`.
       * Positional encodings (RoPE or learned absolute embeddings) are added to inject word order information.
    3. **Multi-Head Self-Attention Transformer Layers**:
       * Through 12 to 32 transformer layers, tokens exchange context dynamically.
       * The representation of `"fast"` in `"fast database"` becomes distinct from `"intermittent fast"`.
    4. **Pooling Layer (Aggregation to Single Vector)**:
       * The sequence of token vectors `[T_1, T_2, ..., T_L]` is aggregated into a single vector:
         * **Mean Pooling** (Standard for Sentence-Transformers/BGE): Average of all token representations.
         * **CLS Token Pooling** (Standard for BERT/OpenAI): Taking the first classification token's final hidden state.
         * **Last-Token Pooling** (Standard for Decoder models like Llama 3): Using the final token's hidden state.
    5. **L2 Normalization**:
       * The final pooled vector is scaled to unit length (`||v|| = 1.0`), ensuring cosine similarity equals the dot product.
    """)

with col_anat2:
    st.markdown("### The Transformer Embedding Generation Flow")
    flow_dot = """
    digraph EmbeddingFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InputText [label="Raw Text: 'Vector databases are fast'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Tokenizer [label="Subword Tokenizer (BPE)\\n[12431, 23981, 460, 2911]", fillcolor="#DBEAFE"];
        TokenLookup [label="Embedding Lookup Matrix + RoPE\\nStatic initial token vectors", fillcolor="#BFDBFE"];
        Transformer [label="N-Layer Transformer Encoder\\nMulti-Head Contextual Self-Attention", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
        TokenOutputs [label="Contextual Token Vectors (L x d)\\nT_1, T_2, T_3, ..., T_L", fillcolor="#DBEAFE"];
        
        Pooling [label="Pooling Layer\\n(Mean Pooling or CLS Token)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        Norm [label="L2 Normalization\\nv = v / ||v||", fillcolor="#A7F3D0"];
        FinalVector [label="Dense Embedding Vector in R^d\\n[0.024, -0.158, 0.891, ..., 0.043]", fillcolor="#6EE7B7", color="#047857", style="filled,bold"];

        InputText -> Tokenizer -> TokenLookup -> Transformer -> TokenOutputs -> Pooling -> Norm -> FinalVector;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# Section 2: Dense vs. Sparse Embeddings
st.header("2. Dense Embeddings vs. Sparse Embeddings", divider="rainbow")

st.markdown("""
Modern vector retrieval is not limited to dense vectors. The most robust retrieval architectures 
combine **Dense Embeddings** and **Sparse Embeddings** in hybrid retrieval systems.
""")

col_ds1, col_ds2 = col_ds1, col_ds2 = st.columns([1, 1])

with col_ds1:
    st.markdown("""
    ### Dense Embeddings
    * **Structure**:
      * Continuous floating-point array where **every single dimension has a non-zero value**.
      * Typical dimensions: 384, 768, 1024, 1536, 3072.
    * **Strengths**:
      * **Deep Semantic Understanding**: Recognizes synonyms (`"automobile"` matches `"car"`).
      * **Cross-Lingual Capabilities**: Connects queries in Spanish to documents in English.
      * **Concept Generalization**: Understands broad context, tone, and intent.
    * **Weaknesses**:
      * **Exact Keyword Blindness**: Struggles with uncommon product serial numbers (e.g. `SKU-8924-X`), rare proper nouns, or exact medical codes.
      * **The Lexical Matching Gap**: May return a semantically related document while missing the document containing the exact requested error code.
    * **Leading Models**: OpenAI `text-embedding-3`, BGE-M3, Voyage-ai, Cohere Embed v3.
    """)

with col_ds2:
    st.markdown("""
    ### Sparse Embeddings
    * **Structure**:
      * High-dimensional vector (e.g. 30,522 dimensions matching tokenizer vocabulary) where **99.9% of values are zero**.
      * Stored as key-value pairs: `{token_id: weight}`.
    * **Strengths**:
      * **Exact Lexical Precision**: Excels at serial numbers, programming API names, patient IDs, and acronyms.
      * **Explainability**: Every non-zero dimension corresponds to an interpretable vocabulary token.
      * **Efficient Inverted Indexing**: Can be stored in standard Lucene/BM25 inverted indexes or vector database sparse indexes.
    * **Weaknesses**:
      * **Vocabulary Mismatch Problem**: Fails completely when query and document express identical concepts using different words without synonym expansion.
    * **Leading Models**: BM25, SPLADE (Sparse Lexical and Expansion Model), BGE-M3 Sparse.
    """)

st.markdown("### Dense vs. Sparse Feature Matrix")
comp_df = pd.DataFrame({
    "Feature": ["Dimensionality", "Zero Values", "Primary Strength", "Primary Weakness", "Storage Structure", "Leading Algorithms"],
    "Dense Embeddings": ["384 to 3072", "Near 0% (Dense)", "Semantic meaning, synonyms, concepts", "Exact keyword & serial code recall", "HNSW / IVF-PQ Vector Index", "OpenAI, BGE-M3, SBERT"],
    "Sparse Embeddings (Learned)": ["30,000+ (Vocab size)", "99.5% to 99.9% (Sparse)", "Exact lexical match + token expansion", "Abstract conceptual understanding", "Inverted Posting Lists / Bitsets", "SPLADE, BGE-M3 Sparse"],
    "BM25 (Classic Sparse)": ["Vocab size", "99.9%+ (Sparse)", "Exact keyword frequency (TF-IDF)", "Zero semantic awareness", "Inverted Index (Lucene)", "BM25, Okapi BM25"]
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.divider()

# Section 3: Sentence, Document & Asymmetric Query Embeddings
st.header("3. Granularity Levels: Sentence, Document & Query Embeddings", divider="rainbow")

st.markdown("""
Not all embeddings are created equal. The length of the source text and the search role (Query vs Document) 
significantly influence embedding behavior.
""")

col_gran1, col_gran2, col_gran3 = col_gran1, col_gran2, col_gran3 = st.columns(3)

with col_gran1:
    st.markdown("""
    #### 1. Sentence Embeddings
    * **Scope**: 10 to 50 tokens (single sentences).
    * **Characteristics**:
      * Extremely dense semantic focus.
      * Highly distinct representations without topic dilution.
    * **Use Cases**:
      * FAQ matching, sentence similarity, semantic duplicate detection.
    * **Challenge**:
      * Misses macro-level context (e.g., pronouns like `"it"` or `"they"` lose referents).
    """)

with col_gran2:
    st.markdown("""
    #### 2. Document / Chunk Embeddings
    * **Scope**: 200 to 1,000 tokens (paragraphs, sections).
    * **Characteristics**:
      * Balances local detail with surrounding context.
      * Standard unit of retrieval in RAG pipelines.
    * **The Semantic Dilution Risk**:
      * Compressing 800 words into a single 768-d vector forces the model to average multiple disparate topics, weakening specific facts.
    """)

with col_gran3:
    st.markdown("""
    #### 3. Asymmetric Query Embeddings
    * **Scope**: Short search queries (3 to 15 tokens) searching long passages.
    * **Asymmetric Mismatch**:
      * A question (`"What is the capital of France?"`) has different syntax from the answer (`"Paris is the capital of France."`).
    * **The Prefix Solution**:
      * Asymmetric models require instruction prefixes:
        * Query: `query: What is the capital of France?`
        * Passage: `passage: Paris is the capital of France.`
      * Examples: E5, BGE, Instructor.
    """)

st.divider()

# Section 4: Interactive Dense vs Sparse & Asymmetric Workbench
st.header("4. Interactive Dense vs. Sparse & Asymmetric Retrieval Workbench", divider="rainbow")

st.markdown("""
Test how text is transformed into dense semantic embeddings and sparse lexical weights. 
Compare symmetric vs asymmetric query formulations and observe how similarity scores shift:
""")

col_wb_in, col_wb_out = st.columns([1, 1])

with col_wb_in:
    st.subheader("Input Text & Configuration")
    user_query = st.text_input(
        "User Search Query",
        value="How to resolve database connection timeout error 504?",
        help="Notice exact keywords ('error 504', 'timeout') vs semantic intent ('database connection failure')"
    )
    
    use_prefix = st.checkbox("Apply Asymmetric Retrieval Prefix ('query:' vs 'passage:')", value=True)
    
    candidate_passages = [
        "Passage 1 (Exact Lexical): Fix HTTP error 504 gateway timeout by adjusting upstream keepalive timeout settings.",
        "Passage 2 (Semantic Synonym): Database network connection dropped after exceeding 30-second handshake threshold.",
        "Passage 3 (Unrelated Topic): Modern cooking recipes recommend slow simmering for rich beef stew broth."
    ]
    
    st.markdown("**Corpus Passages to Search Against:**")
    for p in candidate_passages:
        st.text(p)

with col_wb_out:
    st.subheader("Simulated Retrieval & Scoring Analysis")
    
    # Simple deterministic simulation of dense and sparse scoring
    # Dense: captures semantic concepts (database, connection, timeout, network)
    # Sparse: rewards exact tokens ('error', '504', 'timeout')
    
    q_tokens = set(user_query.lower().replace("?", "").split())
    
    scores = []
    for p in candidate_passages:
        p_tokens = set(p.lower().replace(".", "").replace(":", "").split())
        
        # Sparse lexical score (Jaccard / term overlap)
        overlap = len(q_tokens & p_tokens)
        sparse_score = overlap / len(q_tokens | p_tokens)
        
        # Dense semantic score (concept proximity simulation)
        if "timeout" in p and "504" in p:
            dense_base = 0.88
        elif "connection dropped" in p or "handshake" in p:
            dense_base = 0.85 # High semantic overlap even without 'error 504'
        else:
            dense_base = 0.12 # Unrelated
            
        if use_prefix:
            dense_score = dense_base + 0.04 if dense_base > 0.5 else dense_base
        else:
            dense_score = dense_base
            
        hybrid_score = (0.5 * dense_score) + (0.5 * sparse_score * 2.5) # Normalized hybrid
        
        scores.append({
            "Passage": p[:45] + "...",
            "Dense Semantic Sim": min(dense_score, 0.99),
            "Sparse Lexical Overlap": sparse_score * 2.5,
            "Hybrid Reciprocal Score": hybrid_score
        })
        
    res_df = pd.DataFrame(scores)
    
    st.dataframe(
        res_df.style.format({
            "Dense Semantic Sim": "{:.3f}",
            "Sparse Lexical Overlap": "{:.3f}",
            "Hybrid Reciprocal Score": "{:.3f}"
        }).background_gradient(subset=["Hybrid Reciprocal Score"], cmap="Blues"),
        use_container_width=True,
        hide_index=True
    )
    
    st.info("""
    **Insight**: Notice how Passage 2 has low lexical overlap (0 exact matches for '504') but high dense similarity! 
    A pure keyword engine would miss it completely, whereas a hybrid vector system ranks both Passage 1 and 2 at the top.
    """)

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Building a Hybrid Dense + Sparse Vectorizer", divider="rainbow")

st.markdown("""
The following complete Python program implements a working **Hybrid Vectorizer** that produces 
both dense semantic vectors and sparse vocabulary-weight vectors, demonstrating hybrid score fusion:
""")

code_dense_sparse = '''import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

class SimpleHybridVectorizer:
    """Combines dense semantic embeddings with sparse lexical representations."""
    def __init__(self, dense_dim: int = 64):
        self.dense_dim = dense_dim
        # Global vocabulary for sparse indexing
        self.vocab: Dict[str, int] = {}
        self.vocab_size = 0

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace(".", "").replace("?", "").replace(",", "").split()

    def build_vocab(self, texts: List[str]):
        """Builds sparse vocabulary index."""
        unique_tokens = set()
        for t in texts:
            unique_tokens.update(self._tokenize(t))
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(unique_tokens))}
        self.vocab_size = len(self.vocab)

    def encode_dense(self, text: str, is_query: bool = False) -> np.ndarray:
        """Simulates dense transformer embedding with query/passage prefix projection."""
        np.random.seed(abs(hash(text)) % (2**32))
        raw_vec = np.random.randn(self.dense_dim)
        if is_query:
            # Query prefix bias
            raw_vec[0] += 0.5
        norm = np.linalg.norm(raw_vec)
        return raw_vec / (norm if norm > 0 else 1.0)

    def encode_sparse(self, text: str) -> Dict[int, float]:
        """Generates sparse vector: {token_id: frequency_weight}."""
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        sparse_vec = {}
        for tok, count in counts.items():
            if tok in self.vocab:
                sparse_vec[self.vocab[tok]] = float(count)
        return sparse_vec

    def sparse_dot_product(self, s1: Dict[int, float], s2: Dict[int, float]) -> float:
        """Efficient dot product between two sparse dictionaries."""
        score = 0.0
        # Iterate over smaller dictionary
        smaller, larger = (s1, s2) if len(s1) < len(s2) else (s2, s1)
        for idx, val in smaller.items():
            if idx in larger:
                score += val * larger[idx]
        return score

    def hybrid_search(self, query: str, documents: List[str], alpha: float = 0.7) -> List[Tuple[int, float]]:
        """Alpha: weight for dense similarity (1.0 = pure dense, 0.0 = pure sparse)."""
        q_dense = self.encode_dense(query, is_query=True)
        q_sparse = self.encode_sparse(query)

        results = []
        for i, doc in enumerate(documents):
            d_dense = self.encode_dense(doc, is_query=False)
            d_sparse = self.encode_sparse(doc)

            dense_score = float(np.dot(q_dense, d_dense))
            sparse_score = self.sparse_dot_product(q_sparse, d_sparse)

            # Combined hybrid score
            combined = (alpha * dense_score) + ((1 - alpha) * sparse_score)
            results.append((i, combined))

        return sorted(results, key=lambda x: x[1], reverse=True)

# Demonstration
corpus = [
    "PostgreSQL connection timeout 504 error during heavy load",
    "Database connection pool exhausted due to idle transactions",
    "How to prepare a healthy Mediterranean lunch salad"
]

vec = SimpleHybridVectorizer(dense_dim=8)
vec.build_vocab(corpus + ["connection timeout"])
query = "database connection timeout"
rankings = vec.hybrid_search(query, corpus, alpha=0.5)

print(f"Query: '{query}'")
for rank, (doc_idx, score) in enumerate(rankings, 1):
    print(f"Rank {rank} (Score: {score:.4f}): {corpus[doc_idx]}")
'''

st.code(code_dense_sparse, language="python")

if st.button("🚀 Run Hybrid Vectorizer in Sandbox", type="primary"):
    exec_scope = {}
    exec(code_dense_sparse, globals(), exec_scope)
    rankings_output = exec_scope["rankings"]
    corpus_docs = exec_scope["corpus"]
    
    st.success("Hybrid Vectorizer executed successfully!")
    for rank, (idx, sc) in enumerate(rankings_output, 1):
        st.markdown(f"**Rank {rank}** | `Score: {sc:.4f}` | Text: `{corpus_docs[idx]}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the fundamental difference between Dense and Sparse embeddings, and when should you choose Hybrid Retrieval?"""
a1 = """* **Dense Embeddings** (384-3072 continuous floats):
  * Map input text into continuous latent space where every dimension contains a non-zero value.
  * Superpower: Semantic conceptual matching, synonyms, paraphrase understanding, and cross-lingual alignment.
  * Weakness: Out-of-vocabulary terms, specific product part numbers (e.g. `RTX-4090-TI`), rare proper nouns, and exact regex-like string patterns.
* **Sparse Embeddings** (e.g. SPLADE, BM25):
  * Map text into a high-dimensional vocabulary space (e.g. 30,000+ dimensions) where 99.9% of weights are zero.
  * Superpower: Exact keyword recall, serial codes, medical identifiers, and complete interpretability.
* **Why Hybrid Retrieval is the Production Gold Standard**:
  * Combining dense semantic retrieval with sparse lexical retrieval (via Reciprocal Rank Fusion or linear interpolation) achieves the best of both worlds: robust conceptual understanding without sacrificing exact keyword precision."""

q2 = """2. Why do asymmetric embedding models (such as E5 and BGE) require 'query:' and 'passage:' instruction prefixes?"""
a2 = """* In traditional symmetric embedding models, both inputs are assumed to share identical document styles (e.g., comparing sentence A to sentence B for similarity).
* In search and QA, the query is typically short, vague, and phrased as a question (e.g., `'How to fix 504 gateway timeout?'`), while the document is long, detailed, and written as an affirmative explanation.
* If encoded symmetrically, the model might match the question to other questions rather than to informative answers.
* By adding instruction prefixes (`query:` vs `passage:`), the transformer's attention layers project the query into the **retrieval subspace** representing what the query is looking for, rather than what the query literally looks like."""

q3 = """3. Explain the trade-offs between Mean Pooling, CLS Token Pooling, and Last-Token Pooling in embedding generation."""
a3 = """* **Mean Pooling**: Computes the element-wise average of all non-padding token hidden states.
  * Advantages: Captures contributions from all words across the entire sequence. Highly stable. Standard in Sentence-Transformers and BGE.
* **CLS Token Pooling**: Uses the hidden state of the first special `[CLS]` token.
  * Advantages: Trained explicitly during pre-training to aggregate sequence-level representations (standard in BERT).
  * Disadvantages: Can overfit to pre-training classification objectives if not fine-tuned on contrastive loss.
* **Last-Token Pooling**: Uses the final hidden state of the sequence.
  * Advantages: Natural choice for autoregressive causal decoder models (e.g. Llama, Mistral, Gemma) where every subsequent token attends only to previous tokens, making the final token the only one with full sequence context."""

q4 = """4. What is Matryoshka Representation Learning (MRL), and why is it a game-changer for vector database costs?"""
a4 = """* Traditionally, an embedding model outputs a fixed dimensionality (e.g. 1536 floats). Truncating the vector degrades recall catastrophically.
* **Matryoshka Representation Learning (MRL)** (introduced in models like OpenAI `text-embedding-3` and BGE-M3):
  * Trains the model with nested loss functions across multiple dimensions simultaneously (e.g. first 64, 128, 256, 512, and 1536 dimensions).
  * Forces the model to pack the most important semantic information into the earliest dimensions.
* **Production Impact**:
  * You can truncate a 1536-d vector down to 256-d or 512-d with less than 1-2% loss in retrieval accuracy!
  * This slashes vector database RAM usage, disk storage, and ANN search latency by **66% to 83%** without requiring a new embedding model."""

q5 = """5. What is the Semantic Dilution problem in long-document embeddings, and how do modern architectures solve it?"""
a5 = """* Compressing a 10-page document into a single 768-d vector forces the model to average out thousands of distinct facts, diluting specific technical details into generic background noise.
* **Solutions**:
  1. **Chunking**: Segmenting documents into 200-512 token chunks with 50-token overlap.
  2. **Parent-Document Retrieval**: Embedding small child chunks (128 tokens) for precise search, but hydrating the larger parent section (1024 tokens) into the LLM context.
  3. **Late Interaction (ColBERT / ColPali)**: Storing multi-vector token embeddings and computing MaxSim late interaction across all query tokens rather than collapsing into a single vector."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Embedding Production Traps & War Stories")

st.markdown("""
> [!WARNING]
> **Trap 1: Omission of Asymmetric Prefixes in Production**  
> Models like `intfloat/e5-large-v2` or `BAAI/bge-large-en-v1.5` were explicitly fine-tuned with prefixes 
> (`query: ` and `passage: `). Passing raw queries without the prefix degrades Top-10 recall by up to 15-20%! 
> Always wrap your embedding helper functions with automated prefix injectors based on model metadata.

> [!WARNING]
> **Trap 2: Pooling Layer Implementation Mismatch**  
> If an embedding model was trained with Mean Pooling (e.g. Sentence-Transformers), but your deployment 
> service (e.g. a custom ONNX or TensorRT endpoint) extracts the `[CLS]` token hidden state instead, 
> the resulting vectors will look valid but produce completely corrupted similarity scores. 
> Always verify the pooling configuration in `modules.json` or `config.json`.

> [!WARNING]
> **Trap 3: Truncation Blindness on Long Context Windows**  
> Many developers pass 2,000-word articles into models with an 8k context window, unaware that the model's 
> positional embeddings were only trained on 512 tokens. Unchecked inputs will be silently truncated at token 512, 
> silently discarding the bottom 75% of the document! Always enforce explicit chunking upstream.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Dense Embedding?"):
        st.markdown("""
        **Dense Embedding**:  
        A continuous vector where every dimension contains a non-zero float, capturing deep semantic concepts, 
        synonyms, and broad context.
        """)
    with st.expander("🗂️ Flashcard: What is Sparse Embedding?"):
        st.markdown("""
        **Sparse Embedding**:  
        A high-dimensional vector keyed to vocabulary tokens where >99% of values are zero, excelling at exact 
        serial numbers, proper nouns, and keyword matching.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Matryoshka Learning (MRL)?"):
        st.markdown("""
        **Matryoshka Representation Learning (MRL)**:  
        A training technique where early vector dimensions store primary semantic signals, allowing vector 
        truncation (e.g., 1536 -> 512) to save 66% RAM with minimal accuracy loss.
        """)
    with st.expander("🗂️ Flashcard: Why Use Asymmetric Prefixes?"):
        st.markdown("""
        **Asymmetric Retrieval Prefixes**:  
        Directing the model to project short, interrogative queries and long, affirmative passages into a shared 
        search alignment space using prefixes like `query:` and `passage:`.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can explain the full lifecycle of text becoming a dense embedding through a transformer.", value=True)
chk2 = st.checkbox("I understand the architectural differences and complementary roles of Dense and Sparse embeddings.")
chk3 = st.checkbox("I know when to use Sentence embeddings vs. Document chunks vs. Asymmetric Query embeddings.")
chk4 = st.checkbox("I understand how Asymmetric prefixes (`query:` vs `passage:`) solve search intent mismatches.")
chk5 = st.checkbox("I can articulate how Matryoshka Representation Learning (MRL) reduces vector database storage costs.")
chk6 = st.checkbox("I know how to implement Mean Pooling vs. CLS Pooling vs. Last-Token Pooling.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Embeddings and are ready for Chapter 2.3: Embedding Models!")
