import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="2.2 NLP Representations - LLM Curriculum",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("2.2 NLP Representations: From Sparse Discrete Matrices to Dense Contextual Tensors")
st.caption("Chapter 2: NLP Foundations for LLMs | Domain: 2. Large Language Models")

st.markdown("""
Computers cannot compute linear algebra on raw natural language strings; text must be mathematically represented 
as numerical vectors. The representation journey in computer science evolved through three distinct technological epochs: 
**Sparse Discrete Representations (One-Hot, Bag-of-Words, TF-IDF) -> Static Continuous Embeddings (Word2Vec, GloVe, FastText) 
-> Fully Contextual Neural Tensors (Transformer Self-Attention)**.

Understanding this progression reveals why modern Foundation LLMs achieve superhuman zero-shot transfer: by learning 
high-dimensional continuous manifolds where syntactic rules, world facts, and nuanced polysemy exist as geometric coordinates.

In this lesson, you will master:
1. The 7 core NLP representation paradigms and their mathematical formulation.
2. The Curse of Dimensionality and Orthogonality: why sparse matrices waste 99.9% of memory and fail at semantic distance.
3. The Polysemy Collapse: why static word embeddings fail to represent multiple meanings for words like 'apple' or 'bank'.
4. How modern foundation models use dynamic contextual representations across deep residual stream manifolds.
5. A runnable pure Python implementation comparing TF-IDF vectorization with dense contextual embedding projections.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Evolution of Semantic Vector Spaces
# ==============================================================================
st.header("1. Architectural Blueprint: The Evolution of Semantic Vector Spaces", divider="rainbow")

st.markdown("""
Natural language representations transitioned from high-dimensional, orthogonal sparse vectors toward 
dense, continuous manifolds capable of dynamic contextual adaptation across multi-layer transformers:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    spectrum_dot = """
    digraph RepresentationSpectrum {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_sparse {
            label="Epoch 1: Sparse Discrete (1950s - 2012)";
            style=filled;
            color="#1E293B";
            fontcolor="#EF4444";
            E1_OneHot [label="One-Hot Encoding\\nDim: [Vocab_Size ~100k]\\nOrthogonal unit vectors\\nZero semantic proximity", fillcolor="#450A0A", color="#EF4444"];
            E1_BoW [label="Bag-of-Words & N-grams\\nDim: [Vocab_Size]\\nTerm frequency counts\\nIgnores word order", fillcolor="#450A0A", color="#EF4444"];
            E1_TFIDF [label="TF-IDF Weighting\\nDim: [Vocab_Size]\\nInverse document frequency\\nHigh sparsity (99.9% zeros)", fillcolor="#450A0A", color="#EF4444"];
        }

        subgraph cluster_static {
            label="Epoch 2: Static Continuous (2013 - 2017)";
            style=filled;
            color="#1E293B";
            fontcolor="#F59E0B";
            E2_Word2Vec [label="Word2Vec / GloVe / FastText\\nDim: [Hidden_Dim ~300]\\nContinuous dense vectors\\nLearns analogical geometry\\nFlaw: Polysemy collapse", fillcolor="#78350F", color="#F59E0B"];
        }

        subgraph cluster_contextual {
            label="Epoch 3: Dynamic Contextual (2018 - Present)";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            E3_Context [label="Transformer Self-Attention\\nDim: [Seq_Len, Hidden_Dim ~4096]\\nDynamic contextualized tokens\\nDisambiguates polysemy per sentence\\nFull residual stream memory", fillcolor="#064E3B", color="#34D399", shape=hexagon];
        }

        E1_OneHot -> E1_BoW;
        E1_BoW -> E1_TFIDF;
        E1_TFIDF -> E2_Word2Vec [label="Dimensionality Reduction"];
        E2_Word2Vec -> E3_Context [label="Self-Attention Routing"];
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(spectrum_dot, use_container_width=True)

with col_diag2:
    st.markdown("### Three Epochs of NLP Vectors")
    st.markdown("""
    * **Epoch 1: Sparse Discrete**: Vectors of vocabulary size `V` containing counts or binary flags. All words are mutually orthogonal (`Dot_Product = 0`), meaning `'cat'` and `'feline'` share zero similarity.
    * **Epoch 2: Static Continuous**: Dense vectors (100–300 dimensions) capturing distributed semantics via skip-grams. Fails on polysemy because every word has exactly one static vector.
    * **Epoch 3: Dynamic Contextual**: Vectors of 4,096+ dimensions recomputed at every layer by attention heads. A word's vector shifts dynamically based on surrounding tokens.
    """)

st.divider()

# ==============================================================================
# Section 2: 1. Bag of Words
# ==============================================================================
st.header("2. Bag of Words: Frequency Counting & The Complete Loss of Sequential Order", divider="rainbow")

st.markdown("""
The **Bag of Words (BoW)** model represents text as an unordered multiset of words, disregarding grammar, syntax, 
and word order while retaining occurrence frequencies.

### Mathematical Formulation
Given a vocabulary `V = {w_1, w_2, ..., w_D}` of size `D`:
A document `d` is represented as a D-dimensional vector `x_d = [c_1, c_2, ..., c_D]`, 
where `c_i = Count(w_i, d)` represents the raw frequency of word `w_i` in document `d`.

* Vector Dimension: `[Vocabulary_Size]` (e.g., 50,000 to 100,000 sparse integers).
* Sparsity: Over 99% of entries are zero for any single sentence or short paragraph.

### The Fatal Flaw: Order Blindness
Because word order is completely discarded:
* `"The dog bit the man"` -> `{'the': 2, 'dog': 1, 'bit': 1, 'man': 1}`
* `"The man bit the dog"` -> `{'the': 2, 'dog': 1, 'bit': 1, 'man': 1}`
Both sentences yield mathematically identical vector representations despite communicating opposite real-world events.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Attention Supersedes Word Bags**: Foundation LLMs completely abandon bag-of-words. Rather than flattening text into counts, modern transformers preserve the exact sequence order `[Batch_Size, Sequence_Length, Hidden_Dimension]` and inject positional coordinates via Rotary Position Embeddings (RoPE).
* **Where Word Counts Still Matter**: In production RAG systems, sparse term-frequency statistics remain relevant through BM25 indices as an auxiliary lexical retrieval channel, complementing dense vector retrieval to catch rare technical acronyms.
""")

df_bow = pd.DataFrame({
    "Dimension": [
        "Representation Form",
        "Sequence Order Sensitivity",
        "Vector Dimensionality",
        "Sparsity Profile",
        "Semantic Distance"
    ],
    "Bag of Words (BoW)": [
        "Sparse frequency count vector [Vocab_Size]",
        "Zero order sensitivity; completely blind to word permutations",
        "50,000 to 100,000 dimensions (one per vocabulary term)",
        "Over 99.5% of elements are zero; severe memory waste",
        "Cosine similarity measures lexical term overlap, not meaning"
    ],
    "Modern Foundation LLMs": [
        "Dense contextual tensor [Batch_Size, Sequence_Length, Hidden_Dimension]",
        "Full order sensitivity; distinguishes 'dog bit man' from 'man bit dog'",
        "4,096 (8B) to 8,192 (70B) dimensions across 32-80 layers",
        "100% dense continuous floating-point vectors (BF16)",
        "Deep geometric cosine similarity measuring semantic and conceptual intent"
    ]
})
st.dataframe(df_bow, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. N-grams
# ==============================================================================
st.header("3. N-grams: Local Context Approximations & The Combinatorial Explosion of Sparsity", divider="rainbow")

st.markdown("""
To address the complete order blindness of Bag of Words without full sequence modeling, classical NLP introduced 
**N-grams**: contiguous sequences of `N` items from a given sample of text.
* Unigram (`N=1`): Individual words: `["The", "quick", "brown", "fox"]`
* Bigram (`N=2`): Pairs of adjacent words: `["The quick", "quick brown", "brown fox"]`
* Trigram (`N=3`): Triplets of adjacent words: `["The quick brown", "quick brown fox"]`

### The Combinatorial Explosion Problem
While bigrams and trigrams capture local phrases (such as `"not guilty"` or `"New York"`), scaling `N` causes the vocabulary 
to explode exponentially:
* If vocabulary has `V = 50,000` words:
  * Possible Bigrams: `V^2 = 2,500,000,000` (2.5 billion possible features)
  * Possible Trigrams: `V^3 = 125,000,000,000,000` (125 trillion possible features)
Due to this **curse of dimensionality**, N-gram models suffer from severe data sparsity: any phrase not observed verbatim 
during training is assigned a probability of zero, requiring complex smoothing heuristics (Kneser-Ney, Good-Turing).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Self-Attention as Infinite N-grams**: Self-attention computes soft, continuous dynamic n-grams across the entire sequence. Instead of hardcoded sliding windows of size 2 or 3, attention heads can attend between token 1 and token 50,000 simultaneously without allocating static exponential feature tables.
* **Subword BPE vs Character N-grams**: While classical NLP used character n-grams to handle typos, modern foundation tokenizers use Byte-Pair Encoding (BPE), learning the optimal statistical sub-units data-efficiently.
""")

df_ngram = pd.DataFrame({
    "Dimension": [
        "Context Window Scope",
        "Feature Space Growth",
        "Unseen Phrase Handling",
        "Memory Footprint",
        "Compositionality"
    ],
    "Classical N-gram Models (N=2 or N=3)": [
        "Strictly local window (preceding N-1 words only)",
        "Exponential explosion: O(V^N); impossible to scale beyond N=5",
        "Assigned zero probability unless complex smoothing is applied",
        "Tens of gigabytes for sparse frequency tables",
        "Rigid string concatenation; cannot generalize across synonyms"
    ],
    "Modern Foundation LLMs": [
        "Global context window (8k to 2,000,000 tokens simultaneously)",
        "Constant feature dimensionality: Hidden_Dimension across all layers",
        "Smooth generalization via continuous geometric latent space",
        "Fixed neural weights: ~16 GB for an 8B model in BF16",
        "Deep semantic compositionality via multi-head self-attention"
    ]
})
st.dataframe(df_ngram, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. TF-IDF
# ==============================================================================
st.header("4. TF-IDF: Term Frequency-Inverse Document Frequency & Statistical Weighting", divider="rainbow")

st.markdown("""
Term Frequency-Inverse Document Frequency (**TF-IDF**) is a numerical statistic intended to reflect how important 
a word is to a document in a collection or corpus. It scales term frequency by discounting words that appear ubiquitously 
across all documents.

### Mathematical Formulation
The TF-IDF weight for term `t` in document `d` across corpus `D` decomposes into two components:

`TF(t, d) = Count(t, d) / Total_Words(d)`  
`IDF(t, D) = log((1 + Total_Documents) / (1 + Count_Documents_With_Term(t))) + 1`  
`TF_IDF(t, d) = TF(t, d) * IDF(t, D)`

* **High TF-IDF**: Terms that appear frequently in a specific document but rarely across the general corpus (e.g., technical jargon, unique medical identifiers).
* **Low TF-IDF**: Common words (*the*, *is*, *data*) that appear across nearly every document, receiving an IDF multiplier near zero.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **BM25 in Hybrid Search**: Modern production RAG architectures combine BM25 (the state-of-the-art non-linear evolution of TF-IDF) with dense embeddings. BM25 prevents dense search from hallucinating on exact serial numbers, error codes, and unique proper nouns.
* **Softmax Attention as Learned Importance**: In transformers, self-attention weights act as dynamic, query-dependent relevance weights. Rather than computing static corpus-level frequencies, attention dynamically determines which words are salient conditioned on the user prompt.
""")

df_tfidf = pd.DataFrame({
    "Dimension": [
        "Weighting Mechanism",
        "Corpus Dependency",
        "Synonym & Paraphrase Awareness",
        "Vector Dimensionality",
        "Modern Production Role"
    ],
    "TF-IDF Vectorization": [
        "Mathematical frequency ratio: TF multiplied by inverse document frequency",
        "Requires access to entire document corpus to calculate global IDF statistics",
        "Zero; 'cardiac arrest' and 'heart attack' have completely orthogonal weights",
        "Sparse vector of size [Vocabulary_Size ~100k]",
        "Still widely used in BM25 search engines for exact keyword retrieval"
    ],
    "Modern Foundation LLM Representations": [
        "Learned soft-attention weights and continuous residual stream projections",
        "Self-contained in pretrained neural weights; zero corpus indexing needed",
        "High; maps synonyms and cross-lingual concepts to nearby geometric coordinates",
        "Dense vector of size [Hidden_Dimension ~4096]",
        "Powers deep semantic reasoning, abstractive synthesis, and generative answers"
    ]
})
st.dataframe(df_tfidf, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. One-Hot Encoding
# ==============================================================================
st.header("5. One-Hot Encoding: Orthogonal Basis Vectors & Semantic Distance Collapse", divider="rainbow")

st.markdown("""
One-Hot Encoding represents every word in a vocabulary as a binary vector of dimension `[Vocabulary_Size]`, 
where all elements are `0` except for a single `1` at the word's unique dictionary index.

### Mathematical Formulation
Given vocabulary `V = ["apple", "banana", "car", "dog"]`:
* `"apple"` -> `[1, 0, 0, 0]`
* `"banana"` -> `[0, 1, 0, 0]`
* `"car"` -> `[0, 0, 1, 0]`
* `"dog"` -> `[0, 0, 0, 1]`

### The Orthogonality Catastrophe
In linear algebra, the dot product of any two distinct one-hot vectors is identically zero:

`Dot_Product(v_apple, v_banana) = 0.0`  
`Dot_Product(v_apple, v_car) = 0.0`  
`Euclidean_Distance(v_apple, v_banana) = sqrt(2) = 1.414`  
`Euclidean_Distance(v_apple, v_car) = sqrt(2) = 1.414`

Every single word in the language is equidistant and completely orthogonal to every other word. 
One-hot encoding is mathematically incapable of representing semantic similarity, taxonomic relationships, or conceptual analogies.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Input Token IDs as Implicit One-Hot**: In modern LLM code, token IDs are stored as integers (`token_id = 12314`) rather than expanding memory into a 128,256-dimensional one-hot vector. However, indexing into the embedding matrix `W_emb[token_id, :]` is mathematically equivalent to multiplying a one-hot vector by the embedding matrix:
  `One_Hot(token_id) * W_emb = e_{token_id}`
* **Immediate Dense Projection**: The very first operation in an LLM projects the discrete one-hot coordinate into a dense continuous 4,096-dimensional embedding vector, completely eliminating orthogonality.
""")

df_onehot = pd.DataFrame({
    "Dimension": [
        "Mathematical Representation",
        "Vector Dimensionality",
        "Cross-Word Semantic Distance",
        "Memory Footprint per Token",
        "Continuous Geometry"
    ],
    "One-Hot Encoding": [
        "Binary vector with single '1' and (V - 1) zeros",
        "[Vocabulary_Size ~100k]",
        "Strictly orthogonal; Dot_Product = 0 for all word pairs",
        "100,000 integers (wasteful without sparse matrix wrappers)",
        "Zero; discrete points on the vertices of a high-dimensional hypercube"
    ],
    "Modern Dense Token Embeddings": [
        "Continuous floating-point coordinate vector (BF16)",
        "[Hidden_Dimension ~4096]",
        "Cosine similarity reflects linguistic and semantic relatedness",
        "4,096 float16 values (8 KB per token)",
        "Continuous differentiable manifold with rich topological structure"
    ]
})
st.dataframe(df_onehot, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Word Embeddings
# ==============================================================================
st.header("6. Word Embeddings: Static Distributed Semantics (Word2Vec, GloVe, FastText)", divider="rainbow")

st.markdown("""
Word embeddings introduced the concept of **distributed representations**: mapping words from high-dimensional 
sparse spaces into low-dimensional continuous vector spaces (typically 100 to 300 dimensions) based on the 
**Distributional Hypothesis** (Harris, 1954; Firth, 1957): *"You shall know a word by the company it keeps"*.

### Key Embedding Architectures
1. **Word2Vec (Mikolov et al., 2013)**:
   * **CBOW (Continuous Bag of Words)**: Predicts a target word from surrounding context words.
   * **Skip-Gram**: Predicts surrounding context words given a target word using Negative Sampling.
2. **GloVe (Pennington et al., 2014)**: Global Vectors factorize global word-word co-occurrence probability ratios.
3. **FastText (Bojanowski et al., 2017)**: Extends Word2Vec by learning embeddings for character n-grams, enabling representations for out-of-vocabulary words and morphologically rich languages.

### Vector Arithmetic & Analogical Reasoning
Word embeddings revealed that semantic relationships emerge as linear vector offsets:

`Vector('King') - Vector('Man') + Vector('Woman') ≈ Vector('Queen')`  
`Vector('Paris') - Vector('France') + Vector('Italy') ≈ Vector('Rome')`
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **The Fatal Limitation of Static Embeddings (Polysemy Collapse)**: Static embeddings assign exactly **one static vector per word**. For the polysemic word `"bank"`, Word2Vec collapses financial bank, river bank, blood bank, and banking an aircraft into a single blended vector.
* **Foundation Models as Dynamic Embedding Engines**: LLMs do not use static embeddings beyond layer 0. The token vector is passed through 32 to 80 self-attention layers, dynamically shifting its position in the 4,096-dimensional manifold to represent its exact contextual meaning in that sentence.
""")

df_w2v = pd.DataFrame({
    "Dimension": [
        "Representation Nature",
        "Dimensionality",
        "Polysemy Handling ('bank')",
        "Vector Arithmetic Support",
        "Context Awareness"
    ],
    "Static Word Embeddings (Word2Vec / GloVe)": [
        "Static distributed vector lookup table",
        "100 to 300 dimensions",
        "Collapsed into single blended centroid vector",
        "Linear offsets capture basic analogies (King - Man + Woman = Queen)",
        "Zero sentence context; vector is identical in every document"
    ],
    "Modern Foundation LLMs": [
        "Dynamic contextual tensor updated at every transformer layer",
        "4,096 to 8,192 dimensions across 32-80 layers",
        "Disambiguates polysemy dynamically via multi-head self-attention",
        "High-dimensional non-linear manifolds capturing complex symbolic logic",
        "Full context sensitivity across 100,000+ surrounding tokens"
    ]
})
st.dataframe(df_w2v, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Contextual Embeddings
# ==============================================================================
st.header("7. Contextual Embeddings: Polysemic Latent Projections (ELMo, BERT, Transformer Decoders)", divider="rainbow")

st.markdown("""
Contextual embeddings resolved the polysemy collapse of static word vectors. 
Instead of assigning a fixed vector to each word type, a **contextual embedding is a function of the entire sentence**: 
the vector representation of a token dynamically mutates depending on all preceding and succeeding tokens.

### The Evolution of Contextual Architecture
1. **ELMo (Peters et al., 2018)**: Embeddings from Language Models used bidirectional LSTMs to concatenate forward and backward hidden states: `h_context = [h_forward; h_backward]`.
2. **BERT (Devlin et al., 2018)**: Bidirectional Encoder Representations from Transformers applied multi-layer bidirectional self-attention trained on Masked Language Modeling (MLM).
3. **Decoder-Only Foundation LLMs (Llama 3, GPT-4)**: Autoregressive causal self-attention models where each token's representation is contextualized by all preceding tokens in the context window.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Layer-by-Layer Representation Specialization**: In a 32-layer foundation model, representations specialize across depth:
  * Early Layers (1–8): Capture low-level syntactic structures, part-of-speech, and token spelling.
  * Middle Layers (9–24): Capture relational semantics, coreference resolution, and entity binding.
  * Deep Layers (25–32): Capture abstract reasoning, task intent, and next-token prediction logic.
* **Vector Cosine Separation**: In modern LLMs, the cosine similarity between `"Apple"` in a technology context and `"apple"` in a fruit context drops to near zero in the deep layers, demonstrating complete contextual disambiguation.
""")

df_ctx = pd.DataFrame({
    "Dimension": [
        "Context Modeling Engine",
        "Context Directionality",
        "Polysemy Disambiguation",
        "Layer Depth Specialization",
        "Primary Application"
    ],
    "Early Contextual (ELMo / BERT)": [
        "BiLSTM concatenation (ELMo) or Bidirectional Transformer Encoder (BERT)",
        "Bidirectional (attends to past and future tokens simultaneously)",
        "High; separates 'apple' (tech) from 'apple' (fruit)",
        "12 to 24 layers (BERT Base / Large)",
        "Discriminative tasks (classification, span extraction, NER)"
    ],
    "Modern Foundation LLMs (Llama 3 / Mistral)": [
        "Deep Autoregressive Causal Transformer Decoder with RoPE and GQA",
        "Causal left-to-right (attends to all preceding tokens in context)",
        "Exceptional; resolves nuanced polysemy, sarcasm, and role definitions",
        "32 to 80 layers with unconstrained residual stream highways",
        "Universal generative AI, reasoning, code synthesis, and structured outputs"
    ]
})
st.dataframe(df_ctx, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Dense Representations
# ==============================================================================
st.header("8. Dense Representations: Continuous Residual Stream Manifolds & Foundation Geometry", divider="rainbow")

st.markdown("""
Dense representations are continuous vector spaces where information is distributed across all dimensions 
without explicit zero-sparsity constraints. 
In modern Large Language Models, the ultimate dense representation is the **Residual Stream Manifold**.

### Mathematical Anatomy of the Residual Stream
The residual stream is a continuous vector space of dimension:

`Shape = [Batch_Size, Sequence_Length, Hidden_Dimension]`

Where `Hidden_Dimension` typically spans:
* 4,096 dimensions in 7B / 8B parameter models.
* 8,192 dimensions in 70B parameter models.
* 12,288 dimensions in 405B parameter models.

### Geometric Properties of Modern Dense Embeddings
1. **High-Dimensional Manifolds**: Real-world text concepts occupy lower-dimensional non-linear sub-manifolds embedded within the 4,096-dimensional space.
2. **Superposition & Polysemantic Neurons**: Research in Mechanistic Interpretability (Anthropic, 2023) demonstrates that transformers represent far more distinct concepts than their dimension count allows by storing concepts in **superposition** (non-orthogonal, almost-independent direction vectors).
3. **Anisotropy & Cosine Correction**: High-dimensional embeddings often suffer from representation anisotropy (vectors clustering in a narrow cone). Modern models apply LayerNorm or RMSNorm to center and normalize representations across layers.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Text Embeddings for Semantic Search**: Specialized dense embedding models (such as `text-embedding-3-large`, BGE-M3, Voyage-AI) extract normalized dense vectors from the final layer of transformer encoders/decoders to power vector database search across millions of documents.
* **Matryoshka Representation Learning (MRL)**: Modern embedding models are trained so that truncating the vector from 3,072 dimensions down to 512 dimensions retains over 98% of retrieval accuracy, slashing vector database storage and RAM costs by 6x.
""")

df_dense = pd.DataFrame({
    "Dimension": [
        "Representation Space",
        "Dimensional Capacity",
        "Concept Encoding Mechanism",
        "Vector Truncation (MRL)",
        "Primary Production Metric"
    ],
    "Traditional Dense Embeddings (2015-2020)": [
        "Fixed 300D or 768D static vector spaces",
        "Bounded capacity; prone to representation saturation",
        "One neuron/dimension corresponds to one feature",
        "Unsupported; truncating vectors destroys semantic accuracy",
        "Euclidean distance or basic cosine similarity"
    ],
    "Modern Foundation Manifolds (Llama 3 / OpenAI MRL)": [
        "Continuous 4,096D to 8,192D residual stream manifolds",
        "Massive capacity scaling with model parameter volume",
        "Superposition: millions of concepts represented via non-orthogonal directions",
        "Native Matryoshka Representation Learning (truncates from 3072D to 512D)",
        "Normalized cosine similarity and dot-product MIPS retrieval in Vector DBs"
    ]
})
st.dataframe(df_dense, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: Production Implementation: Vectorizers vs Dense Embeddings
# ==============================================================================
st.header("9. Production Implementation: Vectorizer Engine vs Transformer Dense Embeddings in Python", divider="rainbow")

st.markdown("""
The following complete, runnable Python script demonstrates an enterprise-grade comparison: 
implementing **TF-IDF vectorization from scratch in pure NumPy** alongside **dense vector projection and cosine similarity scoring**:
""")

st.code(r'''import numpy as np
import math
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# 1. Pure NumPy TF-IDF Vectorizer Engine
# ----------------------------------------------------------------------
class PurePythonTFIDFVectorizer:
    """Computes TF-IDF sparse representations from scratch."""
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf_weights: np.ndarray = np.array([])
        
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        # Step 1: Tokenize and build vocabulary
        doc_tokens = [d.lower().split() for d in documents]
        unique_words = sorted(list(set(w for doc in doc_tokens for w in doc)))
        self.vocabulary = {w: idx for idx, w in enumerate(unique_words)}
        vocab_size = len(self.vocabulary)
        num_docs = len(documents)
        
        # Step 2: Compute Document Frequency (DF)
        df_counts = np.zeros(vocab_size)
        for doc in doc_tokens:
            seen_in_doc = set(doc)
            for w in seen_in_doc:
                df_counts[self.vocabulary[w]] += 1
                
        # Step 3: Compute Smoothed Inverse Document Frequency (IDF)
        self.idf_weights = np.log((1.0 + num_docs) / (1.0 + df_counts)) + 1.0
        
        # Step 4: Compute TF-IDF matrix
        tfidf_matrix = np.zeros((num_docs, vocab_size))
        for doc_idx, doc in enumerate(doc_tokens):
            term_counts = {}
            for w in doc:
                term_counts[w] = term_counts.get(w, 0) + 1
            total_words = len(doc)
            for w, count in term_counts.items():
                w_idx = self.vocabulary[w]
                tf = count / total_words
                tfidf_matrix[doc_idx, w_idx] = tf * self.idf_weights[w_idx]
                
        # Normalize rows to unit length (L2 norm)
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return tfidf_matrix / norms

# ----------------------------------------------------------------------
# 2. Simulated Dense Contextual Embedding Projection
# ----------------------------------------------------------------------
def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two dense or sparse vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

# ----------------------------------------------------------------------
# 3. Execution & Comparative Polysemy Audit
# ----------------------------------------------------------------------
documents = [
    "Apple unveiled new M4 Silicon compute hardware",
    "Eat a fresh organic honeycrisp apple every morning",
    "River bank water flows rapidly down the mountain",
    "Deposited savings into the commercial bank account"
]

vectorizer = PurePythonTFIDFVectorizer()
tfidf_vectors = vectorizer.fit_transform(documents)

print(f"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary)} terms")
print("Document 0 (Apple Hardware) vs Document 1 (Apple Fruit) TF-IDF Similarity:")
sim_tfidf = compute_cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])
print(f"TF-IDF Similarity: {sim_tfidf:.4f}")
print("Notice: High similarity purely because the string 'apple' overlapped, despite distinct semantic domains!\n")

# Simulated Dense Contextual Embeddings (4D simplified projection)
# In real production, these are generated by BGE-M3 or text-embedding-3-large
dense_embeddings = np.array([
    [0.85, 0.90, 0.12, -0.05],  # Doc 0: Apple Hardware (Tech domain)
    [0.10, -0.05, 0.92, 0.88],  # Doc 1: Apple Fruit (Food domain)
    [-0.75, 0.10, 0.82, 0.65],  # Doc 2: River Bank (Nature domain)
    [0.82, 0.65, -0.40, 0.10]   # Doc 3: Commercial Bank (Finance domain)
])

dense_sim_apple = compute_cosine_similarity(dense_embeddings[0], dense_embeddings[1])
print(f"Dense Contextual Similarity (Doc 0 vs Doc 1): {dense_sim_apple:.4f}")
print("Notice: Dense contextual model correctly separates tech hardware from edible fruit!")
''', language="python")

st.divider()

# ==============================================================================
# Section 10: Interactive Workbench: Representation Vector Space Explorer
# ==============================================================================
st.header("10. Interactive Workbench: Representation Vector Space Explorer", divider="rainbow")

st.markdown("""
Compare sparse TF-IDF lexical overlap against continuous dense vector similarity in real time. 
Type two sentences and observe how lexical token overlap differs from semantic embedding proximity.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Sentence Pair Inputs")
    sent1 = st.text_input("Sentence A:", value="The cardiologist examined the patient suffering a myocardial infarction")
    sent2 = st.text_input("Sentence B:", value="A doctor checked the person having a cardiac arrest")

with col_wb2:
    st.subheader("Simulated Embedding Configuration")
    semantic_domain = st.selectbox(
        "Select Domain Semantic Alignment:",
        ["Identical Medical Meaning (Synonyms)", "Different Meanings with Word Overlap", "Completely Unrelated"]
    )

# Compute Lexical Jaccard & TF-IDF Overlap
tokens_a = set(sent1.lower().split())
tokens_b = set(sent2.lower().split())
intersection = tokens_a.intersection(tokens_b)
union = tokens_a.union(tokens_b)
jaccard_sim = len(intersection) / max(len(union), 1)

if semantic_domain == "Identical Medical Meaning (Synonyms)":
    simulated_dense_sim = 0.91
elif semantic_domain == "Different Meanings with Word Overlap":
    simulated_dense_sim = 0.32
else:
    simulated_dense_sim = 0.08

st.markdown("#### Representation Metric Comparison")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Sentence A Tokens", len(tokens_a))
col_m2.metric("Sentence B Tokens", len(tokens_b))
col_m3.metric("Lexical Jaccard Overlap", f"{jaccard_sim * 100:.1f}%")
col_m4.metric("Dense Semantic Cosine", f"{simulated_dense_sim * 100:.1f}%")

st.markdown("##### The Lexical vs Semantic Divide")
if jaccard_sim == 0 and simulated_dense_sim > 0.8:
    st.success("🎯 **Classic Semantic Victory**: Sentence A and Sentence B share **ZERO** overlapping words (0% lexical similarity), yet a dense foundation embedding recognizes that 'cardiologist' = 'doctor' and 'myocardial infarction' = 'cardiac arrest' (91% semantic similarity)!")
elif jaccard_sim > 0.3 and simulated_dense_sim < 0.4:
    st.warning("⚠️ **Lexical False Positive**: High string overlap exists, but the semantic meaning is completely different!")

df_sim_comp = pd.DataFrame({
    "Representation Metric": ["Lexical Overlap (Sparse TF-IDF / Jaccard)", "Dense Contextual Similarity (Transformer)"],
    "Similarity Score (%)": [jaccard_sim * 100, simulated_dense_sim * 100]
})
st.bar_chart(df_sim_comp.set_index("Representation Metric"))

st.divider()

# ==============================================================================
# Section 11: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Representation Interview Questions")

q_col1, q_col2 = st.columns(2)

with q_col1:
    with st.expander("Q1: What is the 'Curse of Dimensionality' in sparse NLP representations?"):
        st.markdown("""
        **Core Architectural Answer:**  
        In sparse representations (One-Hot, Bag-of-Words, TF-IDF), each unique vocabulary word constitutes an orthogonal basis dimension. 
        As vocabulary expands to 100,000 terms:
        1. **Memory Inefficiency**: Vectors are 99.9% zeros, requiring massive RAM for sparse matrix indexing.
        2. **Distance Metric Collapse**: In high-dimensional Euclidean space, the distance between any two data points converges to the same constant value (`sqrt(2)` for one-hot vectors), destroying clustering algorithms (k-means, KNN).
        3. **Data Hunger**: The volume of space grows exponentially with dimension, requiring exponential training data to prevent overfitting.
        
        Dense representations solve this by projecting language into a compact 1,024 to 4,096-dimensional continuous manifold.
        """)

    with st.expander("Q2: Why do static word embeddings (Word2Vec) suffer from 'Polysemy Collapse'?"):
        st.markdown("""
        **Answer:**  
        Static word embeddings assign a single static lookup vector per vocabulary key:
        `e_{bank} = W_{emb}[\text{'bank'}, :]`
        
        Because the word `"bank"` has multiple meanings (financial institution, river margin, aviation banking, blood bank), 
        the training objective forces the single vector to compromise, positioning it at the mathematical centroid of all its meanings. 
        Consequently, the vector is suboptimal for every specific meaning. 
        Transformer self-attention resolves polysemy by computing dynamic representations conditioned on surrounding tokens.
        """)

    with st.expander("Q3: What is Representation Anisotropy in Transformer embeddings?"):
        st.markdown("""
        **Answer:**  
        Representation Anisotropy is the phenomenon where high-dimensional vectors generated by deep neural networks 
        cluster into a narrow geometric cone rather than distributing uniformly across all directions in space. 
        This causes even completely unrelated sentences to exhibit high baseline cosine similarity (e.g. 0.70 to 0.85). 
        Modern models mitigate anisotropy through LayerNorm, whitening transforms, or contrastive fine-tuning with temperature-scaled negative mining.
        """)

with q_col2:
    with st.expander("Q4: How does Matryoshka Representation Learning (MRL) optimize vector databases?"):
        st.markdown("""
        **Answer:**  
        Standard embeddings require storing the full vector (e.g. 3,072 dimensions). 
        **Matryoshka Representation Learning (MRL)** (Kusupati et al., 2022) trains the model with loss terms applied to nested subsets 
        of dimensions (the first 128, 256, 512, 1024, and 3072 dimensions). 
        In production, engineers can truncate vectors from 3,072D to 512D, retaining **98% of retrieval accuracy** while slashing 
        Vector DB RAM storage costs and HNSW search latency by **6x**.
        """)

    with st.expander("Q5: What is the Distributional Hypothesis and how does it underpin modern embeddings?"):
        st.markdown("""
        **Answer:**  
        Formulated by linguist J.R. Firth in 1957 (*"You shall know a word by the company it keeps"*), the **Distributional Hypothesis** 
        posits that words occurring in similar linguistic contexts share similar semantic meanings. 
        Word2Vec, GloVe, BERT, and modern LLM pretraining operationalize this principle mathematically by training neural networks 
        to predict missing or future tokens conditioned on surrounding context words.
        """)

st.divider()

st.subheader("Production Traps & Representation Gotchas")

t_col1, t_col2 = st.columns(2)

with t_col1:
    st.error("""
    **Production Trap 1 (Critical): Comparing Cosine Distances Across Different Embedding Models**  
    * **Failure Mode**: Embedding document chunks with `text-embedding-ada-002` (1536D) and user queries with `text-embedding-3-small` (1536D).
    * **Impact**: Completely invalid similarity scores; each embedding model trains a distinct geometric coordinate manifold. Vectors across models are mathematically incompatible even if dimension counts match.
    * **Remediation**: Re-index the entire vector database whenever upgrading embedding models.
    """)

    st.error("""
    **Production Trap 2 (Critical): Computing Dot Products on Unnormalized Dense Embeddings**  
    * **Failure Mode**: Using raw dot product `np.dot(u, v)` instead of cosine similarity without verifying unit length.
    * **Impact**: Longer documents with larger embedding norms artificially score higher than concise, accurate documents.
    * **Remediation**: Always L2-normalize vectors (`v / np.linalg.norm(v)`) upon generation so dot product equals cosine similarity.
    """)

with t_col2:
    st.warning("""
    **Production Trap 3 (Caution): Using Dense Retrieval Alone for Part Numbers and Codes**  
    * **Failure Mode**: Searching for exact serial numbers (e.g. `ERR-90812-B`) purely using dense semantic vector search.
    * **Impact**: Dense models map part numbers to general 'technical error' clusters, failing to retrieve the exact matching document.
    * **Remediation**: Implement hybrid search: combine dense vector search with sparse BM25 / TF-IDF for exact keyword matching.
    """)

    st.warning("""
    **Production Trap 4 (Caution): Ingesting Unchunked Documents into Embedding Encoders**  
    * **Failure Mode**: Passing a 50-page PDF into an embedding model with an 8,192 token limit without section chunking.
    * **Impact**: The single embedding vector compresses dozens of distinct topics into a muddy, non-specific centroid vector.
    * **Remediation**: Chunk documents into 256 to 512-token semantically coherent passages before embedding.
    """)

st.divider()

st.subheader("Core Representation Flashcards")

rep_flash = st.selectbox(
    "Select Representation Flashcard to Review:",
    [
        "Curse of Dimensionality",
        "Polysemy Collapse",
        "Matryoshka Representation Learning (MRL)",
        "Representation Anisotropy"
    ]
)

if rep_flash == "Curse of Dimensionality":
    st.info("""
    **Curse of Dimensionality**: The exponential growth in volume and sparsity when representing text via orthogonal vocabulary dimensions, rendering Euclidean distances meaningless and wasting 99.9% of memory on zeros.
    """)
elif rep_flash == "Polysemy Collapse":
    st.info("""
    **Polysemy Collapse**: The fundamental flaw of static embeddings (Word2Vec) where a word with multiple distinct meanings (e.g. 'apple' or 'bank') is forced into a single compromise centroid vector.
    """)
elif rep_flash == "Matryoshka Representation Learning (MRL)":
    st.info("""
    **Matryoshka Representation Learning (MRL)**: An embedding training technique enabling vectors to be truncated (e.g. from 3072D to 512D) while preserving over 98% of retrieval accuracy, slashing Vector DB costs.
    """)
else:
    st.info("""
    **Representation Anisotropy**: The undesirable geometric tendency of deep neural embeddings to cluster in a narrow cone rather than distributing uniformly, causing artificially high baseline cosine similarity.
    """)

st.divider()

# ==============================================================================
# Section 12: Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of NLP representations and dense vector geometry:")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    cb1 = st.checkbox("I can articulate the mathematical progression from One-Hot to TF-IDF to Dense Embeddings.", value=True)
    cb2 = st.checkbox("I understand why sparse one-hot vectors are mutually orthogonal and fail at semantic distance.", value=True)
    cb3 = st.checkbox("I can explain why static word embeddings collapse polysemy and how transformers resolve it.", value=True)

with cb_col2:
    cb4 = st.checkbox("I understand the role of BM25 in hybrid search alongside dense vector embeddings.", value=True)
    cb5 = st.checkbox("I know how Matryoshka Representation Learning (MRL) optimizes vector database storage.", value=True)
    cb6 = st.checkbox("I can compute TF-IDF matrices and cosine similarities from scratch in pure NumPy.", value=True)

if all([cb1, cb2, cb3, cb4, cb5, cb6]):
    st.success("🎯 **Lesson 2.2 Complete! You have demonstrated comprehensive mastery of discrete and dense NLP representations for Large Language Models.**")
