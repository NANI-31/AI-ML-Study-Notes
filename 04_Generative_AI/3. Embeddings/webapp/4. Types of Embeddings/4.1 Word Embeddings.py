import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.1 Word Embeddings - Embeddings Curriculum",
    page_icon="📖",
    layout="wide"
)

# Title & Overview
st.title("4.1 Word Embeddings: Word2Vec (CBOW vs Skip-Gram), GloVe, FastText, & Static Limitations")
st.caption("Chapter 4: Types of Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Before modern transformers like BERT and GPT, the foundation of modern NLP was built on **Static Word Embeddings**.
Pioneered by Mikolov et al. at Google in 2013 (**Word2Vec**), Pennington et al. at Stanford in 2014 (**GloVe**), 
and Bojanowski et al. at Meta in 2016 (**FastText**), these algorithms demonstrated for the first time that 
neural networks could learn rich semantic geometry from unannotated text.

This lesson explores the three milestone word embedding families:
1. **Word2Vec (CBOW vs. Skip-Gram)**: Local sliding-window neural models predicting context from words or words from context.
2. **GloVe (Global Vectors)**: Log-bilinear matrix factorization combining global co-occurrence statistics with local context windows.
3. **FastText (Subword Character N-Grams)**: Breaking words into character chunks, solving the fatal Out-Of-Vocabulary (OOV) flaw of Word2Vec.
4. **The Polysemy Bottleneck**: Why static embeddings were ultimately superseded by contextual transformer models.

Understanding these foundational algorithms is essential for interview mastery and for resource-constrained edge deployments.
""")

st.divider()

# Section 1: Architecture & Model Comparison
st.header("1. Architectural Comparison: Word2Vec, GloVe, & FastText", divider="rainbow")

col_w1, col_w2 = st.columns([1, 1])

with col_w1:
    st.markdown("""
    ### Three Pillars of Word Representations:
    * **1. Word2Vec — CBOW (Continuous Bag of Words)**:
      * *Mechanism*: Predicts the center target word given the surrounding context words.
      * *Characteristics*: Fast to train; highly accurate on frequent words.
    * **2. Word2Vec — Skip-Gram**:
      * *Mechanism*: Takes the center target word and predicts the surrounding context words.
      * *Characteristics*: Slower training, but captures superior representations for rare words.
    * **3. GloVe (Global Vectors)**:
      * Factorizes the global word-word co-occurrence matrix. If word i and word j appear frequently together across the entire corpus, their dot product `dot(w_i, w_j)` scales with the logarithm of their co-occurrence count.
    * **4. FastText (Subword N-Grams)**:
      * Represents each word as the sum of its character n-grams. For example, `"apple"` with n=3 is composed of `<ap`, `app`, `ppl`, `ple`, `le>`.
      * Enables FastText to generate meaningful embeddings for unseen misspelled words like `"aple"`.
    """)

with col_w2:
    st.markdown("### CBOW vs. Skip-Gram Architectures")
    w2v_dot = """
    digraph Word2VecArch {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_cbow {
            label="CBOW Architecture\\n(Context -> Target)";
            style="dashed"; color="#3B82F6";
            C1 [label="Context: 'deep'\\nContext: 'learning'", fillcolor="#EFF6FF"];
            C_Sum [label="Projection Layer\\n(Sum / Average)", fillcolor="#DBEAFE"];
            Target [label="Target Word:\\n'transforms'", fillcolor="#2563EB", fontcolor="white", style="filled,bold"];
            C1 -> C_Sum -> Target;
        }

        subgraph cluster_skip {
            label="Skip-Gram Architecture\\n(Target -> Context)";
            style="dashed"; color="#10B981";
            Center [label="Target Word:\\n'transforms'", fillcolor="#059669", fontcolor="white", style="filled,bold"];
            Proj [label="Projection Layer", fillcolor="#D1FAE5"];
            CtxOut [label="Predicted Context:\\n'deep', 'learning', 'data'", fillcolor="#ECFDF5"];
            Center -> Proj -> CtxOut;
        }
    }
    """
    st.graphviz_chart(w2v_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Word Embedding Engine in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates vocabulary lookup, FastText character n-gram decomposition, and OOV vector synthesis.
""")

class ProductionWordEmbeddingEngine:
    """
    Simulates static word embedding lookup (Word2Vec/GloVe)
    and FastText subword n-gram vector synthesis for out-of-vocabulary terms.
    """
    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        # Fixed static vocabulary
        self.vocab = {
            "neural": np.array([0.85, 0.75, 0.10, 0.20, -0.15, 0.40]),
            "network": np.array([0.80, 0.70, 0.15, 0.25, -0.10, 0.35]),
            "deep": np.array([0.75, 0.82, 0.05, 0.15, -0.20, 0.50]),
            "learning": np.array([0.70, 0.88, 0.12, 0.18, -0.12, 0.45]),
            "computer": np.array([0.88, 0.20, 0.15, 0.70, 0.10, 0.15])
        }
        # Subword n-gram dictionary for FastText
        self.ngram_weights = {
            "<neu": np.array([0.40, 0.35, 0.05, 0.10, -0.05, 0.20]),
            "neur": np.array([0.45, 0.40, 0.05, 0.10, -0.10, 0.20]),
            "eura": np.array([0.30, 0.25, 0.02, 0.08, -0.05, 0.15]),
            "ural": np.array([0.25, 0.20, 0.03, 0.07, -0.05, 0.10]),
            "ral>": np.array([0.20, 0.15, 0.02, 0.05, -0.02, 0.10]),
            "<net": np.array([0.40, 0.30, 0.08, 0.12, -0.05, 0.18]),
            "work": np.array([0.40, 0.40, 0.07, 0.13, -0.05, 0.17])
        }

    def get_word2vec_embedding(self, word: str) -> Tuple[bool, np.ndarray]:
        """Static lookup: returns vector if word in vocabulary, else OOV error."""
        w_lower = word.lower()
        if w_lower in self.vocab:
            vec = self.vocab[w_lower]
            return True, vec / np.linalg.norm(vec)
        return False, np.zeros(self.dimensions)

    def get_fasttext_embedding(self, word: str) -> Tuple[List[str], np.ndarray]:
        """
        FastText subword synthesis:
        Decomposes word into character n-grams (n=3,4) and sums constituent weights.
        """
        w_lower = word.lower()
        # Direct hit
        if w_lower in self.vocab:
            vec = self.vocab[w_lower]
            return [f"<{w_lower}> (Exact Vocab Match)"], vec / np.linalg.norm(vec)

        # Decompose into n-grams
        wrapped = f"<{w_lower}>"
        ngrams_found = []
        synthesized_vec = np.zeros(self.dimensions)

        for n in [3, 4]:
            for i in range(len(wrapped) - n + 1):
                gram = wrapped[i:i+n]
                if gram in self.ngram_weights:
                    ngrams_found.append(gram)
                    synthesized_vec += self.ngram_weights[gram]

        if len(ngrams_found) == 0:
            # Deterministic fallback based on hash
            np.random.seed(abs(hash(w_lower)) % (2**31))
            synthesized_vec = np.random.uniform(-0.3, 0.3, self.dimensions)
            ngrams_found.append("Random character hash (Unseen subwords)")

        unit_vec = synthesized_vec / np.linalg.norm(synthesized_vec)
        return ngrams_found, unit_vec

code_snippet = r'''
import numpy as np

class WordEmbeddingInspector:
    def word2vec_lookup(self, word: str, vocab: dict) -> np.ndarray:
        # Fails completely on out-of-vocabulary or misspelled words
        if word in vocab:
            return vocab[word]
        raise KeyError(f"Out of Vocabulary (OOV): '{word}'")

    def fasttext_synthesize(self, word: str, ngram_dict: dict, n: int = 3) -> np.ndarray:
        # FastText splits word into character n-grams
        wrapped = f"<{word}>"
        ngrams = [wrapped[i:i+n] for i in range(len(wrapped) - n + 1)]
        
        # Sums all matched subword vectors
        subword_vectors = [ngram_dict[g] for g in ngrams if g in ngram_dict]
        if subword_vectors:
            vector = np.sum(subword_vectors, axis=0)
            return vector / np.linalg.norm(vector)
        return np.random.randn(300)
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Word2Vec vs. FastText Workbench
st.header("3. Interactive Word2Vec vs. FastText OOV Workbench", divider="rainbow")
st.markdown("""
Test how **Word2Vec (Standard Static Lookup)** and **FastText (Subword N-Gram Synthesis)** handle vocabulary words, 
misspellings, and unseen compound words.
""")

word_engine = ProductionWordEmbeddingEngine()

col_in_w1, col_in_w2 = st.columns([1, 1])

with col_in_w1:
    test_word_choice = st.selectbox(
        "Select test word or choose custom:",
        ["neural", "network", "deep", "neuralnet (Compound OOV)", "neuraal (Misspelled OOV)", "quantum (Unknown OOV)"]
    )
with col_in_w2:
    custom_input = st.text_input("Enter custom single word (optional):", value=test_word_choice.split()[0])
    active_word = custom_input.strip() if custom_input.strip() else test_word_choice.split()[0]

st.markdown("### Model Execution Comparison")

col_res_w1, col_res_w2 = st.columns([1, 1])

with col_res_w1:
    st.markdown("#### 1. Word2Vec / GloVe (Static Vocabulary)")
    w2v_found, w2v_vec = word_engine.get_word2vec_embedding(active_word)
    
    if w2v_found:
        st.success(f"VOCABULARY HIT: '{active_word}' exists in dictionary.")
        st.dataframe(pd.DataFrame([w2v_vec], columns=[f"D_{i+1}" for i in range(6)]).style.format("{:.4f}").background_gradient(cmap="coolwarm", axis=1), use_container_width=True)
    else:
        st.error(f"❌ OUT-OF-VOCABULARY (OOV) ERROR: '{active_word}' not found in dictionary!")
        st.warning("Word2Vec cannot compute an embedding for this word. In downstream pipelines, this token is replaced with <UNK> or dropped, causing catastrophic loss of meaning.")

with col_res_w2:
    st.markdown("#### 2. FastText (Subword Character N-Grams)")
    ngrams_used, ft_vec = word_engine.get_fasttext_embedding(active_word)
    
    st.success(f"SUCCESS: Synthesized valid embedding for '{active_word}'!")
    st.write(f"**Constituent Subwords Extracted:** `{', '.join(ngrams_used)}`")
    st.dataframe(pd.DataFrame([ft_vec], columns=[f"D_{i+1}" for i in range(6)]).style.format("{:.4f}").background_gradient(cmap="viridis", axis=1), use_container_width=True)
    st.caption("FastText leverages character n-grams to represent misspelled and compound words without retraining.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Word Embedding Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the Polysemy Bottleneck in static word embeddings, and how was it solved?**
    * **Answer**:
      * **The Bottleneck**: Static word embeddings (Word2Vec, GloVe, FastText) assign exactly one fixed vector per word token in the vocabulary.
      * Ambiguous words like `"bank"` have a single static point in space. That single point must simultaneously represent a financial institution (*"deposit money in a bank"*) and a geographical riverbank (*"sit by the river bank"*), creating a muddled compromise vector.
      * **The Solution**: Transformer encoders (BERT, RoBERTa, GPT) compute **contextual embeddings**: multi-head self-attention dynamically computes different vector representations for `"bank"` based on the surrounding sentence tokens.

    **Q2: What are the differences between CBOW and Skip-Gram, and when should you choose each?**
    * **Answer**:
      * **CBOW (Continuous Bag of Words)**:
        * Predicts target center word w_t from surrounding context: P(w_t given context).
        * Faster to train (O(C) per update); smooths over frequent words effectively.
      * **Skip-Gram**:
        * Predicts context words from target center word: P(context given w_t).
        * Slower to train, but does not average context words together; performs substantially better on rare words and specialized technical jargon.

    **Q3: How does GloVe combine the advantages of Global Matrix Factorization and Local Sliding Windows?**
    * **Answer**:
      * Traditional methods were split: **LSA (Latent Semantic Analysis)** utilized global corpus co-occurrence statistics but struggled with vector analogy arithmetic; **Word2Vec** excelled at analogies using local sliding windows but failed to exploit global co-occurrence counts.
      * **GloVe (Global Vectors)** combines both: it trains a log-bilinear model on the global non-zero co-occurrence matrix $X$, fitting dot products to the log probabilities of co-occurrence:
        `dot(w_i, w_j) = log(P(i | j)) = log(X_ij / X_i)`.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Deploying Word2Vec in Medical / Scientific Pipelines with Heavy Jargon**:
      * *Issue*: Using standard pre-trained Word2Vec on clinical notes or chemistry papers.
      * *Consequence*: Complex compound medical terms (`"hypercholesterolemia"`) trigger 100% OOV failures and get replaced with `<UNK>`.
      * *Fix*: Use FastText (which breaks medical Latin prefixes/suffixes into n-grams) or domain-specific transformers (PubMedBERT).

    * **Trap 2: Comparing Static Word Vectors to Sentence Vectors Directly**:
      * *Issue*: Computing cosine similarity between a Word2Vec word vector and a sentence-transformer document vector.
      * *Consequence*: Different dimensionalities and completely unaligned latent coordinate spaces.
      * *Fix*: Never cross-compare vectors from different model architectures.

    * **Trap 3: Forgetting Negative Sampling in Word2Vec Training**:
      * *Issue*: Computing the full softmax denominator over a vocabulary of 500,000 words on every gradient update.
      * *Consequence*: Training is impossibly slow ($O(|V|)$ per word).
      * *Fix*: Always use **Negative Sampling** (e.g. 5 to 20 noise words) or Hierarchical Softmax.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: FastText Subword N-Grams**
      * *Definition*: Decomposing words into character n-grams so out-of-vocabulary and misspelled terms can be embedded by summing constituent pieces.
    * **Flashcard 2: Continuous Bag-of-Words (CBOW)**
      * *Definition*: A Word2Vec architecture that predicts the target center word given its surrounding context words.
    * **Flashcard 3: Continuous Skip-Gram**
      * *Definition*: A Word2Vec architecture that predicts surrounding context words given the center target word, optimal for rare vocabulary.
    * **Flashcard 4: Polysemy**
      * *Definition*: A word having multiple distinct meanings, which causes static word embeddings to produce conflated compromise vectors.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Word Embeddings:

- [ ] I can describe the architectural differences between Word2Vec CBOW and Skip-Gram.
- [ ] I understand how FastText solves the Out-of-Vocabulary (OOV) problem using character n-grams.
- [ ] I can explain how GloVe fits dot products to global log co-occurrence probabilities.
- [ ] I understand why the polysemy bottleneck caused static embeddings to be replaced by transformers.
- [ ] I know when FastText is still preferred for ultra-fast, lightweight mobile text processing.
""")
