import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.2 Classical Word Embeddings (Word2Vec, CBOW, Skip-Gram, GloVe, FastText)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .takeaway-box {
        background-color: #0B132B;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #1C1917;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .modern-callout {
        background-color: #042F2E;
        border-left: 4px solid #14B8A6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-title">4.2 Classical Embeddings: Word2Vec, CBOW, Skip-gram, GloVe, and FastText</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Local Distributional Semantics to Global Co-occurrence Matrices and Subword N-grams — How Static Word Representations Paved the Path to Foundation Model Weights.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Evolution of Static Distributed Semantics", divider="rainbow")

st.markdown("""
Before deep multi-layer Transformer architectures (GPT, Llama, Mistral) transformed natural language processing, 
computational linguistics transitioned from orthogonal one-hot vectors to **distributed, continuous vector representations**.
Between 2013 and 2017, three landmark paradigms emerged:
1. **Word2Vec (Mikolov et al., 2013)**: Local context window modeling via shallow two-layer feedforward networks (CBOW and Skip-gram).
2. **GloVe (Pennington et al., 2014)**: Global co-occurrence matrix factorization leveraging ratios of word probabilities across full corpuses.
3. **FastText (Bojanowski et al., 2017)**: Subword character n-gram composition enabling morphological generalization and out-of-vocabulary representation.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("CORPUS", "Raw Text Corpus\n(Billions of Unlabeled Tokens)", shape="cylinder", fillcolor="#1E293B", color="#38BDF8")
    
    with arch_graph.subgraph(name="cluster_w2v") as c:
        c.attr(bgcolor="#0F172A", label="Word2Vec (2013) - Local Sliding Window", fontcolor="#38BDF8", color="#38BDF8")
        c.node("CBOW", "CBOW\nPredict Target from Context\nAverage(w_{t-c}...w_{t+c}) -> w_t", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("SG", "Skip-gram\nPredict Context from Target\nw_t -> [w_{t-c}, ..., w_{t+c}]", shape="box", fillcolor="#1E293B", color="#38BDF8")

    with arch_graph.subgraph(name="cluster_glove") as c:
        c.attr(bgcolor="#0F172A", label="GloVe (2014) - Global Co-occurrence Matrix", fontcolor="#A855F7", color="#A855F7")
        c.node("MATRIX", "Global Co-occurrence Matrix X\nX_ij = Count(w_i, w_j within window)", shape="box", fillcolor="#1E293B", color="#A855F7")
        c.node("GLOVE_OBJ", "Weighted Least Squares Objective\nw_i . w_j + b_i + b_j = log(X_ij)", shape="box", fillcolor="#1E293B", color="#A855F7")

    with arch_graph.subgraph(name="cluster_fasttext") as c:
        c.attr(bgcolor="#0F172A", label="FastText (2017) - Subword Character N-grams", fontcolor="#10B981", color="#10B981")
        c.node("NGRAM", "Character N-grams\n<wh, whe, her, ere, re> + <where>", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("FT_SUM", "Sum of N-gram Vectors\nv(where) = Sum(z_g)", shape="box", fillcolor="#1E293B", color="#10B981")

    arch_graph.node("STATIC_LOOKUP", "Static Embedding Lookup Table\nWeight Matrix W in R^{V x D}\n(1 Immutable Vector per Token)", shape="box", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("MODERN_LLM", "Modern Foundation LLM Input Layer\nToken IDs -> nn.Embedding(V, D) -> LayerNorm\n(Dynamically contextualized by Self-Attention)", shape="box", fillcolor="#042F2E", color="#14B8A6")

    # Edges
    arch_graph.edge("CORPUS", "CBOW")
    arch_graph.edge("CORPUS", "SG")
    arch_graph.edge("CORPUS", "MATRIX")
    arch_graph.edge("CORPUS", "NGRAM")
    arch_graph.edge("MATRIX", "GLOVE_OBJ")
    arch_graph.edge("NGRAM", "FT_SUM")
    arch_graph.edge("CBOW", "STATIC_LOOKUP")
    arch_graph.edge("SG", "STATIC_LOOKUP")
    arch_graph.edge("GLOVE_OBJ", "STATIC_LOOKUP")
    arch_graph.edge("FT_SUM", "STATIC_LOOKUP")
    arch_graph.edge("STATIC_LOOKUP", "MODERN_LLM", label="Evolved into Modern Input Tables")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Key Architectural Breakthroughs
    * **Distributional Hypothesis in Action**: Words occurring in similar contexts share similar semantic representations ("You shall know a word by the company it keeps" — J.R. Firth, 1957).
    * **Unsupervised Latent Representation**: Neither Word2Vec, GloVe, nor FastText require manual human labels; they self-supervise on raw unannotated text.
    * **Linear Subspace Isomorphism**: Relationships between concepts manifest as consistent geometric vector displacements (e.g., `Vector("King") - Vector("Man") + Vector("Woman") approx Vector("Queen")`).
    * **The Polysemy Bottleneck**: Every classical model assigns exactly **one static vector per vocabulary item**, rendering it fundamentally incapable of disambiguating "river bank" versus "investment bank".
    """)
    st.info("Modern Transformer Foundation Models (Llama 3, GPT-4, Mistral) retained the input lookup matrix `nn.Embedding(V, D)` established by Word2Vec, but replaced static output pooling with deep multi-head self-attention layers.")

# ==============================================================================
# SECTION 2: 1. WORD2VEC
# ==============================================================================
st.header("2. Word2Vec: Self-Supervised Neural Embeddings", divider="rainbow")

st.markdown("""
Introduced by Tomas Mikolov et al. at Google in 2013, **Word2Vec** transformed natural language processing 
by demonstrating that high-quality, continuous vector representations could be learned efficiently over billions of words in hours rather than weeks.

### Theoretical Foundation
Word2Vec is not a deep neural network; it is a **shallow two-layer architecture** containing:
1. An **Input Projection Matrix** `W_in` of dimensions `[Vocabulary_Size, Hidden_Dimension]`.
2. An **Output Target Matrix** `W_out` of dimensions `[Hidden_Dimension, Vocabulary_Size]`.
3. **No non-linear activation functions** in the hidden layer (it performs pure linear projection and inner products).
""")

w2v_col1, w2v_col2 = st.columns([1, 1])

with w2v_col1:
    st.markdown("""
    ### Training Objectives: Softmax vs. Negative Sampling
    Computing full Softmax across vocabulary V requires calculating:
    ```
    P(w_O | w_I) = exp(v'_wO . v_wI) / Sum_{w=1}^{V} exp(v'_w . v_wI)
    ```
    When vocabulary size V = 1,000,000, computing the partition denominator for every single token update is computationally intractable (O(V)).

    To scale training, Mikolov introduced two training approximations:
    * **Hierarchical Softmax**: Encodes vocabulary as a Huffman binary tree, reducing denominator computation from O(V) to O(log_2(V)).
    * **Negative Sampling (NEG)**: Reformulates multiclass classification into binary logistic regression. For every true target-context pair (w, c), sample k noise words (w, n_i) from unigram distribution P_n(w) = (count(w))^{0.75} / Z:
    ```
    Loss_NEG = -log(sigmoid(v'_c . v_w)) - Sum_{i=1}^{k} log(sigmoid(-v'_{n_i} . v_w))
    ```
    """)

with w2v_col2:
    st.markdown("""
    ### Classical vs. Modern Foundation Model Comparison
    """)
    df_w2v_comp = pd.DataFrame({
        "Attribute": ["Architecture", "Vocabulary Unit", "Representation", "Polysemy Handling", "Training Objective", "Compute Scaling"],
        "Classical Word2Vec (2013)": [
            "2-layer shallow linear projection",
            "Whole Words (dict lookup)",
            "Static, fixed 1 vector per word",
            "Collapses all senses into single centroid",
            "Negative Sampling / Hierarchical Softmax",
            "CPU multi-threaded C code (word2vec.c)"
        ],
        "Modern LLM (Llama 3 / GPT-4)": [
            "32-128 layer Decoder-only Transformer",
            "Subword / Byte tokens (BPE, Tiktoken)",
            "Dynamic contextual hidden states",
            "Full context-conditioned disambiguation",
            "Next-Token Prediction (Causal Cross-Entropy)",
            "GPU/TPU distributed clusters (Megatron/FSDP)"
        ]
    })
    st.dataframe(df_w2v_comp, use_container_width=True, hide_index=True)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Modern LLMs still initialize their first layer with an input embedding table <code>nn.Embedding(vocab_size, hidden_dim)</code>, directly inheriting Word2Vec's linear projection concept. However, modern models discard Word2Vec's negative sampling loss in favor of causal language modeling (next-token cross-entropy), and pass tokens through multi-head self-attention layers so that word representations evolve dynamically at each layer.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: 2. CBOW (CONTINUOUS BAG OF WORDS)
# ==============================================================================
st.header("3. CBOW: Continuous Bag of Words Architecture", divider="rainbow")

st.markdown("""
The **Continuous Bag of Words (CBOW)** architecture learns vector representations by predicting a **single target center word** given a symmetrical context window of surrounding words.
""")

cbow_col1, cbow_col2 = st.columns([1, 1])

with cbow_col1:
    cbow_graph = graphviz.Digraph(format="svg")
    cbow_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.5")
    cbow_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    cbow_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    cbow_graph.node("C1", "Context Token: w_{t-2}\n[One-Hot V]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    cbow_graph.node("C2", "Context Token: w_{t-1}\n[One-Hot V]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    cbow_graph.node("C3", "Context Token: w_{t+1}\n[One-Hot V]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    cbow_graph.node("C4", "Context Token: w_{t+2}\n[One-Hot V]", shape="box", fillcolor="#1E293B", color="#38BDF8")

    cbow_graph.node("POOL", "Projection / Hidden Layer\nMean / Sum: h = 1/(2C) Sum(v_c)\nDimension: [1, D]", shape="ellipse", fillcolor="#312E81", color="#818CF8")
    cbow_graph.node("OUT", "Output Layer\nSoftmax(W_out . h)\nPredict: Target Token w_t\nDimension: [1, V]", shape="box", fillcolor="#042F2E", color="#14B8A6")

    cbow_graph.edge("C1", "POOL", label="W_in")
    cbow_graph.edge("C2", "POOL", label="W_in")
    cbow_graph.edge("C3", "POOL", label="W_in")
    cbow_graph.edge("C4", "POOL", label="W_in")
    cbow_graph.edge("POOL", "OUT", label="W_out")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(cbow_graph, use_container_width=True)

with cbow_col2:
    st.markdown("""
    ### CBOW Algorithmic Properties
    * **Averaging Step**: The context token representations are collapsed into a single mean vector:
      `h = (1 / 2C) * Sum_{i in {-C, ..., +C, i != 0}} v_{w_{t+i}}`
    * **Bag-of-Words Limitation**: Because context vectors are summed or averaged, word order within the window is completely discarded (`"dog bites man"` produces the identical hidden projection h as `"man bites dog"`).
    * **Computational Complexity**: O(2C * D + D * log(V)). Faster to train than Skip-gram because one optimization step updates parameters for 2C context words simultaneously.
    * **Strengths**: Smooths over noise across frequent words; performs exceptionally well on syntactic tasks and common vocabulary items.
    * **Weakness**: Struggles with rare words because context averaging dilutes infrequent token representations into dominant frequent word signals.
    """)

st.markdown("""
<div class="takeaway-box">
<strong>Engineering Insight: Why LLMs Abandoned CBOW Averaging</strong><br>
CBOW treats context as an unweighted bag of tokens. Modern LLMs use <strong>Scaled Dot-Product Attention</strong> instead of simple averaging. Attention allows every token to query and dynamically weight relevant context tokens based on semantic compatibility, rather than assigning uniform 1/(2C) weights to all window positions.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: 3. SKIP-GRAM
# ==============================================================================
st.header("4. Skip-Gram: Context Generation from Center Word", divider="rainbow")

st.markdown("""
The **Skip-gram** architecture flips the CBOW formulation: given a single center word w_t, the model learns vector representations by maximizing the probability of predicting its **surrounding context words** within window radius C.
""")

sg_col1, sg_col2 = st.columns([1, 1])

with sg_col1:
    sg_graph = graphviz.Digraph(format="svg")
    sg_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.5")
    sg_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    sg_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    sg_graph.node("IN", "Input Center Token: w_t\n[One-Hot V]", shape="box", fillcolor="#312E81", color="#818CF8")
    sg_graph.node("PROJ", "Hidden Layer\nh = v_{w_t} = W_in[w_t]\nDimension: [1, D]", shape="ellipse", fillcolor="#1E293B", color="#38BDF8")

    sg_graph.node("OUT1", "Predict: w_{t-2}\nSoftmax(W_out . h)", shape="box", fillcolor="#042F2E", color="#14B8A6")
    sg_graph.node("OUT2", "Predict: w_{t-1}\nSoftmax(W_out . h)", shape="box", fillcolor="#042F2E", color="#14B8A6")
    sg_graph.node("OUT3", "Predict: w_{t+1}\nSoftmax(W_out . h)", shape="box", fillcolor="#042F2E", color="#14B8A6")
    sg_graph.node("OUT4", "Predict: w_{t+2}\nSoftmax(W_out . h)", shape="box", fillcolor="#042F2E", color="#14B8A6")

    sg_graph.edge("IN", "PROJ", label="Lookup")
    sg_graph.edge("PROJ", "OUT1", label="W_out")
    sg_graph.edge("PROJ", "OUT2", label="W_out")
    sg_graph.edge("PROJ", "OUT3", label="W_out")
    sg_graph.edge("PROJ", "OUT4", label="W_out")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(sg_graph, use_container_width=True)

with sg_col2:
    st.markdown("""
    ### Skip-gram Optimization Mechanics
    Given token sequence T = (w_1, w_2, ..., w_T), maximize the average log probability:
    ```
    Objective = (1 / T) * Sum_{t=1}^{T} Sum_{-C <= j <= C, j != 0} log P(w_{t+j} | w_t)
    ```
    
    ### Skip-gram vs. CBOW Empirical Trade-Offs
    * **Rare Word Representation**: In Skip-gram, each context-target occurrence produces an independent gradient update on the target vector v_{w_t}. Even if a rare word appears only 5 times, it generates 5 * 2C parameter updates without being diluted by context averaging.
    * **Training Speed**: Skip-gram requires approximately 2C times more gradient calculations than CBOW, making it substantially slower to train over large text corpuses.
    * **Semantic vs. Syntactic**: Skip-gram generally outperforms CBOW on semantic word analogy benchmarks (e.g., capitals, currencies, familial relations).
    """)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: The Asymmetry of Next-Token Prediction</strong><br>
Skip-gram predicts both past (w_{t-j}) and future (w_{t+j}) context tokens simultaneously (bidirectional local prediction). Modern autoregressive LLMs (Llama 3, GPT-4) enforce a <strong>strict causal mask</strong>: they only predict the single future token w_{t+1} given historical context w_{1:t}. This causal factorization enables efficient KV-caching during autoregressive generation.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: 4. GLOVE (GLOBAL VECTORS)
# ==============================================================================
st.header("5. GloVe: Global Vectors for Word Representation", divider="rainbow")

st.markdown("""
Developed by Jeffrey Pennington, Richard Socher, and Christopher Manning at Stanford University in 2014, 
**GloVe (Global Vectors)** bridged the gap between two major model families:
1. **Global Matrix Factorization Methods** (LSA, Latent Semantic Analysis): Capture global corpus statistics but perform poorly on local word analogies.
2. **Local Context Window Methods** (Skip-gram, CBOW): Excel at capturing local analogies but fail to exploit global co-occurrence frequencies across the entire corpus.
""")

glove_col1, glove_col2 = st.columns([1, 1])

with glove_col1:
    st.markdown("""
    ### The Core Insight: Ratios of Co-occurrence Probabilities
    Let X_{ij} be the number of times word w_j appears in the context of word w_i.
    Let X_i = sum_k(X_{ik}) be the total occurrences of word i.
    The probability that word j appears in the context of word i is:
    ```
    P_{ij} = P(w_j | w_i) = X_{ij} / X_i
    ```
    
    Pennington et al. demonstrated that raw probabilities carry little signal, but **ratios of probabilities** cleanly isolate semantic differences:
    
    | Probe Word (k) | P(k | ice) | P(k | steam) | Ratio: P(k | ice) / P(k | steam) |
    | :--- | :--- | :--- | :--- |
    | **solid** | High (1.9 * 10^{-4}) | Low (2.2 * 10^{-5}) | **8.9** (Large, associated with ice) |
    | **gas** | Low (6.6 * 10^{-5}) | High (7.8 * 10^{-4}) | **0.085** (Small, associated with steam) |
    | **water** | High (3.0 * 10^{-3}) | High (2.2 * 10^{-3}) | **1.36** (Near 1, associated with both) |
    | **fashion** | Low (1.7 * 10^{-5}) | Low (1.8 * 10^{-5}) | **0.96** (Near 1, irrelevant to both) |
    """)

with glove_col2:
    st.markdown("""
    ### The GloVe Weighted Least Squares Objective
    GloVe forces the dot product of word vectors w_i and context vectors w~_j to equal the log of their co-occurrence probability:
    ```
    w_i^T . w~_j + b_i + b~_j = log(X_ij)
    ```
    
    To prevent frequent stop words (e.g., `"the"`, `"of"`) from dominating the loss, GloVe introduces a soft weighting function f(X_{ij}):
    ```
    J = Sum_{i, j=1}^{V} f(X_ij) * (w_i^T . w~_j + b_i + b~_j - log(X_ij))^2
    ```
    Where the weighting function is parameterized by:
    ```
    f(x) = (x / x_max)^alpha  if x < x_max
    f(x) = 1.0                otherwise
    ```
    Typically configured with x_{max} = 100 and \alpha = 0.75.
    """)

    st.markdown("""
    ### Symmetric Vector Averaging
    Because word and context definitions are arbitrary in symmetric windows:
    ```
    w_final = w_i + w~_i
    ```
    Summing the main and context vectors acts as a regularizer, yielding superior downstream performance.
    """)

# ==============================================================================
# SECTION 6: 5. FASTTEXT
# ==============================================================================
st.header("6. FastText: Subword Character N-Grams Representation", divider="rainbow")

st.markdown("""
Developed by Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov at Facebook AI Research (FAIR) in 2017, 
**FastText** resolved the primary architectural flaw of Word2Vec and GloVe: **ignoring internal morphological structure of words**.
""")

ft_col1, ft_col2 = st.columns([1, 1])

with ft_col1:
    st.markdown("""
    ### Character N-gram Decomposition
    In Word2Vec, `"play"`, `"playing"`, `"player"`, and `"playable"` are assigned four completely independent, non-sharing parameters in `W_in`.
    
    FastText decomposes each word into a set of character n-grams bounded by boundary brackets `<` and `>`:
    * Word: `<where>`
    * For n=3: `<wh`, `whe`, `her`, `ere`, `re>`
    * Plus special whole-word token: `<where>`
    
    The final representation of word w is the **sum of its constituent n-gram vectors**:
    ```
    v_w = Sum_{g in G_w} z_g
    ```
    Where G_w is the set of n-grams appearing in w, and z_g is the vector embedding for n-gram g.
    """)

with ft_col2:
    st.markdown("""
    ### Critical Advantages of FastText
    1. **Out-of-Vocabulary (OOV) Generalization**:
       If an inference token (e.g., `"unbiodegradable"`) was never seen during training, Word2Vec returns an unknown token `[UNK]`. FastText extracts character n-grams (`<un`, `bio`, `deg`, `rad`, `able>`) and sums their known representations to form an accurate approximation.
    2. **Morphologically Rich Languages**:
       Languages with extensive conjugation and compounding (German, Finnish, Turkish, Russian) fail under whole-word models because vocabulary sizes explode exponentially. FastText shares subword stems across all morphological inflections.
    3. **Misspelling Robustness**:
       Typographical errors like `"definately"` share 80%+ of their constituent 3-grams and 4-grams with the canonical `"definitely"`, yielding high cosine similarity.
    """)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: From Character N-Grams to Byte-Pair Encoding (BPE)</strong><br>
While FastText showed the power of subwords, summing overlapping character n-grams produces high memory consumption (N_{ngrams} >> V). Modern LLMs solved this cleanly: instead of summing n-grams, modern LLMs use <strong>Byte-Pair Encoding (BPE)</strong> or <strong>Byte-level Unigram Tokenizers</strong> (SentencePiece / Tiktoken). BPE tokenizes words into discrete, non-overlapping subword chunks, feeding each chunk directly into the Transformer embedding matrix.
</div>
""", unsafe_allow_html=True)

# Side-by-side comparison matrix across all classical embedding families
st.markdown("### Master Architectural Comparison Matrix")
df_all_classical = pd.DataFrame({
    "Criterion": ["Pioneering Author / Year", "Core Mechanism", "Window Scope", "Subword Awareness", "OOV Handling", "Matrix Sparsity Issue", "Memory Footprint", "Modern LLM Successor"],
    "Word2Vec (CBOW)": ["Mikolov et al., 2013", "Predict target from averaged context", "Local sliding window (5-10 tokens)", "None (whole word)", "Fails (UNK token)", "None (dense projection)", "O(V * D)", "Transformer Encoder (BERT pooling)"],
    "Word2Vec (Skip-Gram)": ["Mikolov et al., 2013", "Predict context from single center word", "Local sliding window (5-10 tokens)", "None (whole word)", "Fails (UNK token)", "None (dense projection)", "O(V * D)", "Autoregressive Next-Token Head"],
    "GloVe": ["Pennington et al., 2014", "Weighted log-bilinear matrix factorization", "Global corpus co-occurrence matrix", "None (whole word)", "Fails (UNK token)", "Yes (must store large sparse X)", "O(V * D) + Matrix X", "Embedding weight initialization"],
    "FastText": ["Bojanowski et al., 2017", "Summation of character n-gram vectors", "Local sliding window (Skip-gram based)", "High (char 3-6 n-grams)", "Robust (n-gram sum)", "None (hash-based n-grams)", "O((V + N_ngrams) * D)", "Byte-level BPE / Tiktoken"]
})
st.dataframe(df_all_classical, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Classical Word Vector Arithmetic & Co-occurrence Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. A **Co-occurrence Matrix Generator** simulating GloVe's global context statistics.
2. A **Pure-Numpy Word Vector Arithmetic Engine** performing cosine similarity, vector arithmetic analogies (`A - B + C = D`), and FastText character n-gram decomposition.
""")

st.code('''
import numpy as np
import collections
from typing import List, Dict, Tuple, Set

class ClassicalEmbeddingEngine:
    """
    Production-grade educational embedding engine demonstrating:
    1. Symmetric Co-occurrence Matrix Generation (GloVe basis).
    2. Character N-Gram Decomposition & Out-Of-Vocabulary Embedding Synthesis (FastText basis).
    3. Vector Arithmetic & Analogy Resolution (Word2Vec basis).
    """
    def __init__(self, embedding_dim: int = 64, random_seed: int = 42):
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.vectors: np.ndarray = np.empty((0, embedding_dim))
        self.ngram_vectors: Dict[str, np.ndarray] = {}
        
    def build_cooccurrence_matrix(self, corpus: List[List[str]], window_size: int = 2) -> Tuple[np.ndarray, Dict[str, int]]:
        """Constructs symmetric global co-occurrence matrix X."""
        words = [w for doc in corpus for w in doc]
        vocab_counts = collections.Counter(words)
        vocab = {w: i for i, (w, _) in enumerate(vocab_counts.most_common())}
        v_size = len(vocab)
        
        cooccurrence = np.zeros((v_size, v_size), dtype=np.float32)
        for doc in corpus:
            for i, target in enumerate(doc):
                target_idx = vocab[target]
                start = max(0, i - window_size)
                end = min(len(doc), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_idx = vocab[doc[j]]
                        # Harmonic distance weighting: 1 / distance
                        dist = abs(i - j)
                        cooccurrence[target_idx, context_idx] += 1.0 / dist
                        
        return cooccurrence, vocab

    def extract_character_ngrams(self, word: str, min_n: int = 3, max_n: int = 5) -> List[str]:
        """FastText-style character n-gram extraction with boundary tags."""
        tagged = f"<{word}>"
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tagged) - n + 1):
                ngrams.append(tagged[i:i + n])
        ngrams.append(tagged) # Whole word tag
        return ngrams

    def register_vocabulary(self, word_embeddings: Dict[str, np.ndarray]):
        """Registers pretrained word vectors and initializes subword n-grams."""
        self.vocab = {w: i for i, w in enumerate(word_embeddings.keys())}
        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        self.vectors = np.array(list(word_embeddings.values()), dtype=np.float32)
        
        # Normalize all base vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.vectors = self.vectors / norms
        
        # Populate mock subword n-grams
        for word, vec in word_embeddings.items():
            ngrams = self.extract_character_ngrams(word)
            sub_vec = vec / len(ngrams)
            for ng in ngrams:
                if ng not in self.ngram_vectors:
                    self.ngram_vectors[ng] = sub_vec.copy()
                else:
                    self.ngram_vectors[ng] += sub_vec

    def get_vector(self, word: str) -> np.ndarray:
        """Retrieves exact vector if known; falls back to FastText n-gram composition if OOV."""
        if word in self.vocab:
            return self.vectors[self.vocab[word]]
        
        # OOV handling via FastText n-gram summation
        ngrams = self.extract_character_ngrams(word)
        matching = [self.ngram_vectors[ng] for ng in ngrams if ng in self.ngram_vectors]
        if matching:
            v_sum = np.sum(matching, axis=0)
            norm = np.linalg.norm(v_sum)
            return v_sum / (norm + 1e-9)
        
        # Random fallback if zero n-grams match
        rng = np.random.RandomState(abs(hash(word)) % (2**32))
        rand_v = rng.randn(self.embedding_dim).astype(np.float32)
        return rand_v / np.linalg.norm(rand_v)

    def solve_analogy(self, a: str, b: str, c: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Solves vector analogy: A is to B as C is to ? (target = B - A + C).
        Example: Man (A) : King (B) :: Woman (C) : Queen (?)
        """
        v_a = self.get_vector(a)
        v_b = self.get_vector(b)
        v_c = self.get_vector(c)
        
        # Ideal target vector
        query = v_b - v_a + v_c
        query = query / (np.linalg.norm(query) + 1e-9)
        
        # Cosine similarity against all in-vocabulary items
        sims = np.dot(self.vectors, query)
        ranked = np.argsort(sims)[::-1]
        
        results = []
        for idx in ranked:
            candidate = self.inv_vocab[idx]
            if candidate not in [a, b, c]:
                results.append((candidate, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Classical Word Analogy & Subword N-Gram Explorer", divider="rainbow")

st.markdown("""
Test classical vector arithmetic relationships (B - A + C approx D) and inspect how **FastText character n-grams** 
synthesize embeddings for out-of-vocabulary terms.
""")

# Setup synthetic semantic space
@st.cache_data
def get_bench_data():
    seed = 101
    np.random.seed(seed)
    dim = 16
    
    # Semantic latent bases
    royalty = np.array([1.5, 0.2, -0.4, 1.2] + [0.0]*(dim - 4))
    gender = np.array([0.1, 1.8, -0.1, 0.0] + [0.0]*(dim - 4))
    capital = np.array([-0.3, 0.0, 1.9, 0.8] + [0.0]*(dim - 4))
    country = np.array([0.0, -0.2, 1.4, -0.5] + [0.0]*(dim - 4))
    science = np.array([-0.8, -0.5, -0.2, -1.5] + [0.0]*(dim - 4))
    
    vocab_map = {
        "man": gender * 1.0 + np.random.randn(dim)*0.05,
        "woman": gender * -1.0 + np.random.randn(dim)*0.05,
        "king": royalty * 1.0 + gender * 1.0 + np.random.randn(dim)*0.05,
        "queen": royalty * 1.0 + gender * -1.0 + np.random.randn(dim)*0.05,
        "prince": royalty * 0.7 + gender * 0.8 + np.random.randn(dim)*0.05,
        "princess": royalty * 0.7 + gender * -0.8 + np.random.randn(dim)*0.05,
        "paris": capital * 1.0 + country * 0.8 + np.random.randn(dim)*0.05,
        "france": country * 1.2 + np.random.randn(dim)*0.05,
        "tokyo": capital * 1.0 + country * 1.5 + np.random.randn(dim)*0.05,
        "japan": country * 1.8 + np.random.randn(dim)*0.05,
        "rome": capital * 0.9 + country * 0.7 + np.random.randn(dim)*0.05,
        "italy": country * 1.1 + np.random.randn(dim)*0.05,
        "einstein": science * 1.5 + np.random.randn(dim)*0.05,
        "physicist": science * 1.2 + np.random.randn(dim)*0.05,
    }
    return vocab_map, dim

bench_vocab, bench_dim = get_bench_data()

wb_tool = st.radio("Select Interactive Workbench Mode", ["Word Vector Analogy Calculator", "FastText Subword N-Gram Inspector"], horizontal=True)

if wb_tool == "Word Vector Analogy Calculator":
    st.markdown("#### Vector Analogy Formulation: A : B :: C : ? (`Target = B - A + C`)")
    wb_col1, wb_col2, wb_col3 = st.columns(3)
    
    with wb_col1:
        base_a = st.selectbox("Base Word A", options=list(bench_vocab.keys()), index=0)
    with wb_col2:
        target_b = st.selectbox("Analogous Word B", options=list(bench_vocab.keys()), index=2)
    with wb_col3:
        query_c = st.selectbox("Query Word C", options=list(bench_vocab.keys()), index=1)
        
    vec_a = bench_vocab[base_a] / np.linalg.norm(bench_vocab[base_a])
    vec_b = bench_vocab[target_b] / np.linalg.norm(bench_vocab[target_b])
    vec_c = bench_vocab[query_c] / np.linalg.norm(bench_vocab[query_c])
    
    target_pred = vec_b - vec_a + vec_c
    target_pred_norm = target_pred / np.linalg.norm(target_pred)
    
    scores = {}
    for word, v in bench_vocab.items():
        v_norm = v / np.linalg.norm(v)
        scores[word] = float(np.dot(v_norm, target_pred_norm))
        
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown(f"**Target Vector Equation**: `Vector('{target_b}') - Vector('{base_a}') + Vector('{query_c}')`")
    
    res_df = pd.DataFrame([
        {"Rank": i+1, "Candidate Word": w, "Cosine Similarity": f"{s:.4f}", "Status": "Input Word" if w in [base_a, target_b, query_c] else "Top Match"}
        for i, (w, s) in enumerate(sorted_scores[:6])
    ])
    st.dataframe(res_df, use_container_width=True, hide_index=True)

else:
    st.markdown("#### FastText Subword Character N-Gram Decomposition")
    user_test_word = st.text_input("Enter a compound or misspelled word to decompose:", value="subroyalist")
    
    # Extract 3-grams, 4-grams, 5-grams
    t_word = f"<{user_test_word}>"
    ng_3 = [t_word[i:i+3] for i in range(len(t_word)-2)]
    ng_4 = [t_word[i:i+4] for i in range(len(t_word)-3)]
    ng_5 = [t_word[i:i+5] for i in range(len(t_word)-4)]
    
    col_ng1, col_ng2, col_ng3 = st.columns(3)
    with col_ng1:
        st.markdown(f"**3-grams ({len(ng_3)} tokens)**")
        st.code(", ".join(ng_3))
    with col_ng2:
        st.markdown(f"**4-grams ({len(ng_4)} tokens)**")
        st.code(", ".join(ng_4))
    with col_ng3:
        st.markdown(f"**5-grams ({len(ng_5)} tokens)**")
        st.code(", ".join(ng_5))
        
    st.info(f"FastText represents `{user_test_word}` as the normalized vector sum: v('{user_test_word}') = Sum(z_g) across all {len(ng_3) + len(ng_4) + len(ng_5) + 1} constituent subwords.")


# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does Word2Vec Skip-gram use Negative Sampling instead of standard Softmax?",
        "a": "Standard Softmax requires calculating the partition function (denominator sum) over the entire vocabulary V for every training step. With vocabularies exceeding 500,000 words, this takes O(V) operations per token update. Negative Sampling reformulates the task into binary logistic regression: predicting whether a pair is a true context word or noise. It only updates the true target token and k sampled noise words (typically k=5 to 20), reducing training complexity to O(k)."
    },
    {
        "q": "How does GloVe's objective function differ fundamentally from Word2Vec Skip-gram?",
        "a": "Skip-gram trains by streaming local context windows and computing gradient updates on individual co-occurrence events (stochastic local sampling). GloVe builds a global word-word co-occurrence matrix X across the entire corpus upfront, then fits a weighted log-bilinear regression model: w_i . w~_j + b_i + b~_j = log(X_ij). GloVe weights errors by a function f(X_ij) to prevent frequent stopwords from overwhelming the loss."
    },
    {
        "q": "Why can FastText represent unseen out-of-vocabulary (OOV) words at inference time while Word2Vec fails?",
        "a": "Word2Vec maps whole word strings directly to unique integer IDs in a dictionary. Any unseen string has no corresponding row in W_in and must map to a generic [UNK] token. FastText represents every word as the sum of its character n-gram vectors (e.g., 3-to-6 character sequences plus boundary markers). Even if a whole word was never seen, its constituent subwords (stems, prefixes, suffixes) frequently occurred in other words during training."
    },
    {
        "q": "What is the mathematical rationale behind using unigram distribution raised to the 0.75 power for Negative Sampling?",
        "a": "In natural languages, token frequencies follow Zipf's Law: a few words ('the', 'of', 'and') occur millions of times, while rare technical words occur only a few times. Raising counts to the 0.75 power dampens the frequency of ultra-frequent stop words while boosting the sampling probability of rare words. For example, if word A occurs 1,000,000 times and word B occurs 1,000 times, the raw frequency ratio is 1000:1, but the 0.75 power ratio is (10^6)^0.75 : (10^3)^0.75 = 31622 : 177 = 178:1."
    },
    {
        "q": "Why were classical static embeddings entirely replaced by contextual Transformer embeddings in modern LLMs?",
        "a": "Classical static embeddings suffer from the Polysemy Bottleneck: each vocabulary entry receives exactly one immutable coordinate in R^D. For words with multiple senses (e.g., 'apple' as a fruit vs. company, or 'bank' as a river edge vs. financial institution), the static vector becomes an unnatural geometric average of all its senses. Modern Transformer LLMs use self-attention to dynamically synthesize contextualized embeddings conditioned on the surrounding sequence."
    }
]

for i, qa in enumerate(qas, 1):
    with st.expander(f"Q{i}: {qa['q']}"):
        st.markdown(f"**Answer**: {qa['a']}")

st.markdown("""
### Production Traps & Failure Modes
""")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1 (Fatal): Using Static Embedding Lookup as Document Representation**
    * *Failure*: Averaging static Word2Vec or GloVe vectors across an entire document (`doc_vec = mean([v(w) for w in doc])`) destroys word order, syntax, and negation.
    * *Fix*: Use dense bi-encoder sentence transformers (e.g., `BGE-M3`, `all-MiniLM-L6-v2`) which contextualize relationships across full sentences before pooling.
    """)
    st.error("""
    **Trap 2 (Fatal): Training Word2Vec without Subsampling Frequent Words**
    * *Failure*: Words like `"the"` and `"is"` dominate 40%+ of all sliding windows, saturating gradient updates and starving semantic content words of representation learning.
    * *Fix*: Implement Mikolov's subsampling formula: discard word w with probability P(w) = 1 - sqrt(t / f(w)), where threshold t approx 10^-5.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding GloVe Context Vector Summation**
    * *Failure*: Taking only the primary word matrix W and discarding the context matrix W~.
    * *Fix*: Always set the final word vector to W_{final} = W + W~. In symmetric co-occurrence windows, both matrices learn identical semantics with independent noise; summing them acts as a powerful regularizer.
    """)
    st.warning("""
    **Trap 4 (Latency / Memory Trap): FastText N-Gram Hash Collisions**
    * *Failure*: Setting FastText's n-gram hash bucket size too small (< 1,000,000) causes distinct character sequences to map to the same row in memory, causing catastrophic representation crosstalk.
    * *Fix*: Ensure hash bucket sizing exceeds 2 * 10^6 entries or transition to modern Byte-Pair Encoding (BPE) vocabularies.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**CBOW vs. Skip-gram**\n\nCBOW predicts 1 target from context average (faster, good for syntax). Skip-gram predicts context from 1 center word (slower, better for rare words).")
with fc_col2:
    st.info("**Negative Sampling**\n\nReplaces full O(V) softmax with binary logistic regression updates over 1 true target and k sampled noise words (O(k)).")
with fc_col3:
    st.info("**GloVe Core Idea**\n\nRatios of co-occurrence probabilities encode semantic distinctions. Optimized via weighted least squares regression: w_i^T w~_j + b_i + b~_j = log(X_{ij}).")
with fc_col4:
    st.info("**FastText OOV Power**\n\nDecomposes words into character n-grams. Unseen words are represented by summing known subword n-gram vectors.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of classical word embeddings, objective functions, and their bridge to modern LLM input embeddings:
""")

c1 = st.checkbox("I can explain why computing full Softmax over a 500,000-word vocabulary is computationally intractable.")
c2 = st.checkbox("I understand the architectural differences between CBOW (context-to-target) and Skip-gram (target-to-context).")
c3 = st.checkbox("I understand how GloVe exploits ratios of co-occurrence probabilities to isolate semantic distinctions.")
c4 = st.checkbox("I can describe how FastText solves the out-of-vocabulary (OOV) challenge using character n-gram summation.")
c5 = st.checkbox("I know why modern foundation LLMs retained the input lookup matrix `nn.Embedding(V, D)` but discarded static pooling in favor of self-attention.")
c6 = st.checkbox("I can identify the Polysemy Bottleneck and explain why static embeddings fail to disambiguate words with multiple meanings.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered classical word embeddings, their objective formulations, and their structural legacy in modern foundation models!")
