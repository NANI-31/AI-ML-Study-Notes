import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="7.1 Position Representation (Why Position Matters, Order, Absolute vs Relative)",
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
st.markdown('<div class="main-title">7.1 Position Representation: Permutation Invariance and Sequence Order</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Set-Based Bag-of-Words Degeneration to Absolute and Relative Coordinate Systems — How Transformers Learn Sequential Order.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Breaking Permutation Invariance in Self-Attention", divider="rainbow")

st.markdown("""
Unlike Recurrent Neural Networks (RNNs) that inherently process tokens step-by-step in sequential time, 
**Self-Attention is fundamentally permutation equivariant**: it operates on an unordered multiset of vectors. 
To distinguish `"dog bites man"` from `"man bites dog"`, positional information must be explicitly injected 
into the representation manifold:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("TOKENS", "Input Tokens\n['The', 'dog', 'bites', 'the', 'man']\nDiscrete Token IDs: [B, N]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    arch_graph.node("EMBED", "Token Embeddings\nX_tok = W_E[tokens]\nShape: [B, N, d_model]\n(Zero Order Information)", shape="box", fillcolor="#312E81", color="#818CF8")

    with arch_graph.subgraph(name="cluster_pos") as c:
        c.attr(bgcolor="#0F172A", label="Positional Injection Paradigms", fontcolor="#38BDF8", color="#38BDF8")
        c.node("ABS", "Absolute Position (Vaswani / BERT)\nx_i = X_tok[i] + P[i]\nP[i] in R^d_model (Fixed / Learned)", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("REL", "Relative Position (T5 / Shaw)\nBias added to Attention Scores\nS_ij = (q_i . k_j) / sqrt(d_k) + b(i - j)", shape="box", fillcolor="#1E293B", color="#A855F7")
        c.node("ROPE", "Rotary Position Embedding (RoPE - Llama)\nRotate Q and K in complex plane per layer\nq_i_rot = R_Theta(i) . q_i", shape="box", fillcolor="#042F2E", color="#14B8A6")

    arch_graph.node("ATTN", "Position-Aware Multi-Head Attention\nDistinguishes 'dog bites man' from 'man bites dog'\nPreserves syntax, grammar, and narrative flow", shape="box", fillcolor="#042F2E", color="#10B981")

    # Edges
    arch_graph.edge("TOKENS", "EMBED")
    arch_graph.edge("EMBED", "ABS")
    arch_graph.edge("EMBED", "REL")
    arch_graph.edge("EMBED", "ROPE")
    arch_graph.edge("ABS", "ATTN")
    arch_graph.edge("REL", "ATTN")
    arch_graph.edge("ROPE", "ATTN")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### The Fundamental Mathematical Problem
    Let P be an arbitrary permutation matrix (e.g., shuffling token order). 
    In raw self-attention without positional encoding:
    ```
    Attention(P X, P X, P X) = P . Attention(X, X, X)
    ```
    * The output representations are simply permuted by the exact same matrix P.
    * If pooled (e.g., mean-pooled), the representation is **completely identical**:
      `Mean(Attention(P X)) == Mean(Attention(X))`.
    * Without positional encoding, the Transformer degenerates into an un-ordered bag-of-words model!
    """)

# ==============================================================================
# SECTION 2: 1. WHY POSITION MATTERS
# ==============================================================================
st.header("2. Why Position Matters: Syntax, Word Order, and Semantic Inversion", divider="rainbow")

st.markdown("""
In natural human language, **word order is the primary vehicle for syntactic hierarchy and semantic truth conditions**. 
Swapping the positions of two words frequently reverses meaning entirely.
""")

col_wpm1, col_wpm2 = st.columns([1, 1])

with col_wpm1:
    st.markdown("""
    ### Minimal Word Order Pairs
    Consider how identical word bags produce contradictory meanings:
    1. **Agent vs. Patient Inversion**:
       * Sequence A: *"The cat caught the mouse."* (Cat is predator, mouse is prey).
       * Sequence B: *"The mouse caught the cat."* (Biologically absurd, inverted roles).
    2. **Negation & Scope Flips**:
       * Sequence A: *"I believe that he is not coming."*
       * Sequence B: *"I do not believe that he is coming."*
    3. **Programming Code Semantics**:
       * Sequence A: `x = y - z`
       * Sequence B: `x = z - y` (Completely inverted mathematical value).
    """)

with col_wpm2:
    st.markdown("""
    ### Linguistic Phenomena Governed by Position:
    * **Constituency & Phrasing**: Modifiers attach to adjacent noun phrases (`"the tall girl's dog"` vs `"the girl's tall dog"`).
    * **Coreference Proximity**: Pronouns (`"he"`, `"it"`) preferentially bind to recent antecedent mentions in preceding sentences.
    * **Discourse & Argument Flow**: Logical connectors (`"therefore"`, `"however"`, `"because"`) define causal directionality.
    """)

# ==============================================================================
# SECTION 3: 2. SEQUENCE ORDER
# ==============================================================================
st.header("3. Sequence Order: Permutation Equivariance vs. Permutation Invariance", divider="rainbow")

st.markdown("""
To design effective neural architectures, deep learning engineers must distinguish between 
**Permutation Invariance** and **Permutation Equivariance**.
""")

col_so1, col_so2 = st.columns([1, 1])

with col_so1:
    st.markdown("""
    ### Mathematical Definitions
    Let f be a function taking a sequence of vectors X = [x_1, x_2, ..., x_N], and let pi be a permutation operator:
    
    1. **Permutation Invariance**:
       ```
       f(pi(X)) = f(X)
       ```
       * The output does not change when the input is shuffled.
       * *Examples*: Global Mean Pooling, Max Pooling, Sum Pooling, Set Transformer.
       
    2. **Permutation Equivariance**:
       ```
       f(pi(X)) = pi(f(X))
       ```
       * Permuting the input permutes the output in the exact same manner.
       * *Example*: Raw Multi-Head Self-Attention without positional encodings.
    """)

with col_so2:
    st.markdown("""
    ### Why Equivariance Still Fails Sequence Modeling
    * If you feed `["dog", "bites", "man"]` into raw self-attention, the vector for `"dog"` computes dot products with `"bites"` and `"man"`.
    * If you feed `["man", "bites", "dog"]`, the vector for `"dog"` still computes the exact same dot products with `"bites"` and `"man"`!
    * The internal vector representation of `"dog"` is **completely identical** in both sentences. 
    * It cannot know whether `"dog"` acted as the grammatical subject or the direct object!
    """)
    st.error("Conclusion: Self-attention requires breaking permutation equivariance by explicitly binding positional coordinates to token features.")

# ==============================================================================
# SECTION 4: 3. POSITIONAL ENCODING
# ==============================================================================
st.header("4. Positional Encoding: Coordinate Injection into Latent Manifolds", divider="rainbow")

st.markdown("""
**Positional Encoding** is the technique of imbuing each token vector with continuous geometric coordinates 
that denote its position within the sequence.
""")

col_pe1, col_pe2 = st.columns([1, 1])

with col_pe1:
    st.markdown("""
    ### Additive Coordinate Injection
    In early Transformer architectures (Vaswani et al. 2017, BERT, GPT-2), positional information is added directly to token embeddings:
    ```
    z_i = x_i + p_i    # Shape: [d_model]
    ```
    Where:
    * x_i in R^{d_{model}} is the pure lexical token embedding.
    * p_i in R^{d_{model}} is the positional coordinate vector for index i.
    
    ### Why Does Adding Work? Why Doesn't it Corrupt the Word Embedding?
    * In high-dimensional spaces (d_{model} >= 768), vectors have enormous geometric capacity.
    * The network learns to project token semantics into a subset of orthogonal subspaces, while reserving other linear subspaces for positional signals!
    * At initialization, |x_i|_2 and |p_i|_2 are balanced so that neither signal drowns out the other.
    """)

with col_pe2:
    st.markdown("""
    ### Requirements of an Ideal Positional Encoding:
    1. **Deterministic & Unique**: Every position i must map to a unique vector coordinate.
    2. **Distance Sensitivity**: Tokens that are close together (i and i+1) should have high geometric similarity; distant tokens should be nearly orthogonal.
    3. **Generalization / Extrapolation**: The encoding should ideally extrapolate gracefully to sequence lengths unseen during training (N > L_{train}).
    4. **Bounded Norm**: |p_i| must remain bounded to avoid destabilizing layer normalization and gradient scaling.
    """)

# ==============================================================================
# SECTION 5: 4. ABSOLUTE POSITION
# ==============================================================================
st.header("5. Absolute Position: Canonical Coordinates from 0 to N-1", divider="rainbow")

st.markdown("""
**Absolute Positional Encoding** assigns a unique, fixed vector coordinate to each index i in {0, 1, ..., N-1}, 
treating position as an immutable global timestamp along the sequence timeline.
""")

col_abs1, col_abs2 = st.columns([1, 1])

with col_abs1:
    st.markdown("""
    ### Two Dominant Absolute Approaches:
    1. **Sinusoidal Positional Encoding (Vaswani 2017)**:
       Deterministic trigonometric functions with geometric progression of wavelengths:
       ```
       PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
       ```
       * Zero trainable parameters.
       * Low frequencies encode macro-position; high frequencies encode local micro-position.
       
    2. **Learned Absolute Position Embeddings (BERT, GPT-2, RoBERTa)**:
       A trainable parameter matrix W_{pos} in R^{L_{max} x d_{model}}:
       * Position 0 looks up row 0; position 1 looks up row 1.
       * Learns custom positional geometry directly from gradient descent.
    """)

with col_abs2:
    st.markdown("""
    ### Fatal Flaw of Absolute Encodings: The Hard Context Wall
    * **Zero Extrapolation Past L_{max}**: If a model is trained with L_{max} = 2048, position 2049 has no corresponding row in W_{pos}! The model cannot run on even 2049 tokens without error.
    * **Translation Invariance Violation**:
      The relationship between `"The"` and `"dog"` in `"The dog barked"` (positions 0 and 1) is identical to `"I saw that the dog barked"` (positions 4 and 5). 
      Absolute encodings assign completely different coordinate vectors to both pairs, forcing the network to relearn identical grammar rules at every position!
    """)

# ==============================================================================
# SECTION 6: 5. RELATIVE POSITION
# ==============================================================================
st.header("6. Relative Position: Encoding Pairwise Distances (i - j)", divider="rainbow")

st.markdown("""
Human language is inherently **translation invariant**: what matters is the **relative distance** between words 
(i - j), not whether a phrase occurs at token 5 or token 5,000.
""")

col_rel1, col_rel2 = st.columns([1, 1])

with col_rel1:
    st.markdown("""
    ### Relative Positional Formulations
    1. **Shaw et al. (2018) & Transformer-XL**:
       Modifies the attention score calculation to include a learned relative position vector a_{i-j}:
       ```
       Scores_{ij} = (q_i . k_j^T + q_i . a_{i-j}^T) / sqrt(d_k)
       ```
       Distances are clipped to a maximum window k (e.g., k in [-16, +16]).
       
    2. **T5 Relative Position Bias (Raffel et al., 2020)**:
       Adds a scalar learned bias b(i - j) directly to attention logits:
       ```
       Scores_{ij} = (q_i . k_j^T) / sqrt(d_k) + b(i - j)
       ```
       Uses logarithmic bucketed bins: fine resolution for nearby tokens, coarse resolution for distant tokens.
    """)

with col_rel2:
    st.markdown("""
    ### Why Relative Position Won the Modern Era
    * **Translation Invariance**: The relative distance between an adjective and its noun is always -1, regardless of whether the phrase appears at the start, middle, or end of a book.
    * **Natural Extrapolation**: Because the model only conditions on (i - j), it generalizes to much longer contexts than absolute embeddings.
    * **Predecessor to RoPE**: Rotary Position Embedding (RoPE) mathematically achieved the ultimate synthesis: injecting relative distance (i - j) directly through 2D rotations of Query and Key vectors!
    """)

st.markdown("### Absolute vs. Relative Position Paradigm Matrix")
df_pos_master = pd.DataFrame({
    "Criterion": ["Absolute (Vaswani Sinusoid)", "Learned Absolute (BERT / GPT-2)", "Relative Bias (T5)", "Rotary (RoPE - Llama 3)"],
    "Injection Point": ["Input layer (+)", "Input layer (+)", "Attention logit (+ scalar)", "Query & Key rotation per layer"],
    "Trainable Parameters": ["0 (Deterministic)", "L_max * d_model", "32 buckets per head", "0 (Angle frequencies)"],
    "Translation Invariant": ["No (Global coordinates)", "No (Global coordinates)", "Yes (Depends purely on i - j)", "Yes (Rotary dot-product property)"],
    "Context Extrapolation": ["Poor past L_train", "Fails (Hard limit L_max)", "Moderate", "State of the Art (via YaRN / NTK)"]
})
st.dataframe(df_pos_master, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Pure Python Permutation Invariance & Positional Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation proves:
1. That **raw Self-Attention is strictly permutation equivariant** (scrambling words scrambles outputs identically).
2. How adding **Positional Encodings** breaks equivariance, allowing the model to distinguish identical bags of words.
""")

class PositionRepresentationLab:
    """
    Production-grade educational implementation proving:
    1. Permutation equivariance of un-encoded self-attention.
    2. Sinusoidal positional encoding generation.
    3. Breaking equivariance to encode sequential order.
    """
    def __init__(self, d_model: int = 16, seed: int = 42):
        self.d_model = d_model
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.1

    def generate_sinusoidal_pe(self, seq_len: int) -> np.ndarray:
        """Generates classical Vaswani et al. (2017) Sinusoidal Positional Encodings."""
        pe = np.zeros((seq_len, self.d_model), dtype=np.float32)
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def raw_attention(self, X: np.ndarray) -> np.ndarray:
        """Standard self-attention without positional encoding."""
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        safe_s = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(safe_s) / np.sum(np.exp(safe_s), axis=-1, keepdims=True)
        return np.dot(weights, V)

    def position_aware_attention(self, X: np.ndarray) -> np.ndarray:
        """Self-attention with additive sinusoidal positional encoding."""
        seq_len = X.shape[0]
        pe = self.generate_sinusoidal_pe(seq_len)
        X_pos = X + pe
        return self.raw_attention(X_pos)

st.code('''
class PositionRepresentationLab:
    def __init__(self, d_model=16, seed=42):
        self.W_q, self.W_k, self.W_v = ...
        
    def generate_sinusoidal_pe(self, seq_len):
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        return pe
        
    def raw_attention(self, X):
        # Permutation equivariant without positional encodings
        return Softmax((X @ W_q @ (X @ W_k)^T) / sqrt(d)) @ (X @ W_v)
        
    def position_aware_attention(self, X):
        # Breaks equivariance: X_pos = X + PE
        return self.raw_attention(X + self.generate_sinusoidal_pe(len(X)))
''', language="python")


# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Permutation Invariance & Word Order Scrambler", divider="rainbow")

st.markdown("""
Observe the **Permutation Equivariance Failure** in action! 
Scramble the sequence `"The dog bites the man"` into `"The man bites the dog"` 
and watch how un-encoded self-attention produces identical token vectors, whereas position-encoded attention successfully differentiates them!
""")

lab = PositionRepresentationLab(d_model=8, seed=42)

# Synthetic lexical vectors for vocabulary
np.random.seed(101)
vocab_vectors = {
    "The": np.array([1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "dog": np.array([0.0, 2.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    "bites": np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "the": np.array([1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "man": np.array([0.0, 0.0, 0.0, 2.8, 0.0, 0.5, 0.0, 0.0])
}

sent_orig = ["The", "dog", "bites", "the", "man"]
sent_perm = ["The", "man", "bites", "the", "dog"]

X_orig = np.array([vocab_vectors[w] for w in sent_orig])
X_perm = np.array([vocab_vectors[w] for w in sent_perm])

# Compute raw attention (no position)
out_raw_orig = lab.raw_attention(X_orig)
out_raw_perm = lab.raw_attention(X_perm)

# Compute position-aware attention
out_pos_orig = lab.position_aware_attention(X_orig)
out_pos_perm = lab.position_aware_attention(X_perm)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("#### Sentence 1: `'The dog bites the man'`")
    st.markdown("**Vector for `'dog'` (Subject, Position 1)**:")
    st.write("Raw (No Position):", np.round(out_raw_orig[1, :4], 3))
    st.write("Position-Aware:", np.round(out_pos_orig[1, :4], 3))

with col_exp2:
    st.markdown("#### Sentence 2: `'The man bites the dog'`")
    st.markdown("**Vector for `'dog'` (Object, Position 4)**:")
    st.write("Raw (No Position):", np.round(out_raw_perm[4, :4], 3))
    st.write("Position-Aware:", np.round(out_pos_perm[4, :4], 3))

st.markdown("---")
diff_raw = np.linalg.norm(out_raw_orig[1] - out_raw_perm[4])
diff_pos = np.linalg.norm(out_pos_orig[1] - out_pos_perm[4])

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.error(f"❌ Raw Attention Difference: **{diff_raw:.6f}** (Strictly 0.0! The model cannot tell that 'dog' changed from subject to object!)")
with col_res2:
    st.success(f"✅ Position-Aware Difference: **{diff_pos:.4f}** (Distinct coordinates! The model successfully distinguishes subject from object.)")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is Self-Attention permutation equivariant rather than permutation invariant?",
        "a": "A function is permutation equivariant if shuffling the input shuffles the output identically: f(P X) = P f(X). In self-attention, row i of the output matrix is computed from query q_i attending to all keys. Shuffling the rows of input matrix X simply shuffles which row corresponds to which token. The output tensor has the exact same rows in shuffled order. It is NOT permutation invariant because the output does change order; it becomes invariant only if you apply a symmetric pooling operation (e.g., mean pooling over the sequence dimension)."
    },
    {
        "q": "Why did the original Transformer choose sinusoidal functions over simple integer indices (e.g., pos = 1, 2, 3)?",
        "a": "Using raw integer indices pos in [1, 2, ..., N] causes values to grow without bound as sequence length increases. At pos=2000, the positional value would completely overpower the unit-scale token embedding, destabilizing layer normalization and gradients. Normalizing pos to [0, 1] by dividing by N creates a different flaw: pos=0.5 would represent completely different absolute token distances in a 10-word sentence versus a 1000-word sentence. Sinusoidal functions provide bounded, continuous, scale-invariant representations."
    },
    {
        "q": "What is the primary architectural advantage of Relative Positional Encodings over Absolute Positional Encodings?",
        "a": "Relative positional encodings condition attention scores purely on pairwise token distance (i - j). This enforces translation invariance: the syntactic relationship between an adjective and a noun is identical regardless of whether they appear at token 5 or token 5,000. Furthermore, because the model only conditions on local distance deltas, relative encodings extrapolate significantly better to sequence lengths beyond the training context window."
    },
    {
        "q": "Why does adding positional encodings to token embeddings (X + PE) not corrupt lexical word identity?",
        "a": "In high-dimensional vector spaces (e.g., d_model = 4096), there are thousands of orthogonal linear subspaces. The network learns during pretraining to allocate separate subspaces for lexical semantic meaning and positional coordinates. Because high-dimensional vectors can be nearly orthogonal while sharing the same space, the self-attention projection weights learn to project and isolate the semantic features and positional features independently."
    },
    {
        "q": "Why do modern foundation models (Llama 3, Mistral) avoid additive positional encodings in favor of Rotary Position Embeddings (RoPE)?",
        "a": "Additive encodings (x + PE) inject position only at the input layer. As representations propagate through 32+ deep non-linear layers, positional signals degrade and become entangled with semantic abstractions. RoPE applies 2D rotation matrices directly to Query and Key vectors inside each attention head at every layer. The dot product (R(i) Q) . (R(j) K)^T mathematically factors out absolute coordinates, encoding relative distance (i - j) precisely where attention weights are computed."
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
    **Trap 1 (Fatal): Hard Context Wall with Learned Absolute Embeddings**
    * *Failure*: Training a model with learned positional table `nn.Embedding(2048, d_model)` and attempting inference on a 2049-token prompt, triggering an immediate `IndexError: index out of range`.
    * *Fix*: Use Rotary Position Embeddings (RoPE) or relative biases that do not require fixed-size lookup tables.
    """)
    st.error("""
    **Trap 2 (Fatal): Unbalanced Norm Ratios in Custom Additive Encodings**
    * *Failure*: Initializing custom positional embeddings with variance 10x larger than token embeddings, causing position to drown out word identity.
    * *Fix*: Always verify |X_{tok}|_2 \approx |PE|_2, or multiply token embeddings by sqrt{d_{model}} as in original Vaswani.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding Positional Sinks in Extrapolation**
    * *Failure*: Evaluating long context sequences without accounting for attention sinks at token 0, causing high perplexity spikes.
    * *Fix*: Ensure initial tokens remain anchored in position tables.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Materializing Full [N, N] Relative Distance Matrices**
    * *Failure*: Constructing an explicit N x N relative distance matrix in GPU VRAM for N = 32,768, causing 2GB+ memory allocations per head.
    * *Fix*: Use Rotary Position Embeddings (RoPE), which apply relative distance in O(N) memory per token!
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Permutation Equivariance**\n\nRaw self-attention treats sequences as unordered bags of vectors. Positional encoding is essential to break equivariance.")
with fc_col2:
    st.info("**Additive Encoding**\n\nx_i = X_tok[i] + PE[i]. Works because high-dimensional spaces allow orthogonal allocation of syntax and position.")
with fc_col3:
    st.info("**Translation Invariance**\n\nRelative position encodes (i - j). The relationship between adjacent words is identical anywhere in the document.")
with fc_col4:
    st.info("**RoPE Supremacy**\n\nInjects relative distance directly through 2D complex plane rotations of Q and K vectors at every layer.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of position representations, sequence order, and coordinate injection:
""")

c1 = st.checkbox("I can explain why raw self-attention is permutation equivariant and fails without position encodings.")
c2 = st.checkbox("I understand why adding positional vectors (X + PE) does not corrupt lexical token semantics.")
c3 = st.checkbox("I know the mathematical difference between Absolute Position (index i) and Relative Position (i - j).")
c4 = st.checkbox("I understand why learned absolute position embeddings hit a hard context wall at L_max.")
c5 = st.checkbox("I can describe how sinusoidal position encodings encode geometric wavelengths across dimensions.")
c6 = st.checkbox("I know why modern LLMs standardized on Rotary Position Embeddings (RoPE) over additive encodings.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered position representation, permutation equivariance, and absolute vs relative coordinate mechanisms!")
