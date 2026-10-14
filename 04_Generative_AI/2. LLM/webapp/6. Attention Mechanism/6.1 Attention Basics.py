import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.1 Attention Basics (Query, Key, Value, Scores, Weights, Weighted Sum)",
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
st.markdown('<div class="main-title">6.1 Attention Basics: The Query-Key-Value Information Retrieval Paradigm</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From the Fixed-Vector Bottleneck to Soft-Addressing Dynamic Memory — Foundations of Queries, Keys, Values, Attention Scores, and Context Synthesis.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Soft-Addressing Memory Retrieval Paradigm", divider="rainbow")

st.markdown("""
The attention mechanism can be understood through the lens of a **differentiable database retrieval system**. 
In a classical database, a discrete query string matches an exact key to return a single discrete value. 
In a neural network, a continuous **Query vector** computes soft dot-product affinities against multiple **Key vectors**, 
returning a **probabilistically weighted sum of Value vectors**:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("INPUT", "Input Token Hidden States: X\nShape: [Seq_Len, d_model]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    
    with arch_graph.subgraph(name="cluster_projections") as c:
        c.attr(bgcolor="#0F172A", label="Projection Subspace (Learned Weights)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("Q", "Queries (Q = X . W_q)\nWhat each token seeks\nShape: [Seq_Len, d_k]", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("K", "Keys (K = X . W_k)\nWhat each token advertises\nShape: [Seq_Len, d_k]", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("V", "Values (V = X . W_v)\nInformation payload delivered\nShape: [Seq_Len, d_v]", shape="box", fillcolor="#312E81", color="#818CF8")

    arch_graph.node("SCORES", "Raw Alignment Scores: S\nS = Q . K^T / sqrt(d_k)\nUnnormalized Dot-Product Energy", shape="box", fillcolor="#1E293B", color="#F59E0B")
    arch_graph.node("WEIGHTS", "Softmax Probabilities: Alpha\nAlpha = Softmax(S, dim=-1)\nRows sum to 1.0 (Attention Weights)", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("CONTEXT", "Context Output: Out\nOut = Alpha . V\nDynamic Weighted Sum of Values", shape="box", fillcolor="#042F2E", color="#10B981")

    # Edges
    arch_graph.edge("INPUT", "Q")
    arch_graph.edge("INPUT", "K")
    arch_graph.edge("INPUT", "V")
    arch_graph.edge("Q", "SCORES", label="Dot Product")
    arch_graph.edge("K", "SCORES")
    arch_graph.edge("SCORES", "WEIGHTS", label="Scale + Softmax")
    arch_graph.edge("WEIGHTS", "CONTEXT", label="Weighting")
    arch_graph.edge("V", "CONTEXT", label="Payload Delivery")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Fundamental Triad: Queries, Keys, and Values
    * **Query (Q)**: *"What kind of information am I looking for?"* (e.g., a pronoun asking for its antecedent noun).
    * **Key (K)**: *"What kind of content do I contain?"* (e.g., a noun advertising its gender, number, and syntactic case).
    * **Value (V)**: *"What actual semantic payload do I provide if selected?"*
    * **Attention Weight (alpha)**: The degree of compatibility between Query and Key (0.0 <= alpha <= 1.0).
    * **Context Vector**: The blended sum of all Values, modulated by their respective compatibility weights.
    """)

# ==============================================================================
# SECTION 2: 1. WHAT IS ATTENTION?
# ==============================================================================
st.header("2. What is Attention? Soft Addressing and Dynamic Routing", divider="rainbow")

st.markdown("""
In neural networks, **Attention** is a mechanism that allows a model to dynamically focus on specific parts 
of an input sequence when processing a given token, rather than treating all input tokens with equal weight 
or forcing an entire sequence through a static vector bottleneck.
""")

col_att1, col_att2 = st.columns([1, 1])

with col_att1:
    st.markdown("""
    ### Hard vs. Soft Attention
    * **Hard Attention (Non-differentiable)**:
      * Selects exactly one token position i using a discrete categorical decision (z_i in {0, 1}).
      * Requires Reinforcement Learning (policy gradient / REINFORCE) to train because discrete sampling has no derivative (dz / dx is undefined).
    * **Soft Attention (Differentiable - Standard)**:
      * Computes a continuous probability distribution alpha in [0, 1]^N using Softmax.
      * Fully differentiable end-to-end via standard backpropagation and gradient descent.
      * Every position receives a fractional weight, allowing the model to attend to multiple relevant context words simultaneously.
    """)

with col_att2:
    st.markdown("""
    ### Historical Evolution of Attention
    """)
    df_hist_att = pd.DataFrame({
        "Milestone": ["Bahdanau et al. (2014)", "Luong et al. (2015)", "Vaswani et al. (2017)", "Modern LLMs (2024+)"],
        "Architecture": ["Seq2Seq RNN", "Seq2Seq RNN", "Pure Transformer", "Decoder Foundation LLMs"],
        "Alignment Function": ["Additive MLP: v^T tanh(W_q q + W_k k)", "Multiplicative Dot: q^T W k", "Scaled Dot-Product: (Q K^T) / sqrt(d_k)", "FlashAttention-3 + GQA / MLA"],
        "Operation Mode": ["Decoder attends to Encoder", "Local & Global attention variants", "Self-Attention (Tokens attend to tokens)", "Sliding Window, KV Caching, Latent Heads"]
    })
    st.dataframe(df_hist_att, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 3: 2. WHY ATTENTION IS NEEDED
# ==============================================================================
st.header("3. Why Attention is Needed: Overcoming the Fixed-Vector Bottleneck", divider="rainbow")

st.markdown("""
Prior to attention, classical Seq2Seq encoder-decoder models (Sutskever et al., 2014; Cho et al., 2014) 
compressed an entire input sentence—whether 5 words or 100 words long—into a **single, fixed-dimensional hidden state vector** h_{final}.
""")

col_why1, col_why2 = st.columns([1, 1])

with col_why1:
    st.markdown("""
    ### The Information Bottleneck Pathology
    * **Symptom**: Model performance on machine translation plummeted dramatically as sentence length exceeded 15-20 words.
    * **Root Cause**: Compressing 50 complex semantic propositions into a single 512-dimensional vector creates catastrophic loss of detail, word order, and factual precision.
    * **Vanishing Gradients**: Error signals backpropagating from the decoder to early encoder words had to pass through dozens of recurrent time steps, decaying exponentially.

    ### How Attention Solves the Bottleneck
    Instead of discarding intermediate encoder hidden states, attention **preserves all token vectors** h_1, h_2, ..., h_N. 
    When generating target word t, the decoder inspects all source vectors directly, establishing an O(1) direct highway between source and target!
    """)

with col_why2:
    st.markdown("""
    ### Translation BLEU Score vs. Sentence Length
    """)
    df_bleu = pd.DataFrame({
        "Sentence Length (Words)": ["10 Words", "20 Words", "30 Words", "40 Words", "50+ Words"],
        "Classic Seq2Seq (Fixed Vector)": ["31.2 BLEU", "26.4 BLEU", "17.1 BLEU", "9.4 BLEU", "3.2 BLEU (Collapsed)"],
        "Attention-Enabled Seq2Seq": ["32.1 BLEU", "31.8 BLEU", "31.4 BLEU", "30.9 BLEU", "30.2 BLEU (Stable)"]
    })
    st.dataframe(df_bleu, use_container_width=True, hide_index=True)
    st.caption("Attention preserves performance across long sequences by providing direct access to all historical tokens.")

# ==============================================================================
# SECTION 4: 3. THE QUERY
# ==============================================================================
st.header("4. The Query (Q): Active Information Seeking", divider="rainbow")

st.markdown("""
The **Query (Q)** represents the active perspective of a token as it searches across the context sequence. 
It answers the question: *"Given my current identity and role, what information from other tokens do I need to update my representation?"*
""")

q_col1, q_col2 = st.columns([1, 1])

with q_col1:
    st.markdown("""
    ### Mathematical Formulation
    Given a token hidden state x_i in R^{d_{model}}:
    ```
    q_i = x_i . W_Q    # Shape: [1, d_k]
    ```
    Where W_Q in R^{d_{model} * d_k} is a learned projection matrix.
    
    ### Linguistic Probing Example
    In the sentence: `"The animal didn't cross the street because it was too tired."`
    * Token at position i: `"it"`
    * **Query vector of `"it"`**: Projects into a subspace specifically seeking singular nouns and animate agents in prior context.
    * When matched with the Key of `"animal"` (animate, singular noun), it computes high similarity, linking the pronoun to its antecedent!
    """)

with q_col2:
    st.markdown("""
    ### Role of Query in Decoder Generation
    * During autoregressive inference, the **Query vector** is generated exclusively by the newly emitted token at step t:
      `q_t = h_t . W_Q`.
    * This single Query vector probes historical Keys stored in the KV Cache, retrieving relevant background facts to predict step t+1.
    """)

# ==============================================================================
# SECTION 5: 4. THE KEY
# ==============================================================================
st.header("5. The Key (K): Indexing Content and Address Advertising", divider="rainbow")

st.markdown("""
The **Key (K)** acts as an index or tag associated with a stored memory vector. 
It answers the question: *"What features, syntactic roles, and semantic attributes do I offer to any querying token?"*
""")

k_col1, k_col2 = st.columns([1, 1])

with k_col1:
    st.markdown("""
    ### Mathematical Formulation
    Given a token hidden state x_j in R^{d_{model}}:
    ```
    k_j = x_j . W_K    # Shape: [1, d_k]
    ```
    Where W_K in R^{d_{model} * d_k} is a learned projection matrix.
    
    ### Invariance with Query Dimensions
    Notice that the **Query dimension (d_k)** and the **Key dimension (d_k)** must be strictly identical:
    * Their dot product q_i * k_j^T = Sum_{m=1}^{d_k} (q_{i,m} * k_{j,m}) requires identical vector lengths.
    * However, the Value dimension (d_v) is completely independent of d_k!
    """)

with k_col2:
    st.markdown("""
    ### The KV Cache: Storing Keys in Memory
    Because historical tokens do not change in an autoregressive generation loop:
    * Their Key vectors k_1, k_2, ..., k_{t-1} are calculated once and stored in the **KV Cache**.
    * At step t, the new Query q_t computes dot products against all cached Keys in a single matrix multiplication:
      `Scores = q_t @ K_cached.T    # Shape: [1, t]`
    """)

# ==============================================================================
# SECTION 6: 5. THE VALUE
# ==============================================================================
st.header("6. The Value (V): Raw Information Payload Delivery", divider="rainbow")

st.markdown("""
The **Value (V)** contains the actual information payload that will be extracted and blended 
into the querying token's updated representation.
""")

v_col1, v_col2 = st.columns([1, 1])

with v_col1:
    st.markdown("""
    ### Mathematical Formulation
    Given a token hidden state x_j in R^{d_{model}}:
    ```
    v_j = x_j . W_V    # Shape: [1, d_v]
    ```
    Where W_V in R^{d_{model} * d_v} is a learned projection matrix (typically d_v = d_k = d_{model} / H).
    
    ### Decoupling Keys and Values
    Why separate Key and Value into two distinct matrices? Why not simply use K = V?
    * **Matching vs. Payload**: What a token needs to match against (e.g., grammatical gender, position, subjecthood) is often orthogonal to what information it provides (e.g., concrete semantic attributes).
    * Decoupling allows the model to optimize W_K for indexing and address matching, while optimizing W_V for feature representation.
    """)

with v_col2:
    st.markdown("""
    ### Summary: The Q, K, V Triad
    """)
    df_qkv = pd.DataFrame({
        "Component": ["Query (Q)", "Key (K)", "Value (V)"],
        "Analogy": ["Search Query in Search Bar", "Webpage Title, Tags, Meta Description", "Full Webpage Content & Text"],
        "Projection": ["x . W_q", "x . W_k", "x . W_v"],
        "Dimension": ["d_k (e.g., 128)", "d_k (must match Q)", "d_v (typically equals d_k)"],
        "Role in Formula": ["Row vector of dot product", "Column vector of dot product", "Vector multiplied by Softmax weight"]
    })
    st.dataframe(df_qkv, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 7: 6. ATTENTION SCORES
# ==============================================================================
st.header("7. Attention Scores: Pairwise Alignment Energy", divider="rainbow")

st.markdown("""
The **Attention Score** (S_{ij}) measures the unnormalized degree of affinity or alignment between 
Query q_i and Key k_j.
""")

score_col1, score_col2 = st.columns([1, 1])

with score_col1:
    st.markdown("""
    ### Dot-Product Alignment Formula
    ```
    S_{ij} = (q_i . k_j^T) / sqrt(d_k)
    ```
    In matrix form across all sequence positions simultaneously:
    ```
    S = (Q @ K^T) / sqrt(d_k)    # Shape: [Seq_Len, Seq_Len]
    ```
    
    ### Comparison of Alignment Functions:
    1. **Dot Product (Luong 2015)**: S_{ij} = q_i * k_j. Fast, but sensitive to dimension growth.
    2. **Scaled Dot Product (Vaswani 2017)**: S_{ij} = \frac{q_i * k_j}{sqrt(d_k)}. Standard in modern Transformers.
    3. **Additive / MLP (Bahdanau 2014)**: S_{ij} = v_a^T \tanh(W_q q_i + W_k k_j). Slower to compute, discarded in modern LLMs.
    """)

with score_col2:
    st.markdown("""
    ### The Attention Score Matrix Topology (N * N)
    For sequence length N = 4:
    ```
    Scores Matrix S:
    [ [ S_00,  S_01,  S_02,  S_03 ],
      [ S_10,  S_11,  S_12,  S_13 ],
      [ S_20,  S_21,  S_22,  S_23 ],
      [ S_30,  S_31,  S_32,  S_33 ] ]
    ```
    Each entry S_{ij} represents how much token i is attracted to token j.
    """)

# ==============================================================================
# SECTION 8: 7. ATTENTION WEIGHTS
# ==============================================================================
st.header("8. Attention Weights: Probability Normalization via Softmax", divider="rainbow")

st.markdown("""
Raw attention scores are unconstrained real numbers. To turn them into meaningful blending coefficients, 
we apply the **Softmax function** across each row of the score matrix.
""")

w_col1, w_col2 = st.columns([1, 1])

with w_col1:
    st.markdown("""
    ### The Softmax Normalization Equation
    For row i of the attention score matrix:
    ```
    alpha_{ij} = exp(S_{ij}) / Sum_{m=1}^{N} exp(S_{im})
    ```
    
    ### Mathematical Guarantees:
    1. **Non-negativity**: alpha_{ij} >= 0 for all pairs.
    2. **Conservation of Attention**: Sum_{j=1}^{N} alpha_{ij} = 1.0 for each token i.
    3. **Competition**: If one Key scores exceptionally high, the Softmax denominator suppresses the weights of all other competing Keys.
    """)

with w_col2:
    st.markdown("""
    ### Incorporating Attention Masks
    Before applying Softmax, attention masks alter the raw score matrix:
    * **Padding Mask**: For padded tokens, set S_{ij} = -10^9. Then exp(-10^9) approx 0.0, ensuring padding receives exactly zero attention.
    * **Causal Mask**: For future tokens (j > i), set S_{ij} = -10^9, enforcing autoregressive causal visibility.
    """)

# ==============================================================================
# SECTION 9: 8. WEIGHTED SUM
# ==============================================================================
st.header("9. Weighted Sum: Context Vector Synthesis", divider="rainbow")

st.markdown("""
The final step computes the **Weighted Sum** of the Value vectors using the normalized attention weights, 
producing the final context-aware output representation.
""")

ws_col1, ws_col2 = st.columns([1, 1])

with ws_col1:
    st.markdown("""
    ### Vector Synthesis Formula
    For token i:
    ```
    Context_i = Sum_{j=1}^{N} alpha_{ij} * v_j
    ```
    In full matrix notation:
    ```
    Output = Alpha @ V    # Shape: [Seq_Len, d_v]
    ```
    
    ### Physical Interpretation
    * If alpha_{i, 3} = 0.90 and all other alpha approx 0.03, the updated representation of token i will be dominated by the features of token 3 (0.90 * v_3).
    * If alpha_{ij} = 1/N for all j, the output is an unweighted average of all Values (resembling a bag-of-words).
    * Softmax allows the network to smoothly interpolate anywhere between pure selective routing and broad semantic averaging.
    """)

with ws_col2:
    st.markdown("""
    ### Complete Scaled Dot-Product Attention Equation
    Combining all components into the single iconic Vaswani formula:
    ```
    Attention(Q, K, V) = Softmax((Q . K^T) / sqrt(d_k)) . V
    ```
    * Input Dimensions: Q in R^{N * d_k}, K in R^{N * d_k}, V in R^{N * d_v}.
    * Output Dimension: \text{Output} in R^{N * d_v}.
    """)

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Pure Python Scaled Dot-Product Attention Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. Step-by-step logging of **Queries, Keys, Values, Raw Scores, Scaled Scores, and Softmax Weights**.
2. **Causal and Padding Masking**.
3. Output **Weighted Sum Context Synthesis**.
""")

class ScaledDotProductAttentionEngine:
    """
    Production-grade educational implementation of Scaled Dot-Product Attention:
    Attention(Q, K, V) = Softmax((Q @ K^T) / sqrt(d_k) + Mask) @ V
    """
    def __init__(self, d_model: int = 16, d_k: int = 8, d_v: int = 8, seed: int = 42):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_k = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_v = rng.randn(d_model, d_v).astype(np.float32) * 0.1

    def forward(self, X: np.ndarray, is_causal: bool = False, pad_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes step-by-step attention forward pass.
        """
        seq_len = X.shape[0]
        
        # 1. Project into Q, K, V
        Q = np.dot(X, self.W_q) # [seq_len, d_k]
        K = np.dot(X, self.W_k) # [seq_len, d_k]
        V = np.dot(X, self.W_v) # [seq_len, d_v]
        
        # 2. Raw Dot-Product Scores: Q @ K^T
        raw_scores = np.dot(Q, K.T) # [seq_len, seq_len]
        
        # 3. Scaling by 1 / sqrt(d_k)
        scaled_scores = raw_scores / np.sqrt(self.d_k)
        
        # 4. Masking
        masked_scores = scaled_scores.copy()
        if is_causal:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            masked_scores += causal_mask
            
        if pad_mask is not None:
            pad_additive = np.where(pad_mask == 0, -1e9, 0.0)
            masked_scores += pad_additive[np.newaxis, :]
            
        # 5. Softmax Normalization with Logit Shifting
        safe_scores = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(safe_scores)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Weighted Sum of Values: Weights @ V
        context_out = np.dot(weights, V)
        
        intermediates = {
            "Q": Q,
            "K": K,
            "V": V,
            "raw_scores": raw_scores,
            "scaled_scores": scaled_scores,
            "masked_scores": masked_scores
        }
        return context_out, weights, intermediates

st.code('''
class ScaledDotProductAttentionEngine:
    def __init__(self, d_model=16, d_k=8, d_v=8, seed=42):
        # Q, K, V Projection Matrices
        self.W_q, self.W_k, self.W_v = ...
        
    def forward(self, X, is_causal=False, pad_mask=None):
        # 1. Q, K, V = X @ W_q, X @ W_k, X @ W_v
        # 2. Raw Scores = Q @ K.T
        # 3. Scale by 1 / sqrt(d_k)
        # 4. Mask future tokens (if causal) or pad tokens
        # 5. Softmax weights = exp(scaled_scores) / sum(...)
        # 6. Output context = weights @ V
        return context_out, weights, intermediates
''', language="python")


# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Live Attention Weight Heatmap Simulator", divider="rainbow")

st.markdown("""
Interact with a live scaled dot-product attention engine. 
Observe how altering the **Scaling Factor (sqrt(d_k))** and toggling **Causal Masking** alters attention concentration across sequence tokens.
""")

sample_tokens = ["The", "animal", "didn't", "cross", "because", "it", "was", "tired"]
seq_len = len(sample_tokens)

wb_view = st.radio("Select Interactive Workbench View", ["Attention Weight Heatmap", "Step-by-Step Numerical Walkthrough"], horizontal=True)

if wb_view == "Attention Weight Heatmap":
    st.markdown("#### Configure Attention Parameters")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_causal = st.checkbox("Enable Causal Attention Mask (Decoder Mode)", value=True)
    with col_c2:
        custom_scaling = st.slider("Scale Factor Denominator (Default = sqrt(d_k) = 2.83)", min_value=0.5, max_value=10.0, value=2.83, step=0.1)

    # Synthetic semantic affinity matrix
    np.random.seed(42)
    d_k = 8
    X_sample = np.random.randn(seq_len, 16).astype(np.float32)
    # Inject strong semantic link between "it" (index 5) and "animal" (index 1)
    X_sample[5] = X_sample[1] * 0.9 + np.random.randn(16)*0.1

    engine = ScaledDotProductAttentionEngine(d_model=16, d_k=8, d_v=8, seed=42)
    out, weights, inter = engine.forward(X_sample, is_causal=use_causal)

    # Recompute with user custom scaling
    raw_s = inter["raw_scores"] / custom_scaling
    if use_causal:
        raw_s += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    safe_s = raw_s - np.max(raw_s, axis=-1, keepdims=True)
    exp_s = np.exp(safe_s)
    custom_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    df_heatmap = pd.DataFrame(
        custom_weights,
        columns=sample_tokens,
        index=[f"Token {i}: '{t}'" for i, t in enumerate(sample_tokens)]
    )
    st.dataframe(df_heatmap.style.background_gradient(cmap="Blues", vmin=0.0, vmax=1.0), use_container_width=True)
    
    st.caption("Rows represent Queries (tokens seeking context); Columns represent Keys (attended tokens). Each row sums to 1.0.")

else:
    st.markdown("#### Numerical Inspection for Token 5: `'it'`")
    use_causal = True
    custom_scaling = 2.83
    np.random.seed(42)
    d_k = 8
    X_sample = np.random.randn(seq_len, 16).astype(np.float32)
    X_sample[5] = X_sample[1] * 0.9 + np.random.randn(16)*0.1
    engine = ScaledDotProductAttentionEngine(d_model=16, d_k=8, d_v=8, seed=42)
    out, weights, inter = engine.forward(X_sample, is_causal=use_causal)
    raw_s = inter["raw_scores"] / custom_scaling
    raw_s += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    safe_s = raw_s - np.max(raw_s, axis=-1, keepdims=True)
    exp_s = np.exp(safe_s)
    custom_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    row_idx = 5
    st.markdown(f"**Query Token**: `{sample_tokens[row_idx]}`")
    
    breakdown_data = []
    for col_j in range(seq_len):
        breakdown_data.append({
            "Key Token": sample_tokens[col_j],
            "Raw Dot-Product Q . K^T": f"{inter['raw_scores'][row_idx, col_j]:.2f}",
            "Scaled Score / sqrt(d_k)": f"{inter['scaled_scores'][row_idx, col_j]:.2f}",
            "Causal Mask Applied": "Masked (-inf)" if use_causal and col_j > row_idx else "Visible",
            "Final Attention Weight (Alpha)": f"{custom_weights[row_idx, col_j]:.4f}"
        })
    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "What is the fundamental difference between Additive (Bahdanau) Attention and Multiplicative (Scaled Dot-Product) Attention?",
        "a": "Additive attention uses a single-hidden-layer feedforward network: Score(q, k) = v_a^T tanh(W_q q + W_k k). Multiplicative attention computes the dot product: Score(q, k) = (q . k^T) / sqrt(d_k). While both have similar theoretical expressive power, scaled dot-product attention is substantially faster and more space-efficient in practice because it is implemented via highly optimized General Matrix Multiplication (GEMM) routines on GPU tensor cores."
    },
    {
        "q": "Why does Scaled Dot-Product Attention scale the inner product by 1 / sqrt(d_k)?",
        "a": "Assuming query and key components are independent random variables with zero mean and variance 1, their dot product q . k = sum_{i=1}^{d_k} q_i k_i has mean 0 and variance d_k. For large d_k (e.g., 128), dot products reach extreme magnitudes (+20 to +40). Pushing large inputs into Softmax drives probabilities toward 1 or 0, causing gradients to vanish (dSoftmax/dx approaches 0). Dividing by sqrt(d_k) rescales variance to 1.0, preserving responsive gradient flow."
    },
    {
        "q": "Can the Value dimension d_v differ from the Key dimension d_k in Scaled Dot-Product Attention?",
        "a": "Yes. The Query dimension and Key dimension must be identical because their dot product Q @ K^T requires matching inner dimensions. However, the Value dimension d_v is completely decoupled: the operation Softmax((Q @ K^T) / sqrt(d_k)) produces an [N, N] weight matrix, which can multiply any Value matrix V of shape [N, d_v]. While standard Transformers set d_k = d_v = d_model / H for symmetry, this is not a mathematical requirement."
    },
    {
        "q": "Why is Self-Attention permutation-invariant without positional encodings?",
        "a": "Self-Attention treats the sequence as an unordered set of vectors. The attention weight between token i and token j depends solely on the dot product q_i . k_j^T, which is completely independent of where tokens i and j sit in the sequence. Permuting the rows of input matrix X simply permutes the rows and columns of the output matrix identically (permutation equivariance), with zero awareness of sequential order."
    },
    {
        "q": "What is the computational complexity of computing Scaled Dot-Product Attention for sequence length N?",
        "a": "Computing Q @ K^T requires multiplying an [N, d_k] matrix by a [d_k, N] matrix, taking O(N^2 * d_k) operations. Computing Softmax takes O(N^2) operations. Multiplying the [N, N] weights by [N, d_v] Values takes O(N^2 * d_v) operations. Thus, overall computational complexity and memory consumption scale quadratically: O(N^2 * d)."
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
    **Trap 1 (Fatal): Applying Softmax Along the Wrong Tensor Axis**
    * *Failure*: Executing `softmax(scores, dim=-2)` (normalizing across Queries) instead of `dim=-1` (normalizing across Keys).
    * *Fix*: Each Query must distribute 100% of its attention probability across all available Keys. Always specify `Softmax(Scores, dim=-1)`.
    """)
    st.error("""
    **Trap 2 (Fatal): Causal Masking with Large Finite Negatives in Mixed Precision**
    * *Failure*: Masking future positions with `-1e9` inside FP16 tensors, causing immediate underflow to `-inf` or generating `NaN` during subsequent matrix operations.
    * *Fix*: Use `torch.finfo(scores.dtype).min` (approx `-65504` in FP16) to ensure numerical safety.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Omission of the 1 / sqrt(d_k) Scaling Factor**
    * *Failure*: Omitting the scaling factor in custom attention heads where d_k >= 64.
    * *Fix*: Loss will plateau early with near-zero gradients. Always divide dot products by sqrt(d_k).
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Materializing the N x N Attention Matrix for Long Contexts**
    * *Failure*: Computing raw `Q @ K.T` in PyTorch for sequence lengths >16,384, causing out-of-memory errors due to the quadratic memory footprint of the attention matrix.
    * *Fix*: Use **FlashAttention**, which tiles Q, K, and V in GPU SRAM and computes attention without ever materializing the N * N matrix in high-bandwidth memory.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Q, K, V Triad**\n\nQuery: What I seek. Key: What I advertise. Value: What information I deliver if matched.")
with fc_col2:
    st.info("**Scale Factor sqrt(d_k)**\n\nPrevents dot product variance from growing with dimension, keeping Softmax in its responsive gradient zone.")
with fc_col3:
    st.info("**Attention Conservation**\n\nSoftmax ensures attention weights along each Query row sum to exactly 1.0 (probabilistic convex combination).")
with fc_col4:
    st.info("**Quadratic Bottleneck**\n\nPairwise all-to-all matching scales compute and memory as O(N^2), driving modern efficient attention engineering.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of attention fundamentals, the Q/K/V triad, and scaled dot-product mechanics:
""")

c1 = st.checkbox("I can articulate the database retrieval analogy connecting Queries, Keys, and Values.")
c2 = st.checkbox("I understand why Seq2Seq models required attention to eliminate the fixed-vector information bottleneck.")
c3 = st.checkbox("I can explain why Query dimension d_k and Key dimension d_k must match, while Value dimension d_v can differ.")
c4 = st.checkbox("I can mathematically derive why dividing by sqrt(d_k) stabilizes the variance of attention logits.")
c5 = st.checkbox("I understand why attention weights must sum to 1.0 along the Key axis (dim=-1).")
c6 = st.checkbox("I can explain why standard self-attention is computationally quadratic O(N^2) in sequence length.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered attention basics, the Q/K/V paradigm, scaled dot-product mechanics, and context vector synthesis!")
