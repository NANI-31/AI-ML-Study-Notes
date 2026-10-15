import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.2 Self-Attention (Calculation, Q, K, V, QK^T, Scaling, Softmax, Output)",
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
st.markdown('<div class="main-title">6.2 Self-Attention: Full Matrix Algebra and Inner Mechanics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Sequence Projections and Gramian Dot Products to Scaling, Softmax Normalization, and Value Synthesis — The Complete Linear Algebraic Pipeline.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Unified Matrix Algebra of Self-Attention", divider="rainbow")

st.markdown("""
Unlike recurrent networks that process sequences sequentially or convolutional networks that constrain receptive fields 
to local filter kernels, **Self-Attention** computes all-to-all interactions across an entire sequence in a **single matrix multiplication pass**.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("X", "Input Matrix: X\nShape: [N, d_model]\nN tokens in parallel", shape="box", fillcolor="#1E293B", color="#38BDF8")
    
    with arch_graph.subgraph(name="cluster_mats") as c:
        c.attr(bgcolor="#0F172A", label="Projection Matrices (Learned Weights)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("Q", "Q = X . W_Q\n[N, d_k]", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("K", "K = X . W_K\n[N, d_k]", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("V", "V = X . W_V\n[N, d_v]", shape="box", fillcolor="#312E81", color="#818CF8")

    arch_graph.node("QKT", "Gramian Product: Q . K^T\nShape: [N, N]\nPairwise Dot-Product Matrix", shape="box", fillcolor="#1E293B", color="#F59E0B")
    arch_graph.node("SCALE", "Scaled Matrix: (Q . K^T) / sqrt(d_k)\nShape: [N, N]\nVariance stabilized to 1.0", shape="box", fillcolor="#1E293B", color="#F59E0B")
    arch_graph.node("SOFTMAX", "Softmax(Scores, dim=-1)\nShape: [N, N]\nNormalized Attention Weights Matrix A", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("OUT", "Output Matrix: A . V\nShape: [N, d_v]\nContextualized Token Representations", shape="box", fillcolor="#042F2E", color="#10B981")

    # Edges
    arch_graph.edge("X", "Q")
    arch_graph.edge("X", "K")
    arch_graph.edge("X", "V")
    arch_graph.edge("Q", "QKT")
    arch_graph.edge("K", "QKT")
    arch_graph.edge("QKT", "SCALE")
    arch_graph.edge("SCALE", "SOFTMAX")
    arch_graph.edge("SOFTMAX", "OUT")
    arch_graph.edge("V", "OUT")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Matrix Dimension Dimensions & Memory
    Let N be sequence length, d_{model} be model dimension, and d_k be head dimension:
    * X in R^[N, d_model]
    * W_Q, W_K in R^[d_model, d_k],   W_V in R^[d_model, d_v]
    * Q, K in R^[N, d_k],   V in R^[N, d_v]
    * Q K^T in R^[N, N] (The quadratic attention map)
    * A in R^[N, N] (Row-stochastic attention weights)
    * {Output} in R^[N, d_v] (Preserves token count N)
    """)
    st.info("Because all operations are pure matrix multiplications (GEMM), modern GPUs execute this entire pipeline at near-peak theoretical TFLOPs.")

# ==============================================================================
# SECTION 2: 1. SELF-ATTENTION CALCULATION
# ==============================================================================
st.header("2. Self-Attention Calculation: Mathematical Formulation", divider="rainbow")

st.markdown("""
**Self-Attention** (intra-attention) is an attention mechanism relating different positions of a **single sequence** 
in order to compute a representation of the same sequence.
""")

col_calc1, col_calc2 = st.columns([1, 1])

with col_calc1:
    st.markdown("""
    ### The Self-Attention Invariant
    In Cross-Attention, Queries come from one sequence (e.g., target language) while Keys and Values come from another (e.g., source language).
    
    In **Self-Attention**:
    ```
    Q = X W_Q,   K = X W_K,   V = X W_V
    ```
    All three matrices derive from the **exact same input representation X**!
    * Token 1 attends to Token 1, Token 2, ..., Token N.
    * Token N attends to Token 1, Token 2, ..., Token N.
    * The output at position i is a context-modulated blend of all tokens in the sequence, allowing pronouns to bind to nouns, adjectives to bind to subjects, and verbs to bind to objects.
    """)

with col_calc2:
    st.markdown("""
    ### Algorithmic Execution Steps
    1. **Linear Projection**: Multiply input matrix X by projection weights W_Q, W_K, W_V.
    2. **Gramian Similarity**: Compute Q K^T across all token pairs.
    3. **Dimensional Scaling**: Divide scores by sqrt(d_k).
    4. **Softmax Normalization**: Exponentiate and normalize row-wise.
    5. **Value Aggregation**: Multiply weights matrix A by Value matrix V.
    6. **Output Projection**: Multiply result by W_O to return to dimension d_{model}.
    """)

# ==============================================================================
# SECTION 3: 2. QUERY MATRIX
# ==============================================================================
st.header("3. Query Matrix (Q): Linear Projection of Input Tokens", divider="rainbow")

st.markdown("""
The **Query Matrix (Q)** packages the search intentions of every token in the sequence simultaneously.
""")

q_col1, q_col2 = st.columns([1, 1])

with q_col1:
    st.markdown("""
    ### Linear Algebraic Projection
    Given input X in R^[N, d_model] and weight matrix W_Q in R^[d_model, d_k]:
    ```
    Q = X . W_Q = [ q_1^T ]
                  [ q_2^T ]
                  [ ...   ]
                  [ q_N^T ]    # Shape: [N, d_k]
    ```
    * Row i of matrix Q (q_i in R^{d_k}) is the Query vector for token i.
    * It isolates the specific features token i is actively hunting for in other positions.
    """)

with q_col2:
    st.markdown("""
    ### Batch Tensor Dimensions in PyTorch
    In real-world LLM frameworks, inputs are batched:
    * Input Tensor: `x.shape = [Batch_Size, Seq_Len, d_model]`
    * Weight Matrix: `W_q.shape = [d_model, d_k]`
    * Projected Tensor: `Q.shape = [Batch_Size, Seq_Len, d_k]`
    * For Multi-Head Attention, this is reshaped to:
      `[Batch_Size, Num_Heads, Seq_Len, d_k]`.
    """)

# ==============================================================================
# SECTION 4: 3. KEY MATRIX
# ==============================================================================
st.header("4. Key Matrix (K): Indexing Subspace Geometry", divider="rainbow")

st.markdown("""
The **Key Matrix (K)** packages the addressable features and properties of every token in the sequence.
""")

k_col1, k_col2 = st.columns([1, 1])

with k_col1:
    st.markdown("""
    ### Linear Algebraic Projection
    Given input X in R^[N, d_model] and weight matrix W_K in R^[d_model, d_k]:
    ```
    K = X . W_K = [ k_1^T ]
                  [ k_2^T ]
                  [ ...   ]
                  [ k_N^T ]    # Shape: [N, d_k]
    ```
    * Row j of matrix K (k_j in R^{d_k}) represents the Key vector for token j.
    * Transposing matrix K (K^T in R^{d_k * N}) arranges the Key vectors as columns, setting up parallel inner products!
    """)

with k_col2:
    st.markdown("""
    ### Key Invariance with Queries
    * Notice that both W_Q and W_K project into the **exact same dimensionality d_k**.
    * If Q had dimension 64 and K had dimension 128, their inner product Q K^T would be mathematically undefined.
    * Both vectors must live in the same metric space for cosine and dot-product geometry to function.
    """)

# ==============================================================================
# SECTION 5: 4. VALUE MATRIX
# ==============================================================================
st.header("5. Value Matrix (V): Information Content Routing", divider="rainbow")

st.markdown("""
The **Value Matrix (V)** contains the actual information payload that will be routed 
into the updated token representations.
""")

v_col1, v_col2 = st.columns([1, 1])

with v_col1:
    st.markdown("""
    ### Linear Algebraic Projection
    Given input X in R^[N, d_model] and weight matrix W_V in R^[d_model, d_v]:
    ```
    V = X . W_V = [ v_1^T ]
                  [ v_2^T ]
                  [ ...   ]
                  [ v_N^T ]    # Shape: [N, d_v]
    ```
    * Row j of matrix V (v_j in R^{d_v}) represents the Value payload of token j.
    * While K determines *how much* attention token j receives, V determines *what information* token j actually passes along.
    """)

with v_col2:
    st.markdown("""
    ### Independence of Value Dimension (d_v)
    * In standard Transformer implementations, engineers set d_v = d_k = d_{model} / H for symmetry and convenience.
    * However, mathematically, d_v can be completely independent of d_k.
    * Multi-Head Latent Attention (DeepSeek-V2/V3) leverages low-rank compression specifically on Value and Key tensors to compress memory.
    """)

# ==============================================================================
# SECTION 6: 5. QK^T CALCULATION
# ==============================================================================
st.header("6. QK^T Calculation: Pairwise Gramian Dot-Product Matrix", divider="rainbow")

st.markdown("""
Multiplying the Query matrix by the transposed Key matrix (Q K^T) computes all N^2 pairwise inner products 
in a single parallel matrix multiplication.
""")

qkt_col1, qkt_col2 = st.columns([1, 1])

with qkt_col1:
    st.markdown("""
    ### The Gramian Matrix Product
    ```
    Scores = Q @ K^T    # Matrix multiplication: [N, d_k] @ [d_k, N] -> [N, N]
    ```
    Entry (i, j) in the resulting N * N matrix is the scalar dot product between Query i and Key j:
    ```
    Scores_{i, j} = q_i . k_j^T = Sum_{m=1}^{d_k} q_{i, m} * k_{j, m}
    ```
    
    ### The Quadratic Bottleneck: O(N^2)
    * For N = 1000: Matrix has 1,000,000 entries.
    * For N = 32000: Matrix has 1,024,000,000 entries (1.02 Billion floats = 2 GB in FP16 for 1 head!).
    * For 32 heads: 32 * 2{ GB} = 64{ GB} of VRAM just to store raw attention scores!
    * This quadratic scaling O(N^2) is why naive attention struggles past 32k context windows without FlashAttention.
    """)

with qkt_col2:
    st.markdown("""
    ### Geometric Interpretation of Q K^T
    * If token i and token j point in similar directions in projection space, q_i * k_j^T is a large positive number (> 15).
    * If orthogonal, q_i * k_j^T \approx 0.
    * If opposite, q_i * k_j^T is a large negative number (< -15).
    """)

# ==============================================================================
# SECTION 7: 6. SCALING FACTOR
# ==============================================================================
st.header("7. Scaling Factor: Stabilizing Variance on Tensor Dimensions", divider="rainbow")

st.markdown("""
The division by sqrt(d_k) is arguably the most critical mathematical stabilizer in the entire Transformer architecture.
""")

sc_col1, sc_col2 = st.columns([1, 1])

with sc_col1:
    st.markdown("""
    ### Mathematical Variance Derivation
    Assume components of q and k are independent random variables with mean 0 and variance 1:
    ```
    E[q_m] = 0,   Var(q_m) = 1
    E[k_m] = 0,   Var(k_m) = 1
    ```
    The product of two independent zero-mean variables has:
    ```
    E[q_m * k_m] = 0
    Var(q_m * k_m) = Var(q_m) * Var(k_m) = 1 * 1 = 1
    ```
    The full dot product sums d_k such terms:
    ```
    q . k = Sum_{m=1}^{d_k} (q_m * k_m)
    Var(q . k) = Sum_{m=1}^{d_k} Var(q_m * k_m) = d_k
    ```
    The standard deviation is sqrt(d_k)!
    * For d_k = 128: Standard deviation is sqrt{128} \approx 11.3.
    * Dot products naturally swing between -35 and +35.
    """)

with sc_col2:
    st.markdown("""
    ### Why Extreme Variance Destroys Softmax
    When logits enter Softmax with values like [+30, -10, -15]:
    * exp(30) completely dwarfs all other terms: Softmax outputs [1.0000, 0.0000, 0.0000].
    * The Softmax function **saturates into an extreme plateau**.
    * The derivative of saturated Softmax is virtually zero:
      ```
      dSoftmax_i / dx_j = Softmax_i * (delta_ij - Softmax_j) approx 0
      ```
    * Gradients vanish completely during backpropagation, halting all parameter learning!
    * Dividing by sqrt(d_k) rescales the variance back to 1.0, keeping Softmax in its responsive gradient zone.
    """)

# ==============================================================================
# SECTION 8: 7. SOFTMAX
# ==============================================================================
st.header("8. Softmax: Probability Normalization and Competition", divider="rainbow")

st.markdown("""
The **Softmax function** normalizes scaled dot-product scores across the Key dimension (axis -1), 
converting raw affinity energy into a valid row-stochastic attention weights matrix A.
""")

sm_col1, sm_col2 = st.columns([1, 1])

with sm_col1:
    st.markdown("""
    ### Row-Wise Probability Normalization
    For each row i of the scaled score matrix:
    ```
    A_{i, j} = exp(S_{i, j} / sqrt(d_k)) / Sum_{m=1}^{N} exp(S_{i, m} / sqrt(d_k))
    ```
    
    ### Properties of Attention Weight Matrix A:
    1. **Shape**: Exactly [N, N].
    2. **Bounds**: 0.0 <= A_{i, j} <= 1.0 for all i, j.
    3. **Row Sum Invariant**: Sum_{j=1}^{N} A_{i, j} = 1.0 for every token i.
    4. **Soft Selection**: Unlike hard argmax, Softmax provides a continuous, differentiable convex combination over all N tokens.
    """)

with sm_col2:
    st.markdown("""
    ### Numerical Stability in Softmax Execution
    In production code, never execute raw `exp(x)` directly. Always apply logit shifting:
    ```python
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    ```
    This prevents floating-point overflow to `inf` while leaving the output mathematically identical.
    """)

# ==============================================================================
# SECTION 9: 8. ATTENTION OUTPUT
# ==============================================================================
st.header("9. Attention Output: Value Blending and Contextualized Representations", divider="rainbow")

st.markdown("""
The final step multiplies the row-stochastic attention weight matrix A by the Value matrix V, 
producing the updated sequence representation.
""")

out_col1, out_col2 = st.columns([1, 1])

with out_col1:
    st.markdown("""
    ### Value Blending Matrix Product
    ```
    Output = A @ V    # Matrix multiplication: [N, N] @ [N, d_v] -> [N, d_v]
    ```
    Row i of the output matrix is the weighted sum of all Value vectors:
    ```
    Output_i = Sum_{j=1}^{N} A_{i, j} * v_j
    ```
    * Token i's new representation is constructed entirely from the Values of tokens it attended to, weighted by its attention weights A_{i, :}.
    * If token i attended 80% to token 2 and 20% to token 4, its output is 0.80 v_2 + 0.20 v_4.
    """)

with out_col2:
    st.markdown("""
    ### Linear Output Projection (W_O)
    In multi-head implementations, outputs from all H heads are concatenated along the feature dimension:
    `Concatenated = [Head_1, Head_2, ..., Head_H]  # Shape: [N, H * d_v] = [N, d_model]`
    And passed through a final linear projection matrix W_O in R^[d_model, d_model]:
    ```
    Final_Out = Concatenated @ W_O    # Shape: [N, d_model]
    ```
    This blends information learned across different attention heads back into a unified feature space.
    """)

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Pure Python Full Self-Attention Matrix Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. Complete **Q, K, V Projections**.
2. **Gramian QK^T Matrix Multiplication**.
3. **Square Root Scaling (1/sqrt(d_k))**.
4. **Numerically Stable Softmax Normalization**.
5. **Value Matrix Multiplication & Output Projection (W_O)**.
""")

class VectorizedSelfAttentionEngine:
    """
    Production-grade educational implementation of vectorized Self-Attention:
    Out = (Softmax((X @ W_Q @ (X @ W_K)^T) / sqrt(d_k)) @ (X @ W_V)) @ W_O
    """
    def __init__(self, d_model: int = 16, d_k: int = 8, d_v: int = 8, seed: int = 42):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_k = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_v = rng.randn(d_model, d_v).astype(np.float32) * 0.1
        self.W_o = rng.randn(d_v, d_model).astype(np.float32) * 0.1

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes complete vectorized self-attention pipeline.
        """
        N = X.shape[0]
        
        # 1. Projections: [N, d_model] @ [d_model, d_k] -> [N, d_k]
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v) # [N, d_v]
        
        # 2. QK^T Gramian Matrix: [N, d_k] @ [d_k, N] -> [N, N]
        qkt = np.dot(Q, K.T)
        
        # 3. Scaling by 1 / sqrt(d_k)
        scaled_scores = qkt / np.sqrt(self.d_k)
        
        # 4. Softmax with numerical logit shifting: [N, N]
        scores_max = np.max(scaled_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scaled_scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 5. Value Blending: [N, N] @ [N, d_v] -> [N, d_v]
        value_out = np.dot(weights, V)
        
        # 6. Linear Output Projection: [N, d_v] @ [d_v, d_model] -> [N, d_model]
        final_out = np.dot(value_out, self.W_o)
        
        intermediates = {
            "Q": Q,
            "K": K,
            "V": V,
            "qkt": qkt,
            "scaled_scores": scaled_scores,
            "value_out": value_out
        }
        return final_out, weights, intermediates

st.code('''
class VectorizedSelfAttentionEngine:
    def __init__(self, d_model=16, d_k=8, d_v=8, seed=42):
        self.W_q, self.W_k, self.W_v, self.W_o = ... # Initialized projections
        
    def forward(self, X):
        # 1. Linear projections: Q = X @ W_q, K = X @ W_k, V = X @ W_v
        # 2. Pairwise Gramian matrix: QK^T = Q @ K.T
        # 3. Scale by 1 / sqrt(d_k)
        # 4. Softmax weights = exp(scaled) / sum(...)
        # 5. Value blending: V_out = weights @ V
        # 6. Output projection: Final_Out = V_out @ W_o
        return final_out, weights, intermediates
''', language="python")


# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Live Step-by-Step Self-Attention Matrix Calculator", divider="rainbow")

st.markdown("""
Inspect how individual matrices evolve across each step of the **Self-Attention calculation** on a synthetic sequence.
""")

sample_words = ["Language", "Models", "Transform", "Context"]
seq_len = len(sample_words)

np.random.seed(42)
d_model = 8
d_k = 4
X_matrix = np.random.randn(seq_len, d_model).astype(np.float32)

engine = VectorizedSelfAttentionEngine(d_model=d_model, d_k=d_k, d_v=d_k, seed=42)
final_out, weights, inter = engine.forward(X_matrix)

view_mode = st.radio("Select Matrix Stage to Inspect", ["1. Raw Dot-Products (QK^T)", "2. Attention Weights (Softmax)", "3. Final Output Matrix"], horizontal=True)

if view_mode == "1. Raw Dot-Products (QK^T)":
    st.markdown("#### Gramian Product Matrix: Q K^T (N * N)")
    df_qkt = pd.DataFrame(inter["qkt"], columns=sample_words, index=sample_words)
    st.dataframe(df_qkt.style.format("{:.2f}").background_gradient(cmap="coolwarm"), use_container_width=True)
    st.caption("Unnormalized inner products showing pairwise raw affinities between sequence tokens.")
elif view_mode == "2. Attention Weights (Softmax)":
    st.markdown("#### Attention Weight Matrix: Softmax((Q K^T) / sqrt(d_k)) (N * N)")
    df_w = pd.DataFrame(weights, columns=sample_words, index=sample_words)
    st.dataframe(df_w.style.format("{:.4f}").background_gradient(cmap="Blues", vmin=0.0, vmax=1.0), use_container_width=True)
    st.caption("Each row represents one Query token. Notice that every row sums to exactly 1.0.")
else:
    st.markdown("#### Output Representation Matrix (N * d_{model})")
    df_out = pd.DataFrame(final_out, index=sample_words, columns=[f"Dim_{i}" for i in range(d_model)])
    st.dataframe(df_out.style.format("{:.2f}"), use_container_width=True)
    st.caption("Final contextualized vectors after Value blending and linear output projection W_O.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is Self-Attention an O(N^2) operation in sequence length, and which specific matrix multiplication drives this?",
        "a": "The quadratic bottleneck stems from the pairwise similarity calculation: Q @ K^T. Query matrix Q has shape [N, d_k] and Key matrix K^T has shape [d_k, N]. Multiplying them produces an [N, N] score matrix, requiring 2 * N^2 * d_k floating-point operations. Additionally, Softmax must be evaluated over all N^2 elements, and multiplying the [N, N] weights by [N, d_v] Values requires another 2 * N^2 * d_v FLOPs. Both compute and memory scale quadratically with sequence length N."
    },
    {
        "q": "How does FlashAttention optimize the O(N^2) memory footprint without altering mathematical outputs?",
        "a": "Standard PyTorch implementations materialize the intermediate [N, N] attention score matrix in high-bandwidth GPU memory (HBM). For N=16k, this takes gigabytes of RAM and saturates memory bandwidth. FlashAttention (Dao et al.) uses GPU SRAM tiling: it loads small blocks of Q, K, and V into fast on-chip SRAM, computes local attention, normalizes incrementally using online Softmax, and writes only the final [N, d_v] output to HBM. It never materializes the full N x N matrix in HBM, slashing memory from O(N^2) to O(N) and achieving 2x-4x speedups."
    },
    {
        "q": "Why does Self-Attention require an output projection matrix W_O at the end of the block?",
        "a": "In Multi-Head Attention, independent heads project into distinct subspaces (e.g., 32 heads of dimension 128). Their outputs are concatenated into a combined tensor of shape [N, 32 * 128] = [N, 4096]. The output projection matrix W_O in R^[4096 x 4096] linearly mixes and recombines information across all heads into a unified hidden representation before passing it to the residual addition."
    },
    {
        "q": "Under what circumstance does Self-Attention degenerate into a simple Average Pooling operation?",
        "a": "Self-Attention degenerates into average pooling when all Query-Key dot products are identical (q_i . k_j^T = c for all j). In that case, Softmax produces uniform weights: A_{i, j} = 1/N for all j. The output becomes the arithmetic mean of all Value vectors: Output_i = (1/N) * sum_j v_j, completely losing selective routing capability."
    },
    {
        "q": "What is the difference between permutation equivariance and permutation invariance in Self-Attention?",
        "a": "A function is permutation invariant if f(P X) = f(X) (output does not change if input is shuffled). Self-Attention is permutation equivariant: f(P X) = P f(X). If input rows are permuted by permutation matrix P, the output vectors are permuted by the exact same permutation matrix P. It becomes permutation invariant only if all token vectors are pooled (e.g., mean-pooled) into a single vector."
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
    **Trap 1 (Fatal): Transposing Along the Wrong Dimension for K**
    * *Failure*: Executing `Q @ K` instead of `Q @ K.T` or `K.swapaxes(-1, -2)` in PyTorch, triggering a tensor dimension mismatch error or broadcasting across incorrect axes.
    * *Fix*: Always ensure K is transposed along its last two dimensions (`K.transpose(-1, -2)`).
    """)
    st.error("""
    **Trap 2 (Fatal): Floating-Point Softmax Instability**
    * *Failure*: Calculating `torch.exp(scores) / torch.sum(...)` without subtracting the maximum logit, generating `NaN` values whenever scores exceed 88.0 in FP32.
    * *Fix*: Always use `torch.softmax(scores, dim=-1)` which incorporates logit shifting automatically.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Normalizing Softmax Over the Sequence Length Axis**
    * *Failure*: Applying `Softmax(dim=-2)` instead of `dim=-1`, causing each Key to distribute attention over Queries rather than each Query distributing attention over Keys.
    * *Fix*: Softmax must strictly be applied across keys: `dim=-1`.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Storing QK^T During Inference**
    * *Failure*: Storing the entire attention matrix during autoregressive generation when only the final generated token is required.
    * *Fix*: During single-token inference, query length is 1: only compute a `1 x N` vector dot product.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Self-Attention Formula**\n\nAttention(Q, K, V) = Softmax((Q K^T) / sqrt(d_k)) V. Fully vectorized, parallel all-to-all sequence routing.")
with fc_col2:
    st.info("**Gramian Product QK^T**\n\nMultiplies [N, d_k] by [d_k, N] to produce [N, N] pairwise interaction grid. Primary source of quadratic scaling.")
with fc_col3:
    st.info("**Row Stochastic**\n\nSoftmax across dim=-1 ensures each row sums to 1.0, turning dot products into convex blending weights.")
with fc_col4:
    st.info("**Output Projection W_O**\n\nBlends concatenated head outputs [N, d_model] into unified feature space before residual addition.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of the 8 steps of Self-Attention calculation:
""")

c1 = st.checkbox("I can explain why Q, K, and V all derive from the same input matrix X in Self-Attention.")
c2 = st.checkbox("I understand the tensor dimension transitions from [N, d_model] to [N, N] and back to [N, d_model].")
c3 = st.checkbox("I can derive why QK^T has quadratic computational complexity O(N^2 * d_k).")
c4 = st.checkbox("I know why dividing by sqrt(d_k) stabilizes the variance of attention logits to 1.0.")
c5 = st.checkbox("I understand why Softmax must be normalized along dim=-1 (across Keys).")
c6 = st.checkbox("I can describe how FlashAttention avoids materializing the N x N attention matrix in GPU HBM.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered self-attention calculation, full matrix algebra, scaling mechanics, and value synthesis!")
