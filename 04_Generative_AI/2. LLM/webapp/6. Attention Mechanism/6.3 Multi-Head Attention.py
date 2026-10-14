import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.3 Multi-Head Attention (Multiple Heads, Projections, Parallelism, Concat, Output Projection)",
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
st.markdown('<div class="main-title">6.3 Multi-Head Attention: Subspace Specialization and Parallel Execution</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Single-Head Averaging Bottlenecks to Multi-Head Subspace Routing, Concatenation, and Output Projections — The Multi-Perspective Mechanics of Modern Transformers.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Multi-Head Subspace Routing Topology", divider="rainbow")

st.markdown("""
A single attention head can only construct one probability distribution over the sequence at any given step. 
If a token needs to attend to its grammatical subject *and* its governing preposition *and* a distant coreferent pronoun, 
a single head is forced to average these competing signals. 
**Multi-Head Attention (MHA)** solves this by projecting representations into H independent feature subspaces:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("IN", "Input Hidden States: X\nShape: [Batch, Seq_Len, d_model]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    
    with arch_graph.subgraph(name="cluster_heads") as c:
        c.attr(bgcolor="#0F172A", label="H Parallel Independent Attention Heads", fontcolor="#38BDF8", color="#38BDF8")
        c.node("H1", "Head 1\nAttention(Q_1, K_1, V_1)\nShape: [Batch, Seq_Len, d_k]\n(Local Syntax / Modifiers)", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("H2", "Head 2\nAttention(Q_2, K_2, V_2)\nShape: [Batch, Seq_Len, d_k]\n(Long-Range Coreference)", shape="box", fillcolor="#312E81", color="#818CF8")
        c.node("HDOTS", "Head 3 ... Head H-1\nParallel GPU Execution", shape="ellipse", fillcolor="#1E293B", color="#64748B")
        c.node("HH", "Head H\nAttention(Q_H, K_H, V_H)\nShape: [Batch, Seq_Len, d_k]\n(Induction & Semantic Facts)", shape="box", fillcolor="#312E81", color="#818CF8")

    arch_graph.node("CONCAT", "Concatenation Sublayer\nConcat(Head_1, Head_2, ..., Head_H)\nShape: [Batch, Seq_Len, H * d_v] = [Batch, Seq_Len, d_model]", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("WO", "Output Projection (W_O)\nLinear Mix across Subspaces\nShape: [Batch, Seq_Len, d_model]", shape="box", fillcolor="#042F2E", color="#10B981")

    # Edges
    arch_graph.edge("IN", "H1", label="W_Q1, W_K1, W_V1")
    arch_graph.edge("IN", "H2", label="W_Q2, W_K2, W_V2")
    arch_graph.edge("IN", "HDOTS")
    arch_graph.edge("IN", "HH", label="W_QH, W_KH, W_VH")
    arch_graph.edge("H1", "CONCAT")
    arch_graph.edge("H2", "CONCAT")
    arch_graph.edge("HDOTS", "CONCAT")
    arch_graph.edge("HH", "CONCAT")
    arch_graph.edge("CONCAT", "WO")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Fundamental Invariants of Multi-Head Attention
    * **Constant Computational Cost**:
      By setting d_k = d_v = d_{model} / H, the total compute FLOPs and parameter count of H parallel heads are **strictly identical** to single-head attention with dimension d_{model}.
    * **Subspace Disentanglement**:
      Different heads focus on completely orthogonal aspects of the language:
      * Head 1 might track subject-verb agreement (`"The dogs ... bark"`).
      * Head 2 might resolve pronoun anaphora (`"she"` -> `"Alice"`).
      * Head 3 might attend strictly to the immediate preceding token (i-1).
      * Head 4 might function as an induction head, copying repeated sequence patterns.
    """)

# ==============================================================================
# SECTION 2: 1. MULTIPLE ATTENTION HEADS
# ==============================================================================
st.header("2. Multiple Attention Heads: Overcoming Single-Focus Bottlenecks", divider="rainbow")

st.markdown("""
In a single attention head of dimension d_{model}, the model can only place high attention on one or two dominant tokens. 
If an ambiguous sentence requires simultaneously attending to syntactic governors and semantic arguments, 
a single head produces an averaged, washed-out vector.
""")

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("""
    ### The Division of Labor
    Instead of performing a single attention function with d_{model} = 4096, Multi-Head Attention divides the hidden space into H = 32 parallel heads:
    ```
    d_k = d_{model} / H = 4096 / 32 = 128
    ```
    * Each head operates on a compact 128-dimensional subspace.
    * Each head maintains its own independent projection matrices W_Q^i, W_K^i, W_V^i.
    * Head i can place 95% of its probability mass on the sentence subject, while Head j places 90% of its probability on the direct object!
    """)

with col_m2:
    st.markdown("""
    ### Head Dimension Configurations in Modern Models
    | Model Architecture | Hidden Dimension (d_{model}) | Attention Heads (H) | Head Dimension (d_k) |
    | :--- | :--- | :--- | :--- |
    | **BERT-Base** | 768 | 12 | 64 |
    | **BERT-Large** | 1024 | 16 | 64 |
    | **GPT-3 (175B)** | 12288 | 96 | 128 |
    | **Llama 3 (8B)** | 4096 | 32 (8 KV) | 128 |
    | **Llama 3 (70B)** | 8192 | 64 (8 KV) | 128 |
    """)

# ==============================================================================
# SECTION 3: 2. HEAD PROJECTIONS
# ==============================================================================
st.header("3. Head Projections: Independent Linear Mapping Subspaces", divider="rainbow")

st.markdown("""
For each head i in {1, ..., H}, the model applies dedicated linear projection matrices 
to map the common representation X in R^{N * d_{model}} into distinct query, key, and value subspaces.
""")

proj_col1, proj_col2 = st.columns([1, 1])

with proj_col1:
    st.markdown("""
    ### Individual Head Projections
    For head i:
    ```
    Q_i = X . W_Q^i    # Shape: [N, d_k],  where W_Q^i in R^[d_model, d_k]
    K_i = X . W_K^i    # Shape: [N, d_k],  where W_K^i in R^[d_model, d_k]
    V_i = X . W_V^i    # Shape: [N, d_v],  where W_V^i in R^[d_model, d_v]
    ```
    
    ### Unified Matrix Multiplication in Production
    In production deep learning frameworks (PyTorch, JAX), models do **not** run H separate matrix multiplications sequentially. 
    Instead, all head projections are fused into a single giant matrix multiplication:
    ```
    W_Q_all in R^[d_model, H * d_k] = R^[d_model, d_model]
    Q_all = X @ W_Q_all    # Single parallel GEMM! Shape: [B, N, d_model]
    ```
    Then, the output tensor is simply reshaped in memory into [B, H, N, d_k] with zero arithmetic overhead!
    """)

with proj_col2:
    st.markdown("""
    ### Tensor Reshaping Magic
    ```python
    # 1. Single big linear projection
    q = self.W_q(x) # [Batch, Seq_Len, d_model]
    
    # 2. Reshape to split heads
    q = q.view(batch, seq_len, num_heads, d_k)
    
    # 3. Transpose to place Heads before Seq_Len
    q = q.transpose(1, 2) # [Batch, num_heads, Seq_Len, d_k]
    ```
    This allows modern GPU Tensor Cores to process all batches and all heads concurrently as a 4D batch GEMM!
    """)

# ==============================================================================
# SECTION 4: 3. PARALLEL ATTENTION
# ==============================================================================
st.header("4. Parallel Attention: Concurrent Tensor Execution Across Heads", divider="rainbow")

st.markdown("""
Once Queries, Keys, and Values are reshaped into 4D tensors `[Batch, Heads, Seq_Len, d_k]`, 
the attention computation executes across all heads simultaneously in parallel.
""")

par_col1, par_col2 = st.columns([1, 1])

with par_col1:
    st.markdown("""
    ### 4D Batched Matrix Multiplication
    ```
    Scores = (Q @ K.transpose(-1, -2)) / sqrt(d_k)
    ```
    * Input Q: `[B, H, N, d_k]`
    * Input K^T: `[B, H, d_k, N]`
    * Output Scores: `[B, H, N, N]`
    
    Every head h independently computes its own N * N attention grid. 
    Each head is completely isolated from all other heads during this step, with zero inter-head data dependencies!
    """)

with par_col2:
    st.markdown("""
    ### Softmax Normalization per Head
    ```
    Weights = Softmax(Scores, dim=-1)    # Shape: [B, H, N, N]
    ```
    * Because Softmax is applied along `dim=-1`, every head independently normalizes its own rows to sum to 1.0.
    * Head 1 can be sharp and focused (entropy near 0).
    * Head 2 can be broad and diffuse (entropy high).
    * Head outputs:
      `Head_Out = Weights @ V             # Shape: [B, H, N, d_v]`
    """)

# ==============================================================================
# SECTION 5: 4. CONCATENATION
# ==============================================================================
st.header("5. Concatenation: Aggregating Multi-Subspace Representations", divider="rainbow")

st.markdown("""
After all H heads compute their respective context tensors `head_i in R^[N, d_v]`, 
their outputs must be combined back into a single unified sequence representation.
""")

con_col1, con_col2 = st.columns([1, 1])

with con_col1:
    st.markdown("""
    ### The Concatenation Operation
    All H head output vectors are glued together along the feature dimension:
    ```
    Concat_Out = [ head_1 ; head_2 ; ... ; head_H ]
    ```
    * Each head has dimension d_v = d_{model} / H.
    * Concatenating H heads produces:
      `Total_Dimension = H * d_v = H * (d_{model} / H) = d_{model}`.
    * In PyTorch memory layout:
      ```python
      # Transpose back: [B, H, N, d_v] -> [B, N, H, d_v]
      # Contiguous view: [B, N, H * d_v] = [B, N, d_model]
      concat_out = head_out.transpose(1, 2).contiguous().view(B, N, d_model)
      ```
    """)

with con_col2:
    st.markdown("""
    ### Why Concatenate Instead of Average?
    * **Averaging ((1/H) * Sum(head_i))**: Destroys subspace identity by collapsing distinct semantic channels together.
    * **Concatenation ([\text{head}_1; ...; \text{head}_H])**: Preserves the unique signals discovered by each head side-by-side, allowing the subsequent linear projection layer to learn which heads to weight and combine!
    """)

# ==============================================================================
# SECTION 6: 5. OUTPUT PROJECTION
# ==============================================================================
st.header("6. Output Projection (W_O): Linear Mixing Across Subspaces", divider="rainbow")

st.markdown("""
Concatenation simply places head outputs side-by-side in memory. 
The **Output Projection Matrix (W_O)** is the sublayer that allows heads to communicate and mix their discovered signals.
""")

wo_col1, wo_col2 = st.columns([1, 1])

with wo_col1:
    st.markdown("""
    ### Mathematical Formulation
    Given concatenated tensor H_{concat} in R^[B, N, d_model]:
    ```
    MultiHead(Q, K, V) = H_{concat} . W_O    # Shape: [B, N, d_model]
    ```
    Where W_O in R^[d_model, d_model] is a learned linear transformation matrix.
    
    ### Role of W_O:
    1. **Inter-Head Feature Synthesis**: Takes features from Head 1 (e.g., subject identity) and multiplies them against features from Head 4 (e.g., verb tense) to synthesize composite semantic facts.
    2. **Dimensional Alignment**: Ensures the output tensor has the exact shape [B, N, d_{model}] required to add directly into the incoming residual highway (X + \text{MultiHead}).
    """)

with wo_col2:
    st.markdown("""
    ### Complete Multi-Head Attention Equation
    Combining all projection and aggregation steps:
    ```
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) . W_O
    where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
    ```
    """)
    st.info("The output projection matrix W_O contains d_model * d_model parameters, exactly equal to the total parameters of all W_V projection matrices combined.")

# ==============================================================================
# SECTION 7: 6. WHY MULTIPLE HEADS HELP
# ==============================================================================
st.header("7. Why Multiple Heads Help: Empirical Role Specialization", divider="rainbow")

st.markdown("""
Extensive mechanistic interpretability research (Vig et al., Clark et al., Anthropic) reveals that 
individual attention heads in trained LLMs develop **highly specialized linguistic functions**.
""")

emp_col1, emp_col2 = st.columns([1, 1])

with emp_col1:
    st.markdown("""
    ### Discovered Attention Head Archetypes:
    1. **Positional / Offset Heads**:
       Attend almost exclusively to position t-1 (previous token) or position t+1 (next token), acting as local n-gram detectors.
    2. **Syntactic Dependency Heads**:
       Connect verbs to their direct objects, prepositions to noun phrases, or auxiliary verbs to main verbs.
    3. **Coreference Resolution Heads**:
       Connect pronouns (`"they"`, `"it"`, `"she"`) directly to their antecedent entity mentions earlier in the document.
    4. **Induction Heads (Olsson et al., Anthropic)**:
       Two-layer circuits that detect recurring patterns: if `[A][B]` occurred earlier in the prompt and `[A]` appears again, induction heads place high attention on `[B]`, enabling **in-context few-shot learning**!
    """)

with emp_col2:
    st.markdown("""
    ### Head Pruning & Redundancy
    * **The Pruning Paradox**: Michel et al. (2019) demonstrated that in models like BERT, up to **40-60% of attention heads can be completely pruned** at inference time with less than 1% drop in task performance!
    * **Why Over-parameterize?**: During pretraining, having many heads provides a rich exploratory search space. Different random seeds allow different heads to discover specialized circuits. Once trained, only a core subset of heads performs critical reasoning.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Pure Python Vectorized Multi-Head Attention Block", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. Complete **Fused Multi-Head Linear Projections**.
2. **4D Tensor Reshaping** (`[Batch, Heads, Seq_Len, d_k]`).
3. **Batched Parallel Attention Execution** with Causal Masking.
4. **Head Concatenation & Linear Output Projection (W_O)**.
""")

st.code('''
import numpy as np
from typing import Optional, Tuple

class MultiHeadAttentionModule:
    """
    Production-grade educational implementation of vectorized Multi-Head Attention (MHA):
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) @ W_O
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        rng = np.random.RandomState(seed)
        # Fused projection weights: [d_model, d_model]
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * 0.05

    def forward(self, X: np.ndarray, is_causal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes parallel multi-head attention over batched sequence tensor.
        
        Args:
            X: Input tensor of shape [batch_size, seq_len, d_model]
            is_causal: If True, applies lower-triangular causal lookahead mask
            
        Returns:
            out: Contextualized output tensor [batch_size, seq_len, d_model]
            attn_weights: 4D Attention map [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = X.shape
        
        # 1. Project to Q, K, V: [batch, seq_len, d_model]
        Q_fused = np.dot(X, self.W_q)
        K_fused = np.dot(X, self.W_k)
        V_fused = np.dot(X, self.W_v)
        
        # 2. Reshape to 4D tensors: [batch, num_heads, seq_len, d_k]
        Q = Q_fused.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = K_fused.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = V_fused.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        
        # 3. Scaled Dot-Product: [batch, num_heads, seq_len, seq_len]
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        
        # 4. Apply Causal Mask if required
        if is_causal:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask # Broadcasts across heads and batches
            
        # 5. Softmax Normalization along last axis (-1)
        safe_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(safe_scores)
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        
        # 6. Multiply by Values: [batch, num_heads, seq_len, d_k]
        head_outputs = np.matmul(attn_weights, V)
        
        # 7. Concatenate heads back into [batch, seq_len, d_model]
        concat_out = head_outputs.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # 8. Output Linear Projection: [batch, seq_len, d_model]
        final_out = np.dot(concat_out, self.W_o)
        
        return final_out, attn_weights
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Head Subspace Specialization Simulator", divider="rainbow")

st.markdown("""
Observe how **different attention heads specialize in distinct linguistic functions** 
when evaluating the same input sequence.
""")

demo_tokens = ["Alice", "told", "Bob", "she", "liked", "his", "book"]
N_tokens = len(demo_tokens)

# Setup 4 specialized synthetic head patterns
head_patterns = {
    "Head 1 (Previous-Token Offset Head)": np.zeros((N_tokens, N_tokens)),
    "Head 2 (Coreference Resolution Head)": np.zeros((N_tokens, N_tokens)),
    "Head 3 (Syntactic Verb-Object Head)": np.zeros((N_tokens, N_tokens)),
    "Head 4 (Global Subject Broadcast Head)": np.zeros((N_tokens, N_tokens))
}

# Head 1: attends to token i-1
for i in range(N_tokens):
    target = max(0, i - 1)
    head_patterns["Head 1 (Previous-Token Offset Head)"][i, target] = 0.85
    head_patterns["Head 1 (Previous-Token Offset Head)"][i, i] = 0.15

# Head 2: Coreference ("she" -> "Alice", "his" -> "Bob")
for i in range(N_tokens):
    head_patterns["Head 2 (Coreference Resolution Head)"][i, i] = 0.3
    if demo_tokens[i] == "she":
        head_patterns["Head 2 (Coreference Resolution Head)"][i, 0] = 0.9 # Alice
        head_patterns["Head 2 (Coreference Resolution Head)"][i, i] = 0.1
    elif demo_tokens[i] == "his":
        head_patterns["Head 2 (Coreference Resolution Head)"][i, 2] = 0.85 # Bob
        head_patterns["Head 2 (Coreference Resolution Head)"][i, i] = 0.15

# Head 3: Verbs attend to objects ("told" -> "Bob", "liked" -> "book")
for i in range(N_tokens):
    head_patterns["Head 3 (Syntactic Verb-Object Head)"][i, i] = 0.2
    if demo_tokens[i] == "told":
        head_patterns["Head 3 (Syntactic Verb-Object Head)"][i, 2] = 0.8 # Bob
    elif demo_tokens[i] == "liked":
        head_patterns["Head 3 (Syntactic Verb-Object Head)"][i, 6] = 0.8 # book

# Head 4: All tokens attend to "Alice" (Subject)
for i in range(N_tokens):
    head_patterns["Head 4 (Global Subject Broadcast Head)"][i, 0] = 0.75 # Alice
    head_patterns["Head 4 (Global Subject Broadcast Head)"][i, i] = 0.25

col_hw1, col_hw2 = st.columns([1, 1])

with col_hw1:
    selected_head = st.selectbox("Select Specialized Attention Head to Inspect", list(head_patterns.keys()))
    
    if "Head 1" in selected_head:
        st.info("Linguistic Function: Tracks local adjacent tokens (n-gram syntax).")
    elif "Head 2" in selected_head:
        st.info("Linguistic Function: Coreference resolution linking 'she' -> 'Alice' and 'his' -> 'Bob'.")
    elif "Head 3" in selected_head:
        st.info("Linguistic Function: Syntactic argument binding (verbs connecting to objects).")
    else:
        st.info("Linguistic Function: Broadcasts the global agent/subject across all sequence tokens.")

with col_hw2:
    st.markdown(f"#### Attention Heatmap for {selected_head.split('(')[0].strip()}")
    mat = head_patterns[selected_head]
    df_h = pd.DataFrame(mat, index=demo_tokens, columns=demo_tokens)
    st.dataframe(df_h.style.format("{:.2f}").background_gradient(cmap="Purples", vmin=0.0, vmax=1.0), use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does Multi-Head Attention divide the model dimension into d_k = d_model / H instead of giving each head a full d_model dimension?",
        "a": "Dividing d_model by H guarantees that the total parameter count and floating-point operations (FLOPs) remain strictly identical to single-head attention. Specifically, projecting to H heads of dimension d_k takes H * (d_model * d_k) = d_model * (H * d_k) = d_model^2 parameters. Computing attention across H heads takes H * (N^2 * d_k) = N^2 * d_model operations. This provides the multi-subspace representational power of multiple heads without increasing computation or memory."
    },
    {
        "q": "What happens if you remove the output projection matrix W_O from Multi-Head Attention?",
        "a": "Without W_O, the concatenated tensor [head_1; head_2; ...; head_H] simply places the outputs of independent heads side-by-side. The heads would have zero linear mechanism to mix, cross-attend, or synthesize information with one another before entering the residual connection and Feed-Forward Network. The output projection matrix W_O provides the necessary linear recombination across all subspaces."
    },
    {
        "q": "How does PyTorch execute Multi-Head Attention efficiently in hardware without looping over heads?",
        "a": "PyTorch fuses all H Query, Key, and Value projections into single linear layers of size [d_model, d_model]. After one batched matrix multiplication, the output tensor [Batch, Seq_Len, d_model] is reshaped via .view() into [Batch, Seq_Len, H, d_k] and transposed via .transpose(1, 2) into [Batch, H, Seq_Len, d_k]. Modern GPU tensor cores treat [Batch * H] as a unified batch dimension, executing all attention heads simultaneously in a single parallel GEMM kernel."
    },
    {
        "q": "What are 'Induction Heads', and why are they considered the primary driver of in-context learning in LLMs?",
        "a": "Induction Heads (identified by Anthropic) are two-head attention circuits formed across consecutive layers. The first head attends to the previous token [A]. The second head uses that information to search for previous occurrences of [A] earlier in the document and attend to the token [B] that followed it. This allows the model to complete repeating sequences ([A] -> predict [B]), serving as the foundational mechanism behind zero-shot and few-shot in-context learning."
    },
    {
        "q": "What is the difference between Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)?",
        "a": "In MHA, there is an equal number of Query heads and Key/Value heads (1:1 ratio, e.g., 32 Q heads, 32 KV heads). In GQA (used in Llama 3), Query heads are partitioned into groups that share a smaller number of Key/Value heads (e.g., 32 Q heads, 8 KV heads -> 4 Q heads share 1 KV head). GQA slashes the memory size of the inference KV Cache by 4x while maintaining over 99% of full MHA accuracy."
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
    **Trap 1 (Fatal): Memory Non-Contiguity after Head Transposition**
    * *Failure*: Calling `.view()` directly after `.transpose(1, 2)` in PyTorch without calling `.contiguous()`, triggering a runtime `RuntimeError: view size is not compatible with input tensor's size and stride`.
    * *Fix*: Always call `tensor.transpose(1, 2).contiguous().view(...)` to ensure memory strides are contiguous.
    """)
    st.error("""
    **Trap 2 (Fatal): Looping Over Heads with Python `for` Loops**
    * *Failure*: Implementing custom MHA by iterating through `for i in range(num_heads): head_out = attn(Q[i], K[i])`. This destroys GPU kernel concurrency and causes an 8x-15x slowdown.
    * *Fix*: Always vectorize across heads using 4D tensors `[Batch, Heads, Seq_Len, d_k]`.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Selecting d_model Not Divisible by num_heads**
    * *Failure*: Setting `d_model = 768` and `num_heads = 10`, causing fractional head dimensions (d_k = 76.8).
    * *Fix*: Always ensure d_{model} % H = 0, typically setting d_k to power-of-two values (64 or 128) for optimal GPU memory alignment.
    """)
    st.warning("""
    **Trap 4 (Memory Trap): Linear KV Cache Footprint Explosion**
    * *Failure*: Deploying standard MHA at 128k context lengths across hundreds of concurrent users, exhausting GPU VRAM.
    * *Fix*: Transition to Grouped-Query Attention (GQA) or Multi-Head Latent Attention (MLA).
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Head Dimension Invariant**\n\nd_k = d_model / H. Guarantees that MHA has identical parameter and FLOP complexity to single-head attention.")
with fc_col2:
    st.info("**Subspace Routing**\n\nIndependent heads specialize in syntax, coreference, and induction patterns, avoiding single-focus averaging.")
with fc_col3:
    st.info("**Output Matrix W_O**\n\nRecombines concatenated head features [N, d_model] into a unified feature representation.")
with fc_col4:
    st.info("**Induction Heads**\n\nTwo-layer attention circuits that detect recurring patterns, driving in-context few-shot learning in LLMs.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of Multi-Head Attention mechanics, parallel tensor execution, and head specialization:
""")

c1 = st.checkbox("I can explain why Multi-Head Attention has identical compute complexity to single-head attention.")
c2 = st.checkbox("I understand how PyTorch uses fused linear layers and 4D tensor reshaping to parallelize heads without loops.")
c3 = st.checkbox("I know why concatenation is preferred over averaging when combining head outputs.")
c4 = st.checkbox("I can explain the functional role of the output projection matrix W_O.")
c5 = st.checkbox("I understand empirical head specialization (syntactic heads, coreference heads, induction heads).")
c6 = st.checkbox("I can articulate the architectural difference between Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered Multi-Head Attention, subspace projections, parallel GPU tensor mechanics, and head specialization!")
