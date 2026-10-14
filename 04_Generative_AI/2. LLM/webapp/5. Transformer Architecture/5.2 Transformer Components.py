import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.2 Transformer Components (Embeddings, Positional, MHA, FFN, Residual, RMSNorm, Unembedding)",
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
st.markdown('<div class="main-title">5.2 Transformer Components: Deep Mechanical Anatomy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Input Embeddings and Rotary Encodings to Multi-Head Attention, SwiGLU MLPs, RMSNorm, and Output Projection — The Eight Core Subsystems of Modern LLM Blocks.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Modern Pre-LN / RMSNorm Transformer Block", divider="rainbow")

st.markdown("""
A modern foundation LLM (such as Llama 3 or Mistral) is built by stacking 32 to 128 identical **Transformer Blocks**. 
Each block consists of two primary operational sublayers—**Multi-Head Attention (MHA/GQA)** and a **Feed-Forward Network (FFN/SwiGLU)**—connected 
by **Residual Additions** and stabilized by **Root Mean Square Normalization (RMSNorm)**.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("IN", "Input Hidden State: x_l\nShape: [Batch, Seq_Len, d_model]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    
    # Sublayer 1: Attention
    arch_graph.node("NORM1", "RMSNorm Sublayer 1\nx_norm1 = RMSNorm(x_l)", shape="ellipse", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("ROPE", "Rotary Positional Encoding (RoPE)\nRotate Q and K by position angles", shape="box", fillcolor="#1E293B", color="#A855F7")
    arch_graph.node("ATTN", "Grouped-Query Attention (GQA)\nSoftmax((Q . K^T) / sqrt(d_k) + Mask) . V", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("ADD1", "Residual Highway Addition (+)\nx_mid = x_l + Attention_Out", shape="ellipse", fillcolor="#312E81", color="#818CF8")

    # Sublayer 2: FFN
    arch_graph.node("NORM2", "RMSNorm Sublayer 2\nx_norm2 = RMSNorm(x_mid)", shape="ellipse", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("FFN", "SwiGLU MLP Sublayer\n(SiLU(x . W_gate) * (x . W_up)) . W_down", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("ADD2", "Residual Highway Addition (+)\nx_{l+1} = x_mid + FFN_Out", shape="ellipse", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("OUT", "Output Hidden State: x_{l+1}\nShape: [Batch, Seq_Len, d_model]", shape="box", fillcolor="#1E293B", color="#38BDF8")

    # Edges
    arch_graph.edge("IN", "NORM1")
    arch_graph.edge("IN", "ADD1", label="Identity Residual Path", color="#38BDF8", style="dashed")
    arch_graph.edge("NORM1", "ROPE")
    arch_graph.edge("ROPE", "ATTN")
    arch_graph.edge("ATTN", "ADD1")
    
    arch_graph.edge("ADD1", "NORM2")
    arch_graph.edge("ADD1", "ADD2", label="Identity Residual Path", color="#38BDF8", style="dashed")
    arch_graph.edge("NORM2", "FFN")
    arch_graph.edge("FFN", "ADD2")
    arch_graph.edge("ADD2", "OUT")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### The Division of Labor Between Subsystems
    * **Attention Subsystem (Dynamic Routing)**:
      Responsible for **token-to-token communication**. It allows tokens to move information across time steps, query preceding context, and resolve grammatical and semantic dependencies.
    * **Feed-Forward Subsystem (Factual Storage)**:
      Responsible for **position-wise computation and knowledge retrieval**. Research (Geva et al., 2021) shows that MLP layers act as associative key-value memories storing factual knowledge and domain concepts.
    * **Residual Highways & Normalization (Signal Preservation)**:
      Ensure that gradients can backpropagate through 100+ layers without vanishing or exploding, while RMSNorm stabilizes activations across high-dimensional feature dimensions.
    """)

# ==============================================================================
# SECTION 2: 1. INPUT EMBEDDINGS
# ==============================================================================
st.header("2. Input Embeddings: Discrete Tokens to Latent Manifolds", divider="rainbow")

st.markdown("""
Natural language enters a Transformer as discrete integer token IDs (x in {0, 1, ..., V-1}) 
produced by a subword tokenizer (Byte-Pair Encoding or SentencePiece). 
The **Input Embedding Layer** projects these discrete IDs into a continuous d_{model}-dimensional vector space.
""")

emb_col1, emb_col2 = st.columns([1, 1])

with emb_col1:
    st.markdown("""
    ### Linear Algebraic Formulation
    The embedding layer is a parameter matrix W_E in R^{V * d_{model}}:
    * For vocabulary size V = 128,256 (Llama 3) and d_{model} = 4096:
      `Total Embedding Parameters = 128,256 * 4096 = 525,336,576 (~525M parameters)`.
    * Retrieving an embedding for token ID i is mathematically equivalent to multiplying a one-hot vector e_i in {0, 1}^V by W_E:
      ```
      v_i = e_i . W_E = W_E[i, :]
      ```
    * In GPU memory, this is executed as an O(1) memory lookup table (`nn.Embedding(V, d_model)`).
    """)

with emb_col2:
    st.markdown("""
    ### Weight Tying (Embedding vs. LM Head)
    * **Weight Tying (Press & Wolf, 2017)**: Reuses the input embedding matrix W_E as the final output unembedding matrix (W_U = W_E^T). Used in GPT-2, T5, and Gemma. Saves hundreds of millions of parameters.
    * **Untied Weights (Llama 3, Mistral)**: Keeps W_E and W_U completely independent. Allows input representations to optimize for comprehension while output representations optimize for token emission competition.
    """)

# ==============================================================================
# SECTION 3: 2. POSITIONAL ENCODING
# ==============================================================================
st.header("3. Positional Encoding: Breaking Permutation Invariance", divider="rainbow")

st.markdown("""
Self-attention is fundamentally **permutation invariant**: because it computes inner products between all token pairs, 
`"dog bites man"` and `"man bites dog"` produce identical attention scores if position information is omitted!
""")

pos_col1, pos_col2 = st.columns([1, 1])

with pos_col1:
    st.markdown("""
    ### The Three Eras of Position Information:
    1. **Sinusoidal Absolute Encoding (Vaswani 2017)**:
       Adds fixed, deterministic trigonometric waves to input embeddings:
       ```
       PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
       ```
    2. **Learned Absolute Position Embeddings (BERT, GPT-2)**:
       Learns a trainable parameter matrix W_{pos} in R^{L_{max} * d_{model}}. Cannot extrapolate past maximum context length L_{max}.
    3. **Rotary Position Embedding - RoPE (Su et al., 2021 - Llama 3, Mistral)**:
       Applies a 2D rotation matrix to Query and Key vectors in the complex plane at each layer:
       ```
       R_Theta(pos) . v
       ```
       Encodes **relative distance** (pos_i - pos_j) directly into the dot product Q_i * K_j^T!
    """)

with pos_col2:
    st.markdown("""
    ### Comparison of Positional Mechanisms
    """)
    df_pos = pd.DataFrame({
        "Mechanism": ["Sinusoidal (Vaswani)", "Learned Absolute (GPT-2)", "ALiBi (Press et al.)", "RoPE (Llama / Mistral)"],
        "Injection Point": ["Input layer only (+)", "Input layer only (+)", "Attention logit bias (-m * dist)", "Applied to Q & K per layer"],
        "Relative Awareness": ["Weak (implicit linear combos)", "None (pure absolute)", "Strong (linear slope)", "Exact (rotary inner product)"],
        "Length Extrapolation": ["Poor past training limit", "Impossible (hard limit L_max)", "High (decays smoothly)", "Exceptional (via RoPE scaling: YaRN, NTK)"],
        "Industry Standard": ["Historical 2017 baseline", "Legacy (2018-2020)", "Specialized", "Modern Dominant Standard"]
    })
    st.dataframe(df_pos, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 4: 3. SELF-ATTENTION
# ==============================================================================
st.header("4. Self-Attention: Scaled Dot-Product Mechanics", divider="rainbow")

st.markdown("""
**Scaled Dot-Product Self-Attention** is the mathematical core of the Transformer, enabling tokens 
to dynamically route information to one another based on content compatibility.
""")

attn_col1, attn_col2 = st.columns([1, 1])

with attn_col1:
    st.markdown("""
    ### The Fundamental Formula
    Given input sequence representations X in R^{N * d_{model}}:
    1. Project into Queries, Keys, and Values:
       ```
       Q = X W_Q,   K = X W_K,   V = X W_V
       ```
    2. Compute pairwise similarity scores:
       ```
       Scores = (Q . K^T) / sqrt(d_k)
       ```
    3. Normalize to probabilities and weight Values:
       ```
       Attention(Q, K, V) = Softmax(Scores) . V
       ```

    ### Why Scale by 1 / sqrt(d_k)?
    When vector dimension d_k is large (e.g., d_k = 128), the dot product of two independent zero-mean, unit-variance vectors has mean 0 and variance d_k:
    * Without scaling, dot products grow large in magnitude (> 30).
    * Softmax enters regions with extremely small gradients (dSoftmax / dx approx 0).
    * Gradients vanish during backpropagation, halting learning!
    * Dividing by sqrt(d_k) rescales the variance back to 1.0, keeping Softmax in its responsive gradient zone.
    """)

with attn_col2:
    st.markdown("""
    ### Query, Key, and Value Metaphor
    * **Query (Q)**: *"What kind of information am I looking for?"* (e.g., a pronoun asking for its antecedent noun).
    * **Key (K)**: *"What kind of information do I contain?"* (e.g., a noun advertising its gender and number).
    * **Value (V)**: *"What actual semantic content do I provide if selected?"*
    """)
    st.info("When Query matches Key via high dot product, Softmax assigns high attention weight, transferring the Value vector directly into the updated representation.")

# ==============================================================================
# SECTION 5: 4. MULTI-HEAD ATTENTION & GQA
# ==============================================================================
st.header("5. Multi-Head Attention: MHA, MQA, and Grouped-Query Attention (GQA)", divider="rainbow")

st.markdown("""
A single attention head can only focus on one type of relationship at a time. 
**Multi-Head Attention (MHA)** splits the hidden dimension across multiple independent projection heads, 
allowing the model to simultaneously track syntactic agreement, coreference, sentiment, and semantic facts.
""")

mha_col1, mha_col2 = st.columns([1, 1])

with mha_col1:
    st.markdown("""
    ### The Memory Bottleneck: Multi-Query Attention (MQA) and GQA
    During autoregressive LLM inference, the **KV Cache** consumes massive GPU VRAM:
    `KV_Memory = 2 * Batch * Seq_Len * Num_Layers * Num_KV_Heads * d_k * Bytes`.

    To slash KV cache memory consumption:
    1. **Multi-Head Attention (MHA - GPT-3)**: H Query heads, H Key heads, H Value heads. Maximum memory.
    2. **Multi-Query Attention (MQA - Shazeer 2019)**: H Query heads, but only **1 shared Key head and 1 Value head**. Reduces KV cache by Hx, but causes minor quality degradation.
    3. **Grouped-Query Attention (GQA - Ainslie et al. 2023 - Llama 3, Mistral)**: Partitions Query heads into G groups (e.g., 32 Q heads, 8 KV heads -> 4 Q heads share 1 KV head).
       * Slashes KV cache memory by **4x to 8x**.
       * Retains 99.5%+ of full MHA model quality!
    """)

with mha_col2:
    st.markdown("""
    ### Attention Architectures Compared
    """)
    df_mha = pd.DataFrame({
        "Attention Type": ["Multi-Head Attention (MHA)", "Grouped-Query Attention (GQA)", "Multi-Query Attention (MQA)"],
        "Query Heads": ["32", "32", "32"],
        "Key / Value Heads": ["32 (1:1 ratio)", "8 (4:1 sharing ratio)", "1 (32:1 sharing ratio)"],
        "KV Cache Footprint": ["100% (Baseline)", "25% (4x Memory Reduction)", "3.1% (32x Memory Reduction)"],
        "Generation Throughput": ["Standard", "3x - 5x higher batching capacity", "Maximum batching capacity"],
        "Representative Model": ["GPT-3, Original Transformer", "Llama 3, Mistral 7B, Gemma 2", "Falcon 40B, StarCoder"]
    })
    st.dataframe(df_mha, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 6: 5. FEED-FORWARD NETWORK (FFN / SWIGLU)
# ==============================================================================
st.header("6. Feed-Forward Network: Non-Linear Expansion & SwiGLU MLPs", divider="rainbow")

st.markdown("""
While Self-Attention routes information between tokens, the **Feed-Forward Network (FFN)** performs 
intensive position-wise non-linear transformations on individual token representations.
""")

ffn_col1, ffn_col2 = st.columns([1, 1])

with ffn_col1:
    st.markdown("""
    ### Classical FFN vs. Modern SwiGLU
    * **Original Transformer FFN (2017)**:
      Two linear transformations with intermediate ReLU / GELU activation:
      ```
      FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
      ```
      Dimension expanded to d_{ff} = 4 * d_{model}.
      
    * **Modern SwiGLU FFN (Shazeer 2020 - Llama 3, Mistral)**:
      Combines **Swish (SiLU)** activation with **Gated Linear Units (GLU)**:
      ```
      SwiGLU(x) = (SiLU(x W_{gate}) * (x W_{up})) W_{down}
      ```
      Where:
      * W_{gate} in R^{d_{model} * d_{ff}}
      * W_{up} in R^{d_{model} * d_{ff}}
      * W_{down} in R^{d_{ff} * d_{model}}
      * To preserve equal FLOPs with 3 weight matrices, d_{ff} is typically set to \frac{8}{3} d_{model} \approx 2.67 * d_{model} (e.g., 14,336 for Llama 3 8B).
    """)

with ffn_col2:
    st.markdown("""
    ### Why SwiGLU Dominates
    1. **Bilinear Gating**: The element-wise product `SiLU(Gate) * Up` allows the network to dynamically gate feature channels, suppressing noise and amplifying salient factual signals.
    2. **Smoother Gradient Flow**: SiLU (x * sigmoid(x)) possesses continuous non-zero gradients everywhere, eliminating the "dead neuron" pathology of ReLU.
    3. **Parametric Knowledge Storage**: The FFN accounts for approximately **two-thirds of all Transformer parameters** (~66%), functioning as the long-term associative memory of the LLM.
    """)

# ==============================================================================
# SECTION 7: 6. RESIDUAL CONNECTIONS
# ==============================================================================
st.header("7. Residual Connections: Preserving Identity Highways", divider="rainbow")

st.markdown("""
Introduced by Kaiming He et al. in ResNet (2015), **Residual Skip Connections (`Add`)** wrap around every 
Transformer sublayer: `x_{out} = x + Sublayer(x)`.
""")

res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.markdown("""
    ### Mathematical Gradient Flow Highway
    Consider an L-layer Pre-LN Transformer:
    ```
    x_{l+1} = x_l + F_l(LN(x_l))
    ```
    Expanding this recursively reveals that the final state x_L is an uncorrupted sum:
    ```
    x_L = x_0 + Sum_{l=0}^{L-1} F_l(LN(x_l))
    ```
    When computing gradients via the chain rule with respect to input x_0:
    ```
    dLoss / dx_0 = (dLoss / dx_L) * (I + Sum_{l=0}^{L-1} dF_l / dx_0)
    ```
    The presence of the **Identity Matrix I** guarantees that gradients can flow directly from the final loss back to the very first embedding layer without decaying!
    """)

with res_col2:
    st.markdown("""
    ### What Happens Without Residual Connections?
    * At layer depth L >= 12, repeated multiplications by sublayer weight matrices cause signal dissipation or exponential blowup.
    * Gradients vanish: dLoss / dx_0 -> 0, rendering early layers incapable of learning.
    * Training loss stalls and remains flat from step 0.
    """)
    st.info("The identity residual highway is what enabled deep learning to scale from 8 layers to over 100 layers in modern foundation models.")

# ==============================================================================
# SECTION 8: 7. LAYER NORMALIZATION: PRE-LN VS RMSNORM
# ==============================================================================
st.header("8. Layer Normalization: Standard LayerNorm vs. RMSNorm", divider="rainbow")

st.markdown("""
**Layer Normalization** standardizes activation values across the hidden dimension d_{model} 
for each token independently, stabilizing the variance of internal activations.
""")

norm_col1, norm_col2 = st.columns([1, 1])

with norm_col1:
    st.markdown("""
    ### Standard LayerNorm (Ba et al., 2016)
    Computes both mean and variance across the feature dimension:
    ```
    mu = (1 / d) * Sum_{i=1}^{d} x_i
    sigma^2 = (1 / d) * Sum_{i=1}^{d} (x_i - mu)^2
    LN(x) = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta
    ```
    Requires two statistical passes over activations: one for the mean, one for the variance.
    
    ### RMSNorm (Root Mean Square Normalization - Zhang & Sennrich, 2019)
    Hypothesizes that the success of LayerNorm stems from **scaling variance**, not mean-centering:
    ```
    RMS(x) = sqrt((1 / d) * Sum_{i=1}^{d} x_i^2 + eps)
    RMSNorm(x) = (x / RMS(x)) * gamma
    ```
    * Eliminates mean calculation and learnable bias \beta.
    * Reduces memory access overhead and delivers **10% to 50% faster kernel execution on GPUs**!
    * Adopted universally by Llama 2/3, Mistral, Gemma, and DeepSeek.
    """)

with norm_col2:
    st.markdown("""
    ### Normalization Evolution
    """)
    df_norm = pd.DataFrame({
        "Attribute": ["Standard LayerNorm (Post-LN)", "Standard LayerNorm (Pre-LN)", "RMSNorm (Modern Standard)"],
        "Placement": ["After residual addition", "Before attention and FFN", "Before attention and FFN"],
        "Mean Centering": ["Yes (Subtracts mu)", "Yes (Subtracts mu)", "No (Root Mean Square only)"],
        "Learnable Parameters": ["gamma (scale) + beta (shift)", "gamma (scale) + beta (shift)", "gamma (scale) only"],
        "GPU Speedup": ["Baseline", "Baseline", "+10-30% faster execution"],
        "Models": ["Original Vaswani, GPT-2", "GPT-3, T5, BERT", "Llama 3, Mistral, Qwen 2.5"]
    })
    st.dataframe(df_norm, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 9: 8. OUTPUT PROJECTION & UNEMBEDDING
# ==============================================================================
st.header("9. Output Projection & Unembedding: Logits to Vocabulary Probabilities", divider="rainbow")

st.markdown("""
After passing through L Transformer layers, the final hidden state h_L in R^{d_{model}} 
must be projected back into the vocabulary space to sample the next predicted token.
""")

out_col1, out_col2 = st.columns([1, 1])

with out_col1:
    st.markdown("""
    ### The Unembedding Mathematical Pipeline
    1. **Final Layer Normalization**:
       `h_{final} = RMSNorm(h_L)`
    2. **Linear Output Projection (LM Head)**:
       Multiplies by the unembedding matrix W_U in R^{d_{model} * V}:
       ```
       Logits = h_{final} . W_U + b_U    # Shape: [1, V]
       ```
    3. **Softmax with Temperature Scaling**:
       Converts unnormalized logits z into a categorical probability distribution:
       ```
       P(token_i) = exp(z_i / T) / Sum_{j=1}^{V} exp(z_j / T)
       ```
       Where T is the **Sampling Temperature**:
       * T \to 0: Greedy argmax (deterministic, highest probability token).
       * T = 0.7: Balanced creativity and factual coherence.
       * T > 1.5: High entropy, random and erratic text.
    """)

with out_col2:
    st.markdown("""
    ### Sampling Filters: Top-K and Top-P (Nucleus)
    To prevent low-probability nonsensical tokens from being sampled:
    * **Top-K Sampling**: Truncates candidate pool to the K highest-probability tokens (e.g., K=50).
    * **Top-P (Nucleus) Sampling**: Selects the smallest set of tokens whose cumulative probability exceeds threshold P (e.g., P=0.90). Dynamically expands for ambiguous contexts and contracts for strict syntax.
    * **Min-P Sampling (Modern Alternative)**: Filters out all tokens whose probability is less than a fraction of the top token's probability: P(w) < P_{top} * min_p (e.g., min_p = 0.05).
    """)

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Pure Python Modern Transformer Block (RoPE, GQA, SwiGLU, RMSNorm)", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. **RMSNorm** implementation.
2. **Rotary Position Embedding (RoPE)** vector rotation.
3. **Grouped-Query Attention (GQA)**.
4. **SwiGLU Feed-Forward Network**.
5. **Output Projection (LM Head)** with temperature-scaled Softmax sampling.
""")

st.code('''
import numpy as np
from typing import Optional, Tuple

class ModernTransformerBlock:
    """
    Production-grade educational implementation of a modern Llama 3-style block:
    1. RMSNorm (Root Mean Square Layer Normalization).
    2. RoPE (Rotary Position Embedding).
    3. Grouped-Query Attention (GQA).
    4. SwiGLU MLP Feed-Forward Sublayer.
    5. Output LM Head with Temperature Sampling.
    """
    def __init__(self, d_model: int = 64, num_q_heads: int = 4, num_kv_heads: int = 2, d_ff: int = 128, vocab_size: int = 1000, seed: int = 42):
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.heads_per_group = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        
        rng = np.random.RandomState(seed)
        # RMSNorm weights
        self.gamma1 = np.ones(d_model, dtype=np.float32)
        self.gamma2 = np.ones(d_model, dtype=np.float32)
        self.gamma_final = np.ones(d_model, dtype=np.float32)
        
        # GQA Attention weights
        self.W_q = rng.randn(d_model, num_q_heads * self.d_k).astype(np.float32) * 0.02
        self.W_k = rng.randn(d_model, num_kv_heads * self.d_k).astype(np.float32) * 0.02
        self.W_v = rng.randn(d_model, num_kv_heads * self.d_k).astype(np.float32) * 0.02
        self.W_o = rng.randn(num_q_heads * self.d_k, d_model).astype(np.float32) * 0.02
        
        # SwiGLU FFN weights
        self.W_gate = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.W_up = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.W_down = rng.randn(d_ff, d_model).astype(np.float32) * 0.02
        
        # LM Head
        self.W_unembed = rng.randn(d_model, vocab_size).astype(np.float32) * 0.02

    @staticmethod
    def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Root Mean Square Layer Normalization."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * gamma

    @staticmethod
    def apply_rope(x: np.ndarray) -> np.ndarray:
        """Applies Rotary Position Embedding (RoPE) rotation across adjacent dimension pairs."""
        # Simple simulated 2D rotation for demonstration
        x_rot = x.copy()
        x_rot[..., 0::2] = x[..., 0::2] * np.cos(0.5) - x[..., 1::2] * np.sin(0.5)
        x_rot[..., 1::2] = x[..., 0::2] * np.sin(0.5) + x[..., 1::2] * np.cos(0.5)
        return x_rot

    def grouped_query_attention(self, x: np.ndarray) -> np.ndarray:
        """Executes Grouped-Query Attention (GQA) where multiple Q heads share 1 KV head."""
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_q_heads, self.d_k).swapaxes(1, 2)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_kv_heads, self.d_k).swapaxes(1, 2)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_kv_heads, self.d_k).swapaxes(1, 2)
        
        # 2. Apply RoPE
        Q = self.apply_rope(Q)
        K = self.apply_rope(K)
        
        # 3. Repeat KV heads to match Q heads (GQA broadcast)
        K_rep = np.repeat(K, self.heads_per_group, axis=1)
        V_rep = np.repeat(V, self.heads_per_group, axis=1)
        
        # 4. Scaled Dot-Product with Causal Mask
        scores = np.matmul(Q, K_rep.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        scores = scores + mask
        
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        
        out = np.matmul(weights, V_rep).swapaxes(1, 2).reshape(batch_size, seq_len, -1)
        return np.dot(out, self.W_o)

    def swiglu_ffn(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU Feed-Forward Network: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down"""
        gate = np.dot(x, self.W_gate)
        silu_gate = gate * (1.0 / (1.0 + np.exp(-gate))) # x * sigmoid(x)
        up = np.dot(x, self.W_up)
        return np.dot(silu_gate * up, self.W_down)

    def forward_block(self, x: np.ndarray) -> np.ndarray:
        """Complete forward pass of modern Pre-LN / RMSNorm block."""
        # Sublayer 1: GQA with RMSNorm and Residual
        x = x + self.grouped_query_attention(self.rms_norm(x, self.gamma1))
        # Sublayer 2: SwiGLU with RMSNorm and Residual
        x = x + self.swiglu_ffn(self.rms_norm(x, self.gamma2))
        return x

    def generate_next_token(self, hidden_state: np.ndarray, temperature: float = 0.7) -> Tuple[int, np.ndarray]:
        """Projects final hidden state to logits and samples next token."""
        h_norm = self.rms_norm(hidden_state[:, -1, :], self.gamma_final) # Last token
        logits = np.dot(h_norm, self.W_unembed).squeeze()
        
        # Temperature scaling
        scaled_logits = logits / max(temperature, 1e-4)
        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_l / np.sum(exp_l)
        
        sampled_token = int(np.random.choice(self.vocab_size, p=probs))
        return sampled_token, probs
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Transformer Parameter & FLOP Distribution Calculator", divider="rainbow")

st.markdown("""
Calculate exact parameter counts, memory footprints, and architectural ratios 
across different configurations of **Embeddings, Attention, SwiGLU FFN, and Output Projections**.
""")

col_calc1, col_calc2 = st.columns([1, 1])

with col_calc1:
    st.markdown("#### Configure Architecture Hyperparameters")
    calc_d_model = st.selectbox("Hidden Dimension (d_model)", [2048, 4096, 8192], index=1)
    calc_layers = st.slider("Number of Layers", min_value=12, max_value=80, value=32, step=4)
    calc_vocab = st.selectbox("Vocabulary Size (V)", [32000, 64000, 128256], index=2)
    calc_attn_type = st.radio("Attention Topology", ["Multi-Head Attention (MHA)", "Grouped-Query Attention (GQA - 8 KV Heads)"])

    # Calculations
    q_heads = 32
    kv_heads = 32 if "MHA" in calc_attn_type else 8
    d_k = calc_d_model // q_heads
    d_ff = int(calc_d_model * (8.0 / 3.0)) # SwiGLU 8/3 expansion
    
    # Per-layer params
    # Attention: W_q [d, q_heads*d_k], W_k [d, kv_heads*d_k], W_v [d, kv_heads*d_k], W_o [q_heads*d_k, d]
    attn_params = (calc_d_model * q_heads * d_k) + (2 * calc_d_model * kv_heads * d_k) + (q_heads * d_k * calc_d_model)
    # SwiGLU: W_gate [d, d_ff], W_up [d, d_ff], W_down [d_ff, d]
    ffn_params = 3 * (calc_d_model * d_ff)
    # Norms per layer: 2 * d_model
    norm_params = 2 * calc_d_model
    
    layer_total = attn_params + ffn_params + norm_params
    total_layers_params = layer_total * calc_layers
    
    # Embeddings & LM Head
    embed_params = calc_vocab * calc_d_model
    lm_head_params = calc_vocab * calc_d_model # Untied
    
    grand_total = total_layers_params + embed_params + lm_head_params

with col_calc2:
    st.markdown("#### Parameter Allocation Breakdown")
    
    st.metric("Total Model Parameters", f"{grand_total / 1e9:.2f} Billion Parameters")
    
    df_budget = pd.DataFrame([
        {"Subsystem": "Input Embeddings", "Parameters": f"{embed_params / 1e6:.1f} M", "Percentage": f"{(embed_params / grand_total)*100:.1f}%"},
        {"Subsystem": f"Attention Sublayers ({calc_layers}x)", "Parameters": f"{(attn_params * calc_layers) / 1e6:.1f} M", "Percentage": f"{((attn_params * calc_layers) / grand_total)*100:.1f}%"},
        {"Subsystem": f"SwiGLU FFN Sublayers ({calc_layers}x)", "Parameters": f"{(ffn_params * calc_layers) / 1e6:.1f} M", "Percentage": f"{((ffn_params * calc_layers) / grand_total)*100:.1f}%"},
        {"Subsystem": "Output LM Head", "Parameters": f"{lm_head_params / 1e6:.1f} M", "Percentage": f"{(lm_head_params / grand_total)*100:.1f}%"}
    ])
    st.dataframe(df_budget, use_container_width=True, hide_index=True)
    
    st.info(f"Notice: The Feed-Forward Networks account for **{((ffn_params * calc_layers) / grand_total)*100:.1f}%** of total model capacity, serving as the primary factual memory repository.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does SwiGLU achieve superior downstream performance compared to traditional ReLU / GELU activations?",
        "a": "SwiGLU (Swish Gated Linear Unit) computes the element-wise product of a SiLU-activated linear projection and a second linear projection: (SiLU(x W_gate) * (x W_up)) W_down. This bilinear gating mechanism allows the network to dynamically scale or suppress individual feature dimensions based on context. Additionally, SiLU maintains smooth, non-zero gradients everywhere, eliminating the dead neuron problem seen in ReLU."
    },
    {
        "q": "What is the computational and memory advantage of RMSNorm over standard LayerNorm?",
        "a": "Standard LayerNorm enforces zero-mean and unit variance by computing two statistical reductions across the feature dimension: the mean mu and the variance sigma^2. Zhang & Sennrich (2019) demonstrated that the regularization effect comes solely from scaling the root mean square, not from shifting the mean. RMSNorm divides by the square root of mean square activations directly, eliminating the mean calculation, reducing memory read/write traffic, and yielding 10-30% faster GPU execution."
    },
    {
        "q": "How does Grouped-Query Attention (GQA) reduce KV cache memory without sacrificing model quality?",
        "a": "In standard Multi-Head Attention (MHA), every Query head has its own Key and Value head. During autoregressive generation, storing past Keys and Values in GPU VRAM creates severe memory bottlenecks. GQA groups Query heads (e.g., 32 Q heads) and shares a smaller number of KV heads (e.g., 8 KV heads). This slashes KV cache memory consumption by 4x, allowing 4x larger serving batch sizes or 4x longer context windows, while maintaining 99%+ of full MHA accuracy."
    },
    {
        "q": "Why are positional encodings injected by rotating Q and K vectors in RoPE rather than adding them to input embeddings?",
        "a": "Adding absolute positional encodings to input embeddings (x + PE) forces early layers to disentangle semantic vectors from positional vectors, and the positional signal degrades as activations pass through multiple deep non-linear layers. RoPE applies rotation matrices directly to Query and Key vectors inside each attention head at every layer. The dot product (R_Theta(i) Q) . (R_Theta(j) K)^T mathematically factors out absolute positions, encoding pure relative distance (i - j) precisely where attention weights are computed."
    },
    {
        "q": "Why is dividing by sqrt(d_k) mathematically essential in Scaled Dot-Product Attention?",
        "a": "For two independent random vectors q and k with zero mean and unit variance, their dot product sum(q_i * k_i) has mean 0 and variance d_k. For large dimensions (e.g., d_k = 128), raw dot products reach values of +30 to -30. When fed into Softmax, large inputs push probabilities to 1 or 0, saturating the function into regions where gradients are virtually zero. Dividing by sqrt(d_k) normalizes the variance to 1.0, ensuring Softmax remains in its steepest gradient region."
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
    **Trap 1 (Fatal): Forgetting to Rescale SwiGLU Inner Dimension (d_ff)**
    * *Failure*: Replacing a standard ReLU FFN (d_{ff} = 4 d_{model}) with SwiGLU without adjusting d_{ff}. Because SwiGLU requires 3 weight matrices instead of 2, this causes an unintended 50% parameter and compute inflation.
    * *Fix*: Always rescale SwiGLU's inner dimension to d_{ff} \approx \frac{8}{3} d_{model} to maintain strict parameter equivalence with 2-matrix MLPs.
    """)
    st.error("""
    **Trap 2 (Fatal): Precision Underflow in Softmax with Causal Masking**
    * *Failure*: Masking future positions with `-1e4` in FP16 precision. When subtracting the maximum value for numerical stability, values can underflow or produce NaN outputs.
    * *Fix*: In FP16, use `torch.finfo(torch.float16).min` or cast to FP32 before applying the attention softmax.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding RoPE Rotary Application on Key Cache Updates**
    * *Failure*: Caching raw unrotated Keys and applying RoPE dynamically during dot product computation.
    * *Fix*: Apply RoPE to Keys **before** inserting them into the KV cache, ensuring that cached tensors are ready for immediate matrix multiplication without per-step rotation overhead.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): O(N^2) Softmax Memory Allocation in Long Contexts**
    * *Failure*: Materializing the full [Batch, Heads, N, N] attention score matrix in GPU memory for context lengths >8192, triggering an immediate CUDA Out-Of-Memory error.
    * *Fix*: Always use **FlashAttention-2/3**, which tiles the attention computation in SRAM without materializing the N * N matrix in high-bandwidth memory (HBM).
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**RMSNorm Efficiency**\n\nEliminates mean subtraction from LayerNorm, computing root mean square only. Delivers 10-30% faster GPU execution.")
with fc_col2:
    st.info("**SwiGLU Gating**\n\n(SiLU(x W_gate) * (x W_up)) W_down. Bilinear gating provides smoother gradients and superior factual capacity.")
with fc_col3:
    st.info("**GQA Memory Savings**\n\nGroups Q heads to share a smaller number of KV heads (e.g., 32 Q heads to 8 KV heads), slashing KV cache memory by 4x.")
with fc_col4:
    st.info("**RoPE Rotary Relative**\n\nRotates Q and K vectors in the complex plane, injecting relative distance (i - j) directly into the dot product.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of the 8 core Transformer components and their modern foundation LLM variants:
""")

c1 = st.checkbox("I can explain why the embedding layer operates as an O(1) lookup table and understand weight tying.")
c2 = st.checkbox("I know why RoPE rotary encoding is superior to absolute sinusoidal position embeddings.")
c3 = st.checkbox("I can derive why dividing by sqrt(d_k) prevents vanishing gradients in Scaled Dot-Product Attention.")
c4 = st.checkbox("I understand the difference between MHA, MQA, and Grouped-Query Attention (GQA).")
c5 = st.checkbox("I can write out the SwiGLU formula and explain why its inner dimension is scaled to 8/3 d_model.")
c6 = st.checkbox("I know the mathematical difference between standard LayerNorm and RMSNorm.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered all Transformer architectural components, modern foundation LLM variants (RoPE, GQA, SwiGLU, RMSNorm), and parameter budget mechanics!")
