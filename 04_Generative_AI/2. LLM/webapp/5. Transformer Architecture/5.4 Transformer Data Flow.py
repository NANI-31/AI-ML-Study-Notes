import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.4 Transformer Data Flow (Tokens -> Embeddings -> Attention -> Hidden -> Logits -> Probs -> Tokens)",
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
st.markdown('<div class="main-title">5.4 Transformer Data Flow: End-to-End Generative Tensor Trajectory</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Discrete Integer Token IDs to Latent Activations, Logits, Softmax Probabilities, and Emitted Tokens — The Mathematical Life Cycle of a Forward Pass.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The 6-Stage Generative Tensor Pipeline", divider="rainbow")

st.markdown("""
In an autoregressive foundation LLM (such as Llama 3 8B), generating a single new token requires 
routing data through an exact six-stage tensor transformation pipeline. 
Each stage transforms tensor shapes, data representations, and memory spaces:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    arch_graph.node("S1", "Stage 1: Tokens -> Embeddings\nInput IDs: [Batch, Seq_Len] (int64)\nLookup in W_E -> [Batch, Seq_Len, d_model] (float16/bfloat16)", shape="box", fillcolor="#1E293B", color="#38BDF8")
    arch_graph.node("S2", "Stage 2: Embeddings -> Attention\nProject to Q, K, V -> [Batch, Heads, Seq_Len, d_k]\nScaled Dot-Product + Causal Mask -> Values weighted", shape="box", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("S3", "Stage 3: Attention -> Hidden States\nResidual Highway (+) & SwiGLU MLP (x 32 layers)\nFinal Hidden States: [Batch, Seq_Len, d_model]", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("S4", "Stage 4: Hidden States -> Logits\nSlice Last Token: [Batch, d_model] -> RMSNorm\nUnembed: @ W_vocab.T -> [Batch, Vocab_Size]", shape="box", fillcolor="#1E293B", color="#F59E0B")
    arch_graph.node("S5", "Stage 5: Logits -> Probabilities\nTemperature Scaling: Logits / T\nSoftmax Normalization -> [Batch, Vocab_Size] (sum=1.0)", shape="box", fillcolor="#312E81", color="#A855F7")
    arch_graph.node("S6", "Stage 6: Probabilities -> Generated Tokens\nTop-P / Top-K / Min-P Filtering -> Random Categorical Sample\nEmits Token ID: int64 -> Appended to Input Context", shape="box", fillcolor="#042F2E", color="#10B981")

    arch_graph.edge("S1", "S2")
    arch_graph.edge("S2", "S3")
    arch_graph.edge("S3", "S4")
    arch_graph.edge("S4", "S5")
    arch_graph.edge("S5", "S6")
    arch_graph.edge("S6", "S1", label="Autoregressive Loop (Append ID)", color="#10B981", style="dashed")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Tensor Dimensions Across the Pipeline (Llama 3 8B Spec)
    * **Batch Size (B)**: Number of concurrent request streams (e.g., B = 4).
    * **Sequence Length (N)**: Total tokens in current context (e.g., N = 1024).
    * **Hidden Dimension (d_{model})**: Dimensionality of latent state space (d_{model} = 4096).
    * **Attention Heads (H_Q / H_{KV})**: 32 Query heads, 8 Key/Value heads (d_k = 128).
    * **Feed-Forward Dimension (d_{ff})**: Intermediate SwiGLU expansion (d_{ff} = 14336).
    * **Vocabulary Size (V)**: Discrete token inventory (V = 128256).
    """)
    st.info("During training, all N positions compute logits simultaneously in parallel. During inference, only the final position N is projected to logits, generating one token at a time.")

# ==============================================================================
# SECTION 2: 1. TOKENS -> EMBEDDINGS
# ==============================================================================
st.header("2. Tokens -> Embeddings: Discrete IDs to Latent Vectors", divider="rainbow")

st.markdown("""
Raw user text is initially split into a discrete sequence of integer IDs by a tokenizer (e.g., Tiktoken for GPT-4 or SentencePiece for Llama 3). 
The first step of the neural network maps these discrete integers into continuous coordinates.
""")

s1_col1, s1_col2 = st.columns([1, 1])

with s1_col1:
    st.markdown("""
    ### Step 1 Mechanics
    * **Input Tensor**: An integer matrix of token IDs:
      `input_ids in {0, 1, ..., V-1}^[B, N]`
    * **Embedding Lookup Table**: A trainable weight matrix W_E in R^[V, d_model].
    * **Execution**: For each position (b, t), row input_ids[b, t] of W_E is retrieved:
      ```
      X_0 = W_E[input_ids]    # Shape: [B, N, d_model]
      ```
    * **Memory Footprint**:
      For B=4, N=2048, d_{model}=4096 in FP16 (2 bytes/float):
      `Tensor Size = 4 * 2048 * 4096 * 2 bytes = 67.1 MB`.
    """)

with s1_col2:
    st.markdown("""
    ### Positional Preparation
    * In older models (BERT, GPT-2), absolute positional embeddings W_{pos} were added here: `X_0 = W_E + W_{pos}`.
    * In modern models (Llama 3, Mistral), input embeddings remain pure lexical representations. Rotary Position Embeddings (RoPE) are applied downstream inside each attention head rather than at the embedding layer.
    """)

# ==============================================================================
# SECTION 3: 2. EMBEDDINGS -> ATTENTION
# ==============================================================================
st.header("3. Embeddings -> Attention: Multi-Head Projections and Context Routing", divider="rainbow")

st.markdown("""
Once tokens are in latent continuous space, they pass into the first attention layer 
to exchange information and build context-dependent representations.
""")

s2_col1, s2_col2 = st.columns([1, 1])

with s2_col1:
    st.markdown("""
    ### Projection into Query, Key, and Value Tensors
    Given normalized hidden state X_{norm} in R^[B, N, d_model]:
    1. **Linear Projections**:
       * Q = X_{norm} W_Q -> [B, N, H_Q * d_k] -> reshape -> [B, H_Q, N, d_k]
       * K = X_{norm} W_K -> [B, N, H_{KV} * d_k] -> reshape -> [B, H_{KV}, N, d_k]
       * V = X_{norm} W_V -> [B, N, H_{KV} * d_k] -> reshape -> [B, H_{KV}, N, d_k]
    2. **Rotary Encoding**:
       RoPE rotates the 2D coordinate pairs of Q and K based on token index t in [0, N-1].
    3. **Grouped-Query Attention Repeat**:
       K and V heads are repeated \frac{H_Q}{H_{KV}} times to match the Query head count.
    """)

with s2_col2:
    st.markdown("""
    ### Attention Matrix Computation
    ```
    Scores = (Q @ K^T) / sqrt(d_k)      # Shape: [B, H_Q, N, N]
    ```
    * **Add Causal Mask**: Future positions (j > i) are set to -infinity.
    * **Softmax Normalization**:
      ```
      Attn_Weights = Softmax(Scores, dim=-1) # Shape: [B, H_Q, N, N]
      ```
    * **Value Weighting & Output Projection**:
      ```
      Attn_Out = (Attn_Weights @ V)          # Shape: [B, H_Q, N, d_k]
      Attn_Out = Reshape(Attn_Out) @ W_O     # Shape: [B, N, d_model]
      ```
    """)

# ==============================================================================
# SECTION 4: 3. ATTENTION -> HIDDEN STATES
# ==============================================================================
st.header("4. Attention -> Hidden States: Residual Merging & Multi-Layer Depth", divider="rainbow")

st.markdown("""
The output of attention does not replace the token representation; it is **added back** into the 
residual stream, followed by non-linear transformation in the Feed-Forward Network.
""")

s3_col1, s3_col2 = st.columns([1, 1])

with s3_col1:
    st.markdown("""
    ### The Residual & FFN Step Inside Layer l:
    1. **First Residual Skip**:
       `X_{mid} = X_{l-1} + Attn_Out    # Shape: [B, N, d_model]`
    2. **Second RMSNorm**:
       `X_{norm2} = RMSNorm(X_{mid})   # Shape: [B, N, d_model]`
    3. **SwiGLU Expansion**:
       * Gate Projection: `Gate = X_{norm2} @ W_gate  # [B, N, d_ff]`
       * Up Projection: `Up = X_{norm2} @ W_up        # [B, N, d_ff]`
       * Bilinear Gating: `H_ffn = (SiLU(Gate) * Up)  # [B, N, d_ff]`
       * Down Projection: `FFN_Out = H_ffn @ W_down   # [B, N, d_model]`
    4. **Second Residual Skip**:
       `X_l = X_{mid} + FFN_Out         # Shape: [B, N, d_model]`
    """)

with s3_col2:
    st.markdown("""
    ### Propagation Across Deep Towers
    * This process repeats sequentially through all L layers (L=32 for Llama 3 8B, L=80 for 70B).
    * At each layer, token representations become increasingly abstracted:
      * Layer 1-8: Syntactic structure, local phrase chunking.
      * Layer 9-24: Entity linking, coreference resolution, domain terminology.
      * Layer 25-32: High-level reasoning, intent modeling, task completion.
    * The output of layer L is the final hidden state matrix:
      `H_L in R^[B, N, d_model]`.
    """)

# ==============================================================================
# SECTION 5: 4. HIDDEN STATES -> LOGITS
# ==============================================================================
st.header("5. Hidden States -> Logits: Normalization and Vocabulary Unembedding", divider="rainbow")

st.markdown("""
At the end of the Transformer stack, the hidden states contain rich contextual representations. 
To generate text, we must project these continuous vectors back into the vocabulary space.
""")

s4_col1, s4_col2 = st.columns([1, 1])

with s4_col1:
    st.markdown("""
    ### Extracting the Generation Slice
    In autoregressive generation, we only care about predicting the token following the **very last position** of the context sequence:
    ```
    h_last = H_L[:, -1, :]    # Shape: [B, d_model]
    ```
    
    ### Final RMSNorm & Linear LM Head
    1. **Final Normalization**:
       `h_final = RMSNorm(h_last, gamma_final)  # Shape: [B, d_model]`
    2. **Matrix Multiplication with Unembedding Head**:
       `Logits = h_final @ W_vocab.T            # Shape: [B, Vocab_Size]`
    
    Where:
    * W_{vocab} in R^[V, d_model] (e.g., 128,256 * 4096).
    * Logits are **unnormalized real numbers** ([-infinity, +infty]) representing the raw, uncalibrated preference scores for each word in the vocabulary.
    """)

with s4_col2:
    st.markdown("""
    ### Compute Complexity of the LM Head
    * Multiplying [B, 4096] by [4096, 128256] requires:
      `FLOPs = 2 * B * 4096 * 128256 = 1.05 * B GigaFLOPs`.
    * For large vocabularies (V > 100,000), the final unembedding projection is one of the most memory-bandwidth intensive matrix multiplications in the entire forward pass!
    """)
    st.info("Logit values typically range from -15.0 to +25.0. Higher values signify stronger model confidence.")

# ==============================================================================
# SECTION 6: 5. LOGITS -> PROBABILITIES
# ==============================================================================
st.header("6. Logits -> Probabilities: Softmax and Temperature Dynamics", divider="rainbow")

st.markdown("""
Raw logits cannot be directly used for random token sampling because they are uncalibrated and unconstrained. 
**Temperature-scaled Softmax** converts logits into a valid categorical probability distribution.
""")

s5_col1, s5_col2 = st.columns([1, 1])

with s5_col1:
    st.markdown("""
    ### Mathematical Formulation
    For raw logit vector z in R^V and temperature T > 0:
    ```
    P(token_i) = exp(z_i / T) / Sum_{j=1}^{V} exp(z_j / T)
    ```
    
    ### Numerical Stability Trick (Logit Shifting)
    Because exp(25.0) approx 7.2 * 10^{10} and exp(100.0) = Overflow (NaN), 
    production implementations always subtract the maximum logit before exponentiation:
    ```
    z_safe = (z - max(z)) / T
    P(token_i) = exp(z_safe_i) / Sum_{j=1}^{V} exp(z_safe_j)
    ```
    Mathematically, subtracting a constant from all logits cancels out identically in the numerator and denominator, preventing floating-point overflow!
    """)

with s5_col2:
    st.markdown("""
    ### The Impact of Sampling Temperature (T)
    """)
    df_temp = pd.DataFrame({
        "Temperature (T)": ["T -> 0 (Greedy)", "T = 0.2 - 0.5", "T = 0.7 - 0.8", "T = 1.0", "T >= 1.5"],
        "Mathematical Effect": [
            "Dirac delta distribution (P=1.0 on max logit)",
            "Sharply peaks top candidates; suppresses tail",
            "Slightly smooths distribution; balanced",
            "Raw uncalibrated model probabilities",
            "Flattens toward uniform distribution"
        ],
        "Generation Quality": [
            "Deterministic, repetitive, factual",
            "High factual precision (coding, math, extraction)",
            "Ideal balance (creative writing, conversation)",
            "Standard benchmark evaluation",
            "Hallucinatory, erratic, gibberish"
        ]
    })
    st.dataframe(df_temp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 7: 6. PROBABILITIES -> GENERATED TOKENS
# ==============================================================================
st.header("7. Probabilities -> Generated Tokens: Sampling & Autoregressive Loop", divider="rainbow")

st.markdown("""
The final stage transforms the continuous probability distribution over V tokens into a 
single discrete **Token ID**, which is appended to the prompt to initiate the next forward pass.
""")

s6_col1, s6_col2 = st.columns([1, 1])

with s6_col1:
    st.markdown("""
    ### Production Sampling Filter Pipeline
    Before drawing a random sample from probability vector P:
    1. **Top-K Truncation**:
       Keep only the K tokens with the highest probabilities (e.g., K=50); set all others to 0.
    2. **Top-P (Nucleus) Truncation**:
       Sort tokens by probability descending. Find cutoff index where cumulative sum Sum(P_i) >= P (e.g., P=0.90). Truncate the tail.
    3. **Min-P Filtering (Modern Standard)**:
       Discard any token whose probability is less than a dynamic threshold:
       `Threshold = P_top * min_p` (e.g., min_p = 0.05).
    4. **Re-normalization**:
       Divide remaining probabilities by their sum so Sum(P_filtered) = 1.0.
    5. **Categorical Sampling**:
       `sampled_id = np.random.choice(V, p=P_filtered)`
    """)

with s6_col2:
    st.markdown("""
    ### Closing the Autoregressive Loop
    Once `sampled_id` is drawn:
    1. **Check Stop Condition**: If `sampled_id == EOS_TOKEN_ID` or max tokens reached, terminate generation.
    2. **Append to Sequence**:
       `input_ids = [input_ids, sampled_id]`
    3. **Update KV Cache**: In GPU VRAM, append K_{new} and V_{new} to `past_key_values`.
    4. **Emit Token to User**: Stream decoded text character chunk via SSE / WebSocket to user interface.
    5. **Trigger Step t+1**: Run the next forward pass with only the newly emitted token.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Pure Python End-to-End Generative Tensor Pipeline", divider="rainbow")

st.markdown("""
The following self-contained Python implementation traces the exact tensor shapes and transformations 
across all six stages of a generative forward pass.
""")

class GenerativeDataFlowPipeline:
    """
    Production-grade educational engine tracing the exact tensor shapes
    across all 6 stages of the Transformer generative data flow.
    """
    def __init__(self, vocab_size: int = 1000, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        
        rng = np.random.RandomState(seed)
        # Stage 1: Embedding table
        self.W_embed = rng.randn(vocab_size, d_model).astype(np.float32) * 0.02
        # Stage 2: Attention projections
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        # Stage 3: FFN projections
        self.W_gate = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.W_up = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.W_down = rng.randn(d_ff, d_model).astype(np.float32) * 0.02
        # Stage 4: LM Head unembedding
        self.W_unembed = rng.randn(d_model, vocab_size).astype(np.float32) * 0.02

    def trace_forward_pass(self, token_ids: List[int], temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Tuple[str, np.ndarray]]:
        """Executes full 6-stage data flow and logs tensor shapes at each boundary."""
        trace = {}
        
        # --- STAGE 1: Tokens -> Embeddings ---
        input_ids = np.array(token_ids, dtype=np.int64)[np.newaxis, :] # Shape: [1, Seq_Len]
        X_0 = self.W_embed[input_ids]                                  # Shape: [1, Seq_Len, d_model]
        trace["Stage 1 (Embeddings)"] = (f"{list(X_0.shape)} [Batch, Seq_Len, d_model]", X_0)
        
        # --- STAGE 2: Embeddings -> Attention ---
        B, N, D = X_0.shape
        Q = np.dot(X_0, self.W_q).reshape(B, N, self.num_heads, self.d_k).swapaxes(1, 2)
        K = np.dot(X_0, self.W_k).reshape(B, N, self.num_heads, self.d_k).swapaxes(1, 2)
        V = np.dot(X_0, self.W_v).reshape(B, N, self.num_heads, self.d_k).swapaxes(1, 2)
        
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        mask = np.triu(np.full((N, N), -1e9, dtype=np.float32), k=1)
        scores = scores + mask
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        attn_out = np.matmul(weights, V).swapaxes(1, 2).reshape(B, N, D)
        attn_out = np.dot(attn_out, self.W_o)                          # Shape: [1, Seq_Len, d_model]
        trace["Stage 2 (Attention Out)"] = (f"{list(attn_out.shape)} [Batch, Seq_Len, d_model]", attn_out)
        
        # --- STAGE 3: Attention -> Hidden States (Residual + FFN) ---
        X_mid = X_0 + attn_out
        gate = np.dot(X_mid, self.W_gate)
        silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
        up = np.dot(X_mid, self.W_up)
        ffn_out = np.dot(silu_gate * up, self.W_down)
        H_final = X_mid + ffn_out                                      # Shape: [1, Seq_Len, d_model]
        trace["Stage 3 (Final Hidden States)"] = (f"{list(H_final.shape)} [Batch, Seq_Len, d_model]", H_final)
        
        # --- STAGE 4: Hidden States -> Logits ---
        h_last = H_final[:, -1, :]                                     # Shape: [1, d_model]
        # RMSNorm
        rms = np.sqrt(np.mean(h_last ** 2, axis=-1, keepdims=True) + 1e-6)
        h_norm = h_last / rms
        logits = np.dot(h_norm, self.W_unembed)                        # Shape: [1, Vocab_Size]
        trace["Stage 4 (Vocabulary Logits)"] = (f"{list(logits.shape)} [Batch, Vocab_Size]", logits)
        
        # --- STAGE 5: Logits -> Probabilities ---
        scaled_logits = logits / max(temperature, 1e-4)
        safe_logits = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        exp_l = np.exp(safe_logits)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)          # Shape: [1, Vocab_Size]
        trace["Stage 5 (Softmax Probabilities)"] = (f"{list(probs.shape)} [Batch, Vocab_Size]", probs)
        
        # --- STAGE 6: Probabilities -> Sampled Token ---
        p_flat = probs.squeeze()
        # Nucleus Top-P filtering
        sorted_indices = np.argsort(p_flat)[::-1]
        sorted_probs = p_flat[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff_mask = cumsum_probs <= top_p
        cutoff_mask[0] = True # Keep at least 1 token
        valid_indices = sorted_indices[cutoff_mask]
        valid_probs = p_flat[valid_indices] / np.sum(p_flat[valid_indices])
        
        sampled_token_id = int(np.random.choice(valid_indices, p=valid_probs))
        trace["Stage 6 (Sampled Token ID)"] = (f"Scalar int64: {sampled_token_id}", np.array([sampled_token_id]))
        
        return trace

st.code('''
class GenerativeDataFlowPipeline:
    def __init__(self, vocab_size=1000, d_model=64, num_heads=4, d_ff=128, seed=42):
        ... # Full 6-stage forward pass implementation
        
    def trace_forward_pass(self, token_ids, temperature=0.7, top_p=0.9):
        # 1. Tokens -> Embeddings: X_0 = W_embed[token_ids]
        # 2. Embeddings -> Attention: MultiHeadAttention(X_0)
        # 3. Attention -> FFN: X_mid + SwiGLU(X_mid)
        # 4. Hidden States -> Logits: RMSNorm(H_final) @ W_unembed
        # 5. Logits -> Softmax: exp(logits / T) / sum(...)
        # 6. Sampling: Top-P Nucleus Selection -> Next Token ID
        return trace
''', language="python")


# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Generative Tensor Transformation Inspector", divider="rainbow")

st.markdown("""
Step interactively through the **6 stages of the Transformer Data Flow** with a live simulated forward pass. 
Observe exact tensor shape mutations, activation distributions, and output sampling choices.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("#### Input Context & Sampling Controls")
    prompt_tokens_str = st.text_input("Input Token IDs (comma-separated integers)", value="105, 420, 891, 12, 54")
    user_temp = st.slider("Sampling Temperature (T)", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    user_topp = st.slider("Nucleus Top-P Cutoff", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    try:
        input_ids = [int(x.strip()) for x in prompt_tokens_str.split(",") if x.strip()]
    except ValueError:
        input_ids = [105, 420, 891]

    # Instantiate pipeline
    pipe = GenerativeDataFlowPipeline(vocab_size=1000, d_model=64, num_heads=4, d_ff=128, seed=42)
    trace_results = pipe.trace_forward_pass(input_ids, temperature=user_temp, top_p=user_topp)

with wb_col2:
    st.markdown("#### Step-by-Step Tensor Shape Audit")
    
    trace_rows = []
    for stage_name, (shape_desc, tensor_val) in trace_results.items():
        trace_rows.append({
            "Pipeline Stage": stage_name,
            "Tensor Dimensions & Shape": shape_desc,
            "Min / Max Value": f"{np.min(tensor_val):.2f} / {np.max(tensor_val):.2f}"
        })
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
    
    sampled_id = trace_results["Stage 6 (Sampled Token ID)"][1][0]
    st.success(f"**Emitted Token ID**: `{sampled_id}` -> Appended to prompt context for next iteration.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is the Softmax stability trick (subtracting max logit) mathematically invariant?",
        "a": "Given logits z and constant c = max(z): exp(z_i - c) / sum_j exp(z_j - c) = (exp(z_i) * exp(-c)) / sum_j (exp(z_j) * exp(-c)). Factoring out exp(-c) from the denominator yields exp(-c) / exp(-c) = 1, canceling it completely. The resulting probabilities are mathematically identical to raw Softmax, but the largest exponent is now exp(0) = 1.0, guaranteeing that values never overflow into IEEE floating-point infinity (NaN)."
    },
    {
        "q": "What is the memory and FLOP difference between the Training forward pass and the Inference forward pass?",
        "a": "During training, all N tokens are processed in parallel: the attention matrix computes an N x N score grid, and the LM Head multiplies [B, N, d_model] by [d_model, V], computing loss across all N positions simultaneously (O(N^2) FLOPs). During inference, past Keys and Values are retrieved from the KV cache. The query sequence length is 1, so attention computes a 1 x N score vector, and the LM Head only multiplies the single final position [B, 1, d_model] by [d_model, V] (O(N) FLOPs per step)."
    },
    {
        "q": "What is Min-P sampling, and why does it outperform Top-P (Nucleus) in modern LLMs?",
        "a": "Top-P truncates the probability distribution when the cumulative sum reaches threshold P (e.g., 0.90). When the model is uncertain and the distribution is flat, Top-P includes hundreds of low-probability, low-quality tokens. Min-P sets a dynamic floor scaled relative to the highest probability token: Threshold = P_top * min_p (e.g., min_p = 0.05). If P_top is 0.80, tokens below 0.04 are filtered. If P_top is 0.10, tokens below 0.005 are filtered. This dynamically adapts candidate breadth to the model's confidence without letting nonsensical tails pollute responses."
    },
    {
        "q": "Why is the final Unembedding Head (LM Head) often the primary memory-bandwidth bottleneck during single-batch inference?",
        "a": "In single-batch generation, arithmetic intensity (FLOPs per byte of memory loaded) is low. Computing logits requires loading the entire unembedding weight matrix W_vocab (e.g., 128,256 x 4096 in FP16 = 1.05 GB) from GPU High Bandwidth Memory (HBM) into SRAM just to multiply by a single vector [1, 4096]. At 1 token per step, generating 100 tokens requires reading that 1.05 GB matrix 100 times, saturating GPU memory bus bandwidth."
    },
    {
        "q": "Why does Temperature = 0 cause a division by zero in Softmax, and how do inference engines implement it?",
        "a": "Softmax with temperature scales logits by 1/T. At T=0, 1/T approaches infinity. Production inference engines never compute Softmax at T=0; instead, they completely bypass Softmax exponentiation and execute a deterministic greedy argmax: sampled_id = argmax(logits), saving GPU memory and latency."
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
    **Trap 1 (Fatal): Materializing Full [B, N, V] Logits Tensor during Training**
    * *Failure*: Computing `logits = hidden_states @ W_vocab.T` across all sequence positions simultaneously when N=8192 and V=128256, requiring [4, 8192, 128256] * 4 \text{ bytes} approx 16.8 \text{ GB} of VRAM just for logits, causing immediate OOM.
    * *Fix*: Use **Fused Cross-Entropy Loss** (e.g., Liger-Kernel or Flash-CrossEntropy), which computes cross-entropy chunk-by-chunk in GPU SRAM without materializing the full 3D logits tensor in VRAM.
    """)
    st.error("""
    **Trap 2 (Fatal): Floating-Point Softmax Overflow without Logit Shifting**
    * *Failure*: Passing raw logits directly to `np.exp(logits / T)` where logits exceed 88.0 in FP32, generating `inf` values that become `NaN` after normalization.
    * *Fix*: Always subtract the row maximum: `z_safe = (z - max(z)) / T`.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Applying Temperature After Top-K Truncation**
    * *Failure*: Filtering Top-K tokens first and then applying temperature scaling.
    * *Fix*: Temperature modifies the relative sharpness of logits and must be applied **before** computing probabilities and filtering thresholds.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Redundant LM Head Projections for Prompt Tokens during Inference**
    * *Failure*: Unembedding all prompt tokens 1...N-1 during the prefill phase when the user only requested generated tokens.
    * *Fix*: Slice hidden states to position -1 (`h[:, -1, :]`) before multiplying by `W_vocab.T`.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Logit Shifting**\n\nSubtract max(z) before exponentiation. Mathematically identical, prevents floating-point overflow and NaNs.")
with fc_col2:
    st.info("**Prefill vs Decode**\n\nPrefill evaluates prompt tokens in parallel. Decode emits 1 token per step sequentially using cached Keys and Values.")
with fc_col3:
    st.info("**Min-P Filtering**\n\nFilters candidates below P_top * min_p, dynamically scaling candidate pool to model confidence.")
with fc_col4:
    st.info("**LM Head Bottleneck**\n\nMultiplying [1, d_model] by [d_model, V] is memory-bandwidth bound during single-stream token generation.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of the 6-stage generative data flow, tensor transformations, and sampling mechanics:
""")

c1 = st.checkbox("I can trace the exact tensor shapes from token IDs [B, N] to logits [B, V] in a modern LLM.")
c2 = st.checkbox("I understand why the Softmax numerical stability trick (subtracting max logit) is mathematically invariant.")
c3 = st.checkbox("I know why the inference forward pass only projects the final hidden state h[:, -1, :] to logits.")
c4 = st.checkbox("I can articulate the difference between Temperature, Top-K, Top-P, and Min-P sampling.")
c5 = st.checkbox("I understand why fused cross-entropy kernels are necessary to avoid multi-gigabyte logit VRAM explosions during training.")
c6 = st.checkbox("I know how the autoregressive loop closes by appending the sampled token ID and updating the KV cache.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered the Transformer generative data flow, tensor life cycles, and production sampling pipelines!")
