import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.4 Causal Attention (Causal Masking, Autoregressive Generation, Lower-Triangular Mask, Next-Token)",
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
st.markdown('<div class="main-title">6.4 Causal Attention: Masking, Autoregressive Loops, and Next-Token Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From the Mathematical Foundations of the Lower-Triangular Mask to KV-Cache Token Emission and Teacher Forcing — Enforcing the Arrow of Time in Foundation Models.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Arrow of Time in Causal Generative Models", divider="rainbow")

st.markdown("""
In generative language modeling, text is produced sequentially from left to right. 
When predicting the word at position t+1, the model must never be allowed to "peek" at token t+1 or any future token. 
**Causal Attention** enforces this fundamental constraint during parallel training by imposing a **strict lower-triangular visibility mask**:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("TRAIN_IN", "Training Prompt Tokens (Parallel)\nTokens: [w_1, w_2, w_3, w_4]\nEvaluated simultaneously on GPU", shape="box", fillcolor="#1E293B", color="#38BDF8")
    arch_graph.node("RAW_SCORES", "Raw Score Matrix: Q . K^T / sqrt(d_k)\nShape: [4, 4] Full Grid\nTokens 1-4 attend to tokens 1-4", shape="box", fillcolor="#312E81", color="#818CF8")
    arch_graph.node("CAUSAL_MASK", "Additive Causal Mask Matrix M\nM_ij = 0 if j <= i; M_ij = -inf if j > i\nShape: [4, 4] Upper-Triangular -inf", shape="box", fillcolor="#1E293B", color="#F59E0B")
    arch_graph.node("SOFTMAX", "Softmax(Scores + M, dim=-1)\nexp(-inf) = 0.0 -> Future strictly zeroed\nShape: [4, 4] Lower-Triangular Weights", shape="box", fillcolor="#042F2E", color="#14B8A6")
    arch_graph.node("NEXT_TOK", "Parallel Next-Token Prediction Head\nPosition 1 predicts w_2 | Position 2 predicts w_3\nPosition 3 predicts w_4 | Position 4 predicts w_5", shape="box", fillcolor="#042F2E", color="#10B981")

    # Edges
    arch_graph.edge("TRAIN_IN", "RAW_SCORES")
    arch_graph.edge("RAW_SCORES", "CAUSAL_MASK", label="Add Mask")
    arch_graph.edge("CAUSAL_MASK", "SOFTMAX")
    arch_graph.edge("SOFTMAX", "NEXT_TOK")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Why Causal Attention Enabled the LLM Revolution
    * **Parallel Training Efficiency**: Even though text is generated one token at a time during inference, causal masking allows GPUs to train on all N tokens **in a single parallel matrix multiplication**!
    * **Teacher Forcing**: The model predicts token t+1 conditioned on true historical ground-truth tokens w_{1:t}, regardless of whether its previous predictions were correct.
    * **Mathematical Equivalence**: The representations learned by position t during parallel training are strictly identical to the representations computed when token t is generated during live inference!
    """)

# ==============================================================================
# SECTION 2: 1. CAUSAL MASKING
# ==============================================================================
st.header("2. Causal Masking: Preserving Unidirectional Information Flow", divider="rainbow")

st.markdown("""
Without a causal mask, self-attention is bidirectional: Query q_1 can compute a high dot-product score with Key k_5. 
In an autoregressive generative model, this would constitute catastrophic **information leakage**: 
the model would simply copy future tokens rather than learning to predict them.
""")

col_cm1, col_cm2 = st.columns([1, 1])

with col_cm1:
    st.markdown("""
    ### Autoregressive Probability Factorization
    By the chain rule of probability, any joint probability distribution over a sequence of T tokens factors into:
    ```
    P(w_1, w_2, ..., w_T) = Product_{t=1}^{T} P(w_t | w_1, w_2, ..., w_{t-1})
    ```
    To model this exact conditional probability P(w_t | w_<t), the representation of token t must depend **strictly on tokens 1 through t**:
    ```
    h_t = Transformer(w_1, w_2, ..., w_t)
    ```
    Causal masking ensures that no information from tokens t+1, t+2, ... can ever flow backward into h_t.
    """)

with col_cm2:
    st.markdown("""
    ### Bidirectional vs. Causal Attention Topology
    """)
    df_mask_types = pd.DataFrame({
        "Attribute": ["BERT (Bidirectional)", "GPT / Llama (Causal)", "Prefix LM (Hybrid)"],
        "Attention Matrix Structure": ["Full N * N dense grid", "Lower-triangular matrix", "Bidirectional block + Lower-triangular tail"],
        "Future Token Visibility": ["Visible (Can see all future tokens)", "Strictly Invisible (Future masked)", "Prompt visible; Generation masked"],
        "Training Task": ["Masked LM (Reconstruct [MASK])", "Next-Token Prediction", "Conditioned Sequence Generation"],
        "Autoregressive Generation": ["Fails completely", "Native and mathematically exact", "Native and mathematically exact"]
    })
    st.dataframe(df_mask_types, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 3: 2. AUTOREGRESSIVE GENERATION
# ==============================================================================
st.header("3. Autoregressive Generation: Sequential Emission & KV Caching", divider="rainbow")

st.markdown("""
At inference time, text generation is inherently sequential: we cannot know token t+1 until token t is emitted by the model. 
This process is divided into two distinct execution regimes: **Prefill** and **Decode**.
""")

col_ar1, col_ar2 = st.columns([1, 1])

with col_ar1:
    st.markdown("""
    ### 1. Prefill Phase (Prompt Ingestion)
    * The user submits an initial prompt of length P (e.g., P = 500 tokens).
    * All P tokens are processed simultaneously in parallel through the model.
    * The attention scores compute a P * P lower-triangular causal matrix.
    * **Time-To-First-Token (TTFT)** is determined by the speed of this prefill forward pass.
    * The Key and Value vectors for all P tokens are stored in GPU memory as the **KV Cache**.
    
    ### 2. Decode Phase (Token Generation Loop)
    * The model predicts the first new token (P+1).
    * On subsequent steps, we do **not** re-run tokens 1...P.
    * Only token P+1 is passed through the model: Query length is 1!
    * The single Query vector q_{new} attends to the entire historical KV Cache (P cached Keys).
    * Emits token P+2, appends its K and V to cache, and repeats.
    """)

with col_ar2:
    st.markdown("""
    ### Prefill vs. Decode Hardware Characteristics
    | Metric | Prefill Phase | Decode Phase |
    | :--- | :--- | :--- |
    | **Query Length** | N_{prompt} (e.g., 1024) | Exactly 1 |
    | **Compute Regime** | Compute-bound (High TFLOPs) | Memory-bandwidth bound |
    | **Matrix Operation** | GEMM ([B, N, D] * [D, D]) | GEMV ([B, 1, D] * [D, D]) |
    | **GPU Bottleneck** | Tensor Core saturation | VRAM read speed (Memory bus) |
    | **KV Cache Role** | Populates the initial cache | Reads cache and appends 1 new entry |
    """)

# ==============================================================================
# SECTION 4: 3. FUTURE-TOKEN MASKING
# ==============================================================================
st.header("4. Future-Token Masking: Mathematical Implementation via Additive Negative Infinity", divider="rainbow")

st.markdown("""
In neural network implementations, future tokens are not "removed" from the tensor; 
rather, their raw attention scores are overridden by **adding negative infinity (-infinity)** prior to Softmax.
""")

col_ftm1, col_ftm2 = st.columns([1, 1])

with col_ftm1:
    st.markdown("""
    ### Why Additive Masking Works
    Recall the Softmax equation for attention weights:
    ```
    alpha_{i, j} = exp(Scores_{i, j}) / Sum_{m=1}^{N} exp(Scores_{i, m})
    ```
    If we set Scores_{i, j} = -infinity for any future position (j > i):
    ```
    exp(-infinity) = 0.0
    ```
    The numerator becomes exactly 0.0, and the term contributes 0.0 to the denominator sum. 
    Consequently:
    ```
    alpha_{i, j} = 0.0    for all j > i
    ```
    The attention weight is strictly zero, and no information from token j can reach token i!
    """)

with col_ftm2:
    st.markdown("""
    ### Numerical Safety in Production (FP16 / BF16 Limits)
    * In IEEE 754 floating-point, true `-inf` can produce `NaN` if subtracted or multiplied.
    * Furthermore, in **FP16 (Half Precision)**, the minimum representable finite number is `-65,504`.
    * If you use `-1e9` inside an FP16 tensor, it underflows to `-inf` and can trigger numerical instabilities.
    * **Production Standard**:
      ```python
      # PyTorch robust masking:
      mask_value = torch.finfo(scores.dtype).min
      scores = scores.masked_fill(causal_mask, mask_value)
      ```
    """)

# ==============================================================================
# SECTION 5: 4. LOWER-TRIANGULAR ATTENTION MASKS
# ==============================================================================
st.header("5. Lower-Triangular Attention Masks: Topology and Tensor Broadcasting", divider="rainbow")

st.markdown("""
A **Lower-Triangular Mask** is a square binary matrix where all entries on and below the main diagonal are valid (1 or 0), 
while all entries strictly above the main diagonal are masked (0 or -infinity).
""")

col_lt1, col_lt2 = st.columns([1, 1])

with col_lt1:
    st.markdown("""
    ### The 4x4 Causal Attention Grid
    For a sequence of 4 tokens:
    ```
    Causal Mask M:
    [ [  0.0,   -inf,   -inf,   -inf ],    <- Token 1 sees ONLY Token 1
      [  0.0,    0.0,   -inf,   -inf ],    <- Token 2 sees Tokens 1 and 2
      [  0.0,    0.0,    0.0,   -inf ],    <- Token 3 sees Tokens 1, 2, and 3
      [  0.0,    0.0,    0.0,    0.0 ] ]   <- Token 4 sees ALL tokens
    ```
    
    ### Constructing the Mask in NumPy and PyTorch
    * In NumPy:
      ```python
      # np.triu with k=1 selects all strictly upper-triangular entries
      mask = np.triu(np.full((N, N), -1e9, dtype=np.float32), k=1)
      ```
    * In PyTorch:
      ```python
      mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
      ```
    """)

with col_lt2:
    st.markdown("""
    ### 4D Tensor Broadcasting
    The attention scores tensor in Multi-Head Attention has shape:
    `Scores.shape = [Batch_Size, Num_Heads, Seq_Len, Seq_Len]`
    
    A 2D causal mask of shape `[Seq_Len, Seq_Len]` or `[1, 1, Seq_Len, Seq_Len]` is automatically **broadcast across all batches and all heads** without copying memory!
    """)
    st.info("The causal mask is identical across all attention heads and all batch items; it depends purely on sequence length N.")

# ==============================================================================
# SECTION 6: 5. NEXT-TOKEN PREDICTION
# ==============================================================================
st.header("6. Next-Token Prediction: Teacher Forcing vs. Free-Running Generation", divider="rainbow")

st.markdown("""
The objective of pretraining a causal language model is **Next-Token Prediction**. 
The interaction between training (Teacher Forcing) and inference (Autoregressive Rollout) defines the model's behavior.
""")

col_ntp1, col_ntp2 = st.columns([1, 1])

with col_ntp1:
    st.markdown("""
    ### Teacher Forcing During Training
    Consider target sentence: `"The cat sat on the mat."`
    * Model Input: `["The", "cat", "sat", "on", "the"]`
    * Target Labels: `["cat", "sat", "on", "the", "mat."]`
    * Position 1 receives `"The"` -> computes hidden state h_1 -> loss compares to `"cat"`.
    * Position 2 receives `"The cat"` -> computes hidden state h_2 -> loss compares to `"sat"`.
    * Position 3 receives `"The cat sat"` -> computes hidden state h_3 -> loss compares to `"on"`.
    
    Even if position 1 predicted `"dog"` instead of `"cat"`, position 2 still receives the ground-truth token `"cat"`! 
    This is **Teacher Forcing**: it stabilizes training and prevents errors from compounding during gradient descent.
    """)

with col_ntp2:
    st.markdown("""
    ### Free-Running Generation at Inference
    * During real-world generation, there is no teacher.
    * If the model samples an incorrect or hallucinated token at step t, that hallucinated token is permanently appended to the prompt context for step t+1.
    * This distribution shift between training (ground-truth history) and inference (model-generated history) is known as **Exposure Bias**.
    """)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Pure Python Causal Attention Engine & KV-Cache Tracker", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. **Vectorized Lower-Triangular Causal Masking**.
2. **Step-by-Step KV Cache Tracking** simulating both the Prefill and Decode phases.
3. An **Autoregressive Generation Loop** emitting tokens sequentially.
""")

st.code('''
import numpy as np
from typing import Dict, List, Tuple, Optional

class CausalAttentionEngine:
    """
    Production-grade educational engine demonstrating:
    1. Lower-triangular causal masking.
    2. KV Cache accumulation during sequential token generation.
    3. Prefill vs. Decode phase execution.
    """
    def __init__(self, d_model: int = 16, d_k: int = 8, vocab_size: int = 50, seed: int = 42):
        self.d_model = d_model
        self.d_k = d_k
        self.vocab_size = vocab_size
        
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_k = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_v = rng.randn(d_model, d_k).astype(np.float32) * 0.1
        self.W_unembed = rng.randn(d_k, vocab_size).astype(np.float32) * 0.1

    @staticmethod
    def create_causal_mask(q_len: int, k_len: int) -> np.ndarray:
        """
        Creates causal attention mask for query length q_len and key length k_len.
        """
        # For prefill (q_len == k_len): standard lower-triangular
        # For decode (q_len == 1): query at position (k_len - 1) can see all past keys (no mask needed)
        mask = np.zeros((q_len, k_len), dtype=np.float32)
        if q_len > 1:
            # Mask out future positions
            for i in range(q_len):
                for j in range(k_len):
                    if j > i:
                        mask[i, j] = -1e9
        return mask

    def prefill_phase(self, X_prompt: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes parallel prefill pass over prompt of length P.
        Returns initial logits and populated KV Cache.
        """
        P = X_prompt.shape[0]
        Q = np.dot(X_prompt, self.W_q)
        K = np.dot(X_prompt, self.W_k)
        V = np.dot(X_prompt, self.W_v)
        
        # Parallel causal attention
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        mask = self.create_causal_mask(P, P)
        scores += mask
        
        safe_s = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(safe_s) / np.sum(np.exp(safe_s), axis=-1, keepdims=True)
        context = np.dot(weights, V)
        
        # Unembed last position
        logits_last = np.dot(context[-1:], self.W_unembed)
        
        kv_cache = {"K": K, "V": V}
        return logits_last, kv_cache

    def decode_step(self, x_new: np.ndarray, kv_cache: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, np.ndarray], np.ndarray]:
        """
        Executes single decode step for 1 new token using cached Keys and Values.
        """
        # 1. Project single new token: [1, d_k]
        q_new = np.dot(x_new, self.W_q)
        k_new = np.dot(x_new, self.W_k)
        v_new = np.dot(x_new, self.W_v)
        
        # 2. Append to KV Cache
        K_all = np.vstack([kv_cache["K"], k_new]) # [total_len, d_k]
        V_all = np.vstack([kv_cache["V"], v_new]) # [total_len, d_k]
        
        # 3. Attention: single query dot-product against all cached keys
        scores = np.dot(q_new, K_all.T) / np.sqrt(self.d_k) # Shape: [1, total_len]
        # Query at end can see all historical keys; no mask needed!
        safe_s = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(safe_s) / np.sum(exp_s := np.exp(safe_s), axis=-1, keepdims=True)
        
        context_new = np.dot(weights, V_all) # Shape: [1, d_k]
        logits = np.dot(context_new, self.W_unembed) # Shape: [1, vocab_size]
        
        sampled_id = int(np.argmax(logits))
        updated_cache = {"K": K_all, "V": V_all}
        return sampled_id, updated_cache, weights
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Live Autoregressive Generation & Causal Mask Inspector", divider="rainbow")

st.markdown("""
Step through a live simulated **Autoregressive Generation Loop**. 
Observe how the **Causal Attention Matrix** expands row-by-row as new tokens are emitted, 
and inspect how the single new Query attends across the accumulated KV Cache.
""")

sample_prompt = ["The", "quick", "brown", "fox"]
P_len = len(sample_prompt)

# State tracking
if "gen_tokens" not in st.session_state:
    st.session_state.gen_tokens = sample_prompt.copy()

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Generate Next Token (+1 step)"):
        synthetic_vocab = ["jumps", "over", "the", "lazy", "sleeping", "hound", "dog", "fence"]
        next_tok = synthetic_vocab[(len(st.session_state.gen_tokens) - P_len) % len(synthetic_vocab)]
        st.session_state.gen_tokens.append(next_tok)
with col_btn2:
    if st.button("Reset to Initial Prompt"):
        st.session_state.gen_tokens = sample_prompt.copy()

curr_seq = st.session_state.gen_tokens
N_curr = len(curr_seq)

st.markdown(f"**Current Sequence**: `{' '.join(curr_seq)}` (Total Length: {N_curr} tokens)")

# Render lower-triangular causal attention weights
causal_grid = np.zeros((N_curr, N_curr))
for i in range(N_curr):
    for j in range(i + 1):
        # Decay attention over distance for visual realism
        causal_grid[i, j] = np.exp(-0.4 * (i - j))
    # Normalize row to sum to 1.0
    causal_grid[i] = causal_grid[i] / np.sum(causal_grid[i])

df_causal = pd.DataFrame(causal_grid, index=curr_seq, columns=curr_seq)
st.dataframe(df_causal.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0.0, vmax=1.0), use_container_width=True)

st.caption("Lower-Triangular Visibility Grid: Notice that every newly generated row can attend to all prior tokens, while upper-triangular lookahead cells are strictly zeroed out.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is the additive causal mask matrix set to negative infinity (-inf) rather than zero (0.0)?",
        "a": "Softmax computes exp(score). If un-attended entries were set to 0.0, exp(0.0) = 1.0, which would grant future masked tokens equal baseline probability weight in the Softmax partition function. Setting masked entries to -infinity ensures exp(-inf) = 0.0, which completely nullifies their contribution in both the numerator and the denominator, resulting in strictly zero attention weight."
    },
    {
        "q": "What is the primary operational difference between the Prefill phase and the Decode phase in production LLM inference?",
        "a": "In the Prefill phase, all prompt tokens are processed concurrently: the attention computation is compute-bound (GEMM) with an N x N lower-triangular mask, and Key/Value states are written to the KV cache. In the Decode phase, tokens are generated one-by-one: query length is 1, so attention computes a 1 x N vector dot product against the cached Keys. Decode is memory-bandwidth bound (GEMV), requiring high VRAM bandwidth to stream large KV caches into GPU SRAM for every single generated token."
    },
    {
        "q": "What is 'Exposure Bias' in autoregressive language models, and how does it relate to Teacher Forcing?",
        "a": "During training with Teacher Forcing, the model is conditioned exclusively on ground-truth historical tokens. During inference (free-running generation), the model is conditioned on its own previous predictions. If the model makes an early error during generation, it enters an out-of-distribution state that was never seen during training, causing errors to compound rapidly into hallucinations. This discrepancy between training conditioning and inference conditioning is called Exposure Bias."
    },
    {
        "q": "Why doesn't the Decode phase require a causal attention mask?",
        "a": "During the decode phase, the model only passes the single newly generated token into the attention sublayer (query length = 1). The KV cache contains only historical tokens that have already been emitted in past time steps. Because there are no future tokens present in either the query tensor or the KV cache tensor, lookahead leakage is physically impossible, making causal masking completely redundant during decode steps."
    },
    {
        "q": "How does FlashAttention handle causal masking with hardware speedups?",
        "a": "FlashAttention uses causal block masking: in the lower-triangular score matrix, blocks strictly below the diagonal are computed with full attention, while blocks strictly above the diagonal are skipped entirely without loading Keys/Values into SRAM or executing tensor core instructions. This cuts total FLOPs and memory reads by exactly 50% compared to bidirectional attention."
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
    **Trap 1 (Fatal): Inverting the Causal Mask Diagonal**
    * *Failure*: Setting the diagonal to `-inf` or using `k=0` instead of `k=1` in `np.triu`, causing tokens to be unable to attend even to *themselves*.
    * *Fix*: Position (i, i) (self-attention) must always be allowed (0.0). Only positions strictly where j > i must receive `-inf`.
    """)
    st.error("""
    **Trap 2 (Fatal): FP16 Overflow/NaN with Extreme Mask Values**
    * *Failure*: Masking future positions with `-1e20` inside an FP16 tensor, which underflows to IEEE `-inf` and causes `NaN` during subsequent matrix additions.
    * *Fix*: Always use `torch.finfo(tensor.dtype).min` for precision safety.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Re-Evaluating Full Context at Every Decode Step**
    * *Failure*: Running prompt + all emitted tokens through the full model at every generation step without implementing a KV Cache (O(N^2) latency).
    * *Fix*: Implement PagedAttention (vLLM) or `past_key_values` caching, reducing per-token decode step latency to O(1).
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Padding Past Causal Masks in Dynamic Batches**
    * *Failure*: Applying uniform causal masks across batched sequences with left-padding or right-padding, inadvertently allowing padding tokens to pollute attention weights.
    * *Fix*: Combine causal masks with attention masks using element-wise addition: `total_mask = causal_mask + padding_mask`.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Causal Lower-Triangular**\n\nM_ij = 0 if j <= i; -inf if j > i. Guarantees that token t can only see tokens from the past and present.")
with fc_col2:
    st.info("**Teacher Forcing**\n\nConditions parallel training on ground-truth history, allowing all N sequence positions to train simultaneously.")
with fc_col3:
    st.info("**Prefill vs Decode**\n\nPrefill evaluates prompt tokens in parallel (compute-bound). Decode emits 1 token per step via KV cache (bandwidth-bound).")
with fc_col4:
    st.info("**Exposure Bias**\n\nDiscrepancy between ground-truth training history and model-generated inference history that drives hallucinations.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of causal attention, autoregressive decoding, and next-token prediction:
""")

c1 = st.checkbox("I can explain why causal attention uses an additive -infinity mask instead of 0.0.")
c2 = st.checkbox("I understand the difference between parallel Teacher Forcing during training and free-running autoregression during inference.")
c3 = st.checkbox("I know why the Decode phase is memory-bandwidth bound while the Prefill phase is compute-bound.")
c4 = st.checkbox("I can construct a lower-triangular causal attention mask in NumPy or PyTorch.")
c5 = st.checkbox("I understand why the Decode phase does not require a causal attention mask.")
c6 = st.checkbox("I know what Exposure Bias is and how compounding errors manifest during inference.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered causal attention, autoregressive token emission, lower-triangular masking, and next-token prediction mechanics!")
