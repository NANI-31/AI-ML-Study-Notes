import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.5 Attention Engineering (Complexity, Memory, Long-Context, FlashAttention, Efficient Attention)",
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
st.markdown('<div class="main-title">6.5 Attention Engineering: Complexity, Long Contexts, and FlashAttention</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From the Quadratic O(N^2) Barrier to IO-Aware GPU SRAM Tiling (FlashAttention-2/3), Sliding Windows, and Enterprise KV-Cache Optimization.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Engineering Frontier of Scaled Attention", divider="rainbow")

st.markdown("""
While the theoretical formulation of scaled dot-product attention is mathematically elegant, 
scaling it to production context windows (from 2,048 tokens in GPT-3 to 128,000+ tokens in Llama 3 and 1,000,000+ in Gemini) 
requires overcoming severe **computational complexity (O(N^2))** and **memory bandwidth bottlenecks**:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.4")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("BOTTLENECK", "The Scaling Bottleneck\nCompute: O(N^2 . d) FLOPs\nMemory: O(N^2) intermediate attention matrix\nVRAM Traffic: Hundreds of GBs streamed per step", shape="box", fillcolor="#312E81", color="#818CF8")
    
    with arch_graph.subgraph(name="cluster_algo") as c:
        c.attr(bgcolor="#0F172A", label="Algorithmic Innovations", fontcolor="#38BDF8", color="#38BDF8")
        c.node("GQA", "Grouped-Query Attention (GQA)\nShare KV heads (4x-8x KV cache reduction)", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("SWA", "Sliding Window Attention (SWA)\nConstrain attention to window W (e.g., 4096)", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("SINKS", "Attention Sinks (StreamingLLM)\nKeep token 0 permanently in cache", shape="box", fillcolor="#1E293B", color="#38BDF8")

    with arch_graph.subgraph(name="cluster_hardware") as c:
        c.attr(bgcolor="#0F172A", label="Hardware-Aware Innovations", fontcolor="#10B981", color="#10B981")
        c.node("FLASH", "FlashAttention (Dao et al.)\nSRAM Block Tiling + Online Softmax\nNever materializes N x N matrix in HBM\nO(N) memory, 3x-4x speedup", shape="box", fillcolor="#042F2E", color="#14B8A6")
        c.node("PAGED", "PagedAttention (vLLM)\nVirtual memory paging for non-contiguous KV blocks\nEliminates internal fragmentation", shape="box", fillcolor="#042F2E", color="#14B8A6")

    arch_graph.edge("BOTTLENECK", "GQA")
    arch_graph.edge("BOTTLENECK", "SWA")
    arch_graph.edge("BOTTLENECK", "FLASH")
    arch_graph.edge("BOTTLENECK", "PAGED")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### The Memory-Bound Reality of Modern GPUs
    * **Compute vs. Memory Bandwidth**: Modern GPUs (NVIDIA H100) deliver massive compute (approx 2,000 TFLOPs of FP16), but memory bandwidth is capped at 3.35 TB/sec.
    * **Memory Traffic Bottleneck**: In naive attention, reading and writing the N * N attention matrix to High Bandwidth Memory (HBM) takes far longer than computing the actual arithmetic operations!
    * **The FlashAttention Breakthrough**: By computing attention in small tiles inside ultra-fast on-chip SRAM (approx 33 TB/sec bandwidth), FlashAttention achieves a 3x-4x wall-clock speedup without changing a single float of model output!
    """)

# ==============================================================================
# SECTION 2: 1. ATTENTION COMPLEXITY
# ==============================================================================
st.header("2. Attention Complexity: The O(N^2) Time and Space Barrier", divider="rainbow")

st.markdown("""
The computational and memory complexity of standard Multi-Head Attention scales **quadratically** with sequence length N.
""")

comp_col1, comp_col2 = st.columns([1, 1])

with comp_col1:
    st.markdown("""
    ### Derivation of Attention FLOPs
    For a sequence of length N with hidden dimension d_{model}:
    1. **Q, K, V Projections**:
       3 * (2 * N * d_{model}^2) = 6 N d_{model}^2 FLOPs.
    2. **Q K^T Dot Product**:
       2 * N * N * d_{model} = 2 N^2 d_{model} FLOPs.
    3. **Softmax**:
       3 * N^2 operations (Max, Exp, Sum, Div).
    4. **Weights * V Multiplication**:
       2 * N * N * d_{model} = 2 N^2 d_{model} FLOPs.
    5. **Output Projection W_O**:
       2 * N * d_{model}^2 FLOPs.
       
    **Total FLOPs per Attention Layer**:
    ```
    Total FLOPs = 4 N^2 d_{model} + 8 N d_{model}^2
    ```
    * When N << d_{model} (short prompts): Linear term 8 N d_{model}^2 dominates.
    * When N >> d_{model} (long context, e.g., N = 32,000): Quadratic term 4 N^2 d_{model} completely overwhelms total compute!
    """)

with comp_col2:
    st.markdown("""
    ### Scaling Law Impact: Sequence Length vs. FLOPs
    """)
    df_flops = pd.DataFrame({
        "Sequence Length (N)": ["1,024", "4,096", "16,384", "65,536", "131,072"],
        "N^2 Relative Growth": ["1x (Baseline)", "16x", "256x", "4,096x", "16,384x"],
        "N x N FP16 Matrix Memory (32 Heads)": ["64 MB", "1.0 GB", "16.0 GB", "256.0 GB (OOM)", "1.0 TB (Impossible)"],
        "Execution Mode": ["Standard GEMM", "Standard GEMM", "FlashAttention Required", "FlashAttention Required", "Distributed RingAttention Required"]
    })
    st.dataframe(df_flops, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 3: 2. MEMORY REQUIREMENTS
# ==============================================================================
st.header("3. Memory Requirements: The KV Cache Sizing Formula", divider="rainbow")

st.markdown("""
In autoregressive text generation, storing historical Keys and Values in GPU VRAM (the **KV Cache**) 
dominates total system memory consumption.
""")

mem_col1, mem_col2 = st.columns([1, 1])

with mem_col1:
    st.markdown("""
    ### The Universal KV Cache Formula
    For batch size B, sequence length N, layer count L, number of KV heads H_{KV}, and head dimension d_k:
    ```
    KV_Cache_Bytes = 2 * B * N * L * H_KV * d_k * Bytes_per_Element
    ```
    Where:
    * The factor of **2** accounts for storing both Keys and Values.
    * `Bytes_per_Element`: 2 bytes for FP16/BF16, 1 byte for FP8.
    
    ### Example: Llama 3 8B (32 Layers, H_{KV} = 8, d_k = 128)
    Per token stored in memory:
    `2 * 32 * 8 * 128 * 2 bytes = 131,072 bytes = 128 KB per token`.
    * For a single user (B=1) at N = 8,192 context:
      `KV Cache = 8,192 * 128 KB = 1.05 GB VRAM`.
    * For B = 32 concurrent users at N = 32,768 context:
      `KV Cache = 32 * 32,768 * 128 KB = 134.2 GB VRAM`!
      (Exceeds the total memory of an 80GB A100 GPU solely for cached tokens!)
    """)

with mem_col2:
    st.markdown("""
    ### KV Cache Sizing Comparison Across Architectures
    """)
    df_kv_comp = pd.DataFrame({
        "Model": ["GPT-3 175B (MHA)", "Llama 2 70B (GQA)", "Llama 3 8B (GQA)", "Mistral 7B (GQA)", "DeepSeek-V2 (MLA)"],
        "Layers (L)": ["96", "80", "32", "32", "60"],
        "KV Heads (H_kv)": ["96 (Full MHA)", "8 (GQA 8x)", "8 (GQA 4x)", "8 (GQA 4x)", "1 (Latent MLA)"],
        "Head Dim (d_k)": ["128", "128", "128", "128", "576 (Latent)"],
        "KV Bytes per Token": ["4.72 MB", "320 KB", "128 KB", "128 KB", "135 KB"],
        "Memory Reduction": ["1x (Baseline)", "14.7x smaller", "36.8x smaller", "36.8x smaller", "35.0x smaller"]
    })
    st.dataframe(df_kv_comp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 4: 3. LONG-CONTEXT PROBLEMS
# ==============================================================================
st.header("4. Long-Context Problems: Extrapolation, Anisotropy, and Sinks", divider="rainbow")

st.markdown("""
Simply expanding context length from 4k to 128k introduces three major architectural pathologies: 
**Positional Extrapolation Breakdown**, the **Lost in the Middle phenomenon**, and **Attention Sinks**.
""")

long_col1, long_col2 = st.columns([1, 1])

with long_col1:
    st.markdown("""
    ### 1. Positional Extrapolation Breakdown
    * Models trained on sequences up to length L_{train} suffer severe perplexity degradation when evaluated at L > L_{train}.
    * Rotary Position Embeddings (RoPE) assign unseen high rotation angles to positions > L_{train}, throwing query-key dot products out of distribution.
    * **Remedies**: RoPE interpolation, **NTK-Aware scaling**, and **YaRN (Yet another RoPE extensioN)**.

    ### 2. The "Lost in the Middle" Dilemma
    * Large models exhibit a U-shaped attention distribution: they attend heavily to tokens at the very beginning (prefix) and very end (recency), but frequently overlook facts placed in the center of long prompts.
    * **Remedies**: Reordering retrieved chunks in RAG to place critical facts at the beginning and end of the context.
    """)

with long_col2:
    st.markdown("""
    ### 3. Attention Sinks (Xiao et al., StreamingLLM, 2023)
    * **Discovery**: In autoregressive LLMs, the first 1 to 4 tokens (e.g., `<s>` and token 1) receive an extraordinarily large fraction of attention weights (up to 40%+), regardless of their semantic content!
    * **Mechanism**: Because Softmax requires probabilities to sum to 1.0, when a token needs to attend to *nothing* in a given head, it dumps its excess probability onto the initial tokens as a "sink".
    * **Failure**: Evicting the initial tokens from the KV cache causes immediate model collapse and NaN perplexity!
    * **StreamingLLM Fix**: Always retain the initial 4 sink tokens permanently in the KV cache, sliding the rest of the window dynamically.
    """)

# ==============================================================================
# SECTION 5: 4. ATTENTION VISUALIZATION
# ==============================================================================
st.header("5. Attention Visualization: Inspecting Head Specialization and Entropy", divider="rainbow")

st.markdown("""
Visualizing internal attention distributions reveals how individual heads partition work across 
syntax, entity tracking, and factual memory retrieval.
""")

viz_col1, viz_col2 = st.columns([1, 1])

with viz_col1:
    st.markdown("""
    ### Attention Entropy Metric
    The **Entropy** of an attention distribution measures whether a head is sharp and selective, or diffuse and broad:
    ```
    H(alpha_i) = - Sum_{j=1}^{N} alpha_{ij} * log(alpha_{ij} + eps)
    ```
    * **Low Entropy (H -> 0)**: The head concentrates 99% of its attention on 1 or 2 specific tokens (e.g., punctuation, coreference antecedents).
    * **High Entropy (H -> log(N))**: The head spreads attention uniformly across all tokens, acting like an unweighted average pooling layer.
    
    ### Tooling Standards
    * **BertViz (DeRose et al.)**: Interactive browser visualization tool for inspecting multi-head and cross-attention matrices in Jupyter notebooks.
    * **Mechanistic Circuits (Anthropic)**: Traces direct feature interactions across attention heads and MLP layers using Sparse Autoencoders (SAEs).
    """)

with viz_col2:
    st.markdown("""
    ### Head Functional Categorization
    """)
    df_heads = pd.DataFrame({
        "Discovered Head Type": ["Previous-Token Head", "Duplicate-Token Head", "Induction Head", "Punctuation Sink Head", "Syntactic Dependency Head"],
        "Attention Target": ["Token at position i - 1", "Identical words seen earlier in prompt", "Token that followed previous instance of current word", "First token <s> or period [.]", "Grammatical subject or governing verb"],
        "Entropy Profile": ["Near Zero (Ultra-sharp)", "Low", "Low to Moderate", "Near Zero (Sinks all excess mass)", "Moderate"],
        "Functional Role": ["Local n-gram modeling", "Entity recognition and tracking", "Core driver of in-context few-shot learning", "Maintains Softmax normalization stability", "Sentence syntactic parsing"]
    })
    st.dataframe(df_heads, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 6: 5. EFFICIENT ATTENTION
# ==============================================================================
st.header("6. Efficient Attention: Sparse Attention and Sliding Windows", divider="rainbow")

st.markdown("""
To break the quadratic compute barrier (O(N^2)), researchers developed **Efficient Attention approximations** 
that restrict attention visibility to sparse geometric patterns.
""")

eff_col1, eff_col2 = st.columns([1, 1])

with eff_col1:
    st.markdown("""
    ### Sparse Attention Paradigms
    1. **Sliding Window Attention (SWA - Mistral 7B)**:
       * Each token attends only to the preceding W tokens (e.g., window W = 4,096).
       * Slashes compute complexity from O(N^2) to O(N * W).
       * Information still propagates across long distances through layer depth: after L layers, effective receptive field is L * W tokens (e.g., 32 * 4096 = 131,072 tokens!).
    2. **Global + Local Attention (Longformer / BigBird)**:
       * Combines sliding local window with a few designated "global" tokens (e.g., `[CLS]` or question tokens) that can attend to all tokens.
    3. **State-Space Models (Mamba / RWKV)**:
       * Completely replaces attention with linear recurrent state-space models (O(N) inference and constant memory).
    """)

with eff_col2:
    st.markdown("""
    ### Efficient Attention Architectures Compared
    """)
    df_eff_comp = pd.DataFrame({
        "Mechanism": ["Full Attention", "Sliding Window Attention (SWA)", "BigBird Sparse Attention", "Linear Attention (Mamba SSM)"],
        "Time Complexity": ["O(N^2 * d)", "O(N * W * d)", "O(N * d)", "O(N * d)"],
        "Memory Footprint": ["O(N^2) or O(N) with Flash", "O(N * W)", "O(N)", "O(1) constant inference state"],
        "Global Receptive Field": ["Instant (1 layer)", "Hierarchical (L * W layers)", "Approximate graph", "Continuous hidden state decay"],
        "Representative Model": ["Llama 3, GPT-4", "Mistral 7B", "Longformer, BigBird", "Mamba-2, Jamba"]
    })
    st.dataframe(df_eff_comp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 7: 6. FLASHATTENTION
# ==============================================================================
st.header("7. FlashAttention: IO-Aware SRAM Tiling and Online Softmax", divider="rainbow")

st.markdown("""
Introduced by Tri Dao et al. at Stanford in 2022, **FlashAttention** is arguably the most impactful 
systems engineering breakthrough in modern deep learning. 
It makes attention **3x-5x faster and slashes memory consumption from O(N^2) to O(N)**—without any approximation!
""")

flash_col1, flash_col2 = st.columns([1, 1])

with flash_col1:
    st.markdown("""
    ### The GPU Memory Hierarchy Problem
    A GPU consists of two primary memory tiers:
    * **High-Bandwidth Memory (HBM - GPU VRAM)**: Large (80 GB), but relatively slow bandwidth (3.35\text{ TB/sec}).
    * **Static RAM (SRAM - On-Chip Cache)**: Extremely small (256\text{ KB} per Streaming Multiprocessor), but blindingly fast (33\text{ TB/sec}—10x faster than HBM!).
    
    ### The Standard Attention Flaw:
    1. Loads Q, K from HBM -> computes S = Q K^T -> **writes N * N matrix back to HBM**!
    2. Loads S from HBM -> computes Softmax -> **writes N * N weights back to HBM**!
    3. Loads weights and V from HBM -> computes O = A V -> writes output to HBM.
    
    **The bottleneck is not math; it is reading and writing the N * N matrix to HBM!**
    """)

with flash_col2:
    st.markdown("""
    ### How FlashAttention Solves It:
    1. **Tiling into SRAM Blocks**: Splits Q, K, V into small blocks (e.g., 128 * 128) that fit entirely within fast on-chip SRAM.
    2. **Online Softmax Algorithm**:
       Computes Softmax incrementally across blocks without materializing the entire row denominator at once! Maintains running maximums m and normalization sums l.
    3. **Recomputation in Backward Pass**:
       Never stores the intermediate N * N attention matrix for backpropagation. Instead, it recomputes attention on the fly in SRAM during the backward pass, which is faster than reading it from slow HBM!
    """)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: FlashAttention-2 and FlashAttention-3</strong><br>
<strong>FlashAttention-2</strong> (Dao, 2023) redesigned work partitioning to achieve 73% of theoretical peak GPU FLOPS on A100. <strong>FlashAttention-3</strong> (Dao et al., 2024) exploits NVIDIA Hopper H100 hardware asynchronous Tensor Memory Accelerators (TMA) and FP8 Tensor Cores to deliver 1.5 PFLOPS of throughput (up to 2x faster than FlashAttention-2).
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Pure Python Online Softmax Tiling Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation demonstrates the **Online Softmax Tiling Algorithm** 
that enables FlashAttention to compute exact Softmax incrementally across memory blocks without materializing the full score matrix.
""")

st.code('''
import numpy as np
from typing import Tuple

class OnlineSoftmaxTilingEngine:
    """
    Production-grade educational implementation demonstrating the mathematical
    core of FlashAttention: Incremental Online Softmax across tiled memory blocks.
    
    Computes exact Attention(Q, K, V) = Softmax((Q @ K^T) / sqrt(d_k)) @ V
    without ever storing the full [N, N] matrix in memory!
    """
    def __init__(self, block_size: int = 4):
        self.block_size = block_size

    def compute_tiled_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Executes block-tiled attention over Q, K, V matrices of shape [N, d].
        
        Args:
            Q, K, V: Tensors of shape [N, d]
            
        Returns:
            O: Final attention output tensor [N, d] mathematically identical to standard attention.
        """
        N, d = Q.shape
        scale = 1.0 / np.sqrt(d)
        B_size = self.block_size
        
        # Output accumulator and running statistics per query row
        O = np.zeros((N, d), dtype=np.float32)
        # Running max m_i and running sum l_i initialized
        m = np.full((N, 1), -np.inf, dtype=np.float32)
        l = np.zeros((N, 1), dtype=np.float32)
        
        num_k_blocks = int(np.ceil(N / B_size))
        
        # Outer loop over Key/Value memory blocks (tiled in SRAM)
        for j_block in range(num_k_blocks):
            start_j = j_block * B_size
            end_j = min(N, (j_block + 1) * B_size)
            
            K_j = K[start_j:end_j] # Shape: [B_size, d]
            V_j = V[start_j:end_j] # Shape: [B_size, d]
            
            # Compute local block scores: [N, B_size]
            S_ij = np.dot(Q, K_j.T) * scale
            
            # Update running statistics for each query row
            m_prev = m.copy()
            m_curr = np.max(S_ij, axis=-1, keepdims=True)
            m_new = np.maximum(m_prev, m_curr) # Running maximum
            
            # Rescale previous accumulator and compute local exponentiated scores
            P_ij = np.exp(S_ij - m_new)
            alpha = np.exp(m_prev - m_new)
            
            # Update running normalizer sum
            l_new = alpha * l + np.sum(P_ij, axis=-1, keepdims=True)
            
            # Update running output accumulator
            O = alpha * O + np.dot(P_ij, V_j)
            
            m = m_new
            l = l_new
            
        # Final normalization by running sum
        O = O / l
        return O
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Enterprise Attention Memory & Scalability Calculator", divider="rainbow")

st.markdown("""
Explore exact VRAM requirements, activation memory, and throughput trade-offs across 
**Context Length (N)**, **Batch Size (B)**, and **Attention Optimizations (MHA vs GQA vs FlashAttention)**.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.markdown("#### Hardware & Workload Specification")
    w_context_len = st.select_slider(
        "Context Window Length (Tokens)",
        options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
        value=32768
    )
    w_batch_size = st.slider("Concurrent Serving Batch Size", min_value=1, max_value=64, value=8)
    w_precision = st.selectbox("Precision Format", ["FP16 / BF16 (2 bytes)", "FP8 (1 byte)"])
    bytes_per_elem = 2 if "16" in w_precision else 1

    # Standard Llama 3 8B spec: L=32, H_Q=32, H_KV=8, d_k=128
    L = 32
    H_Q = 32
    H_KV = 8
    d_k = 128
    
    # KV Cache Calculation
    kv_mha_bytes = 2 * w_batch_size * w_context_len * L * H_Q * d_k * bytes_per_elem
    kv_gqa_bytes = 2 * w_batch_size * w_context_len * L * H_KV * d_k * bytes_per_elem
    
    # Intermediate Attention Matrix Memory (Naive vs FlashAttention)
    # Naive: [B, H_Q, N, N] * bytes
    naive_attn_bytes = w_batch_size * H_Q * (w_context_len ** 2) * bytes_per_elem
    flash_attn_bytes = 0.0 # FlashAttention does not materialize N x N in HBM!

with col_wb2:
    st.markdown("#### Memory Allocation Analysis")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("GQA KV Cache (Active)", f"{kv_gqa_bytes / (1024**3):.2f} GB")
        st.caption("Standard production format in Llama 3.")
    with col_m2:
        st.metric("MHA KV Cache (Unoptimized)", f"{kv_mha_bytes / (1024**3):.2f} GB")
        st.caption(f"4x larger than GQA! Saves {(kv_mha_bytes - kv_gqa_bytes) / (1024**3):.2f} GB VRAM.")

    st.markdown("---")
    st.markdown("#### Intermediate Activation Memory to Store Attention Matrix (N * N)")
    if naive_attn_bytes / (1024**3) > 80:
        st.error(f"❌ Naive PyTorch Attention: **{naive_attn_bytes / (1024**3):.2f} GB** -> Instant CUDA Out-Of-Memory Crash!")
    else:
        st.warning(f"⚠️ Naive PyTorch Attention: **{naive_attn_bytes / (1024**3):.2f} GB** in HBM.")
        
    st.success("✅ FlashAttention Memory: **~0.0 GB in HBM** (Tiled in 256 KB on-chip SRAM via Online Softmax).")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does FlashAttention achieve a 3x-4x speedup over standard PyTorch attention without altering model outputs?",
        "a": "Standard PyTorch attention is memory-bound: it repeatedly writes and reads the large [N, N] score and probability matrices to slow High-Bandwidth Memory (HBM). FlashAttention reorganizes the computation to be IO-aware: it loads small tiles of Q, K, and V into fast on-chip SRAM (33 TB/sec bandwidth), computes local attention, and updates the output incrementally using the Online Softmax trick. Because it never writes or reads the [N, N] matrix to HBM, it slashes memory traffic by an order of magnitude, speeding up execution while producing mathematically exact outputs."
    },
    {
        "q": "What are 'Attention Sinks', and why does removing token 0 cause catastrophic perplexity collapse in streaming LLMs?",
        "a": "Xiao et al. (StreamingLLM, 2023) discovered that autoregressive language models consistently assign high attention probability (up to 40%+) to the first 1 to 4 initial tokens of a sequence, regardless of semantic content. Because Softmax requires probabilities to sum to 1.0, the model uses the initial tokens as an attention sink to dump excess probability when current tokens have no strong contextual affinity. Evicting token 0 from a sliding-window KV cache removes this sink, causing attention weights to oscillate uncontrollably and driving perplexity to infinity."
    },
    {
        "q": "How does Sliding Window Attention (SWA) in Mistral 7B still allow tokens to access information beyond the 4096-token window?",
        "a": "While each individual attention layer only looks back W = 4,096 tokens, information propagates through layer depth. Layer 1 allows token t to see tokens [t-W, t]. Layer 2 allows token t to attend to layer 1 states that already attended to [t-2W, t-W]. After L stacked layers, the effective receptive field of token t expands hierarchically to L * W tokens (e.g., 32 layers * 4096 = 131,072 tokens), achieving broad long-context reasoning with linear per-layer compute."
    },
    {
        "q": "What is the difference between FlashAttention-1, FlashAttention-2, and FlashAttention-3?",
        "a": "FlashAttention-1 introduced SRAM block tiling and online Softmax, slashing memory from O(N^2) to O(N). FlashAttention-2 reduced non-matmul operations (rescaling), parallelized over the sequence dimension rather than only the batch/head dimensions, and raised GPU Tensor Core utilization from 35% to 73% on A100. FlashAttention-3 targets NVIDIA Hopper (H100) architectures, exploiting hardware Tensor Memory Accelerators (TMA), warp-group asynchronous instructions, and FP8 precision to reach 1.5 PFLOPS (up to 2x faster than FlashAttention-2)."
    },
    {
        "q": "How does PagedAttention (vLLM) resolve KV cache memory fragmentation?",
        "a": "In traditional inference, systems allocate contiguous virtual memory buffers for the maximum possible sequence length (e.g., reserving 8k tokens of VRAM per request upfront), causing 60-80% of VRAM to be wasted on internal and external fragmentation. Inspired by OS virtual memory paging, PagedAttention partitions the KV cache into fixed-size physical blocks (e.g., 16 tokens per block). Blocks are allocated dynamically on demand and can be non-contiguous in physical VRAM, boosting serving throughput by 2x-4x."
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
    **Trap 1 (Fatal): Evicting the First 4 Tokens in Sliding Window KV Caching**
    * *Failure*: Implementing a simple FIFO queue for the KV cache that pops the oldest tokens when memory is full, immediately removing token 0.
    * *Fix*: Always pin the first 4 tokens permanently in the KV cache as **Attention Sinks**, sliding only the subsequent window (StreamingLLM).
    """)
    st.error("""
    **Trap 2 (Fatal): Deploying Multi-Head Attention without GQA at Scale**
    * *Failure*: Hosting a 70B parameter model with standard MHA at 32k context length, causing KV cache memory to consume >150\text{ GB} per concurrent user stream and starving batch throughput.
    * *Fix*: Use Grouped-Query Attention (GQA) checkpoints (e.g., Llama 3 70B with 8 KV heads) to slash KV cache memory by 8x.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Disabling FlashAttention During Training**
    * *Failure*: Training long-context models (N > 8,192) using standard PyTorch attention kernels, causing 4x slower training times and GPU memory thrashing.
    * *Fix*: Always verify FlashAttention is active (`torch.nn.functional.scaled_dot_product_attention` automatically dispatches FlashAttention when available).
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Unpaged Contiguous Memory Allocation in Multi-User Serving**
    * *Failure*: Pre-allocating contiguous static buffers for 100 requests at max context length, running out of VRAM while 70% of memory remains unutilized.
    * *Fix*: Deploy serving engines with **PagedAttention** (vLLM or TensorRT-LLM) to dynamically allocate non-contiguous KV blocks.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**FlashAttention IO Tiling**\n\nTiles Q, K, V in 256 KB on-chip SRAM via Online Softmax, eliminating the N x N matrix from slow HBM.")
with fc_col2:
    st.info("**KV Cache Equation**\n\n2 * B * N * L * H_kv * d_k * bytes. Dominates inference memory footprint in long-context serving.")
with fc_col3:
    st.info("**Attention Sinks**\n\nInitial tokens (0-4) absorb excess Softmax probability mass. Must never be evicted from sliding window caches.")
with fc_col4:
    st.info("**PagedAttention (vLLM)**\n\nPartitions KV cache into non-contiguous physical memory blocks, eliminating internal fragmentation.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of attention engineering, computational complexity, memory limits, and FlashAttention:
""")

c1 = st.checkbox("I can calculate the exact FLOP count and memory footprint of an attention layer for sequence length N.")
c2 = st.checkbox("I know the exact formula for KV Cache memory sizing and understand why GQA achieves a 4x-8x memory reduction.")
c3 = st.checkbox("I understand the 'Lost in the Middle' problem and know why Attention Sinks are essential for streaming generation.")
c4 = st.checkbox("I can explain how FlashAttention's online Softmax and SRAM tiling eliminate the N x N matrix from GPU HBM.")
c5 = st.checkbox("I understand how Sliding Window Attention achieves an L * W effective receptive field across layers.")
c6 = st.checkbox("I know how PagedAttention resolves memory fragmentation in production inference engines.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered attention engineering, memory bandwidth bottlenecks, FlashAttention tiling, and long-context scaling architectures!")
