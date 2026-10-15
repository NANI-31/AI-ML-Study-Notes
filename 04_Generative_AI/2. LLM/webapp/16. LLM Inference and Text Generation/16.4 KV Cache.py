import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.4 KV Cache - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionKVCacheEngine:
    """
    Self-contained educational KV Cache simulation engine:
    - Calculates exact memory footprint across MHA, GQA, and MQA
    - Simulates Prefill phase (parallel KV tensor population)
    - Simulates Decode phase (single-token KV appending)
    - Demonstrates O(1) step complexity vs O(N) recomputation
    """
    def __init__(
        self,
        num_layers: int = 32,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,  # GQA default (e.g. Llama-3-8B)
        head_dim: int = 128,
        bytes_per_element: int = 2  # BF16 / FP16
    ):
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bytes_per_element = bytes_per_element

    def calculate_kv_memory_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Exact formula: 2 (K and V) * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_elem
        """
        bytes_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element
        return bytes_per_token * seq_len * batch_size

    def format_memory_gb(self, num_bytes: int) -> float:
        """Converts bytes to gigabytes."""
        return num_bytes / (1024**3)

    def simulate_cache_lifecycle(self, prompt_len: int, gen_tokens: int) -> pd.DataFrame:
        """Simulates step-by-step KV cache growth during Prefill and Decode."""
        records = []
        
        # Step 0: Prefill
        prefill_bytes = self.calculate_kv_memory_bytes(prompt_len, batch_size=1)
        records.append({
            "Step": "Prefill (Prompt)",
            "Sequence Length": prompt_len,
            "KV Memory (MB)": prefill_bytes / (1024**2),
            "Compute FLOPs Scaled": prompt_len ** 2,
            "Phase": "Prefill (Compute-Bound)"
        })

        # Decode steps
        current_len = prompt_len
        for step in range(1, gen_tokens + 1):
            current_len += 1
            decode_bytes = self.calculate_kv_memory_bytes(current_len, batch_size=1)
            records.append({
                "Step": f"Decode {step}",
                "Sequence Length": current_len,
                "KV Memory (MB)": decode_bytes / (1024**2),
                "Compute FLOPs Scaled": current_len,  # Only 1 query attending to N keys
                "Phase": "Decode (Memory-Bound)"
            })

        return pd.DataFrame(records)

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("16.4 KV Cache")
st.caption("Chapter 16: LLM Inference and Text Generation | Domain: 2. Large Language Models")

st.markdown("""
The **KV Cache (Key-Value Cache)** is the single most critical runtime optimization in modern Large Language Model serving.
Without the KV cache, generating text with an LLM would suffer from **catastrophic quadratic slowdown**, 
recomputing the entire prompt and all previously emitted tokens from scratch for every single new word.

By storing the intermediate **Key** and **Value** projection tensors in GPU VRAM across self-attention layers, 
the model reduces the per-step decode computation from **`O(N)` down to `O(1)`**.
However, this algorithmic speedup comes at a steep price: **massive GPU memory consumption**.

In this lesson, you will master:
1. What the KV Cache is and how it avoids recomputing attention matrices.
2. The complexity leap: transforming `O(N^2)` generation latency into linear `O(N)` overall complexity.
3. The Prefill Phase (compute-bound parallel caching) vs Decode Phase (memory-bandwidth-bound cache reading).
4. The exact mathematical formula for calculating KV Cache VRAM consumption.
5. Modern architectural optimizations: Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and PagedAttention (vLLM).
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The KV Cache Data Flow & Prefill-Decode Disaggregation", divider="rainbow")

st.markdown("""
Self-attention decomposes into Queries (`Q`), Keys (`K`), and Values (`V`). 
During generation, historical `K` and `V` vectors never change—only the new token's `Q` vector needs to attend to them.
""")

kv_blueprint_dot = """
digraph KVCacheBlueprint {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_cache {
        label = "Persistent GPU VRAM Buffer (KV Cache)";
        color = "#334155";
        fontcolor = "#94A3B8";
        CachedK [label="Cached Keys (Past Tokens)\\n[Layers, KV_Heads, Seq_Len-1, Head_Dim]", fillcolor="#0F172A"];
        CachedV [label="Cached Values (Past Tokens)\\n[Layers, KV_Heads, Seq_Len-1, Head_Dim]", fillcolor="#0F172A"];
    }

    subgraph cluster_step {
        label = "Time Step t (Generating 1 Token)";
        color = "#334155";
        fontcolor = "#94A3B8";
        NewToken [label="Single New Token x_t\\n[1, 1, Hidden_Dim]", fillcolor="#0369A1"];
        NewQ [label="Query Vector Q_t\\n[Q_Heads, 1, Head_Dim]", fillcolor="#0369A1"];
        NewK [label="New Key K_t\\n[KV_Heads, 1, Head_Dim]", fillcolor="#047857"];
        NewV [label="New Value V_t\\n[KV_Heads, 1, Head_Dim]", fillcolor="#047857"];
        
        NewToken -> NewQ;
        NewToken -> NewK;
        NewToken -> NewV;
    }

    subgraph cluster_attention {
        label = "O(1) Step Attention Computation";
        color = "#334155";
        fontcolor = "#94A3B8";
        ConcatK [label="Concat K = [Cached_K ; K_t]\\n[Seq_Len, Head_Dim]", fillcolor="#1E293B"];
        ConcatV [label="Concat V = [Cached_V ; V_t]\\n[Seq_Len, Head_Dim]", fillcolor="#1E293B"];
        AttnScore [label="Score = Softmax(Q_t @ Concat_K^T)\\nShape: [1, Seq_Len]", fillcolor="#B45309"];
        AttnOut [label="Output = Score @ Concat_V\\nShape: [1, Head_Dim]", fillcolor="#7C3AED"];
        
        ConcatK -> AttnScore;
        AttnScore -> AttnOut;
        ConcatV -> AttnOut;
    }

    CachedK -> ConcatK;
    NewK -> ConcatK;
    CachedV -> ConcatV;
    NewV -> ConcatV;
    NewQ -> AttnScore;
    
    NewK -> CachedK [label="Persist in Cache"];
    NewV -> CachedV [label="Persist in Cache"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(kv_blueprint_dot, use_container_width=True)

st.markdown("""
### The Fundamental Insight
* At step `t`, we compute `Q_t`, `K_t`, and `V_t` **only for the single new incoming token**.
* `K_t` and `V_t` are appended to the pre-existing KV Cache in GPU memory.
* The query `Q_t` attends across the entire concatenated `Concat_K` and `Concat_V` buffer.
* We **never recompute `K` and `V` for past tokens**!
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - WHAT IS KV CACHE?
# ==============================================================================
st.header("2. What is KV Cache?", divider="rainbow")

st.markdown("""
In a standard Causal Multi-Head Self-Attention layer:
```text
Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k)) @ V
```
Where `Q = X @ W_q`, `K = X @ W_k`, and `V = X @ W_v`.

Because modern autoregressive models use a **causal mask** (lower-triangular attention mask), 
a token at position `i` only attends to tokens at positions `<= i`.
This means that as new tokens are added at position `t > i`:
1. The Key vector `K_i` for token `i` **never changes**.
2. The Value vector `V_i` for token `i` **never changes**.
3. Therefore, calculating `K_i` and `V_i` repeatedly on every future token is completely redundant!

The **KV Cache** allocates a dedicated memory buffer in GPU VRAM to store these `K` and `V` activation tensors 
for all past tokens across every transformer layer.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - WHY KV CACHE MATTERS
# ==============================================================================
st.header("3. Why KV Cache Matters: O(N) vs. O(N^2) Complexity", divider="rainbow")

st.markdown("""
The performance impact of the KV Cache is the difference between sub-second interactive AI and complete computational paralysis.
""")

col_cmp1, col_cmp2 = st.columns(2)
with col_cmp1:
    st.subheader("Without KV Cache: Quadratic O(N^2) Degradation")
    st.markdown("""
    * **Step 1**: Run forward pass on 100 prompt tokens -> emit token 101.
    * **Step 2**: Re-run forward pass on all 101 tokens -> emit token 102.
    * **Step 3**: Re-run forward pass on all 102 tokens -> emit token 103.
    * **Total Tokens Processed**:
      ```text
      Sum_{i=P}^{P+G} i = O(G * P + G^2 / 2)
      ```
    * For a 2,000-token prompt generating 1,000 tokens: **Over 2.5 Million token forward passes!** Latency explodes with every new word.
    """)

with col_cmp2:
    st.subheader("With KV Cache: Linear O(N) Total Cost")
    st.markdown("""
    * **Prefill**: Run 1 parallel forward pass on 100 prompt tokens -> cache all 100 K and V tensors.
    * **Step 1**: Run forward pass on **1 token** -> append KV -> emit token 101.
    * **Step 2**: Run forward pass on **1 token** -> append KV -> emit token 102.
    * **Total Tokens Processed**: Exactly `P + G` token passes (**3,000 token passes vs 2.5 Million**)!
    * Generation speed remains nearly constant per token.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - PREFILL PHASE
# ==============================================================================
st.header("4. Prefill Phase (Compute-Bound)", divider="rainbow")

st.markdown("""
The **Prefill Phase** (also known as the Prompt Phase or Context Phase) occurs at step 0 of inference.
""")

col_prf1, col_prf2 = st.columns(2)
with col_prf1:
    st.markdown("#### Hardware & Compute Characteristics")
    st.markdown("""
    * **High Parallelism**: All `N` prompt tokens are fed into the model simultaneously as a 2D matrix `[Batch, N, Hidden_Dim]`.
    * **Compute-Bound (FLOPS)**: Tensor cores execute large Matrix-Matrix Multiplications (GEMM), achieving near-peak hardware utilization (TFLOPS).
    * **Metrics**: Governs **Time-To-First-Token (TTFT)**.
    """)

with col_prf2:
    st.markdown("#### The Cache Population Step")
    st.markdown("""
    * Every transformer block projects `K = Prompt @ W_k` and `V = Prompt @ W_v`.
    * These matrices of shape `[Batch, KV_Heads, N, Head_Dim]` are written to GPU memory as the initial KV Cache buffer.
    * The logits of the final prompt token `N` are emitted to sample the first output token.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 5 - DECODE PHASE
# ==============================================================================
st.header("5. Decode Phase (Memory-Bound)", divider="rainbow")

st.markdown("""
The **Decode Phase** (Token Generation Phase) begins immediately after the first token is emitted.
""")

col_dec1, col_dec2 = st.columns(2)
with col_dec1:
    st.markdown("#### Hardware & Bandwidth Characteristics")
    st.markdown("""
    * **Sequential Bottleneck**: Exactly 1 token is processed per step: `[Batch, 1, Hidden_Dim]`.
    * **Matrix-Vector Multiplications (GEMV)**: Rather than dense matrix multiplication, tensor cores perform vector-matrix projections.
    * **Memory-Bandwidth Bound**: The GPU memory bus must read all model weights and the entire historical KV Cache for every single emitted token.
    * **Metrics**: Governs **Inter-Token Latency (ITL)** and streaming tokens/second.
    """)

with col_dec2:
    st.markdown("#### The KV Cache Read Penalty")
    st.markdown("""
    * As sequence length grows from 100 to 10,000 tokens, the amount of data read from the KV Cache increases by **100x**!
    * At long context lengths (32k+), **reading the KV Cache from memory consumes more bandwidth than reading the model weights themselves**!
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - KV-CACHE MEMORY
# ==============================================================================
st.header("6. KV-Cache Memory Calculation", divider="rainbow")

st.markdown("""
Estimating the exact VRAM required by the KV Cache is the most critical calculation in AI capacity planning.
""")

st.markdown("""
#### The Exact KV Cache Memory Equation
```text
KV_Cache_Bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * bytes_per_element
```
Where:
* **`2`**: Represents both Key and Value tensors.
* **`n_layers`**: Total transformer blocks (e.g. 32 in Llama-3-8B, 80 in Llama-3-70B).
* **`n_kv_heads`**: Number of Key/Value attention heads (depends on MHA vs GQA).
* **`d_head`**: Head dimension (typically `128`).
* **`seq_len`**: Total sequence length (`Prompt_Tokens + Generated_Tokens`).
* **`batch_size`**: Number of concurrent requests.
* **`bytes_per_element`**: 2 bytes for FP16 / BF16 (1 byte for FP8).
""")

st.markdown("""
#### Concrete Example: Llama-3-70B (80 Layers, 8 KV Heads, d_head=128, BF16)
```text
Bytes_per_token = 2 * 80 * 8 * 128 * 2 = 327,680 bytes = ~320 KB per token
```
* For **1 request** with 8,192 tokens: `320 KB * 8,192 = 2.62 GB` of VRAM.
* For **32 concurrent requests** with 8,192 tokens: `2.62 GB * 32 =` **`83.8 GB of VRAM`**!
* **The KV Cache consumes more GPU memory than the entire 70B model itself in 4-bit!**
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - KV-CACHE OPTIMIZATION
# ==============================================================================
st.header("7. KV-Cache Optimization (GQA, MQA, & PagedAttention)", divider="rainbow")

st.markdown("""
To prevent the KV Cache from overwhelming GPU memory, modern LLM architectures employ three breakthrough innovations:
""")

opt_col1, opt_col2, opt_col3 = st.columns(3)

with opt_col1:
    st.subheader("1. GQA & MQA")
    st.markdown("""
    * **Multi-Head Attention (MHA)**: 1 KV head per Query head (`n_kv = n_q`). Huge memory waste.
    * **Multi-Query Attention (MQA)**: 1 single KV head shared across all Query heads. Reduces KV cache by 32x!
    * **Grouped-Query Attention (GQA)**: 1 KV head shared per group of 4 or 8 Query heads (Llama-3, Mistral). **Reduces KV Cache by 8x** with zero accuracy loss!
    """)

with opt_col2:
    st.subheader("2. PagedAttention (vLLM)")
    st.markdown("""
    * **The Problem**: Naive memory allocation pre-allocates contiguous virtual memory for `max_seq_len` (e.g. 4096), causing **60% to 80% memory waste via internal fragmentation**.
    * **The Solution**: Inspired by OS virtual memory paging, PagedAttention stores KV tensors in non-contiguous physical memory blocks (e.g. 16 tokens/block).
    * **Result**: Eliminates 100% of memory fragmentation, boosting server throughput by **2x to 4x**!
    """)

with opt_col3:
    st.subheader("3. KV Cache Quantization")
    st.markdown("""
    * Compresses cached Key and Value vectors from 16-bit (BF16) down to **8-bit (FP8)** or **4-bit (INT4)** on the fly.
    * **Result**: Instantly cuts KV Cache VRAM consumption in half (`2x`), allowing doubling concurrent batch sizes with under 0.2 perplexity loss.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End KV Cache Tensor Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates exact KV Cache memory allocation, 
simulated prefill and decode stepping, and GQA vs MHA accounting.
""")

st.code("""
import numpy as np
import pandas as pd

class ProductionKVCacheEngine:
    \"\"\"
    Production-grade educational KV Cache simulation engine.
    \"\"\"
    def __init__(self, num_layers=32, num_q_heads=32, num_kv_heads=8, head_dim=128, bytes_per_element=2):
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bytes_per_element = bytes_per_element

    def calculate_kv_memory_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        \"\"\"2 * layers * kv_heads * head_dim * seq_len * batch_size * bytes_per_element\"\"\"
        bytes_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element
        return bytes_per_token * seq_len * batch_size

    def simulate_cache_lifecycle(self, prompt_len: int, gen_tokens: int) -> pd.DataFrame:
        records = []
        # Prefill
        p_bytes = self.calculate_kv_memory_bytes(prompt_len, batch_size=1)
        records.append({"Step": "Prefill", "Length": prompt_len, "KV_MB": p_bytes / (1024**2)})
        
        # Decode steps
        curr = prompt_len
        for s in range(1, gen_tokens + 1):
            curr += 1
            d_bytes = self.calculate_kv_memory_bytes(curr, batch_size=1)
            records.append({"Step": f"Decode {s}", "Length": curr, "KV_MB": d_bytes / (1024**2)})
        return pd.DataFrame(records)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: KV Cache Memory Calculator & Prefill/Decode Step Simulator", divider="rainbow")

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("Model Architecture & Concurrency Sizer")
    model_choice = st.selectbox(
        "Target Foundation Model",
        ["Llama-3-8B (32 Layers, 8 KV Heads, GQA)", "Llama-3-70B (80 Layers, 8 KV Heads, GQA)", "Legacy 7B (32 Layers, 32 KV Heads, MHA)"]
    )
    if "Llama-3-8B" in model_choice:
        n_lay, n_qh, n_kvh = 32, 32, 8
    elif "Llama-3-70B" in model_choice:
        n_lay, n_qh, n_kvh = 80, 64, 8
    else:
        n_lay, n_qh, n_kvh = 32, 32, 32

    user_seq_len = st.select_slider("Context Length (Tokens)", options=[512, 1024, 2048, 4096, 8192, 16384, 32768], value=4096)
    user_batch_size = st.slider("Concurrent Batch Size", min_value=1, max_value=64, value=8)
    kv_prec = st.radio("KV Cache Precision", ["16-bit (BF16 / FP16 - 2 bytes)", "8-bit (FP8 - 1 byte)"])
    b_per_elem = 2 if "16-bit" in kv_prec else 1

    sim_engine = ProductionKVCacheEngine(
        num_layers=n_lay,
        num_q_heads=n_qh,
        num_kv_heads=n_kvh,
        head_dim=128,
        bytes_per_element=b_per_elem
    )
    total_kv_bytes = sim_engine.calculate_kv_memory_bytes(user_seq_len, user_batch_size)
    total_kv_gb = sim_engine.format_memory_gb(total_kv_bytes)

with wb_c2:
    st.subheader("KV Cache VRAM Consumption")
    st.metric(
        "Total KV Cache VRAM Required",
        f"{total_kv_gb:.2f} GB",
        f"{user_batch_size} concurrent streams @ {user_seq_len} tokens"
    )

    if total_kv_gb > 80:
        st.error(f"🚨 Requires {int(np.ceil(total_kv_gb / 80))}x 80GB GPUs strictly for KV Cache storage!")
    elif total_kv_gb > 24:
        st.warning("⚠️ Exceeds consumer 24GB GPUs; requires enterprise datacenter GPU (A100/H100).")
    else:
        st.success("✅ Fits within a single 24GB GPU allocation.")

    st.markdown("##### Step-by-Step Lifecycle Simulation (Batch=1):")
    lifecycle_df = sim_engine.simulate_cache_lifecycle(prompt_len=256, gen_tokens=10)
    st.line_chart(lifecycle_df.set_index("Sequence Length")["KV Memory (MB)"])
    st.caption("Linear memory growth during decode step-by-step token appending.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: What is the KV Cache and why is it essential for autoregressive inference?"):
        st.write("""
        The KV Cache stores the Key and Value activation tensors computed during prior self-attention forward passes across all layers.
        Without it, generating token N requires re-running tokens 1 through N-1, causing O(N^2) quadratic computational complexity and unbearable latency.
        With the KV Cache, each decode step only computes Query, Key, and Value for the single new token, appending Key and Value to the cache and achieving O(1) step computation.
        """)
    with st.expander("Q2: How does Grouped Query Attention (GQA) reduce KV Cache memory?"):
        st.write("""
        In traditional Multi-Head Attention (MHA), every Query head has its own dedicated Key and Value head (e.g. 32 Q heads = 32 KV heads).
        Grouped Query Attention (GQA) shares a single Key and Value head across a group of Query heads (e.g. 32 Q heads grouped into 8 KV heads = 4:1 ratio).
        This directly reduces the memory footprint of the KV Cache by 4x to 8x without significant loss in downstream model reasoning accuracy.
        """)
    with st.expander("Q3: What problem does PagedAttention (vLLM) solve in production serving?"):
        st.write("""
        Traditional inference engines pre-allocate a contiguous memory block for the maximum possible sequence length for each request.
        Because requests have unpredictable lengths and arrive dynamically, up to 60-80% of VRAM is wasted through internal and external memory fragmentation.
        PagedAttention partitions the KV Cache into fixed-size physical blocks (e.g. 16 tokens), mapping virtual sequence tokens to non-contiguous physical memory blocks.
        This eliminates fragmentation, allowing 2x-4x more concurrent requests to fit in GPU VRAM.
        """)

with q2:
    with st.expander("Q4: Why does long-context inference cause KV Cache memory to exceed model weights?"):
        st.write("""
        Model weights are fixed in size (e.g. 70B in 4-bit is ~35GB).
        The KV Cache scales linearly with both sequence length and batch size: `Bytes = 2 * L * H_kv * d * seq_len * batch_size * 2`.
        At 16k context with batch size 32 on a 70B model, the KV Cache requires over 160 GB of memory—more than 4x the weight of the quantized model itself!
        """)
    with st.expander("Q5: What is Chunked Prefill and why is it used in high-throughput engines?"):
        st.write("""
        When a huge prompt (e.g. 32k tokens) enters a serving system, executing the entire prefill at once monopolizes the GPU for seconds, causing massive latency spikes (jitter) for ongoing decode streams.
        Chunked Prefill breaks large prompts into smaller chunks (e.g. 512 tokens), interleaving prompt prefill chunks with ongoing token decode steps to preserve consistent streaming SLAs.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Failing to Account for KV Cache in Sizing Capacity**
    * **Failure**: Calculating that a 70B 4-bit model fits in 40GB VRAM and concluding an A100 40GB is sufficient for production.
    * **Impact**: The container boots up, but crashes with CUDA OOM on the first concurrent multi-user request as the KV Cache exceeds remaining memory.
    * **Remedy**: Always size total GPU VRAM as: `Weight_Memory + (Peak_Batch_Size * KV_Cache_per_Stream) + Activation_Headroom`.
    """)
    st.error("""
    **🚨 Production Trap 2: Disabling use_cache in Transformers Generation**
    * **Failure**: Accidentally setting `use_cache=False` in `model.generate()`.
    * **Impact**: Latency drops by 50x; every new token re-runs the entire prompt through all layers.
    * **Remedy**: Ensure `model.config.use_cache = True` during inference.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Memory Fragmentation Without PagedAttention**
    * **Failure**: Running high-concurrency production endpoints using raw PyTorch loops without vLLM or Hugging Face TGI.
    * **Impact**: GPU OOM errors occur at only 30% memory utilization due to memory allocation fragmentation.
    * **Remedy**: Deploy with PagedAttention-enabled engines (vLLM / SGLang / TensorRT-LLM).
    """)
    st.warning("""
    **⚠️ Production Trap 4: Excessive Context Window Allocation**
    * **Failure**: Defaulting all user requests to allocate 128k context buffers.
    * **Impact**: Severely throttles maximum concurrent requests per server.
    * **Remedy**: Enforce dynamic sliding-window allocation or chunked context limits based on actual user query length.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: KV Cache Complexity"):
        st.write("What is the per-step token generation complexity with vs without KV Cache?")
        st.info("With KV Cache: O(1) step complexity. Without KV Cache: O(N) step complexity.")
    with st.expander("⚡ Flashcard 2: GQA KV Cache Reduction"):
        st.write("How much does Grouped Query Attention (GQA 8:1) reduce KV Cache memory compared to MHA?")
        st.info("Reduces KV Cache memory by 8x.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: KV Cache Memory Formula"):
        st.write("What are the key terms in the KV Cache memory equation?")
        st.info("`2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_element`.")
    with st.expander("⚡ Flashcard 4: PagedAttention Creator"):
        st.write("What serving library introduced PagedAttention?")
        st.info("vLLM (Kwon et al., UC Berkeley).")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of KV Cache before advancing to Chapter 17 (Quantization & Optimization):
""")

chk1 = st.checkbox("I understand why the KV Cache eliminates quadratic O(N^2) computational re-execution.", value=True)
chk2 = st.checkbox("I can calculate the exact byte footprint of the KV Cache given model hyperparameters.", value=True)
chk3 = st.checkbox("I understand the difference between compute-bound Prefill and memory-bandwidth-bound Decode.", value=True)
chk4 = st.checkbox("I know how Grouped Query Attention (GQA) cuts KV Cache memory by 8x.", value=True)
chk5 = st.checkbox("I understand how PagedAttention eliminates internal memory fragmentation in vLLM.", value=True)
chk6 = st.checkbox("I know why the KV Cache can exceed the total size of model weights at long contexts.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! Chapter 16: LLM Inference and Text Generation is fully completed.")
