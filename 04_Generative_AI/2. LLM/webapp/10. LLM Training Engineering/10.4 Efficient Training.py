"""
10.4 Efficient Training
=======================
Interactive Master Lesson exploring Batch Processing, Gradient Accumulation,
Activation Checkpointing, Mixed Precision (BF16/FP16), and 3D Distributed Parallelism (DP, TP, PP, FSDP).
"""

from typing import Dict, List, Tuple
import graphviz
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="10.4 Efficient Training - LLM Architecture",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS for aesthetic typography & dark theme styling
st.markdown(
    """
<style>
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .callout-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TOP NAVIGATION / HERO
# -----------------------------------------------------------------------------
st.title("⚡ 10.4 Efficient Training")
st.caption(
    "Chapter 10: LLM Training Engineering | Module 4 of 4"
)

st.markdown(
    """
Scaling Large Language Models to hundreds of billions of parameters requires conquering
the memory and compute constraints of physical hardware. Through the synergistic combination
of **Activation Checkpointing**, **Gradient Accumulation**, **BF16 Mixed Precision**,
and **3D Distributed Parallelism** (Data, Tensor, Pipeline, and FSDP/ZeRO), engineering teams
achieve near-linear scaling across clusters of tens of thousands of GPUs.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE EFFICIENCY TRILEMMA
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Efficiency Trilemma", divider="rainbow")

st.markdown(
    """
### The 3D Parallelism Coordination Matrix
A single 70B or 405B model cannot fit onto a single GPU's VRAM. Training requires sharding across three orthogonal dimensions:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "efficient_training_parallelism",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("dp", "Data Parallelism (DP / FSDP / ZeRO)\\nShards dataset batches & optimizer state\\nInfiniBand All-Reduce", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("tp", "Tensor Parallelism (TP / Megatron)\\nSplits weight matrices intra-layer\\nHigh-speed NVLink (Intra-Node)", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("pp", "Pipeline Parallelism (PP)\\nSplits 80 layers across GPU stages\\nInter-node P2P communication", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("cp", "Context Parallelism (CP / Ring)\\nSplits 128k token sequences across GPUs\\nFlashAttention Ring All-to-All", fillcolor="#7C3AED", fontcolor="#F8FAFC")

arch_dot.edge("tp", "pp", label="Combined into 3D Parallelism")
arch_dot.edge("pp", "dp", label="Scaled across Supercluster")
arch_dot.edge("dp", "cp", label="Enables Long Contexts")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **FSDP (Fully Sharded Data Parallel) Dominance**: While Megatron 3D parallelism dominated early LLMs, PyTorch FSDP (ZeRO-3) has become the de-facto standard for training models up to 70B parameters on standard clusters, sharding weights, gradients, and optimizer states with minimal code overhead.
* **FP8 Hopper Acceleration**: Frontier training clusters leverage NVIDIA H100 FP8 Tensor Cores to nearly double effective GEMM throughput over BF16 while cutting activation memory in half.
"""
)

# Memory Component Breakdown Table
mem_breakdown_df = pd.DataFrame(
    {
        "Memory Component": ["Model Parameters (BF16)", "Gradients (BF16)", "Optimizer States (AdamW FP32)", "Intermediate Activations", "KV Cache (Decoding)"],
        "7B Model (Bytes)": ["14 GB", "14 GB", "84 GB", "~20 GB (w/o checkpointing)", "~1.5 GB per stream"],
        "70B Model (Bytes)": ["140 GB", "140 GB", "840 GB", "~120 GB", "~15 GB per stream"],
        "Reduction Strategy": ["FSDP / ZeRO-3 sharding", "ZeRO-2 gradient sharding", "ZeRO-1 optimizer sharding", "Activation checkpointing", "GQA, Quantization (FP8/INT4)"],
    }
)
st.dataframe(mem_breakdown_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: BATCH PROCESSING DYNAMICS
# -----------------------------------------------------------------------------
st.header("2. Batch Processing Dynamics (Micro-Batches vs. Global Batches)", divider="rainbow")

st.markdown(
    """
In foundation LLMs, batch size operates at two distinct granularities:
1. **Micro-Batch Size (`B_micro`)**: The number of sequences processed by a single GPU in a single forward/backward step. Typically `1` or `2` due to VRAM limits.
2. **Global Batch Size (`B_global`)**: The effective statistical batch size over which gradients are averaged before an optimizer step:
   ```text
   B_global = B_micro * Num_Data_Parallel_GPUs * Gradient_Accumulation_Steps
   ```

### The Throughput Trade-Off
* Small micro-batches underutilize GPU Tensor Cores (low arithmetic intensity).
* Large micro-batches trigger Out of Memory (OOM) crashes from activation memory.
* Engineering teams benchmark the largest possible `B_micro` that fits into VRAM, and scale `B_global` using gradient accumulation.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: GRADIENT ACCUMULATION MECHANICS
# -----------------------------------------------------------------------------
st.header("3. Gradient Accumulation Mechanics & VRAM Decoupling", divider="rainbow")

st.markdown(
    """
Gradient accumulation decouples the physical GPU memory limit from the mathematical batch size:
* Instead of communicating gradients on every micro-batch, PyTorch sums gradients locally in `param.grad` over `K` accumulation steps.
* Communication hooks (all-reduce) are suppressed for `K-1` iterations, saving valuable network bandwidth.
* On the `K`-th iteration, gradients are averaged by dividing by `K`, distributed across ranks, and stepped by the optimizer.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: ACTIVATION CHECKPOINTING
# -----------------------------------------------------------------------------
st.header("4. Activation Checkpointing (Gradient Checkpointing)", divider="rainbow")

st.markdown(
    """
During standard backpropagation, all intermediate activations computed during the forward pass
must be stored in GPU VRAM until the backward pass reaches that layer.
In an 80-layer model, stored activations can consume over **120 GB of memory**, causing instant OOM.

### The Selective Recomputation Trade-Off
Activation checkpointing divides the transformer stack into blocks:
1. **During Forward Pass**: Store activations only at block boundaries (e.g., input to each layer). Intermediate activations inside attention and MLP blocks are discarded immediately.
2. **During Backward Pass**: When backpropagating through layer `l`, re-run the forward pass for layer `l` on-the-fly using the cached input to re-generate the discarded activations right when needed.
* **Memory Savings**: **Up to 80% reduction** in activation memory.
* **Compute Cost**: Adds approximately **33% extra FLOPs** (one additional forward pass per layer).
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: MIXED PRECISION ACCELERATION ARCHITECTURES
# -----------------------------------------------------------------------------
st.header("5. Mixed Precision Acceleration Architectures (AMP Engine)", divider="rainbow")

st.markdown(
    """
Automatic Mixed Precision (AMP) dynamically casts operations to lower precision formats
to exploit dedicated hardware Tensor Cores:
* **GEMM Matrix Multiplications (Attention QK^T, Dense Projections)**: Executed in BF16 / FP8 at up to 989 TFLOPS.
* **Sensitive Reductions (RMSNorm, Softmax, Cross-Entropy Loss)**: Maintained in FP32 to prevent numerical underflow and precision loss.
* **Master Parameter Copies**: Kept in FP32 so minute optimizer updates (`lr * step approx 1e-6`) are not rounded to zero.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: BFLOAT16 (BF16) DEEP DIVE ON MODERN HARDWARE
# -----------------------------------------------------------------------------
st.header("6. Bfloat16 (BF16) Deep Dive on Modern Hardware", divider="rainbow")

st.markdown(
    """
Introduced by Google Brain and standardized on NVIDIA Ampere (A100) and Hopper (H100):
* **Structure**: 1 sign bit + **8 exponent bits** + 7 mantissa bits.
* **Dynamic Range**: Matches single-precision FP32 (`~1e-38 to 3.4e38`).
* **Why it Won**: Models can train in BF16 without any dynamic loss scaling or complex underflow protections. If a calculation does not overflow in FP32, it will not overflow in BF16.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: FP16 PRECISION & LEGACY LOSS SCALING OVERHEAD
# -----------------------------------------------------------------------------
st.header("7. FP16 Precision & The Legacy Loss Scaling Overhead", divider="rainbow")

st.markdown(
    """
In legacy architectures (NVIDIA Volta V100, Turing T4), FP16 was the only 16-bit hardware format:
* **Structure**: 1 sign bit + **5 exponent bits** + 10 mantissa bits.
* **Max Value**: `65,504`.
* **Min Positive Value**: `~6e-5`.
* **The Loss Scaling Tax**: Because backward gradients frequently fall below `6e-5`, training in FP16 requires dynamic loss scaling (`GradScaler`). If loss spikes occur, the scaler decreases, skipping gradient steps and causing compute wastage. Modern LLMs avoid FP16 in favor of BF16.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: DISTRIBUTED TRAINING PARADIGMS
# -----------------------------------------------------------------------------
st.header("8. Distributed Training Paradigms (3D Parallelism: DP, TP, PP, and FSDP)", divider="rainbow")

st.markdown(
    """
Modern pretraining clusters orchestrate four complementary parallelism axes:

### 1. Fully Sharded Data Parallelism (FSDP / ZeRO-3)
Shards model weights, gradients, and optimizer states across data-parallel ranks.
Before executing layer `l`, GPUs perform an all-gather to reconstruct layer `l` weights, compute the forward pass, and immediately discard the weights, maintaining minimal per-GPU VRAM usage.

### 2. Tensor Parallelism (TP, Shoeybi et al., Megatron-LM)
Splits individual weight matrices across GPUs within the same physical node:
* **Column-Parallel Linear**: Splits `W` by columns across 8 GPUs.
* **Row-Parallel Linear**: Splits `W` by rows, followed by an intra-node all-reduce over high-speed NVLink (900 GB/s on H100).

### 3. Pipeline Parallelism (PP)
Partitions layers sequentially across nodes (e.g. GPU 0 handles layers 1-20, GPU 1 handles layers 21-40).
Uses 1F1B (One Forward, One Backward) scheduling to minimize pipeline bubbles.

### 4. Context Parallelism (CP / Ring Attention)
Splits long sequences (e.g. 128k tokens) across GPUs, overlapping attention computation with Key-Value block communication rings.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE DISTRIBUTED TRAINING ENGINE
# -----------------------------------------------------------------------------
st.header("9. Production-Grade Distributed Training Engine (FSDP / DDP Pipeline)", divider="rainbow")

st.markdown(
    """
A complete PyTorch code template illustrating modern FSDP wrapping and activation checkpointing:
"""
)

code_snippet = """import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

def build_production_fsdp_model(
    model: nn.Module,
    transformer_layer_cls,
    device_id: int,
) -> FSDP:
    # 1. Configure BF16 Mixed Precision policy
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 2. Configure Auto-Wrap Policy per Transformer Block
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={transformer_layer_cls},
    )

    # 3. Apply Activation Checkpointing to each transformer block
    check_wrapper = lambda module: checkpoint_wrapper(
        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )

    # 4. Wrap model in FSDP (ZeRO-3 full sharding)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.device(f"cuda:{device_id}"),
        limit_all_gathers=True,
    )
    return fsdp_model
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE GPU VRAM & DISTRIBUTED SCALING WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive GPU VRAM Allocation & Distributed Scaling Workbench", divider="rainbow")

st.markdown(
    """
Calculate exact per-GPU VRAM consumption and parallel communication overhead across different model sizes
and distributed sharding strategies.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    model_size_b = st.select_slider("Model Parameter Count", options=[7, 13, 34, 70, 405], value=70)
    sharding_mode = st.selectbox("Parallelism Strategy", options=["Standard DDP (Replicated)", "ZeRO-1 (Sharded Opt)", "ZeRO-2 (Sharded Opt + Grad)", "FSDP / ZeRO-3 (Full Sharding)"], index=3)
with col_ctrl2:
    gpu_cluster_count = st.select_slider("Cluster GPU Count", options=[8, 64, 256, 512, 1024, 2048], value=64)
    act_checkpoint = st.checkbox("Enable Activation Checkpointing", value=True)

# Memory calculations (BF16 params = 2B, grads = 2B, AdamW opt = 12B)
param_bytes = model_size_b * 2e9
grad_bytes = model_size_b * 2e9
opt_bytes = model_size_b * 12e9

if sharding_mode == "Standard DDP (Replicated)":
    vram_param = param_bytes
    vram_grad = grad_bytes
    vram_opt = opt_bytes
elif sharding_mode == "ZeRO-1 (Sharded Opt)":
    vram_param = param_bytes
    vram_grad = grad_bytes
    vram_opt = opt_bytes / gpu_cluster_count
elif sharding_mode == "ZeRO-2 (Sharded Opt + Grad)":
    vram_param = param_bytes
    vram_grad = grad_bytes / gpu_cluster_count
    vram_opt = opt_bytes / gpu_cluster_count
elif sharding_mode == "FSDP / ZeRO-3 (Full Sharding)":
    vram_param = param_bytes / gpu_cluster_count
    vram_grad = grad_bytes / gpu_cluster_count
    vram_opt = opt_bytes / gpu_cluster_count

# Activation estimates (heuristic)
vram_act = (model_size_b * 0.15e9) if act_checkpoint else (model_size_b * 1.5e9)

total_per_gpu_gb = (vram_param + vram_grad + vram_opt + vram_act) / (1024 ** 3)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total VRAM per GPU", f"{total_per_gpu_gb:.2f} GB")
with col_m2:
    st.metric("H100 80GB Feasibility", "FITS ON GPU!" if total_per_gpu_gb < 80 else "OUT OF MEMORY (OOM)", delta_color="normal" if total_per_gpu_gb < 80 else "inverse")
with col_m3:
    st.metric("Memory Savings Factor", f"{((param_bytes + grad_bytes + opt_bytes) / (1024**3)) / max(0.1, total_per_gpu_gb):.1f}x vs Single GPU")

mem_df = pd.DataFrame(
    {
        "Component": ["Model Parameters", "Gradients", "Optimizer State (AdamW)", "Activations"],
        "VRAM Allocated (GB)": [
            vram_param / (1024 ** 3),
            vram_grad / (1024 ** 3),
            vram_opt / (1024 ** 3),
            vram_act / (1024 ** 3),
        ],
    }
)
st.subheader("Per-GPU Memory Footprint Breakdown")
st.dataframe(mem_df.round(2), use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Explain the difference between ZeRO-1, ZeRO-2, and ZeRO-3."):
    st.markdown(
        """
        **Answer:**
        * **ZeRO-1 (Optimizer State Sharding)**: Shards the 12 bytes/param AdamW state across GPUs. Reduces memory by 4x with zero communication overhead.
        * **ZeRO-2 (Gradient Sharding)**: Shards both optimizer states and gradients (14 bytes/param total). Each GPU reduces and stores only its assigned gradient slice.
        * **ZeRO-3 (Full Parameter Sharding / FSDP)**: Shards optimizer states, gradients, and model parameters (16 bytes/param total). Layers are gathered on-demand during forward/backward passes and discarded immediately, allowing arbitrary model sizes to fit across clusters.
        """
    )

with st.expander("Q2: Why does Activation Checkpointing add approximately 33% compute overhead?"):
    st.markdown(
        """
        **Answer:**
        In standard training, 1 forward pass + 1 backward pass = 3 FLOPs units (1 for forward, 2 for backward).
        With activation checkpointing, intermediate activations inside a block are recomputed during the backward pass by running a second forward pass through that block.
        Total compute becomes `1 (forward) + 1 (recomputed forward) + 2 (backward) = 4 units`.
        The ratio `4 / 3 = 1.333`, representing exactly a **33% compute increase**.
        """
    )

with st.expander("Q3: Why is Tensor Parallelism typically restricted to within a single physical node?"):
    st.markdown(
        """
        **Answer:**
        Tensor Parallelism splits weight matrices within individual linear layers, requiring **two all-reduce communications per transformer layer** (one after attention, one after MLP).
        In an 80-layer model, this requires 160 all-reduce communications per single forward-backward pass.
        This extreme communication frequency requires high-bandwidth NVLink interconnects (900 GB/s on H100). Running TP across nodes over InfiniBand (50-100 GB/s) causes severe network bottlenecks, stalling execution.
        """
    )

with st.expander("Q4: What is the 1F1B schedule in Pipeline Parallelism?"):
    st.markdown(
        """
        **Answer:**
        In naive pipeline parallelism, all micro-batches execute forward passes before executing backward passes, causing activation memory to accumulate linearly with the number of micro-batches.
        The **1F1B (One Forward, One Backward)** schedule alternates: after a brief warmup phase, each pipeline stage executes one backward pass for an earlier micro-batch for every forward pass of a new micro-batch, capping activation memory at the number of pipeline stages.
        """
    )

with st.expander("Q5: How does FlashAttention reduce memory access overhead?"):
    st.markdown(
        """
        **Answer:**
        Standard attention materializes the `N x N` attention score matrix in GPU High-Bandwidth Memory (HBM), incurring massive IO transfer bottlenecks.
        FlashAttention tiles Query, Key, and Value blocks into ultra-fast on-chip SRAM (228 KB per SM on H100). It computes Softmax online using running maximums and sums, never writing the intermediate `N x N` attention matrix to DRAM and reducing memory reads/writes by an order of magnitude.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Running Tensor Parallelism Across Ethernet / InfiniBand Nodes**
        *Root Cause*: Configuring `tp_size = 16` across two 8-GPU nodes connected via standard InfiniBand.
        *Production Fix*: GPU execution stalls on inter-node all-reduce latency. Always limit Tensor Parallelism to `tp_size <= 8` (within single NVLink nodes) and scale across nodes using FSDP, PP, or DP.
        """
    )
    st.warning(
        """
        **Trap 2: Excessive Pipeline Parallelism Bubbles**
        *Root Cause*: Setting too few micro-batches relative to pipeline depth `P`.
        *Production Fix*: The pipeline bubble fraction is `(P - 1) / (M + P - 1)`, where `M` is micro-batch count. Ensure `M >= 4 * P` to keep bubble idle time under 20%.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Memory Leaks from Custom Non-Reentrant Checkpointing**
        *Root Cause*: Using legacy PyTorch reentrant checkpointing with autograd hooks.
        *Production Fix*: Always configure modern non-reentrant activation checkpointing (`checkpoint_impl=CheckpointImpl.NO_REENTRANT`) in PyTorch.
        """
    )
    st.warning(
        """
        **Trap 4: Unbalanced Micro-Batch Shapes in Heterogeneous GPUs**
        *Root Cause*: Sharding uneven sequence lengths across data-parallel ranks, causing ranks with shorter sequences to idle waiting at all-reduce barriers.
        *Production Fix*: Use sequence packing to ensure every micro-batch contains the exact same number of valid tokens.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Activation Checkpointing Compute Overhead"):
        st.write("How much extra compute does activation checkpointing add?")
        st.info("Approximately 33% extra FLOPs (1 additional forward pass per block).")

    with st.expander("⚡ Flashcard 2: NVLink Necessity"):
        st.write("Which parallelism strategy strictly requires NVLink interconnects?")
        st.info("Tensor Parallelism (due to high-frequency intra-layer all-reduce steps).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: FSDP Full Sharding Scope"):
        st.write("What three components does FSDP (ZeRO-3) shard across GPUs?")
        st.info("Parameters, Gradients, and Optimizer States.")

    with st.expander("⚡ Flashcard 4: 1F1B Pipeline Schedule"):
        st.write("What is the primary benefit of the 1F1B pipeline schedule?")
        st.info("Caps activation memory at the number of pipeline stages.")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the memory breakdown of large models across parameters, gradients, optimizer states, and activations.",
    "I can explain the 33% compute vs. 80% memory trade-off in Activation Checkpointing.",
    "I understand how Gradient Accumulation decouples physical micro-batch limits from statistical global batch sizes.",
    "I can distinguish between ZeRO-1, ZeRO-2, and FSDP (ZeRO-3) sharding strategies.",
    "I know why Tensor Parallelism is restricted to intra-node NVLink while FSDP scales across nodes.",
    "I understand how FlashAttention computes attention in GPU SRAM without materializing the N x N attention matrix.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Efficient LLM Training, Activation Checkpointing, and Distributed Parallelism Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
