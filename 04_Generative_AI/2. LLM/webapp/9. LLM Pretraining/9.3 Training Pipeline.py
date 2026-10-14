"""
9.3 Training Pipeline
=====================
Interactive Master Lesson exploring Token Ingestion, Micro-Batching, Forward Passes,
Loss Computation, Backpropagation, Gradient Accumulation, and Optimizer Weight Updates in Modern LLMs.
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
    page_title="9.3 Training Pipeline - LLM Architecture",
    page_icon="⚙️",
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
st.title("⚙️ 9.3 Training Pipeline")
st.caption(
    "Chapter 9: LLM Pretraining | Module 3 of 4"
)

st.markdown(
    """
The pretraining loop is the high-performance engine of foundation models.
Executing across thousands of GPUs continuously for months, every micro-step involves
coordinated data streaming, forward activation passes, causal cross-entropy loss computation,
reverse-mode automatic differentiation, inter-GPU gradient all-reduce communication, and
optimizer moment updates in **AdamW**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & END-TO-END PIPELINE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The End-to-End Execution Pipeline", divider="rainbow")

st.markdown(
    """
### The Pretraining Iteration Lifecycle
Every training step executes a tightly synchronized sequence of computational and communication phases.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "training_pipeline_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("batch", "Data Ingestion\\nToken Batches [B, T]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("fwd", "Forward Pass\\nAttention + MLP", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("loss", "Loss Computation\\nCross-Entropy", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("bwd", "Backpropagation\\ndL / dW", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("allreduce", "All-Reduce\\nInfiniBand Sum", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("opt", "AdamW Step\\nm_t, v_t Updates", fillcolor="#1E293B", fontcolor="#F8FAFC")

arch_dot.edge("batch", "fwd")
arch_dot.edge("fwd", "loss", label="Logits")
arch_dot.edge("loss", "bwd", label="Scalar Loss")
arch_dot.edge("bwd", "allreduce", label="Gradients")
arch_dot.edge("allreduce", "opt", label="Averaged Grads")
arch_dot.edge("opt", "fwd", label="New Weights")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Gradient Accumulation at Mega-Batch Scale**: Frontier LLMs train with global batch sizes of **4 million to 16 million tokens per step**. To fit within GPU memory, each GPU processes a small micro-batch (e.g. 2 sequences of 8,192 tokens = 16,384 tokens) and accumulates gradients locally across multiple forward-backward passes before triggering an all-reduce communication step.
* **Mixed Precision (BF16 / FP8)**: Forward and backward computations are executed in 16-bit Brain Floating Point (BF16) or Hopper FP8 Tensor Cores, while the master copy of optimizer weights is maintained in 32-bit FP32 for numerical stability.
"""
)

# Training Stage Profiling Table
profile_df = pd.DataFrame(
    {
        "Pipeline Stage": ["Data Streaming", "Forward Pass", "Loss Evaluation", "Backward Pass", "Gradient All-Reduce", "Optimizer Step"],
        "Computational Nature": ["IO / Network Bound", "Compute Bound (FLOPs)", "Memory Bandwidth Bound", "Compute Bound (2x FWD FLOPs)", "Interconnect Bandwidth (NVLink)", "Memory Bandwidth Bound"],
        "Time Share (%)": ["< 1% (Asynchronous)", "~25% - 30%", "< 2%", "~50% - 60%", "~5% - 10%", "~3% - 5%"],
        "Precision": ["INT32 / UINT16 IDs", "BF16 / FP8", "FP32 (LogSumExp)", "BF16 / FP8", "FP32 / BF16 Gradients", "FP32 Master Weights"],
    }
)
st.dataframe(profile_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: DATASET TO TOKENIZER INGESTION ARCHITECTURE
# -----------------------------------------------------------------------------
st.header("2. Dataset to Tokenizer Ingestion Architecture", divider="rainbow")

st.markdown(
    """
Feeding thousands of GPUs without stalling requires an asynchronous data loader pipeline:
1. **Sharded Disk Storage**: The multi-trillion token binary dataset is partitioned into thousands of equal-sized chunks (`shard_00000.bin` to `shard_09999.bin`).
2. **Distributed Data Sampler**: Each GPU rank receives a unique deterministic slice of shards based on its global `rank` and `world_size`.
3. **Double Buffering & Prefetching**: While the GPU executes matrix multiplications for step `t`, CPU worker threads asynchronously prefetch and uncompress tokens for step `t+1` directly into pinned host memory.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: TOKEN BATCHES, MICRO-BATCHES & GLOBAL BATCHES
# -----------------------------------------------------------------------------
st.header("3. Token Batches, Micro-Batches & Global Batch Construction", divider="rainbow")

st.markdown(
    """
Modern LLMs decouple GPU memory limits from the mathematical global batch size:

```text
Global_Batch_Size_Tokens = Micro_Batch_Size * Seq_Length * Data_Parallel_Ranks * Gradient_Accumulation_Steps
```

### Concrete Pretraining Example (Llama 3 70B)
* Sequence length: `8,192` tokens.
* Micro-batch size per GPU: `1` sequence.
* Number of Data Parallel GPUs: `512` ranks.
* Gradient accumulation steps: `4` steps.
* **Global Tokens per Step**: `1 * 8,192 * 512 * 4 = 16,777,216 tokens (16.7M tokens/step)`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: FORWARD PASS EXECUTION & FLASHATTENTION TILING
# -----------------------------------------------------------------------------
st.header("4. Forward Pass Execution & FlashAttention Tiling", divider="rainbow")

st.markdown(
    """
During the forward pass, tokens flow sequentially through `L` transformer blocks:
1. **Embedding Lookup**: Gathers initial hidden vectors `h_0 = Embedding(x)`.
2. **Layer Norm (RMSNorm)**: Normalizes hidden states across channels.
3. **Grouped-Query Attention (GQA)**: Computes FlashAttention in on-chip SRAM, updating `h = h + Attn(RMSNorm(h))`.
4. **Feed-Forward MLP (SwiGLU)**: Expands hidden dimension to `8/3 * d_model` with gated activation: `h = h + MLP(RMSNorm(h))`.
5. **Final Unembedding**: Projects `h_L` to logits `z in R^[B, T, V]`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: LOSS CALCULATION & CAUSAL SHIFT MECHANICS
# -----------------------------------------------------------------------------
st.header("5. Loss Calculation & Causal Shift Mechanics", divider="rainbow")

st.markdown(
    """
Cross-entropy loss measures how well logits at step `t` predict the true token at step `t+1`:

```text
Shift_Logits  = Logits[:, 0 : T-1, :]
Shift_Targets = Tokens[:, 1 : T, :]
Loss = F.cross_entropy(Shift_Logits.view(-1, V), Shift_Targets.view(-1), ignore_index=-100)
```

Loss computation is kept in **FP32** precision to ensure the Log-Sum-Exp partition function does not saturate.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: BACKPROPAGATION & REVERSE-MODE AUTODIFF
# -----------------------------------------------------------------------------
st.header("6. Backpropagation & Reverse-Mode Automatic Differentiation", divider="rainbow")

st.markdown(
    """
Backpropagation traverses the computational graph in reverse:
* **The 2x FLOP Rule**: The backward pass requires **twice as many floating-point operations** as the forward pass (one GEMM to compute gradients with respect to weights `dL / dW`, and one GEMM to propagate gradients to the previous activation layer `dL / dX`).
* **Activation Checkpointing (Gradient Checkpointing)**: Storing all intermediate activations across 80 layers would require hundreds of gigabytes of VRAM. Instead, models discard intermediate activations and recompute them on-the-fly during the backward pass, trading 33% extra compute for a 5x memory reduction.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: GRADIENT COMPUTATION, ACCUMULATION & REDUCTION
# -----------------------------------------------------------------------------
st.header("7. Gradient Computation, Accumulation & Reduction", divider="rainbow")

st.markdown(
    """
### Gradient Accumulation Mechanics
For `accumulation_steps = K`:
1. For steps `1` through `K-1`: Compute loss and call `loss.backward()`. Gradients are summed into `param.grad` without triggering optimizer updates or distributed communication.
2. At step `K`: Normalize gradients by `1 / K`.
3. **Distributed All-Reduce (Ring / Tree)**: GPUs exchange gradient tensors via NVLink / InfiniBand, summing them so every rank has identical global average gradients.
4. **Gradient Clipping**: Scale gradients if global norm exceeds threshold:
   ```text
   if ||g||_2 > max_norm:
       g = g * (max_norm / ||g||_2)
   ```
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: OPTIMIZER STEP DYNAMICS (ADAMW MOMENTS)
# -----------------------------------------------------------------------------
st.header("8. Optimizer Step Dynamics (AdamW First and Second Moments)", divider="rainbow")

st.markdown(
    """
AdamW updates parameters by tracking exponential moving averages of gradients:

```text
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t          (First Moment: Direction)
v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ^ 2)    (Second Moment: Variance)
m_hat = m_t / (1 - beta_1 ^ t)
v_hat = v_t / (1 - beta_2 ^ t)
Delta_W = - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * W)
```
* **AdamW Memory Footprint**: For every parameter, AdamW stores 4 bytes for `m_t`, 4 bytes for `v_t`, and 4 bytes for FP32 master weights = **12 bytes per parameter** in optimizer state alone!
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: WEIGHT UPDATES & PARAMETER CHECKPOINTING
# -----------------------------------------------------------------------------
st.header("9. Weight Updates & Parameter Checkpointing", divider="rainbow")

st.markdown(
    """
Once `Delta_W` is applied to the master weights, new BF16 weights are copied back into the model.
* **Checkpointing Frequency**: Checkpoints are written to distributed network storage (Lustre / Ceph) every 500 to 2,000 steps.
* **Asynchronous Checkpointing**: Saving a 140GB model snapshot takes 30 seconds if synchronous, halting training. Modern engines (PyTorch Distributed Checkpoint) dump tensors to host RAM asynchronously while GPUs resume the next step immediately.
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE PRETRAINING LOOP IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Pretraining Loop Implementation (PyTorch DDP)", divider="rainbow")

st.markdown(
    """
A complete, standalone training step script with gradient accumulation, mixed precision, and clipping:
"""
)

code_snippet = """import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def pretraining_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_tokens: torch.Tensor,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> Dict[str, float]:
    # batch_tokens shape: [micro_batch_size, seq_len]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    
    # Split micro-batch if simulating gradient accumulation
    micro_batches = torch.chunk(batch_tokens, grad_accum_steps, dim=0)
    
    for step_idx, micro_batch in enumerate(micro_batches):
        micro_batch = micro_batch.to(device)
        input_ids = micro_batch[:, :-1].contiguous()
        target_ids = micro_batch[:, 1:].contiguous()

        # BF16 Autocast forward pass
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # Backward pass (accumulate gradients)
        loss.backward()
        total_loss += loss.item()

    # Gradient clipping
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Optimizer weight update
    optimizer.step()

    return {
        "step_loss": total_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE TRAINING STEP & ACCUMULATION WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Training Step & Gradient Accumulation Workbench", divider="rainbow")

st.markdown(
    """
Calculate global pretraining throughput, tokens per step, and memory consumption across varying cluster configurations.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    gpu_count = st.select_slider("Number of GPUs in Cluster", options=[8, 64, 256, 512, 1024, 2048, 4096, 16384], value=512)
    seq_len = st.select_slider("Sequence Length", options=[2048, 4096, 8192, 16384], value=8192)
with col_ctrl2:
    micro_batch = st.slider("Micro-Batch per GPU", min_value=1, max_value=8, value=1)
    accum_steps = st.slider("Gradient Accumulation Steps", min_value=1, max_value=16, value=4)

# Calculations
tokens_per_gpu_step = micro_batch * seq_len
global_batch_tokens = tokens_per_gpu_step * gpu_count * accum_steps
target_15t_steps = 15_000_000_000_000 / global_batch_tokens

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Global Tokens per Step", f"{global_batch_tokens / 1e6:.2f} Million")
with col_m2:
    st.metric("Steps for 15T Tokens", f"{int(target_15t_steps):,} Steps")
with col_m3:
    step_time_sec = 1.2  # Typical H100 iteration time
    total_days = (target_15t_steps * step_time_sec) / (3600 * 24)
    st.metric("Estimated Training Days", f"{total_days:.1f} Days")

config_summary_df = pd.DataFrame(
    {
        "Parameter": ["Cluster GPUs", "Sequence Length", "Tokens / Micro-Batch", "Gradient Accumulation", "Global Batch Tokens", "Target 15T Epoch Duration"],
        "Value": [f"{gpu_count} GPUs", f"{seq_len} tokens", f"{tokens_per_gpu_step:,} tokens", f"{accum_steps} iterations", f"{global_batch_tokens:,} tokens", f"{total_days:.1f} continuous cluster days"],
    }
)
st.subheader("Training Configuration Blueprint")
st.dataframe(config_summary_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is Gradient Accumulation necessary in large-scale LLM training?"):
    st.markdown(
        """
        **Answer:**
        Large language models require massive global batch sizes (e.g., 4M to 16M tokens) for gradient descent stability.
        However, individual GPU VRAM is strictly limited (80GB on H100). A single GPU can only fit 1 or 2 sequences of 8,192 tokens in its memory without running out of memory (OOM).
        Gradient accumulation computes loss and gradients sequentially over smaller micro-batches, accumulating gradients in `param.grad` before executing a single optimizer update, achieving massive batch training without hardware OOM.
        """
    )

with st.expander("Q2: Why does the backward pass take approximately twice the compute (FLOPs) of the forward pass?"):
    st.markdown(
        """
        **Answer:**
        For any linear layer `Y = X * W`:
        1. **Forward Pass**: 1 matrix multiplication (`X * W`).
        2. **Backward Pass**: Requires 2 distinct matrix multiplications:
           * `dL / dW = X^T * (dL / dY)` to update parameter gradients.
           * `dL / dX = (dL / dY) * W^T` to propagate error backward to the previous layer.
        Hence, backward compute is naturally `2x` forward compute.
        """
    )

with st.expander("Q3: What is the purpose of Gradient Clipping and what happens if it is omitted?"):
    st.markdown(
        """
        **Answer:**
        Gradient clipping rescales the global gradient vector if its L2 norm exceeds a threshold (typically `1.0`).
        Without gradient clipping, encountering an anomalous training document or extreme activation produces massive gradient spikes. A huge step in AdamW pushes weights into regions where the loss explodes or produces `NaN` activations, ruining weeks of training progress.
        """
    )

with st.expander("Q4: Why does optimizer.zero_grad(set_to_none=True) improve performance over zeroing tensors?"):
    st.markdown(
        """
        **Answer:**
        Calling `.zero_grad()` writes floating-point zeros to all memory addresses allocated for `.grad`.
        Passing `set_to_none=True` deallocates the gradient tensor memory completely, allowing PyTorch to avoid unnecessary memory writes and letting subsequent backpropagation operations allocate fresh memory without caching overhead.
        """
    )

with st.expander("Q5: What are the three memory components stored per parameter in AdamW?"):
    st.markdown(
        """
        **Answer:**
        In mixed-precision AdamW:
        1. **FP32 Master Weight** (4 bytes).
        2. **First Moment Vector m_t** (4 bytes).
        3. **Second Moment Vector v_t** (4 bytes).
        Total: **12 bytes per model parameter** in optimizer state alone (excluding activations and model weights).
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Forgetting to Scale Loss During Gradient Accumulation**
        *Root Cause*: Calling `loss.backward()` without dividing by `grad_accum_steps`.
        *Production Fix*: The accumulated gradient becomes `K` times too large, effectively multiplying the learning rate by `K` and causing training divergence. Always execute `(loss / K).backward()`.
        """
    )
    st.warning(
        """
        **Trap 2: Mismatched Random Seeds in Distributed Data Loading**
        *Root Cause*: Failing to set unique seeds across data-parallel ranks, causing all 512 GPUs to train on the exact same token batch.
        *Production Fix*: Ensure `DistributedSampler` properly offsets seeds by `rank` so every GPU sees unique text shards.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Synchronizing Gradients on Every Micro-Batch Step**
        *Root Cause*: Using naive DistributedDataParallel without disabling gradient reduction hooks during accumulation steps.
        *Production Fix*: Use `with model.no_sync():` during all accumulation steps except the final one to prevent unnecessary InfiniBand communication overhead.
        """
    )
    st.warning(
        """
        **Trap 4: Gradient Clipping on Local Rank Instead of Global Norm**
        *Root Cause*: Computing gradient norm across local parameters rather than globally across all tensor-parallel and pipeline-parallel shards.
        *Production Fix*: Use Megatron-LM or FSDP global gradient norm reduction before applying the clipping factor.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Backward vs Forward Compute"):
        st.write("What is the ratio of backward pass compute to forward pass compute?")
        st.info("Approximately 2:1 (Backward requires 2x the FLOPs of forward).")

    with st.expander("⚡ Flashcard 2: AdamW Optimizer State Size"):
        st.write("How many bytes of optimizer state does AdamW require per parameter?")
        st.info("12 bytes (4 bytes master weight + 4 bytes m_t + 4 bytes v_t).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Gradient Accumulation Purpose"):
        st.write("Why use gradient accumulation in pretraining?")
        st.info("To simulate large global batch sizes without exceeding per-GPU VRAM limits.")

    with st.expander("⚡ Flashcard 4: Set to None Benefit"):
        st.write("Why pass set_to_none=True to zero_grad()?")
        st.info("Deallocates memory instead of writing zeros, saving memory bandwidth.")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the end-to-end execution lifecycle of a single pretraining step.",
    "I can explain why the backward pass requires twice the computational FLOPs of the forward pass.",
    "I know how gradient accumulation enables multi-million token global batch sizes.",
    "I understand why AdamW requires 12 bytes of optimizer state per model parameter.",
    "I can explain how activation checkpointing trades 33% compute for an 80% memory reduction.",
    "I know why gradient clipping prevents optimizer divergence when encountering training loss spikes.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered the LLM Pretraining Pipeline, Distributed Execution, and Backpropagation Mechanics."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
