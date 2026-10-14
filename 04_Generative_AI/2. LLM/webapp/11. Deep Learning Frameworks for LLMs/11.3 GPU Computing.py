"""
11.3 GPU Computing
==================
Interactive Master Lesson exploring CPU vs. GPU Architectures, CUDA Programming Model,
GPU Memory Hierarchy, Host-to-Device Transfers, Device Management, MFU, and Memory Optimizations.
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
    page_title="11.3 GPU Computing - LLM Architecture",
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
st.title("⚡ 11.3 GPU Computing")
st.caption(
    "Chapter 11: Deep Learning Frameworks for LLMs | Module 3 of 4"
)

st.markdown(
    """
Training and serving Large Language Models is fundamentally a high-performance GPU computing challenge.
Understanding the **CUDA execution model** (Warps, Thread Blocks, Streaming Multiprocessors),
the physical **memory hierarchy** (HBM vs. SRAM registers), **host-to-device PCIe/NVLink bottlenecks**,
and the **Roofline performance model** allows engineers to squeeze maximum arithmetic throughput
out of modern NVIDIA Hopper and Blackwell superclusters.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE GPU COMPUTE ENGINE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The GPU Compute Engine", divider="rainbow")

st.markdown(
    """
### The Physical Hierarchy of GPU Computing
A modern AI accelerator (like the NVIDIA H100 SXM5) is organized into a massive hierarchy of parallel processors and memory tiers:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "gpu_memory_hierarchy",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("host", "Host System CPU (x86_64 / ARM)\\nDDR5 RAM (512 GB) | PCIe Gen 5 (64 GB/s)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("hbm", "GPU Device Memory (HBM3)\\n80 GB VRAM | 3.35 TB/s Bandwidth", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("l2", "GPU On-Chip L2 Cache\\n50 MB Cache | ~12 TB/s Bandwidth", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("sm", "132 Streaming Multiprocessors (SMs)\\nSRAM Shared Memory (228 KB/SM) | 33 TB/s", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("cores", "Tensor Cores (4th Gen Hopper)\\nFP8 / BF16 Matrix Engines (989 TFLOPS)", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("host", "hbm", label="PCIe / NVLink Transfers")
arch_dot.edge("hbm", "l2", label="L2 Line Loads")
arch_dot.edge("l2", "sm", label="TMA / Shared Memory")
arch_dot.edge("sm", "cores", label="Registers (Zero-Cycle)")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Memory Bandwidth Bound vs. Compute Bound**: LLM execution splits into two distinct hardware regimes:
  1. **Prefill Phase (Prompt Ingestion)**: Highly parallel matrix-matrix multiplication (`GEMM`). Compute bound! Fully saturates Hopper Tensor Cores.
  2. **Decode Phase (Token-by-Token Generation)**: Matrix-vector multiplication (`GEMV`). Memory-bandwidth bound! The GPU must fetch the entire 70B model weights and KV cache from HBM for every single generated token.
* **NVLink Cluster Fabric**: Modern 8-GPU nodes utilize **NVLink 4.0** delivering 900 GB/s bidirectional bandwidth per GPU (over 14x faster than PCIe Gen 5), enabling high-speed Tensor Parallelism without bus contention.
"""
)

# Hardware Architecture Comparison Table
gpu_specs_df = pd.DataFrame(
    {
        "Hardware Metric": ["Peak BF16 Tensor TFLOPS", "Peak FP8 Tensor TFLOPS", "Device Memory (HBM)", "Memory Bandwidth", "L2 Cache Size", "Interconnect Speed"],
        "NVIDIA V100 (Volta, 2017)": ["125 TFLOPS (FP16)", "Unsupported", "32 GB HBM2", "900 GB/s", "6 MB", "300 GB/s (NVLink 2)"],
        "NVIDIA A100 (Ampere, 2020)": ["312 TFLOPS", "Unsupported", "80 GB HBM2e", "2.0 TB/s", "40 MB", "600 GB/s (NVLink 3)"],
        "NVIDIA H100 (Hopper, 2023)": ["989 TFLOPS", "1,979 TFLOPS", "80 GB HBM3", "3.35 TB/s", "50 MB", "900 GB/s (NVLink 4)"],
        "NVIDIA B200 (Blackwell, 2024)": ["2,250 TFLOPS", "4,500 TFLOPS", "192 GB HBM3e", "8.0 TB/s", "128 MB", "1,800 GB/s (NVLink 5)"],
    }
)
st.dataframe(gpu_specs_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: CPU VS. GPU ARCHITECTURAL COMPARISON
# -----------------------------------------------------------------------------
st.header("2. CPU vs. GPU Architectural Comparison (Latency vs. Throughput)", divider="rainbow")

st.markdown(
    """
CPUs and GPUs are engineered for diametrically opposed computing philosophies:

### CPU: Latency-Oriented Computing
* Contains 16 to 96 powerful cores with massive out-of-order execution logic, branch predictors, and multi-megabyte L3 caches.
* Optimized to execute complex sequential code with minimal latency per instruction.

### GPU: Throughput-Oriented Computing
* Contains thousands of small, energy-efficient Arithmetic Logic Units (ALUs) working in lockstep (SIMT: Single Instruction, Multiple Threads).
* Hides high memory latency through massive thread-level parallelism: while one thread group waits for DRAM memory loads, another thread group immediately executes arithmetic on the execution units.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: CUDA PROGRAMMING MODEL & EXECUTION HIERARCHY
# -----------------------------------------------------------------------------
st.header("3. CUDA Programming Model & Execution Hierarchy (Grids, Blocks, Warps, Threads)", divider="rainbow")

st.markdown(
    """
CUDA decomposes parallel execution into a three-tier hierarchy:

```text
Grid -> Thread Blocks -> Warps (32 Threads) -> Threads
```

### 1. The Warp: Fundamental Execution Unit
Hardware executes instructions in groups of **32 threads called a Warp**:
* All 32 threads in a warp execute the same instruction simultaneously.
* **Warp Divergence Penalty**: If threads within a warp take different branches (`if/else`), both branches must be executed serially with masked threads, cutting throughput in half.

### 2. Thread Blocks & Streaming Multiprocessors (SMs)
* A **Thread Block** contains up to 1,024 threads that share ultra-fast on-chip Shared Memory (SRAM) and hardware barrier synchronizations (`__syncthreads()`).
* Thread blocks are assigned to independent SMs across the chip.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: GPU MEMORY HIERARCHY
# -----------------------------------------------------------------------------
st.header("4. GPU Memory Hierarchy (HBM, L2 Cache, SRAM, Registers)", divider="rainbow")

st.markdown(
    """
Understanding the memory speed pyramid is the key to GPU optimization:

```text
Registers:       ~30 TB/s Bandwidth  | 0 Cycles Latency    | ~256 KB per SM
Shared Memory:   ~15 TB/s Bandwidth  | ~20 Cycles Latency  | 228 KB per SM
L2 Cache:        ~12 TB/s Bandwidth  | ~200 Cycles Latency | 50 MB on-chip
HBM3 DRAM:       3.35 TB/s Bandwidth | ~400 Cycles Latency | 80 GB off-chip
PCIe Gen 5:      64 GB/s Bandwidth   | Microseconds        | Host RAM
```

### The SRAM Revolution (FlashAttention)
Writing intermediate data to HBM DRAM is 10x slower than keeping it in on-chip SRAM.
Algorithms that tile computation to stay inside SRAM (like FlashAttention) completely bypass HBM bandwidth bottlenecks.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: TENSOR MOVEMENT DYNAMICS & HOST-TO-DEVICE BOTTLENECKS
# -----------------------------------------------------------------------------
st.header("5. Tensor Movement Dynamics & Host-to-Device Bottlenecks (PCIe vs. NVLink)", divider="rainbow")

st.markdown(
    """
Moving tensors across physical hardware buses is often the slowest link in the pipeline:
* **Host-to-Device (`.to("cuda")`)**: Crosses the PCIe bus (64 GB/s on Gen 5). Transferring a 140GB model snapshot takes over 2.2 seconds!
* **Pinned Memory (`pin_memory=True`)**: Allocates page-locked host memory, enabling Direct Memory Access (DMA) engines to transfer tensors asynchronously without CPU intervention.
* **Non-Blocking Transfers**: `tensor.to(device, non_blocking=True)` overlaps host-to-device transfers with GPU kernel computation.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: DEVICE MANAGEMENT & MULTI-GPU ORCHESTRATION
# -----------------------------------------------------------------------------
st.header("6. Device Management & Multi-GPU Orchestration in PyTorch", divider="rainbow")

st.markdown(
    """
PyTorch provides low-level device management APIs:
* **Current Device Context**: `torch.cuda.set_device(local_rank)` binds the current OS process to a specific physical GPU.
* **CUDA Streams**: Independent command queues where GPU kernels execute asynchronously. Operations on different streams execute concurrently on the GPU:
  ```python
  stream = torch.cuda.Stream()
  with torch.cuda.stream(stream):
      # Computes asynchronously alongside the default stream
      y = model(x)
  ```
* **Synchronization Barriers**: `torch.cuda.synchronize()` blocks host CPU execution until all queued GPU tasks complete (critical for accurate runtime profiling).
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: GPU UTILIZATION METRICS & THE ROOFLINE MODEL
# -----------------------------------------------------------------------------
st.header("7. GPU Utilization Metrics & Kernel Profiling (MFU, Roofline Model)", divider="rainbow")

st.markdown(
    """
### The Roofline Performance Model
The Roofline model plots achievable performance (TFLOPS) against **Operational Intensity** (FLOPs per byte of memory loaded):

```text
Attainable_Performance = min( Peak_FLOPS, Operational_Intensity * Memory_Bandwidth )
```
* **Memory-Bound Ceiling**: If operational intensity is low (e.g. element-wise additions, LayerNorm, autoregressive decoding), throughput is capped by memory bandwidth.
* **Compute-Bound Ceiling**: If operational intensity is high (e.g. dense GEMMs in prompt prefill), throughput hits the hardware Tensor Core ceiling.

### Model FLOPs Utilization (MFU)
Measures the percentage of peak theoretical GPU capability achieved by the model arithmetic. A world-class LLM pretraining setup achieves an MFU of **38% to 45%**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: MEMORY OPTIMIZATION TECHNIQUES
# -----------------------------------------------------------------------------
st.header("8. Memory Optimization Techniques (PagedAttention, Quantization, FlashAttention)", divider="rainbow")

st.markdown(
    """
State-of-the-art inference engines (vLLM, TensorRT-LLM) deploy three critical memory innovations:

### 1. PagedAttention (vLLM, Kwon et al., 2023)
Traditional frameworks allocate contiguous VRAM buffers for KV caches, resulting in up to 80% memory fragmentation.
PagedAttention treats GPU VRAM like virtual memory operating systems: KV cache tensors are partitioned into fixed-size pages (e.g. 16 tokens), stored in non-contiguous physical memory blocks, eliminating internal fragmentation and boosting throughput by 2x to 4x.

### 2. Weight Quantization (AWQ, GPTQ, BitsAndBytes)
Compresses 16-bit model weights down to 8-bit or 4-bit integers:
* **W4A16**: 4-bit weights with 16-bit activations. Cuts weight VRAM from 140GB down to 35GB, allowing a 70B model to fit onto a single 80GB H100!
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE CUDA DEVICE MANAGEMENT IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("9. Production-Grade CUDA Device Management & Memory Tracking Engine", divider="rainbow")

st.markdown(
    """
A production PyTorch utility for monitoring GPU memory allocations, cache reserves, and non-blocking streaming:
"""
)

code_snippet = """import torch
from typing import Dict, Any

class GPUMonitor:
    # Production GPU VRAM profiler and device orchestrator
    @staticmethod
    def get_vram_summary(device: int = 0) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"status": "CUDA Unavailable"}
            
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

        return {
            "allocated_gb": allocated,
            "reserved_cache_gb": reserved,
            "peak_allocated_gb": max_allocated,
            "total_capacity_gb": total,
            "utilization_pct": (allocated / total) * 100.0,
        }

    @staticmethod
    def benchmark_streamed_transfer(tensor: torch.Tensor, device: int = 0) -> torch.Tensor:
        # Asynchronous non-blocking transfer using dedicated CUDA stream
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            gpu_tensor = tensor.to(device=f"cuda:{device}", non_blocking=True)
        # Synchronize stream before consuming
        stream.synchronize()
        return gpu_tensor

    @staticmethod
    def clear_cache():
        # Deallocates unused PyTorch caching allocator reserves
        torch.cuda.empty_cache()
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE GPU MEMORY & ROOFLINE WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive GPU Memory & Roofline Performance Workbench", divider="rainbow")

st.markdown(
    """
Explore the **Roofline Model** for an NVIDIA H100 GPU and determine whether a given transformer
operation is **Memory-Bound** or **Compute-Bound**.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    gpu_hardware = st.selectbox("Hardware Platform", options=["NVIDIA H100 SXM (989 TFLOPS, 3.35 TB/s)", "NVIDIA A100 SXM (312 TFLOPS, 2.0 TB/s)"], index=0)
    op_type = st.selectbox("Operation Type", options=["Autoregressive Decode (Batch=1)", "Prefill GEMM (Batch=4, Seq=4096)", "RMSNorm Normalization", "FlashAttention Kernel"], index=1)
with col_ctrl2:
    batch_sim = st.slider("Concurrent Batch Size", min_value=1, max_value=32, value=4)
    precision_mode = st.selectbox("Precision Format", options=["BF16 (16-bit)", "FP8 (8-bit)"], index=0)

# Roofline constants
peak_flops = 989.0 if "H100" in gpu_hardware else 312.0
if precision_mode == "FP8 (8-bit)":
    peak_flops *= 2.0
bandwidth_tb = 3.35 if "H100" in gpu_hardware else 2.0

# Empirical Operational Intensities (FLOPs / Byte)
intensities = {
    "Autoregressive Decode (Batch=1)": 1.0 * batch_sim,
    "Prefill GEMM (Batch=4, Seq=4096)": 250.0,
    "RMSNorm Normalization": 2.0,
    "FlashAttention Kernel": 85.0,
}
current_intensity = intensities[op_type]

# Attainable throughput: min(Peak_FLOPS, Intensity * Bandwidth)
attainable_tflops = min(peak_flops, current_intensity * bandwidth_tb)
is_compute_bound = attainable_tflops >= peak_flops

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Operational Intensity", f"{current_intensity:.1f} FLOPs/Byte")
with col_m2:
    st.metric("Attainable Throughput", f"{attainable_tflops:.1f} TFLOPS")
with col_m3:
    st.metric("Execution Regime", "COMPUTE-BOUND" if is_compute_bound else "MEMORY-BANDWIDTH BOUND", delta_color="normal" if is_compute_bound else "inverse")

# Roofline Curve Plot
intensity_range = np.logspace(0, 3, 50)
roofline_tflops = np.minimum(peak_flops, intensity_range * bandwidth_tb)

roofline_df = pd.DataFrame(
    {
        "Operational Intensity (FLOPs/Byte)": intensity_range,
        "Attainable Throughput (TFLOPS)": roofline_tflops,
    }
)

st.subheader("Hardware Roofline Ceiling Trajectory")
st.line_chart(roofline_df.set_index("Operational Intensity (FLOPs/Byte)"))

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is LLM generation (decoding) memory-bandwidth bound rather than compute-bound?"):
    st.markdown(
        """
        **Answer:**
        During the decode phase, the model generates 1 token at a time.
        For a 70B parameter model in FP16, the GPU must stream all 140 GB of parameter weights plus the KV cache from HBM into the compute cores for every single generated token.
        Because each parameter is multiplied by only a single activation vector (operational intensity is `1 FLOP / Byte`), the Tensor Cores spend over 90% of their time idling waiting for memory transfers from DRAM.
        """
    )

with st.expander("Q2: What is Warp Divergence and why does it penalize GPU execution?"):
    st.markdown(
        """
        **Answer:**
        A warp consists of 32 threads executing in SIMT lockstep.
        If code contains conditional branches (`if (threadIdx.x < 16) { A } else { B }`), the warp cannot execute both paths concurrently. It serializes execution: it runs path A with half the threads disabled, then runs path B with the other half disabled. This cuts peak theoretical execution throughput by 50%.
        """
    )

with st.expander("Q3: How does PagedAttention eliminate memory fragmentation in vLLM?"):
    st.markdown(
        """
        **Answer:**
        Standard systems pre-allocate a contiguous memory block for a request's maximum context length (e.g. 8k tokens), even if the request only uses 500 tokens.
        PagedAttention partitions KV cache memory into small, fixed-size pages (e.g. 16 tokens).
        Pages are allocated dynamically on-the-fly and linked via a page table (identical to operating system virtual memory paging). This eliminates internal memory fragmentation and boosts serving throughput by 2x to 4x.
        """
    )

with st.expander("Q4: What is the difference between PyTorch memory_allocated() and memory_reserved()?"):
    st.markdown(
        """
        **Answer:**
        * `torch.cuda.memory_allocated()`: The exact amount of GPU VRAM actively holding live PyTorch tensor data.
        * `torch.cuda.memory_reserved()`: The total memory currently held by PyTorch's internal Caching Allocator. To avoid expensive CUDA driver allocations (`cudaMalloc`), PyTorch keeps deallocated memory in a reserve cache.
        """
    )

with st.expander("Q5: Why is pinned memory (page-locked memory) faster for host-to-device transfers?"):
    st.markdown(
        """
        **Answer:**
        Standard host memory is pageable and can be swapped to disk by the operating system.
        When copying pageable memory to the GPU, the CUDA driver must first copy the data to a temporary internal pinned buffer, then initiate Direct Memory Access (DMA).
        Allocating pinned memory (`pin_memory=True`) guarantees the memory physical address will not change, allowing the GPU DMA engine to read directly from host RAM at full PCIe line rate.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Synchronizing Tensors Inside the Generation Loop**
        *Root Cause*: Calling `.item()` or `print(tensor)` on the GPU during autoregressive decoding.
        *Production Fix*: Forces an expensive host-device synchronization barrier (`cudaStreamSynchronize()`), stalling the GPU pipeline and destroying generation throughput.
        """
    )
    st.warning(
        """
        **Trap 2: Excessive Calls to torch.cuda.empty_cache()**
        *Root Cause*: Calling `empty_cache()` after every forward pass to prevent OOM.
        *Production Fix*: Forces the driver to execute slow `cudaFree` and `cudaMalloc` calls, degrading execution speed by up to 5x. Let the PyTorch caching allocator manage reserves.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Blocking CPU-to-GPU Transfers in Data Loaders**
        *Root Cause*: Omitting `pin_memory=True` and `non_blocking=True` in the PyTorch DataLoader.
        *Production Fix*: The GPU idles while waiting for CPU worker threads to copy batches across the PCIe bus. Always engage pinned asynchronous prefetching.
        """
    )
    st.warning(
        """
        **Trap 4: Benchmarking Without Warmup Steps**
        *Root Cause*: Measuring GPU kernel runtimes on step 0 without warming up the CUDA context and JIT compilation caches.
        *Production Fix*: Always run 5 to 10 warmup steps before recording timing profiles with `torch.cuda.Event`.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: CUDA Warp Size"):
        st.write("How many physical threads execute in a single CUDA Warp?")
        st.info("Exactly 32 threads.")

    with st.expander("⚡ Flashcard 2: Decode Phase Bottleneck"):
        st.write("What hardware constraint bounds LLM autoregressive token generation?")
        st.info("Memory Bandwidth (HBM to compute transfer speed).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: H100 SRAM Shared Memory"):
        st.write("How much ultra-fast shared memory does an NVIDIA H100 possess per SM?")
        st.info("228 KB per Streaming Multiprocessor (SM).")

    with st.expander("⚡ Flashcard 4: NVLink 4.0 Bandwidth"):
        st.write("What is the bidirectional bandwidth of NVLink 4 on an H100?")
        st.info("900 GB/s per GPU (over 14x faster than PCIe Gen 5).")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the difference between throughput-oriented GPUs and latency-oriented CPUs.",
    "I can explain the CUDA programming execution hierarchy: Grid, Thread Block, Warp, Thread.",
    "I understand the speed disparity between on-chip SRAM registers and off-chip HBM3 memory.",
    "I know why prompt prefill is compute-bound while autoregressive decoding is memory-bandwidth bound.",
    "I can explain the Roofline Performance Model and Operational Intensity.",
    "I understand how PagedAttention eliminates memory fragmentation in high-throughput LLM serving.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered GPU Computing, CUDA Memory Hierarchies, and Roofline Performance Modeling."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
