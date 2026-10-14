"""
11.1 PyTorch Fundamentals
=========================
Interactive Master Lesson exploring Tensor Storage & Strides, Multi-Dimensional Shapes,
Vectorized Operations, Autograd, Dynamic Computational Graphs, nn.Module, and Optimizers.
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
    page_title="11.1 PyTorch Fundamentals - LLM Architecture",
    page_icon="🔥",
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
st.title("🔥 11.1 PyTorch Fundamentals")
st.caption(
    "Chapter 11: Deep Learning Frameworks for LLMs | Module 1 of 4"
)

st.markdown(
    """
PyTorch is the undisputed foundational framework powering modern Large Language Model research
and production training. From its low-level strided C++ **Tensor storage engine** and tape-based
**Autograd engine** to modular **nn.Module** hierarchies and parameter abstractions, PyTorch
combines intuitive Python ergonomics with metal-level GPU compute performance.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE FRAMEWORK ENGINE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Deep Learning Engine Framework", divider="rainbow")

st.markdown(
    """
### The PyTorch Core Execution Stack
A high-level Python tensor operation is dispatched through a multi-tier C++ runtime down to hardware acceleration kernels:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "pytorch_core_stack",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("python", "Python Frontend API\\ntorch.Tensor, nn.Module, torch.autograd", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("c10", "ATen / c10 Core C++ Engine\\nTensor Storage, Strides, Dispatcher", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("autograd", "Autograd Tape Engine\\nDynamic Backward DAG Node Evaluation", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("cuda", "Hardware Execution Layer\\ncuBLAS, cuDNN, FlashAttention, Triton", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("hardware", "Hardware Accelerator\\nGPU HBM, Tensor Cores, NVLink", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("python", "c10", label="PyBind11 Dispatch")
arch_dot.edge("c10", "autograd", label="Records Graph")
arch_dot.edge("autograd", "cuda", label="Kernel Invocations")
arch_dot.edge("cuda", "hardware", label="SRAM / Tensor Cores")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **PyTorch 2.0+ `torch.compile`**: Rather than executing Python operations eagerly line-by-line, modern foundation stacks compile subgraphs using TorchDynamo and OpenAI Triton, fusing pointwise operations (e.g. `RMSNorm`, `SwiGLU`) directly into single GPU SRAM kernels for up to 30% speedup.
* **Triton Custom Kernels**: State-of-the-art transformer layers (RoPE, FlashAttention, fused Cross-Entropy) are increasingly authored in OpenAI Triton (embedded in PyTorch), bypassing slow C++ CUDA compilation while matching handwritten assembly speeds.
"""
)

# PyTorch Abstraction Table
pt_abstractions_df = pd.DataFrame(
    {
        "PyTorch Abstraction": ["torch.Tensor", "torch.autograd", "torch.nn.Module", "torch.nn.Parameter", "torch.optim.Optimizer", "torch.compile"],
        "Core Responsibility": ["Multi-dimensional strided array", "Automatic reverse differentiation", "Stateful modular computation block", "Trainable tensor registered in state_dict", "Momentum & gradient update step", "JIT graph capture & kernel fusion"],
        "C++ Subsystem": ["at::Tensor (ATen)", "torch::autograd::Node", "torch::nn::Module", "at::Tensor with requires_grad=True", "C++ Optimizer backend", "TorchDynamo + Inductor"],
        "Role in LLM Engineering": ["Houses weights, tokens, activations", "Computes dL/dW across 80 layers", "Defines Attention and MLP layers", "Encapsulates weight matrices W", "Executes AdamW updates", "Fuses RMSNorm + SwiGLU kernels"],
    }
)
st.dataframe(pt_abstractions_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: TENSOR BASICS (STORAGE & STRIDES)
# -----------------------------------------------------------------------------
st.header("2. Tensor Basics (Storage, Strides, Contiguity & Memory Layout)", divider="rainbow")

st.markdown(
    """
In PyTorch, a `torch.Tensor` is a lightweight Python metadata view over a 1-dimensional contiguous block of memory called a **Storage**:

```text
Tensor Metadata = (Pointer to Storage, Size/Shape, Strides, Offset, Dtype, Device)
```

### Strides and Memory Offsets
The **stride** defines the number of elements in physical memory that must be jumped to move to the next item along each dimension.
For a 2D tensor of shape `[rows, cols]`:
* Row stride: `cols`
* Column stride: `1`
* Element at `(r, c)` is located at physical memory index: `offset = r * stride[0] + c * stride[1]`.

### Contiguity & Zero-Copy Views
Operations like `.transpose()`, `.permute()`, and `.view()` **do not copy data in memory**:
they merely create a new Tensor view with modified strides!
However, when an operation requires contiguous memory (e.g. `.view(-1)`), calling `.contiguous()` triggers an actual memory reallocation and copy if the tensor strides are non-standard.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: TENSOR SHAPES & MULTI-DIMENSIONAL INDEXING
# -----------------------------------------------------------------------------
st.header("3. Tensor Shapes & Multi-Dimensional Indexing", divider="rainbow")

st.markdown(
    """
In transformer architectures, multi-dimensional tensor shapes represent batches, sequences, heads, and feature dimensions:

```text
Inputs:              [Batch_Size, Sequence_Length]
Embeddings:          [Batch_Size, Sequence_Length, Hidden_Dimension]
Query / Key / Value: [Batch_Size, Num_Heads, Sequence_Length, Head_Dimension]
Attention Scores:    [Batch_Size, Num_Heads, Sequence_Length, Sequence_Length]
Logits:              [Batch_Size, Sequence_Length, Vocab_Size]
```

### Shape Manipulation Primitives
* **`view` vs. `reshape`**: `.view()` is guaranteed zero-copy (fails if non-contiguous). `.reshape()` returns a view if possible, or copies memory if necessary.
* **`unsqueeze` & `squeeze`**: Injects or removes singleton dimensions (critical for broadcasting attention masks).
* **Broadcasting**: Automatically expands singleton dimensions (`dim = 1`) across matching dimensions without copying data in memory.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: VECTORIZED TENSOR OPERATIONS
# -----------------------------------------------------------------------------
st.header("4. Vectorized Tensor Operations (GEMM, BMM, Einsum)", divider="rainbow")

st.markdown(
    """
Transformer self-attention is built on three core vectorized matrix operations:

### 1. General Matrix Multiply (GEMM)
`torch.matmul(X, W)` computes linear feed-forward projections `[B, T, d_in] * [d_in, d_out] = [B, T, d_out]`.

### 2. Batched Matrix Multiply (BMM)
`torch.bmm(Q, K.transpose(-1, -2))` multiplies thousands of head matrices in parallel across all batch sequences.

### 3. Einstein Summation (`torch.einsum`)
Enables concise, error-free multi-dimensional tensor contractions:
```python
# Batched multi-head attention inner product:
attn_scores = torch.einsum("b h i d, b h j d -> b h i j", Q, K)
```
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: AUTOMATIC DIFFERENTIATION ENGINE (AUTOGRAD)
# -----------------------------------------------------------------------------
st.header("5. Automatic Differentiation Engine (Autograd Mechanics)", divider="rainbow")

st.markdown(
    """
PyTorch's **Autograd** engine implements **Reverse-Mode Automatic Differentiation**.
Every tensor with `requires_grad=True` tracks the mathematical operation that created it via its `.grad_fn` attribute:

```text
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 5
# y.grad_fn points to <AddBackward0>
# y.grad_fn.next_functions points to <PowBackward0>
```

When `loss.backward()` is called, Autograd traces backward from `loss` through the chain of `grad_fn` nodes, applying vector-Jacobian products (VJPs) and accumulating partial derivatives into `param.grad`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: DYNAMIC COMPUTATIONAL GRAPHS
# -----------------------------------------------------------------------------
st.header("6. Dynamic Computational Graphs & Tape-Based Backpropagation", divider="rainbow")

st.markdown(
    """
PyTorch uses a **Dynamic (Eager / Define-by-Run)** computational graph:
* The computational graph is constructed dynamically on-the-fly as the forward pass executes.
* Different iterations can have different sequence lengths, conditional branches (`if/else`), and dynamic loops without graph recompilation.
* Once `loss.backward()` finishes, the intermediate graph is destroyed (freeing activation memory), and a fresh graph is created on the next step.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: MODULAR NEURAL NETWORK CONSTRUCTION (NN.MODULE)
# -----------------------------------------------------------------------------
st.header("7. Modular Neural Network Construction (nn.Module Lifecycle)", divider="rainbow")

st.markdown(
    """
`nn.Module` is the base class for all neural network components in PyTorch:
* **Recursive Parameter Registration**: Submodules assigned as class attributes (`self.attn = ...`) are automatically registered, making their parameters discoverable via `model.parameters()`.
* **Execution Hooks**: Supports forward and backward pre/post hooks for profiling, activation logging, and distributed gradient all-reduce triggers.
* **Mode Switching**:
  * `model.train()`: Enables Dropout and sets running statistics updates.
  * `model.eval()`: Freezes LayerNorm statistics and deactivates Dropout for deterministic inference.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PARAMETERS, BUFFERS & STATE DICTIONARIES
# -----------------------------------------------------------------------------
st.header("8. Model Parameters, Buffers & State Dictionaries (Parameter vs. Buffer)", divider="rainbow")

st.markdown(
    """
Understanding the distinction between Parameters and Buffers is critical for LLM architecture:

### `nn.Parameter`
A tensor registered with `requires_grad=True` that is updated by the optimizer (e.g. `W_q`, `W_k`, `W_mlp`).
Included in `model.parameters()` and `model.state_dict()`.

### Persistent Buffer (`register_buffer(..., persistent=True)`)
A tensor that is NOT updated by gradient descent, but is part of the model state and must be saved to disk checkpoints (e.g. RoPE precomputed cosine/sine frequency tables, causal attention masks).
Included in `model.buffers()` and `model.state_dict()`.

### Non-Persistent Buffer (`register_buffer(..., persistent=False)`)
A cached tensor generated dynamically on startup that should NOT be serialized to checkpoint files (e.g. intermediate scratch pads).
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PYTORCH OPTIMIZERS PIPELINE
# -----------------------------------------------------------------------------
st.header("9. PyTorch Optimizers Pipeline & Learning Rate Integration", divider="rainbow")

st.markdown(
    """
PyTorch optimizers (`torch.optim.Optimizer`) manage parameter groups:
* **Parameter Groups**: Allow different hyperparameters for different submodules:
  ```python
  optimizer = torch.optim.AdamW([
      {"params": decay_params, "weight_decay": 0.1, "lr": 3e-4},
      {"params": no_decay_params, "weight_decay": 0.0, "lr": 3e-4},
  ])
  ```
* **Optimizer State Storage**: Tracks per-parameter tensors in `optimizer.state[param]`.
* **Zero Grad**: `optimizer.zero_grad(set_to_none=True)` clears gradient pointers to save memory bandwidth.
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE CUSTOM TRANSFORMER BLOCK
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Custom Transformer Building Block (Autograd & nn.Module)", divider="rainbow")

st.markdown(
    """
A complete, idiomatic PyTorch implementation of a modular Transformer block with custom RMSNorm, SwiGLU MLP, and buffer registration:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    # Root Mean Square Layer Normalization
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUMLP(nn.Module):
    # SwiGLU feed-forward network matching Llama 3 architecture
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w_down(silu(w_gate(x)) * w_up(x))
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class TransformerBlock(nn.Module):
    # Idiomatic Transformer block showing parameters, submodules, and buffers
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # SwiGLU hidden dim: 8/3 * d_model
        hidden_dim = int(8.0 * d_model / 3.0)
        self.mlp = SwiGLUMLP(d_model, hidden_dim)

        # Register non-persistent buffer for causal upper-triangular mask
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Attention block with Pre-RMSNorm
        norm_x = self.attn_norm(x)
        # Scaled dot-product attention
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)
        
        # Apply causal mask slice
        mask_slice = self.causal_mask[:seq_len, :seq_len]
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_slice)
        x = x + self.out_proj(attn_out)
        
        # MLP block with Pre-RMSNorm
        x = x + self.mlp(self.mlp_norm(x))
        return x
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE TENSOR STRIDE & COMPUTATIONAL GRAPH WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Tensor Stride, Einsum & Computational Graph Workbench", divider="rainbow")

st.markdown(
    """
Inspect how tensor reshaping and permutation alter underlying **Memory Strides**,
and trace dynamic **Autograd** gradient propagation.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    b_dim = st.slider("Batch Dimension (B)", min_value=1, max_value=4, value=2)
    s_dim = st.slider("Sequence Dimension (S)", min_value=2, max_value=8, value=4)
with col_ctrl2:
    h_dim = st.slider("Hidden Dimension (D)", min_value=4, max_value=16, value=8, step=4)
    transpose_mode = st.selectbox("Operation View", options=["Standard [B, S, D]", "Permuted [S, B, D]", "Flattened [B * S, D]"])

# Compute strides
sample_tensor = np.arange(b_dim * s_dim * h_dim).reshape(b_dim, s_dim, h_dim)
orig_shape = sample_tensor.shape
orig_strides = (s_dim * h_dim, h_dim, 1)

if transpose_mode == "Permuted [S, B, D]":
    view_tensor = np.transpose(sample_tensor, (1, 0, 2))
    cur_shape = view_tensor.shape
    cur_strides = (h_dim, s_dim * h_dim, 1)
    is_contiguous = False
elif transpose_mode == "Flattened [B * S, D]":
    view_tensor = sample_tensor.reshape(b_dim * s_dim, h_dim)
    cur_shape = view_tensor.shape
    cur_strides = (h_dim, 1)
    is_contiguous = True
else:
    cur_shape = orig_shape
    cur_strides = orig_strides
    is_contiguous = True

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Tensor Shape", str(cur_shape))
with col_m2:
    st.metric("Memory Strides", str(cur_strides))
with col_m3:
    st.metric("Is Contiguous Memory?", "YES (Contiguous)" if is_contiguous else "NO (Non-Contiguous View)", delta_color="normal" if is_contiguous else "inverse")

st.subheader("Physical Storage Offset Mapping (First 6 Elements)")
offset_df = pd.DataFrame(
    {
        "Tensor Index": [f"(0, 0, {i})" for i in range(min(6, h_dim))],
        "Strided Offset in RAM": [i * cur_strides[-1] for i in range(min(6, h_dim))],
        "Data Value": [int(sample_tensor[0, 0, i]) for i in range(min(6, h_dim))],
    }
)
st.dataframe(offset_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: What is the architectural difference between a Parameter and a Buffer in PyTorch?"):
    st.markdown(
        """
        **Answer:**
        * **`nn.Parameter`**: A tensor wrapped with `requires_grad=True`. It is automatically tracked in `model.parameters()` and updated by optimizers.
        * **Buffer (`register_buffer`)**: A tensor that holds state but does not require gradients and is not updated by optimizers (e.g. running mean in BatchNorm, causal attention masks, RoPE precomputed frequency tables). Persistent buffers are saved to checkpoints, while non-persistent buffers are recomputed dynamically.
        """
    )

with st.expander("Q2: Why does calling .contiguous() sometimes matter before .view()?"):
    st.markdown(
        """
        **Answer:**
        `.view()` requires that elements are laid out contiguously in memory according to standard row-major order.
        Operations like `.transpose()` or `.permute()` only modify tensor strides without reordering elements in memory, producing a non-contiguous tensor.
        Calling `.view()` on a non-contiguous tensor raises a runtime error: `RuntimeError: view size is not compatible with input tensor's size and stride`.
        Calling `.contiguous()` allocates new memory and copies elements in order, allowing `.view()` to succeed.
        """
    )

with st.expander("Q3: How does PyTorch's dynamic computational graph differ from static graphs (TensorFlow 1.x)?"):
    st.markdown(
        """
        **Answer:**
        * **Static Graph**: Define-and-run. The graph is compiled ahead of time into a static execution engine. Very fast, but difficult to debug and inflexible with variable sequence lengths.
        * **Dynamic Graph (PyTorch)**: Define-by-run. The graph is built as Python lines execute. Facilitates native Python debugging (pdb), dynamic control flows (`if/else`), and variable sequence lengths, rebuilding the backward graph on each step.
        """
    )

with st.expander("Q4: Why should optimizer.zero_grad(set_to_none=True) be used instead of zero_grad()?"):
    st.markdown(
        """
        **Answer:**
        Default `zero_grad()` writes floating-point zeros into the allocated gradient memory buffers.
        Passing `set_to_none=True` deletes the references, deallocating the memory. During backpropagation, PyTorch allocates fresh memory for gradients, saving unnecessary memory-write bandwidth on large models.
        """
    )

with st.expander("Q5: What does torch.compile do under the hood in PyTorch 2.0+?"):
    st.markdown(
        """
        **Answer:**
        `torch.compile` uses:
        1. **TorchDynamo**: Inspects Python bytecode and extracts pure computation subgraphs.
        2. **AOTAutograd**: Traces both forward and backward graphs ahead of execution.
        3. **TorchInductor**: Compiles the graphs into optimized C++ or OpenAI Triton GPU kernels, fusing consecutive pointwise operations (e.g. Norm + Add + Act) to eliminate DRAM read/write roundtrips.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Memory Leaks from Storing Tensors in Lists Without .detach()**
        *Root Cause*: Appending intermediate activation tensors or loss tensors (`losses.append(loss)`) to a list for logging.
        *Production Fix*: The tensor maintains its `.grad_fn` and the entire backward computational graph in memory, causing rapid GPU VRAM exhaustion. Always log scalar values: `losses.append(loss.item())`.
        """
    )
    st.warning(
        """
        **Trap 2: Forgetting model.eval() and with torch.no_grad() During Inference**
        *Root Cause*: Evaluating models in training mode.
        *Production Fix*: Keeps Autograd graph tracking active (wasting VRAM) and leaves Dropout active, injecting random stochastic noise into evaluation benchmarks.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: In-Place Operations Corrupting Autograd Version Counters**
        *Root Cause*: Executing in-place operations like `x += 5` or `x.add_()` on tensors needed for backward differentiation.
        *Production Fix*: Raises `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. Avoid in-place operations on activation paths.
        """
    )
    st.warning(
        """
        **Trap 4: Missing Non-Reentrant Flag in Checkpointing**
        *Root Cause*: Using legacy PyTorch reentrant gradient checkpointing.
        *Production Fix*: Breaks autograd hooks and causes silent gradient tracking failures. Always set `checkpoint_impl=CheckpointImpl.NO_REENTRANT`.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: PyTorch Memory View"):
        st.write("Does calling .transpose() copy data in physical memory?")
        st.info("No. It creates a zero-copy view by altering tensor strides.")

    with st.expander("⚡ Flashcard 2: Parameter vs Buffer"):
        st.write("Which tensor type is included in model.state_dict() but excluded from optimizer updates?")
        st.info("A persistent Buffer (register_buffer).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Log Scalar Loss"):
        st.write("How should loss be extracted for logging to prevent memory leaks?")
        st.info("loss.item() (extracts raw Python float without computational graph).")

    with st.expander("⚡ Flashcard 4: PyTorch 2.0 Compiler Engine"):
        st.write("What backend compiler does PyTorch 2.0 use to generate Triton GPU kernels?")
        st.info("TorchInductor.")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the relationship between physical Tensor Storage, Dimensions, and Strides.",
    "I can explain why .transpose() creates a non-contiguous view and when .contiguous() is required.",
    "I understand how PyTorch Autograd constructs dynamic computation graphs via .grad_fn.",
    "I know the difference between nn.Parameter and registered Buffers (persistent vs. non-persistent).",
    "I understand how optimizer parameter groups enable selective weight decay and learning rates.",
    "I know how to prevent memory leaks by logging loss.item() and using torch.no_grad() in evaluation.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered PyTorch Fundamentals, Tensor Strides, Autograd, and nn.Module Lifecycle."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
