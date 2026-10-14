import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="14.1 PEFT Concepts - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionPEFTProfiler:
    """
    Production-grade PEFT Memory & Parameter Profiler.
    Calculates parameter counts, AdamW memory states, gradient buffers,
    and storage footprints across Full Fine-Tuning vs. PEFT paradigms.
    """
    def __init__(self, base_params_billion: float = 7.0, bytes_per_param: int = 2):
        self.base_params_billion = base_params_billion
        self.total_base_params = int(base_params_billion * 1e9)
        self.bytes_per_param = bytes_per_param  # BF16 = 2 bytes

    def profile_full_fine_tuning(self) -> Dict[str, Any]:
        """Calculates memory profile for 100% parameter update."""
        weight_mem_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        grad_mem_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        # FP32 Master Weights (4B) + Momentum (4B) + Variance (4B) = 12-16 bytes/param
        optim_mem_gb = (self.total_base_params * 12) / (1024**3)
        total_vram = weight_mem_gb + grad_mem_gb + optim_mem_gb

        return {
            "trainable_params": self.total_base_params,
            "trainable_pct": 100.0,
            "weight_gb": weight_mem_gb,
            "grad_gb": grad_mem_gb,
            "optim_gb": optim_mem_gb,
            "total_vram_gb": total_vram,
            "checkpoint_disk_mb": weight_mem_gb * 1024
        }

    def profile_peft(self, adapter_ratio: float = 0.002) -> Dict[str, Any]:
        """Calculates memory profile for PEFT adapter (e.g. 0.2% trainable parameters)."""
        trainable_params = int(self.total_base_params * adapter_ratio)
        frozen_params = self.total_base_params - trainable_params

        # Base weights in BF16
        base_weight_mem_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        # Trainable adapter weights in BF16
        adapter_weight_mem_gb = (trainable_params * self.bytes_per_param) / (1024**3)
        # Gradients strictly for adapter
        grad_mem_gb = (trainable_params * self.bytes_per_param) / (1024**3)
        # Optimizer states strictly for adapter
        optim_mem_gb = (trainable_params * 12) / (1024**3)
        
        total_vram = base_weight_mem_gb + adapter_weight_mem_gb + grad_mem_gb + optim_mem_gb

        return {
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_pct": adapter_ratio * 100.0,
            "weight_gb": base_weight_mem_gb + adapter_weight_mem_gb,
            "grad_gb": grad_mem_gb,
            "optim_gb": optim_mem_gb,
            "total_vram_gb": total_vram,
            "checkpoint_disk_mb": (trainable_params * self.bytes_per_param) / (1024**2)
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("14.1 PEFT Concepts")
st.caption("Chapter 14: Parameter-Efficient Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
As frontier foundation models scaled from 7 billion to 70+ billion parameters, **Full Fine-Tuning (FFT)** 
became commercially and computationally prohibitive for all but the largest research labs. 
Allocating 16 to 20 bytes per parameter for optimizer states, gradients, and master weights turns a 70B model into a 1.2 Terabyte memory monster.

**Parameter-Efficient Fine-Tuning (PEFT)** revolutionized post-training by demonstrating a profound empirical truth:
**Freezing 99%+ of the base model's parameters and training a tiny fraction of modular weights achieves downstream accuracy on par with full fine-tuning.**

In this lesson, you will master:
1. The economic, computational, and architectural motivations behind PEFT.
2. The core differences between Full Fine-Tuning and PEFT across VRAM, latency, storage, and multi-tenancy.
3. The mechanics of Trainable Parameters: selective gradient graphs and rank decomposition.
4. The mechanics of Frozen Parameters: protecting foundation representations and zero optimizer memory allocation.
5. Exact memory accounting: how PEFT collapses optimizer and gradient memory by over 95%.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Parameter-Efficient Adaptation Landscape & Memory Hierarchy", divider="rainbow")

st.markdown("""
PEFT methods fundamentally alter how computational graphs are constructed and updated during backpropagation.
""")

peft_arch_dot = """
digraph PEFTArchitecture {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_full {
        label = "Full Fine-Tuning (FFT) Graph";
        color = "#334155";
        fontcolor = "#94A3B8";
        FullBase [label="Base Model W\\n(requires_grad = True)\\n7,000,000,000 Params", fillcolor="#B91C1C"];
        FullGrad [label="Gradient Buffer dW\\n14 GB (BF16)", fillcolor="#B91C1C"];
        FullOpt [label="AdamW States (m, v)\\n84 GB (FP32)", fillcolor="#B91C1C"];
        FullBase -> FullGrad -> FullOpt;
    }

    subgraph cluster_peft {
        label = "PEFT (LoRA / Adapters) Graph";
        color = "#334155";
        fontcolor = "#94A3B8";
        PeftBase [label="Base Model W_0 (FROZEN)\\n(requires_grad = False)\\nZero Grad, Zero Optimizer", fillcolor="#0369A1"];
        PeftAdapter [label="Trainable Adapter (B x A)\\n(requires_grad = True)\\n14,000,000 Params (0.2%)", fillcolor="#047857"];
        PeftGrad [label="Adapter Grad d(B,A)\\n28 MB (BF16)", fillcolor="#047857"];
        PeftOpt [label="Adapter AdamW\\n168 MB (FP32)", fillcolor="#047857"];
        
        PeftBase -> PeftAdapter [label="Forward Addition\\nh = W_0 x + B A x"];
        PeftAdapter -> PeftGrad -> PeftOpt;
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(peft_arch_dot, use_container_width=True)

st.markdown("""
### Key Structural Takeaways
* **Frozen Base Model (`requires_grad = False`)**: The billions of base parameters require **zero gradient storage** and **zero optimizer state memory**.
* **Trainable Adapter (`requires_grad = True`)**: Only the tiny adapter matrices allocate memory for backpropagation and AdamW moments.
* **Storage Modularity**: The base model stays unmodified on disk; adapters are serialized as lightweight 20MB to 50MB artifacts that plug in dynamically.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - WHY PEFT?
# ==============================================================================
st.header("2. Why PEFT?", divider="rainbow")

st.markdown("""
The rapid ascension of PEFT from an academic novelty to an industry standard is driven by four powerful macroeconomic and technical forces:
""")

col_why1, col_why2 = st.columns(2)
with col_why1:
    st.subheader("1. Breaking the GPU Memory Wall")
    st.markdown("""
    * **The 16-Bytes-per-Param Problem**: Updating every parameter requires storing the parameter (2B), its gradient (2B), and its first/second optimizer moments (8-12B in FP32).
    * **Hardware Reality**: Full fine-tuning of an 8B model requires multiple A100/H100 80GB GPUs. PEFT enables adapting that exact same model on a **single 16GB or 24GB consumer GPU** (e.g. RTX 4090).
    """)

    st.subheader("2. Eliminating Catastrophic Forgetting")
    st.markdown("""
    * In Full Fine-Tuning, large gradient steps rewrite the foundational representation vectors, destroying common sense reasoning.
    * In PEFT, the base weights `W_0` are **mathematically frozen**. The model's core linguistic knowledge cannot be erased by narrow downstream task tuning.
    """)

with col_why2:
    st.subheader("3. Multi-Tenant Deployment Economics")
    st.markdown("""
    * **The Full-FT Serving Dilemma**: If an enterprise creates 50 customized models for 50 clients via full fine-tuning, serving them requires loading 50 separate 16GB model instances into GPU clusters (800GB of GPU memory).
    * **The PEFT Multi-Adapter Solution**: Serve **1 single frozen base model** in GPU VRAM, and swap tiny 30MB client adapters dynamically on the fly per request with zero server restarts!
    """)

    st.subheader("4. Checkpoint Storage & Distribution")
    st.markdown("""
    * Full FT produces a 15GB - 30GB model checkpoint for every training run.
    * PEFT generates a 20MB - 100MB adapter file. Thousands of domain adapters can be saved on a single developer laptop or pushed to Git repositories in seconds.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - FULL FINE-TUNING VS PEFT
# ==============================================================================
st.header("3. Full Fine-Tuning vs. PEFT", divider="rainbow")

st.markdown("""
A direct technical and operational comparison between Full Fine-Tuning and Parameter-Efficient Adaptation:
""")

compare_peft_data = {
    "Operational Metric": [
        "Trainable Parameter %",
        "VRAM for Optimizer States (7B Model)",
        "Minimum Required Hardware",
        "Checkpoint File Size",
        "Multi-Tenant Serving Support",
        "Catastrophic Forgetting Risk",
        "Downstream Task Accuracy",
        "Training Wall-Clock Speed"
    ],
    "Full Fine-Tuning (FFT)": [
        "100.0%",
        "56 GB to 84 GB (FP32 AdamW)",
        "2x to 4x A100 (80GB) with FSDP",
        "14 GB to 16 GB per task",
        "Impossible (requires separate model instances)",
        "High (representation drift)",
        "State-of-the-art upper bound",
        "Standard (1.0x baseline)"
    ],
    "PEFT (e.g. LoRA / QLoRA)": [
        "0.05% to 0.5%",
        "100 MB to 300 MB",
        "1x RTX 3090 / 4090 (24GB) or QLoRA on 16GB",
        "20 MB to 100 MB per task",
        "Seamless (Dynamic LoRA swapping in vLLM / SGLang)",
        "Virtually Zero (base weights immutable)",
        "Comparable (>= 98-99% of Full FT)",
        "Equal to ~1.2x faster (fewer gradients to communicate)"
    ]
}
st.dataframe(pd.DataFrame(compare_peft_data), use_container_width=True, hide_index=True)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Leading AI platforms (OpenAI fine-tuning API, Mistral Fine-Tuning, Together AI, Predibase) **use LoRA-based PEFT as the default engine**. 
Full fine-tuning is rarely offered or is priced at 10x-50x higher premiums due to the dedicated multi-GPU serving infrastructure required.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - TRAINABLE PARAMETERS
# ==============================================================================
st.header("4. Trainable Parameters", divider="rainbow")

st.markdown("""
In PyTorch and deep learning frameworks, a parameter is "trainable" if its tensor flag `requires_grad = True`.
In PEFT, the ratio of trainable parameters to total parameters is called the **Trainable Fraction**:
```text
Trainable_Fraction = (Count_Of_Parameters_With_Requires_Grad) / (Total_Model_Parameters)
```
Typically:
* Full Fine-Tuning: `Trainable_Fraction = 1.0` (100%)
* Prefix Tuning / Prompt Tuning: `Trainable_Fraction ~ 0.0001` (0.01%)
* Standard LoRA (r=16 on Attention projections): `Trainable_Fraction ~ 0.001 to 0.005` (0.1% to 0.5%)
* Comprehensive LoRA (r=64 on All Linear layers): `Trainable_Fraction ~ 0.01 to 0.02` (1% to 2%)
""")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.markdown("#### How PyTorch Manages Trainable Parameters")
    st.code("""
# Inspecting trainable parameter count in PyTorch
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable_params:,} | Total: {all_params:,}")
print(f"Trainable %: {100 * trainable_params / all_params:.3f}%")
""", language="python")

with col_tr2:
    st.markdown("#### The Intrinsic Rank Hypothesis")
    st.markdown("""
    The foundational paper by Aghajanyan et al. (2020) and Hu et al. (LoRA, 2021) demonstrated that 
    the parameter updates for downstream tasks have a **low "intrinsic dimension"**.
    Even though the model operates in a 4,096-dimensional hidden space, task adaptation occupies a low-dimensional manifold (rank 8 to 32) 
    that can be captured with 1,000x fewer parameters.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - FROZEN PARAMETERS
# ==============================================================================
st.header("5. Frozen Parameters", divider="rainbow")

st.markdown("""
A parameter is "frozen" when its gradient tracking is explicitly deactivated:
```python
for param in model.parameters():
    param.requires_grad = False
```
Freezing parameters fundamentally changes both computation and memory allocation:
""")

col_frz1, col_frz2 = st.columns(2)
with col_frz1:
    st.subheader("What Happens to Frozen Weights During Forward Pass?")
    st.markdown("""
    * Frozen weights still perform forward matrix multiplications: `y = x W_frozen`.
    * They remain active in computing intermediate activations.
    * They can be quantized into 4-bit or 8-bit precision (e.g. QLoRA) without affecting gradient calculation.
    """)

with col_frz2:
    st.subheader("What Happens to Frozen Weights During Backward Pass?")
    st.markdown("""
    * Backpropagation computes activation gradients through the frozen layer (`dL / dx = (dL / dy) W_frozen^T`) to pass gradients backward to earlier layers.
    * However, the parameter gradient `dL / dW_frozen` is **never computed and never stored in memory**.
    * The optimizer allocates **zero bytes** for frozen parameters.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - MEMORY EFFICIENCY
# ==============================================================================
st.header("6. Memory Efficiency", divider="rainbow")

st.markdown("""
To understand why PEFT enables fine-tuning on consumer hardware, we must analyze the exact breakdown of GPU VRAM.
""")

st.markdown("""
### The Fine-Tuning Memory Equation
```text
Total_VRAM = Static_Memory (Weights + Gradients + Optimizer) + Dynamic_Memory (Activations + KV Cache)
```
""")

mem_col1, mem_col2 = st.columns(2)
with mem_col1:
    st.markdown("#### Full Fine-Tuning VRAM (7B Model, BF16 + AdamW)")
    st.markdown("""
    * **Model Weights (BF16)**: `7B * 2 bytes = 14 GB`
    * **Gradients (BF16)**: `7B * 2 bytes = 14 GB`
    * **AdamW FP32 Momentum (`m`)**: `7B * 4 bytes = 28 GB`
    * **AdamW FP32 Variance (`v`)**: `7B * 4 bytes = 28 GB`
    * **FP32 Master Copy**: `7B * 4 bytes = 28 GB` *(often used in mixed precision)*
    * **Static VRAM Subtotal**: **~84 GB to 112 GB** *(Excluding activations!)*
    """)

with mem_col2:
    st.markdown("#### LoRA PEFT VRAM (7B Model, BF16 Base + 0.2% Adapter)")
    st.markdown("""
    * **Base Model Weights (BF16 Frozen)**: `7B * 2 bytes = 14 GB`
    * **Adapter Weights (BF16)**: `14M * 2 bytes = 0.028 GB`
    * **Adapter Gradients (BF16)**: `14M * 2 bytes = 0.028 GB`
    * **Adapter AdamW States (FP32)**: `14M * 12 bytes = 0.168 GB`
    * **Static VRAM Subtotal**: **~14.22 GB** *(A massive ~85% reduction in static memory!)*
    """)

st.info("""
**QLoRA (4-bit Quantization)** pushes this even further:
By compressing the 7B frozen base model into 4-bit NormalFloat (NF4), base weights consume only **~3.5 GB to 4 GB**. 
The entire static memory drops to under **5 GB**, making fine-tuning comfortable on a 12GB or 16GB GPU!
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End PEFT Parameter and Memory Profiler in Python", divider="rainbow")

st.markdown("""
The following self-contained class computes exact parameter counts, AdamW memory allocation, 
and disk footprint across Full Fine-Tuning and arbitrary PEFT adapter ratios.
""")

st.code("""
from typing import Dict, Any

class ProductionPEFTProfiler:
    \"\"\"
    Production-grade PEFT Memory & Parameter Profiler.
    \"\"\"
    def __init__(self, base_params_billion: float = 7.0, bytes_per_param: int = 2):
        self.base_params_billion = base_params_billion
        self.total_base_params = int(base_params_billion * 1e9)
        self.bytes_per_param = bytes_per_param

    def profile_full_fine_tuning(self) -> Dict[str, Any]:
        weight_mem_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        grad_mem_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        optim_mem_gb = (self.total_base_params * 12) / (1024**3)
        total_vram = weight_mem_gb + grad_mem_gb + optim_mem_gb
        return {
            \"trainable_params\": self.total_base_params,
            \"total_vram_gb\": total_vram,
            \"checkpoint_disk_mb\": weight_mem_gb * 1024
        }

    def profile_peft(self, adapter_ratio: float = 0.002) -> Dict[str, Any]:
        trainable_params = int(self.total_base_params * adapter_ratio)
        base_weight_gb = (self.total_base_params * self.bytes_per_param) / (1024**3)
        grad_mem_gb = (trainable_params * self.bytes_per_param) / (1024**3)
        optim_mem_gb = (trainable_params * 12) / (1024**3)
        total_vram = base_weight_gb + grad_mem_gb + optim_mem_gb
        return {
            \"trainable_params\": trainable_params,
            \"total_vram_gb\": total_vram,
            \"checkpoint_disk_mb\": (trainable_params * self.bytes_per_param) / (1024**2)
        }
""", language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Parameter Sizer & Memory Budget Simulator", divider="rainbow")

st.markdown("""
Compare GPU VRAM requirements and checkpoint storage between Full Fine-Tuning and PEFT across arbitrary model scales.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Model & Adapter Configuration")
    user_model_size = st.selectbox(
        "Base Model Scale",
        [1.0, 3.0, 7.0, 8.0, 13.0, 70.0],
        index=3,
        format_func=lambda x: f"{x}B Parameters (e.g. Llama-3-{int(x) if x.is_integer() else x}B)"
    )
    user_adapter_pct = st.slider(
        "PEFT Trainable Fraction (%)",
        min_value=0.01,
        max_value=2.0,
        value=0.2,
        step=0.01,
        format="%.2f%%"
    )
    quantize_base = st.checkbox("Enable 4-bit Base Quantization (QLoRA Mode)", value=False)

    profiler = ProductionPEFTProfiler(
        base_params_billion=user_model_size,
        bytes_per_param=0.5 if quantize_base else 2
    )
    fft_profile = profiler.profile_full_fine_tuning()
    peft_profile = profiler.profile_peft(adapter_ratio=user_adapter_pct / 100.0)

with wb_col2:
    st.subheader("VRAM Comparison (Static Memory)")
    
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.metric("Full Fine-Tuning VRAM", f"{fft_profile['total_vram_gb']:.2f} GB", "100% Parameters")
    with col_met2:
        st.metric(
            "PEFT VRAM Required", 
            f"{peft_profile['total_vram_gb']:.2f} GB", 
            f"-{((fft_profile['total_vram_gb'] - peft_profile['total_vram_gb'])/fft_profile['total_vram_gb'])*100:.1f}% Savings"
        )

    chart_df = pd.DataFrame({
        "Method": ["Full Fine-Tuning", "PEFT Adapter"],
        "Static VRAM (GB)": [fft_profile["total_vram_gb"], peft_profile["total_vram_gb"]],
        "Checkpoint Size (MB)": [fft_profile["checkpoint_disk_mb"], peft_profile["checkpoint_disk_mb"]]
    })
    st.bar_chart(chart_df.set_index("Method")["Static VRAM (GB)"])

    st.markdown("##### Storage Footprint Impact:")
    st.info(f"""
    * **Full FT Checkpoint**: `{fft_profile['checkpoint_disk_mb']/1024:.2f} GB` on disk.
    * **PEFT Adapter Checkpoint**: `{peft_profile['checkpoint_disk_mb']:.2f} MB` on disk (**{int(fft_profile['checkpoint_disk_mb'] / max(1, peft_profile['checkpoint_disk_mb']))}x smaller!**).
    """)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why does freezing parameters save memory if weights still exist in VRAM?"):
        st.write("""
        While frozen parameters still reside in VRAM to perform forward pass multiplications, freezing them eliminates their **gradients** and **optimizer states**.
        In FP32 AdamW, optimizer states (first moment `m` and second moment `v`) require 8 bytes per parameter, and FP32 master weights require 4 bytes.
        By freezing 99.8% of the weights, you eliminate 12 to 16 bytes of memory per parameter for almost the entire model.
        """)
    with st.expander("Q2: What is the 'Intrinsic Rank Hypothesis' and how does it justify PEFT?"):
        st.write("""
        The Intrinsic Rank Hypothesis states that parameter updates necessary to adapt a large pretrained model to a specific downstream task lie in a subspace of significantly lower dimension (intrinsic dimension) than the full parameter space.
        Because pretrained representations already capture syntax, semantics, and world knowledge, task adaptation only requires reorienting attention heads and projection pathways along low-rank trajectories.
        """)
    with st.expander("Q3: How does PEFT enable cost-effective multi-tenant serving?"):
        st.write("""
        In multi-tenant systems, loading separate full fine-tuned models for each customer is cost-prohibitive.
        With PEFT, the deployment infrastructure loads a single shared frozen base model in GPU memory. 
        When a request arrives for Customer A, the inference engine applies Customer A's 30MB adapter weights on the fly (via dynamic kernel merging or batched LoRA routing in vLLM/SGLang).
        """)

with q2:
    with st.expander("Q4: Can frozen weights be quantized without losing fine-tuning accuracy?"):
        st.write("""
        Yes, this is the exact principle behind QLoRA.
        Because frozen weights are never updated, they do not suffer from optimizer quantization noise or rounding drift. 
        Quantizing frozen weights to 4-bit (NF4) reduces weight memory by 75%, while the small adapter remains in 16-bit precision to maintain gradient fidelity.
        """)
    with st.expander("Q5: What are the three primary families of PEFT methods?"):
        st.write("""
        1. **Reparameterization-based**: LoRA and DoRA (inject low-rank decomposition matrices parallel to existing layers).
        2. **Additive-based**: Bottleneck Adapters (inject small feed-forward layers between transformer blocks).
        3. **Prefix / Prompt-based**: Prefix Tuning and Prompt Tuning (prepend learnable virtual token embeddings to input sequences or key/value caches).
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Accidentally Training Unfrozen Embeddings**
    * **Failure**: Forgetting to freeze the token embedding layer (`wte` / `embed_tokens`).
    * **Impact**: Token embeddings have massive parameter counts (`vocab_size * d_model = 128,000 * 4096 = 524M params`). Unfreezing them creates huge AdamW state allocations, triggering instant OOM!
    * **Remedy**: Always verify `modules_to_save` in your PEFT configuration and ensure `embed_tokens` remains frozen unless adding new vocabulary.
    """)
    st.error("""
    **🚨 Production Trap 2: Missing requires_grad Verification**
    * **Failure**: Assuming `peft_model = get_peft_model(model, config)` automatically froze the base weights without checking.
    * **Impact**: If a custom model class or wrapper is used, base weights may remain trainable, quietly executing full fine-tuning and running out of GPU memory.
    * **Remedy**: Always print `model.print_trainable_parameters()` before launching the training loop.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Excessive LoRA Rank Setting**
    * **Failure**: Setting LoRA rank `r = 256` or `r = 512` assuming "higher rank is always better".
    * **Impact**: Increases parameter count unnecessarily, triggers overfitting on small datasets, and increases latency during inference.
    * **Remedy**: Stick to `r = 8` or `r = 16` for standard SFT tasks; rarely exceed `r = 64` even for complex coding tasks.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Merging Adapters at Wrong Precision**
    * **Failure**: Merging a 16-bit LoRA adapter into an 8-bit or 4-bit quantized base model.
    * **Impact**: Destroys weight distribution, introducing severe numerical artifacts and gibberish outputs.
    * **Remedy**: Always dequantize the base model to 16-bit (BF16/FP16) before performing `merge_and_unload()`.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Typical PEFT Parameter Ratio"):
        st.write("What fraction of parameters is typically trainable in a LoRA PEFT setup?")
        st.info("Between 0.05% and 0.5% of total base model parameters.")
    with st.expander("⚡ Flashcard 2: Optimizer Memory for Frozen Weights"):
        st.write("How much optimizer memory is allocated for frozen parameters (`requires_grad=False`)?")
        st.info("Zero bytes.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Checkpoint Size Ratio"):
        st.write("How does a PEFT adapter checkpoint size compare to a full model checkpoint?")
        st.info("~20MB - 100MB vs. ~15GB - 30GB (typically 100x to 500x smaller).")
    with st.expander("⚡ Flashcard 4: Primary Serving Advantage"):
        st.write("How does PEFT improve multi-tenant serving architectures?")
        st.info("Allows serving 1 shared frozen base model and dynamically routing requests to client adapters in VRAM.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of PEFT Concepts before advancing to Lesson 14.2 (LoRA):
""")

chk1 = st.checkbox("I understand why freezing base weights eliminates optimizer memory allocation.", value=True)
chk2 = st.checkbox("I can calculate the static VRAM required for full fine-tuning vs PEFT on a 7B model.", value=True)
chk3 = st.checkbox("I understand the Intrinsic Rank Hypothesis that justifies low-rank adaptation.", value=True)
chk4 = st.checkbox("I know how multi-tenant serving leverages dynamic adapter swapping on a single base model.", value=True)
chk5 = st.checkbox("I know why unfreezing token embeddings can accidentally trigger an out-of-memory crash.", value=True)
chk6 = st.checkbox("I understand the 3 primary PEFT families: reparameterization, additive, and prompt/prefix.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 14.2: LoRA.")
