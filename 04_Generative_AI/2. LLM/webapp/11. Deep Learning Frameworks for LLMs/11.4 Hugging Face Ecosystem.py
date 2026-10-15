"""
11.4 Hugging Face Ecosystem
===========================
Interactive Master Lesson exploring Transformers, Tokenizers, Datasets, Accelerate,
PEFT (LoRA/QLoRA), TRL (DPO/PPO), Safetensors, and the Open-Source Model Hub.
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
    page_title="11.4 Hugging Face Ecosystem - LLM Architecture",
    page_icon="🤗",
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
st.title("🤗 11.4 Hugging Face Ecosystem")
st.caption(
    "Chapter 11: Deep Learning Frameworks for LLMs | Module 4 of 4"
)

st.markdown(
    """
The Hugging Face ecosystem is the open-source operating system of the modern AI revolution.
Connecting multi-petabyte datasets (**Datasets**), ultra-fast Rust tokenization (**Tokenizers**),
standardized model architectures (**Transformers**), distributed cluster orchestrators (**Accelerate**),
parameter-efficient fine-tuning (**PEFT**), and post-training alignment (**TRL**), Hugging Face provides
the unified toolchain for developing, fine-tuning, and deploying frontier foundation models.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & OPEN-SOURCE STACK
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Open-Source LLM Ecosystem", divider="rainbow")

st.markdown(
    """
### The Unified Hugging Face Lifecycle Stack
The ecosystem spans the full lifecycle from raw data ingestion to post-training human alignment and hub distribution:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "hf_ecosystem_stack",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("hub", "Hugging Face Hub\\nModels, Datasets, Safetensors", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("data", "Datasets & Tokenizers\\nArrow Zero-Copy, Rust BPE", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("accel", "Accelerate Engine\\nFSDP, DeepSpeed, Multi-GPU", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("trans", "Transformers Library\\nAutoModelForCausalLM", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("peft", "PEFT & TRL\\nLoRA, QLoRA, SFT, DPO", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("hub", "data", label="Stream Data")
arch_dot.edge("data", "accel", label="Batched Tokens")
arch_dot.edge("accel", "trans", label="Distributed Execution")
arch_dot.edge("trans", "peft", label="Base Weights")
arch_dot.edge("peft", "hub", label="Push Checkpoints")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Safetensors Over Pickle**: PyTorch historically serialized model weights using Python's `pickle` library, exposing users to arbitrary remote code execution exploits. Hugging Face developed **Safetensors**: a memory-mapped, zero-copy, cryptographically safe binary tensor format that has become the universal industry standard for all open weights.
* **Direct Alignment via TRL & DPO**: Rather than building complex multi-model PPO reinforcement learning loops with separate reward models, modern practitioners utilize Hugging Face **TRL (Transformer Reinforcement Learning)** with **Direct Preference Optimization (DPO)** to align models directly on preference pairs.
"""
)

# Ecosystem Library Matrix Table
hf_matrix_df = pd.DataFrame(
    {
        "Library": ["transformers", "tokenizers", "datasets", "accelerate", "peft", "trl"],
        "Core Abstraction": ["AutoModelForCausalLM", "TokenizerFast (Rust)", "Dataset (Apache Arrow)", "Accelerator", "LoraConfig, get_peft_model", "SFTTrainer, DPOTrainer"],
        "Primary Functionality": ["Model architectures & generation", "Subword segmentation (BPE/Unigram)", "Zero-copy streaming data loading", "Hardware-agnostic distributed launches", "Low-Rank Adaptation (<1% params)", "Supervised fine-tuning & RLHF alignment"],
        "Underlying Backend": ["PyTorch / FlashAttention", "Pure Rust (Rayon SIMD)", "PyArrow C++ / Memory mapping", "torch.distributed / FSDP / DeepSpeed", "torch.nn Linear hooks", "Accelerate + Transformers"],
    }
)
st.dataframe(hf_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: HUGGING FACE TRANSFORMERS CORE ARCHITECTURE
# -----------------------------------------------------------------------------
st.header("2. Hugging Face Transformers Core Architecture (AutoModel, AutoConfig, Pipelines)", divider="rainbow")

st.markdown(
    """
The `transformers` library organizes model interactions through standardized unified interfaces:

### The Auto-Class Factory Pattern
* **`AutoConfig`**: Loads model architectural metadata from `config.json` (`hidden_size`, `num_attention_heads`, `rope_theta`).
* **`AutoModelForCausalLM`**: Dynamically inspects the configuration architecture and instantiates the correct causal LM class (e.g. `LlamaForCausalLM`, `MistralForCausalLM`).
* **`AutoTokenizer`**: Downloads and instantiates the matching subword tokenizer and vocabulary mapping.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
```
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: HUGGING FACE TOKENIZERS
# -----------------------------------------------------------------------------
st.header("3. Hugging Face Tokenizers (Fast Rust Tokenizers & Byte-Pair Encoding)", divider="rainbow")

st.markdown(
    """
Tokenizing billions of tokens in pure Python is a major CPU bottleneck.
Hugging Face's `tokenizers` library is authored entirely in **Rust**:
* **Rayon Multi-Threading**: Processes text chunks concurrently across all available CPU cores.
* **Zero-Copy Byte Handling**: Evaluates byte-pair merges directly in memory buffers.
* **Pre-Tokenization & Normalization**: Executes Unicode NFC normalization and regex pre-tokenization in microseconds, achieving throughput exceeding **1 million words per second** on modern CPUs.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: HUGGING FACE DATASETS
# -----------------------------------------------------------------------------
st.header("4. Hugging Face Datasets (Arrow Memory-Mapping & Streaming Terabyte Corpora)", divider="rainbow")

st.markdown(
    """
The `datasets` library eliminates RAM limitations through **Apache Arrow**:
* **Zero-Copy Memory Mapping**: Datasets stored on NVMe SSDs are memory-mapped into user space using PyArrow. Reading a 500GB dataset consumes almost zero RAM because columns are paged in directly by the operating system kernel.
* **Dataset Streaming (`streaming=True`)**: Iterates through multi-terabyte web archives (e.g. FineWeb) record-by-record directly from HTTP without downloading the archive to local disk.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: HUGGING FACE ACCELERATE
# -----------------------------------------------------------------------------
st.header("5. Hugging Face Accelerate (Multi-GPU Distributed & FSDP Launch)", divider="rainbow")

st.markdown(
    """
`accelerate` abstracts distributed infrastructure:
* The same PyTorch training loop runs seamlessly on a single GPU, multi-GPU DDP, multi-node FSDP, or DeepSpeed ZeRO-3 without code changes.
* Instead of manual device management (`.to(device)`), users wrap components with `accelerator.prepare(model, optimizer, dataloader)`.
* Launches via the CLI: `accelerate launch --config_file fsdp_config.yaml train.py`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: PARAMETER-EFFICIENT FINE-TUNING (PEFT, LORA & QLORA)
# -----------------------------------------------------------------------------
st.header("6. Parameter-Efficient Fine-Tuning (PEFT, LoRA & QLoRA Integration)", divider="rainbow")

st.markdown(
    """
Full fine-tuning of a 70B model requires updating and storing gradients for all 70 billion parameters (requiring multiple 80GB GPUs).
**PEFT** freezes the base model and injects lightweight trainable parameters:

### Low-Rank Adaptation (LoRA, Hu et al., 2021)
Decomposes parameter updates into low-rank factor matrices:

```text
W = W_0 + Delta_W = W_0 + (B * A) * (alpha / r)
```
Where:
* `W_0 in R^[d, k]` is frozen.
* `A in R^[r, k]` is initialized with Gaussian noise.
* `B in R^[d, r]` is initialized to zeros.
* `r << d` (e.g. rank `r = 16` while `d = 4096`).
* **Memory Savings**: Trains **less than 0.2%** of total parameters while matching full fine-tuning accuracy.

### QLoRA (Dettmers et al., 2023)
Quantizes the frozen base model to 4-bit NormalFloat (NF4), allowing a 70B model to be fine-tuned on a single 48GB GPU!
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: TRANSFORMER REINFORCEMENT LEARNING (TRL & DPO)
# -----------------------------------------------------------------------------
st.header("7. Transformer Reinforcement Learning (TRL, SFTTrainer & DPO Alignments)", divider="rainbow")

st.markdown(
    """
**TRL (Transformer Reinforcement Learning)** provides the post-training alignment stack:
1. **`SFTTrainer`**: High-performance Supervised Fine-Tuning supporting sequence packing and PEFT integration out-of-the-box.
2. **`DPOTrainer`**: Direct Preference Optimization. Eliminates the complex reinforcement learning reward model by mathematically reformulating policy optimization as a binary cross-entropy loss over chosen vs. rejected completions:
   ```text
   L_DPO = - log sigma( beta * log( pi_theta(y_w | x) / pi_ref(y_w | x) ) - beta * log( pi_theta(y_l | x) / pi_ref(y_l | x) ) )
   ```
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: HUGGING FACE MODEL HUB
# -----------------------------------------------------------------------------
st.header("8. Hugging Face Model Hub (Model Cards, Safetensors, Quantized GGUF/AWQ)", divider="rainbow")

st.markdown(
    """
The Model Hub acts as the global registry for foundation models:
* **`safetensors` Format**: Fast, memory-mapped, safe serialization preventing pickle vulnerabilities.
* **GGUF Format**: Quantized CPU/Mac inference format for llama.cpp.
* **AWQ & GPTQ**: 4-bit GPU inference formats optimized for vLLM serving.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE HUGGING FACE TRAINING PIPELINE
# -----------------------------------------------------------------------------
st.header("9. Production-Grade Complete Hugging Face Training & Alignment Pipeline", divider="rainbow")

st.markdown(
    """
A complete, production-ready script demonstrating LoRA configuration, model wrapping, and SFT training with PEFT:
"""
)

code_snippet = """from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

def train_lora_model(
    base_model_name: str = "meta-llama/Meta-Llama-3-8B",
    output_dir: str = "./lora_output",
):
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 3. Configure LoRA Parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                         # Rank dimension
        lora_alpha=32,                # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 4. Wrap with PEFT Adapter Hooks
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,
        save_strategy="epoch",
    )

    return model, peft_config, training_args
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE HUGGING FACE LORA WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive Hugging Face Library Ecosystem & LoRA Calculator Workbench", divider="rainbow")

st.markdown(
    """
Calculate exact trainable parameter percentages and VRAM savings when applying **LoRA Adapters**
across different foundation model architectures.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    hf_model_choice = st.selectbox("Base Model Architecture", options=["Llama 3 8B (32L, 4096D)", "Llama 3 70B (80L, 8192D)", "Mistral 7B (32L, 4096D)"], index=0)
    lora_r = st.select_slider("LoRA Rank (r)", options=[4, 8, 16, 32, 64], value=16)
with col_ctrl2:
    lora_targets = st.multiselect("Target Linear Modules", options=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], default=["q_proj", "v_proj"])
    quant_base = st.selectbox("Base Model Precision", options=["BF16 (16-bit)", "4-Bit NF4 (QLoRA)"], index=1)

# Model Specs
if "70B" in hf_model_choice:
    layers, d_model, base_params = 80, 8192, 70e9
else:
    layers, d_model, base_params = 32, 4096, 8e9

# Calculate LoRA parameters: 2 * (d_model * r) per target module per layer
modules_count = len(lora_targets)
lora_params = layers * modules_count * (2 * d_model * lora_r)
lora_pct = (lora_params / base_params) * 100.0

# VRAM Estimates
base_vram_gb = (base_params * 0.5) / (1024 ** 3) if "4-Bit" in quant_base else (base_params * 2.0) / (1024 ** 3)
lora_vram_gb = (lora_params * 16.0) / (1024 ** 3)  # Optimizer + Grads for LoRA params only

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Trainable LoRA Parameters", f"{lora_params / 1e6:.2f} Million", f"{lora_pct:.3f}% of Base Model")
with col_m2:
    st.metric("Base Weights VRAM", f"{base_vram_gb:.1f} GB")
with col_m3:
    st.metric("Adapter + Optimizer VRAM", f"{lora_vram_gb:.2f} GB", f"Total: {base_vram_gb + lora_vram_gb:.1f} GB")

st.subheader("Fine-Tuning Hardware Footprint Comparison")
st.dataframe(
    pd.DataFrame(
        {
            "Strategy": ["Full Fine-Tuning (BF16)", "Standard LoRA (BF16 Base)", "QLoRA (4-bit NF4 Base)"],
            "Trainable Params": [f"{base_params / 1e9:.1f} Billion (100%)", f"{lora_params / 1e6:.1f} Million ({lora_pct:.2f}%)", f"{lora_params / 1e6:.1f} Million ({lora_pct:.2f}%)"],
            "VRAM Required": [f"~{base_params * 16 / (1024**3):.0f} GB (Multi-GPU)", f"~{base_vram_gb * 1.3:.1f} GB", f"~{base_vram_gb + lora_vram_gb + 2.0:.1f} GB (Single GPU!)"],
            "Hardware Feasibility": ["Multi-Node Supercluster", "Dual A100 / H100", "Consumer RTX 4090 / A10G"],
        }
    ),
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Mathematically, how does LoRA reduce fine-tuning memory?"):
    st.markdown(
        """
        **Answer:**
        Instead of learning a full parameter update matrix `Delta_W in R^[d, k]` (which requires storing gradients and optimizer states for `d * k` weights), LoRA decomposes the update into two low-rank matrices:
        `Delta_W = B * A`, where `A in R^[r, k]` and `B in R^[d, r]` with rank `r << min(d, k)`.
        For `d = 4096` and `r = 16`, parameter count drops from `16,777,216` to `131,072` (a **128x parameter reduction**).
        Because the base weights `W_0` are frozen, zero optimizer states or gradients are stored for them.
        """
    )

with st.expander("Q2: Why is the Safetensors format superior to PyTorch's default .pt / .bin checkpoints?"):
    st.markdown(
        """
        **Answer:**
        1. **Security**: PyTorch `.pt` uses Python `pickle`, which can execute arbitrary shell commands upon deserialization. Safetensors is pure binary data with zero executable code.
        2. **Speed (Zero-Copy Memory Mapping)**: Safetensors supports direct `mmap`, mapping model tensors directly into memory without deserialization parsing, loading weights 2x to 5x faster.
        3. **Header Metadata**: Stores clean JSON headers allowing partial weight inspection without loading the full file.
        """
    )

with st.expander("Q3: What is the difference between Direct Preference Optimization (DPO) and PPO in TRL?"):
    st.markdown(
        """
        **Answer:**
        * **PPO (Proximal Policy Optimization)**: Requires 4 separate models running simultaneously: Actor (policy), Critic (value), Reward Model, and Reference Model. High memory overhead and fragile hyperparameter stability.
        * **DPO (Direct Preference Optimization)**: Mathematically derives the optimal policy directly from the reward function, proving that human preference alignment can be optimized using a simple binary cross-entropy loss over chosen vs. rejected pairs. Requires only the policy model and frozen reference model.
        """
    )

with st.expander("Q4: How does QLoRA achieve 4-bit fine-tuning without losing precision?"):
    st.markdown(
        """
        **Answer:**
        QLoRA introduces:
        1. **NormalFloat4 (NF4)**: An information-theoretically optimal quantile quantization data type for normally distributed weights.
        2. **Double Quantization**: Quantizes the quantization constants themselves, saving 0.37 bits per parameter.
        3. **Paged Optimizers**: Offloads optimizer state spikes to CPU RAM during activation surges.
        """
    )

with st.expander("Q5: What role does Hugging Face Accelerate play in distributed training?"):
    st.markdown(
        """
        **Answer:**
        Accelerate decouples the PyTorch training loop from the specific distributed execution engine.
        The exact same script can be launched locally on a MacBook, on a multi-GPU workstation with DDP, or on a 100-node cluster using DeepSpeed ZeRO-3 or PyTorch FSDP simply by changing the launch configuration file (`accelerate launch`).
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Forgetting to Add Chat Templates to Prompts**
        *Root Cause*: Fine-tuning or running inference on conversational data without applying `tokenizer.apply_chat_template()`.
        *Production Fix*: Causes the model to miss special system delimiters (`<|start_header_id|>`), degrading instruction-following performance.
        """
    )
    st.warning(
        """
        **Trap 2: Mismatched Padding Token in Tokenizer**
        *Root Cause*: Llama 3 does not define a default `pad_token`, causing `tokenizer.pad_token_id` to be `None`.
        *Production Fix*: Set `tokenizer.pad_token = tokenizer.eos_token` or register a dedicated `<|finetune_right_pad_id|>` token.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Merging LoRA Adapters with Quantized 4-Bit Weights**
        *Root Cause*: Calling `.merge_and_unload()` on a 4-bit QLoRA model directly.
        *Production Fix*: Merging requires dequantizing weights back to FP16/BF16 first, otherwise quantization rounding destroys the adapter weights.
        """
    )
    st.warning(
        """
        **Trap 4: Loading Datasets Without Streaming on Limited Disk**
        *Root Cause*: Running `load_dataset("HuggingFaceFW/fineweb")` without `streaming=True` on a local machine.
        *Production Fix*: Attempts to download 44 terabytes of data to local disk, immediately exhausting storage. Always stream massive web datasets.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: LoRA Target Modules"):
        st.write("Which modules in a transformer are typically targeted by LoRA?")
        st.info("Query, Key, Value, and MLP projection matrices (q_proj, v_proj, etc.).")

    with st.expander("⚡ Flashcard 2: Safetensors Primary Benefit"):
        st.write("Why was Safetensors developed to replace PyTorch .bin/.pt files?")
        st.info("To eliminate arbitrary code execution vulnerabilities and enable fast zero-copy memory mapping.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: DPO vs PPO Complexity"):
        st.write("Why is DPO simpler than PPO for post-training alignment?")
        st.info("Eliminates the separate reward and critic models, training directly on preference pairs via binary cross-entropy.")

    with st.expander("⚡ Flashcard 4: Fast Tokenizer Language"):
        st.write("What programming language powers Hugging Face's Fast Tokenizers?")
        st.info("Pure Rust.")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the unified ecosystem mapping across Transformers, Tokenizers, Datasets, and Accelerate.",
    "I can explain why Safetensors is secure and loads faster than standard PyTorch pickle serialization.",
    "I understand the mathematical formulation of LoRA low-rank decomposition Delta_W = B * A.",
    "I know how QLoRA enables fine-tuning 70B models on consumer GPUs using 4-bit NF4 quantization.",
    "I can explain how DPO simplifies RLHF by eliminating separate reward and critic models.",
    "I know how to set up an end-to-end SFT training pipeline using TRL and PEFT.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered the Hugging Face Ecosystem, PEFT, LoRA, and TRL Alignment Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
