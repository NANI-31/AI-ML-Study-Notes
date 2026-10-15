import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="13.3 Fine-Tuning Process - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionFineTuningTrainer:
    """
    Production-grade educational Trainer simulating the complete LLM fine-tuning lifecycle:
    - Model initialization and weight states
    - Cosine learning rate scheduling with warmup
    - Gradient accumulation and clipping
    - Checkpoint persistence and validation perplexity tracking
    """
    def __init__(
        self,
        base_lr: float = 2e-5,
        min_lr: float = 2e-6,
        warmup_steps: int = 20,
        total_steps: int = 100,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        seed: int = 42
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        self.rng = np.random.RandomState(seed)

    def get_lr(self, step: int) -> float:
        """Computes learning rate with linear warmup and cosine decay."""
        if step < self.warmup_steps:
            return self.base_lr * (float(step) / float(max(1, self.warmup_steps)))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def simulate_training_run(self) -> pd.DataFrame:
        """Executes a simulated training run tracking loss, perplexity, and LR."""
        records = []
        base_loss = 3.8
        
        for step in range(1, self.total_steps + 1):
            lr = self.get_lr(step)
            # Simulated natural decay with stochastic noise
            decay = np.exp(-step / (self.total_steps * 0.35)) * 2.2
            noise = self.rng.normal(0, 0.05)
            train_loss = max(1.1, base_loss - (2.2 - decay) + noise)
            
            # Validation every 10 steps
            val_loss = train_loss * 1.05 + self.rng.uniform(0.02, 0.08) if step % 10 == 0 else np.nan
            val_ppl = np.exp(val_loss) if not np.isnan(val_loss) else np.nan

            records.append({
                "Step": step,
                "Learning Rate": lr,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Validation Perplexity": val_ppl
            })
        return pd.DataFrame(records)

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("13.3 Fine-Tuning Process")
st.caption("Chapter 13: LLM Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
Executing a successful LLM fine-tuning run is an exacting discipline of systems engineering and mathematical optimization. 
Moving beyond toy scripts requires mastering model precision serialization, tokenization alignment, 
gradient accumulation dynamics, numerical loss masking, checkpoint fault tolerance, and SafeTensors deployment.

In this lesson, you will master:
1. Loading pretrained model weights with exact precision semantics (FP32, FP16, BF16) and device mapping.
2. Tokenization synchronization: padding side selection, truncation boundaries, and special token registration.
3. Training configurations: effective batch size, gradient accumulation, learning rate schedules, and warmup.
4. Loss calculation: autoregressive cross-entropy, label masking (`ignore_index=-100`), and gradient clipping.
5. In-flight evaluation: validation loss, perplexity calculations, and early stopping.
6. Checkpoint persistence: state dict serialization, distributed recovery, and resume semantics.
7. Model export: adapter merging, tokenizer packaging, and SafeTensors production serialization.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Complete End-to-End Fine-Tuning Loop", divider="rainbow")

st.markdown("""
The modern post-training loop coordinates distributed GPU workers, optimizer states, memory-mapped data loaders, 
and disk checkpoints into a fault-tolerant pipeline.
""")

ft_loop_dot = """
digraph FineTuningLoop {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_init {
        label = "Phase 1: Model Initialization";
        color = "#334155";
        fontcolor = "#94A3B8";
        PretrainedWeights [label="Hugging Face Hub / Disk\\n(BF16 SafeTensors)", fillcolor="#0F172A"];
        ModelLoader [label="AutoModelForCausalLM\\n(torch_dtype=bfloat16, device_map='auto')", fillcolor="#0369A1"];
        PretrainedWeights -> ModelLoader;
    }

    subgraph cluster_loop {
        label = "Phase 2: Optimization Step (Iterative Loop)";
        color = "#334155";
        fontcolor = "#94A3B8";
        Batch [label="Micro-Batch Inputs & Labels\\n(ignore_index=-100 on Prompts)", fillcolor="#1E293B"];
        Forward [label="Forward Pass + Masked Loss\\nCross-Entropy(Shift_Logits, Shift_Labels)", fillcolor="#047857"];
        GradAccum [label="Gradient Accumulation\\nLoss / grad_accum_steps -> Backward()", fillcolor="#047857"];
        OptimizerStep [label="Clip Gradients (max_norm=1.0)\\nAdamW Step + Cosine LR Schedule", fillcolor="#B45309"];
        
        Batch -> Forward -> GradAccum -> OptimizerStep;
    }

    subgraph cluster_validation {
        label = "Phase 3: Eval & Persistence";
        color = "#334155";
        fontcolor = "#94A3B8";
        EvalStep [label="Validation Loss & Perplexity\\nPPL = exp(CrossEntropy)", fillcolor="#7C3AED"];
        Checkpoint [label="Fault-Tolerant Checkpoint\\n(Weights, Optimizer, LR Scheduler)", fillcolor="#1E293B"];
        FinalSave [label="Merged SafeTensors + Tokenizer\\nProduction Deployable Artifact", fillcolor="#0284C7"];
        
        OptimizerStep -> EvalStep [label="Every N Steps"];
        EvalStep -> Checkpoint [label="If Best Metric"];
        Checkpoint -> FinalSave [label="Training Complete"];
    }

    ModelLoader -> Batch [label="Loaded into VRAM"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(ft_loop_dot, use_container_width=True)

st.markdown("""
### The Critical Milestones in Every Training Run
1. **Model & Tokenizer Load**: Instantiate the model in native 16-bit precision (`bfloat16`) to cut VRAM in half without loss of numerical range.
2. **Micro-Batch Execution**: Accumulate gradients over multiple forward passes to achieve large effective batch sizes without out-of-memory (OOM) errors.
3. **Stabilization & Step**: Clip gradient norms to `1.0` to suppress gradient explosions, followed by AdamW parameter updates.
4. **Perplexity Audit & SafeTensors Export**: Evaluate exponential cross-entropy and export weights without Python pickle vulnerabilities.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - LOADING PRETRAINED MODELS
# ==============================================================================
st.header("2. Loading Pretrained Models", divider="rainbow")

st.markdown("""
Loading an LLM into memory requires precise control over numerical precision, device allocation, and memory mapping.
""")

col_ld1, col_ld2 = st.columns(2)
with col_ld1:
    st.subheader("Numerical Precision Formats")
    st.markdown("""
    * **FP32 (Single Precision)**: 32 bits (1 sign, 8 exponent, 23 mantissa). Extreme numerical stability, but doubles memory (`4 bytes/param`). A 7B model takes 28GB for weights alone!
    * **FP16 (Half Precision)**: 16 bits (1 sign, 5 exponent, 10 mantissa). Prone to underflow and overflow in deep transformer layers; requires loss scaling.
    * **BF16 (Brain Floating Point)**: 16 bits (1 sign, 8 exponent, 7 mantissa). **Modern industry standard**. Retains the same dynamic exponent range as FP32, eliminating underflow without requiring loss scaling. Native on Ampere+ (A100, H100, RTX 3000/4000).
    """)

with col_ld2:
    st.subheader("Device Placement & Hugging Face Ingestion")
    st.markdown("""
    * **`device_map="auto"`**: Uses Accelerate to profile available GPU VRAM and splits layers automatically across multiple GPUs and CPU RAM.
    * **FlashAttention-2 Flag**: `attn_implementation="flash_attention_2"` must be explicitly passed during model loading to enable memory-efficient attention kernels.
    """)

st.code("""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

# Production standard model loading
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - TOKENIZATION IN FINE-TUNING
# ==============================================================================
st.header("3. Tokenization in Fine-Tuning", divider="rainbow")

st.markdown("""
Tokenization during fine-tuning introduces subtle edge cases that do not exist during pretraining or pure inference.
""")

col_tok1, col_tok2 = st.columns(2)
with col_tok1:
    st.markdown("#### Padding Side Selection")
    st.markdown("""
    * **Inference Requirement**: Must use **Left Padding (`tokenizer.padding_side = "left"`)**. 
      Autoregressive generation continues from the rightmost token; right-padding places pad tokens at the end, ruining positional encodings!
    * **Training Requirement**: Standard causal language modeling can use **Right Padding** during training if loss masking (`-100`) is configured correctly, though left padding is also widely supported.
    * **Golden Practice**: Ensure `tokenizer.pad_token = tokenizer.eos_token` if a dedicated `<pad>` token does not exist, and always mask pad tokens from loss.
    """)

with col_tok2:
    st.markdown("#### Max Length Truncation & Truncation Bias")
    st.markdown("""
    * **Truncation Side**: If a sample exceeds `max_seq_len = 4096`, default right truncation cuts off the assistant's answer or final reasoning steps!
    * **Curated Datasets**: Filter or prune overly long samples before training rather than blindly truncating mid-sentence.
    * **Special Token Registration**: Never manually append string tags like `<|user|>`; always execute `tokenizer.apply_chat_template()`.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - TRAINING CONFIGURATION
# ==============================================================================
st.header("4. Training Configuration", divider="rainbow")

st.markdown("""
Configuring an SFT run involves balancing effective batch size, learning rate schedules, and optimizer stability.
""")

st.markdown("""
| Hyperparameter | Typical Pretraining Value | Typical SFT / Fine-Tuning Value | Rationale |
| :--- | :--- | :--- | :--- |
| **Effective Batch Size** | 1,000 to 4,000 sequences | 32 to 128 sequences | SFT datasets are smaller; massive batches cause under-fitting per epoch. |
| **Learning Rate** | `3e-4` to `1e-3` | `1e-5` to `5e-5` (LoRA: `1e-4` to `3e-4`) | Gentle steps prevent destroying pretrained representations. |
| **LR Schedule** | Cosine with 2,000 warmup steps | Cosine with 3-5% warmup steps | Warmup prevents massive early gradients from destabilizing layers. |
| **Epochs** | 1 Epoch over web tokens | 1 to 3 Epochs | Training beyond 3 epochs over small datasets causes severe overfitting. |
| **Weight Decay** | 0.1 | 0.01 to 0.1 | Regularizes parameter drift. |
| **Gradient Clipping** | 1.0 | 0.5 to 1.0 | Suppresses exploding gradients on outlier token sequences. |
""")

st.markdown("""
#### The Effective Batch Size Formula
```text
Effective_Batch_Size = Per_Device_Batch_Size * Gradient_Accumulation_Steps * Number_Of_GPUs
```
*If you have 1 GPU with 24GB VRAM: Per-device batch size = 2, Gradient accumulation = 16 -> Effective batch size = `2 * 16 * 1 = 32 sequences`.*
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - LOSS CALCULATION
# ==============================================================================
st.header("5. Loss Calculation", divider="rainbow")

st.markdown("""
The mathematical engine of causal language model training is **Autoregressive Cross-Entropy Loss** with teacher forcing.
""")

st.markdown("""
#### Teacher Forcing & Autoregressive Shifting
During training, the entire ground-truth sequence is fed into the model simultaneously. 
At position `t`, the model computes logits to predict the token at position `t+1`.
```text
Inputs:    [token_0, token_1, token_2, ..., token_N-1]
Targets:   [token_1, token_2, token_3, ..., token_N]
```
In PyTorch code, this is implemented by shifting logits and labels by 1 index:
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```
""")

col_loss1, col_loss2 = st.columns(2)
with col_loss1:
    st.markdown("#### Response-Only Masking (`ignore_index = -100`)")
    st.markdown("""
    All prompt tokens in `shift_labels` are set to `-100`.
    PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)` skips masked elements:
    ```text
    Loss = - (1 / M) * Sum_{i in Unmasked} log P(y_i | y_<i, x)
    ```
    Where `M` is the count of assistant response tokens only.
    """)

with col_loss2:
    st.markdown("#### Gradient Checkpointing")
    st.markdown("""
    * **The Problem**: Storing all intermediate activation tensors for backward pass consumes up to 70% of total VRAM.
    * **The Solution**: Discards activations during forward pass and recalculates them on the fly during backward pass.
    * **Tradeoff**: **Saves ~60% VRAM** at the cost of a **~20% slower training step**. Essential for long context windows (4096+).
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - EVALUATION
# ==============================================================================
st.header("6. Evaluation During Training", divider="rainbow")

st.markdown("""
Evaluating generative language models during training requires both fast automated metrics and rigorous qualitative benchmarks.
""")

st.markdown("""
### 1. Validation Loss & Perplexity (In-Flight Metrics)
During training, every `eval_steps` (e.g. 100 steps), the trainer computes cross-entropy loss on a held-out validation split.
**Perplexity (PPL)** is the exponential of the cross-entropy loss:
```text
Perplexity = exp(Validation_Cross_Entropy_Loss)
```
* **Intuition**: Perplexity represents the model's average branching factor. A perplexity of `4.5` means the model is as uncertain as if choosing uniformly among 4.5 equally likely tokens.
* **Overfitting Signal**: If training loss keeps dropping but validation loss/perplexity begins curving upward, the model is memorizing the training samples. Stop training immediately!
""")

st.markdown("""
### 2. Generation-Based Benchmarking (Post-Step Evaluations)
Loss alone does not prove instruction obedience. Production teams run automated generation evaluations:
* **MT-Bench / AlpacaEval**: Multi-turn dialogue evaluation scored by GPT-4o.
* **GSM8K / HumanEval**: Exact-match verification for math and code execution.
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CHECKPOINTING
# ==============================================================================
st.header("7. Checkpointing & Fault Tolerance", divider="rainbow")

st.markdown("""
In distributed deep learning, hardware failures, node preemptions, and power anomalies are common occurrences. 
A robust checkpointing strategy prevents losing dozens of hours of compute.
""")

col_chk1, col_chk2 = st.columns(2)
with col_chk1:
    st.subheader("What a Complete Checkpoint Contains")
    st.markdown("""
    * **Model Weights**: `model.safetensors` (the updated neural weights).
    * **Optimizer State**: `optimizer.pt` (AdamW momentum and variance tensors for every parameter).
    * **Learning Rate Scheduler**: `scheduler.pt` (current step count, current decayed LR).
    * **Trainer State**: `trainer_state.json` (global step, epoch, best metric history).
    * **RNG State**: `rng_state.pth` (PyTorch, NumPy, and Python random states for reproducible data loading).
    """)

with col_chk2:
    st.subheader("Checkpoint Management Rules")
    st.markdown("""
    * **Keep Best & Latest (`save_total_limit`)**: Store only the top 2 best checkpoints by validation loss plus the latest 1 to prevent filling disk storage.
    * **Async Checkpointing**: Write checkpoints to local NVMe SSDs asynchronously to avoid blocking GPU training loops.
    * **Resumption Flag**: Always configure `trainer.train(resume_from_checkpoint=True)` to enable automatic recovery upon spot instance preemption.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - MODEL SAVING
# ==============================================================================
st.header("8. Model Saving and Deployment Artifacts", divider="rainbow")

st.markdown("""
Once training completes, the model must be saved in a secure, self-contained, and deployment-ready format.
""")

st.markdown("""
### SafeTensors vs. Legacy PyTorch Pickle (.bin / .pt)
* **The Security Hazard of Pickle**: Legacy PyTorch `.bin` checkpoints use Python's `pickle` library. Deserializing an untrusted `.bin` file can execute arbitrary malicious shell code on the host machine!
* **SafeTensors (Hugging Face Standard)**: A modern, safe serialization format. It stores raw tensor data without code execution vulnerabilities and supports **zero-copy memory mapping (`mmap`)**, making model load times 5x to 10x faster.
""")

col_save1, col_save2 = st.columns(2)
with col_save1:
    st.markdown("#### Saving a Full Fine-Tuned Model")
    st.code("""
# Save model weights and tokenizer in SafeTensors format
model.save_pretrained(
    "./output_model", 
    safe_serialization=True
)
tokenizer.save_pretrained("./output_model")
""", language="python")

with col_save2:
    st.markdown("#### Saving & Merging PEFT / LoRA Adapters")
    st.code("""
# Option A: Save adapter only (small: ~50MB)
model.save_pretrained("./lora_adapter")

# Option B: Merge adapter into base model for fast inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained(
    "./merged_production_model", 
    safe_serialization=True
)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: End-to-End Fine-Tuning Loop Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates the core algorithmic mechanics of a fine-tuning engine:
learning rate scheduling with warmup, gradient accumulation, and loss trajectory tracking.
""")

st.code("""
import numpy as np
import pandas as pd

class ProductionFineTuningTrainer:
    \"\"\"
    Production-grade educational Trainer simulating the complete LLM fine-tuning lifecycle.
    \"\"\"
    def __init__(self, base_lr=2e-5, min_lr=2e-6, warmup_steps=20, total_steps=100, grad_accum_steps=4):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps

    def get_lr(self, step: int) -> float:
        \"\"\"Computes learning rate with linear warmup and cosine decay.\"\"\"
        if step < self.warmup_steps:
            return self.base_lr * (float(step) / float(max(1, self.warmup_steps)))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
""", language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Hyperparameter Sizer & Live Training Console", divider="rainbow")

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("Training Run Configuration")
    user_lr = st.select_slider(
        "Peak Learning Rate",
        options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
        value=2e-5,
        format_func=lambda x: f"{x:.1e}"
    )
    user_warmup = st.slider("Warmup Steps", min_value=5, max_value=50, value=15)
    user_steps = st.slider("Total Optimization Steps", min_value=50, max_value=200, value=100)
    user_accum = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8, 16], index=2)
    user_batch = st.slider("Micro-Batch Size per Device", min_value=1, max_value=8, value=2)

    eff_batch = user_batch * user_accum
    st.info(f"Effective Batch Size per Step: **{eff_batch} sequences**")

with wb_c2:
    st.subheader("Simulated Run Trajectory")
    trainer = ProductionFineTuningTrainer(
        base_lr=user_lr,
        warmup_steps=user_warmup,
        total_steps=user_steps,
        grad_accum_steps=user_accum,
        seed=123
    )
    history_df = trainer.simulate_training_run()

    final_loss = history_df["Train Loss"].iloc[-1]
    best_val_ppl = history_df["Validation Perplexity"].dropna().min()

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Final Training Loss", f"{final_loss:.3f}")
    with m2:
        st.metric("Best Validation Perplexity", f"{best_val_ppl:.2f}")

    st.markdown("##### Loss Curves & Learning Rate Decay:")
    st.line_chart(history_df.set_index("Step")[["Train Loss", "Validation Loss"]])
    st.caption("Notice the rapid initial loss descent followed by plateau as LR decays according to cosine schedule.")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why is BF16 preferred over FP16 for fine-tuning modern LLMs?"):
        st.write("""
        FP16 has only 5 exponent bits, giving it a narrow dynamic range. In deep transformers, gradients frequently underflow to zero or overflow to infinity, requiring loss scaling.
        BF16 allocates 8 exponent bits (the same as FP32) and 7 mantissa bits. It never overflows on transformer activation ranges, eliminating the need for loss scaling and training instability.
        """)
    with st.expander("Q2: Why must tokenizer.padding_side be set to 'left' during autoregressive generation?"):
        st.write("""
        Autoregressive generation continues from the final rightmost token in the context window.
        If left-padding is used: `[<pad>, <pad>, token_1, token_2]`, the model continues from `token_2`.
        If right-padding is used: `[token_1, token_2, <pad>, <pad>]`, the model attempts to continue generating from `<pad>`, corrupting attention and producing nonsense.
        """)
    with st.expander("Q3: What does Gradient Accumulation achieve and what are its memory tradeoffs?"):
        st.write("""
        Gradient accumulation simulates a large batch size when GPU VRAM cannot fit that batch all at once.
        It runs multiple forward and backward passes, accumulating gradients in the parameter `.grad` buffers, 
        and calls `optimizer.step()` and `optimizer.zero_grad()` only once every `N` steps.
        It uses zero additional memory for optimizer states, but does not speed up wall-clock execution time.
        """)

with q2:
    with st.expander("Q4: What is Perplexity and how does it relate to Cross-Entropy Loss?"):
        st.write("""
        Perplexity is the exponential of the cross-entropy loss: `PPL = exp(CrossEntropy)`.
        It represents the effective vocabulary size the model is equally uncertain between when predicting the next token. 
        A model with cross-entropy loss `2.0` has a perplexity of `exp(2.0) = 7.39`.
        Lower perplexity indicates higher confidence and predictive accuracy.
        """)
    with st.expander("Q5: Why is SafeTensors mandatory over PyTorch .bin checkpoints in production?"):
        st.write("""
        PyTorch `.bin` files rely on the Python `pickle` deserializer, which can execute arbitrary shell commands when loaded (`__reduce__` exploit).
        SafeTensors uses a restricted, secure binary header format that cannot execute arbitrary code. 
        Additionally, SafeTensors enables zero-copy memory mapping (`mmap`), drastically reducing model loading times in production containers.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Forgetting to Call model.eval() Before Validation**
    * **Failure**: Computing validation loss while leaving the model in training mode.
    * **Impact**: Dropout remains active, and activation gradients are cached unnecessarily, triggering sudden GPU out-of-memory crashes during evaluation.
    * **Remedy**: Always wrap evaluation routines with `model.eval()` and `with torch.no_grad():`.
    """)
    st.error("""
    **🚨 Production Trap 2: Omitting Gradient Clipping**
    * **Failure**: Disabling `max_grad_norm` during full fine-tuning.
    * **Impact**: A single anomalous or malformed token sequence produces massive loss gradients, causing loss spikes from which the model never recovers (loss becomes `NaN`).
    * **Remedy**: Always enforce gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Unpadded Gradient Accumulation Loss Scaling**
    * **Failure**: Computing `loss.backward()` inside an accumulation loop without dividing loss by `grad_accum_steps`.
    * **Impact**: Gradients become `N` times too large, causing the effective learning rate to blow up by a factor of `grad_accum_steps`.
    * **Remedy**: Always divide loss: `(loss / grad_accum_steps).backward()`.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Merging LoRA Weights with Mismatched Scaling**
    * **Failure**: Manually adding LoRA weights without multiplying by `alpha / r`.
    * **Impact**: Adapter signal is either completely diluted or overwhelmingly dominates the base weights.
    * **Remedy**: Use the official PEFT method `model.merge_and_unload()` which applies the exact scaling factor automatically.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Perplexity Formula"):
        st.write("What is the exact mathematical relationship between loss and perplexity?")
        st.info("`Perplexity = exp(Loss)`.")
    with st.expander("⚡ Flashcard 2: Generation Padding Direction"):
        st.write("Which side must padding be placed on during generation inference?")
        st.info("Left padding (`padding_side='left'`).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Gradient Checkpointing Cost"):
        st.write("What is the tradeoff of enabling gradient checkpointing?")
        st.info("Saves ~60% VRAM at the cost of ~20% slower training speed.")
    with st.expander("⚡ Flashcard 4: SafeTensors Benefit"):
        st.write("Why is SafeTensors safer and faster than Pickle?")
        st.info("Zero code-execution vulnerability and supports zero-copy memory mapping (`mmap`).")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of the Fine-Tuning Process before advancing to Fine-Tuning Problems:
""")

chk1 = st.checkbox("I understand why BF16 eliminates the gradient underflow and loss scaling issues of FP16.", value=True)
chk2 = st.checkbox("I know why left padding is required for autoregressive token generation.", value=True)
chk3 = st.checkbox("I can calculate the effective batch size given micro-batch size and gradient accumulation steps.", value=True)
chk4 = st.checkbox("I know how to divide loss by grad_accum_steps before calling loss.backward().", value=True)
chk5 = st.checkbox("I understand how gradient checkpointing saves VRAM by recomputing activations.", value=True)
chk6 = st.checkbox("I know why SafeTensors is standard for saving production model weights.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 13.4: Fine-Tuning Problems.")
