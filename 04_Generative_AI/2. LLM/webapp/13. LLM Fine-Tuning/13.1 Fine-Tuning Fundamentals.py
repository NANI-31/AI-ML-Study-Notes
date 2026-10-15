import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="13.1 Fine-Tuning Fundamentals - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionSFTEngine:
    """
    Self-contained educational Supervised Fine-Tuning Engine demonstrating
    autoregressive token generation, response-only loss masking, and gradient step simulation.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 32, ignore_index: int = -100, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ignore_index = ignore_index
        rng = np.random.RandomState(seed)
        self.W_embed = rng.randn(vocab_size, d_model).astype(np.float32) * 0.05
        self.W_head = rng.randn(d_model, vocab_size).astype(np.float32) * 0.05

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Computes projection logits: [Seq_Len, Vocab_Size]."""
        h = self.W_embed[input_ids]
        logits = np.dot(h, self.W_head)
        return logits

    def compute_masked_loss(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, int]:
        """
        Calculates Cross-Entropy Loss with strict response-only label masking.
        Tokens matching ignore_index contribute 0 to the loss and gradient.
        """
        shift_logits = logits[:-1, :]
        shift_labels = labels[1:]
        
        mask = (shift_labels != self.ignore_index)
        active_tokens = int(np.sum(mask))
        if active_tokens == 0:
            return 0.0, 0
            
        valid_logits = shift_logits[mask]
        valid_labels = shift_labels[mask]
        
        # Numerically stable log-softmax
        max_logits = np.max(valid_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(valid_logits - max_logits)
        log_probs = (valid_logits - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
        
        nll = -log_probs[np.arange(len(valid_labels)), valid_labels]
        return float(np.mean(nll)), active_tokens

    def train_step(self, input_ids: np.ndarray, labels: np.ndarray, lr: float = 0.01) -> Dict[str, Any]:
        """Simulates an SFT optimization step and parameter update."""
        logits = self.forward(input_ids)
        loss, active_tokens = self.compute_masked_loss(logits, labels)
        # Update simulation on unembedding projection
        self.W_head -= lr * (np.random.randn(*self.W_head.shape).astype(np.float32) * 0.005)
        return {
            "loss": loss,
            "active_tokens": active_tokens,
            "total_tokens": len(input_ids)
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("13.1 Fine-Tuning Fundamentals")
st.caption("Chapter 13: LLM Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
A raw foundation model fresh from self-supervised pretraining is essentially a statistical text completer. 
When prompted with a query like *"What is the capital of France?"*, an un-tuned base model is just as likely to complete the sequence with 
*"What is the capital of Spain? What is the capital of Italy?"* as it is to answer *"Paris"*. 
To transform raw autocomplete engines into instruction-following, task-specialized, and domain-expert production systems, 
practitioners employ **Supervised Fine-Tuning (SFT)**, **Instruction Tuning**, and **Domain Adaptation**.

In this lesson, you will master:
1. The mathematical definition of fine-tuning as parameter shifts from general web priors to task-specific posteriors.
2. The architectural differences between self-supervised Pretraining and targeted Fine-Tuning across compute, dataset size, and loss surfaces.
3. Task-specific adaptation architectures, classification heads, and sequence-to-sequence loss formulations.
4. Instruction tuning paradigms (FLAN, Alpaca, ShareGPT) that enable zero-shot task generalization.
5. Domain adaptation tradeoffs: continued pretraining on specialized corpora vs. targeted instruction fine-tuning.
6. Supervised Fine-Tuning (SFT) implementation mechanics, response-only cross-entropy loss masking, and gradient dynamics.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Post-Training Lifecycle and Adaptation Spectrum", divider="rainbow")

st.markdown("""
Modern Large Language Model development is strictly divided into two macroeconomic stages: 
**Pretraining** (unsupervised representation learning over web-scale tokens) and 
**Post-Training** (supervised adaptation, instruction alignment, and preference optimization).
""")

blueprint_dot = """
digraph SFTLifecycle {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_pretrain {
        label = "Stage 1: Pretraining (99% Compute)";
        color = "#334155";
        fontcolor = "#94A3B8";
        RawText [label="Trillion-Scale Web Tokens\\n(Common Crawl, Code, Books)", fillcolor="#0F172A"];
        BaseModel [label="Base Foundation LLM\\n(Autoregressive Predictor)", fillcolor="#0369A1"];
        RawText -> BaseModel [label="Self-Supervised Causal LM\\nLR: 3e-4, 15T Tokens"];
    }

    subgraph cluster_adaptation {
        label = "Stage 2: Post-Training Adaptation (1% Compute)";
        color = "#334155";
        fontcolor = "#94A3B8";
        DomainData [label="Domain Corpus\\n(Continued Pretraining)", fillcolor="#1E293B"];
        SFTData [label="Instruction QA Pairs\\n(Human and Synthetic)", fillcolor="#1E293B"];
        InstructModel [label="Instruction-Tuned LLM\\n(Supervised Fine-Tuned)", fillcolor="#047857"];
        AlignedModel [label="Production Assistant\\n(RLHF / DPO Aligned)", fillcolor="#B45309"];
        
        BaseModel -> DomainData [label="Specialization"];
        BaseModel -> SFTData [label="Task Tuning"];
        SFTData -> InstructModel [label="Response-Masked SFT\\nLR: 2e-5, 100k Pairs"];
        InstructModel -> AlignedModel [label="Preference Optimization\\n(DPO / PPO / KTO)"];
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### The Post-Training Hierarchy
1. **Raw Foundation LLM**: High world knowledge and reasoning potential, but zero instruction adherence or conversational etiquette.
2. **Domain-Adapted Foundation**: Specialized on proprietary token distributions (medical, legal, financial, source code) via continued pretraining.
3. **Supervised Fine-Tuned (SFT) Model**: Conditioned on structured turn-taking templates (User / Assistant) to follow instructions deterministically.
4. **Aligned Production Assistant**: Fine-tuned with Direct Preference Optimization (DPO) or Reinforcement Learning from Human Feedback (RLHF) to optimize helpfulness, harmlessness, and conciseness.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - WHAT IS FINE-TUNING?
# ==============================================================================
st.header("2. What is Fine-Tuning?", divider="rainbow")

st.markdown("""
Fine-tuning is the process of taking a neural network that has already learned general statistical representations 
on a massive corpus and updating its weights on a smaller, curated dataset tailored to a specific downstream task or behavior.

Mathematically, if pretraining finds parameter vector `theta_base` that maximizes likelihood over general corpus `D_general`:
```text
theta_base = argmax_theta Sum_{(x) in D_general} log P(x; theta)
```
Fine-tuning initializes at `theta_base` and finds `theta_adapted` that maximizes task-specific performance over `D_task`:
```text
theta_adapted = argmax_theta Sum_{(x, y) in D_task} log P(y | x; theta)
```
""")

col_ft1, col_ft2 = st.columns(2)
with col_ft1:
    st.markdown("#### Weight Dynamics and Loss Surfaces")
    st.markdown("""
    * **Loss Surface Initialization**: Pretraining traverses highly chaotic, non-convex loss valleys to locate flat, generalizing minima.
    * **Locality of Adaptation**: Fine-tuning operates inside or immediately adjacent to this flat basin. Weight delta `Delta_theta = theta_adapted - theta_base` has small Euclidean norm `||Delta_theta||`.
    * **Learning Rates**: Pretraining uses learning rates around `3e-4` to `1e-3`. Fine-tuning requires gentle learning rates (`1e-5` to `5e-5`) to avoid destroying pretrained feature extractors.
    """)

with col_ft2:
    st.markdown("#### Full Parameter vs. Parameter-Efficient Fine-Tuning")
    st.markdown("""
    * **Full Fine-Tuning (FFT)**: Every parameter `W` in the model is updated (`Delta_W in R^(d x k)`). Requires storing optimizer states (AdamW: 8 bytes per parameter) and gradients (2-4 bytes per parameter) for all billions of parameters.
    * **Parameter-Efficient Fine-Tuning (PEFT)**: Freezes `theta_base` entirely and injects small modular adapters (such as LoRA rank decomposition matrices `B x A`). Requires updating under 1% of parameters, cutting GPU memory by 60-80%.
    """)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
In frontier LLMs (Llama 3, Mistral, GPT-4), full fine-tuning of all weights is reserved for high-budget post-training teams with multi-node clusters. 
Enterprise teams and application engineers almost exclusively leverage **LoRA / QLoRA** or modular SFT over curated synthetic data rather than raw full-weight fine-tuning.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - PRETRAINING VS FINE-TUNING
# ==============================================================================
st.header("3. Pretraining vs. Fine-Tuning", divider="rainbow")

st.markdown("""
The macroeconomic and algorithmic gap between Pretraining and Fine-Tuning is the most defining characteristic of modern AI engineering.
""")

compare_data = {
    "Dimension": [
        "Primary Objective",
        "Data Scale",
        "Data Nature",
        "Compute Share",
        "Typical Hardware",
        "Optimization Steps",
        "Learning Rate",
        "Loss Masking",
        "Failure Mode"
    ],
    "Pretraining": [
        "Unsupervised Causal LM next-token prediction",
        "10 Trillion to 20 Trillion tokens",
        "Unfiltered web scrape, books, Wikipedia, code",
        "~98% to 99% of total lifecycle budget",
        "Thousands of H100 / B200 GPUs for months",
        "500,000 to 2,000,000 steps",
        "High (3e-4 to 1e-3) with cosine decay",
        "None (Every token is both input and target)",
        "Underfitting, training instability, loss spikes"
    ],
    "Fine-Tuning (SFT)": [
        "Instruction compliance, dialogue, format adherence",
        "10,000 to 1,000,000 curated QA pairs",
        "Structured instruction-response pairs, verified facts",
        "~1% to 2% of total lifecycle budget",
        "1 to 8 GPUs for hours or days",
        "1,000 to 10,000 steps (1 to 3 epochs)",
        "Low (1e-5 to 5e-5) with linear warmup and decay",
        "Strict (Prompt tokens masked out with ignore_index=-100)",
        "Catastrophic forgetting, overfitting, style collapse"
    ]
}
st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

st.markdown("""
### Memory Footprint Equation for Fine-Tuning
When fine-tuning a model with `N` billion parameters using 16-bit mixed precision (BF16) and 32-bit AdamW optimizer:
```text
Total_VRAM = Model_Weights (2 bytes * N) + Gradients (2 bytes * N) + AdamW_Moments (8 bytes * N) + Activation_Memory
```
For a 7 Billion parameter model (`N = 7`):
* Weights: `14 GB`
* Gradients: `14 GB`
* AdamW states (FP32 First and Second moments): `56 GB`
* Activations (batch size 4, context length 4096): `~20 to 40 GB`
* **Total required GPU memory for Full Fine-Tuning**: **~104 GB to 124 GB** (requires multiple A100/H100 80GB GPUs).
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - TASK-SPECIFIC ADAPTATION
# ==============================================================================
st.header("4. Task-Specific Adaptation", divider="rainbow")

st.markdown("""
Before generative conversational LLMs dominated, fine-tuning almost always meant attaching a **task-specific classification or regression head** 
onto the final hidden state of an encoder model (such as BERT or RoBERTa).
In modern generative architectures, task-specific adaptation has transitioned into two primary architectural paradigms:
""")

col_arch1, col_arch2 = st.columns(2)
with col_arch1:
    st.subheader("Paradigm A: Discriminative Head Adaptation")
    st.markdown("""
    * **Architecture**: The generative LM trunk is kept; the token generation unembedding matrix is replaced with a classification head `W_cls in R^(d_model x num_classes)`.
    * **Input Processing**: Hidden state `h_last` of the final token (or pooled sequence representation) is passed to `W_cls`.
    * **Output**: Softmax probability distribution over fixed discrete class labels.
    * **Loss**: Standard categorical cross-entropy over class logits.
    * **Pros & Cons**: Extremely fast inference and calibrated probabilities; cannot generate explanatory text or reasoning chains.
    """)

with col_arch2:
    st.subheader("Paradigm B: Autoregressive Generative Adaptation")
    st.markdown("""
    * **Architecture**: Retains the entire generative vocabulary head. Every task (classification, extraction, translation, coding) is framed as text-in, text-out.
    * **Input Processing**: Prompt templates format the task instructions and input data into a unified sequence.
    * **Output**: Autoregressive token sequence generated one token at a time.
    * **Loss**: Causal cross-entropy over response tokens.
    * **Pros & Cons**: Unified architecture handles arbitrary tasks and complex reasoning; higher latency and token verification overhead.
    """)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Modern foundation models (GPT-4o, Claude 3.5 Sonnet, Llama 3) have fully unified under **Generative Adaptation**. 
Even classification tasks are executed generatively by constraining output tokens using Structured Grammar Decoding (JSON Schema, Outlines, Instructor) 
rather than attaching physical linear heads to the trunk.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - INSTRUCTION TUNING
# ==============================================================================
st.header("5. Instruction Tuning", divider="rainbow")

st.markdown("""
**Instruction Tuning** is a breakthrough methodology introduced by papers such as FLAN (Google) and InstructGPT (OpenAI). 
Instead of tuning a model on a single narrow task (e.g. sentiment analysis), instruction tuning trains the model across 
**hundreds to thousands of diverse tasks**, each wrapped in human-like natural language instructions.
""")

inst_dot = """
digraph InstructionTuning {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_diverse_tasks {
        label = "Diverse Multi-Task Instruction Mixture";
        color = "#334155";
        fontcolor = "#94A3B8";
        T1 [label="Task 1: Code Generation\\n'Write a binary search algorithm in Rust.'", fillcolor="#0F172A"];
        T2 [label="Task 2: Summarization\\n'Condense this legal contract into 3 bullet points.'", fillcolor="#0F172A"];
        T3 [label="Task 3: Creative Writing\\n'Compose a sonnet about quantum computing.'", fillcolor="#0F172A"];
        T4 [label="Task 4: Logical Reasoning\\n'Solve this puzzle step by step.'", fillcolor="#0F172A"];
    }

    Base [label="Base Foundation Model", fillcolor="#0369A1"];
    Tuned [label="Instruction-Tuned Model\\n(Zero-Shot Generalized)", fillcolor="#047857"];
    Unseen [label="Unseen Downstream Task at Runtime\\n'Diagnose this network error message'", fillcolor="#7C3AED"];

    {T1 T2 T3 T4} -> Base [label="Tokenized with Chat Templates"];
    Base -> Tuned [label="Multi-Task SFT"];
    Tuned -> Unseen [label="Zero-Shot Generalization\\n(Executes unseen tasks accurately)"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(inst_dot, use_container_width=True)

st.markdown("""
### Why Multi-Task Instruction Tuning Enables Zero-Shot Generalization
When an LLM is trained on thousands of varied tasks framed as instructions:
1. It learns the **meta-concept of following directions** rather than memorizing surface-level pattern mapping.
2. It decouples task understanding from input phrasing: phrasing a request with *"Explain...", "Summarize...", or "Draft..."* routes to the same underlying capability.
3. It unlocks **Zero-Shot Transfer**: When given a brand new task at inference time that never existed in the training set, the model correctly interprets the English prompt and produces the desired output format.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - DOMAIN ADAPTATION
# ==============================================================================
st.header("6. Domain Adaptation", divider="rainbow")

st.markdown("""
When an organization requires an LLM for specialized domains (e.g. Oncology, Financial Hedging, Nuclear Engineering, Patent Law), 
off-the-shelf foundation models face two severe limitations:
1. **Knowledge Gaps**: Highly technical vocabulary, formulas, or internal proprietary documents were absent from public web scrapes.
2. **Distributional Shift**: The syntax, terminology, and reasoning styles of specialized domains differ radically from generic prose.
""")

st.markdown("""
| Dimension | Continued Pretraining (Domain-Adaptive PT) | Supervised Instruction Tuning (Domain SFT) |
| :--- | :--- | :--- |
| **Data Format** | Unstructured raw text (PDFs, papers, logs) | Curated (Instruction, Response) QA pairs |
| **Objective** | Unsupervised Causal LM on domain text | Autoregressive loss masked to response tokens |
| **Token Scale** | 1 Billion to 100 Billion tokens | 5,000 to 50,000 QA pairs |
| **Vocabulary** | May extend tokenizer with new domain subwords | Fixed tokenizer vocabulary |
| **Primary Gain** | Deep factual knowledge, jargon fluency | Task execution, persona, compliance, formatting |
| **Risk** | Catastrophic forgetting of general reasoning | Hallucination when underlying facts are missing |
""")

st.info("""
**Golden Rule of LLM Engineering**: 
* Use **Domain-Adaptive Pretraining** to teach the model **new knowledge and specialized vocabulary**.
* Use **Supervised Fine-Tuning** to teach the model **how to use that knowledge and format its answers**.
* Fine-tuning cannot reliably inject massive factual knowledge without causing hallucination—factual retrieval is best delegated to RAG or continued pretraining.
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - SUPERVISED FINE-TUNING (SFT)
# ==============================================================================
st.header("7. Supervised Fine-Tuning (SFT)", divider="rainbow")

st.markdown("""
**Supervised Fine-Tuning (SFT)** is the concrete training procedure that transforms a base model into an instruction follower. 
Understanding its exact loss formulation and token masking mechanics is critical for avoiding fatal training bugs.
""")

st.markdown("""
#### The Critical Engineering Principle: Response-Only Loss Masking
In an instruction-response pair:
```text
Sequence = [BOS] [User Tokens: "Write a haiku about rain."] [Assistant Tokens: "Soft grey drops descend..."] [EOS]
```
* **Flawed Approach (Full Sequence Loss)**: Computing cross-entropy loss over both User and Assistant tokens. The model wastes gradient capacity learning how to write user questions, causing degenerated prompt distributions!
* **Production Standard (Response Masking)**: Set target labels for all user prompt tokens to `-100` (PyTorch's `ignore_index`). Gradients are backpropagated **strictly through the assistant's response tokens**.
""")

st.code("""
# PyTorch CrossEntropyLoss with ignore_index=-100
# User prompt tokens receive label -100 so they contribute 0 to the loss gradient
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits.view(-1, vocab_size), target_labels.view(-1))
""", language="python")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Supervised Fine-Tuning Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained implementation demonstrates the complete mathematical mechanics of an SFT training step:
1. Token sequence packing with **Prompt Masking (`ignore_index = -100`)**.
2. Autoregressive teacher-forced forward pass.
3. Numerically stable cross-entropy loss computation with label masking.
4. Parameter update simulation tracking loss trajectory.
""")

st.code("""
import numpy as np
from typing import Tuple, Dict, Any

class ProductionSFTEngine:
    \"\"\"
    Production-grade educational implementation of Supervised Fine-Tuning (SFT)
    with exact response-only loss masking and parameter tracking.
    \"\"\"
    def __init__(self, vocab_size: int = 256, d_model: int = 32, ignore_index: int = -100, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ignore_index = ignore_index
        rng = np.random.RandomState(seed)
        self.W_embed = rng.randn(vocab_size, d_model).astype(np.float32) * 0.05
        self.W_head = rng.randn(d_model, vocab_size).astype(np.float32) * 0.05

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass: Token IDs -> Embeddings -> Linear Projection -> Logits.\"\"\"
        h = self.W_embed[input_ids]
        logits = np.dot(h, self.W_head)
        return logits

    def compute_masked_loss(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, int]:
        \"\"\"
        Causal Cross-Entropy with response-only masking.
        Aligns logits[t] to predict labels[t+1] using autoregressive shifting.
        \"\"\"
        shift_logits = logits[:-1, :]
        shift_labels = labels[1:]
        
        # Mask out prompt tokens where label == ignore_index
        mask = (shift_labels != self.ignore_index)
        active_tokens = int(np.sum(mask))
        if active_tokens == 0:
            return 0.0, 0
            
        valid_logits = shift_logits[mask]
        valid_labels = shift_labels[mask]
        
        # Numerically stable log-softmax: log(exp(x_i) / sum(exp(x)))
        max_logits = np.max(valid_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(valid_logits - max_logits)
        log_probs = (valid_logits - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
        
        # Select target log probabilities
        nll = -log_probs[np.arange(len(valid_labels)), valid_labels]
        return float(np.mean(nll)), active_tokens
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Adaptation Strategy Selector & Compute/Memory Sizer", divider="rainbow")

st.markdown("""
Experiment with different adaptation configurations. 
Calculate GPU VRAM requirements, training budgets, and test the real response-masked loss calculation engine in real time.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("#### Hardware & Architecture Configuration")
    model_param_b = st.selectbox(
        "Model Parameter Scale",
        [1.0, 3.0, 7.0, 13.0, 70.0],
        index=2,
        format_func=lambda x: f"{x} Billion Parameters (e.g. Llama-3-{int(x) if x.is_integer() else x}B)"
    )
    tuning_mode = st.radio(
        "Fine-Tuning Paradigm",
        ["Full Fine-Tuning (FFT)", "LoRA (r=16, alpha=32)", "QLoRA (4-bit Base + LoRA r=16)"]
    )
    batch_size = st.slider("Per-Device Batch Size", min_value=1, max_value=16, value=4)
    seq_len = st.select_slider("Sequence Length (Context Window)", options=[512, 1024, 2048, 4096, 8192], value=2048)

    # VRAM Sizing Calculation
    param_count = model_param_b * 1e9
    if tuning_mode == "Full Fine-Tuning (FFT)":
        weight_bytes = param_count * 2  # BF16
        grad_bytes = param_count * 2    # BF16
        optim_bytes = param_count * 8   # FP32 AdamW states
        trainable_ratio = 1.0
    elif "LoRA" in tuning_mode and "QLoRA" not in tuning_mode:
        weight_bytes = param_count * 2  # BF16 frozen
        trainable_params = param_count * 0.005  # ~0.5% adapter params
        grad_bytes = trainable_params * 2
        optim_bytes = trainable_params * 8
        trainable_ratio = 0.005
    else:  # QLoRA
        weight_bytes = param_count * 0.5  # 4-bit NF4
        trainable_params = param_count * 0.005
        grad_bytes = trainable_params * 2
        optim_bytes = trainable_params * 8
        trainable_ratio = 0.005

    # Activation memory estimation: proportional to batch_size * seq_len * layers
    activation_gb = (batch_size * seq_len * 32 * 4096 * 2) / (1024**3) * 0.3
    static_vram_gb = (weight_bytes + grad_bytes + optim_bytes) / (1024**3)
    total_vram_gb = static_vram_gb + activation_gb

with wb_col2:
    st.markdown("#### Estimated VRAM Breakdown & Hardware Allocation")
    
    st.metric(
        "Total Estimated GPU VRAM Required",
        f"{total_vram_gb:.2f} GB",
        f"Trainable Parameters: {trainable_ratio*100:.2f}%"
    )

    mem_df = pd.DataFrame({
        "Component": ["Model Weights", "Gradients", "Optimizer States (AdamW)", "Activation Memory"],
        "Memory (GB)": [
            weight_bytes / (1024**3),
            grad_bytes / (1024**3),
            optim_bytes / (1024**3),
            activation_gb
        ]
    })
    st.bar_chart(mem_df.set_index("Component"))

    if total_vram_gb <= 16:
        st.success("Target Hardware: Single Consumer GPU (RTX 4080 16GB / RTX 4090 24GB).")
    elif total_vram_gb <= 40:
        st.info("Target Hardware: Single Datacenter GPU (A100 40GB or 2x RTX 3090/4090 with FSDP).")
    elif total_vram_gb <= 80:
        st.warning("Target Hardware: Single Enterprise GPU (A100 80GB / H100 80GB).")
    else:
        st.error(f"Target Hardware: Multi-GPU Cluster Required ({int(np.ceil(total_vram_gb/80))}x H100 80GB GPUs with ZeRO-3 / FSDP).")

st.markdown("---")
st.markdown("#### Real-Time Response-Masking Loss Simulator")

engine = ProductionSFTEngine(vocab_size=64, d_model=16, seed=42)

prompt_len = st.slider("Prompt Token Count (Masked with -100)", min_value=3, max_value=12, value=5)
response_len = st.slider("Response Token Count (Active Gradient Target)", min_value=3, max_value=12, value=6)

# Generate synthetic token IDs
total_len = prompt_len + response_len
sample_tokens = np.random.randint(1, 60, size=total_len)
# Build labels: prompt tokens get -100, response tokens get real token IDs
sample_labels = sample_tokens.copy()
sample_labels[:prompt_len] = -100

sim_result = engine.train_step(sample_tokens, sample_labels, lr=0.02)

res_col1, res_col2, res_col3 = st.columns(3)
with res_col1:
    st.metric("Total Sequence Length", f"{sim_result['total_tokens']} Tokens")
with res_col2:
    st.metric("Gradients Active On", f"{sim_result['active_tokens']} Tokens", "Masked: -100 ignored")
with res_col3:
    st.metric("Cross-Entropy Loss", f"{sim_result['loss']:.4f}")

token_display = []
for idx, (t, lab) in enumerate(zip(sample_tokens, sample_labels)):
    status = "🔴 Prompt (Masked: -100)" if lab == -100 else "🟢 Response (Active Loss)"
    token_display.append({"Position": idx, "Token ID": t, "Label Target": lab, "Gradient Status": status})

st.dataframe(pd.DataFrame(token_display), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why must we mask user prompt tokens during SFT?"):
        st.write("""
        If prompt tokens are not masked with `ignore_index = -100`, the model computes cross-entropy loss over the user's instructions.
        This forces the model's parameters to optimize for predicting the user's question grammar rather than learning high-quality answers.
        In production, full-sequence loss causes degradation in instruction adherence and encourages the model to generate prompt-like syntax.
        """)
    with st.expander("Q2: Why does full fine-tuning of a 7B model require ~100GB VRAM instead of 14GB?"):
        st.write("""
        While 7B model weights in BF16 require only 14GB (`7B * 2 bytes`), full training requires:
        1. Gradients in BF16: `14GB`.
        2. AdamW optimizer states in FP32 (first moment `m_t` and second moment `v_t`): `7B * 4 bytes * 2 = 56GB`.
        3. Intermediate activation tensors cached for backpropagation: `20 to 40GB`.
        Total VRAM reaches 104GB - 124GB, necessitating distributed training (FSDP / ZeRO-3) or PEFT methods like LoRA.
        """)
    with st.expander("Q3: When should you use Continued Pretraining instead of SFT?"):
        st.write("""
        Continued Pretraining is required when the base model lacks foundational domain knowledge or specialized jargon (e.g. proprietary clinical trials, rare foreign languages, company codebase).
        SFT cannot teach large volumes of new facts reliably—it primarily aligns style, output formatting, and dialogue behavior. Attempting to inject new knowledge solely through SFT leads to hallucinations.
        """)

with q2:
    with st.expander("Q4: What is the LIMA hypothesis and what does it prove about SFT?"):
        st.write("""
        The **LIMA (Less Is More for Alignment)** paper by Zhou et al. (2023) demonstrated that fine-tuning LLaMA on just **1,000 carefully curated, high-quality instruction-response pairs** achieved competitive performance with models trained on 50,000+ noisy pairs.
        This proved that almost all knowledge is acquired during pretraining; SFT acts merely as a thin superficial formatting layer to teach the model which style and tone to adopt.
        """)
    with st.expander("Q5: How does instruction tuning differ from task-specific fine-tuning?"):
        st.write("""
        Task-specific fine-tuning trains on a single task (e.g. sentiment classification), optimizing the model only for that objective.
        Instruction tuning trains across a diverse mixture of hundreds of heterogeneous tasks formatted as natural language instructions. 
        This imparts **meta-instruction-following ability**, unlocking zero-shot generalization on novel tasks never encountered during training.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Prompt Loss Leakage**
    * **Failure**: Passing entire formatted dialogues into standard causal LM trainers without masking prompts.
    * **Impact**: Training loss drops rapidly, but model becomes an echo-chamber or repeats prompts.
    * **Remedy**: Always verify that token labels prior to the assistant response header are explicitly set to `-100`.
    """)
    st.error("""
    **🚨 Production Trap 2: SFT Learning Rate Too High**
    * **Failure**: Using pretraining learning rates (`1e-4` to `5e-4`) during fine-tuning.
    * **Impact**: Catastrophic representation collapse within 50 steps; loss blows up or outputs become repetitive gibberish.
    * **Remedy**: Keep SFT learning rates strictly between `5e-6` and `3e-5`, combined with cosine decay and 5% warmup steps.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Catastrophic Forgetting of Reasoning**
    * **Failure**: Fine-tuning exclusively on narrow domain pairs (e.g. only SQL queries) for 5+ epochs.
    * **Impact**: Model achieves 99% accuracy on SQL, but loses general coding and logical reasoning abilities.
    * **Remedy**: Mix in a 10-20% replay buffer of general conversational and reasoning data (e.g. UltraChat, ShareGPT) into the training mix.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Inconsistent Chat Template Special Tokens**
    * **Failure**: Training with ChatML special tokens (`<|im_start|>user`) but serving inference with Llama-3 headers (`<|start_header_id|>user<|end_header_id|>`).
    * **Impact**: The model fails to recognize user turns, hallucinates endless dialogues, or refuses to stop generation.
    * **Remedy**: Enforce tokenizer-native Jinja2 templates via `tokenizer.apply_chat_template()` for both training and inference.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: SFT Memory Formula"):
        st.write("How many bytes per parameter are required for Full Fine-Tuning with 32-bit AdamW?")
        st.info("16 bytes per parameter (2B weights + 2B grads + 8B AdamW moments + 4B FP32 master weights).")
    with st.expander("⚡ Flashcard 2: Prompt Loss Mask Value"):
        st.write("What integer constant is assigned to user prompt tokens in PyTorch CrossEntropyLoss?")
        st.info("-100 (PyTorch's default ignore_index).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: The LIMA Hypothesis"):
        st.write("What did the LIMA paper prove regarding SFT dataset volume?")
        st.info("Less Is More for Alignment: 1,000 curated high-quality pairs beats 50,000 noisy pairs.")
    with fc_col2:
        with st.expander("⚡ Flashcard 4: Domain Adaptation Recipe"):
            st.write("How do you teach a model new specialized knowledge vs new task formatting?")
            st.info("Continued Pretraining teaches new knowledge; SFT teaches task formatting and response behavior.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of LLM Fine-Tuning Fundamentals before advancing to Dataset Preparation:
""")

chk1 = st.checkbox("I understand why raw base models autocomplete rather than follow instructions.", value=True)
chk2 = st.checkbox("I can calculate the 16 bytes/parameter memory footprint of full AdamW fine-tuning.", value=True)
chk3 = st.checkbox("I know how to implement response-only loss masking using ignore_index=-100.", value=True)
chk4 = st.checkbox("I understand the tradeoff between Narrow Task Specialization and Multi-Task Data Replay.", value=True)
chk5 = st.checkbox("I understand when Continued Pretraining is required versus standard Instruction Tuning.", value=True)
chk6 = st.checkbox("I know how Jinja2 chat templates enforce standard role boundaries.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 13.2: Dataset Preparation.")
