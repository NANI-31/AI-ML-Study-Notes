"""
8.4 Evaluation
==============
Interactive Master Lesson exploring Perplexity (PPL), Validation Loss, Training Loss Dynamics,
Overfitting, Underfitting, and Generalization frontiers in Modern Large Language Models.
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
    page_title="8.4 Evaluation - LLM Architecture",
    page_icon="📊",
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
st.title("📊 8.4 Evaluation")
st.caption(
    "Chapter 8: Language Modeling | Module 4 of 4"
)

st.markdown(
    """
Evaluating Large Language Models requires measuring both continuous intrinsic statistical metrics
(**Cross-Entropy Loss**, **Perplexity**) and discrete downstream reasoning capabilities.
Understanding how training loss curves interact with validation loss, learning rate schedules,
and compute budgets governs whether a foundation model underfits its data budget, memorizes its corpus,
or achieves powerful zero-shot and few-shot **generalization**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & MODERN EVALUATION PARADIGM
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Modern Evaluation Paradigm", divider="rainbow")

st.markdown(
    """
### The Dual-Track LLM Evaluation Framework
Modern LLM evaluation operates across two synchronized methodologies:
1. **Intrinsic Perplexity & Loss**: Continuously tracked every 100 training steps across held-out corpora (WikiText, C4, RedPajama validation splits).
2. **Downstream Task Benchmarks**: Evaluated checkpoint-by-checkpoint across standard academic reasoning suites (MMLU, GSM8K, HumanEval, ARC, HellaSwag).
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "evaluation_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("checkpoint", "Model Checkpoint\\n(Step K)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("loss_eval", "Intrinsic Evaluation\\n- Valid Loss (CE)\\n- Perplexity (exp(Loss))\\n- Bits Per Character (BPC)", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("bench_eval", "Downstream Benchmarks\\n- MMLU (General Knowledge)\\n- GSM8K (Math Reasoning)\\n- HumanEval (Code Synthesis)", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("scaling", "Chinchilla Scaling Fit\\nPower-Law Verification\\nL(N, D) = A/N^alpha + B/D^beta + E", fillcolor="#7C3AED", fontcolor="#F8FAFC")

arch_dot.edge("checkpoint", "loss_eval")
arch_dot.edge("checkpoint", "bench_eval")
arch_dot.edge("loss_eval", "scaling", label="Continuous Signal")
arch_dot.edge("bench_eval", "scaling", label="Discrete Emergence")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Perplexity as the North Star**: Continuous validation loss correlates monotonically with downstream capability. A 0.1 reduction in cross-entropy loss consistently unlocks noticeable jumps in 5-shot MMLU and GSM8K scores.
* **The Emergence Illusion**: Early research argued that complex abilities (like multi-step math) appear discontinuously as "emergent abilities." Recent analysis demonstrates that the underlying token probabilities improve smoothly; it is only the non-linear thresholding of 0/1 accuracy metrics that creates the illusion of sudden jumps.
* **Data Contamination Audits**: With web-scraped test sets, models risk memorizing benchmark questions. Frontier training runs employ strict 13-gram decontamination filters to purge evaluation splits from training datasets.
"""
)

# Evaluation Metric Matrix Table
eval_matrix_df = pd.DataFrame(
    {
        "Evaluation Metric": ["Cross-Entropy Loss (nats)", "Perplexity (PPL)", "Bits Per Byte (BPB)", "Few-Shot Accuracy (%)"],
        "Formula": ["- 1/N sum log P(y_i)", "exp(Loss)", "Loss / (log(2) * Bytes_Per_Token)", "Correct / Total Questions"],
        "Tokenizer Independent?": ["No (Vocab Dependent)", "No (Vocab Dependent)", "Yes (Normalized by Byte Length)", "Yes (Direct Accuracy)"],
        "Evaluation Frequency": ["Every step / Continuous", "Every 500-1000 steps", "Across heterogeneous models", "At major checkpoints / Evals"],
        "Benchmark Examples": ["Held-out validation sets", "WikiText-103, C4 validation", "LLaMA vs Mistral byte comp", "MMLU, HumanEval, GSM8K"],
    }
)
st.dataframe(eval_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: PERPLEXITY (PPL) MATHEMATICAL FOUNDATIONS
# -----------------------------------------------------------------------------
st.header("2. Perplexity (PPL) Mathematical Foundations & Branching Factor", divider="rainbow")

st.markdown(
    """
Perplexity is the standard intrinsic metric for evaluating language models.
Mathematically, it is the exponentiated cross-entropy loss over a sequence `W = (w_1, w_2, ..., w_N)`:

```text
PPL(W) = exp( - 1/N * SUM_{i=1}^{N} log P(w_i | w_{<i}) ) = exp( Cross_Entropy_Loss )
```

### The Intuitive Meaning: Weighted Branching Factor
Perplexity represents the **effective number of equally likely tokens** among which the model is uncertain at each step:
* A perplexity of `1.0` means the model is 100% confident on every token and makes zero errors.
* A perplexity of `10.0` means the model is, on average, as confused as if it were choosing uniformly from 10 plausible words.
* A perplexity equal to the vocabulary size `V` (e.g. 128,256) means the model is completely untrained (pure uniform random guessing).

### The Tokenizer Sensitivity Trap
**Perplexity is NOT directly comparable between models with different tokenizers!**
If Model A has a vocabulary of 32,000 tokens (longer subword chunks), and Model B has a vocabulary of 128,000 tokens (shorter subword pieces):
* Model B assigns probability over fewer, shorter tokens per word, altering the token-level cross-entropy sum.
* To compare models with different tokenizers, always normalize by text byte length using **Bits Per Byte (BPB)**:
  ```text
  BPB = (Total_Loss_in_bits) / (Total_UTF8_Bytes)
  ```
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: VALIDATION LOSS TRACKING & CONVERGENCE INDICATORS
# -----------------------------------------------------------------------------
st.header("3. Validation Loss Tracking & Convergence Indicators", divider="rainbow")

st.markdown(
    """
Validation loss is computed on a static, held-out slice of data that the optimizer never sees during backpropagation.

### Critical Convergence Behaviors
1. **Healthy Learning**: Both training and validation loss decline steadily in parallel. The validation loss tracks slightly higher than training loss.
2. **Learning Rate Decay Impact**: Under cosine learning rate decay, validation loss drops sharply during the learning rate warmup, plateaus during the steady state, and experiences a final rapid plunge during the final 10-20% learning rate cooldown.
3. **Loss Spikes & Recovery**: In multi-billion parameter runs, models occasionally experience transient loss spikes (e.g. from 1.8 jumping to 4.5). If gradient clipping is healthy and data sharding is stable, the optimizer typically recovers within 50-100 steps.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: TRAINING LOSS DYNAMICS & SCALING LAW TRAJECTORIES
# -----------------------------------------------------------------------------
st.header("4. Training Loss Dynamics & Scaling Law Trajectories (Chinchilla)", divider="rainbow")

st.markdown(
    """
Kaplan et al. (2020) and Hoffmann et al. (Chinchilla, 2022) proved that LLM training loss follows predictable **power-law scaling laws**:

```text
Loss(N, D) = (A / N^alpha) + (B / D^beta) + E
```
Where:
* `N` is the number of non-embedding model parameters.
* `D` is the number of training tokens.
* `E` is the irreducible entropy of natural language (approximately 1.69 nats for English).
* `alpha approx 0.34`, `beta approx 0.28`.

### The Chinchilla Optimal Compute Rule
For compute-optimal pretraining, **model parameters and training tokens should scale in equal proportion**:
`D approx 20 * N`.
* A 7B model requires `~140 billion tokens` to be compute-optimal.
* However, modern open-weight models (Llama 3 8B trained on 15 trillion tokens) intentionally **overtrain** by 100x beyond Chinchilla optimality to produce radically smaller, faster models for production inference!
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: OVERFITTING IN LLMS
# -----------------------------------------------------------------------------
st.header("5. Overfitting in LLMs (Memorization vs. Reasoning)", divider="rainbow")

st.markdown(
    """
In classical machine learning, overfitting occurs when a model memorizes training noise and fails on unseen test sets.
In Large Language Models, overfitting exhibits distinct characteristics:

### The Single-Epoch Pretraining Standard
In frontier pretraining (GPT-4, Llama 3), models are trained for **exactly one single epoch** over 10 to 15 trillion tokens:
* Every token seen by the model is novel; the model almost never observes the exact same document twice.
* Therefore, **classical pretraining overfitting is virtually non-existent** in large runs.

### Overfitting in Fine-Tuning (SFT & Alignment)
Where overfitting DOES severely threaten LLMs is during Supervised Fine-Tuning (SFT):
* SFT datasets typically consist of only 10,000 to 500,000 instruction pairs.
* Training for 3-5 epochs causes the model to memorize exact prompt phrasings, degrading its general reasoning, style flexibility, and benchmark scores (the *alignment tax*).
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: UNDERFITTING & MODEL CAPACITY BOTTLENECKS
# -----------------------------------------------------------------------------
st.header("6. Underfitting & Model Capacity Bottlenecks", divider="rainbow")

st.markdown(
    """
Underfitting occurs when the model architecture lacks the parameter capacity, depth, or compute budget
to capture the underlying distribution of the training data.

### Hallmarks of Underfitting
* The training loss curve flattens prematurely at a high value (> 2.5 nats).
* The model produces syntactically fluent sentences but hallucinates facts and fails simple logical reasoning.
* Increasing the training token count further yields diminishing returns without expanding model width (`d_model`) or layer depth (`L`).
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: GENERALIZATION FRONTIERS
# -----------------------------------------------------------------------------
st.header("7. Generalization Frontiers (In-Domain vs. Out-of-Distribution Transfer)", divider="rainbow")

st.markdown(
    """
Generalization in LLMs is evaluated across two primary axes:
1. **In-Domain Generalization**: Predicting unseen tokens drawn from the same distribution as the training data (e.g. held-out web text from Common Crawl).
2. **Out-of-Distribution (OOD) Transfer**: Executing novel reasoning tasks, unseen programming languages, or zero-shot translation that was never explicitly labeled in the corpus.

### In-Context Learning (ICL) as Generalization
The hallmark of foundation LLMs is that few-shot demonstrations provided in the prompt act as algorithmic task specifications, allowing the model to generalize to new tasks at inference time without any parameter weight updates.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE EVALUATION PIPELINE IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("8. Production-Grade Perplexity & Downstream Benchmark Evaluation Pipeline", divider="rainbow")

st.markdown(
    """
A complete PyTorch implementation for computing exact, sliding-window perplexity on long documents:
"""
)

code_snippet = """import torch
import torch.nn as nn
import math

def compute_sliding_window_perplexity(
    model: nn.Module,
    encoded_tokens: torch.Tensor,
    max_context_length: int = 4096,
    stride: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    # Computes perplexity over long documents using sliding-window stride to avoid boundary artifacts
    model.eval()
    seq_len = encoded_tokens.size(1)
    
    nlls = []
    total_tokens = 0
    
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_context_length, seq_len)
            trg_len = end_loc - max(begin_loc, end_loc - stride)
            
            input_ids = encoded_tokens[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            
            # Mask out context tokens so loss is only computed over newly introduced stride tokens
            target_ids[:, :-trg_len] = -100
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift for causal evaluation
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            
            nlls.append(nll.item())
            total_tokens += trg_len
            
            if end_loc == seq_len:
                break
                
    total_nll = sum(nlls)
    avg_loss = total_nll / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "cross_entropy_loss": avg_loss,
        "perplexity": perplexity,
        "total_evaluated_tokens": total_tokens,
    }
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE EVALUATION & PERPLEXITY WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Perplexity & Generalization Dynamics Workbench", divider="rainbow")

st.markdown(
    """
Model the relationship between **Cross-Entropy Validation Loss**, **Perplexity**, and predicted
**Downstream MMLU Benchmark Accuracy**.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    sim_loss = st.slider("Validation Cross-Entropy Loss (nats)", min_value=1.2, max_value=4.0, value=2.1, step=0.05)
    model_param_b = st.select_slider("Model Parameter Scale", options=[1, 3, 7, 13, 34, 70, 405], value=7)
with col_ctrl2:
    eval_domain = st.selectbox("Validation Corpus Domain", options=["Wikipedia / STEM", "Code Repositories", "Multi-Turn Dialogue", "Raw Common Crawl"], index=0)

# Compute metrics
sim_ppl = np.exp(sim_loss)
bits_per_token = sim_loss / np.log(2.0)

# Empirical heuristic correlation with MMLU accuracy
# Typically, loss 3.0 -> ~25% (random guessing), loss 1.5 -> ~80%
mmlu_est = np.clip(100.0 * (1.0 - (sim_loss - 1.2) / (3.2 - 1.2)) ** 1.3, 25.0, 88.0)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Effective Perplexity (PPL)", f"{sim_ppl:.2f}")
with col_m2:
    st.metric("Bits per Token", f"{bits_per_token:.2f} bits")
with col_m3:
    st.metric("Estimated MMLU 5-Shot", f"{mmlu_est:.1f}%")

# Generate simulated training curve comparison
steps = np.linspace(0, 100000, 100)
# Cosine schedule learning curve simulation
train_curve = 4.0 - 2.2 * (1.0 - np.exp(-steps / 15000.0)) - 0.2 * (steps / 100000.0)
val_curve = train_curve + 0.12 + 0.05 * (steps / 100000.0) ** 2

loss_plot_df = pd.DataFrame(
    {
        "Step": steps,
        "Train Loss": np.round(train_curve, 3),
        "Validation Loss": np.round(val_curve, 3),
    }
)

st.subheader("Simulated Foundation Pretraining Convergence Trajectory")
st.line_chart(loss_plot_df.set_index("Step"))

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why cannot Perplexity be used to compare two models with different tokenizers?"):
    st.markdown(
        """
        **Answer:**
        Perplexity is computed per token: `exp(Loss / N_tokens)`.
        A tokenizer with a large vocabulary (e.g. 128k) merges longer character chunks into single tokens, resulting in fewer total tokens per sentence than a 32k tokenizer. Because the loss denominator `N_tokens` changes, comparing raw perplexities is invalid. To compare across tokenizers, normalize by text byte length using Bits Per Byte (BPB).
        """
    )

with st.expander("Q2: What did the Chinchilla scaling law paper reveal about compute-optimal training?"):
    st.markdown(
        """
        **Answer:**
        Kaplan et al. initially suggested prioritizing model parameter size over dataset size.
        Hoffmann et al. (Chinchilla, 2022) proved that models like GPT-3 (175B params trained on 300B tokens) were severely undertrained. For compute-optimal training, model size and token count should scale equally (`D approx 20 * N`). Chinchilla 70B outperformed Gopher 280B by training on 1.4T tokens using equal compute.
        """
    )

with st.expander("Q3: What causes loss spikes during large-scale pretraining, and how are they handled?"):
    st.markdown(
        """
        **Answer:**
        Loss spikes are typically triggered by:
        1. Encounters with malformed or extreme web data batches that generate outlier activations.
        2. Unstable attention FP16 underflows/overflows.
        3. Optimizer state corruption in AdamW second-moment estimates (`v_t`).
        **Mitigations**: Gradient clipping, reverting to an earlier checkpoint, skipping the corrupted data shard, and switching to BF16 or FP32 Softmax normalization.
        """
    )

with st.expander("Q4: Why does validation loss continue to decrease even after downstream task accuracy saturates?"):
    st.markdown(
        """
        **Answer:**
        Downstream benchmarks use 0/1 accuracy (either the correct option was chosen or not).
        Even after the model correctly ranks the true answer as #1, continuing pretraining increases the confidence probability `P(target)` from 60% to 95%. This continuous probability improvement drives validation cross-entropy loss lower and enhances out-of-distribution robustness, even if top-1 benchmark accuracy appears flat.
        """
    )

with st.expander("Q5: What is benchmark contamination and how is it detected in modern LLMs?"):
    st.markdown(
        """
        **Answer:**
        Benchmark contamination occurs when evaluation test questions inadvertently leak into the massive multi-terabyte web training corpus.
        **Detection**: N-gram overlap filtering (e.g., matching 13 consecutive tokens between training data and evaluation benchmarks). Contaminated documents are either purged prior to pretraining or reported with decontamination asterisks.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Evaluating Perplexity with Stride Equal to Context Length**
        *Root Cause*: Chunking a document into disjoint 4096-token blocks causes early tokens in each chunk to have zero conditioning context, artificially inflating evaluated perplexity.
        *Production Fix*: Use a sliding-window stride (e.g., stride = 512) so tokens are always evaluated conditioned on a full historical context buffer.
        """
    )
    st.warning(
        """
        **Trap 2: Comparing Validation Loss Across Different Sequence Lengths**
        *Root Cause*: Shorter sequences have higher proportions of initial, hard-to-predict tokens, producing naturally higher average loss than 8192-token packed sequences.
        *Production Fix*: Always evaluate validation loss using consistent packing configurations and identical context window limits.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: SFT Catastrophic Forgetting via Excessive Epochs**
        *Root Cause*: Training on a domain-specific SFT dataset for 5-10 epochs causes loss to drop near zero, but the model loses general instruction following.
        *Production Fix*: Train SFT for only 1 to 2 epochs, monitoring validation loss on a general conversation benchmark.
        """
    )
    st.warning(
        """
        **Trap 4: Averaging Perplexities Across Documents Instead of Losses**
        *Root Cause*: Computing `mean(PPL_doc1, PPL_doc2)` instead of `exp(mean(Loss_doc1, Loss_doc2))`.
        *Production Fix*: Exponentiation is non-linear; averaging perplexities heavily biases metrics toward the single worst outlier document. Always average cross-entropy loss first, then exponentiate.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Perplexity Lower Bound"):
        st.write("What is the theoretical minimum possible perplexity for a language model?")
        st.info("PPL = 1.0 (corresponding to 0.0 nats cross-entropy loss).")

    with st.expander("⚡ Flashcard 2: Chinchilla Optimal Ratio"):
        st.write("What is the Chinchilla compute-optimal ratio of training tokens to model parameters?")
        st.info("Approximately 20 tokens per parameter (D = 20 * N).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Bits Per Byte (BPB)"):
        st.write("Why is Bits Per Byte superior to Perplexity for comparing different model families?")
        st.info("BPB normalizes by physical character/byte length, making it tokenizer-independent.")

    with st.expander("⚡ Flashcard 4: Pretraining Epoch Count"):
        st.write("How many epochs are frontier models typically trained for during pretraining?")
        st.info("Exactly 1 epoch (every token seen once) to avoid memorization and maximize diverse coverage.")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the mathematical derivation of Perplexity as exponentiated cross-entropy loss.",
    "I can explain why Perplexity cannot be compared directly between models with different tokenizers.",
    "I understand the Chinchilla compute-optimal scaling rule and why modern models overtrain.",
    "I know how sliding-window evaluation prevents boundary artifacts when measuring document perplexity.",
    "I understand the difference between pretraining underfitting and fine-tuning overfitting.",
    "I can explain how 13-gram decontamination filters prevent test set leakage in web pretraining.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Language Model Evaluation, Perplexity, and Scaling Dynamics."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
