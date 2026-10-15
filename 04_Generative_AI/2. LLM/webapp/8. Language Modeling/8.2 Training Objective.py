"""
8.2 Training Objective
======================
Interactive Master Lesson exploring Cross-Entropy Loss, Negative Log Likelihood,
Log Probabilities, Sequence Loss Masking, and Teacher Forcing in Large Language Models.
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
    page_title="8.2 Training Objective - LLM Architecture",
    page_icon="🎯",
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
st.title("🎯 8.2 Training Objective")
st.caption(
    "Chapter 8: Language Modeling | Module 2 of 4"
)

st.markdown(
    """
The foundational training objective of all modern autoregressive Large Language Models
is **Maximum Likelihood Estimation (MLE)** implemented via **Cross-Entropy Loss** and
**Negative Log Likelihood (NLL)**. By training models under **Teacher Forcing**, the entire
sequence is optimized in parallel, pushing the network to allocate maximum probability density
to the actual next tokens appearing in massive multi-trillion token corpora.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE OPTIMIZATION OBJECTIVE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Optimization Objective", divider="rainbow")

st.markdown(
    """
### The End-to-End Optimization Feedback Loop
During pretraining, batch sequences are processed through the model to produce raw logits.
The training objective measures the distance between predicted probability distributions and the true one-hot token targets.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "training_objective_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("input", "Input Sequence\\nx = [x_1, ..., x_{T-1}]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("target", "Target Sequence\\ny = [x_2, ..., x_T]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("model", "Causal Transformer\\nForward Pass", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("logits", "Logits Matrix\\nz in R^[B, T-1, V]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("loss", "Cross-Entropy / NLL\\n-log P(y_t | x_{<=t})", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("grad", "Backpropagation\\ndL / dW", fillcolor="#047857", fontcolor="#F8FAFC")

arch_dot.edge("input", "model")
arch_dot.edge("model", "logits")
arch_dot.edge("logits", "loss", label="Predictions")
arch_dot.edge("target", "loss", label="Ground Truth")
arch_dot.edge("loss", "grad", label="Scalar Loss")
arch_dot.edge("grad", "model", label="Optimizer Step (AdamW)")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
While the core formula `L = -log P(y)` remains unchanged from early neural networks:
* **Auxiliary Loss Terms**: Modern pretraining pipelines add **Z-Loss** (`c * log(sum(exp(z)))^2`) to penalize large logit magnitudes and prevent FP16 numerical overflow during 100,000-step runs (PaLM, Gemma).
* **Target Masking & Packed Sequences**: Modern training does not pad sequences with zeros. Document sequences are packed into continuous 8,192-token buffers with attention boundary masks and ignore index (`-100`) flags to ensure compute is never wasted on padding.
* **Loss Spikes & Gradient Clipping**: Gradient norm clipping (`max_norm = 1.0`) protects against optimizer divergence when models encounter malformed web data or anomalous tokens.
"""
)

# Objective Comparison Table
obj_df = pd.DataFrame(
    {
        "Objective Component": ["Negative Log Likelihood (NLL)", "Cross-Entropy Loss", "Label Smoothing", "Z-Loss Regularization"],
        "Mathematical Form": ["-log P(y | x)", "-sum y_i log p_i", "(1-eps) CE + eps Uniform", "c * (log sum exp(z))^2"],
        "Role in LLM Training": ["Core probabilistic objective", "Standard multi-class loss", "Prevents overconfident logits", "Stabilizes FP16/BF16 logits"],
        "Default Configuration": ["Standard for Next-Token", "Used in 100% of LLMs", "Rare in Pretraining, common in RL", "Used in PaLM, Gemma, Llama 3"],
    }
)
st.dataframe(obj_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: CROSS-ENTROPY LOSS & INFORMATION THEORY FOUNDATIONS
# -----------------------------------------------------------------------------
st.header("2. Cross-Entropy Loss & Information Theory Foundations", divider="rainbow")

st.markdown(
    """
Cross-entropy measures the average number of bits required to encode data from a true distribution `p`
using an estimated model distribution `q`:

```text
H(p, q) = - SUM_{v=1}^{V} p(v) * log q(v)
```

In standard language modeling, the ground truth `p` is a **one-hot distribution** centered entirely on the true target token `y`:
* `p(v) = 1` if `v = y`
* `p(v) = 0` if `v != y`

Substituting this into the cross-entropy formula eliminates all terms except `v = y`:

```text
H(p, q) = - 1.0 * log q(y) = -log q(y)
```
Thus, for one-hot targets, **Cross-Entropy is mathematically identical to Negative Log Likelihood**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: NEGATIVE LOG LIKELIHOOD (NLL) & MLE
# -----------------------------------------------------------------------------
st.header("3. Negative Log Likelihood (NLL) & Maximum Likelihood Estimation", divider="rainbow")

st.markdown(
    """
### Derivation from Maximum Likelihood Estimation (MLE)
We model our language dataset `D = {x^(1), x^(2), ..., x^(N)}` as samples drawn from the true underlying language distribution.
The goal of Maximum Likelihood Estimation is to find parameter weights `theta` that maximize the likelihood of the training corpus:

```text
theta_MLE = argmax_{theta} PRODUCT_{i=1}^{N} P(x^(i); theta)
```

Taking the natural logarithm converts the product into a sum:

```text
theta_MLE = argmax_{theta} SUM_{i=1}^{N} log P(x^(i); theta)
```

Because gradient optimization libraries minimize rather than maximize functions, we multiply by `-1`, converting the objective into **Negative Log Likelihood (NLL)** minimization:

```text
Loss(theta) = - 1/N * SUM_{i=1}^{N} log P(x^(i); theta)
```
Minimizing NLL directly minimizes the Kullback-Leibler (KL) divergence between the true data distribution and the model distribution:
`D_KL(P_data || P_model) = H(P_data, P_model) - H(P_data)`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: LOG PROBABILITIES & NUMERICAL PRECISION DYNAMICS
# -----------------------------------------------------------------------------
st.header("4. Log Probabilities & Numerical Precision Dynamics", divider="rainbow")

st.markdown(
    """
### The Danger of Naive Softmax Followed by Log
A naive implementation computes:
```python
probs = torch.softmax(logits, dim=-1)
loss = -torch.log(probs[target])
```
This is **numerically catastrophic** in deep learning:
1. If a logit for an incorrect token is large, `probs[target]` can round down to `0.0` in FP16 / BF16.
2. Evaluating `torch.log(0.0)` produces `-inf`, causing the loss to become `+inf` and corrupting all model gradients with `NaN`.

### The Log-Sum-Exp Trick
PyTorch's `nn.CrossEntropyLoss` combines `log_softmax` and `nll_loss` internally using the **Log-Sum-Exp (LSE)** trick:

```text
log P(y) = log( exp(z_y) / SUM_j exp(z_j) )
         = z_y - log( SUM_j exp(z_j) )
         = z_y - ( max(z) + log( SUM_j exp(z_j - max(z)) ) )
```
By subtracting `max(z)` before exponentiation, the largest exponent evaluated is `exp(0) = 1.0`, completely eliminating floating-point overflow and underflow.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: PER-TOKEN LOSS & HARD-TOKEN GRADIENT DISTRIBUTION
# -----------------------------------------------------------------------------
st.header("5. Per-Token Loss & Hard-Token Gradient Distribution", divider="rainbow")

st.markdown(
    """
Not all tokens contribute equally to the loss or gradient magnitude.
At each position `t`:

```text
Loss_t = -log P(x_{t+1} | x_{<=t})
```

### Gradient Magnitude Analysis
Taking the gradient of Cross-Entropy with respect to logit `z_i`:

```text
dL / dz_i = P(x = i) - 1   (if i is the true target token)
dL / dz_i = P(x = i)       (if i is an incorrect token)
```
* **Easy Tokens (P(target) = 0.99)**: The gradient `dL / dz_target = 0.99 - 1 = -0.01`. The model generates almost zero parameter updates.
* **Hard Tokens (P(target) = 0.01)**: The gradient `dL / dz_target = 0.01 - 1 = -0.99`. Generates large, aggressive weight updates.

### Token Types Across Corpora
* **Low Loss Tokens**: Predictable function words, boilerplate syntax (`"def"`, `"import"`, `"the"`, `"{"`). Loss is typically `< 0.2`.
* **High Loss Tokens**: Rare proper nouns, specific numerical values, entities introduced for the first time. Loss can exceed `8.0`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: SEQUENCE LOSS AGGREGATION & MASKED NORMALIZATION
# -----------------------------------------------------------------------------
st.header("6. Sequence Loss Aggregation & Masked Normalization", divider="rainbow")

st.markdown(
    """
For a batch of sequences, individual token losses must be aggregated into a single scalar for backpropagation:

```text
Sequence_Loss = (1 / M) * SUM_{b=1}^{B} SUM_{t=1}^{T} m_{b, t} * Loss_{b, t}
```

Where:
* `m_{b, t} in {0, 1}` is a binary mask indicating valid tokens (`1`) versus ignored tokens (`0`).
* `M = SUM m_{b, t}` is the exact count of unmasked target tokens in the batch.

### Masking Scenarios in Production LLMs
1. **Padding Masking**: Padding tokens appended to equalize sequence lengths in a batch must have target label `-100` (`ignore_index = -100` in PyTorch).
2. **Prompt Masking in Instruction Fine-Tuning (SFT)**: During chat fine-tuning (e.g. ChatML), the model should learn to generate the assistant response, not memorize the user prompt. User prompt tokens are masked out with loss weight `0`, computing gradients exclusively over assistant tokens!
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: TEACHER FORCING & AUTOREGRESSIVE PARALLELISM
# -----------------------------------------------------------------------------
st.header("7. Teacher Forcing & Autoregressive Pretraining Parallelism", divider="rainbow")

st.markdown(
    """
**Teacher Forcing** is the training strategy where the model always receives the ground-truth previous tokens as input conditioning, regardless of whether its own previous prediction was correct or incorrect.

### The Exposure Bias Dilemma
While teacher forcing enables massive parallelism during training, it creates an architectural mismatch called **Exposure Bias**:
* **During Training**: The model is always conditioned on 100% perfect, flawless human text.
* **During Inference**: The model is conditioned on its own previously sampled outputs. If it generates an awkward phrase or slight error at step 10, it has never been trained to recover from imperfect historical contexts, potentially causing errors to cascade.
Modern RLHF (Reinforcement Learning from Human Feedback) and DPO directly address exposure bias by training on model-generated completions.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE AUTOREGRESSIVE LOSS PIPELINE
# -----------------------------------------------------------------------------
st.header("8. Production-Grade Autoregressive Loss Pipeline", divider="rainbow")

st.markdown(
    """
Production PyTorch implementation incorporating Causal Target Shifting, Prompt Masking, and Gemma-style Z-Loss:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductionLMLoss(nn.Module):
    # Production loss module featuring causal alignment, masking, and z-loss
    def __init__(self, ignore_index: int = -100, z_loss_weight: float = 1e-4, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.z_loss_weight = z_loss_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        
        # Shift so that token t predicts t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        # Reshape for cross entropy
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_targets = shift_targets.view(-1)

        # Standard cross entropy with ignore_index
        ce_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        # Auxiliary Z-Loss: penalizes large log-sum-exp to stabilize training
        if self.z_loss_weight > 0:
            log_z = torch.logsumexp(flat_logits, dim=-1)
            # Mask out ignored tokens from z-loss
            valid_mask = flat_targets != self.ignore_index
            z_loss = torch.mean(log_z[valid_mask] ** 2)
            total_loss = ce_loss + self.z_loss_weight * z_loss
        else:
            z_loss = torch.tensor(0.0)
            total_loss = ce_loss

        metrics = {
            "ce_loss": ce_loss.item(),
            "z_loss": z_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE LOSS & LABEL SMOOTHING WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Cross-Entropy Loss & Label Smoothing Workbench", divider="rainbow")

st.markdown(
    """
Visualize how model confidence `P(target)` directly dictates per-token Cross-Entropy loss
and see how **Label Smoothing** redistributes probability mass.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    target_prob = st.slider("Target Token Probability P(y)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
with col_ctrl2:
    smoothing_factor = st.slider("Label Smoothing Factor (epsilon)", min_value=0.0, max_value=0.2, value=0.0, step=0.02)

# Compute loss curves
ce_loss = -np.log(target_prob)
smoothed_loss = (1.0 - smoothing_factor) * ce_loss + smoothing_factor * (-np.log(0.0001))

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Raw Cross-Entropy Loss", f"{ce_loss:.3f} nats")
with col_m2:
    st.metric("Effective Perplexity", f"{np.exp(ce_loss):.2f}")
with col_m3:
    st.metric("Target Token Gradient (dL/dz)", f"{target_prob - 1.0:.3f}")

# Loss across probability range
p_range = np.linspace(0.01, 0.99, 50)
loss_curve = -np.log(p_range)
loss_df = pd.DataFrame({"P(Target)": p_range, "Loss (-log P)": loss_curve})

st.subheader("Cross-Entropy Loss vs. Prediction Confidence")
st.line_chart(loss_df.set_index("P(Target)"))

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is Cross-Entropy loss equivalent to Negative Log Likelihood for one-hot targets?"):
    st.markdown(
        """
        **Answer:**
        Cross-entropy is defined as `H(p, q) = - sum p_i log q_i`.
        When targets are one-hot, the true distribution `p` has `p_k = 1` for the correct target class and `p_i = 0` for all other classes `i != k`.
        The summation collapses to a single non-zero term: `- 1.0 * log q_k = -log q_k`.
        This is exactly the Negative Log Likelihood of the true target class.
        """
    )

with st.expander("Q2: What is Z-Loss, and why was it introduced in models like PaLM and Gemma?"):
    st.markdown(
        """
        **Answer:**
        Z-loss adds an auxiliary penalty `c * (log sum exp(z))^2` to the training objective.
        In deep transformers trained over billions of steps in lower precision (BF16), logit values can slowly drift toward large positive numbers. Large logits do not change Softmax probabilities (which are scale-invariant under constant shifts), but they cause internal activations to approach floating-point overflow limits and degrade gradient accuracy.
        Z-loss penalizes large partition functions, keeping logits centered near zero.
        """
    )

with st.expander("Q3: Explain the concept of 'Exposure Bias' caused by Teacher Forcing."):
    st.markdown(
        """
        **Answer:**
        During training with teacher forcing, the model is always fed the ground-truth prefix tokens at every step.
        During inference, the model must feed its own generated tokens back into itself.
        If the model generates an unusual token at step `t`, it enters a hidden state distribution it never experienced during pretraining, leading to cascading errors. SFT and DPO mitigate exposure bias by training on model rollouts.
        """
    )

with st.expander("Q4: How should loss be handled for prompt tokens during Instruction Fine-Tuning (SFT)?"):
    st.markdown(
        """
        **Answer:**
        Prompt tokens should be masked out using `ignore_index = -100`.
        The model's goal during SFT is to learn conditioned response generation, not to memorize prompt instructions. Computing loss on the prompt would waste capacity on language modeling the user input. Only the completion/assistant tokens should incur loss and backpropagate gradients.
        """
    )

with st.expander("Q5: What is the mathematical connection between Cross-Entropy Loss and Perplexity?"):
    st.markdown(
        """
        **Answer:**
        Perplexity is the exponentiated cross-entropy loss: `PPL = exp(Cross_Entropy_Loss)`.
        If a model achieves a cross-entropy loss of `2.0 nats` per token, its perplexity is `exp(2.0) = 7.39`.
        This intuitively means the model is, on average, as uncertain about the next token as if it were choosing uniformly among 7.39 equally likely words.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Forgetting to Mask Padding Tokens in Loss Aggregation**
        *Root Cause*: Passing padding tokens (ID 0) without setting `ignore_index=-100` in `nn.CrossEntropyLoss`.
        *Production Fix*: The model expends significant gradient budget predicting repetitive pad tokens, diluting actual language learning. Always set target padding to `-100`.
        """
    )
    st.warning(
        """
        **Trap 2: Normalizing Loss by Batch Size Instead of Valid Token Count**
        *Root Cause*: Using standard batch reduction when sequences have varying lengths of padding or prompt masks.
        *Production Fix*: Divide sequence loss by the exact count of unmasked target tokens (`sum(targets != -100)`), not by `batch_size * max_seq_len`.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Logit Drift Causing BF16 Softmax Underflow**
        *Root Cause*: Without Z-loss or logit capping, logits drift past 100.0, causing intermediate exponentiations in custom kernels to saturate.
        *Production Fix*: Implement soft-capping on logits (`logits = cap * tanh(logits / cap)`) as seen in Gemma 2 and Grok, or add Z-loss.
        """
    )
    st.warning(
        """
        **Trap 4: Excessive Label Smoothing in Foundation Pretraining**
        *Root Cause*: Setting `label_smoothing=0.1` during pretraining forces the model to assign 10% uniform probability to random tokens across a 128k vocabulary.
        *Production Fix*: Never use label smoothing during pretraining; it distorts exact factual knowledge memorization.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Perplexity Formula"):
        st.write("How is perplexity computed from cross-entropy loss L?")
        st.info("PPL = exp(L). It represents the effective branching factor of the model.")

    with st.expander("⚡ Flashcard 2: PyTorch Ignore Index"):
        st.write("What is the standard PyTorch sentinel value for ignored loss tokens?")
        st.info("-100 (nn.CrossEntropyLoss default ignore_index).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Easy Token Gradient"):
        st.write("What is the gradient magnitude for a target token predicted with 99% probability?")
        st.info("Approximately -0.01 (dL/dz = P - 1 = 0.99 - 1 = -0.01).")

    with st.expander("⚡ Flashcard 4: Teacher Forcing Benefit"):
        st.write("What is the main computational advantage of teacher forcing during training?")
        st.info("Enables full parallelization across all sequence positions via causal masking.")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why Cross-Entropy and Negative Log Likelihood are mathematically equivalent for one-hot targets.",
    "I can derive the Log-Sum-Exp trick and explain why it prevents FP16 floating-point underflow.",
    "I understand the difference between per-token loss and masked batch sequence loss aggregation.",
    "I know why prompt tokens should be masked with ignore_index=-100 during Instruction Fine-Tuning.",
    "I can explain how teacher forcing enables parallel pretraining and why it causes exposure bias.",
    "I understand why frontier models use Z-Loss or logit soft-capping to stabilize training runs.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Training Objectives, Cross-Entropy Loss, and Teacher Forcing Dynamics."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
