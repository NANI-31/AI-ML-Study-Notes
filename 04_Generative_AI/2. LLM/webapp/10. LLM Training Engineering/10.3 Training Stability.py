"""
10.3 Training Stability
=======================
Interactive Master Lesson exploring Vanishing/Exploding Gradients, Gradient Clipping,
Numerical Precision Stability, FP16/BF16/FP8 Mixed Precision, and Dynamic Loss Scaling.
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
    page_title="10.3 Training Stability - LLM Architecture",
    page_icon="🛡️",
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
st.title("🛡️ 10.3 Training Stability")
st.caption(
    "Chapter 10: LLM Training Engineering | Module 3 of 4"
)

st.markdown(
    """
In large-scale foundation model pretraining, training stability is a multimillion-dollar engineering imperative.
Loss spikes, numeric overflows (`NaN`), gradient explosions, and underflow crashes can silently corrupt
months of supercomputer cluster execution. Mastering numerical precision, **BF16 vs. FP16 dynamics**,
**loss scaling**, and **logit soft-capping** ensures flawless convergence.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & STABILITY FRONTIER
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Stability Frontier", divider="rainbow")

st.markdown(
    """
### The Numerical Stability Defense Architecture
Modern LLM stacks employ multiple coordinated layers of defense against numerical collapse:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "stability_architecture",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("input", "Inputs / Embeddings\\nRMSNorm / muP", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("attn", "Attention Blocks\\nScaled QK / FlashAttn", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("cap", "Logit Soft-Capping\\ncap * tanh(z / cap)", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("loss", "FP32 Loss Calculation\\nLog-Sum-Exp Trick", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("grad", "Grad Clip & Scale\\n||g|| <= 1.0, Dynamic Scale", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("master", "FP32 Master Update\\nAdamW Step", fillcolor="#1E293B", fontcolor="#F8FAFC")

arch_dot.edge("input", "attn", label="Bounded Variance")
arch_dot.edge("attn", "cap", label="Raw Activations")
arch_dot.edge("cap", "loss", label="Bounded Logits")
arch_dot.edge("loss", "grad", label="Stable Backprop")
arch_dot.edge("grad", "master", label="Safe Updates")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **The Transition from FP16 to BF16**: FP16 possesses only a 5-bit exponent (maximum representable value is `65,504`), causing frequent arithmetic overflow during multi-billion parameter runs. **Bfloat16 (BF16)** adopts an 8-bit exponent matching FP32 (max value `3.4e38`), eliminating overflow and rendering dynamic loss scaling obsolete on modern NVIDIA Ampere/Hopper GPUs!
* **Logit Soft-Capping (Gemma 2 / Grok)**: To prevent runaway logits in attention and LM heads, modern architectures apply hyperbolic tangent capping:
  ```text
  Logits = cap * tanh(Logits / cap)
  ```
  Where `cap = 50.0` for attention logits and `cap = 30.0` for final vocabulary logits.
"""
)

# Precision Format Comparison Table
prec_df = pd.DataFrame(
    {
        "Format": ["FP32 (Single Precision)", "FP16 (Half Precision)", "BF16 (Brain Floating Point)", "FP8 E4M3", "FP8 E5M2"],
        "Total Bits": ["32 Bits", "16 Bits", "16 Bits", "8 Bits", "8 Bits"],
        "Exponent Bits": ["8 Bits", "5 Bits (Overflow risk!)", "8 Bits (Same as FP32!)", "4 Bits (High precision)", "5 Bits (Wide range)"],
        "Mantissa Bits": ["23 Bits", "10 Bits", "7 Bits", "3 Bits", "2 Bits"],
        "Dynamic Range": ["1e-38 to 3e38", "6e-5 to 65,504", "1e-38 to 3e38", "1e-6 to 448", "6e-5 to 57,344"],
        "Loss Scaling Needed?": ["No", "Yes (Mandatory)", "No (Native stability)", "Yes (Amax scaling)", "Optional"],
    }
)
st.dataframe(prec_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: VANISHING GRADIENTS IN ULTRA-DEEP NETWORKS
# -----------------------------------------------------------------------------
st.header("2. Vanishing Gradients in Ultra-Deep Networks & Residual Mitigations", divider="rainbow")

st.markdown(
    """
In networks with 80+ stacked layers, backpropagating gradients through repeated matrix multiplications
can cause gradient magnitudes to decay exponentially toward zero:

```text
||g_1|| approx ||g_L|| * PRODUCT_{l=1}^{L} ||W_l||
```
If the singular values of `W_l` are slightly less than 1.0 (e.g. `0.95`), then across 80 layers:
`0.95^80 = 0.016` (a 98.4% decay in learning signal).

### How Pre-RMSNorm Residuals Solved Vanishing Gradients
Modern transformers compute `x_{l+1} = x_l + SubBlock(RMSNorm(x_l))`.
By the product rule, the derivative with respect to `x_l` is:
```text
dL / dx_l = dL / dx_{l+1} + dL / dx_{l+1} * dSubBlock / dx_l
```
The unscaled identity term `dL / dx_{l+1}` flows straight back through the residual backbone, guaranteeing that early layers receive full gradient signals regardless of network depth.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: EXPLODING GRADIENTS & ATTENTION LOGIT INSTABILITIES
# -----------------------------------------------------------------------------
st.header("3. Exploding Gradients & Attention Logit Instabilities", divider="rainbow")

st.markdown(
    """
Conversely, exploding gradients occur when gradient norms grow exponentially across layers,
triggering `inf` values and optimizer crashes.

### The Attention Score Explosion Mechanism
In self-attention, query and key vectors are multiplied: `z = Q * K^T`.
* If head dimension `d_k = 128`, and entries of `Q` and `K` are independent random variables with zero mean and unit variance:
  ```text
  Var(q * k^T) = SUM_{i=1}^{d_k} Var(q_i * k_i) = d_k = 128
  ```
* Standard deviation is `sqrt(128) = 11.31`.
* Without division by `sqrt(d_k)`, dot products reach `+40` to `+60`.
* In Softmax: `exp(60) = 1.14e26`, which completely overflows FP16 (`max = 65,504`)!
* Scaling by `1 / sqrt(d_k)` brings variance back to `1.0`, keeping attention scores in safe numerical ranges.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: GRADIENT NORM CLIPPING & OUTLIER THRESHOLDING
# -----------------------------------------------------------------------------
st.header("4. Gradient Norm Clipping & Outlier Thresholding", divider="rainbow")

st.markdown(
    """
Gradient clipping prevents parameter weights from making catastrophic updates when encountering
anomalous training batches:

```text
g_clipped = g * min(1.0, max_norm / ||g||_2)
```

### Production Norm Monitoring
During pretraining, the global gradient norm is logged on every step:
* **Healthy Training**: `||g||_2` fluctuates smoothly between `0.2` and `0.8`.
* **Approaching Instability**: `||g||_2` surges past `5.0` or `10.0`.
* **Active Clipping**: Rescales the gradient vector back to `1.0`, preserving the optimization trajectory's direction while capping its velocity.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: NUMERICAL STABILITY DYNAMICS
# -----------------------------------------------------------------------------
st.header("5. Numerical Stability Dynamics (Epsilon, Log-Sum-Exp, Soft-Capping)", divider="rainbow")

st.markdown(
    """
Every mathematical operator in the training loop requires defensive engineering:

### 1. The Epsilon Floor
In LayerNorm / RMSNorm, variance is computed as `sigma = sqrt( 1/d sum x^2 + eps )`.
* If `eps` is too small (`1e-12`), evaluating inputs with near-zero variance produces `1 / sqrt(1e-12) = 1e6`, causing catastrophic activation explosion.
* Modern LLMs fix `eps = 1e-5` or `1e-6` for RMSNorm.

### 2. The Log-Sum-Exp Trick
Softmax evaluates:
```text
Softmax(z_i) = exp(z_i - max(z)) / SUM_j exp(z_j - max(z))
```
Subtracting `max(z)` ensures the largest evaluated exponent is `exp(0) = 1.0`, guaranteeing zero overflow.

### 3. Logit Soft-Capping
Applied in Gemma 2:
```text
z_capped = cap * tanh(z / cap)
```
Ensures logits never exceed `[-cap, +cap]` regardless of how large the dot products grow.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: MIXED PRECISION TRAINING FORMATS
# -----------------------------------------------------------------------------
st.header("6. Mixed Precision Training Formats (FP32, FP16, BF16, FP8)", divider="rainbow")

st.markdown(
    """
Training in pure 32-bit float (FP32) is computationally unviable for multi-trillion token LLMs.
Modern systems deploy **Automatic Mixed Precision (AMP)**:
"""
)

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("BF16: The Modern LLM Workhorse")
    st.markdown(
        """
        * Identical 8-bit dynamic range as FP32.
        * Eliminates underflow and overflow issues.
        * Supported natively by NVIDIA Ampere (A100), Hopper (H100), and Google TPUs.
        * Standard pretraining precision for Llama 3, Mistral, and Gemma.
        """
    )
with col_p2:
    st.subheader("FP8: The Hopper Frontier")
    st.markdown(
        """
        * Supported on NVIDIA H100 Hopper Tensor Cores.
        * Doubles matrix arithmetic throughput over 16-bit.
        * Uses **E4M3** for forward passes (maximizing precision) and **E5M2** for backward gradients (maximizing dynamic range).
        * Requires delayed block-wise scaling factors (`amax`).
        """
    )

# -----------------------------------------------------------------------------
# SECTION 7: DYNAMIC LOSS SCALING MECHANICS
# -----------------------------------------------------------------------------
st.header("7. Dynamic Loss Scaling Mechanics in FP16 Pretraining", divider="rainbow")

st.markdown(
    """
When training in **FP16**, gradients are frequently smaller than `6e-5` (the minimum representable positive number in FP16).
Without loss scaling, these small gradient values round down to `0.0`, resulting in gradient underflow.

### The Loss Scaler Algorithm
1. **Scale Up Loss**: Multiply the loss by a large factor `S` (e.g. `S = 65,536`):
   ```text
   Loss_scaled = Loss * S
   ```
2. **Backpropagation**: Gradients are scaled up by `S`, shifting them into the representable FP16 range.
3. **Unscaling & Check**: Before updating weights, divide gradients by `S`:
   `g = g_scaled / S`.
4. **Dynamic Adjustment**:
   * If any gradient contains `inf` or `NaN`: Skip the optimizer step and shrink the scale: `S = S / 2`.
   * If gradients remain healthy for `N` consecutive steps (e.g. 2,000 steps): Increase the scale: `S = S * 2`.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE NUMERICAL STABILITY MONITOR IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("8. Production-Grade Dynamic Loss Scaler & Numerical Stability Monitor", divider="rainbow")

st.markdown(
    """
A complete PyTorch implementation of a dynamic loss scaler with logit soft-capping and gradient norm monitoring:
"""
)

code_snippet = """import torch
import torch.nn as nn
from typing import Tuple, Dict

class StabilityMonitor:
    # Production numerical stability monitor and logit capper
    @staticmethod
    def soft_cap_logits(logits: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
        # Gemma 2 style logit soft-capping: cap * tanh(logits / cap)
        return cap * torch.tanh(logits / cap)

    @staticmethod
    def audit_gradient_health(model: nn.Module) -> Tuple[bool, float]:
        total_norm_sq = 0.0
        has_nan_or_inf = False
        
        for p in model.parameters():
            if p.grad is not None:
                grad = p.grad.detach()
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    has_nan_or_inf = True
                    break
                param_norm = grad.norm(2).item()
                total_norm_sq += param_norm ** 2
                
        global_norm = total_norm_sq ** 0.5
        is_healthy = (not has_nan_or_inf) and (global_norm < 100.0)
        return is_healthy, global_norm

class DynamicLossScaler:
    # Pure-Python dynamic loss scaler for FP16 training
    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._consecutive_healthy = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def step(self, optimizer: torch.optim.Optimizer, model: nn.Module) -> bool:
        # Unscale gradients
        inv_scale = 1.0 / self.scale
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(inv_scale)

        is_healthy, norm = StabilityMonitor.audit_gradient_health(model)
        
        if not is_healthy:
            # Skip step and decrease scale
            self.scale *= self.backoff_factor
            self._consecutive_healthy = 0
            optimizer.zero_grad(set_to_none=True)
            return False
            
        # Execute optimizer step
        optimizer.step()
        self._consecutive_healthy += 1
        
        if self._consecutive_healthy >= self.growth_interval:
            self.scale *= self.growth_factor
            self._consecutive_healthy = 0
            
        return True
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE STABILITY & MIXED PRECISION WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Mixed Precision & Gradient Stability Workbench", divider="rainbow")

st.markdown(
    """
Experiment with simulated activations across **FP32**, **BF16**, and **FP16** precision limits,
and observe how **Logit Soft-Capping** protects against Softmax saturation.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    test_logit_val = st.slider("Raw Test Logit Magnitude", min_value=1.0, max_value=120.0, value=65.0, step=5.0)
    enable_capping = st.checkbox("Enable Logit Soft-Capping (cap = 30.0)", value=True)
with col_ctrl2:
    soft_cap_limit = st.slider("Capping Threshold (cap)", min_value=10.0, max_value=50.0, value=30.0, step=5.0)
    head_dim_sim = st.selectbox("Head Dimension d_k", options=[64, 128, 256], index=1)

# Apply capping
if enable_capping:
    effective_logit = soft_cap_limit * np.tanh(test_logit_val / soft_cap_limit)
else:
    effective_logit = test_logit_val

# Compare precision limits
fp16_max = 65504.0
bf16_max = 3.4e38

is_fp16_overflow = np.exp(min(effective_logit, 100)) > fp16_max

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Effective Logit", f"{effective_logit:.2f}", f"Raw: {test_logit_val:.1f}")
with col_m2:
    st.metric("FP16 Exponent State", "OVERFLOW!" if is_fp16_overflow else "Safe Range", delta_color="inverse" if is_fp16_overflow else "normal")
with col_m3:
    st.metric("BF16 Exponent State", "Safe Range (Max 1e38)")

# Interactive capping curve
x_vals = np.linspace(-100, 100, 200)
uncapped_vals = x_vals
capped_vals = soft_cap_limit * np.tanh(x_vals / soft_cap_limit)

curve_df = pd.DataFrame(
    {
        "Raw Input Logit": x_vals,
        "Uncapped Logit": uncapped_vals,
        "Soft-Capped Logit": capped_vals,
    }
)

st.subheader("Logit Soft-Capping vs. Uncapped Linear Activation")
st.line_chart(curve_df.set_index("Raw Input Logit"))

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why has BF16 almost entirely replaced FP16 in modern LLM pretraining?"):
    st.markdown(
        """
        **Answer:**
        FP16 has only a 5-bit exponent, limiting its maximum representable value to `65,504`.
        In deep networks with trillions of tokens, activations and attention dot products routinely cross this threshold, triggering arithmetic overflow (`inf` and `NaN`) and requiring complex dynamic loss scalers.
        BF16 allocates an 8-bit exponent (identical to FP32), expanding its maximum representable value to `~3.4e38`.
        This completely eliminates overflow, allowing stable pretraining without loss scaling.
        """
    )

with st.expander("Q2: What is Logit Soft-Capping and why did Gemma 2 and Grok introduce it?"):
    st.markdown(
        """
        **Answer:**
        In extreme contexts, query-key dot products and output logits can drift to large values (> 50.0).
        This forces the Softmax distribution into a saturated one-hot state, killing gradient flow through non-target tokens and destabilizing training.
        Logit soft-capping computes `cap * tanh(logits / cap)`, smoothly bounding logits within `[-cap, +cap]`.
        This guarantees numerical stability without setting hard unnatural clamp boundaries.
        """
    )

with st.expander("Q3: How does dividing by sqrt(d_k) in attention prevent vanishing gradients?"):
    st.markdown(
        """
        **Answer:**
        The dot product of two vectors of dimension `d_k` has variance proportional to `d_k`.
        For `d_k = 128`, dot products reach values of +40 to +60.
        When passed into Softmax, the distribution collapses into a sharp delta peak on the highest value.
        The derivative of Softmax at saturated regions is near zero (`P * (1 - P) approx 0`), completely killing gradients flowing back to the query and key projections. Scaling by `1 / sqrt(d_k)` keeps the Softmax input variance near 1.0, preserving healthy gradient flow.
        """
    )

with st.expander("Q4: What causes Loss Spikes during multi-month cluster pretraining?"):
    st.markdown(
        """
        **Answer:**
        Common root causes include:
        1. **Data Outliers**: Ingesting corrupted web documents with thousands of repetitive identical characters or malformed encodings.
        2. **Second Moment Decay**: Optimizer variance estimates `v_t` underestimating sudden curvature shifts.
        3. **Hardware Silent Data Corruption (SDC)**: Transistor-level bit flips in GPU Tensor Cores.
        **Mitigations**: Global gradient clipping, robust data filtering, and reverting to the most recent checkpoint while skipping the affected data shard.
        """
    )

with st.expander("Q5: Why is Loss computed in FP32 even when forward activations are in BF16?"):
    st.markdown(
        """
        **Answer:**
        Computing Cross-Entropy requires the Log-Sum-Exp partition function across the entire vocabulary (e.g. 128,256 tokens).
        Summing thousands of exponentiated logits in 16-bit precision leads to severe accumulation rounding errors due to limited mantissa bits.
        Computing Log-Sum-Exp in FP32 ensures exact numerical normalization and preserves precise gradient magnitudes.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Omitting FP32 Casting in Cross-Entropy Loss**
        *Root Cause*: Executing `F.cross_entropy` directly on raw BF16 or FP16 logits over a 128k vocabulary.
        *Production Fix*: Always cast logits to float before loss computation: `F.cross_entropy(logits.float(), targets)`.
        """
    )
    st.warning(
        """
        **Trap 2: Using Epsilon = 1e-12 in RMSNorm with BF16**
        *Root Cause*: The lowest positive normal number in BF16 is `1.17e-38`, but small mantissa precision causes `1e-12` additions to be rounded to zero.
        *Production Fix*: Set `eps = 1e-5` or `1e-6` in RMSNorm layers to prevent division-by-zero instability.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Loss Scaler Scale Exploding Beyond FP32 Limits**
        *Root Cause*: Never capping `max_scale` in dynamic loss scalers, causing the scale factor itself to overflow.
        *Production Fix*: Cap dynamic loss scale to `2^24` or switch entirely to BF16.
        """
    )
    st.warning(
        """
        **Trap 4: Silent Gradient Underflow in Pure FP16 Without Scaler**
        *Root Cause*: Running mixed-precision training in FP16 without engaging PyTorch's `GradScaler`.
        *Production Fix*: Over 80% of backward gradients underflow to zero, causing the model to freeze learning. Always engage `torch.cuda.amp.GradScaler` if training in FP16.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: FP16 Max Value"):
        st.write("What is the maximum representable value in IEEE-754 FP16?")
        st.info("65,504 (exceeding this value produces +inf).")

    with st.expander("⚡ Flashcard 2: BF16 Exponent Size"):
        st.write("How many exponent bits does BF16 possess?")
        st.info("8 bits (identical dynamic range to FP32).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Logit Soft-Capping Formula"):
        st.write("What mathematical function is used for logit soft-capping?")
        st.info("cap * tanh(logits / cap).")

    with st.expander("⚡ Flashcard 4: Attention Scaling Factor"):
        st.write("Why scale Q*K^T by 1 / sqrt(d_k)?")
        st.info("To maintain unit variance and prevent Softmax saturation and vanishing gradients.")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why Pre-RMSNorm residual connections prevent vanishing gradients in deep networks.",
    "I can explain why unscaled attention dot products cause Softmax saturation and overflow.",
    "I understand the bit allocation and dynamic range differences between FP32, FP16, BF16, and FP8.",
    "I know how Dynamic Loss Scaling prevents gradient underflow when training in FP16.",
    "I understand how logit soft-capping (Gemma 2) stabilizes logits across long training runs.",
    "I can explain why cross-entropy loss must be computed in FP32 precision.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Training Stability, Precision Formats, and Numerical Defense Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
