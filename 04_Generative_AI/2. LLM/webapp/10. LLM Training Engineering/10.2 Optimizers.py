"""
10.2 Optimizers
==============
Interactive Master Lesson exploring SGD, Momentum, Adam, AdamW, Decoupled Weight Decay,
and Distributed Optimizer States (ZeRO-1) in Modern Foundation Model Pretraining.
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
    page_title="10.2 Optimizers - LLM Architecture",
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
st.title("⚡ 10.2 Optimizers")
st.caption(
    "Chapter 10: LLM Training Engineering | Module 2 of 4"
)

st.markdown(
    """
Optimizers are the navigational algorithms that guide billions of model parameters
through steep, non-convex, and poorly conditioned loss terrains. While **SGD with Momentum**
dominated early computer vision, modern generative LLMs universally depend upon
**AdamW (Decoupled Weight Decay)** and memory-efficient variations like **8-bit Adam** and **Adafactor**.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & GRADIENT OPTIMIZATION LANDSCAPE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Gradient Optimization Landscape", divider="rainbow")

st.markdown(
    """
### The Evolution of Neural Network Optimizers
From static directional steps to second-moment coordinate scaling and decoupled weight regularization:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "optimizer_evolution",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("sgd", "Vanilla SGD (1950s)\\nw_{t+1} = w_t - lr * g_t\\nOscillates in ravines", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("momentum", "SGD with Momentum (1986)\\nv_t = beta * v_{t-1} + g_t\\nDampens transverse oscillations", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("adam", "Adam (Kingma & Ba, 2014)\\nFirst & Second Moments\\nL2 weight decay bug", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("adamw", "AdamW (Loshchilov & Hutter, 2017)\\nDecoupled Weight Decay\\nUniversal LLM Pretraining Standard", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("modern", "Memory-Efficient Optimizers\\n8-bit Adam, Adafactor, ZeRO-1\\nShards 12 bytes/param across GPUs", fillcolor="#7C3AED", fontcolor="#F8FAFC")

arch_dot.edge("sgd", "momentum", label="Polyak Velocity")
arch_dot.edge("momentum", "adam", label="Adaptive Coordinate Scaling")
arch_dot.edge("adam", "adamw", label="Decouple Regularization")
arch_dot.edge("adamw", "modern", label="Scale Memory & Quantization")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **AdamW as the Universal Default**: 100% of major foundation models (GPT-4, Claude 3.5, Llama 3, Mistral, Gemma 2) use **AdamW** with `beta_1 = 0.9`, `beta_2 = 0.95` (lower than the default 0.999 to forget historical variance faster in massive runs), and `eps = 1e-8`.
* **The Memory Tax**: AdamW requires **12 bytes of optimizer state per parameter**. For a 70B model, AdamW consumes **840 GB of GPU VRAM** just for optimizer tensors!
* **ZeRO-1 Sharding**: Modern pretraining frameworks (DeepSpeed, PyTorch FSDP) shard optimizer states across data-parallel ranks, reducing per-GPU optimizer memory to a fraction of the total.
"""
)

# Comparison Table
opt_matrix_df = pd.DataFrame(
    {
        "Optimizer": ["SGD", "SGD + Momentum", "Adam", "AdamW", "Adafactor", "8-Bit AdamW"],
        "Moments Maintained": ["None (0 bytes)", "Velocity (4 bytes)", "m_t + v_t (8 bytes)", "m_t + v_t (8 bytes)", "Row/Col Factored (~2 bytes)", "Quantized m_t, v_t (2 bytes)"],
        "Weight Decay Method": ["L2 Gradient Penalty", "L2 Gradient Penalty", "Coupled L2 Penalty", "True Decoupled Decay", "Decoupled Decay", "Decoupled Decay"],
        "Optimizer VRAM / Param": ["0 bytes", "4 bytes", "12 bytes (w/ master)", "12 bytes (w/ master)", "~4-6 bytes", "~6 bytes"],
        "LLM Adoption": ["Never used in pretraining", "Early vision / CNNs", "Deprecated in transformers", "100% Pretraining Standard", "T5, PaLM", "Fine-tuning / Low-memory"],
    }
)
st.dataframe(opt_matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: STOCHASTIC GRADIENT DESCENT (SGD) & CURVATURE
# -----------------------------------------------------------------------------
st.header("2. Stochastic Gradient Descent (SGD) & Curvature Pathology", divider="rainbow")

st.markdown(
    """
In Stochastic Gradient Descent, weights are updated proportionally to the negative gradient:

```text
W_{t+1} = W_t - lr * g_t
```

### The Ill-Conditioned Ravine Problem
In deep transformers, the loss surface features **severe ill-conditioning**:
* In directions of high curvature, gradients are massive, causing SGD to oscillate wildly across the canyon walls.
* In directions of low curvature, gradients are minuscule, causing SGD to crawl at a snail's pace along the canyon floor.
Finding a single global learning rate `lr` that prevents divergence in steep directions while making progress in flat directions is mathematically impossible in deep attention networks.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: MOMENTUM & POLYAK HEAVY-BALL ACCELERATION
# -----------------------------------------------------------------------------
st.header("3. Momentum & Polyak Heavy-Ball Acceleration", divider="rainbow")

st.markdown(
    """
Momentum simulates a heavy physical ball rolling down the loss surface, accumulating velocity:

```text
v_t = beta * v_{t-1} + g_t
W_{t+1} = W_t - lr * v_t
```
Where `beta in [0.9, 0.99]`.

### How Momentum Solves Transverse Oscillations
1. **Dampening High-Frequency Noise**: Across canyon walls, gradient vectors oscillate in opposite directions (+ and -). Averaging them cancels out transverse motion.
2. **Accelerating Consistent Directions**: Along the canyon floor, gradient vectors consistently point in the same forward direction. The velocity vector compounds, accelerating progress by `1 / (1 - beta)` (up to 10x faster).
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: ADAPTIVE MOMENT ESTIMATION (ADAM) MECHANICS
# -----------------------------------------------------------------------------
st.header("4. Adaptive Moment Estimation (Adam) Mechanics (Kingma & Ba, 2014)", divider="rainbow")

st.markdown(
    """
Adam computes individual, adaptive step sizes for every parameter coordinate by tracking both the **mean (first moment)** and **uncentered variance (second moment)** of gradients:

```text
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t           (First Moment: Velocity)
v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ^ 2)     (Second Moment: Squared Gradients)

m_hat = m_t / (1 - beta_1 ^ t)                        (Bias Correction)
v_hat = v_t / (1 - beta_2 ^ t)                        (Bias Correction)

W_{t+1} = W_t - lr * ( m_hat / (sqrt(v_hat) + eps) )
```

### Coordinate-Wise Invariance
If a parameter has large historical gradients, `sqrt(v_hat)` is large, scaling the update down to prevent explosion.
If a parameter has small historical gradients, `sqrt(v_hat)` is small, boosting the effective step size to ensure rapid feature learning.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: ADAMW & DECOUPLED WEIGHT DECAY FORMULATION
# -----------------------------------------------------------------------------
st.header("5. AdamW & Decoupled Weight Decay Formulation (Loshchilov & Hutter)", divider="rainbow")

st.markdown(
    """
In standard L2 regularization, an extra penalty `lambda/2 * ||W||^2` is added to the loss function `L`.
When computing gradients:
```text
g_t_regularized = g_t + lambda * W_t
```

### The Fatal Flaw in Standard Adam (Coupled L2 Penalty)
When `g_t_regularized` is passed into Adam, the weight decay term `lambda * W_t` gets divided by `sqrt(v_hat)`:
```text
Update = (m_hat) / (sqrt(v_hat) + eps)
```
* Parameters with **large historical gradients** (frequently active features) have their weight decay **suppressed**!
* Parameters with **small historical gradients** experience **disproportionately aggressive weight decay**!
This directly inverts the intended regularization behavior.

### The AdamW Decoupled Solution
Loshchilov & Hutter (2017) restored true weight decay by decoupling it from gradient moment updates:

```text
W_{t+1} = W_t - lr * ( m_hat / (sqrt(v_hat) + eps) ) - lr * lambda * W_t
```
The weight vector shrinks directly by `(1 - lr * lambda)` on every step, independent of gradient scale!
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: WEIGHT DECAY REGULARIZATION VS. L2 PENALTY
# -----------------------------------------------------------------------------
st.header("6. Weight Decay Regularization vs. L2 Penalty Divergence", divider="rainbow")

st.markdown(
    """
* **In SGD**: L2 penalty and decoupled weight decay are mathematically identical up to a constant learning rate scale.
* **In Adaptive Optimizers (Adam, RMSProp)**: L2 regularization and weight decay diverge completely.
* **Empirical Result**: Training transformers with AdamW achieves significantly lower generalization perplexity than standard Adam with L2 regularization, establishing AdamW as the gold standard for all modern LLMs.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: OPTIMIZER STATE MEMORY DYNAMICS & ZERO-1
# -----------------------------------------------------------------------------
st.header("7. Optimizer State Memory Dynamics & Sharded Optimizer States (ZeRO-1)", divider="rainbow")

st.markdown(
    """
### The 16-Byte-Per-Parameter Memory Tax
For a mixed-precision BF16/FP32 pretraining run, every parameter requires:
1. **Model Weights (BF16)**: 2 bytes
2. **Gradients (BF16)**: 2 bytes
3. **Master Weights (FP32)**: 4 bytes
4. **AdamW First Moment m_t (FP32)**: 4 bytes
5. **AdamW Second Moment v_t (FP32)**: 4 bytes
**Total: 16 bytes per parameter** (where 12 bytes are purely optimizer-related!).

### ZeRO-1 (Zero Redundancy Optimizer Stage 1)
Instead of replicating the 12 bytes of optimizer state on every data-parallel GPU:
* Partition optimizer states evenly across `N_data` GPUs.
* Each GPU updates only `1 / N_data` of the parameters, then executes an all-gather to broadcast the updated weights.
* Cuts optimizer VRAM consumption by `N_data` times with zero communication penalty!
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: PRODUCTION-GRADE PYTORCH ADAMW IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("8. Production-Grade PyTorch AdamW & 8-Bit Optimizer Implementation", divider="rainbow")

st.markdown(
    """
A complete, pure-Python reference implementation of the vectorized AdamW algorithm with decoupled decay:
"""
)

code_snippet = """import torch
from torch.optim.optimizer import Optimizer
import math

class CustomAdamW(Optimizer):
    # Pure-PyTorch implementation of AdamW with decoupled weight decay
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 1. Decoupled Weight Decay: W = W * (1 - lr * decay)
                if decay != 0:
                    p.mul_(1.0 - lr * decay)

                # 2. Update First Moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Update Second Moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Bias Correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 5. Parameter Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 9: INTERACTIVE OPTIMIZER TRAJECTORY WORKBENCH
# -----------------------------------------------------------------------------
st.header("9. Interactive Optimizer Trajectory & Weight Decay Workbench", divider="rainbow")

st.markdown(
    """
Simulate optimization convergence trajectories on an anisotropic Rosenbrock valley
comparing **Vanilla SGD**, **SGD + Momentum**, and **AdamW**.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    sim_optimizer = st.selectbox("Select Optimizer Algorithm", options=["Vanilla SGD", "SGD with Momentum", "AdamW"], index=2)
    sim_lr = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
with col_ctrl2:
    sim_weight_decay = st.slider("Decoupled Weight Decay", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    sim_steps = st.slider("Optimization Steps", min_value=20, max_value=150, value=60, step=10)

# Simulate 2D optimization on ill-conditioned ellipse: f(x, y) = 0.5 * (x^2 + 20 * y^2)
x, y = 5.0, 5.0
path_x, path_y, losses = [x], [y], [0.5 * (x ** 2 + 20 * y ** 2)]

vx, vy = 0.0, 0.0
m_x, m_y = 0.0, 0.0
v_x, v_y = 0.0, 0.0

for t in range(1, sim_steps + 1):
    gx = x
    gy = 20.0 * y

    if sim_optimizer == "Vanilla SGD":
        x = x - sim_lr * gx - sim_lr * sim_weight_decay * x
        y = y - sim_lr * gy - sim_lr * sim_weight_decay * y
    elif sim_optimizer == "SGD with Momentum":
        vx = 0.9 * vx + gx
        vy = 0.9 * vy + gy
        x = x - sim_lr * vx - sim_lr * sim_weight_decay * x
        y = y - sim_lr * vy - sim_lr * sim_weight_decay * y
    elif sim_optimizer == "AdamW":
        m_x = 0.9 * m_x + 0.1 * gx
        m_y = 0.9 * m_y + 0.1 * gy
        v_x = 0.95 * v_x + 0.05 * (gx ** 2)
        v_y = 0.95 * v_y + 0.05 * (gy ** 2)

        m_hat_x = m_x / (1.0 - 0.9 ** t)
        m_hat_y = m_y / (1.0 - 0.9 ** t)
        v_hat_x = v_x / (1.0 - 0.95 ** t)
        v_hat_y = v_y / (1.0 - 0.95 ** t)

        x = (x * (1.0 - sim_lr * sim_weight_decay)) - sim_lr * (m_hat_x / (np.sqrt(v_hat_x) + 1e-8))
        y = (y * (1.0 - sim_lr * sim_weight_decay)) - sim_lr * (m_hat_y / (np.sqrt(v_hat_y) + 1e-8))

    path_x.append(x)
    path_y.append(y)
    losses.append(0.5 * (x ** 2 + 20 * y ** 2))

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.subheader("Convergence Loss Trajectory")
    st.line_chart(pd.DataFrame({"Step": range(len(losses)), "Loss": losses}).set_index("Step"))
with col_m2:
    st.subheader("2D Parameter Coordinates")
    st.dataframe(pd.DataFrame({"Step": range(len(path_x)), "Param X": np.round(path_x, 3), "Param Y": np.round(path_y, 3)}).tail(8), use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is beta_2 set to 0.95 in LLM pretraining instead of the default 0.999?"):
    st.markdown(
        """
        **Answer:**
        At `beta_2 = 0.999`, the second moment estimate `v_t` has an effective memory window of `1 / (1 - 0.999) = 1,000 steps`.
        In multi-billion token training runs with changing data distributions and learning rate schedules, an optimizer that remembers 1,000 steps into the past responds too slowly to sudden curvature shifts. Setting `beta_2 = 0.95` shrinks the memory window to `1 / (1 - 0.95) = 20 steps`, allowing the optimizer to adapt rapidly and avoid loss instability.
        """
    )

with st.expander("Q2: Mathematically, why did Loshchilov & Hutter create AdamW?"):
    st.markdown(
        """
        **Answer:**
        In classical Adam, L2 regularization was added to the loss function, meaning weight decay was scaled inversely by `sqrt(v_t)`.
        Features with large gradients received almost zero weight decay, while features with tiny gradients received excessive weight decay.
        AdamW decoupled weight decay by subtracting `lr * lambda * W` directly from parameters during the update step, ensuring all weights experience proportional regularization regardless of their gradient magnitude.
        """
    )

with st.expander("Q3: How much GPU VRAM does AdamW consume for a 70B parameter model?"):
    st.markdown(
        """
        **Answer:**
        AdamW stores:
        1. FP32 master weights: `70B * 4 bytes = 280 GB`
        2. FP32 first moment `m_t`: `70B * 4 bytes = 280 GB`
        3. FP32 second moment `v_t`: `70B * 4 bytes = 280 GB`
        Total: **840 GB of GPU VRAM** solely for optimizer states (excluding model weights, gradients, and activations).
        """
    )

with st.expander("Q4: What is the difference between ZeRO-1 and standard Data Parallelism?"):
    st.markdown(
        """
        **Answer:**
        In standard DDP, every GPU holds an identical full copy of the 840 GB optimizer state.
        In ZeRO-1 (Zero Redundancy Optimizer Stage 1), the optimizer states are sharded equally across the data-parallel GPUs (e.g. across 64 GPUs, each GPU holds only `840 / 64 = 13.1 GB`). Each GPU updates its assigned shard, and updated weights are shared via an all-gather communication step.
        """
    )

with st.expander("Q5: What is 8-bit Adam (BitsAndBytes) and how does it save memory?"):
    st.markdown(
        """
        **Answer:**
        8-bit Adam quantizes the optimizer state tensors (`m_t` and `v_t`) from 32-bit floats down to non-linear 8-bit block-wise floating point representations.
        This reduces optimizer memory footprint from 12 bytes per parameter to approximately **6 bytes per parameter** (a 50-75% reduction) with zero degradation in pretraining perplexity.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Setting Epsilon Too Small in FP16 Precision**
        *Root Cause*: Setting `eps = 1e-16` causes `sqrt(v_t) + eps` to round down to zero if `v_t` is tiny in FP16 precision, triggering division by zero and `NaN` gradients.
        *Production Fix*: In mixed-precision training, always set `eps = 1e-8` or `1e-6` to guarantee numerical stability.
        """
    )
    st.warning(
        """
        **Trap 2: Applying Weight Decay to Biases and Normalization Scales**
        *Root Cause*: Passing all parameters uniformly into AdamW with weight decay.
        *Production Fix*: Always split parameters into two optimizer groups: 2D weights receive weight decay, while 1D LayerNorm scales and biases have `weight_decay = 0.0`.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Omitting Bias Correction in Custom Kernels**
        *Root Cause*: Forgetting to divide `m_t` and `v_t` by `(1 - beta^step)` in custom Triton or CUDA optimizer implementations.
        *Production Fix*: In early steps, `m_t` and `v_t` are severely biased toward zero, causing initial step sizes to be wildly inaccurate. Always enforce bias correction.
        """
    )
    st.warning(
        """
        **Trap 4: Naive Master Weight Casting**
        *Root Cause*: Updating BF16 weights directly without maintaining a 32-bit FP32 master weight copy.
        *Production Fix*: In BF16, adding small updates (`lr * step approx 1e-6`) to weights of magnitude `1.0` underflows to zero due to 8-bit mantissa limits. Always update FP32 master weights.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Standard Beta_2 for LLMs"):
        st.write("What value of beta_2 is standard in Llama 3 and modern pretraining?")
        st.info("beta_2 = 0.95 (instead of PyTorch default 0.999).")

    with st.expander("⚡ Flashcard 2: AdamW Total Memory Per Param"):
        st.write("How many total bytes does mixed-precision AdamW require per parameter?")
        st.info("16 bytes (2B weight + 2B grad + 4B master weight + 4B m_t + 4B v_t).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Decoupled Weight Decay Formula"):
        st.write("How does AdamW update weights with weight decay lambda?")
        st.info("W_{t+1} = W_t * (1 - lr * lambda) - lr * (m_hat / (sqrt(v_hat) + eps)).")

    with st.expander("⚡ Flashcard 4: ZeRO-1 Purpose"):
        st.write("What specific memory does ZeRO Stage 1 eliminate across data-parallel ranks?")
        st.info("Optimizer state redundancy (shards the 12B/param optimizer state across GPUs).")

# -----------------------------------------------------------------------------
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why Vanilla SGD fails in deep ill-conditioned transformer loss landscapes.",
    "I can mathematically derive the first and second moment updates in Adam.",
    "I understand why AdamW decouples weight decay from gradient second-moment scaling.",
    "I know why beta_2 is lowered to 0.95 during multi-trillion token LLM pretraining.",
    "I can calculate the exact 16-byte-per-parameter memory breakdown in mixed-precision AdamW.",
    "I understand how ZeRO-1 shards optimizer states across data-parallel ranks without communication penalties.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Deep Learning Optimizers, AdamW Dynamics, and Sharded Optimizer Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
