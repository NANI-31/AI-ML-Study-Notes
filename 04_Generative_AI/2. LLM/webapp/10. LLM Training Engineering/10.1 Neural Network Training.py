"""
10.1 Neural Network Training
============================
Interactive Master Lesson exploring Forward Propagation, Backpropagation,
Gradient Descent, Adaptive Optimizers, Learning Rate Schedules, Weight Initialization, and Gradient Clipping.
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
    page_title="10.1 Neural Network Training - LLM Architecture",
    page_icon="🧠",
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
st.title("🧠 10.1 Neural Network Training")
st.caption(
    "Chapter 10: LLM Training Engineering | Module 1 of 4"
)

st.markdown(
    """
Training an 80-layer, 70-billion-parameter neural network across billions of parameter matrices
pushes the fundamentals of numerical optimization to the limit. Navigating forward tensor flows,
reverse automatic differentiation, non-convex gradient descent, learning rate schedules, and
stable weight initialization (such as **Maximal Update Parametrization / muP**) forms the backbone
of modern foundation model training engineering.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & DEEP LEARNING ENGINE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Deep Learning Engine", divider="rainbow")

st.markdown(
    """
### The Optimization Cycle of Deep Transformer Networks
Training deep neural networks relies on the continuous loop of activation propagation, error derivation, and parameter updates.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "nn_training_engine",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("init", "Weight Initialization\\nmuP / Scaled Normal", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("fwd", "Forward Propagation\\nLayerNorm + Attn + MLP", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("loss", "Loss Evaluation\\nCross-Entropy L(y, p)", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("bwd", "Reverse Autodiff\\ndL / dW", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("clip", "Gradient Clipping\\n||g|| <= max_norm", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("step", "Optimizer Step (AdamW)\\nCosine LR Schedule", fillcolor="#1E293B", fontcolor="#F8FAFC")

arch_dot.edge("init", "fwd", label="W_0")
arch_dot.edge("fwd", "loss", label="Activations")
arch_dot.edge("loss", "bwd", label="Scalar Loss")
arch_dot.edge("bwd", "clip", label="Raw Gradients")
arch_dot.edge("clip", "step", label="Clipped Grads")
arch_dot.edge("step", "fwd", label="W_{t+1}")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Warmup-Stable-Decay (WSD) Schedules**: While earlier models used static Cosine Annealing, modern models (MiniCPM, DeepSeek-V2) adopt **WSD schedules**. The model trains at a flat maximum learning rate for 80-90% of the run, allowing researchers to evaluate intermediate checkpoints and branch off continual pretraining runs before executing a rapid 10% cooldown.
* **Maximal Update Parametrization (muP)**: Traditional initialization causes optimal hyperparameters (learning rate, weight decay) to drift as model width increases. muP scales weight matrices such that optimal learning rates found on small 100M proxy models transfer directly to 70B+ models without tuning.
"""
)

# Training Engineering Dimensions Table
eng_df = pd.DataFrame(
    {
        "Training Component": ["Weight Initialization", "Forward Pass", "Backpropagation", "Gradient Clipping", "Learning Rate Schedule", "Optimizer Step"],
        "Mathematical Focus": ["Var(W) scaling, muP", "Tensor contractions, GEMM", "Jacobian vector products", "L2 norm rescaling", "Cosine / WSD decay", "Adaptive moment updates"],
        "Precision Format": ["FP32", "BF16 / FP8 Tensor Cores", "BF16 / FP8 Gradients", "FP32 Norm Reduction", "FP32 Scalar Scaling", "FP32 Master Weights"],
        "Key Failure Mode": ["Vanishing/Exploding signals", "Out of Memory (OOM)", "Activation overflow", "Gradient explosion", "Premature plateauing", "Second moment bias drift"],
    }
)
st.dataframe(eng_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: FORWARD PROPAGATION IN DEEP TRANSFORMER STACKS
# -----------------------------------------------------------------------------
st.header("2. Forward Propagation in Deep Transformer Stacks", divider="rainbow")

st.markdown(
    """
In an 80-layer foundation model, forward propagation passes token representations through stacked residual connections:

```text
x_{l+1} = x_l + Attention(RMSNorm(x_l))
x_{l+2} = x_{l+1} + MLP(RMSNorm(x_{l+1}))
```

### The Residual Stream Backbone
* The residual connection `x + f(x)` acts as an unimpeded gradient highway.
* Pre-LayerNorm (or Pre-RMSNorm) normalizes inputs **before** entering attention and MLP sub-blocks, ensuring activations remain bounded across all 80 layers and preventing forward variance explosion.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: BACKPROPAGATION & REVERSE ACCUMULATION
# -----------------------------------------------------------------------------
st.header("3. Backpropagation & Reverse Accumulation", divider="rainbow")

st.markdown(
    """
Backpropagation computes the exact partial derivatives of the scalar loss `L` with respect to all parameter matrices `W`:

```text
dL / dW = SUM_{t=1}^{T} (dL / dy_t) * (dy_t / dW)
```

### The Chain Rule Through Residual Streams
Because `x_{l+1} = x_l + f(x_l)`:

```text
dL / dx_l = (dL / dx_{l+1}) * (I + df(x_l) / dx_l) = dL / dx_{l+1} + dL / dx_{l+1} * df / dx_l
```
The identity matrix `I` allows gradients to backpropagate directly from layer 80 to layer 1 without passing through decaying weight matrices, completely eliminating the historical vanishing gradient problem.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: GRADIENT DESCENT DYNAMICS
# -----------------------------------------------------------------------------
st.header("4. Gradient Descent Dynamics in High-Dimensional Loss Landscapes", divider="rainbow")

st.markdown(
    """
In a 70B parameter model, the optimization landscape exists in a 70-billion-dimensional non-convex space.
* **Saddle Points vs. Local Minima**: High-dimensional spaces are populated by millions of saddle points (where gradients vanish in some directions while curving upwards/downwards in others) rather than true local minima.
* **Escaping Saddle Points**: First-order momentum and stochastic noise from batch sampling allow optimizers to roll past saddle points and navigate ill-conditioned ravines.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: MODERN OPTIMIZERS OVERVIEW
# -----------------------------------------------------------------------------
st.header("5. Modern Optimizers Overview (Adaptive vs. Non-Adaptive)", divider="rainbow")

st.markdown(
    """
### Why Standard SGD Fails in LLMs
Standard Stochastic Gradient Descent (SGD) applies a uniform learning rate across all parameters:
`W_{t+1} = W_t - lr * g_t`.
* In transformers, parameters in different layers have wildly different gradient magnitudes. Unembedding matrices and attention projections require fine-grained adaptive step sizes.
* **Adaptive Optimizers (AdamW)** scale updates inversely by the square root of historical gradient variance, ensuring frequent features do not cause explosive steps while sparse features receive sufficient learning signal.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: LEARNING RATE SENSITIVITY
# -----------------------------------------------------------------------------
st.header("6. Learning Rate Sensitivity & The Scale of Maximum Updates", divider="rainbow")

st.markdown(
    """
The learning rate `lr` is the single most sensitive hyperparameter in LLM pretraining:
* **Too High (`> 1e-3`)**: Causes loss spikes, numeric overflow, or permanent divergence.
* **Too Low (`< 5e-5`)**: The model converges slowly, leaving capacity undertrained within the compute budget.
* **Standard Regimes**:
  * 7B Models: `lr approx 3e-4`
  * 70B Models: `lr approx 1.5e-4`
  * 405B Models: `lr approx 8e-5`
  Notice: Larger models require systematically lower learning rates to maintain numerical stability!
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: LEARNING-RATE SCHEDULES
# -----------------------------------------------------------------------------
st.header("7. Learning-Rate Schedules (Warmup, Cosine Annealing, WSD)", divider="rainbow")

st.markdown(
    """
### 1. Linear Warmup
During the first 2,000 steps, AdamW's second moment estimates `v_t` are uninitialized and noisy. Warmup ramps `lr` linearly from `0.0` to `lr_max`, allowing optimizer statistics to stabilize.

### 2. Cosine Annealing
Gradually decays `lr` following a cosine curve down to `0.1 * lr_max`.

### 3. Warmup-Stable-Decay (WSD)
* **Warmup**: First 1-2% of tokens.
* **Stable**: Flat maximum learning rate for 80-90% of tokens.
* **Decay**: Rapid linear cooldown for final 10% of tokens.
* **Strategic Advantage**: In 15T token runs, researchers can evaluate the stable plateau and seamlessly extend training by adding another 5T tokens without having to restart from scratch!
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: WEIGHT INITIALIZATION STRATEGIES
# -----------------------------------------------------------------------------
st.header("8. Weight Initialization Strategies (Xavier, He, MuP / Maximal Update)", divider="rainbow")

st.markdown(
    """
### Classical vs. Modern Transformer Initialization
* **Xavier / Glorot (2010)**: Sets `Var(W) = 2 / (fan_in + fan_out)`. Designed for tanh/sigmoid networks.
* **He / Kaiming (2015)**: Sets `Var(W) = 2 / fan_in`. Designed for ReLU networks.
* **Deep Transformer Residual Scaling**: To prevent activation variance from growing linearly with depth `L`, residual projection weights (`W_out` in attention, `W_down` in MLP) are initialized with standard deviation:
  ```text
  std = (1 / sqrt(2 * L)) * (1 / sqrt(d_model))
  ```

### Maximal Update Parametrization (muP, Yang et al., 2022)
muP decouples parameter learning rates from hidden width `d_model`, ensuring that activations and updates maintain constant norms regardless of whether width is 512 or 16,384.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: GRADIENT CLIPPING MECHANICS
# -----------------------------------------------------------------------------
st.header("9. Gradient Clipping Mechanics & Outlier Taming", divider="rainbow")

st.markdown(
    """
Gradient clipping limits the magnitude of global gradient vectors:

```text
Global_Norm = sqrt( SUM_{p in Params} ||g_p||_2^2 )
if Global_Norm > max_norm:
    g_p = g_p * (max_norm / Global_Norm)
```
Where `max_norm` is typically set to `1.0`.
* Preserves gradient **direction** while capping step magnitude.
* Protects optimizer states from corruption when the model encounters noisy or anomalous web documents.
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE TRAINING STEP ENGINE
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Training Step Engine & MuP Initialization Module", divider="rainbow")

st.markdown(
    """
A production PyTorch module implementing scaled residual initialization and dynamic learning rate scheduling:
"""
)

code_snippet = """import torch
import torch.nn as nn
import math

def initialize_transformer_weights(model: nn.Module, num_layers: int, d_model: int):
    # Scaled residual initialization matching Llama 3 / GPT-3 standards
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            if "out_proj" in name or "down_proj" in name:
                # Scaled residual projection: std = 1 / sqrt(2 * L * d_model)
                std = (1.0 / math.sqrt(2.0 * num_layers)) * (1.0 / math.sqrt(d_model))
                nn.init.normal_(param, mean=0.0, std=std)
            else:
                # Standard input projection: std = 1 / sqrt(d_model)
                std = 1.0 / math.sqrt(d_model)
                nn.init.normal_(param, mean=0.0, std=std)
        elif "bias" in name:
            nn.init.zeros_(param)

def get_wsd_schedule(step: int, total_steps: int, warmup_steps: int = 2000, decay_ratio: float = 0.1):
    # Warmup-Stable-Decay learning rate multiplier
    decay_start = int(total_steps * (1.0 - decay_ratio))
    
    # 1. Warmup
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    # 2. Stable
    elif step < decay_start:
        return 1.0
    # 3. Decay
    else:
        decay_steps = total_steps - decay_start
        progress = float(step - decay_start) / float(max(1, decay_steps))
        return max(0.0, 1.0 - progress)
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE LEARNING RATE & INITIALIZATION WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Learning Rate Schedules & Initialization Workbench", divider="rainbow")

st.markdown(
    """
Compare **Cosine Annealing** vs. **Warmup-Stable-Decay (WSD)** schedules and observe weight standard deviation scaling across network depth.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    total_steps_sim = st.select_slider("Total Training Steps", options=[10000, 50000, 100000, 500000], value=50000)
    warmup_steps_sim = st.slider("Warmup Steps", min_value=500, max_value=5000, value=2000, step=500)
with col_ctrl2:
    wsd_decay_pct = st.slider("WSD Cooldown Ratio (%)", min_value=5, max_value=25, value=10, step=5)
    model_layers_sim = st.slider("Model Depth (Layers)", min_value=12, max_value=80, value=32, step=4)

steps = np.linspace(0, total_steps_sim, 200)

# Simulate schedules
cosine_lrs = []
wsd_lrs = []
decay_start_step = total_steps_sim * (1.0 - wsd_decay_pct / 100.0)

for s in steps:
    # Cosine
    if s < warmup_steps_sim:
        c_lr = s / warmup_steps_sim
    else:
        prog = (s - warmup_steps_sim) / (total_steps_sim - warmup_steps_sim)
        c_lr = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * prog))
    cosine_lrs.append(c_lr)

    # WSD
    if s < warmup_steps_sim:
        w_lr = s / warmup_steps_sim
    elif s < decay_start_step:
        w_lr = 1.0
    else:
        w_lr = max(0.0, 1.0 - (s - decay_start_step) / (total_steps_sim - decay_start_step))
    wsd_lrs.append(w_lr)

sched_df = pd.DataFrame(
    {
        "Step": steps,
        "Cosine Annealing LR": np.round(cosine_lrs, 3),
        "WSD Schedule LR": np.round(wsd_lrs, 3),
    }
)

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("Learning Rate Multiplier Trajectories")
    st.line_chart(sched_df.set_index("Step"))
with col_p2:
    st.subheader("Weight Initialization std across Depth")
    d_model_sim = 4096
    layer_range = np.arange(1, model_layers_sim + 1)
    std_residuals = (1.0 / np.sqrt(2.0 * layer_range)) * (1.0 / np.sqrt(d_model_sim))
    st.line_chart(pd.DataFrame({"Layer Depth": layer_range, "Residual Proj std": std_residuals}).set_index("Layer Depth"))

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why are residual projection weights scaled by 1 / sqrt(2 * L) during initialization?"):
    st.markdown(
        """
        **Answer:**
        In a pre-LN transformer, each layer computes `x_{l+1} = x_l + f(x_l)`.
        If both `x_l` and `f(x_l)` have unit variance, the variance of the residual sum doubles: `Var(x + f(x)) = Var(x) + Var(f(x))`.
        Across `L` layers, the activation variance would grow linearly with depth `L`, causing activations in later layers to explode in magnitude and corrupt FP16 precision.
        Scaling residual weights by `1 / sqrt(2 * L)` ensures that total activation variance remains bounded near 1.0 across the entire network depth.
        """
    )

with st.expander("Q2: What is the Warmup-Stable-Decay (WSD) schedule, and why has it overtaken Cosine Annealing?"):
    st.markdown(
        """
        **Answer:**
        Cosine annealing forces the engineering team to commit to an exact token budget upfront; decaying the learning rate prematurely locks the model into a local minimum.
        WSD maintains a flat maximum learning rate for 80-90% of training. If team goals change, training can be extended indefinitely on additional data tokens without retraining from scratch, only executing the steep 10% cooldown when final evaluation is needed.
        """
    )

with st.expander("Q3: Why do larger models require systematically smaller learning rates?"):
    st.markdown(
        """
        **Answer:**
        As model width (`d_model`) and parameter count increase, the spectral norm of weight matrices and intermediate activation updates scale upwards.
        Using a 7B model learning rate (e.g. `3e-4`) on a 70B model causes large activation update norms that destabilize LayerNorm and trigger loss divergence. To keep update steps invariant, the learning rate must scale inversely with model width (`lr is proportional to 1 / sqrt(d_model)`).
        """
    )

with st.expander("Q4: How does Pre-LayerNorm prevent vanishing gradients compared to Post-LayerNorm?"):
    st.markdown(
        """
        **Answer:**
        In Post-LN (`x_{l+1} = LN(x_l + f(x_l))`), the layer normalization normalizes the entire output, dampening gradients during backpropagation as they pass back through the normalization layer.
        In Pre-LN (`x_{l+1} = x_l + f(LN(x_l))`), the residual stream `x_l` passes directly through an addition operation, preserving an unscaled gradient highway `dL / dx_l = dL / dx_{l+1} + ...` straight back to the input embeddings.
        """
    )

with st.expander("Q5: What is Maximal Update Parametrization (muP)?"):
    st.markdown(
        """
        **Answer:**
        muP is a mathematical scaling framework that scales layer initializations and learning rates according to matrix width dimensions.
        Under standard parametrization, the optimal learning rate shifts as models grow wider, requiring expensive hyperparameter sweeps on full-scale models.
        Under muP, optimal learning rates and initialization scales remain constant regardless of width, allowing 100M proxy models to predict 70B hyperparameter configurations with zero tuning.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Omitting Linear Warmup with AdamW**
        *Root Cause*: Starting training immediately at `lr_max` when AdamW's second moment estimate `v_t` is initialized to zero.
        *Production Fix*: The initial steps make wildly inaccurate updates because `v_t` underestimates gradient variance. Always implement at least 1,000 to 2,000 warmup steps.
        """
    )
    st.warning(
        """
        **Trap 2: Mismatched Gradient Clipping Across Distributed Tensor Slices**
        *Root Cause*: Computing gradient norms locally on individual GPUs before communicating, resulting in unequal clipping factors across ranks.
        *Production Fix*: Always aggregate squared gradient norms across all ranks via an all-reduce operation before evaluating the clipping condition.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Weight Decay Applied to LayerNorm and Biases**
        *Root Cause*: Passing all model parameters into a single optimizer group with `weight_decay = 0.1`.
        *Production Fix*: Penalizing 1D scale/bias parameters pulls normalizations toward zero, degrading representation capacity. Exclude 1D parameters and biases from weight decay.
        """
    )
    st.warning(
        """
        **Trap 4: Setting Too Low a Learning Rate Floor in Cosine Decay**
        *Root Cause*: Decaying `lr` all the way to `0.0` early in training.
        *Production Fix*: Set `min_lr` to `0.1 * lr_max` to ensure the model continues learning through late-stage tokens.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Residual Initialization"):
        st.write("Why scale residual projection weights by 1 / sqrt(2 * L)?")
        st.info("To keep activation variance bounded near 1.0 across deep network layers.")

    with st.expander("⚡ Flashcard 2: Gradient Clipping Threshold"):
        st.write("What is the standard production threshold for gradient clipping?")
        st.info("max_norm = 1.0 (L2 norm).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: WSD Cooldown Timing"):
        st.write("When does the decay phase occur in a Warmup-Stable-Decay schedule?")
        st.info("During the final 10% to 15% of total pretraining tokens.")

    with st.expander("⚡ Flashcard 4: Pre-LN vs Post-LN"):
        st.write("Which normalization layout enables training deep transformers without warmup instability?")
        st.info("Pre-LayerNorm / Pre-RMSNorm.")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand how residual connections provide an unimpeded gradient highway during backpropagation.",
    "I can explain why Pre-RMSNorm is preferred over Post-LayerNorm for training stability.",
    "I understand why residual projections are initialized with 1 / sqrt(2 * L) scaling.",
    "I know how the Warmup-Stable-Decay (WSD) schedule facilitates flexible continual pretraining.",
    "I understand why larger models require systematically lower learning rates.",
    "I can explain the role of gradient clipping in protecting optimizer states from loss spikes.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Neural Network Training Engineering, Initialization, and Optimization Schedules."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
