"""
9.4 Scaling
===========
Interactive Master Lesson exploring Model Parameters, Dataset Size, Compute Budgets (FLOPs),
Critical Batch Sizes, Sequence Lengths, and Chinchilla Scaling Laws in Foundation LLMs.
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
    page_title="9.4 Scaling - LLM Architecture",
    page_icon="📈",
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
st.title("📈 9.4 Scaling")
st.caption(
    "Chapter 9: LLM Pretraining | Module 4 of 4"
)

st.markdown(
    """
The development of modern Artificial General Intelligence is governed by empirical **Scaling Laws**.
Understanding how model parameters (`N`), training tokens (`D`), and total floating-point operations (`C`)
interact allows engineering teams to accurately forecast model performance, loss trajectories,
and emergent reasoning benchmarks before launching multimillion-dollar cluster training runs.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & COMPUTE SCALING FRONTIER
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Frontier of Compute Scaling", divider="rainbow")

st.markdown(
    """
### The Triad of Model Scaling: Parameters, Data, and Compute
Foundation model performance is strictly predictable as a power-law function of compute budget:
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "scaling_triad",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("compute", "Total Compute Budget (FLOPs)\\nC approx 6 * N * D", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("params", "Model Parameters (N)\\nLayers, Hidden Dim, Heads", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("data", "Training Tokens (D)\\nCorpus Quality, Diversity", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("loss", "Predictable Power-Law Loss\\nL(N, D) = A/N^alpha + B/D^beta + E", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("capabilities", "Downstream Benchmark Emergence\\nMMLU, Code, Reasoning", fillcolor="#1E293B", fontcolor="#F8FAFC")

arch_dot.edge("compute", "params", label="Allocated Params")
arch_dot.edge("compute", "data", label="Allocated Tokens")
arch_dot.edge("params", "loss")
arch_dot.edge("data", "loss")
arch_dot.edge("loss", "capabilities", label="Smooth Token Density")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **From Chinchilla to Inference-Optimal Overtraining**: The Chinchilla scaling law dictates `D = 20 * N` for compute-optimal pretraining. However, modern foundation models (Llama 3 8B trained on 15.6T tokens = ~1,950 tokens/parameter!) intentionally **overtrain by 100x**. While this wastes pretraining FLOPs, it produces radically smaller, cheaper models to serve across billions of inference queries.
* **The 6 * N * D Rule of Thumb**: For standard decoder-only transformers, the total floating-point operations required for pretraining is approximately:
  ```text
  Compute (FLOPs) approx 6 * N * D
  ```
  Where `2 * N * D` is forward pass compute and `4 * N * D` is backward pass compute.
"""
)

# Scaling Milestones Table
scale_df = pd.DataFrame(
    {
        "Model": ["GPT-2", "GPT-3", "Chinchilla", "Llama 1 65B", "Llama 2 70B", "Llama 3 70B", "Llama 3 405B"],
        "Parameters (N)": ["1.5 Billion", "175 Billion", "70 Billion", "65 Billion", "70 Billion", "70 Billion", "405 Billion"],
        "Tokens (D)": ["40 Billion", "300 Billion", "1.4 Trillion", "1.4 Trillion", "2.0 Trillion", "15.6 Trillion", "15.6 Trillion"],
        "Ratio (D / N)": ["26.7x", "1.7x (Undertrained)", "20.0x (Optimal)", "21.5x", "28.5x", "222x (Overtrained)", "38.5x"],
        "Estimated FLOPs": ["3.6e20", "3.1e23", "5.9e23", "5.5e23", "8.4e23", "6.5e24", "3.8e25"],
    }
)
st.dataframe(scale_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: MODEL PARAMETERS ALLOCATION
# -----------------------------------------------------------------------------
st.header("2. Model Parameters Allocation (Width vs. Depth vs. MLP Expansion)", divider="rainbow")

st.markdown(
    """
A model's total non-embedding parameter budget is partitioned across structural dimensions:
* **Layer Depth (`L`)**: Number of stacked transformer blocks.
* **Hidden Width (`d_model`)**: Continuous representation dimension.
* **MLP Expansion (`d_ff`)**: Typically `4 * d_model` in standard ReLU/GELU, or `8/3 * d_model` in SwiGLU.
* **Attention Heads (`H_q`, `H_kv`)**: Number of Query and Key-Value heads.

### Width vs. Depth Scaling
Empirical research indicates that making models excessively deep (>120 layers) slows latency due to sequential layer dependencies. Modern architectures prefer balanced aspect ratios (`d_model / L approx 50 to 100`).
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: DATASET SIZE & UNIQUE TOKEN HORIZONS
# -----------------------------------------------------------------------------
st.header("3. Dataset Size & Unique Token Quality Horizons", divider="rainbow")

st.markdown(
    """
Scaling dataset size `D` requires maintaining high linguistic quality:
* **Unique Token Saturation**: Repeating low-quality web text leads to early validation loss plateauing.
* **Multilingual Expansion**: Expanding from 2T to 15T tokens requires incorporating non-English text (50+ languages) and domain-specific code.
* **Synthetic Infusion**: Adding billions of verified synthetic reasoning tokens prevents catastrophic forgetting during massive training runs.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: TRAINING TOKENS SCALING DYNAMICS
# -----------------------------------------------------------------------------
st.header("4. Training Tokens Scaling Dynamics (Overtraining Frontiers)", divider="rainbow")

st.markdown(
    """
### Why Modern Labs Overtrain Small Models
Consider two models achieving the exact same validation loss:
1. **Model A (Large, Chinchilla-optimal)**: 70 Billion parameters trained on 1.4 Trillion tokens.
2. **Model B (Small, Overtrained)**: 8 Billion parameters trained on 15.6 Trillion tokens.

* **Pretraining Cost**: Model B consumes roughly the same pretraining compute as Model A.
* **Inference Cost**: Serving Model B is **8.7x faster and consumes 8.7x less GPU VRAM** per generated token than Model A!
Because foundation models serve trillions of queries post-launch, overtraining smaller architectures yields massive organizational ROI.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: COMPUTE REQUIREMENTS & FLOPS FORMULATION
# -----------------------------------------------------------------------------
st.header("5. Compute Requirements & Floating-Point FLOPs Formulation", divider="rainbow")

st.markdown(
    """
The total training compute `C` measured in floating-point operations (FLOPs) is derived from matrix operations:

```text
C approx 6 * N * D
```

### Breakdown:
* **Forward Pass**: Each token requires approximately `2 * N` FLOPs (1 multiply + 1 add per parameter weight).
* **Backward Pass**: Requires `4 * N` FLOPs (computing input activations `2 * N` + computing parameter gradients `2 * N`).
* **Total per Token**: `2 * N + 4 * N = 6 * N` FLOPs.

### Hardware Utilization (MFU)
A cluster of `P` GPUs each rated at peak theoretical throughput `T_peak` (e.g. 989 TFLOPS BF16 on H100 SXM) over time `t_sec`:

```text
MFU (Model FLOPs Utilization) = (6 * N * D) / (P * T_peak * t_sec)
```
Frontier training clusters achieve an MFU of **38% to 45%**; the remaining 55% of peak GPU speed is lost to memory bandwidth limits, communications, and bubble pipeline stalls.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: BATCH SIZE DYNAMICS & CRITICAL BATCH SIZE
# -----------------------------------------------------------------------------
st.header("6. Batch Size Dynamics & The Critical Batch Size Limit", divider="rainbow")

st.markdown(
    """
Scaling batch size increases parallelization across GPUs. However, there is a physical upper limit known as the **Critical Batch Size (`B_crit`)**:

```text
B_crit approx B_star / Loss^gamma
```
* **Below B_crit**: Doubling the batch size cuts training time in half without increasing total token consumption (Linear Scaling Regime).
* **Above B_crit**: Doubling the batch size yields diminishing returns. The gradient estimates become redundant, wasting compute (Diminishing Returns Regime).

Frontier LLMs dynamically scale batch sizes from `2M tokens/step` in early training to `16M tokens/step` as loss decreases.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: SEQUENCE LENGTH SCALING & QUADRATIC FRONTIERS
# -----------------------------------------------------------------------------
st.header("7. Sequence Length Scaling & Quadratic Compute Frontiers", divider="rainbow")

st.markdown(
    """
Increasing pretraining sequence length `T` (e.g. from 2,048 to 8,192 tokens) expands the context window but introduces quadratic attention overhead:
* Linear layers and MLPs scale linearly: `O(N * T)`.
* Raw attention scales quadratically: `O(L * H * T^2 * d_k)`.
At `T = 8,192`, attention represents ~15% of total compute. At `T = 128,000`, attention dominates unless mitigated by FlashAttention and Context Parallelism.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: TRAINING STEPS & LEARNING RATE COOLDOWNS
# -----------------------------------------------------------------------------
st.header("8. Training Steps & Learning Rate Cooldown Trajectories", divider="rainbow")

st.markdown(
    """
Pretraining learning rates follow carefully calibrated schedules:
1. **Warmup Phase (First 1-3% of steps)**: Linear ramp from 0 to `lr_max` (e.g. `3e-4`). Prevents early gradient instability.
2. **Cosine Decay (Bulk of training)**: Gradually decays learning rate down to `0.1 * lr_max`.
3. **Cooldown Phase (Final 10-20% of tokens)**: Rapid linear or aggressive cosine decay down to zero. The model absorbs high-quality domain knowledge and sharpens its loss by up to 0.15 nats during cooldown.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: SCALING LAWS (KAPLAN VS. CHINCHILLA)
# -----------------------------------------------------------------------------
st.header("9. Scaling Laws (Kaplan vs. Chinchilla vs. Inference-Optimal)", divider="rainbow")

st.markdown(
    """
The definitive empirical scaling formulation by Hoffmann et al. (Chinchilla, 2022):

```text
L(N, D) = E + A / (N^alpha) + B / (D^beta)
```
Where:
* `E = 1.69` (Irreducible entropy of English).
* `A = 406.4`, `alpha = 0.34` (Parameter scaling exponent).
* `B = 410.7`, `beta = 0.28` (Data scaling exponent).

### Key Takeaways
Because `alpha approx beta`, parameters and tokens contribute almost equally to loss reduction.
Under a constrained compute budget `C`:
```text
N_opt is proportional to C^(beta / (alpha + beta)) approx C^0.45
D_opt is proportional to C^(alpha / (alpha + beta)) approx C^0.55
```
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE SCALING CALCULATOR IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("10. Production-Grade FLOPs & Compute Budget Calculator Implementation", divider="rainbow")

st.markdown(
    """
A complete, standalone Python class for computing exact transformer FLOPs, GPU cluster requirements, and Chinchilla optimal targets:
"""
)

code_snippet = """class LLMScalingCalculator:
    # Production compute and scaling laws calculator for transformer pretraining
    @staticmethod
    def calculate_training_flops(num_params: float, num_tokens: float) -> float:
        # Standard 6 * N * D rule of thumb
        return 6.0 * num_params * num_tokens

    @staticmethod
    def chinchilla_optimal_tokens(num_params: float) -> float:
        # Chinchilla rule: D_opt approx 20 * N
        return 20.0 * num_params

    @staticmethod
    def estimate_cluster_time(
        flops: float,
        num_gpus: int,
        gpu_peak_tflops: float = 989.0,  # NVIDIA H100 SXM FP8/BF16
        mfu: float = 0.40,               # Model FLOPs Utilization
    ) -> Dict[str, float]:
        effective_tflops_per_gpu = gpu_peak_tflops * mfu
        total_cluster_tflops = num_gpus * effective_tflops_per_gpu
        total_cluster_flops_per_sec = total_cluster_tflops * 1e12

        total_seconds = flops / total_cluster_flops_per_sec
        total_hours = total_seconds / 3600.0
        total_days = total_hours / 24.0

        return {
            "total_flops": flops,
            "total_hours": total_hours,
            "total_days": total_days,
            "effective_cluster_tflops": total_cluster_tflops,
        }

    @staticmethod
    def predict_chinchilla_loss(num_params: float, num_tokens: float) -> float:
        # L(N, D) = E + A / N^alpha + B / D^beta
        E = 1.69
        A = 406.4
        B = 410.7
        alpha = 0.34
        beta = 0.28
        return E + (A / (num_params ** alpha)) + (B / (num_tokens ** beta))
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE LLM SCALING LAWS WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive LLM Scaling Laws & GPU Cluster Planner Workbench", divider="rainbow")

st.markdown(
    """
Plan a pretraining run by adjusting parameter scale, token budget, and hardware cluster size.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    model_b = st.select_slider("Model Parameters (Billions)", options=[1, 3, 7, 8, 13, 34, 70, 405], value=8)
    tokens_t = st.select_slider("Training Tokens (Trillions)", options=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0], value=15.0)
with col_ctrl2:
    gpu_type = st.selectbox("Cluster GPU Type", options=["NVIDIA H100 SXM (989 TFLOPS)", "NVIDIA A100 SXM (312 TFLOPS)"], index=0)
    gpu_cluster_size = st.select_slider("Cluster GPU Count", options=[64, 128, 256, 512, 1024, 2048, 4096, 16384], value=1024)

# Compute numbers
params_num = model_b * 1e9
tokens_num = tokens_t * 1e12
total_flops = 6.0 * params_num * tokens_num

peak_tflops = 989.0 if "H100" in gpu_type else 312.0
mfu = 0.40  # 40% efficiency
effective_flops_sec = gpu_cluster_size * (peak_tflops * 1e12) * mfu
total_seconds = total_flops / effective_flops_sec
total_days = total_seconds / (3600 * 24)

# Predict Loss
loss_pred = 1.69 + (406.4 / (params_num ** 0.34)) + (410.7 / (tokens_num ** 0.28))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Compute Budget", f"{total_flops:.2e} FLOPs")
with col_m2:
    st.metric("Estimated Training Days", f"{total_days:.1f} Days")
with col_m3:
    st.metric("Predicted Final Loss", f"{loss_pred:.3f} nats")
with col_m4:
    st.metric("Chinchilla Over-ratio", f"{(tokens_num / (20 * params_num)):.1f}x Optimal")

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.subheader("Cluster Feasibility Blueprint")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Hardware Type", "Total Active GPUs", "Model FLOPs Utilization (MFU)", "Estimated GPU-Hours", "Projected Completion"],
                "Value": [gpu_type, f"{gpu_cluster_size} GPUs", "40.0%", f"{int(total_days * 24 * gpu_cluster_size):,} GPU-hours", f"{total_days:.1f} continuous days"],
            }
        ),
        use_container_width=True,
    )
with col_w2:
    st.subheader("Predicted Loss vs. Token Scale")
    token_scales = np.linspace(1.0, 20.0, 20)
    sim_losses = [1.69 + (406.4 / (params_num ** 0.34)) + (410.7 / ((t * 1e12) ** 0.28)) for t in token_scales]
    st.line_chart(pd.DataFrame({"Tokens (Trillions)": token_scales, "Predicted Loss": sim_losses}).set_index("Tokens (Trillions)"))

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Derive the 6 * N * D formula for transformer pretraining compute."):
    st.markdown(
        """
        **Answer:**
        For each token and each parameter:
        1. **Forward Pass**: 1 multiply-accumulate operation = `2 FLOPs` (1 multiplication + 1 addition). Forward compute = `2 * N * D`.
        2. **Backward Pass**: Computing activation gradients (`2 FLOPs`) and parameter weight gradients (`2 FLOPs`) = `4 FLOPs`. Backward compute = `4 * N * D`.
        Summing forward and backward: `(2 + 4) * N * D = 6 * N * D FLOPs`.
        """
    )

with st.expander("Q2: Why does modern LLM training deliberately violate Chinchilla optimality?"):
    st.markdown(
        """
        **Answer:**
        Chinchilla optimality assumes you want to minimize **pretraining compute only**.
        In industrial deployment, the cost of pretraining is paid once, while the cost of inference is paid billions of times. An 8B model trained on 15T tokens is much cheaper to serve at low latency on edge or single-node GPUs than a Chinchilla-optimal 70B model trained on 1.4T tokens.
        """
    )

with st.expander("Q3: What is Model FLOPs Utilization (MFU) and why is it typically below 50%?"):
    st.markdown(
        """
        **Answer:**
        MFU is the ratio of theoretical matrix arithmetic FLOPs executed to the hardware peak FLOPS capability.
        It rarely exceeds 45% due to:
        1. **Memory-Bandwidth Bound Layers**: LayerNorm, Softmax, and RoPE do not utilize Tensor Cores.
        2. **Distributed Communication Bubbles**: GPUs idle during pipeline-parallel bubbles and gradient all-reduce synchronization.
        3. **Host-to-Device Latencies**: Asynchronous data loading overhead.
        """
    )

with st.expander("Q4: What is the 'Critical Batch Size' and why does it change during pretraining?"):
    st.markdown(
        """
        **Answer:**
        The critical batch size is the maximum batch size where data parallelism yields linear speedup.
        Early in pretraining, the gradient signal-to-noise ratio is low, so smaller batch sizes suffice. As training progresses and the loss landscape flattens, the gradient noise decreases, allowing the critical batch size to expand to millions of tokens without causing optimizer degradation.
        """
    )

with st.expander("Q5: What is the irreducible loss E in Chinchilla scaling laws?"):
    st.markdown(
        """
        **Answer:**
        `E approx 1.69 nats` is the irreducible entropy of human language.
        Even an infinitely large model with infinite compute cannot achieve zero cross-entropy loss because human language has inherent stochastic ambiguity (synonyms, open-ended narrative choices, multiple valid continuations).
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Counting Embedding Parameters in Scaling Law Calculations**
        *Root Cause*: Including the `V * d_model` embedding table when calculating non-embedding compute scaling `N`.
        *Production Fix*: Embeddings only execute sparse lookups per token rather than dense matrix multiplies. Exclude embedding parameters when computing `N` for Chinchilla loss predictions.
        """
    )
    st.warning(
        """
        **Trap 2: Overestimating Theoretical Cluster Throughput**
        *Root Cause*: Assuming 100% of GPU peak TFLOPS without accounting for Model FLOPs Utilization (MFU).
        *Production Fix*: Always budget training cluster runtimes using realistic MFU estimates between 35% and 42%.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Under-Allocating Training Cluster Storage Bandwidth**
        *Root Cause*: Saving multi-terabyte checkpoints from 2,048 GPUs concurrently onto standard NFS storage, locking the cluster for hours.
        *Production Fix*: Deploy high-performance parallel file systems (Lustre / GPFS) with dedicated InfiniBand storage fabric.
        """
    )
    st.warning(
        """
        **Trap 4: Premature Learning Rate Cooldown**
        *Root Cause*: Starting the learning rate cooldown schedule too early, trapping the model in a local minimum before seeing the full token budget.
        *Production Fix*: Restrict the steep cooldown decay phase to the final 10% to 15% of total pretraining tokens.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: 6 * N * D Rule"):
        st.write("What does 6 * N * D calculate in LLM pretraining?")
        st.info("The total floating-point operations (FLOPs) required to train an N-parameter model on D tokens.")

    with st.expander("⚡ Flashcard 2: Typical MFU Target"):
        st.write("What is a world-class Model FLOPs Utilization (MFU) for LLM pretraining clusters?")
        st.info("Around 38% to 45% on NVIDIA H100 clusters.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Chinchilla Parameter Exponent"):
        st.write("What is the power-law parameter exponent alpha in Chinchilla scaling?")
        st.info("alpha approx 0.34 (scaling parameters scales loss as N^(-0.34)).")

    with st.expander("⚡ Flashcard 4: Inference-Optimal Motivation"):
        st.write("Why train small models on 15T tokens?")
        st.info("To minimize downstream inference serving cost and latency while matching larger model accuracy.")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the mathematical derivation of the 6 * N * D pretraining FLOPs rule.",
    "I can explain why modern models overtrain small architectures by 100x beyond Chinchilla optimality.",
    "I understand Model FLOPs Utilization (MFU) and why hardware efficiency peaks around 40-45%.",
    "I know how the Critical Batch Size governs the transition between linear speedup and diminishing returns.",
    "I understand the role of learning rate cooldowns in unlocking downstream reasoning performance.",
    "I can calculate cluster GPU count, runtime days, and storage bandwidth for multi-trillion token runs.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered LLM Scaling Laws, Compute FLOPs, and GPU Cluster Planning."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
