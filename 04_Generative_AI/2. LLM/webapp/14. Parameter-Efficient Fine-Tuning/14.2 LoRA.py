import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="14.2 LoRA - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionLoRALayer:
    """
    Self-contained educational implementation of a LoRA (Low-Rank Adaptation) Linear Layer.
    Demonstrates:
    - Base weight freezing (W_0)
    - Low-rank decomposition matrices (A and B)
    - Strict initialization: A ~ Gaussian, B = 0
    - Scaling factor: alpha / r
    - Mathematical weight merging: W_merged = W_0 + (alpha / r) * (B @ A)
    """
    def __init__(
        self,
        in_features: int = 16,
        out_features: int = 16,
        r: int = 4,
        lora_alpha: float = 8.0,
        seed: int = 42
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        rng = np.random.RandomState(seed)
        # Frozen base weight W_0: [out_features, in_features]
        self.W_0 = rng.randn(out_features, in_features).astype(np.float32) * 0.1
        
        # Trainable Adapter Matrix A: [r, in_features] ~ Gaussian
        self.A = rng.randn(r, in_features).astype(np.float32) * (1.0 / np.sqrt(in_features))
        
        # Trainable Adapter Matrix B: [out_features, r] = Zeros
        self.B = np.zeros((out_features, r), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: h = x @ W_0^T + (alpha / r) * (x @ A^T @ B^T)
        Supports input tensor shape: [Seq_Len, in_features]
        """
        base_out = np.dot(x, self.W_0.T)
        adapter_out = np.dot(np.dot(x, self.A.T), self.B.T) * self.scaling
        return base_out + adapter_out

    def merge_weights(self) -> np.ndarray:
        """
        Merges adapter weights into the base weight:
        W_merged = W_0 + (alpha / r) * (B @ A)
        """
        delta_W = np.dot(self.B, self.A) * self.scaling
        return self.W_0 + delta_W

    def parameter_counts(self) -> Dict[str, int]:
        """Returns parameter accounting for base vs adapter."""
        base_params = self.out_features * self.in_features
        adapter_params = (self.in_features * self.r) + (self.out_features * self.r)
        return {
            "base_params": base_params,
            "adapter_params": adapter_params,
            "total_params": base_params + adapter_params
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("14.2 LoRA (Low-Rank Adaptation)")
st.caption("Chapter 14: Parameter-Efficient Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
Introduced by Edward Hu et al. (Microsoft, 2021), **LoRA (Low-Rank Adaptation)** is the definitive standard 
for modern parameter-efficient fine-tuning. 
By hypothesizing that weight updates during downstream adaptation possess a low **intrinsic dimension**, 
LoRA freezes the multi-billion-parameter base weights and injects trainable rank-decomposition matrices into linear projections.

In this lesson, you will master:
1. The mathematical formulation of Low-Rank Adaptation and rank bottlenecking.
2. Low-rank matrix decomposition: transforming large weight updates into compact factored matrices.
3. Rank (`r`): balancing adaptation expressiveness, memory footprint, and overfitting.
4. Alpha (`alpha`): constant scaling dynamics and learning rate stabilization.
5. LoRA target modules: attention projections (`q, k, v, o`) vs. feed-forward projections (`gate, up, down`).
6. Training dynamics: asymmetric Gaussian/zero initialization and gradient routing.
7. LoRA weight merging: zero-latency inference deployments via `merge_and_unload()`.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Mathematical Decomposition and Forward Injection of LoRA", divider="rainbow")

st.markdown("""
LoRA operates by adding a parallel bypass branch to transformer linear layers, decomposing weight delta `Delta W` into two small matrices.
""")

lora_dot = """
digraph LoRABlueprint {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    Input [label="Input Vector x\\n[Batch, Seq_Len, d_in]", fillcolor="#0F172A"];

    subgraph cluster_base {
        label = "Frozen Base Pathway (requires_grad = False)";
        color = "#334155";
        fontcolor = "#94A3B8";
        W0 [label="Base Weight W_0\\n[d_out, d_in]\\n(Immutable Pretrained Knowledge)", fillcolor="#0369A1"];
        BaseOut [label="W_0 @ x\\n[Batch, Seq_Len, d_out]", fillcolor="#1E293B"];
        W0 -> BaseOut;
    }

    subgraph cluster_lora {
        label = "Trainable LoRA Adapter Pathway (requires_grad = True)";
        color = "#334155";
        fontcolor = "#94A3B8";
        MatrixA [label="Down-Projection A\\n[r, d_in]\\nInitialized ~ Gaussian", fillcolor="#047857"];
        Bottleneck [label="Bottleneck State\\n[Batch, Seq_Len, r]", fillcolor="#1E293B"];
        MatrixB [label="Up-Projection B\\n[d_out, r]\\nInitialized = 0", fillcolor="#047857"];
        Scale [label="Scaling Factor\\n* (alpha / r)", fillcolor="#B45309"];
        
        MatrixA -> Bottleneck -> MatrixB -> Scale;
    }

    AddNode [label="Elementwise Addition (+)\\nh = W_0 x + (alpha / r) B A x", fillcolor="#7C3AED"];
    Output [label="Output Hidden State h\\n[Batch, Seq_Len, d_out]", fillcolor="#0F172A"];

    Input -> W0;
    Input -> MatrixA;
    BaseOut -> AddNode;
    Scale -> AddNode;
    AddNode -> Output;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(lora_dot, use_container_width=True)

st.markdown("""
### The Core Forward Equation
```text
h = W_0 @ x + Delta W @ x = W_0 @ x + (alpha / r) * B @ (A @ x)
```
* **Base Linear Layer `W_0`**: Frozen matrix of shape `[d_out, d_in]`.
* **Adapter Matrix `A`**: Down-projects input from `d_in` to rank `r`.
* **Adapter Matrix `B`**: Up-projects activations from `r` to `d_out`.
* **Zero Initial Perturbation**: Because `B` is initialized to all zeros at step 0, `Delta W = 0`, ensuring training starts exactly at the base model's loss baseline.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - LOW-RANK ADAPTATION
# ==============================================================================
st.header("2. Low-Rank Adaptation", divider="rainbow")

st.markdown("""
Traditional fine-tuning modifies the dense parameter matrix `W` by directly updating all `d_out x d_in` elements:
```text
W = W_0 + Delta W
```
When `d_in = 4096` and `d_out = 4096`, `Delta W` contains **16,777,216 parameters** for a single layer. 
Across 32 layers, updating all weights consumes dozens of gigabytes of optimizer memory.

**Low-Rank Adaptation** constrains the update matrix `Delta W` to have low rank `r`, where `r << min(d_in, d_out)`.
Instead of storing the full dense matrix, `Delta W` is factorized as the product of two rank-`r` matrices:
```text
Delta W = B @ A,   where B in R^(d_out x r) and A in R^(r x d_in)
```
""")

col_lra1, col_lra2 = st.columns(2)
with col_lra1:
    st.subheader("Why Does Low-Rank Adaptation Work?")
    st.markdown("""
    * **Pretrained Manifold**: Trillion-token pretraining establishes a rich, high-dimensional conceptual space.
    * **Task Subspaces**: Adapting to a downstream task (such as Python code generation, medical QA, or JSON extraction) does not require reshaping the entire foundation space.
    * **Low Intrinsic Dimension**: The necessary parameter adjustments reside on a low-dimensional tangent manifold, perfectly captured by small rank bottlenecks.
    """)

with col_lra2:
    st.subheader("Key Empirical Proof from Hu et al.")
    st.markdown("""
    * Fine-tuning RoBERTa, DeBERTa, and GPT-3 (175B) with LoRA rank `r = 4` or `r = 8` achieved validation performance **matching or exceeding full fine-tuning**.
    * Trainable parameters dropped from 175 Billion to **under 35 Million** (> 99.98% parameter reduction).
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - LOW-RANK MATRICES
# ==============================================================================
st.header("3. Low-Rank Matrices", divider="rainbow")

st.markdown("""
The mathematical foundation of LoRA rests upon matrix factorization and rank decomposition.
""")

st.markdown("""
#### Matrix Dimensions and Parameter Compression
Consider a standard Llama-3-8B attention projection where `d_in = 4096` and `d_out = 4096`:
* **Full Update Matrix `Delta W`**: Shape `[4096, 4096]` = **16,777,216 parameters** (`~33.5 MB` in BF16).
* **LoRA Matrix A**: Shape `[r, 4096]`. With `r = 8`: `8 * 4096` = **32,768 parameters**.
* **LoRA Matrix B**: Shape `[4096, r]`. With `r = 8`: `4096 * 8` = **32,768 parameters**.
* **Combined LoRA Parameters**: `32,768 + 32,768` = **65,536 parameters** (`~131 KB` in BF16).
""")

st.success("""
⚡ **Compression Ratio**: LoRA with rank `r = 8` reduces the parameter footprint of this layer by **256x** (from 16.7M to 65.5K parameters)!
""")

st.markdown("""
#### The SVD Connection
Any arbitrary matrix `Delta W` can be factored via Singular Value Decomposition (SVD):
`Delta W = U Sigma V^T`. 
When the singular values `sigma_i` drop off exponentially, truncating to the top `r` singular values retains almost all variance.
LoRA essentially learns this low-rank subspace directly through gradient descent.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - RANK
# ==============================================================================
st.header("4. Rank (r)", divider="rainbow")

st.markdown("""
The **Rank `r`** hyperparameter specifies the inner dimension of matrices `A` and `B`. 
It controls the expressiveness, parameter capacity, and memory footprint of the adapter.
""")

rank_df = pd.DataFrame({
    "Rank (r)": [
        "r = 4",
        "r = 8",
        "r = 16",
        "r = 32",
        "r = 64",
        "r = 128+"
    ],
    "Typical Use Case": [
        "Classification, sentiment analysis, simple style formatting",
        "Standard instruction tuning, QA, conversational alignment (Industry Default)",
        "Complex domain adaptation, reasoning tasks, medical QA",
        "Code generation, complex multi-turn logic, multi-step math",
        "Large-scale continued pretraining on new domain corpora",
        "Rarely needed; approaches full fine-tuning with risk of overfitting"
    ],
    "Parameter Share": [
        "~0.05% of base",
        "~0.10% of base",
        "~0.20% of base",
        "~0.40% of base",
        "~0.80% of base",
        "> 1.5% of base"
    ],
    "Overfitting Risk": [
        "Negligible",
        "Very Low",
        "Low",
        "Moderate",
        "High on small datasets",
        "Severe on small datasets"
    ]
})
st.dataframe(rank_df, use_container_width=True, hide_index=True)

st.info("""
**Engineering Rule of Thumb**:
Start with **`r = 8`** or **`r = 16`**. 
Increasing rank beyond 16 yields diminishing returns for standard instruction tuning and increases risk of overfitting on small datasets.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - ALPHA
# ==============================================================================
st.header("5. Alpha (lora_alpha)", divider="rainbow")

st.markdown("""
The **`lora_alpha` (`alpha`)** hyperparameter is a constant scaling factor applied to the adapter output:
```text
Scaling_Factor = alpha / r
```
During the forward pass, the adapter contribution is multiplied by this scaling factor:
```text
Output = W_0 @ x + (alpha / r) * (B @ A @ x)
```
""")

col_alp1, col_alp2 = st.columns(2)
with col_alp1:
    st.subheader("Why Do We Need Alpha?")
    st.markdown("""
    * **Stabilizing Learning Rate Across Ranks**: If you change rank `r` from 8 to 64 without scaling, the magnitude of the adapter activations grows, requiring complete retuning of the optimizer learning rate!
    * **Decoupling Rank from Magnitude**: By dividing by `r`, the expected scale of the gradient update remains constant regardless of the chosen rank.
    """)

with col_alp2:
    st.subheader("Standard Conventions for Setting Alpha")
    st.markdown("""
    * **Convention 1: `alpha = 2 * r` (Most Common)**: If `r = 16`, set `alpha = 32`. Scaling factor is `32 / 16 = 2.0`.
    * **Convention 2: `alpha = r`**: Scaling factor is `1.0`. Requires slightly higher learning rate.
    * **Convention 3: Fixed Alpha (e.g. `alpha = 16`)**: As you sweep rank `r in [4, 8, 16, 32]`, the scaling factor naturally down-weights higher ranks to prevent gradient explosions.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - LORA TARGET MODULES
# ==============================================================================
st.header("6. LoRA Target Modules", divider="rainbow")

st.markdown("""
In a Transformer architecture, linear projections exist in both the **Multi-Head Self-Attention** blocks 
and the **Feed-Forward Network (MLP)** blocks. Which layers should receive LoRA adapters?
""")

st.markdown("""
### Attention vs. All Linear Projections
""")

mod_col1, mod_col2 = st.columns(2)
with mod_col1:
    st.subheader("Strategy A: Attention Projections Only (`q_proj, v_proj`)")
    st.markdown("""
    * **Historical Origin**: The original LoRA paper (Hu et al. 2021) only applied adapters to Query (`W_q`) and Value (`W_v`) projections.
    * **Parameter Efficiency**: Minimal parameters (~0.05% of base).
    * **Limitation**: Modern research (Dettmers et al. QLoRA) showed that attention-only adaptation leaves performance on the table for complex domain tasks.
    """)

with mod_col2:
    st.subheader("Strategy B: All Linear Modules (Modern Production Standard)")
    st.markdown("""
    * **Target Modules**: 
      * Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
      * Feed-Forward (MLP): `gate_proj`, `up_proj`, `down_proj`
    * **Empirical Advantage**: Adapting MLP layers allows the model to learn new domain-specific factual patterns and associations.
    * **Configuration**: In Hugging Face PEFT, set `target_modules="all-linear"`.
    """)

st.code("""
# Modern Hugging Face PEFT LoRA Configuration
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - LORA TRAINING
# ==============================================================================
st.header("7. LoRA Training Dynamics", divider="rainbow")

st.markdown("""
LoRA training involves asymmetric initialization, selective backpropagation, and specialized learning rates.
""")

col_trn1, col_trn2 = st.columns(2)
with col_trn1:
    st.markdown("#### Asymmetric Initialization Mechanics")
    st.markdown("""
    * **Matrix A**: Initialized with zero-mean Gaussian random values: `A ~ N(0, 1 / d_in)`.
    * **Matrix B**: Initialized strictly to **ALL ZEROS**: `B = 0`.
    * **Why Asymmetry is Mandatory**:
      * If both `A` and `B` were random, step 0 would add noise to the base model, spiking initial perplexity!
      * If both `A` and `B` were zero, gradients would vanish at step 0 because `dL / dA = B^T * grad = 0 * grad = 0`.
      * Gaussian `A` and Zero `B` guarantees `Delta W = 0` at step 0 while ensuring immediate non-zero gradients for Matrix `A`!
    """)

with col_trn2:
    st.markdown("#### Optimal Learning Rates for LoRA")
    st.markdown("""
    * In Full Fine-Tuning, learning rates are very small: `1e-5` to `3e-5`.
    * In LoRA, because the base weights are frozen and only a few million parameters are updated, **higher learning rates are required**:
      * **Standard LoRA LR**: **`1e-4` to `3e-4`** (10x higher than Full FT).
      * **Warmup**: 3% to 5% of total steps.
      * **LR Decay**: Cosine decay to `1e-5`.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - LORA MERGING
# ==============================================================================
st.header("8. LoRA Merging & Inference Deployment", divider="rainbow")

st.markdown("""
During training, running two parallel branches (`W_0 x` and `B A x`) introduces a small computational overhead. 
In production inference, however, **LoRA has ZERO latency penalty** because the adapter can be permanently folded into the base weights!
""")

st.markdown("""
### The Weight Merging Equation
Due to the distributive property of matrix multiplication:
```text
h = W_0 @ x + (alpha / r) * B @ (A @ x) = [ W_0 + (alpha / r) * (B @ A) ] @ x = W_merged @ x
```
Before deployment, compute the dense update matrix `Delta W = (alpha / r) * (B @ A)` and add it directly to `W_0`.
The resulting `W_merged` has the exact same dimensions as the original base model, running with **zero additional inference latency and zero memory overhead**!
""")

col_mrg1, col_mrg2 = st.columns(2)
with col_mrg1:
    st.markdown("#### Hugging Face Merge & Unload")
    st.code("""
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Load trained adapter
lora_model = PeftModel.from_pretrained(base_model, "./my_lora_checkpoint")

# Merge adapter weights into base model
merged_model = lora_model.merge_and_unload()

# Save as standalone production model
merged_model.save_pretrained("./production_merged_model", safe_serialization=True)
""", language="python")

with col_mrg2:
    st.markdown("#### Dynamic Multi-LoRA Serving (vLLM / SGLang)")
    st.markdown("""
    When serving multiple distinct tasks concurrently:
    * Do **not** merge weights into the base model.
    * Keep the base model loaded in GPU VRAM.
    * Use serving engines like **vLLM** or **SGLang** with dynamic multi-LoRA support:
      ```python
      # Client specifies adapter in inference request
      outputs = llm.generate(
          prompts, 
          lora_request=LoRARequest("legal_adapter", 1, "/path/to/legal")
      )
      ```
    """)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: End-to-End LoRA Layer Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained class implements a complete LoRA linear layer:
asymmetric initialization, low-rank forward pass with alpha scaling, parameter accounting, and weight merging.
""")

st.code("""
import numpy as np
from typing import Dict

class ProductionLoRALayer:
    \"\"\"
    Production-grade educational LoRA Linear Layer.
    \"\"\"
    def __init__(self, in_features: int = 16, out_features: int = 16, r: int = 4, lora_alpha: float = 8.0, seed: int = 42):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        rng = np.random.RandomState(seed)
        self.W_0 = rng.randn(out_features, in_features).astype(np.float32) * 0.1
        self.A = rng.randn(r, in_features).astype(np.float32) * (1.0 / np.sqrt(in_features))
        self.B = np.zeros((out_features, r), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        base_out = np.dot(x, self.W_0.T)
        adapter_out = np.dot(np.dot(x, self.A.T), self.B.T) * self.scaling
        return base_out + adapter_out

    def merge_weights(self) -> np.ndarray:
        delta_W = np.dot(self.B, self.A) * self.scaling
        return self.W_0 + delta_W
""", language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Low-Rank Matrix Decomposer & Weight Merging Simulator", divider="rainbow")

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("LoRA Layer Dimensions & Hyperparameters")
    dim_in = st.selectbox("Input Dimension (d_in)", [64, 128, 256, 512, 1024, 4096], index=2)
    dim_out = st.selectbox("Output Dimension (d_out)", [64, 128, 256, 512, 1024, 4096], index=2)
    user_r = st.select_slider("Rank (r)", options=[2, 4, 8, 16, 32, 64], value=8)
    user_alpha = st.slider("LoRA Alpha", min_value=2, max_value=64, value=16)

    scaling_val = user_alpha / user_r
    st.info(f"Effective Scaling Factor (alpha / r): **{scaling_val:.2f}**")

    # Compute parameters
    base_params = dim_in * dim_out
    adapter_params = (dim_in * user_r) + (dim_out * user_r)
    reduction = base_params / max(1, adapter_params)

    layer = ProductionLoRALayer(in_features=dim_in, out_features=dim_out, r=user_r, lora_alpha=user_alpha)

with wb_c2:
    st.subheader("Compression & Parameter Savings")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Base Layer Parameters", f"{base_params:,}", "Frozen (0 grad)")
    with col_p2:
        st.metric("LoRA Adapter Parameters", f"{adapter_params:,}", f"{reduction:.1f}x Compression")

    param_df = pd.DataFrame({
        "Component": ["Base Weights W_0", "Adapter Matrix A", "Adapter Matrix B"],
        "Parameters": [base_params, dim_in * user_r, dim_out * user_r]
    })
    st.bar_chart(param_df.set_index("Component"))

    st.markdown("##### Simulated Forward Pass & Merge Verification:")
    sample_x = np.random.randn(2, dim_in).astype(np.float32)
    # Give B small weights to simulate trained state
    layer.B = np.random.randn(dim_out, user_r).astype(np.float32) * 0.05
    
    unmerged_out = layer.forward(sample_x)
    W_merged = layer.merge_weights()
    merged_out = np.dot(sample_x, W_merged.T)

    max_diff = float(np.max(np.abs(unmerged_out - merged_out)))
    if max_diff < 1e-5:
        st.success(f"✅ Merge Invariance Verified! Max numerical difference: {max_diff:.2e} (Exact mathematical equivalence).")
    else:
        st.warning(f"Numerical difference: {max_diff}")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why is Matrix B initialized to zero while Matrix A is initialized with Gaussian noise?"):
        st.write("""
        Matrix B is initialized to all zeros so that at step 0 of training, the adapter update `Delta W = (alpha / r) * B @ A = 0`.
        This guarantees that training begins exactly at the base model's pretrained baseline without perturbing initial outputs.
        Matrix A is initialized with random Gaussian values so that gradients `dL / dA = B^T * dL / dy` can flow immediately once B receives its first non-zero gradient update.
        If both were zero, gradients would vanish at step 0.
        """)
    with st.expander("Q2: Why does LoRA introduce zero latency overhead in production deployment?"):
        st.write("""
        Because matrix multiplication is distributive: `W_0 @ x + Delta W @ x = (W_0 + Delta W) @ x`.
        Before deployment, we can compute `Delta W = (alpha / r) * (B @ A)` and add it directly to `W_0` using `merge_and_unload()`.
        The deployed model is a single dense weight matrix with identical architecture and latency to the original base model.
        """)
    with st.expander("Q3: What is the purpose of the alpha / r scaling factor in LoRA?"):
        st.write("""
        The `alpha / r` scaling factor stabilizes training dynamics when experimenting with different rank values.
        Without this scaling, increasing rank `r` increases the magnitude of activations flowing from the adapter branch, requiring re-tuning the optimizer learning rate.
        By dividing by `r`, the expected magnitude of updates remains constant across rank changes.
        """)

with q2:
    with st.expander("Q4: Which transformer layers should be targeted with LoRA adapters?"):
        st.write("""
        While original LoRA targeted only `q_proj` and `v_proj`, modern empirical practice (QLoRA, Dettmers et al.) proves that targeting **all linear projections** (`q, k, v, o` in attention and `gate, up, down` in MLP) achieves significantly higher accuracy.
        Adapting MLP layers is especially critical for tasks requiring domain factual adaptation.
        """)
    with st.expander("Q5: Why does LoRA require a higher learning rate than Full Fine-Tuning?"):
        st.write("""
        In full fine-tuning, every single parameter is updating, meaning a large learning rate (e.g. `2e-4`) causes massive global parameter shifts that destroy pretrained representations.
        In LoRA, the base model is frozen and only a tiny subset of parameters (< 1%) receives gradient updates. 
        A higher learning rate (`1e-4` to `3e-4`) is necessary to rapidly guide the small adapter matrices to task convergence.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Initializing Both Matrices with Random Weights**
    * **Failure**: Writing custom LoRA code where Matrix B is initialized with Gaussian noise instead of zeros.
    * **Impact**: At step 0, the adapter injects massive random noise into every attention projection, causing initial loss to skyrocket from 2.0 to 12.0+ and destroying early training stability.
    * **Remedy**: Always verify `torch.nn.init.zeros_(self.lora_B.weight)`.
    """)
    st.error("""
    **🚨 Production Trap 2: Training Only q_proj and v_proj for Domain Tasks**
    * **Failure**: Leaving MLP projections un-adapted when fine-tuning on specialized medical or legal corpora.
    * **Impact**: Attention heads learn syntax, but feed-forward layers (which store factual token associations) cannot adapt, leading to high validation perplexity.
    * **Remedy**: Always set `target_modules="all-linear"`.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Double Alpha Scaling During Manual Merging**
    * **Failure**: Computing `W_0 + (alpha / r) * B @ A` when the forward hook or PEFT layer already incorporated scaling internally.
    * **Impact**: Adapter weights are applied with squared scaling (`(alpha/r)^2`), completely overwhelming the base model.
    * **Remedy**: Always rely on official `model.merge_and_unload()` rather than manual matrix math.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Attempting to Merge 16-bit LoRA into 4-bit Base Model**
    * **Failure**: Calling `merge_and_unload()` on a 4-bit quantized QLoRA model.
    * **Impact**: 4-bit quantized weights cannot represent continuous 16-bit float additions without severe truncation noise.
    * **Remedy**: Dequantize base weights to BF16/FP16 before executing weight merging.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: LoRA Forward Equation"):
        st.write("What is the exact forward equation of a LoRA layer?")
        st.info("`h = W_0 @ x + (alpha / r) * B @ (A @ x)`.")
    with st.expander("⚡ Flashcard 2: Matrix B Initialization"):
        st.write("Why is Matrix B initialized to zero?")
        st.info("To ensure `Delta W = 0` at step 0, preserving the base model's exact initial performance.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Typical SFT Rank Range"):
        st.write("What is the standard rank recommendation for instruction fine-tuning?")
        st.info("`r = 8` or `r = 16` with `alpha = 2 * r`.")
    with st.expander("⚡ Flashcard 4: Zero Latency Mechanism"):
        st.write("How does LoRA achieve zero latency overhead at inference time?")
        st.info("By mathematically merging `(alpha / r) * (B @ A)` directly into `W_0` before deployment.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of LoRA before advancing to Lesson 14.3 (QLoRA):
""")

chk1 = st.checkbox("I can derive the mathematical forward pass: h = W_0 x + (alpha / r) B A x.", value=True)
chk2 = st.checkbox("I understand why Matrix A is Gaussian and Matrix B is initialized to zero.", value=True)
chk3 = st.checkbox("I know why alpha / r decouples learning rate tuning from rank adjustments.", value=True)
chk4 = st.checkbox("I understand why targeting all linear modules (attention + MLP) outperforms q/v only.", value=True)
chk5 = st.checkbox("I know how merge_and_unload() eliminates all inference latency overhead.", value=True)
chk6 = st.checkbox("I know why LoRA requires a higher learning rate (1e-4) than full fine-tuning (2e-5).", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 14.3: QLoRA.")
