import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="14.4 Other PEFT Methods - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionAlternativePEFTSuite:
    """
    Unified architectural simulator demonstrating alternative PEFT methods:
    - Houlsby Bottleneck Adapters (Down-projection -> Non-linearity -> Up-projection + Residual)
    - Prompt Tuning (Trainable soft prompt virtual token embeddings)
    - IA^3 (Element-wise learned scaling vectors on K, V, and FFN activations)
    """
    def __init__(self, d_model: int = 64, bottleneck_dim: int = 8, num_virtual_tokens: int = 5, seed: int = 42):
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_virtual_tokens = num_virtual_tokens

        rng = np.random.RandomState(seed)
        # Houlsby Adapter weights
        self.W_down = rng.randn(bottleneck_dim, d_model).astype(np.float32) * 0.1
        self.W_up = rng.randn(d_model, bottleneck_dim).astype(np.float32) * 0.1
        
        # Prompt Tuning Virtual Embeddings: [num_virtual_tokens, d_model]
        self.soft_prompt_embeddings = rng.randn(num_virtual_tokens, d_model).astype(np.float32) * 0.05
        
        # IA^3 learned vectors: [d_model]
        self.ia3_l_k = np.ones(d_model, dtype=np.float32)
        self.ia3_l_v = np.ones(d_model, dtype=np.float32)
        self.ia3_l_ffn = np.ones(d_model, dtype=np.float32)

    def forward_houlsby_adapter(self, x: np.ndarray) -> np.ndarray:
        """
        Bottleneck adapter: h = x + W_up @ GeLU(W_down @ x)
        Sequential execution after transformer layer.
        """
        # Down-projection: [Batch, Seq, bottleneck_dim]
        down = np.dot(x, self.W_down.T)
        # GeLU non-linearity approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        act = 0.5 * down * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (down + 0.044715 * np.power(down, 3))))
        # Up-projection: [Batch, Seq, d_model]
        up = np.dot(act, self.W_up.T)
        # Residual addition
        return x + up

    def forward_prompt_tuning(self, input_embeddings: np.ndarray) -> np.ndarray:
        """
        Prepends trainable continuous virtual tokens to sequence:
        Output shape: [Batch, num_virtual_tokens + Seq_Len, d_model]
        """
        batch_size = input_embeddings.shape[0]
        # Tile soft prompt across batch
        tiled_prompt = np.tile(self.soft_prompt_embeddings[np.newaxis, :, :], (batch_size, 1, 1))
        return np.concatenate([tiled_prompt, input_embeddings], axis=1)

    def forward_ia3(self, k: np.ndarray, v: np.ndarray, ffn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IA^3: Scales inner activations elementwise by learned vectors:
        k_scaled = l_k (hadamard) k
        v_scaled = l_v (hadamard) v
        ffn_scaled = l_ffn (hadamard) ffn
        """
        return k * self.ia3_l_k, v * self.ia3_l_v, ffn * self.ia3_l_ffn

    def get_parameter_comparison(self, base_params: int = 7_000_000_000, num_layers: int = 32) -> pd.DataFrame:
        """Computes trainable parameter overhead across PEFT methods for a 7B model."""
        houlsby_params = num_layers * 2 * (2 * self.d_model * self.bottleneck_dim)
        prefix_params = num_layers * 2 * (self.num_virtual_tokens * self.d_model)
        prompt_params = self.num_virtual_tokens * self.d_model
        ia3_params = num_layers * (3 * self.d_model)
        lora_params = num_layers * 4 * (2 * self.d_model * 8)

        data = [
            {"Method": "LoRA (r=8 on q, v)", "Trainable Params": lora_params, "Pct Base": (lora_params / base_params) * 100, "Latency Overhead": "Zero (Foldable)"},
            {"Method": "Houlsby Adapters", "Trainable Params": houlsby_params, "Pct Base": (houlsby_params / base_params) * 100, "Latency Overhead": "+10-20% (Sequential)"},
            {"Method": "Prefix Tuning", "Trainable Params": prefix_params, "Pct Base": (prefix_params / base_params) * 100, "Latency Overhead": "Minimal (Prefix KV)"},
            {"Method": "Prompt Tuning", "Trainable Params": prompt_params, "Pct Base": (prompt_params / base_params) * 100, "Latency Overhead": "Minimal (Prefix Tokens)"},
            {"Method": "IA^3 (Infused Adapters)", "Trainable Params": ia3_params, "Pct Base": (ia3_params / base_params) * 100, "Latency Overhead": "Zero (Foldable)"}
        ]
        return pd.DataFrame(data)

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("14.4 Other PEFT Methods")
st.caption("Chapter 14: Parameter-Efficient Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
While LoRA and QLoRA command the majority of mindshare in generative AI fine-tuning, 
the broader parameter-efficient landscape includes specialized architectures engineered for distinct deployment constraints.
Techniques such as **Bottleneck Adapters**, **Prefix Tuning**, **Prompt Tuning**, and **IA^3** offer unique advantages 
in multi-task composition, ultra-low-compute adaptation (< 0.01% parameters), and parameter sharing.

In this lesson, you will master:
1. The comparative taxonomy of alternative PEFT families.
2. Bottleneck Adapters: The seminal Houlsby and Pfeiffer architectures.
3. Prefix Tuning: Conditioning every attention layer with virtual key-value tensor prefixes.
4. Prompt Tuning: Learning continuous virtual soft token embeddings at the input layer.
5. IA^3 (Infused Adapter by Inhibiting and Amplifying Inner Activations): Scaling activations with 10x fewer parameters than LoRA.
6. Adapter-based training pipelines, multi-adapter routing, and composability.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Complete Taxonomy of Alternative PEFT Architectures", divider="rainbow")

st.markdown("""
Different PEFT methods modify different structural layers of the Transformer architecture:
input tokens, attention key/values, internal linear activations, or post-layer representations.
""")

alt_peft_dot = """
digraph AltPEFTTaxonomy {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_input {
        label = "Input Layer Injections";
        color = "#334155";
        fontcolor = "#94A3B8";
        PromptTune [label="Prompt Tuning\\n(Lester et al., 2021)\\nPrepends Virtual Soft Tokens", fillcolor="#047857"];
    }

    subgraph cluster_attention {
        label = "Attention Mechanism Injections";
        color = "#334155";
        fontcolor = "#94A3B8";
        PrefixTune [label="Prefix Tuning\\n(Li and Liang, 2021)\\nPrepends P_K and P_V to Every Layer", fillcolor="#0369A1"];
        IA3 [label="IA^3\\n(Liu et al., 2022)\\nLearned Scaling Vectors l_k and l_v", fillcolor="#7C3AED"];
    }

    subgraph cluster_ffn {
        label = "Feed-Forward and Sequential Injections";
        color = "#334155";
        fontcolor = "#94A3B8";
        Houlsby [label="Houlsby Adapters\\n(Houlsby et al., 2019)\\nSequential Bottleneck MLPs", fillcolor="#B45309"];
        Pfeiffer [label="Pfeiffer Adapters\\n(Pfeiffer et al., 2020)\\nSingle Bottleneck per Block", fillcolor="#B45309"];
    }

    PromptTune -> PrefixTune [label="Generalization into all layers"];
    PrefixTune -> IA3 [label="Vector scaling replaces tensors"];
    Houlsby -> Pfeiffer [label="Architectural simplification"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(alt_peft_dot, use_container_width=True)

st.markdown("""
### The Structural Classification of PEFT Methods
1. **Additive Methods (Bottleneck Adapters)**: Insert small neural sub-networks (MLPs) sequentially into the transformer backbone.
2. **Soft Prompt Methods (Prompt & Prefix Tuning)**: Prepend continuous, learnable virtual parameter vectors directly into the sequence context or key-value caches.
3. **Multiplicative Methods (IA^3)**: Multiply internal hidden activation vectors by learned elementwise scale vectors without adding new matrix multiplications.
4. **Reparameterization Methods (LoRA / DoRA)**: Represent weight updates as parallel low-rank matrix products.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - ADAPTERS
# ==============================================================================
st.header("2. Adapters (Bottleneck Adapters)", divider="rainbow")

st.markdown("""
Introduced by Neil Houlsby et al. (Google Research, 2019), **Bottleneck Adapters** were the original pioneer of parameter-efficient deep learning.
Instead of modifying existing weights, an adapter inserts a miniature feed-forward bottleneck module directly into the transformer block.
""")

col_houl1, col_houl2 = st.columns(2)
with col_houl1:
    st.subheader("1. Houlsby Architecture (2019)")
    st.markdown("""
    * **Insertion Points**: Injects **two adapters per transformer block**:
      1. Immediately after multi-head self-attention (before LayerNorm 2).
      2. Immediately after feed-forward MLP (before block output).
    * **Internal Bottleneck**:
      ```text
      Output = x + W_up @ Activation(W_down @ LayerNorm(x))
      ```
    * `W_down`: Projects from `d_model` (e.g. 4096) down to bottleneck `m` (e.g. 64).
    * `W_up`: Projects from `m` back up to `d_model`.
    * **Residual Connection**: Guarantees identity mapping when initialized close to zero.
    """)

with col_houl2:
    st.subheader("2. Pfeiffer Architecture (AdapterHub, 2020)")
    st.markdown("""
    * **Simplification**: Injects **only one adapter per transformer block**, positioned strictly after the Feed-Forward layer and LayerNorm.
    * **Parameter Efficiency**: Cuts adapter parameter count in half compared to Houlsby with zero loss in downstream classification accuracy.
    * **Limitation of Both**: Because adapters execute **sequentially in series**, they cannot be merged into the base weights. 
      They add a **+10% to 20% latency penalty** during inference!
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - PREFIX TUNING
# ==============================================================================
st.header("3. Prefix Tuning", divider="rainbow")

st.markdown("""
Introduced by Xiang Lisa Li and Percy Liang (Stanford, 2021), **Prefix Tuning** freezes the entire language model 
and prepends trainable continuous virtual parameter tensors to the Keys (`K`) and Values (`V`) of **every attention layer**.
""")

st.markdown("""
#### Mathematical Formulation
In standard multi-head attention:
```text
Head_i = Softmax(Q K^T / sqrt(d_k)) V
```
In Prefix Tuning, continuous prefix matrices `P_K in R^(l_p x d_k)` and `P_V in R^(l_p x d_k)` (length `l_p` virtual tokens) are prepended:
```text
K_prefixed = [ P_K ; K_input ]
V_prefixed = [ P_V ; V_input ]
Head_i = Softmax(Q (K_prefixed)^T / sqrt(d_k)) V_prefixed
```
The query tokens `Q` can attend to the virtual prefix parameters as if they were a fixed sequence of conditioning tokens.
""")

col_pref1, col_pref2 = st.columns(2)
with col_pref1:
    st.markdown("#### Reparameterization Trick for Stability")
    st.markdown("""
    * Direct optimization of `P_K` and `P_V` is notoriously unstable, causing loss divergence.
    * Li & Liang introduced an MLP reparameterization: `P = MLP(E_prefix)`.
    * During training, gradients flow through the MLP. 
    * Once training finishes, the MLP is discarded, and only the resulting static tensors `P_K, P_V` are saved!
    """)

with col_pref2:
    st.markdown("#### Tradeoffs")
    st.markdown("""
    * **Pros**: Highly expressive for NLG tasks (table-to-text, summarization); preserves 100% of base model weights.
    * **Cons**: Consumes context window capacity (e.g. 20 to 50 tokens of available context are permanently occupied by the prefix).
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - PROMPT TUNING
# ==============================================================================
st.header("4. Prompt Tuning", divider="rainbow")

st.markdown("""
Introduced by Brian Lester et al. (Google Research, 2021), **Prompt Tuning** is the most minimal soft prompt adaptation technique possible.
Instead of injecting parameters into every layer (like Prefix Tuning), Prompt Tuning injects trainable parameters **only at the input embedding layer**.
""")

col_pt1, col_pt2 = st.columns(2)
with col_pt1:
    st.markdown("#### Input Embedding Injection")
    st.markdown("""
    * A set of `P` virtual "soft tokens" `P_soft in R^(P x d_model)` is prepended to the token embeddings:
      ```text
      Input_Sequence = [ P_soft_1, P_soft_2, ..., P_soft_P, Token_1, Token_2, ..., Token_N ]
      ```
    * All 32+ transformer layers, attention heads, and MLP weights are **100% frozen**.
    * Only `P_soft` is updated via backpropagation (`< 0.01% of base parameters`).
    """)

with col_pt2:
    st.markdown("#### The Scale Asymptotic Phenomenon")
    st.markdown("""
    * On small models (T5-Small: 60M, T5-Base: 220M), Prompt Tuning significantly underperforms full fine-tuning.
    * **The Scale Breakthrough**: As model scale crosses **10 Billion parameters (e.g. T5-XXL 11B)**, Prompt Tuning **matches full fine-tuning accuracy**!
    * Larger foundation models have such rich representations that steering the input embedding space is sufficient for complete task control.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - IA³
# ==============================================================================
st.header("5. IA³ (Infused Adapter by Inhibiting & Amplifying Inner Activations)", divider="rainbow")

st.markdown("""
Introduced by Haokun Liu et al. (2022) in the **T-Few** paper, **IA^3** is an ultra-parameter-efficient method 
that updates **10x fewer parameters than LoRA** while achieving competitive or superior few-shot accuracy.
""")

col_ia1, col_ia2 = st.columns(2)
with col_ia1:
    st.subheader("The Mathematical Mechanism: Elementwise Rescaling")
    st.markdown("""
    Instead of adding matrices, IA^3 learns three elementwise scaling vectors:
    1. **Key Activation Scaling**: `k = l_k (hadamard) (W_k x)`
    2. **Value Activation Scaling**: `v = l_v (hadamard) (W_v x)`
    3. **Position-Wise Feed-Forward Scaling**: `ffn_out = l_ffn (hadamard) GeLU(W_1 x)`
    Where `l_k in R^(d_k)`, `l_v in R^(d_v)`, and `l_ffn in R^(d_ffn)`.
    """)

with col_ia2:
    st.subheader("Why IA³ is an Engineering Marvel")
    st.markdown("""
    * **Ultra-Low Parameter Footprint**: For an 8B model, IA^3 introduces only **~300,000 parameters (< 0.005%)** across all layers!
    * **Zero Inference Latency**: Because scaling is multiplicative, `l_k` can be mathematically folded into the frozen projection weights:
      `W_k_merged = diag(l_k) @ W_k`. Merged models run with **zero latency overhead**!
    * **Extreme Few-Shot Prowess**: Designed specifically for few-shot learning where data is scarce (< 100 samples).
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - ADAPTER-BASED TRAINING
# ==============================================================================
st.header("6. Adapter-Based Training Pipelines & Composability", divider="rainbow")

st.markdown("""
The true production power of modular PEFT architectures is **Composability**: 
combining multiple independently trained adapters into a unified model without retraining.
""")

comp_dot = """
digraph AdapterComposition {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    BaseLLM [label="Shared Frozen Base LLM\\n(Llama-3-8B)", fillcolor="#0369A1"];

    subgraph cluster_adapters {
        label = "Specialized Modular Adapters";
        color = "#334155";
        fontcolor = "#94A3B8";
        AdapterLang [label="Language Adapter\\n(Spanish Medical)", fillcolor="#047857"];
        AdapterTask [label="Task Adapter\\n(Summarization)", fillcolor="#047857"];
        AdapterFormat [label="Format Adapter\\n(JSON Schema)", fillcolor="#047857"];
    }

    Router [label="Adapter Fusion / Router\\n(Dynamic Weighting)", fillcolor="#B45309"];
    Output [label="Composed Output\\nSpanish Medical JSON Summary", fillcolor="#7C3AED"];

    BaseLLM -> Router;
    {AdapterLang AdapterTask AdapterFormat} -> Router;
    Router -> Output;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(comp_dot, use_container_width=True)

st.markdown("""
### Modern Adapter Composition Techniques
1. **AdapterFusion (Pfeiffer et al.)**: Uses an attention router over multiple frozen task adapters to learn dynamic task combinations.
2. **LoRA Weight Averaging / SVD Merging**: Combines multiple domain LoRAs by computing a weighted average: `Delta W_combined = lambda_1 Delta W_1 + lambda_2 Delta W_2`.
3. **MoA (Mixture of Adapters)**: Dynamically routes individual tokens to specialized adapters using a top-k router, similar to Mixture of Experts (MoE).
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Method PEFT Adapter Suite in Python", divider="rainbow")

st.markdown("""
The following self-contained module demonstrates the complete forward pass and parameter mechanics 
of Houlsby Bottleneck Adapters, Prompt Tuning, and IA^3.
""")

st.code("""
import numpy as np
from typing import Tuple

class ProductionAlternativePEFTSuite:
    \"\"\"
    Production-grade educational suite of alternative PEFT methods.
    \"\"\"
    def __init__(self, d_model=64, bottleneck_dim=8, num_virtual_tokens=5):
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_virtual_tokens = num_virtual_tokens

        # Houlsby Adapter matrices
        self.W_down = np.random.randn(bottleneck_dim, d_model).astype(np.float32) * 0.1
        self.W_up = np.random.randn(d_model, bottleneck_dim).astype(np.float32) * 0.1
        
        # Prompt Tuning soft prompt embeddings
        self.soft_prompt = np.random.randn(num_virtual_tokens, d_model).astype(np.float32) * 0.05
        
        # IA^3 scaling vectors
        self.ia3_l_k = np.ones(d_model, dtype=np.float32)

    def forward_houlsby(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Bottleneck: x + W_up @ Activation(W_down @ x)\"\"\"
        down = np.dot(x, self.W_down.T)
        act = np.maximum(0, down)  # ReLU
        up = np.dot(act, self.W_up.T)
        return x + up

    def forward_ia3(self, k: np.ndarray) -> np.ndarray:
        \"\"\"IA^3: Elementwise activation scaling\"\"\"
        return k * self.ia3_l_k
""", language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Alternative PEFT Architecture Comparator & Latency Profiler", divider="rainbow")

suite = ProductionAlternativePEFTSuite(d_model=128, bottleneck_dim=16, num_virtual_tokens=10)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Model & Parameter Sizing")
    base_scale = st.selectbox(
        "Base Model Scale",
        [1.0, 7.0, 13.0, 70.0],
        index=1,
        format_func=lambda x: f"{x}B Parameters"
    )
    b_dim = st.slider("Houlsby Bottleneck Dimension (m)", min_value=4, max_value=64, value=16)
    v_tokens = st.slider("Prompt / Prefix Virtual Token Count", min_value=2, max_value=50, value=10)

    suite.bottleneck_dim = b_dim
    suite.num_virtual_tokens = v_tokens

    comp_df = suite.get_parameter_comparison(base_params=int(base_scale * 1e9), num_layers=32)

with wb_col2:
    st.subheader("Trainable Parameter Footprint Across PEFT Methods")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("##### Parameter Efficiency Visualizer:")
    st.bar_chart(comp_df.set_index("Method")["Pct Base"])
    st.caption("Notice how IA^3 and Prompt Tuning require dramatically fewer parameters than Bottleneck Adapters or LoRA.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why do Bottleneck Adapters add inference latency while LoRA does not?"):
        st.write("""
        Bottleneck Adapters (Houlsby/Pfeiffer) contain non-linear activation functions (e.g. GeLU or ReLU) sandwiched between down-projection and up-projection layers: `W_up @ Act(W_down @ x)`.
        Because non-linear functions are not distributive over matrix addition, adapters cannot be folded or merged into the frozen linear weights. 
        They must execute sequentially in series during every inference forward pass, introducing +10% to 20% latency overhead. 
        LoRA uses strictly linear updates `(alpha/r) * B @ A`, which can be pre-merged into `W_0` before deployment.
        """)
    with st.expander("Q2: What is the primary difference between Prompt Tuning and Prefix Tuning?"):
        st.write("""
        **Prompt Tuning** injects trainable virtual tokens strictly at the **input embedding layer (Layer 0)**; all subsequent transformer layers are completely frozen.
        **Prefix Tuning** prepends trainable virtual Key and Value tensors `P_K, P_V` to **every single transformer attention layer** in the model.
        Prefix Tuning is significantly more expressive and stable on smaller models, while Prompt Tuning only reaches parity on 10B+ scale models.
        """)
    with st.expander("Q3: How does IA^3 achieve parameter efficiency even higher than LoRA?"):
        st.write("""
        LoRA introduces low-rank matrices `B @ A` containing `2 * d * r` parameters per adapted layer.
        IA^3 introduces learned elementwise scaling vectors `l_k, l_v, l_ffn` containing only `d` parameters per adapted layer (no rank multiplication).
        For an 8B model, IA^3 adapts the network with only ~300,000 parameters (10x fewer than LoRA rank 8) while remaining completely foldable into frozen weights.
        """)

with q2:
    with st.expander("Q4: Why does Prompt Tuning perform poorly on smaller models (< 1B)?"):
        st.write("""
        Prompt Tuning only modifies the input representation. 
        In smaller models, internal transformer representations are not sufficiently rich or flexible to translate simple input token steering into complex downstream reasoning.
        At 10B+ parameters, the model's latent manifold is vast, allowing soft prompt embeddings to easily steer pre-existing representations toward the target task.
        """)
    with st.expander("Q5: What is AdapterFusion and how does it prevent catastrophic interference?"):
        st.write("""
        AdapterFusion (Pfeiffer et al., 2021) decouples multi-task learning into two stages:
        1. Train task-specific adapters independently on separate datasets (each frozen after training).
        2. Combine adapters using a learned attention-based routing mechanism (Fusion layer).
        Because base adapters remain completely frozen during fusion training, task capabilities cannot overwrite or interfere with each other.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Directly Optimizing Prefix Tensors Without MLP Reparameterization**
    * **Failure**: Creating raw trainable parameter tensors for `P_K` and `P_V` in Prefix Tuning.
    * **Impact**: Training is exceptionally unstable; gradients explode or oscillate violently, resulting in failure to converge.
    * **Remedy**: Always initialize with a two-layer MLP bottleneck (`P = MLP(Embedding)`), then discard the MLP after training completes.
    """)
    st.error("""
    **🚨 Production Trap 2: Using Prompt Tuning on Short Context Windows**
    * **Failure**: Prepending 100 soft tokens on a model with a 512-token context window.
    * **Impact**: Consumes 20% of available sequence length for prefix tokens alone, truncating user prompts and generation outputs.
    * **Remedy**: Keep soft prompt token count under 10-20 tokens, or use LoRA which consumes zero context window tokens.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Deploying Sequential Adapters in Low-Latency SLA Services**
    * **Failure**: Using Houlsby adapters in a real-time voice or search pipeline with strict <50ms SLAs.
    * **Impact**: The extra non-linear layers break kernel fusion and increase sequential CUDA kernel launch overhead, violating latency budgets.
    * **Remedy**: Use LoRA or IA^3 which fold directly into base weights with 0ms latency penalty.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Combining Adapters with Incompatible Pre-Tokenizers**
    * **Failure**: Merging adapters trained on different tokenizer vocabularies or chat templates.
    * **Impact**: Token ID distributions conflict, producing immediate generation degradation.
    * **Remedy**: Enforce strict base tokenizer homogeneity across all modular adapters in a multi-adapter suite.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Foldable Alternative PEFT"):
        st.write("Which alternative PEFT method can be mathematically folded into base weights for zero latency?")
        st.info("IA^3 (Infused Adapters).")
    with st.expander("⚡ Flashcard 2: Prompt Tuning Scale Threshold"):
        st.write("At what model scale does Prompt Tuning match full fine-tuning performance?")
        st.info("Approximately 10 Billion parameters (e.g. T5-XXL 11B).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Context Window Consumption"):
        st.write("Which PEFT methods consume tokens from the model's context window?")
        st.info("Prompt Tuning and Prefix Tuning (consume virtual prefix tokens).")
    with st.expander("⚡ Flashcard 4: Houlsby vs Pfeiffer Difference"):
        st.write("What is the primary difference between Houlsby and Pfeiffer adapters?")
        st.info("Houlsby has 2 adapters per block (attention + FFN); Pfeiffer has 1 adapter per block (FFN only).")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Alternative PEFT Methods before advancing to Chapter 15 (Prompt Engineering):
""")

chk1 = st.checkbox("I understand why Houlsby adapters introduce sequential latency while LoRA does not.", value=True)
chk2 = st.checkbox("I know the architectural difference between Prompt Tuning (input only) and Prefix Tuning (all layers).", value=True)
chk3 = st.checkbox("I understand how IA^3 scales activations with learned vectors using 10x fewer parameters than LoRA.", value=True)
chk4 = st.checkbox("I know why soft prompts consume available context window capacity.", value=True)
chk5 = st.checkbox("I understand the scale asymptotic phenomenon of Prompt Tuning at 10B+ parameters.", value=True)
chk6 = st.checkbox("I know how AdapterFusion combines modular adapters without catastrophic interference.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! Chapter 14: PEFT is fully completed. You are ready for Chapter 15: Prompt Engineering.")
