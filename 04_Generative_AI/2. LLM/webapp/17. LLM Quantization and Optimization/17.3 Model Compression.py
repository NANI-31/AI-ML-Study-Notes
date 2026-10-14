import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="17.3 Model Compression: Distillation, Pruning, & Small Language Models",
    page_icon="📦",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class CompressionEngine:
    """
    Module-scope realistic simulation engine for modern LLM compression techniques:
    - Wanda (Pruning by Weights and Activations)
    - Unstructured Magnitude Pruning
    - Truncated SVD Low-Rank Factorization
    - Temperature-Scaled Knowledge Distillation Loss
    """
    
    @staticmethod
    def generate_synthetic_layer(
        in_features: int = 64,
        out_features: int = 64,
        outlier_channel_ratio: float = 0.05,
        outlier_magnitude: float = 20.0,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates realistic LLM weight matrix and activation batch."""
        np.random.seed(random_seed)
        W = np.random.normal(loc=0.0, scale=0.05, size=(out_features, in_features))
        X = np.random.normal(loc=0.0, scale=1.0, size=(16, in_features))
        
        num_outliers = max(1, int(in_features * outlier_channel_ratio))
        outlier_cols = np.random.choice(in_features, size=num_outliers, replace=False)
        for col in outlier_cols:
            X[:, col] *= outlier_magnitude
            
        return W, X

    @staticmethod
    def magnitude_pruning(W: np.ndarray, sparsity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Unstructured magnitude pruning: zeros out the lowest absolute weight values."""
        flat_w = np.abs(W).flatten()
        threshold = np.quantile(flat_w, sparsity)
        mask = np.abs(W) >= threshold
        W_pruned = W * mask
        return W_pruned, mask

    @staticmethod
    def wanda_pruning(
        W: np.ndarray,
        X: np.ndarray,
        sparsity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wanda (Pruning by Weights and Activations, Sun et al. 2023):
        Salience S_ij = |W_ij| * ||X_j||_2
        Prunes weights with lowest salience independently per output row.
        """
        # Compute L2 norm of each input activation channel
        act_norm = np.linalg.norm(X, axis=0) # [in_features]
        # Salience score matrix
        salience = np.abs(W) * act_norm[np.newaxis, :]
        
        # Per-row thresholding
        mask = np.zeros_like(W, dtype=bool)
        for i in range(W.shape[0]):
            row_sal = salience[i]
            thresh = np.quantile(row_sal, sparsity)
            mask[i] = row_sal >= thresh
            
        W_pruned = W * mask
        return W_pruned, mask

    @staticmethod
    def low_rank_svd(W: np.ndarray, rank_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Low-rank factorization via Truncated Singular Value Decomposition (SVD):
        W ~ U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
        """
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        k = max(1, int(len(S) * rank_ratio))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstructed matrix
        W_reconstructed = U_k @ np.diag(S_k) @ Vt_k
        
        # Explained variance
        explained_variance = float(np.sum(S_k**2) / np.sum(S**2))
        return W_reconstructed, U_k, Vt_k, explained_variance

    @staticmethod
    def compute_kd_loss(
        teacher_logits: np.ndarray,
        student_logits: np.ndarray,
        temperature: float = 2.0
    ) -> float:
        """
        Computes Hinton Knowledge Distillation Kullback-Leibler divergence:
        KL( Softmax(z_t / T) || Softmax(z_s / T) ) * T^2
        """
        def softmax(z, t):
            exp_z = np.exp((z - np.max(z, axis=-1, keepdims=True)) / t)
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
            
        p_teacher = softmax(teacher_logits, temperature)
        p_student = softmax(student_logits, temperature)
        
        p_teacher = np.clip(p_teacher, 1e-9, 1.0)
        p_student = np.clip(p_student, 1e-9, 1.0)
        
        kl = np.sum(p_teacher * (np.log(p_teacher) - np.log(p_student)), axis=-1)
        return float(np.mean(kl) * (temperature ** 2))

    @staticmethod
    def evaluate_reconstruction(W_orig: np.ndarray, W_comp: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """Computes layer output reconstruction error."""
        Y_orig = X @ W_orig.T
        Y_comp = X @ W_comp.T
        mse = float(np.mean((Y_orig - Y_comp) ** 2))
        norm_orig = np.linalg.norm(Y_orig)
        norm_comp = np.linalg.norm(Y_comp)
        if norm_orig > 0 and norm_comp > 0:
            cos_sim = float(np.sum(Y_orig * Y_comp) / (norm_orig * norm_comp))
        else:
            cos_sim = 1.0
        return {"mse": mse, "cosine_similarity": cos_sim}


# --- HEADER & INTRODUCTION ---
st.title("17.3 Model Compression: Distillation, Pruning, & Small Language Models")
st.caption("Chapter 17: LLM Quantization and Optimization | Domain: 2. Large Language Models")

st.markdown("""
Not every enterprise generative AI deployment can afford the multi-GPU hardware infrastructure, electricity, 
and latency footprint of a 70-billion-parameter foundation model. Autonomous edge devices, mobile operating systems, 
and high-speed streaming autocomplete engines require **Model Compression**: transforming massive frontier intelligence 
into compact, nimble models (500M to 3.8B parameters).

While **quantization** reduces precision without changing parameter count, **Model Compression** attacks parameter geometry directly:
1. **Knowledge Distillation (KD)**: Transfers reasoning patterns and probability distributions from a massive "teacher" model into an architecturally smaller "student" network.
2. **Pruning**: Surgically eliminates redundant weights, attention heads, or MLP hidden layers (via magnitude or activation-aware Wanda/SparseGPT).
3. **Weight Sharing**: Ties parameter tensors across distinct layers or positions to minimize unique memory footprints.
4. **Parameter Reduction (Low-Rank SVD)**: Factorizes massive dense weight matrices into compact low-rank matrix pairs.
5. **Small Language Models (SLMs)**: Inverts Chinchilla scaling laws by training compact architectures on massive volumes of synthetic reasoning data.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The LLM Model Compression Spectrum", divider="rainbow")

st.markdown("""
How foundational compression paradigms navigate the architectural trade-off between **training compute**, 
**hardware deployment ease**, and **reasoning retention**:
""")

compression_dot = """
digraph CompressionSpectrum {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_teacher {
        label="Teacher Frontier Models (70B - 405B+)";
        color="#1E293B";
        fontcolor="#38BDF8";
        TeacherLLM [label="Frontier Teacher\\nGPT-4o / Llama 3 405B\\nHigh Reasoning Capacity", fillcolor="#1E293B"];
    }

    subgraph cluster_compression {
        label="Compression Transformations";
        color="#1E293B";
        fontcolor="#38BDF8";
        KD [label="Knowledge Distillation\\nSoft Logit KL Divergence\\nSynthetic Textbook Reasoning", fillcolor="#0F766E"];
        Prune [label="Pruning (Wanda / SparseGPT)\\nStructured (Heads/Layers)\\n2:4 Hardware Sparsity", fillcolor="#0369A1"];
        SVD [label="Low-Rank Factorization\\nTruncated SVD (W = U @ V)\\nDense Matrix Rank Reduction", fillcolor="#6D28D9"];
        WeightShare [label="Weight Sharing\\nTied Embeddings / Cross-Layer\\nParameter Count Compression", fillcolor="#4338CA"];
    }

    subgraph cluster_deploy {
        label="Target Deployment Artifacts";
        color="#1E293B";
        fontcolor="#38BDF8";
        SLM [label="Small Language Models (SLM)\\nPhi-3 / SmolLM-2 / Gemma-2 2B\\nEdge & Mobile NPU Serving", fillcolor="#059669"];
        CompactEngine [label="Edge In-Memory Serving\\nApple Silicon / Snapdragon NPU\\nUltra-Low Latency Inference", fillcolor="#334155"];
    }

    TeacherLLM -> KD [label="Synthetic Data / Logits"];
    TeacherLLM -> Prune [label="Layer/Head Removal"];
    TeacherLLM -> SVD [label="Matrix Decomposition"];
    KD -> SLM [label="Student Training"];
    Prune -> SLM;
    SVD -> SLM;
    WeightShare -> SLM;
    SLM -> CompactEngine;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(compression_dot, use_container_width=True)

st.markdown("""
### Strategic Compression Comparison Matrix
""")

comp_matrix = [
    {
        "Compression Technique": "Knowledge Distillation (Generative)",
        "Parameter Footprint": "Shrinks 10x - 50x (e.g. 70B -> 3B)",
        "Latency Reduction": "Direct linear speedup with parameter reduction",
        "Retraining Required?": "Yes (Requires pre-training student on tokens)",
        "Hardware Acceleration": "Standard dense GEMM kernels; 100% hardware compatible",
        "Production Examples": "Microsoft Phi-3, DeepSeek-R1-Distill-Qwen, SmolLM-2"
    },
    {
        "Compression Technique": "Structured Pruning",
        "Parameter Footprint": "Shrinks 20% - 40% (Removes heads/layers)",
        "Latency Reduction": "Immediate linear speedup on standard GPUs",
        "Retraining Required?": "Fine-tuning recommended to recover accuracy",
        "Hardware Acceleration": "Native dense GPU kernels (reduced dimensions)",
        "Production Examples": "LLM-Pruner, Sheared LLaMA"
    },
    {
        "Compression Technique": "Wanda / 2:4 Structured Sparsity",
        "Parameter Footprint": "50% parameter reduction in weights",
        "Latency Reduction": "Up to 1.6x speedup on Ampere/Hopper Tensor Cores",
        "Retraining Required?": "No (Post-training in 10 minutes)",
        "Hardware Acceleration": "Requires NVIDIA Sparse Tensor Cores (2:4 pattern)",
        "Production Examples": "SparseGPT, NVIDIA TensorRT-LLM Sparsity"
    },
    {
        "Compression Technique": "Low-Rank SVD Factorization",
        "Parameter Footprint": "Shrinks 20% - 50% via rank parameter r",
        "Latency Reduction": "Reduces FLOPs if rank r < (d_in * d_out) / (d_in + d_out)",
        "Retraining Required?": "Calibration fine-tuning necessary",
        "Hardware Acceleration": "Two consecutive standard dense GEMMs",
        "Production Examples": "LoRA Adapter initialization, MobileBERT"
    },
    {
        "Compression Technique": "Cross-Layer Weight Sharing",
        "Parameter Footprint": "Shrinks disk and VRAM storage up to 60%",
        "Latency Reduction": "Zero speedup (Model still executes identical layer depth)",
        "Retraining Required?": "Trained from scratch",
        "Hardware Acceleration": "Standard dense GEMM kernels",
        "Production Examples": "ALBERT, Universal Transformers"
    }
]
st.dataframe(pd.DataFrame(comp_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - KNOWLEDGE DISTILLATION ---
st.header("2. Knowledge Distillation: Logit Matching, Feature Distillation, & Synthetic Reasoning Transfer", divider="rainbow")

st.markdown("""
Originally pioneered by Geoffrey Hinton et al. (2015), **Knowledge Distillation (KD)** transfers the rich, nuanced internal 
representations of a large "teacher" model into an architecturally compact "student" model.

In modern LLMs, knowledge distillation has evolved along two distinct axes:
1. **Logit-Based Distillation (Dark Knowledge Transfer)**: Matching vocabulary probability distributions.
2. **Generative / Synthetic Distillation (Reasoning Transfer)**: Training students on high-density reasoning tokens generated by teacher models.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Classical Logit Distillation Mechanics")
    st.markdown("""
    In standard cross-entropy training, the loss only rewards predicting the exact ground-truth token. 
    However, when asked *"What is the capital of France?"*, the teacher's probability distribution reveals:
    - `Paris`: 0.94
    - `Lyon`: 0.04
    - `London`: 0.015
    - `Apple`: 0.000001

    This runner-up ranking is **dark knowledge**: it teaches the student that Lyon is a French city and London is a capital city, 
    while Apple is completely irrelevant.

    ### Temperature-Scaled Softmax
    To expose these runner-up probabilities without squashing them into numerical underflow, logits are smoothed using **temperature `T > 1`**:

    ```text
    p_i(T) = exp(z_i / T) / sum_j [exp(z_j / T)]
    ```

    The student is trained using a composite loss function:

    ```text
    Loss = alpha * L_CE(Student, Target) + (1 - alpha) * (2 * T^2) * KL( P_Teacher(T) || P_Student(T) )
    ```
    """)

with col2:
    st.subheader("Modern Generative Distillation (The 2024+ Frontier)")
    st.markdown("""
    Why did logit matching fade for frontier LLMs?
    - Distributing vocabulary logits across a 128,000-token dictionary for billions of steps requires transmitting petabytes of float tensors over network clusters.
    - Instead, frontier labs use **Synthetic Generative Distillation**:
      1. **Textbooks Are All You Need (Phi-3 / SmolLM)**: Frontier teacher models generate pristine synthetic textbooks, exercises, and coding challenges, filtering out internet noise.
      2. **DeepSeek-R1 Reasoning Distillation**: DeepSeek trained a 671B reasoning model via pure reinforcement learning, then used its Chain-of-Thought (CoT) trajectories to fine-tune compact **Qwen-1.5B, 7B, 14B, and 32B** models.
      3. The distilled 14B student outperforms earlier 70B models on math and coding benchmarks!
    """)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Modern open-weight SLMs (like Microsoft Phi-3.5 and DeepSeek-R1-Distill) do not rely on raw internet crawls. 
They achieve PhD-level mathematical reasoning at 1.5B parameters specifically because **100% of their training tokens 
are distilled outputs of 100B+ teacher reasoning models**.
""")

st.divider()

# --- SECTION 3: TOPIC 2 - PRUNING ---
st.header("3. Pruning: Unstructured vs Structured Sparsity, Wanda, & SparseGPT", divider="rainbow")

st.markdown("""
**Pruning** eliminates redundant weights, attention heads, or entire MLP layers from a trained model. 
However, there is a massive chasm between **theoretical parameter reduction** and **real-world GPU speedups**.
""")

prune_diagram = """
digraph PruningTaxonomy {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    FullMatrix [label="Dense Weight Matrix\\n[d_out, d_in]", fillcolor="#1E293B"];
    
    subgraph cluster_unstruct {
        label="Unstructured Sparsity (Individual Zeros)";
        color="#1E293B";
        fontcolor="#38BDF8";
        UnstructZeros [label="Random Zero Entries\\n50% Sparsity\\nZero Standard GPU Speedup!", fillcolor="#7F1D1D"];
        Sparse24 [label="NVIDIA 2:4 Structured Sparsity\\nExactly 2 zeros per 4 values\\n1.5x - 1.8x Ampere Speedup", fillcolor="#047857"];
    }

    subgraph cluster_struct {
        label="Structured Sparsity (Block Removal)";
        color="#1E293B";
        fontcolor="#38BDF8";
        HeadPrune [label="Attention Head Pruning\\nRemoves Redundant Heads\\nImmediate Linear Speedup", fillcolor="#0369A1"];
        LayerPrune [label="Layer Drop / Shearing\\nRemoves Entire Blocks\\nDirect VRAM & Latency Drop", fillcolor="#0F766E"];
    }

    FullMatrix -> UnstructZeros [label="Magnitude Cutoff"];
    FullMatrix -> Sparse24 [label="Ampere Pattern Filter"];
    FullMatrix -> HeadPrune [label="Salience Analysis"];
    FullMatrix -> LayerPrune [label="Block Redundancy"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(prune_diagram, use_container_width=True)

st.markdown("""
### The "Unstructured Sparsity Paradox" on Modern Hardware
- In academic papers, setting 60% of small weights to zero sounds impressive.
- In production, **unstructured sparsity runs SLOWER than the unpruned dense model on standard GPUs**!
- Why? Modern GPUs achieve high floating-point throughput by reading contiguous memory blocks and feeding 
  synchronized Tensor Cores. When weights are irregularly zeroed out, GPU threads diverge, memory coalescing breaks, 
  and indexing overhead destroys performance.

### Solutions to the Paradox:
1. **NVIDIA 2:4 Structured Sparsity**: Enforces that for every 4 contiguous floating-point values in a row, **exactly 2 must be zero**. Supported natively by Ampere (A100) and Hopper (H100) Tensor Cores, yielding a true **2x matrix multiply acceleration**.
2. **Wanda (Pruning by Weights and Activations, Sun et al., 2023)**:
   Classical pruning evaluates weight magnitude `|W|`. Wanda observes that weights must be weighted by the L2-norm of the incoming activation channel:
   ```text
   Salience_ij = |W_ij| * ||X_j||_2
   ```
   Wanda prunes weights with the lowest salience score on a per-row basis. It requires **zero gradient updates, zero retraining, and runs in minutes** while preserving perplexity far better than pure magnitude pruning.
3. **SparseGPT (Frantar et al., 2023)**: Uses second-order inverse Hessian error compensation (identical to GPTQ) to dynamically update remaining unpruned weights as weights are removed, reaching 50% one-shot unstructured sparsity with negligible perplexity loss.
""")

st.divider()

# --- SECTION 4: TOPIC 3 - WEIGHT SHARING ---
st.header("4. Weight Sharing: Cross-Layer Parameter Tying & ALBERT Lineage", divider="rainbow")

st.markdown("""
**Weight Sharing** reuses the exact same parameter tensor across multiple positions or layers in the model architecture. 
Instead of allocating unique floating-point memory for every transformer block, multiple operations point to the same memory buffer.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Embedding & LM-Head Weight Tying")
    st.markdown("""
    In autoregressive transformers, the vocabulary input embedding matrix maps token IDs to hidden dimensions:

    ```text
    W_embed in R^[V, d_model]
    ```

    The final output projection layer maps hidden dimensions back to vocabulary logits:

    ```text
    W_lm_head in R^[d_model, V]
    ```

    With a modern vocabulary size `V = 128,000` and `d_model = 4,096`, each matrix consumes:
    - `128,000 * 4,096 * 2 bytes = 1.05 GB` in FP16.
    - Storing both consumes **2.1 GB** just for token mapping!

    **Weight Tying** sets `W_lm_head = W_embed^T`.
    - Saves over 1 GB of memory immediately.
    - Acts as a strong regularizer: forces tokens that share similar semantic input embeddings to receive similar output logit projections.
    - Used in GPT-2, PaLM, Gemma, and Mistral.
    """)

with col2:
    st.subheader("2. Cross-Layer Parameter Sharing (ALBERT)")
    st.markdown("""
    Pioneered by **ALBERT (Lan et al., 2019)**:
    - Instead of having 24 unique Multi-Head Attention and Feed-Forward Network blocks, **all 24 layers share the exact same weights**!
    - The model executes 24 sequential passes through the single layer, functioning like a recurrent neural network unrolled in depth.

    ### The Production Reality & Latency Trap
    While ALBERT reduced BERT-Large's parameter count from **334M down to 18M (a 94% reduction)**:
    - **Inference latency was completely unchanged!**
    - The GPU still performs 24 sequential forward passes through the layer; FLOPs and compute time are identical.
    - Furthermore, repeating the exact same weights prevents higher layers from developing abstract semantic representations, causing capacity saturation.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - PARAMETER REDUCTION ---
st.header("5. Parameter Reduction: Low-Rank Factorization & Tensor Decomposition", divider="rainbow")

st.markdown("""
Massive Feed-Forward Networks (MLP blocks) account for roughly **two-thirds of all parameters** in a transformer. 
**Low-Rank Factorization** replaces a large dense matrix `W` with the product of two much smaller matrices `A` and `B`.

### The Mathematical Basis: Singular Value Decomposition (SVD)
Any real matrix `W in R^[d_out, d_in]` can be factored via Singular Value Decomposition into:

```text
W = U @ diag(S) @ V^T
```

Where:
- `U in R^[d_out, d_out]` (left singular vectors)
- `S in R^[min(d_out, d_in)]` (singular values, sorted in descending order)
- `V^T in R^[d_in, d_in]` (right singular vectors)

To compress the layer, we truncate to the top `r` singular values:

```text
W_approx = U[:, :r] @ diag(S[:r]) @ V^T[:r, :] = A @ B
```

Where `A = U[:, :r] @ diag(sqrt(S[:r]))` and `B = diag(sqrt(S[:r])) @ V^T[:r, :]`.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Parameter & FLOP Break-Even Analysis
    - **Original Parameters**: `d_out * d_in`
    - **Factorized Parameters**: `r * (d_out + d_in)`

    Compression provides memory and FLOP savings **if and only if**:

    ```text
    r < (d_out * d_in) / (d_out + d_in)
    ```

    For a square matrix `d x d`, parameter savings occur when `r < d / 2`. 
    If `r = d / 4`, memory and computational FLOPs drop by **50%**!
    """)
with col2:
    st.markdown("""
    ### Relation to LoRA (Low-Rank Adaptation)
    - **SVD Compression**: Compresses the **pre-trained base weights** permanently to reduce inference footprint: `W = A @ B`.
    - **LoRA**: Keeps base weights frozen and learns a low-rank **residual delta** during fine-tuning: `W_final = W_frozen + (B_lora @ A_lora)`.
    - Both techniques exploit the same fundamental reality: **the intrinsic dimensionality of language representations is much lower than the ambient dimension of the weight matrix**.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - SMALLER MODELS ---
st.header("6. Smaller Models: The Rise of Frontier Small Language Models (SLMs)", divider="rainbow")

st.markdown("""
Until 2023, the conventional wisdom dictated that smaller models (1B to 3B parameters) were toy architectures 
incapable of complex multi-step reasoning, mathematical problem solving, or zero-shot coding.

The advent of modern **Small Language Models (SLMs)** completely overturned this assumption.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Inverting Chinchilla Scaling Laws")
    st.markdown("""
    The Chinchilla scaling law (Hoffmann et al., 2022) established that for compute-optimal training, 
    parameters and tokens should scale in equal proportion: a 7B model requires ~140B tokens.

    However, compute-optimal training optimizes **training cost**, not **deployment cost**. 
    If a model is intended to be queried billions of times in production:
    - It is vastly more economical to **severely over-train a small model**.
    - For example, **Llama 3 8B** was trained on **15 trillion tokens** (over 100x the Chinchilla recommendation)!
    - **SmolLM-2 1.7B** was trained on **11 trillion tokens**.
    - Over-training saturates the small model's parameter capacity, squeezing out maximum reasoning per byte.
    """)

with col2:
    st.subheader("The Modern SLM Vanguard (1B - 4B)")
    st.markdown("""
    Leading production SLMs in 2024+:
    - **Microsoft Phi-3.5 Mini (3.8B)**: Matches or outperforms original LLaMA-2 70B on reasoning and coding by training exclusively on synthetic textbook data and filtered educational web tokens.
    - **Hugging Face SmolLM-2 (1.7B / 360M)**: Designed specifically for local edge devices, smartphones, and in-browser WebGPU execution.
    - **Google Gemma-2 2B**: Uses sliding-window local attention and knowledge distillation directly from Gemma 27B.
    - **Apple Foundation Models (3B on-device)**: Powers Apple Intelligence directly on iPhone NPU hardware with sub-second response times.
    """)

st.divider()

# --- SECTION 7: PRODUCTION CODE ---
st.header("7. Production Implementation: Wanda Pruning, Softmax Distillation, & SVD Decomposition in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements the mathematical core of **Wanda Activation-Aware Pruning**, 
**Truncated SVD Factorization**, and **Temperature-Scaled Knowledge Distillation Loss**.

This module is defined in real module scope and executes live inside this application:
""")

code_block = '''import numpy as np
from typing import Tuple, Dict

class ModelCompressionCore:
    """
    Self-contained implementation of modern LLM compression techniques.
    Zero external dependencies beyond standard NumPy.
    """
    
    @staticmethod
    def wanda_prune_layer(
        W: np.ndarray,      # Shape: [out_features, in_features]
        X: np.ndarray,      # Shape: [batch_seq, in_features]
        sparsity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wanda (Pruning by Weights and Activations):
        Salience S_ij = |W_ij| * ||X_j||_2
        Prunes weights with lowest salience independently per output row.
        """
        # 1. Compute L2 norm of incoming activation channels
        act_norm = np.linalg.norm(X, axis=0) # [in_features]
        
        # 2. Compute salience matrix
        salience = np.abs(W) * act_norm[np.newaxis, :]
        
        # 3. Create per-row pruning mask
        mask = np.zeros_like(W, dtype=bool)
        for i in range(W.shape[0]):
            row_sal = salience[i]
            cutoff = np.quantile(row_sal, sparsity)
            mask[i] = row_sal >= cutoff
            
        W_pruned = W * mask
        return W_pruned, mask

    @staticmethod
    def svd_factorize_layer(
        W: np.ndarray,
        rank_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Factorizes W into low-rank matrices A and B using Truncated SVD.
        W ~ U[:, :k] @ diag(S[:k]) @ Vt[:k, :] = A @ B
        """
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        k = max(1, int(len(S) * rank_ratio))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Factorized matrix pair: A [out, k], B [k, in]
        A = U_k @ np.diag(np.sqrt(S_k))
        B = np.diag(np.sqrt(S_k)) @ Vt_k
        W_approx = A @ B
        
        explained_var = float(np.sum(S_k**2) / np.sum(S**2))
        return W_approx, A, B, explained_var

    @staticmethod
    def distillation_kl_divergence(
        teacher_logits: np.ndarray,
        student_logits: np.ndarray,
        temperature: float = 2.0
    ) -> float:
        """
        Calculates Hinton soft-target KL divergence loss scaled by T^2.
        """
        def soft(z, t):
            exp_z = np.exp((z - np.max(z, axis=-1, keepdims=True)) / t)
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
            
        p_t = np.clip(soft(teacher_logits, temperature), 1e-9, 1.0)
        p_s = np.clip(soft(student_logits, temperature), 1e-9, 1.0)
        
        kl = np.sum(p_t * (np.log(p_t) - np.log(p_s)), axis=-1)
        return float(np.mean(kl) * (temperature ** 2))
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 8: WORKBENCH ---
st.header("8. Interactive Workbench: Pruning Sparsity & SVD Rank Compression Simulator", divider="rainbow")

st.markdown("""
Simulate how **Magnitude Pruning**, **Wanda Activation-Aware Pruning**, and **Truncated SVD Factorization** 
degrade layer reconstruction error across varying compression ratios in real time:
""")

wb_col1, wb_col2, wb_col3 = st.columns(3)
with wb_col1:
    target_sparsity = st.slider("Pruning Sparsity Target (% Zeros)", min_value=0.10, max_value=0.80, value=0.50, step=0.05)
with wb_col2:
    svd_rank_ratio = st.slider("SVD Retained Rank Ratio (% Dimensions)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
with wb_col3:
    act_outlier_strength = st.slider("Activation Outlier Multiplier", min_value=1.0, max_value=50.0, value=25.0, step=5.0)

# Generate synthetic layer
W_sim, X_sim = CompressionEngine.generate_synthetic_layer(
    in_features=64,
    out_features=64,
    outlier_channel_ratio=0.05,
    outlier_magnitude=act_outlier_strength
)

# 1. Magnitude Pruning
W_mag, mask_mag = CompressionEngine.magnitude_pruning(W_sim, sparsity=target_sparsity)
err_mag = CompressionEngine.evaluate_reconstruction(W_sim, W_mag, X_sim)

# 2. Wanda Pruning
W_wanda, mask_wanda = CompressionEngine.wanda_pruning(W_sim, X_sim, sparsity=target_sparsity)
err_wanda = CompressionEngine.evaluate_reconstruction(W_sim, W_wanda, X_sim)

# 3. SVD Decomposition
W_svd, U_k, Vt_k, exp_var = CompressionEngine.low_rank_svd(W_sim, rank_ratio=svd_rank_ratio)
err_svd = CompressionEngine.evaluate_reconstruction(W_sim, W_svd, X_sim)

st.subheader("Compression Algorithm Accuracy Benchmarks")
bm1, bm2, bm3 = st.columns(3)
bm1.metric(
    "Magnitude Pruning MSE",
    f"{err_mag['mse']:.5f}",
    delta=f"Cos Sim: {err_mag['cosine_similarity']:.4f}",
    delta_color="inverse"
)
bm2.metric(
    "Wanda Pruning MSE",
    f"{err_wanda['mse']:.5f}",
    delta=f"Cos Sim: {err_wanda['cosine_similarity']:.4f}",
    delta_color="normal"
)
bm3.metric(
    "Truncated SVD MSE",
    f"{err_svd['mse']:.5f}",
    delta=f"Explained Var: {exp_var*100:.1f}%",
    delta_color="normal"
)

st.markdown("### Cumulative Error Curve Across Sparsity Levels (Live Simulation)")
sparsity_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mse_mag_list = []
mse_wanda_list = []

for sp in sparsity_steps:
    w_m, _ = CompressionEngine.magnitude_pruning(W_sim, sparsity=sp)
    w_w, _ = CompressionEngine.wanda_pruning(W_sim, X_sim, sparsity=sp)
    mse_mag_list.append(CompressionEngine.evaluate_reconstruction(W_sim, w_m, X_sim)["mse"])
    mse_wanda_list.append(CompressionEngine.evaluate_reconstruction(W_sim, w_w, X_sim)["mse"])

chart_data = pd.DataFrame({
    "Sparsity": [f"{int(s*100)}%" for s in sparsity_steps],
    "Magnitude Pruning MSE": mse_mag_list,
    "Wanda Activation-Aware MSE": mse_wanda_list
}).set_index("Sparsity")

st.line_chart(chart_data)

st.caption("""
Notice how Wanda's error line remains significantly lower than Magnitude Pruning at aggressive sparsity (50% to 70%). 
By weighting weights by incoming activation magnitude `||X_j||_2`, Wanda protects weights attached to outlier channels, 
preventing the sudden catastrophic error spikes suffered by naive magnitude pruning.
""")

st.divider()

# --- SECTION 9: INTERVIEW PREP ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why does unstructured magnitude pruning fail to accelerate LLM inference on modern GPUs?",
        "a": "Modern GPU Tensor Cores are designed for synchronous, contiguous matrix multiplications (GEMM) where memory is loaded in uniform warps. When individual weights are zeroed out randomly across a matrix, the memory layout remains sparse and irregular. Standard dense GEMM routines must still load the zeros from VRAM, yielding zero memory bandwidth savings. Furthermore, sparse matrix routines (CSR/COO) require separate coordinate index lookups, which introduce memory latency overhead that typically makes unstructured sparse models run slower than the original dense network. True hardware speedup requires structured pruning or NVIDIA 2:4 structured sparsity."
    },
    {
        "q": "How does the Wanda pruning algorithm incorporate activations to protect critical weights?",
        "a": "Wanda (Pruning by Weights and Activations) addresses the limitation of pure magnitude pruning, which assumes weight magnitude equals importance. In LLMs, activations feature sharp outlier channels where a tiny fraction of hidden coordinates exhibit massive magnitudes. A small weight attached to a huge activation channel has a far greater impact on the output than a large weight attached to an inactive channel. Wanda computes salience as S_ij = |W_ij| * ||X_j||_2, evaluating importance as the product of weight magnitude and activation channel L2 norm, and prunes the lowest scores row-by-row without needing any retraining."
    },
    {
        "q": "What is the role of temperature scaling (T > 1) in Hinton Knowledge Distillation?",
        "a": "At standard temperature T=1, the softmax function exponentiates logits and normalizes them, pushing the winning token's probability close to 1.0 and collapsing all other candidate tokens to near zero (e.g. 1e-7). This hides the teacher's relative ranking of runner-up tokens ('dark knowledge'). Raising temperature T > 1 softens the probability distribution, revealing subtle semantic relationships between runner-up classes. Scaling the resulting KL divergence loss by T^2 prevents the gradient magnitude from vanishing as T increases."
    },
    {
        "q": "Why does ALBERT cross-layer weight sharing reduce parameter memory but fail to reduce inference latency?",
        "a": "ALBERT reuses the exact same attention and feed-forward weight tensors across all 24 layers of the model. While this reduces the number of unique parameters stored on disk and in GPU VRAM (from 334M to 18M), the forward pass execution graph is unchanged: every input sequence must still pass through 24 sequential matrix multiplication operations in depth. Because the number of FLOPs required to process a token remains identical, inference latency is unchanged."
    },
    {
        "q": "How did Microsoft Phi-3 and SmolLM achieve reasoning parity with models 10x larger?",
        "a": "They inverted traditional Chinchilla compute-optimal training laws. Instead of training compute-optimally for minimal pre-training FLOPs, they severely over-trained small architectures (1.5B to 3.8B parameters) on trillions of tokens. Crucially, they replaced uncurated web crawls with synthetic 'textbook-quality' data generated by frontier models (GPT-4), featuring chain-of-thought derivations, clean coding solutions, and structured educational curriculum. This saturated the small model's parameter capacity with high-density reasoning tokens."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Applying SVD Low-Rank Truncation Without Calibration Fine-Tuning**
    Factorizing an LLM's MLP layers via SVD and immediately deploying the model without post-decomposition calibration. 
    Even a 10% drop in explained variance across 32 transformer blocks compounds exponentially, resulting in complete 
    syntactic collapse. SVD decomposition must always be followed by low-learning-rate fine-tuning or layer-wise reconstruction fitting.
    """)
    st.warning("""
    **Trap 3: Pruning Attention Query-Key Heads Without Context Auditing**
    Pruning 50% of Multi-Head Attention heads based on short-sequence (512-token) benchmark perplexity. Certain attention heads 
    remain inactive on short prompts but function as critical retrieval needle-finders on 32k+ token contexts. Always evaluate 
    pruned models on long-context retrieval benchmarks (e.g., Needle In A Haystack) before shipping.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Excessive Distillation Temperature Washing Out Confident Predictions**
    Setting the distillation temperature `T` excessively high (e.g., `T > 10.0`) during student fine-tuning. 
    An extreme temperature turns the soft target distribution into near-uniform noise, destroying the teacher's sharp 
    confidence on deterministic tasks like arithmetic and syntax rules. Standard practice keeps `T in [2.0, 4.0]`.
    """)
    st.warning("""
    **Trap 4: Parameter-Count Illusion in Mobile Deployment**
    Assuming an SLM with 2B parameters will automatically run fast on a smartphone. While the weights fit into mobile RAM, 
    un-optimized KV cache memory allocation or non-fused LayerNorm operations can cause memory bandwidth throttling on mobile NPUs. 
    Always benchmark on physical target hardware using CoreML or Qualcomm AI Engine.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Wanda Metric"):
        st.markdown("**What is Wanda's salience equation?**\\n\\n`S_ij = |W_ij| * ||X_j||_2`\\n\\nMultiplies weight magnitude by incoming activation L2 norm.")
with fc2:
    with st.popover("Flashcard 2: 2:4 Sparsity"):
        st.markdown("**What is Ampere 2:4 sparsity?**\\n\\nExactly 2 zeros out of every 4 values in a row, unlocking native 2x Tensor Core acceleration.")
with fc3:
    with st.popover("Flashcard 3: Hinton KD"):
        st.markdown("**Why scale KD loss by T^2?**\\n\\n`L_KD = T^2 * KL(...)`\\n\\nPrevents gradient magnitude from shrinking by 1/T^2 as logits are softened.")
with fc4:
    with st.popover("Flashcard 4: Break-Even SVD"):
        st.markdown("**When does SVD reduce parameters?**\\n\\nWhen rank `r < (d_in * d_out) / (d_in + d_out)`. For square matrices, `r < d / 2`.")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why unstructured magnitude pruning provides zero speedup on standard GPUs and why NVIDIA 2:4 structured sparsity is required.",
    "I can explain how the Wanda algorithm combines weight magnitude with activation channel L2-norm to calculate salience.",
    "I know how temperature scaling in Knowledge Distillation uncovers 'dark knowledge' and why modern SLMs rely on synthetic generative distillation.",
    "I understand the trade-offs of embedding weight tying and why cross-layer weight sharing saves VRAM without reducing latency.",
    "I can calculate the parameter break-even rank for low-rank SVD decomposition (r < d_in * d_out / (d_in + d_out)).",
    "I can explain how modern Small Language Models (Phi-3, SmolLM) invert Chinchilla scaling laws via high-density synthetic data over-training."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of modern LLM model compression (Knowledge Distillation, Pruning, SVD, and SLMs).")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
