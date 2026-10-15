import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.1 Inference Fundamentals - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionAutoregressiveEngine:
    """
    Self-contained educational autoregressive inference engine demonstrating:
    - Token generation loop
    - Softmax temperature scaling
    - Greedy vs stochastic sampling
    - Context window tracking
    """
    def __init__(self, vocab_size: int = 64, d_model: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = np.random.RandomState(seed)
        # Simplified linear language model projection
        self.W_embed = self.rng.randn(vocab_size, d_model).astype(np.float32) * 0.1
        self.W_head = self.rng.randn(d_model, vocab_size).astype(np.float32) * 0.1

    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Computes next-token logits for input sequence: [Vocab_Size]."""
        # Embed last token (or mean of tokens for toy forward pass)
        h = np.mean(self.W_embed[token_ids], axis=0)
        logits = np.dot(h, self.W_head)
        return logits

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        greedy: bool = True,
        stop_token: int = 0
    ) -> List[Dict[str, Any]]:
        """Executes autoregressive generation loop recording step-by-step telemetry."""
        generated = list(prompt_tokens)
        steps_telemetry = []

        for step in range(max_new_tokens):
            logits = self.forward(generated)
            
            # Apply temperature
            if temperature > 0 and not greedy:
                scaled_logits = logits / max(1e-5, temperature)
                # Numerically stable softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token = int(self.rng.choice(self.vocab_size, p=probs))
            else:
                next_token = int(np.argmax(logits))

            generated.append(next_token)
            steps_telemetry.append({
                "step": step + 1,
                "input_length": len(generated) - 1,
                "emitted_token": next_token,
                "total_context": len(generated)
            })

            if next_token == stop_token:
                break

        return steps_telemetry

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("16.1 Inference Fundamentals")
st.caption("Chapter 16: LLM Inference and Text Generation | Domain: 2. Large Language Models")

st.markdown("""
Inference is where trained weights meet real-world production users. 
While training optimizes parameter updates via backward passes, **inference is strictly a forward-pass operation** 
repeated iteratively to produce text token-by-token.
Understanding inference fundamentals—including compute characteristics, model loading precision, 
unembedding projections, autoregressive generation loops, and context window limits—is crucial for building low-latency, scalable AI systems.

In this lesson, you will master:
1. The profound differences between Training and Inference across compute, memory, and parallelization.
2. The forward pass mechanics during generation and unembedding projections.
3. Model loading precision: FP16, BF16, and INT8/INT4 weight ingestion into GPU VRAM.
4. Token generation mathematics: from hidden states to probability distributions via Softmax.
5. The Autoregressive Generation Loop: how each generated token becomes the prefix for the next step.
6. Context window limits: how the quadratic complexity of self-attention governs memory and latency.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Autoregressive Inference Lifecycle & Token Loop", divider="rainbow")

st.markdown("""
Inference operates in two distinct phases: the **Prefill Phase** (parallel ingestion of user prompt) 
followed by the **Decode Phase** (iterative sequential token emission).
""")

infer_blueprint_dot = """
digraph InferenceLifecycle {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_prefill {
        label = "Phase 1: Prefill Phase (Compute-Bound)";
        color = "#334155";
        fontcolor = "#94A3B8";
        Prompt [label="User Prompt Tokens\\n[Batch, Seq_Len]", fillcolor="#0F172A"];
        PrefillForward [label="Full Parallel Forward Pass\\nComputes KV for all prompt tokens", fillcolor="#0369A1"];
        InitialLogits [label="Logits for Final Token\\n[Batch, Vocab_Size]", fillcolor="#0369A1"];
        Prompt -> PrefillForward -> InitialLogits;
    }

    subgraph cluster_decode {
        label = "Phase 2: Decode Phase (Memory-Bound Iterative Loop)";
        color = "#334155";
        fontcolor = "#94A3B8";
        SampleToken [label="Sample Next Token\\nGreedy / Top-p / Top-k", fillcolor="#047857"];
        CheckEOS [label="Check Stop Condition\\n(EOS or Max Tokens?)", fillcolor="#B45309"];
        AppendKV [label="Append to KV Cache\\nand Sequence Buffer", fillcolor="#1E293B"];
        SingleForward [label="Single-Token Forward Pass\\n[Batch, 1, Hidden_Dim]", fillcolor="#047857"];
        
        InitialLogits -> SampleToken;
        SampleToken -> CheckEOS;
        CheckEOS -> AppendKV [label="Not Finished"];
        AppendKV -> SingleForward;
        SingleForward -> SampleToken [label="Next Logits"];
    }

    Complete [label="Final Response Stream\\nDelivered to User", fillcolor="#7C3AED"];
    CheckEOS -> Complete [label="Finished (EOS)"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(infer_blueprint_dot, use_container_width=True)

st.markdown("""
### Key Structural Milestones
1. **Prefill (Time-To-First-Token / TTFT)**: Processes all prompt tokens simultaneously using parallel tensor cores. Compute-bound.
2. **Decode (Inter-Token Latency / ITL)**: Generates one token at a time. Memory bandwidth-bound because model weights must be streamed from HBM to SRAM for every single generated token.
3. **Termination**: Generation continues until the model emits an End-Of-Sequence (`<|eos|>`) token or reaches `max_new_tokens`.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - TRAINING VS INFERENCE
# ==============================================================================
st.header("2. Training vs. Inference", divider="rainbow")

st.markdown("""
The operational and architectural requirements of training and inference are fundamentally distinct.
""")

compare_infer_df = pd.DataFrame({
    "Dimension": [
        "Primary Goal",
        "Computation Graph",
        "Memory Footprint",
        "Hardware Bottleneck",
        "Sequence Parallelism",
        "Latency Sensitivity",
        "Batch Sizes"
    ],
    "Training": [
        "Optimize weight parameters via gradient descent",
        "Forward pass + Backward pass (autograd tape)",
        "Weights (2B) + Gradients (2B) + AdamW (12B) + Activations",
        "Compute-bound (FLOPS) on massive tensor cores",
        "Full sequence parallel (all tokens processed at once)",
        "Hours to weeks (throughput matters, latency does not)",
        "Large batches (e.g. 512 - 4,096 sequences)"
    ],
    "Inference": [
        "Generate probability distributions and emit text",
        "Forward pass only (no gradients, torch.no_grad())",
        "Weights (2B) + KV Cache (dynamic memory)",
        "Memory Bandwidth-bound (GB/s memory bus during decode)",
        "Sequential autoregressive loop (one token at a time)",
        "Sub-second SLAs (TTFT < 200ms, ITL < 30ms)",
        "Small dynamic batches (e.g. 1 - 32 concurrent requests)"
    ]
})
st.dataframe(compare_infer_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - FORWARD PASS IN INFERENCE
# ==============================================================================
st.header("3. Forward Pass Mechanics in Inference", divider="rainbow")

st.markdown("""
During inference, a forward pass transforms input token indices into output vocabulary logits without caching intermediate gradients.
""")

col_fp1, col_fp2 = st.columns(2)
with col_fp1:
    st.markdown("#### Forward Pass Tensor Transformations")
    st.markdown("""
    1. **Input IDs**: Tensor of integers `[Batch_Size, Sequence_Length]`.
    2. **Embedding Lookup**: `W_emb[input_ids] -> [Batch_Size, Sequence_Length, Hidden_Dimension]`.
    3. **Transformer Blocks (x N)**:
       * RMSNorm -> Self-Attention (Query, Key, Value) -> Residual Connection.
       * RMSNorm -> Feed-Forward Network (SwiGLU / MLP) -> Residual Connection.
    4. **Final LayerNorm / RMSNorm**: Normalizes hidden states.
    5. **Unembedding Head**: Linear projection to vocabulary size.
    """)

with col_fp2:
    st.markdown("#### Disabling Gradient Tape Overhead")
    st.markdown("""
    In PyTorch, executing inference requires strictly wrapping the forward call in `torch.inference_mode()` or `torch.no_grad()`:
    ```python
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids).logits
    ```
    * **`torch.no_grad()`**: Disables autograd tracking.
    * **`torch.inference_mode()`**: Even faster; completely disables version tracking and view tracking tensors.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - MODEL LOADING
# ==============================================================================
st.header("4. Model Loading & Precision in Production", divider="rainbow")

st.markdown("""
Loading multi-billion-parameter foundation models into GPU VRAM requires strict precision management.
""")

col_ld1, col_ld2 = st.columns(2)
with col_ld1:
    st.subheader("Weight Precision and Memory Footprint")
    st.markdown("""
    * **FP32 (4 bytes/param)**: 7B model requires **28 GB** VRAM. Obsolete for LLM inference.
    * **FP16 (2 bytes/param)**: 7B model requires **14 GB** VRAM. Prone to underflow in deep layers.
    * **BF16 (2 bytes/param)**: 7B model requires **14 GB** VRAM. **Modern standard**. Matches FP32 dynamic range.
    * **INT8 (1 byte/param)**: 7B model requires **7 GB** VRAM. 50% memory reduction with near-zero accuracy loss.
    * **INT4 / NF4 (0.5 bytes/param)**: 7B model requires **3.5 GB** VRAM. Runs on consumer laptop GPUs.
    """)

with col_ld2:
    st.subheader("SafeTensors & Zero-Copy Memory Mapping")
    st.markdown("""
    * Legacy `.bin` (Pickle) checkpoints require serializing tensors through Python objects, taking minutes to load.
    * **SafeTensors** enables zero-copy memory mapping (`mmap`):
      Tensors are mapped directly from NVMe SSDs into GPU memory address spaces in seconds, drastically accelerating container cold-starts.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - TOKEN GENERATION
# ==============================================================================
st.header("5. Token Generation & Unembedding Projection", divider="rainbow")

st.markdown("""
How does an abstract hidden vector `h_last` of shape `[Hidden_Dimension]` become an English word or subword?
Through the **Unembedding Projection (LM Head)** and the **Softmax function**.
""")

st.markdown("""
#### The Mathematical Formulation
```text
Logits = h_last @ W_head^T,   where W_head in R^(Vocab_Size x Hidden_Dimension)
```
The resulting vector `Logits` has length equal to the vocabulary size (e.g. 128,256 tokens in Llama-3).
To convert raw unbounded logits into a normalized probability distribution over tokens:
```text
P(token_i) = exp(Logits_i / T) / Sum_{j=1}^{Vocab_Size} exp(Logits_j / T)
```
Where `T` is the sampling **Temperature**.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - AUTOREGRESSIVE GENERATION
# ==============================================================================
st.header("6. Autoregressive Generation & Loop Dynamics", divider="rainbow")

st.markdown("""
The defining characteristic of modern generative language models is **Autoregression**: 
the model cannot predict sentence tokens in parallel; each token is predicted conditionally on all preceding tokens.
""")

st.markdown("""
#### The Autoregressive Loop Sequence
```text
Iteration 1: Input: ["The", "sky", "is"]       -> Predicts: "blue"
Iteration 2: Input: ["The", "sky", "is", "blue"]  -> Predicts: "because"
Iteration 3: Input: ["The", "sky", "is", "blue", "because"] -> Predicts: "Rayleigh"
Iteration 4: Input: ["The", "sky", "is", "blue", "because", "Rayleigh"] -> Predicts: "scattering"
```
""")

col_ar1, col_ar2 = st.columns(2)
with col_ar1:
    st.markdown("#### The Memory-Bandwidth Wall")
    st.markdown("""
    * At each iteration, the entire 14 GB of model weights (for a 7B model) must be read from High Bandwidth Memory (HBM) into GPU compute cores to produce **a single token**.
    * If an H100 GPU has 3.35 TB/s memory bandwidth:
      ```text
      Max_Tokens_Per_Sec = (3.35 TB/s) / (14 GB / token) = ~240 tokens/sec
      ```
    * Compute cores sit idle waiting for weights to transfer from memory.
    """)

with col_ar2:
    st.markdown("#### The Arithmetic Intensity Problem")
    st.markdown("""
    * **Arithmetic Intensity** = FLOPS / Memory Access Bytes.
    * In training, batching gives high arithmetic intensity (~100+ FLOPs per byte).
    * In single-user decoding, arithmetic intensity is **~1 FLOP per byte**. 
    * Generation is fundamentally bounded by memory speed, not GPU computing power!
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CONTEXT WINDOW
# ==============================================================================
st.header("7. Context Window Constraints & The Quadratic Attention Wall", divider="rainbow")

st.markdown("""
The **Context Window** defines the maximum number of tokens a model can attend to during a single inference call 
(e.g., 8k tokens in Llama-3, 128k in GPT-4o, 1M+ in Gemini 1.5).
""")

col_ctx1, col_ctx2 = st.columns(2)
with col_ctx1:
    st.subheader("Quadratic Attention Scaling: O(N^2)")
    st.markdown("""
    * Standard Self-Attention computes:
      ```text
      Attention_Scores = Softmax(Q @ K^T / sqrt(d_k))
      ```
    * `Q @ K^T` produces an `[N, N]` matrix where `N` is sequence length.
    * Doubling context from 4k to 8k quadruples memory and compute (`4x`).
    * Scaling from 8k to 128k increases attention memory by **256x**!
    """)

with col_ctx2:
    st.subheader("Architectural Mitigations in Modern Models")
    st.markdown("""
    * **FlashAttention-2**: Tiles attention computation in GPU SRAM, avoiding materializing the `[N, N]` matrix in HBM.
    * **Grouped Query Attention (GQA)**: Shares Key/Value heads across Query heads, cutting KV Cache memory by 8x.
    * **RoPE Scaling (YaRN, LongLoRA)**: Extends rotary positional encodings to extrapolate across 100k+ tokens.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Autoregressive Generator in Python", divider="rainbow")

st.markdown("""
The following self-contained class implements the core autoregressive token generation loop:
unembedding projection, temperature scaling, greedy vs stochastic selection, and context expansion.
""")

st.code("""
import numpy as np
from typing import List, Dict, Any

class ProductionAutoregressiveEngine:
    \"\"\"
    Production-grade educational autoregressive inference engine.
    \"\"\"
    def __init__(self, vocab_size: int = 64, d_model: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = np.random.RandomState(seed)
        self.W_embed = self.rng.randn(vocab_size, d_model).astype(np.float32) * 0.1
        self.W_head = self.rng.randn(d_model, vocab_size).astype(np.float32) * 0.1

    def forward(self, token_ids: List[int]) -> np.ndarray:
        h = np.mean(self.W_embed[token_ids], axis=0)
        return np.dot(h, self.W_head)

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 10, temperature: float = 1.0, greedy: bool = True) -> List[int]:
        generated = list(prompt_tokens)
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            if temperature > 0 and not greedy:
                scaled = logits / max(1e-5, temperature)
                probs = np.exp(scaled - np.max(scaled))
                probs /= np.sum(probs)
                next_token = int(self.rng.choice(self.vocab_size, p=probs))
            else:
                next_token = int(np.argmax(logits))
            generated.append(next_token)
        return generated
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Autoregressive Generator & Context Window Simulator", divider="rainbow")

engine = ProductionAutoregressiveEngine(vocab_size=32, d_model=16, seed=101)

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("Generation Hyperparameters")
    prompt_str = st.text_input("Initial Prompt Token IDs (Comma Separated)", value="5, 12, 18, 3")
    try:
        init_tokens = [int(x.strip()) for x in prompt_str.split(",") if x.strip()]
    except Exception:
        init_tokens = [5, 12, 18, 3]

    max_steps = st.slider("Max New Tokens", min_value=1, max_value=25, value=8)
    sample_mode = st.radio("Sampling Strategy", ["Greedy Decoding (Deterministic)", "Temperature Sampling"])
    temp_val = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1) if sample_mode != "Greedy Decoding (Deterministic)" else 0.0

    telemetry = engine.generate(
        init_tokens,
        max_new_tokens=max_steps,
        temperature=temp_val,
        greedy=(sample_mode == "Greedy Decoding (Deterministic)")
    )
    telem_df = pd.DataFrame(telemetry)

with wb_c2:
    st.subheader("Step-by-Step Generation Telemetry")
    st.metric("Final Generated Sequence Length", f"{len(init_tokens) + len(telem_df)} Tokens", f"+{len(telem_df)} generated")
    
    st.markdown("##### Autoregressive Emission Table:")
    st.dataframe(telem_df, use_container_width=True, hide_index=True)
    st.caption("Each row represents one full forward pass where the emitted token is appended to the context buffer.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why is LLM inference memory-bandwidth bound rather than compute-bound during decode?"):
        st.write("""
        During the decode phase, the model generates one token at a time with batch size 1 or small batch sizes.
        To predict that single token, all billions of model parameters must be read from GPU High Bandwidth Memory (HBM) into on-chip cache.
        The arithmetic intensity is ~1 FLOP per byte transferred. 
        Because modern GPU tensor cores can execute hundreds of trillions of FLOPS per second, compute cores sit idle waiting for weights to stream over the memory bus.
        """)
    with st.expander("Q2: What is the difference between Time-To-First-Token (TTFT) and Inter-Token Latency (ITL)?"):
        st.write("""
        **Time-To-First-Token (TTFT)** measures the latency of the Prefill phase: processing all prompt tokens in parallel and generating the first output token. It is compute-bound.
        **Inter-Token Latency (ITL)** measures the time required to generate each subsequent token during the Decode phase. It is memory-bandwidth bound and governs the user's perceived streaming speed.
        """)
    with st.expander("Q3: Why should you use torch.inference_mode() over torch.no_grad()?"):
        st.write("""
        `torch.no_grad()` disables autograd recording but still creates tensors with version counters and view tracking hooks.
        `torch.inference_mode()` completely disables all autograd metadata, view tracking, and version counters, providing a lower Python dispatch overhead and ~5-10% faster forward pass execution.
        """)

with q2:
    with st.expander("Q4: Why does self-attention have quadratic O(N^2) complexity with sequence length?"):
        st.write("""
        In self-attention, every token in the sequence computes an attention dot product against every other token in the sequence (`Q @ K^T`).
        For sequence length N, this produces an `[N, N]` attention score matrix.
        Computing and storing this matrix requires `O(N^2)` FLOPS and `O(N^2)` memory, becoming a severe bottleneck for long contexts.
        """)
    with st.expander("Q5: What is the Unembedding Head and what are tied weights?"):
        st.write("""
        The unembedding head is a final linear projection layer `W_head in R^(Vocab_Size x Hidden_Dim)` that maps final layer hidden states into vocabulary logits.
        In **Weight Tying** (e.g. GPT-2, Gemma), `W_head` shares the exact same weight tensor as the input embedding matrix `W_embed^T`, saving hundreds of millions of parameters.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Leaving model.train() Active During Inference**
    * **Failure**: Running generation without calling `model.eval()`.
    * **Impact**: Dropout layers remain active, randomly zeroing out activations; generation becomes noisy and non-deterministic even with temperature 0.
    * **Remedy**: Always invoke `model.eval()` before calling generate.
    """)
    st.error("""
    **🚨 Production Trap 2: Re-Tokenizing Full Sequences at Every Step Without KV Cache**
    * **Failure**: Passing the full growing text string back to the tokenizer and model at every iteration.
    * **Impact**: Prefill is re-executed from scratch for every token, turning `O(N)` decode into catastrophic `O(N^2)` quadratic latency.
    * **Remedy**: Enable `use_cache=True` to retain intermediate Key-Value activations.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Memory Spikes from Inappropriate Batch Sizes**
    * **Failure**: Setting inference batch size to 64 on 4096-token prompts.
    * **Impact**: KV Cache memory exceeds GPU VRAM, triggering a sudden CUDA OOM crash.
    * **Remedy**: Use dynamic batching and continuous batching engines like vLLM.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Loading FP32 Weights for Production Serving**
    * **Failure**: Omitting `torch_dtype` during `from_pretrained`, defaulting to FP32.
    * **Impact**: Doubles GPU memory usage (28GB for 7B model) and cuts inference throughput in half.
    * **Remedy**: Always specify `torch_dtype=torch.bfloat16`.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Prefill vs Decode Bottleneck"):
        st.write("Which hardware resource bottlenecks the Prefill phase vs the Decode phase?")
        st.info("Prefill is Compute-Bound (FLOPS); Decode is Memory-Bandwidth Bound (GB/s).")
    with st.expander("⚡ Flashcard 2: Autoregressive Step Count"):
        st.write("How many forward passes are required to generate N tokens?")
        st.info("N forward passes (one forward pass per token).")

with fc_col2:
    with st.expander("⚡ Flashcard 3: TTFT vs ITL Definitions"):
        st.write("What do TTFT and ITL stand for?")
        st.info("Time-To-First-Token (prefill latency) and Inter-Token Latency (streaming speed).")
    with st.expander("⚡ Flashcard 4: Recommended Inference Context Mode"):
        st.write("What PyTorch context manager provides the lowest overhead during inference?")
        st.info("`torch.inference_mode()`.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Inference Fundamentals before advancing to Lesson 16.2 (Decoding Strategies):
""")

chk1 = st.checkbox("I understand the architectural differences between Training and Inference.", value=True)
chk2 = st.checkbox("I know why the Decode phase is memory-bandwidth bound with arithmetic intensity ~1.", value=True)
chk3 = st.checkbox("I can explain Time-To-First-Token (TTFT) vs Inter-Token Latency (ITL).", value=True)
chk4 = st.checkbox("I understand how the unembedding head maps hidden states to vocabulary logits.", value=True)
chk5 = st.checkbox("I know why torch.inference_mode() is preferred over torch.no_grad().", value=True)
chk6 = st.checkbox("I understand the quadratic O(N^2) complexity of attention over long context windows.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 16.2: Decoding Strategies.")
