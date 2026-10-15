import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.3 LLM Architecture Overview - LLM Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("1.3 LLM Architecture Overview: From Raw Text to Token Probabilities")
st.caption("Chapter 1: Introduction to Large Language Models | Domain: 2. Large Language Models")

st.markdown("""
Understanding a Large Language Model requires peeling back the internal data transformations of the 
**Decoder-Only Transformer forward pass** from the moment a user submits an input prompt to the moment 
the next token is predicted. 

Every modern foundation model (including Llama 3, Mistral, Qwen, and GPT-4) executes a unified 9-stage 
transformation pipeline: **Raw UTF-8 Text -> Subword Tokenizer -> Token Embeddings -> Positional Encoding (RoPE) 
-> Residual Stream Highway -> Multi-Head / Grouped-Query Attention -> SwiGLU Feed-Forward Networks -> 
RMSNorm & Linear Unembedding Head -> Softmax & Temperature Sampling**.

In this lesson, you will master:
1. The 9 sequential stages converting raw human strings into categorical token probability distributions.
2. The mathematics and tensor dimensions governing each stage in plain English without broken notation.
3. How modern foundation models (Llama 3, Mistral, GPT-4) implement Pre-RMSNorm, RoPE, GQA, and SwiGLU.
4. The role of the central Residual Stream highway in preventing gradient degradation across 100+ layers.
5. A runnable single-layer Transformer forward pass implemented in pure Python and NumPy.
""")

st.divider()

# ==============================================================================
# Section 1: The End-to-End Forward Pass Architecture
# ==============================================================================
st.header("1. The End-to-End Forward Pass Architecture", divider="rainbow")

st.markdown("""
Below is the complete architectural flowchart of a state-of-the-art autoregressive foundation model. 
Notice how the central **Residual Stream** acts as an additive highway: layers read information from the highway, 
compute contextual updates, and write additive deltas back onto the stream without destructive overwrites.
""")

col_arch1, col_arch2 = st.columns([1.3, 0.7])

with col_arch1:
    arch_dot = """
    digraph LLMArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=polyline;
        nodesep=0.35;
        ranksep=0.55;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        // Stage 1-2: Text Ingestion
        subgraph cluster_input {
            label="Stage 1 & 2: Ingestion and Discretization";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            RawText [label="1. Raw UTF-8 Text Input\\n'The capital of France is...'", fillcolor="#1E293B"];
            Tokenizer [label="2. Subword Tokenizer (BPE / Tiktoken)\\nConverts string to IDs: [791, 6827, 315, 8740, 374]\\nShape: [Batch_Size, Sequence_Length]", fillcolor="#0F172A", color="#38BDF8"];
        }

        // Stage 3-4: Embedding & RoPE
        subgraph cluster_embed {
            label="Stage 3 & 4: Latent Projection and Position";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            EmbedLookup [label="3. Token Embedding Lookup (W_emb)\\nParameter Shape: [Vocab_Size, Hidden_Dim]\\nOutput: [Batch_Size, Sequence_Length, Hidden_Dim]", fillcolor="#0C4A6E", color="#38BDF8"];
            RoPE [label="4. Rotary Position Embeddings (RoPE)\\nRotates Q & K in 2D complex planes by m * theta\\nInjects relative distance awareness into attention", fillcolor="#0C4A6E", color="#38BDF8"];
        }

        // Stage 5-7: Transformer Block
        subgraph cluster_transformer {
            label="Stage 5-7: Stacked Transformer Blocks (N = 32 to 128 Layers)";
            style=filled;
            color="#1E293B";
            fontcolor="#818CF8";
            RMSNorm1 [label="Pre-RMSNorm 1\\nx_norm1 = RMSNorm(x_residual)", fillcolor="#1E1B4B", color="#818CF8"];
            GQA [label="6. Grouped-Query Attention (GQA)\\nQueries look up Keys, aggregate Values\\nShape: [Batch_Size, Sequence_Length, Hidden_Dim]", fillcolor="#1E1B4B", color="#818CF8"];
            Add1 [label="5. Residual Add Highway (1)\\nx_residual = x_residual + GQA_Output", fillcolor="#312E81", color="#A5B4FC"];
            RMSNorm2 [label="Pre-RMSNorm 2\\nx_norm2 = RMSNorm(x_residual)", fillcolor="#1E1B4B", color="#818CF8"];
            SwiGLU [label="7. SwiGLU Feed-Forward Network\\nSwish(x * W_gate) * (x * W_up) * W_down\\nParametric factual memory store", fillcolor="#1E1B4B", color="#818CF8"];
            Add2 [label="5. Residual Add Highway (2)\\nx_residual = x_residual + FFN_Output", fillcolor="#312E81", color="#A5B4FC"];
        }

        // Stage 8-9: Output & Sampling
        subgraph cluster_output {
            label="Stage 8 & 9: Unembedding and Sampling";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            FinalRMS [label="Final Layer RMSNorm\\nStabilizes residual variance across 100+ layers", fillcolor="#1E293B"];
            Unembed [label="8. Linear Unembedding Head (W_unembed)\\nProjects [Batch_Size, 1, Hidden_Dim] -> [Batch_Size, 1, Vocab_Size]\\nProduces raw unnormalized logits z", fillcolor="#064E3B", color="#34D399"];
            Softmax [label="9. Softmax & Temperature Sampling\\nP(w) = Softmax(z / T)\\nEmits next token ID -> Feeds back to Stage 1", fillcolor="#065F46", color="#34D399"];
        }

        RawText -> Tokenizer;
        Tokenizer -> EmbedLookup;
        EmbedLookup -> RMSNorm1;
        RoPE -> GQA [style=dashed, label="Applies to Q & K"];

        RMSNorm1 -> GQA;
        EmbedLookup -> Add1 [label="Residual Skip"];
        GQA -> Add1;
        Add1 -> RMSNorm2;
        Add1 -> Add2 [label="Residual Skip"];
        RMSNorm2 -> SwiGLU;
        SwiGLU -> Add2;
        Add2 -> FinalRMS;
        FinalRMS -> Unembed;
        Unembed -> Softmax;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_dot, use_container_width=True)

with col_arch2:
    st.markdown("### The 9 Transformer Stages")
    st.markdown("""
    * **Stage 1: Input Text**: Discrete UTF-8 human text prompt.
    * **Stage 2: Tokenization**: Integer token sequence via subword BPE.
    * **Stage 3: Token Embeddings**: High-dimensional continuous lookup vector.
    * **Stage 4: Positional Information**: Relative rotation via RoPE.
    * **Stage 5: Transformer Layers**: Pre-RMSNorm and the additive Residual Highway.
    * **Stage 6: Attention Mechanism**: Grouped-Query Attention (GQA) token routing.
    * **Stage 7: Feed-Forward Networks**: SwiGLU non-linear factual memory.
    * **Stage 8: Output Probabilities**: Linear unembedding head and Softmax.
    * **Stage 9: Token Generation**: Temperature sampling and KV-cache autoregression.
    """)

st.divider()

# ==============================================================================
# Section 2: 1. Input Text
# ==============================================================================
st.header("2. Input Text: Unicode Byte Streams & Context Windows", divider="rainbow")

st.markdown("""
The transformation pipeline begins with raw human input text. 
At the machine boundary, text is not an abstract concept; it is an ordered sequence of **UTF-8 encoded byte streams**. 
Every character—from basic ASCII alphanumeric characters to complex multilingual scripts and emojis—is represented 
by 1 to 4 bytes in standard UTF-8 encoding.

### The Context Window Boundary
The input text is placed inside a finite **Context Window buffer**. The context window represents the maximum sequence length 
`Sequence_Length` that the model can process concurrently in a single forward pass:
* **Batch Shape**: `[Batch_Size, Sequence_Length]`, where each element is a discrete token integer.
* **Prompt Prefill vs Generation Phase**: In the prefill phase, all `Sequence_Length` input tokens are processed in parallel across all transformer layers. In the generation phase, only 1 new token is ingested at each autoregressive step.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Massive Context Expansion**: While original transformers (BERT, GPT-2) were constrained to 512 or 1,024 tokens, modern models handle 128,000 tokens (Llama 3), 200,000 tokens (Claude 3.5 Sonnet), and 2,000,000 tokens (Gemini 1.5 Pro).
* **Chunked Prefill**: Large prompt texts (e.g., a 100-page PDF containing 50,000 tokens) cannot be loaded onto a GPU in a single monolithic matrix multiplication without causing out-of-memory (OOM) or stalling interactive users. Modern serving systems (vLLM, SGLang) chunk input text into 512-token segments, interleaving prefill computation with live token generation steps.
* **Byte-Fallback Encodings**: Rather than replacing unknown characters with an out-of-vocabulary (`<unk>`) token, modern foundation tokenizers fall back directly to raw UTF-8 byte representations, ensuring zero information loss on arbitrary character sequences.
""")

df_s1 = pd.DataFrame({
    "Dimension": [
        "Ingestion Representation",
        "Out-of-Vocabulary Handling",
        "Context Window Scale",
        "Prefill Processing Mode",
        "Special Control Tokens"
    ],
    "Legacy NLP (RNN / Word2Vec / BERT)": [
        "Whitespace-split strings or static word vocabularies",
        "Mapped to generic <unk> token; permanent loss of rare words and typos",
        "Strictly 512 tokens (BERT) or recurrent state with memory decay",
        "Sequential word-by-word recurrent step or unpadded batch truncation",
        "Basic [CLS] and [SEP] delimiters"
    ],
    "Modern Foundation LLMs (Llama 3 / GPT-4)": [
        "UTF-8 byte-level subword tokens with byte-fallback preservation",
        "Zero <unk> tokens; arbitrary Unicode and binary bytes cleanly tokenized",
        "32,000 to 2,000,000 tokens with FlashAttention-3 and chunked prefill",
        "Fully parallel causal self-attention across prompt tokens with KV-cache setup",
        "Structured header tokens: <|begin_of_text|>, <|start_header_id|>, <|eot_id|>"
    ]
})
st.dataframe(df_s1, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Tokenization
# ==============================================================================
st.header("3. Tokenization: Subword Discretization & Vocabulary Mapping", divider="rainbow")

st.markdown("""
Computers cannot directly ingest natural language characters; neural networks require numerical matrices. 
**Tokenization** is the deterministic discretization process that maps continuous strings into ordered arrays of token integers 
according to a pre-built static vocabulary table `V`.

### Subword Algorithmic Foundations
Modern LLMs use subword tokenization algorithms—primarily **Byte-Pair Encoding (BPE)** (OpenAI Tiktoken, Llama 3) 
and **SentencePiece / Unigram** (Mistral, Gemini):
1. **Initial Vocabulary**: Begins with 256 individual UTF-8 bytes.
2. **Iterative Merge Learning**: The algorithm iteratively merges the most frequently co-occurring adjacent byte/character pairs across a trillion-token training corpus.
3. **Vocabulary Size**: The resulting vocabulary typically spans 32,000 tokens (Llama 2, Mistral) to 128,256 tokens (Llama 3) or 200,000 tokens (GPT-4o).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Expanding Vocabularies**: Increasing vocabulary size from 32k (Llama 2) to 128k (Llama 3) reduced token fertility on non-Latin scripts (Hindi, Arabic, Japanese, code) by 30% to 50%, significantly accelerating inference speed and reducing cost.
* **Regex Pre-Splitting Rules**: Rather than merging arbitrary whitespace and punctuation, modern tokenizers (such as `cl100k_base` and Llama 3's tokenizer) apply sophisticated regex patterns before BPE merging to prevent numbers from being grouped into irregular multi-digit blocks (e.g. splitting `1234` into distinct numeric tokens).
* **Deterministic Frozen Parameters**: Tokenizers have zero neural weights and do not update during model pretraining. Their mappings are entirely deterministic integer lookups.
""")

df_s2 = pd.DataFrame({
    "Dimension": [
        "Vocabulary Size",
        "Character Coverage",
        "Fertility on Non-English Scripts",
        "Numeric Tokenization",
        "Whitespace & Indentation Handling"
    ],
    "Legacy Tokenizers (WordPiece / GPT-2 BPE)": [
        "30,000 to 50,257 tokens",
        "Prone to <unk> on unmapped characters and rare emojis",
        "High fertility (5 to 8 tokens per single word in Asian/Indic scripts)",
        "Merges multiple digits randomly (e.g., '12' as one token, '34' as another)",
        "Inefficient; encodes leading whitespace space-by-space"
    ],
    "Modern Foundation Tokenizers (Tiktoken / Llama 3)": [
        "100,000 to 128,256 tokens",
        "100% byte-fallback coverage across all Unicode code points and emojis",
        "Low fertility (1 to 2 tokens per word across global multilingual scripts)",
        "Regex rules isolate individual digits, aiding arithmetic reasoning",
        "Multi-space indentation tokens optimized for Python and code blocks"
    ]
})
st.dataframe(df_s2, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Token Embeddings
# ==============================================================================
st.header("4. Token Embeddings: High-Dimensional Latent Lookup Matrices", divider="rainbow")

st.markdown("""
Once the text string is converted into token IDs `[t_1, t_2, ..., t_L]`, the model projects each discrete integer 
into a continuous, high-dimensional geometric vector space via the **Token Embedding Matrix** `W_emb`.

### Mathematical Formulation
The embedding matrix `W_emb` is a learned parameter tensor:
* **Shape**: `[Vocabulary_Size, Hidden_Dimension]` (e.g., `[128256, 4096]` in Llama-3-8B).
* **Lookup Operation**: For a token integer ID `i`, the embedding vector `e_i` is retrieved by indexing the `i-th` row of `W_emb`:

`e_i = W_emb[i, :]`

* **Tensor Transformation**:
  * Input Shape: `[Batch_Size, Sequence_Length]` (integer IDs)
  * Output Shape: `[Batch_Size, Sequence_Length, Hidden_Dimension]` (continuous floating-point vectors, e.g. BF16)
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Untied vs Tied Embeddings**: In early transformers (GPT-2, T5), the embedding matrix `W_emb` and the unembedding matrix `W_unembed` shared weights (Weight Tying) to save parameters. Modern foundation models (Llama 3, Mistral) decouple them (Untied Embeddings), allowing input representations and output probability projections to specialize independently.
* **Initialization & Embedding Scaling**: In standard architectures, embeddings are initialized with a standard deviation of `1 / sqrt(Hidden_Dimension)`. Some architectures (such as Gemma) multiply embeddings by `sqrt(Hidden_Dimension)` before adding positional information to stabilize initial gradient magnitudes.
* **VRAM Overhead**: For large vocabularies (128,256 tokens) and wide hidden dimensions (8,192 in 70B models), `W_emb` alone consumes over 2 GB of GPU VRAM in BF16 precision.
""")

df_s3 = pd.DataFrame({
    "Dimension": [
        "Embedding Dimensionality",
        "Weight Sharing Paradigm",
        "Precision Format",
        "VRAM Footprint (8B vs 70B)",
        "Normalization"
    ],
    "Legacy Word Vectors (Word2Vec / GloVe)": [
        "100 to 300 dimensions",
        "Static lookup table; static cosine similarity irrespective of context",
        "FP32 (32-bit floating point)",
        "Under 50 MB total",
        "Static unit vector normalization without learned gains"
    ],
    "Modern Foundation LLM Embeddings (Llama 3 / Qwen)": [
        "4,096 dimensions (8B) to 8,192 dimensions (70B)",
        "Untied embeddings; independent W_emb and W_unembed parameter tensors",
        "BF16 (bfloat16) or FP8 (8-bit floating point) in quantized runtimes",
        "1.05 GB (8B: 128k x 4096) to 2.1 GB (70B: 128k x 8192) in BF16",
        "Layer-normed or RMS-normed before residual stream injection"
    ]
})
st.dataframe(df_s3, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Positional Information
# ==============================================================================
st.header("5. Positional Information: Absolute, Relative & Rotary Position Embeddings (RoPE)", divider="rainbow")

st.markdown("""
A fundamental mathematical property of self-attention is **permutation equivariance**: without positional information, 
the sentence `"The dog bit the man"` and `"The man bit the dog"` produce identical attention distributions. 
To inject order awareness, transformer architectures must modulate token representations with positional coordinates.

### Evolution of Positional Modulation
1. **Absolute Positional Embeddings (APE)** (Original Transformer, GPT-2): A learned or sinusoidal vector `P_pos` of shape `[Sequence_Length, Hidden_Dimension]` is added directly to the token embedding: `x = e_token + P_pos`.
   * *Limitation*: Cannot extrapolate beyond the fixed pretraining sequence length.
2. **Relative Positional Encodings** (T5, ALiBi): Injects scalar distance penalties into the attention score matrix: `Score(i, j) = Q_i * K_j^T - m * |i - j|`.
3. **Rotary Position Embeddings (RoPE)** (Su et al., 2021): Rotates Query and Key vectors in 2D coordinate planes by angles proportional to their position index `m * theta`.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Ubiquity of RoPE**: Every top-tier open-weights foundation model (Llama 3, Mistral, Qwen, DeepSeek) uses **RoPE**. Instead of adding a vector to the residual stream, RoPE rotates `Q` and `K` inside each attention head immediately before computing dot-product attention.
* **Inner Product Relative Preservation**: When computing `Q_m` at position `m` and `K_n` at position `n`, the inner product `Q_m * K_n^T` depends strictly on the relative distance `m - n`, granting the model inherent relative position sensitivity.
* **RoPE Frequency Scaling for Long Context**: To scale context windows from 8k to 128k tokens, modern models scale the base theta parameter (e.g. Llama 3 increased `theta_base` from 10,000 to 500,000) or apply YaRN (Yet another RoPE extensioN) interpolation to prevent high-frequency rotational distortion.
""")

df_s4 = pd.DataFrame({
    "Dimension": [
        "Positional Mechanism",
        "Injection Point",
        "Context Extrapolation",
        "Theoretical Property",
        "Modern Adoption"
    ],
    "Absolute Positional Embeddings (APE: GPT-2 / BERT)": [
        "Additive learned positional vectors P_pos",
        "Added to token embeddings once at layer 0",
        "Fails completely beyond pretraining length (hard ceiling)",
        "Encodes absolute coordinates m; ignores relative distance m - n",
        "Largely obsolete in modern generative foundation models"
    ],
    "Rotary Position Embeddings (RoPE: Llama 3 / Mistral)": [
        "Multiplicative 2D rotation of Q and K vectors by angle m * theta",
        "Applied dynamically inside every attention head at every layer",
        "Extrapolates smoothly to 128k+ tokens via base theta scaling and YaRN",
        "Inner product depends strictly on relative offset (m - n)",
        "Universal gold standard across all modern frontier models"
    ]
})
st.dataframe(df_s4, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Transformer Layers
# ==============================================================================
st.header("6. Transformer Layers: Pre-RMSNorm & Residual Stream Highways", divider="rainbow")

st.markdown("""
The core processing engine of a modern foundation model consists of a deep stack of identical **Transformer Blocks** 
(typically `N = 32` layers in an 8B model, up to `N = 128` layers in a 70B+ model). 
Two architectural principles govern every modern transformer block: **Pre-Layer Normalization** and the **Residual Stream Highway**.

### The Residual Stream Highway
The residual stream is the central additive vector running uninterrupted from layer 1 to layer N. 
Rather than passing activations through deep non-linear functions sequentially (`x_{l+1} = f(x_l)`), each sublayer reads from 
the stream, computes a delta, and **adds** the delta back onto the stream:

`x_{mid} = x_{in} + Attention(Norm(x_{in}))`  
`x_{out} = x_{mid} + FFN(Norm(x_{mid}))`

Because the skip connections are purely additive, gradients flow backwards during backpropagation along an unobstructed 
linear pathway straight from the loss head to the input embeddings, eliminating vanishing gradients.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Pre-LayerNorm over Post-LayerNorm**: Original transformers used Post-LN (`x = Norm(x + SubLayer(x))`), which caused severe gradient instability requiring warm-up schedules. Modern models use Pre-LN (`x = x + SubLayer(Norm(x))`), ensuring the residual stream remains unnormalized.
* **RMSNorm replacing LayerNorm**: Standard LayerNorm calculates both the mean and variance across the hidden dimension: `Norm(x) = (x - mean) / sqrt(var + eps) * gamma + beta`. Modern models (Llama 3, Mistral, Gemma) replace LayerNorm with **Root Mean Square Normalization (RMSNorm)**:
  `RMSNorm(x) = x / sqrt(Mean(x^2) + eps) * gamma`
  RMSNorm assumes zero-mean activation distributions, eliminating the mean calculation and bias parameter `beta`, delivering a 7% to 15% speedup on GPU memory bandwidth.
""")

df_s5 = pd.DataFrame({
    "Dimension": [
        "Normalization Technique",
        "Normalization Placement",
        "Skip Connection Topology",
        "Gradient Highway Health",
        "GPU Memory Efficiency"
    ],
    "Legacy Transformer (Original 2017 / BERT)": [
        "Standard LayerNorm with mean subtraction and learned bias beta",
        "Post-LayerNorm: x = LayerNorm(x + SubLayer(x))",
        "Additive skip connection, but interrupted by post-layer normalization",
        "Unstable; gradients vanish or explode without extensive learning rate warm-up",
        "Requires two reduction passes per token across hidden dimension"
    ],
    "Modern Foundation Transformer (Llama 3 / Mistral)": [
        "RMSNorm: Root Mean Square normalization without mean or bias parameters",
        "Pre-LayerNorm: x = x + SubLayer(RMSNorm(x))",
        "Uninterrupted additive residual stream highway across all 128 layers",
        "Extremely stable; zero gradient vanishing, allowing 100+ layer scaling",
        "Single reduction pass per token; 10% to 15% faster kernel execution"
    ]
})
st.dataframe(df_s5, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Attention Mechanism
# ==============================================================================
st.header("7. Attention Mechanism: Scaled Dot-Product, Multi-Head & Grouped-Query Attention (GQA)", divider="rainbow")

st.markdown("""
The self-attention mechanism is the cognitive routing core of the Transformer: it enables every token in the sequence 
to dynamically broadcast informational queries, evaluate relevance against past tokens, and aggregate contextual values.

### Mathematical Formulation of Scaled Dot-Product Attention
Given projection matrices `W_Q`, `W_K`, and `W_V`, input states are projected into Queries `Q`, Keys `K`, and Values `V`:
* `Q = x * W_Q`, `K = x * W_K`, `V = x * W_V`
* Scaled Dot-Product Attention:

`Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(Head_Dim) + Mask) * V`

* **Causal Masking**: In autoregressive decoder-only models, future tokens are masked out with negative infinity (`-inf`) before the Softmax operation, preventing the model from cheating by looking ahead.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Multi-Head (MHA) vs Multi-Query (MQA) vs Grouped-Query Attention (GQA)**:
  * **Multi-Head Attention (MHA)** (GPT-3): Each Query head has its own Key and Value head (`Num_Q = Num_KV`). During autoregressive inference, caching all KV heads consumes massive GPU VRAM.
  * **Grouped-Query Attention (GQA)** (Llama 3, Mistral): Multiple Query heads (e.g. 8 Query heads) share a single Key-Value head (`Num_Q = 32, Num_KV = 8`). GQA slashes the inference KV-cache size by 4x to 8x with zero loss in reasoning capacity.
* **FlashAttention-2 and FlashAttention-3**: Standard attention materializes the intermediate `[Sequence_Length, Sequence_Length]` attention matrix in slow GPU High-Bandwidth Memory (HBM). FlashAttention tiles the computation across fast on-chip SRAM, eliminating quadratic HBM memory access and providing a 3x to 5x speedup.
""")

df_s6 = pd.DataFrame({
    "Dimension": [
        "Attention Architecture",
        "Key-Value Head Ratio",
        "KV-Cache Memory Footprint",
        "Inference Throughput (Tokens/sec)",
        "Kernel Execution"
    ],
    "Legacy Multi-Head Attention (MHA: GPT-3 / Llama 1)": [
        "Multi-Head Attention (MHA)",
        "1:1 (32 Query heads paired with 32 KV heads)",
        "Massive (e.g., 64 GB of KV-cache for a 70B model at 8k context)",
        "Memory bandwidth bound; low batch size ceiling during generation",
        "Naive PyTorch matrix multiplication; quadratic HBM reads"
    ],
    "Modern Grouped-Query Attention (GQA: Llama 3 / Mistral)": [
        "Grouped-Query Attention (GQA)",
        "4:1 or 8:1 (32 Query heads sharing 4 or 8 KV heads)",
        "Compact (slashes KV-cache by 4x to 8x, fitting 128k context on 1 GPU)",
        "High throughput; supports 4x larger batch sizes during inference",
        "FlashAttention-3 IO-aware tiling executed directly in SRAM"
    ]
})
st.dataframe(df_s6, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Feed-Forward Networks
# ==============================================================================
st.header("8. Feed-Forward Networks: Two-Layer MLPs, SwiGLU Gating & Mixture-of-Experts (MoE)", divider="rainbow")

st.markdown("""
While the attention mechanism routes information **between** different tokens across the sequence, the 
**Feed-Forward Network (FFN)** operates position-wise on **each individual token in isolation**. 
Research indicates that while attention handles relational syntax, the FFN layers store the model's vast 
**factual world knowledge, factual associations, and associative memories**.

### Evolution from ReLU to SwiGLU
1. **Classic Two-Layer MLP** (Original Transformer, GPT-2): Used ReLU or GELU activations:
   `FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2`
2. **SwiGLU Gated Multi-Layer Perceptron** (Llama 3, Mistral, PaLM): Replaces simple activation with a gated linear unit:
   `SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down`
   * `W_gate`: Projects `[Hidden_Dim, Intermediate_Dim]` to compute gating coefficients.
   * `W_up`: Projects `[Hidden_Dim, Intermediate_Dim]` to produce content representations.
   * `*`: Element-wise multiplication (Hadamard product).
   * `W_down`: Projects `[Intermediate_Dim, Hidden_Dim]` back to the residual stream dimension.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Intermediate Dimension Sizing**: In classic transformers, `Intermediate_Dim = 4 * Hidden_Dim`. In SwiGLU architectures, to keep total parameter count balanced despite having 3 weight matrices instead of 2, the intermediate dimension is set to approximately `(8 / 3) * Hidden_Dim` (e.g. 14,336 in Llama-3-8B).
* **Sparse Mixture-of-Experts (MoE)**: Frontier models (Mixtral 8x7B, DeepSeek-V3, GPT-4) replace dense FFNs with Sparse Mixture-of-Experts. Instead of passing every token through a massive 100B parameter FFN, a lightweight router network routes each token to only top-2 out of 8 (or 16) specialized expert FFNs. This delivers the reasoning capacity of a 50B+ model while consuming the inference FLOPs of a 12B model.
""")

df_s7 = pd.DataFrame({
    "Dimension": [
        "FFN Activation Architecture",
        "Parameter Matrices per Block",
        "Intermediate Expansion Ratio",
        "Sparse Routing Support (MoE)",
        "Factual Knowledge Retention"
    ],
    "Legacy Dense FFN (GPT-2 / BERT)": [
        "Standard GELU / ReLU two-layer MLP",
        "2 matrices (W_1, W_2) plus learned additive biases b_1, b_2",
        "Strictly 4 * Hidden_Dim",
        "Dense only; all parameters execute for every token",
        "Prone to catastrophic interference across diverse factual domains"
    ],
    "Modern SwiGLU / MoE FFN (Llama 3 / Mixtral)": [
        "SwiGLU gated activation: Swish(x * W_gate) * (x * W_up) * W_down",
        "3 matrices (W_gate, W_up, W_down) with zero additive biases",
        "Approximately (8 / 3) * Hidden_Dim (e.g., 14,336 in 8B)",
        "MoE routing: routes tokens to top-k experts dynamically",
        "High factual capacity with superior parameter efficiency"
    ]
})
st.dataframe(df_s7, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: 8. Output Probabilities
# ==============================================================================
st.header("9. Output Probabilities: Linear Unembedding Head & Softmax Normalization", divider="rainbow")

st.markdown("""
After traversing all `N` transformer layers, the final token state emerges onto the residual stream with dimension 
`[Batch_Size, Sequence_Length, Hidden_Dimension]`. To generate a prediction, this continuous internal vector 
must be projected back into the discrete vocabulary space via the **Unembedding Head** (also known as the LM Head).

### Mathematical Projection to Probabilities
1. **Final Layer Normalization**: The final residual state is normalized using RMSNorm to ensure stable unit variance:
   `h_final = RMSNorm(x_residual_final)`
2. **Linear Unembedding (`W_unembed`)**: The normalized hidden state is multiplied by the unembedding matrix:
   `z = h_final * W_unembed`
   * `h_final` shape: `[Batch_Size, 1, Hidden_Dimension]` (focusing on the final token position)
   * `W_unembed` shape: `[Hidden_Dimension, Vocabulary_Size]`
   * Raw logits `z` shape: `[Batch_Size, 1, Vocabulary_Size]`
3. **Softmax Probability Normalization**: Raw logits `z` are unnormalized real numbers spanning `(-inf, +inf)`. The Softmax operator normalizes them into categorical probabilities summing strictly to 1.0:
   `P(w_i) = exp(z_i) / Sum_{j=1}^{Vocab_Size} exp(z_j)`
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Logit Soft-Capping**: In ultra-large models (such as Gemma 2), logit magnitudes can grow excessively large, causing probability collapse. Modern architectures apply soft-capping using the hyperbolic tangent function: `z_capped = cap_value * tanh(z / cap_value)` (where `cap_value = 30.0` or `50.0`), preventing numerical overflow and keeping probabilities stable.
* **Vocabulary Size Scaling**: Projecting a 4,096-dimensional hidden vector to 128,256 vocabulary logits requires substantial compute. Modern inference runtimes compute logits strictly for the final generated token position during decoding, skipping the unembedding projection for earlier prompt tokens to conserve GPU FLOPs.
""")

df_s8 = pd.DataFrame({
    "Dimension": [
        "Unembedding Matrix Shape",
        "Weight Tying with Input W_emb",
        "Logit Normalization Stabilizer",
        "Decoding Compute Scope",
        "Precision in Production"
    ],
    "Legacy Output Heads (GPT-2 / T5)": [
        "[Hidden_Dim, 32000 to 50257]",
        "Shared weights (W_unembed = W_emb^T)",
        "None; raw logits passed directly to Softmax",
        "Often computed across all sequence positions naively",
        "FP32"
    ],
    "Modern Foundation Unembedding (Llama 3 / Gemma 2)": [
        "[Hidden_Dim, 128256 to 256000]",
        "Untied independent parameter matrix for maximum expressive capacity",
        "Tanh logit soft-capping (cap = 30.0) in frontier models",
        "Computed strictly on final token index [:, -1, :] during autoregressive steps",
        "BF16 or FP8 with selective FP32 upcasting for Softmax stability"
    ]
})
st.dataframe(df_s8, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: 9. Token Generation
# ==============================================================================
st.header("10. Token Generation: Autoregressive Sampling & KV-Cache Dynamics", divider="rainbow")

st.markdown("""
The final stage transforms the vocabulary probability distribution into an actual emitted token string, 
appends it to the conversation history, and prepares the model for the next forward pass.

### Sampling Heuristics & Temperature Dynamics
Once probabilities `P(w)` are computed, the decoding algorithm selects the winning token index:
1. **Greedy Argmax Decoding** (`Temperature = 0.0`): Deterministically picks the highest probability token: `token_id = argmax(P(w))`. Produces reproducible, factual outputs but suffers from repetitive loops in creative tasks.
2. **Temperature Scaling** (`T > 0`): Divides logits by `T` before Softmax. Higher temperature flattens the distribution, increasing diversity.
3. **Top-k Filtering**: Restricts sampling to the top `K` most probable tokens (e.g., `K = 50`).
4. **Nucleus (Top-p) Sampling**: Sorts tokens in descending probability and selects from the minimal set whose cumulative sum reaches threshold `p` (typically `p = 0.9`).
5. **Min-P Sampling**: A modern alternative that discards tokens whose probability is less than `Min_P * Max_Probability` (e.g. `Min_P = 0.05`), dynamically adapting the candidate pool based on model confidence.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **The KV-Cache Engine**: In naive generation, emitting 100 new tokens requires running 100 forward passes where the sequence grows from length 1 to 100, recomputing attention for all past tokens (`O(N^2)` complexity). Modern inference runtimes store past Key and Value tensors in the **KV-Cache** in GPU HBM. At step `t`, only token `t` is projected into Query `Q_t`, which attends against cached Keys `[K_1, ..., K_{t-1}, K_t]`, reducing per-step computation to `O(1)`.
* **PagedAttention (vLLM)**: Similar to virtual memory paging in operating systems, PagedAttention allocates non-contiguous physical GPU memory blocks for the KV-cache, eliminating memory fragmentation and enabling 4x to 8x higher concurrency.
* **Stop Tokens**: Generation halts cleanly when the model samples a designated terminal token (e.g., `<|eot_id|>`, `<|end_of_text|>`, or `</s>`).
""")

df_s9 = pd.DataFrame({
    "Dimension": [
        "Inference Compute Complexity",
        "Memory Management",
        "Sampling Modern Heuristics",
        "Multi-Token Speculation",
        "Batching Strategy"
    ],
    "Legacy Autoregressive Generation": [
        "O(N^2) quadratic recomputation of entire prompt history per token",
        "Static contiguous GPU memory buffers; frequent Out-Of-Memory crashes",
        "Greedy or standard Top-k / Top-p only",
        "Strictly 1 token per full model forward pass",
        "Static padding batches; GPU cores idle waiting for shortest sequences"
    ],
    "Modern Foundation Serving (vLLM / TensorRT-LLM)": [
        "O(1) per-step attention compute via Grouped-Query KV-caching",
        "PagedAttention: Non-contiguous dynamic page allocation with zero waste",
        "Min-P sampling, Dry repetition penalties, and grammar-constrained masks",
        "Speculative decoding: Small draft model generates 5 tokens; target verifies in 1 step",
        "Continuous in-flight iteration-level batching maximizing GPU saturation"
    ]
})
st.dataframe(df_s9, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 11: Production Implementation: Pure Python Forward Pass
# ==============================================================================
st.header("11. Production Implementation: Full Single-Layer Transformer in Pure Python", divider="rainbow")

st.markdown("""
The following complete, runnable Python implementation demonstrates the exact mathematical forward pass of a 
modern Transformer block using pure NumPy. It includes **RMSNorm**, **Rotary Position Embeddings (RoPE)**, 
**Multi-Head Attention with Causal Masking**, and **SwiGLU Feed-Forward Networks**.
""")

st.code(r'''import numpy as np

# ----------------------------------------------------------------------
# 1. Mathematical Helper Primitives: RMSNorm, RoPE, Swish
# ----------------------------------------------------------------------
def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square Normalization across hidden dimension."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma

def apply_rope(x: np.ndarray, seq_len: int, head_dim: int) -> np.ndarray:
    """Applies 2D Rotary Position Embeddings to Query or Key tensors."""
    # Compute inverse frequency theta
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2).astype(float) / head_dim))
    t = np.arange(seq_len)
    freqs = np.outer(t, inv_freq)  # [seq_len, head_dim // 2]
    
    # Construct sin and cos representations
    cos_f = np.cos(freqs)
    sin_f = np.sin(freqs)
    
    # Split into even and odd indices for 2D rotation
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    out_even = x_even * cos_f - x_odd * sin_f
    out_odd = x_even * sin_f + x_odd * cos_f
    
    out = np.zeros_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out

def swish(x: np.ndarray) -> np.ndarray:
    """Swish (SiLU) activation function: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ----------------------------------------------------------------------
# 2. Executable Single-Layer Decoder Transformer Forward Pass
# ----------------------------------------------------------------------
class SingleLayerTransformer:
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_heads: int = 4, intermediate_dim: int = 128):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.intermediate_dim = intermediate_dim
        
        # Initialize synthetic parameters (BF16 simulated as FP32)
        np.random.seed(42)
        self.W_emb = np.random.randn(vocab_size, hidden_dim) * 0.02
        self.gamma1 = np.ones(hidden_dim)
        self.gamma2 = np.ones(hidden_dim)
        
        # Self-Attention Projections
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_out = np.random.randn(hidden_dim, hidden_dim) * 0.02
        
        # SwiGLU FFN Projections
        self.W_gate = np.random.randn(hidden_dim, intermediate_dim) * 0.02
        self.W_up = np.random.randn(hidden_dim, intermediate_dim) * 0.02
        self.W_down = np.random.randn(intermediate_dim, hidden_dim) * 0.02
        
        # Final Unembedding
        self.gamma_final = np.ones(hidden_dim)
        self.W_unembed = np.random.randn(hidden_dim, vocab_size) * 0.02

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape
        
        # Stage 3: Embedding Lookup -> [Batch, Seq_Len, Hidden_Dim]
        x_residual = self.W_emb[token_ids]
        
        # Stage 5 & 6: Pre-RMSNorm and Attention Sublayer
        x_norm1 = rms_norm(x_residual, self.gamma1)
        
        # Compute Projections -> [Batch, Seq_Len, Hidden_Dim]
        Q = np.matmul(x_norm1, self.W_q)
        K = np.matmul(x_norm1, self.W_k)
        V = np.matmul(x_norm1, self.W_v)
        
        # Reshape to Multi-Head: [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Stage 4: Apply RoPE
        Q = apply_rope(Q, seq_len, self.head_dim)
        K = apply_rope(K, seq_len, self.head_dim)
        
        # Scaled Dot-Product Attention Scores: Q * K^T / sqrt(Head_Dim)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
        
        # Apply Causal Masking (Upper Triangle = -inf)
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + causal_mask
        
        # Softmax Attention Weights -> [Batch, Num_Heads, Seq_Len, Seq_Len]
        attn_weights = softmax(scores, axis=-1)
        
        # Weighted Value Aggregation -> [Batch, Num_Heads, Seq_Len, Head_Dim]
        attn_out = np.matmul(attn_weights, V)
        
        # Concatenate Heads -> [Batch, Seq_Len, Hidden_Dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)
        attn_proj = np.matmul(attn_out, self.W_out)
        
        # Stage 5: Residual Add Highway (1)
        x_residual = x_residual + attn_proj
        
        # Stage 7: Pre-RMSNorm and SwiGLU FFN
        x_norm2 = rms_norm(x_residual, self.gamma2)
        gate = swish(np.matmul(x_norm2, self.W_gate))
        up = np.matmul(x_norm2, self.W_up)
        ffn_out = np.matmul(gate * up, self.W_down)
        
        # Residual Add Highway (2)
        x_residual = x_residual + ffn_out
        
        # Stage 8: Final RMSNorm and Unembedding
        x_final = rms_norm(x_residual, self.gamma_final)
        
        # Logits strictly for the final token position: [Batch, 1, Vocab_Size]
        final_token_h = x_final[:, -1:, :]
        logits = np.matmul(final_token_h, self.W_unembed)
        
        # Stage 9: Softmax Probability Distribution
        probs = softmax(logits, axis=-1)
        return probs

# Test execution
model = SingleLayerTransformer(vocab_size=5000, hidden_dim=64, num_heads=4, intermediate_dim=128)
sample_prompt_ids = np.array([[42, 108, 994, 2031]])  # Batch=1, Seq_Len=4
output_probabilities = model.forward(sample_prompt_ids)
print("Forward Pass Executed Successfully!")
print(f"Output Probability Tensor Shape: {output_probabilities.shape}")
print(f"Top-1 Predicted Token ID: {np.argmax(output_probabilities[0, 0, :])}")
''', language="python")

st.divider()

# ==============================================================================
# Section 12: Interactive Workbench: Foundation Model Architecture Explorer
# ==============================================================================
st.header("12. Interactive Workbench: Foundation Model Architecture Explorer", divider="rainbow")

st.markdown("""
Explore the real-world parameter counts, tensor dimensions, and memory footprints of frontier foundation models. 
Compare architectural choices across Llama 3, Mistral, and custom configurations.
""")

col_wb1, col_wb2 = st.columns([1, 1.2])

with col_wb1:
    model_preset = st.selectbox(
        "Select Foundation Model Preset:",
        [
            "Llama-3-8B (Meta)",
            "Llama-3-70B (Meta)",
            "Mistral-7B-v0.3 (Mistral AI)",
            "Custom Architecture"
        ]
    )

    if model_preset == "Llama-3-8B (Meta)":
        d_hidden = 4096
        n_layers = 32
        n_q_heads = 32
        n_kv_heads = 8
        d_inter = 14336
        v_size = 128256
        ctx_max = 8192
    elif model_preset == "Llama-3-70B (Meta)":
        d_hidden = 8192
        n_layers = 80
        n_q_heads = 64
        n_kv_heads = 8
        d_inter = 28672
        v_size = 128256
        ctx_max = 8192
    elif model_preset == "Mistral-7B-v0.3 (Mistral AI)":
        d_hidden = 4096
        n_layers = 32
        n_q_heads = 32
        n_kv_heads = 8
        d_inter = 14336
        v_size = 32768
        ctx_max = 32768
    else:
        d_hidden = st.number_input("Hidden Dimension (Hidden_Dim)", value=4096, step=512)
        n_layers = st.number_input("Number of Layers (N)", value=32, step=4)
        n_q_heads = st.number_input("Query Heads", value=32, step=4)
        n_kv_heads = st.number_input("KV Heads (GQA)", value=8, step=2)
        d_inter = st.number_input("Intermediate Dimension (FFN)", value=14336, step=1024)
        v_size = st.number_input("Vocabulary Size", value=128256, step=10000)
        ctx_max = st.number_input("Target Context Length", value=8192, step=4096)

    active_ctx = st.slider("Active Context Length for Calculation", min_value=512, max_value=int(ctx_max), value=min(4096, int(ctx_max)), step=512)
    batch_size = st.slider("Serving Concurrency (Batch Size)", min_value=1, max_value=64, value=8, step=1)

# Theoretical Calculations
head_dim = d_hidden // n_q_heads

# Weight parameters in one layer:
# Attention: W_q [hidden, q_heads*head_dim], W_k [hidden, kv_heads*head_dim], W_v [hidden, kv_heads*head_dim], W_out [q_heads*head_dim, hidden]
attn_params = (d_hidden * (n_q_heads * head_dim)) + (2 * d_hidden * (n_kv_heads * head_dim)) + ((n_q_heads * head_dim) * d_hidden)
# SwiGLU FFN: W_gate, W_up, W_down
ffn_params = 3 * (d_hidden * d_inter)
# RMSNorm per layer: 2 * hidden_dim
layer_params = attn_params + ffn_params + (2 * d_hidden)

# Total Model Parameters: Embeddings + N * Layer + Final Norm + Unembed
embed_params = v_size * d_hidden
unembed_params = d_hidden * v_size
total_params = embed_params + (n_layers * layer_params) + d_hidden + unembed_params

# Memory calculations in BF16 (2 bytes per parameter)
model_vram_gb = (total_params * 2) / (1024 ** 3)

# KV Cache calculation per layer: 2 * batch * kv_heads * seq_len * head_dim * 2 bytes
kv_per_layer_bytes = 2 * batch_size * n_kv_heads * active_ctx * head_dim * 2
total_kv_vram_gb = (n_layers * kv_per_layer_bytes) / (1024 ** 3)

# MHA comparison: what if MHA was used instead of GQA?
mha_kv_per_layer_bytes = 2 * batch_size * n_q_heads * active_ctx * head_dim * 2
mha_total_kv_vram_gb = (n_layers * mha_kv_per_layer_bytes) / (1024 ** 3)
gqa_savings_factor = mha_total_kv_vram_gb / max(total_kv_vram_gb, 1e-6)

with col_wb2:
    st.subheader("Architectural Dimension Specifications")

    dim_m1, dim_m2 = st.columns(2)
    dim_m1.metric("Total Parameters", f"{total_params / 1e9:.2f} Billion")
    dim_m2.metric("Base Model VRAM (BF16)", f"{model_vram_gb:.2f} GB")

    kv_m1, kv_m2 = st.columns(2)
    kv_m1.metric("GQA KV-Cache VRAM", f"{total_kv_vram_gb:.2f} GB", delta=f"{gqa_savings_factor:.1f}x smaller than MHA")
    kv_m2.metric("Total Generation VRAM", f"{(model_vram_gb + total_kv_vram_gb):.2f} GB")

    st.markdown("##### Dimension Blueprint Table")
    df_dims = pd.DataFrame({
        "Component": [
            "Hidden Dimension (Hidden_Dim)",
            "Query Heads (Q) vs KV Heads (KV)",
            "Per-Head Dimension (Head_Dim)",
            "FFN Intermediate Dimension",
            "Vocabulary Size (Tokens)",
            "GQA Compression Ratio"
        ],
        "Value": [
            f"{d_hidden:,}",
            f"{n_q_heads} Q Heads / {n_kv_heads} KV Heads",
            f"{head_dim} dimensions per head",
            f"{d_inter:,} dimensions",
            f"{v_size:,} tokens",
            f"{n_q_heads // n_kv_heads}:1 Query-to-KV Head Grouping"
        ]
    })
    st.dataframe(df_dims, use_container_width=True, hide_index=True)

    # KV Cache Comparison Chart
    df_kv_chart = pd.DataFrame({
        "Attention Mode": ["Legacy MHA KV-Cache", "Modern GQA KV-Cache"],
        "VRAM Footprint (GB)": [mha_total_kv_vram_gb, total_kv_vram_gb]
    })
    st.bar_chart(df_kv_chart.set_index("Attention Mode"))

st.divider()

# ==============================================================================
# Section 13: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Architecture Interview Questions")

qa_c1, qa_c2 = st.columns(2)

with qa_c1:
    with st.expander("Q1: Why does Grouped-Query Attention (GQA) reduce KV-cache memory without hurting accuracy?"):
        st.markdown("""
        **Core Architectural Answer:**  
        In standard Multi-Head Attention (MHA), each Query head has an independent Key and Value head. During autoregressive decoding, 
        caching Key and Value tensors scales linearly with `Num_Heads * Sequence_Length`, bottlenecking GPU memory bandwidth.
        
        Research reveals that while Query heads must remain numerous to project diverse representational subspaces, 
        Key and Value features exhibit substantial redundancy across heads. **Grouped-Query Attention (GQA)** groups multiple Query heads 
        (e.g., 8 Q-heads) to share a single Key and Value head. This reduces the KV-cache tensor size by **8x** with negligible accuracy degradation, 
        allowing 4x to 8x higher serving concurrency on a single GPU.
        """)

    with st.expander("Q2: What is the mathematical advantage of RMSNorm over standard LayerNorm?"):
        st.markdown("""
        **Answer:**  
        Standard LayerNorm normalizes inputs by computing both the mean and the variance:
        `LayerNorm(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta`
        
        Research by Zhang & Sennrich (2019) demonstrated that the shift invariance provided by mean subtraction (`x - mean`) 
        contributes almost nothing to training stability. **RMSNorm** discards mean subtraction entirely:
        `RMSNorm(x) = (x / sqrt(Mean(x^2) + eps)) * gamma`
        
        By eliminating the mean calculation and the bias parameter `beta`, RMSNorm reduces GPU memory reduction passes, 
        yielding a **7% to 15% wall-clock speedup** across transformer layer executions.
        """)

    with st.expander("Q3: Why are the skip connections in the Residual Stream strictly additive rather than concatenated?"):
        st.markdown("""
        **Answer:**  
        The Residual Stream uses additive skip connections (`x = x + SubLayer(x)`) for two mathematical reasons:
        1. **Constant Dimensionality**: Adding maintains the exact hidden dimension `[Batch_Size, Sequence_Length, Hidden_Dimension]` across all 100+ layers. Concatenation (like DenseNet) would cause tensor dimensions to explode linearly with layer depth.
        2. **Unobstructed Gradient Highways**: In backpropagation, the derivative of addition is the identity matrix (`d(x + f(x))/dx = I + df/dx`). Even if `df/dx` approaches zero, the identity term `I` allows gradients from the output loss to flow backwards through all layers without vanishing.
        """)

with qa_c2:
    with st.expander("Q4: How does Rotary Position Embedding (RoPE) extrapolate to longer contexts?"):
        st.markdown("""
        **Answer:**  
        RoPE does not add static positional vectors; it applies a 2D rotation matrix to Query and Key vectors in coordinate slices 
        by angle `m * theta`, where `m` is the token index. The inner product `Q_m * K_n^T` naturally depends on the relative distance `(m - n)`.
        
        To extrapolate beyond pretraining length, modern architectures apply **RoPE Base Frequency Scaling**: 
        by increasing `theta_base` (e.g. from 10,000 to 500,000 in Llama 3), the rotation speed for high-dimensional components slows down, 
        allowing position indices up to 128k to map within the model's familiar angular coordinates without high-frequency collision.
        """)

    with st.expander("Q5: Why did SwiGLU replace ReLU and GELU in modern transformer feed-forward networks?"):
        st.markdown("""
        **Answer:**  
        SwiGLU (Swish Gated Linear Unit) incorporates multiplicative gating:
        `SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down`
        
        Unlike monotonic activations (ReLU, GELU) that apply a fixed non-linear curve to each scalar independently, 
        gated linear units allow the network to dynamically modulate the flow of information: one projection (`W_up`) computes the content, 
        while a parallel projection (`W_gate`) acts as a continuous dynamic valve. Shazeer (2020) empirically demonstrated that SwiGLU 
        consistently achieves significantly lower perplexity than standard GELU networks at equivalent compute budgets.
        """)

st.divider()

st.subheader("Production Traps & Architecture Gotchas")

trap_c1, trap_c2 = st.columns(2)

with trap_c1:
    st.error("""
    **Production Trap 1 (Critical): Neglecting KV-Cache Memory Growth During Long-Context Serving**  
    * **Failure Mode**: Sizing GPU memory strictly based on model parameter weights (e.g. 16 GB for an 8B model) without accounting for the KV-cache.
    * **Impact**: Immediate Out-Of-Memory (OOM) crashes when serving multi-turn chat sessions or documents exceeding 32k tokens.
    * **Remediation**: Always reserve dedicated VRAM for the KV-cache: `VRAM_KV = 2 * Batch * Num_KV_Heads * Seq_Len * Head_Dim * Num_Layers * 2 bytes`. Use GQA and PagedAttention (vLLM) in production.
    """)

    st.error("""
    **Production Trap 2 (Critical): RoPE Out-Of-Distribution Context Degradation**  
    * **Failure Mode**: Querying an 8k-pretrained model with a 32k prompt without adjusting the RoPE theta frequency.
    * **Impact**: Perplexity explodes to infinity; model outputs garbled repetitive characters due to unseen rotational angles.
    * **Remediation**: Apply YaRN or RoPE base frequency scaling before attempting inference beyond the pretraining context window limit.
    """)

with trap_c2:
    st.warning("""
    **Production Trap 3 (Caution): Tying Embeddings on High-Vocabulary Models**  
    * **Failure Mode**: Sharing weights between input embeddings and the output unembedding head (`W_unembed = W_emb^T`) when vocabulary exceeds 100k tokens.
    * **Impact**: Severely constrains representational capacity; input semantic parsing and output token generation require distinct geometric properties.
    * **Remediation**: Untie input embeddings and unembedding heads in large vocabulary models despite the slight VRAM increase.
    """)

    st.warning("""
    **Production Trap 4 (Caution): Inefficient Quadratic Attention on Long Contexts**  
    * **Failure Mode**: Using standard PyTorch `torch.matmul(Q, K.T)` on sequence lengths exceeding 4,096 tokens.
    * **Impact**: GPU execution slows down dramatically as memory bandwidth saturates with quadratic HBM memory reads.
    * **Remediation**: Always compile with FlashAttention-2 or FlashAttention-3 to tile attention computations directly inside fast on-chip SRAM.
    """)

st.divider()

st.subheader("Core Architecture Flashcards")

arch_flash_choice = st.selectbox(
    "Select Architecture Flashcard to Review:",
    [
        "Grouped-Query Attention (GQA)",
        "Residual Stream Highway",
        "Rotary Position Embedding (RoPE)",
        "SwiGLU Gated Activation"
    ]
)

if arch_flash_choice == "Grouped-Query Attention (GQA)":
    st.info("""
    **Grouped-Query Attention (GQA)**: An attention architecture where multiple Query heads share a single Key and Value head. Slashes the autoregressive KV-cache memory footprint by 4x to 8x while preserving the expressive reasoning power of full multi-head attention.
    """)
elif arch_flash_choice == "Residual Stream Highway":
    st.info("""
    **Residual Stream Highway**: The central additive vector backbone running uninterrupted through all transformer layers (`x = x + SubLayer(x)`). Guarantees direct gradient flow during backpropagation, preventing vanishing gradients across 100+ layers.
    """)
elif arch_flash_choice == "Rotary Position Embedding (RoPE)":
    st.info("""
    **Rotary Position Embedding (RoPE)**: Encodes position by rotating Query and Key vectors in 2D coordinate planes by angles proportional to their index `m * theta`. Preserves relative token distance directly within attention dot-products.
    """)
else:
    st.info("""
    **SwiGLU Gated Activation**: A feed-forward activation function (`Swish(x * W_gate) * (x * W_up) * W_down`) combining non-linear gating with linear projection. Significantly improves factual capacity and convergence over standard GELU and ReLU MLPs.
    """)

st.divider()

# ==============================================================================
# Section 14: Self-Assessment Checklist
# ==============================================================================
st.header("14. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of the complete Decoder-Only Transformer architecture:")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    cb1 = st.checkbox("I can trace the complete 9-stage tensor lifecycle from raw UTF-8 string to categorical probability distribution.", value=True)
    cb2 = st.checkbox("I understand why the additive Residual Stream highway prevents gradient vanishing across 100+ layers.", value=True)
    cb3 = st.checkbox("I understand how Rotary Position Embeddings (RoPE) encode relative distance into Q and K dot-products.", value=True)

with cb_col2:
    cb4 = st.checkbox("I know why Grouped-Query Attention (GQA) reduces KV-cache memory consumption by 4x to 8x.", value=True)
    cb5 = st.checkbox("I understand how SwiGLU gated activations enhance factual knowledge storage in feed-forward networks.", value=True)
    cb6 = st.checkbox("I can calculate the exact GPU VRAM required for model weights and dynamic KV-caching during serving.", value=True)

if all([cb1, cb2, cb3, cb4, cb5, cb6]):
    st.success("🎯 **Lesson 1.3 Complete! You have demonstrated comprehensive mastery of the complete Large Language Model architecture lifecycle.**")
