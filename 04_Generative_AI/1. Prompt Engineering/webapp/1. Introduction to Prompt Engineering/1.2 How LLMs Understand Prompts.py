import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

# ==============================================================================
# Page Configuration & Guardrails
# ==============================================================================
st.set_page_config(
    page_title="1.2 How LLMs Understand Prompts | GenAI Roadmap",
    page_icon="🧠",
    layout="wide",
)

# Custom Styling for GPU-accelerated sidebar & Glassmorphic cards
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .token-chip {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🧠 1.2 How LLMs Understand Prompts — Mechanics, Tokenization & Latent Dynamics")
st.markdown(
    """
Welcome to **Lesson 1.2: How LLMs Understand Prompts**.  
To write optimal prompts, an AI engineer must understand how foundation models mechanically ingest, tokenize, attend to, and sample from tokens. 
In this module, you will explore **autoregressive decoding**, analyze **Byte-Pair Encoding (BPE)**, master the mathematics of **Temperature & Nucleus Sampling**, evaluate **Context Window attention degradation**, and examine why models behave non-deterministically.
"""
)

st.divider()

# ==============================================================================
# 1. Large Language Models Overview
# ==============================================================================
st.header("1. Large Language Models (LLM) Architectural Overview", divider="rainbow")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown(
        r"""
### What is an LLM?
A **Large Language Model (LLM)** is a deep neural network based on the **Decoder-only Transformer** architecture 
(e.g., GPT-4o, Claude 3.5 Sonnet, Llama 3, Gemini 1.5, DeepSeek-V3). 

Fundamentally, LLMs do **not** "think", "search the web", or "comprehend" in the human sense. They perform 
**conditional probability distribution modeling over discrete token vocabularies**:

$$P(w_1, w_2, \dots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_1, w_2, \dots, w_{i-1})$$

### The 3 Stages of Foundation Model Training
1. **Self-Supervised Pre-Training (Base Model)**:
   - Trained on trillions of web tokens (Common Crawl, GitHub, books, papers).
   - Objective: Predict next token $w_t$ given preceding context $w_{<t}$.
   - Result: World knowledge and syntax internalized into billions of latent vector representations, but acts as an unfiltered text completion engine.
2. **Supervised Fine-Tuning (SFT / Instruction Tuning)**:
   - Trained on curated datasets of high-quality `(Prompt, Response)` demonstration pairs.
   - Converts an undisciplined text-completer into a helpful, conversational assistant capable of executing imperatives (*"Summarize"*, *"Translate"*).
3. **Alignment (RLHF / DPO / RLAIF)**:
   - Reinforcement Learning from Human Feedback (PPO) or Direct Preference Optimization (DPO).
   - Penalizes hallucinations, unsafe advice, and toxicity; optimizes for **Helpfulness, Honesty, and Harmlessness (HHH)**.
"""
    )

with col2:
    st.markdown("#### ⚡ The Autoregressive Inference Loop")
    st.info(
        r"""
**The Inference Step-by-Step:**
1. **Tokenize**: Convert user prompt text into discrete token IDs.
2. **Embedding & Positional Encoding**: Map token IDs to continuous $d$-dimensional vector space.
3. **Multi-Head Self-Attention**: Each token computes attention scores against all prior tokens.
4. **Logits Computation**: Output vector is projected via Unembedding matrix to a vocabulary-wide logit vector $z \in \mathbb{R}^{|V|}$ (size ~100k-200k).
5. **Sampling**: Softmax + Temperature + Top-P samples **one single next token**.
6. **Append & Repeat**: The generated token is appended to the prompt, and the loop repeats until `<|end_of_text|>` or `max_tokens` is hit.
        """
    )
    
    # Interactive Latent Space Metric Cards
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Modern Vocabulary Size $|V|$", "128k – 256k", "Tokens")
    with col_m2:
        st.metric("Transformer Hidden Dim $d_{model}$", "4,096 – 8,192", "Dimensions")

st.divider()

# ==============================================================================
# 2. Tokens & Tokenization
# ==============================================================================
st.header("2. Tokens & Tokenization — The Discrete Building Blocks", divider="rainbow")

st.markdown(
    """
### What is a Token?
LLMs cannot read raw strings of Unicode characters or words directly. Text is preprocessed into numerical chunks called **Tokens** 
using subword algorithms—most notably **Byte-Pair Encoding (BPE)** (OpenAI `tiktoken`) or **SentencePiece** (Llama/Google).
"""
)

col_tok1, col_tok2 = st.columns([3, 2])

with col_tok1:
    st.markdown(
        """
#### Key Tokenization Rules of Thumb:
* **1 Token $\\approx$ 0.75 English Words** (or $\\approx 4$ characters).
* **100 Tokens $\\approx$ 75 Words**.
* **Common words** are single tokens: ` "the"`, ` " apple"`, ` " Python"`.
* **Rare or technical words** are fragmented: `"neuro"` + `"diversity"`, `"hetero"` + `"oscedasticity"`.
* **Numbers and code** are split unpredictably: `" 9821"` might be `[" 98", "21"]` or `[" 9", "821"]`.
* **Whitespace matters**: Notice the leading space! `"Apple"` $\\ne$ `" Apple"` to a tokenizer.
"""
    )

    st.subheader("🧪 Live Interactive Tokenizer Simulator")
    test_text = st.text_input(
        "Enter text to simulate token decomposition:",
        value="Prompt Engineering is 95% about context and 5% about syntax.",
        key="tok_sim_input"
    )

    # BPE simulation for demonstration
    chips_palette = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4", "#6366F1"]
    simulated_tokens = []
    words = test_text.split(" ")
    for i, w in enumerate(words):
        prefix = " " if i > 0 else ""
        if len(w) > 6:
            half = len(w) // 2
            simulated_tokens.append(prefix + w[:half])
            simulated_tokens.append(w[half:])
        else:
            simulated_tokens.append(prefix + w)

    st.markdown("**Simulated Token Chunks (Colored Alternating Boundaries):**")
    html_chips = ""
    for idx, tok in enumerate(simulated_tokens):
        c = chips_palette[idx % len(chips_palette)]
        html_chips += f'<span class="token-chip" style="background-color: {c}33; border: 1px solid {c}; color: #f8fafc;">"{tok}" <small style="color: {c};">({len(tok)}c)</small></span> '
    st.markdown(html_chips, unsafe_allow_html=True)
    st.caption(f"Estimated Token Count: **{len(simulated_tokens)} tokens** | Character Count: **{len(test_text)} chars** | Ratio: **{len(test_text)/max(len(simulated_tokens), 1):.2f} chars/token**")

with col_tok2:
    with st.container(border=True):
        st.markdown("#### 🚨 Why Tokenization Causes Model Blindspots")
        st.error(
            """
**1. The 'Strawberry' Problem (Character Counting):**
* Question: *"How many 'r's in strawberry?"*
* Why LLMs fail: The model sees token `["straw", "berry"]` as two atomic vector IDs (e.g., `[49621, 15302]`). It never inspects individual letters unless explicitly prompted to decompose the string into characters!

**2. Reversal Failures:**
* Asking an LLM to reverse words (e.g. *"reverse 'prompt'"*) fails because the attention mechanism processes tokens, not character sequences.

**3. Arithmetic Blindspots:**
* Multi-digit multiplication fails because numbers like `128` and `439` are tokenized inconsistently, destroying positional digit alignment necessary for manual carrying.
            """
        )
        st.success("💡 **Prompt Engineering Fix**: When asking for math, spelling, or character counts, instruct the model: *'First break down the word into individual hyphenated letters: s-t-r-a-w-b-e-r-r-y, then count.'*")

st.divider()

# ==============================================================================
# 3. Context Window & Attention
# ==============================================================================
st.header("3. Context Window & The Attention Mechanism", divider="rainbow")

col_ctx1, col_ctx2 = st.columns([3, 2])

with col_ctx1:
    st.markdown(
        r"""
### What is the Context Window?
The **Context Window** represents the maximum cumulative number of tokens (Input Prompt + Output Completion) that a model 
can process in a single inference session.

| Model Generation | Typical Context Window | Key Architectural Innovations |
| :--- | :--- | :--- |
| **GPT-3 (2020)** | 2,048 tokens (~1.5k words) | Standard absolute positional embeddings |
| **GPT-4 (2023)** | 8,192 – 32,768 tokens | FlashAttention-1, grouped-query attention (GQA) |
| **Llama 3.1 / 3.3 (2024)** | 128,000 tokens | RoPE (Rotary Position Embeddings) scaling, FlashAttention-2 |
| **Gemini 1.5 Pro (2024+)** | 2,000,000 tokens | Linear / sparse multi-head latent attention |

---

### The Attention Mechanism: Quadratic Complexity $O(N^2)$
In standard full self-attention, every token $i$ computes a dot-product attention score against every other token $j$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* For a sequence of length $N$, the attention matrix requires $N \times N$ calculations.
* Doubling the prompt length **quadruples** the attention computational requirement.
* To make long contexts (128k+) tractable, modern architectures employ **FlashAttention**, **PagedAttention** (vLLM), and **KV-Caching**.
"""
    )

with col_ctx2:
    st.markdown("#### 📉 The 'Lost in the Middle' Phenomenon")
    st.markdown(
        """
Empirical research (*Liu et al., Stanford 2023*) proved that LLM retrieval accuracy degrades sharply when relevant 
facts are placed in the **middle** of long prompts. Models exhibit a **U-shaped recall curve** (Primacy & Recency bias).
        """
    )

    # Plotly U-Curve
    pos_pct = np.linspace(0, 100, 50)
    retrieval_acc = 95 - 45 * np.sin(np.pi * pos_pct / 100) + np.random.normal(0, 2, 50)
    retrieval_acc = np.clip(retrieval_acc, 20, 100)

    df_curve = pd.DataFrame({"Position in Context (%)": pos_pct, "Fact Recall Accuracy (%)": retrieval_acc})
    fig_u = px.line(
        df_curve,
        x="Position in Context (%)",
        y="Fact Recall Accuracy (%)",
        title="Fact Retrieval Accuracy vs Token Placement",
        color_discrete_sequence=["#EF4444"],
    )
    fig_u.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_u, use_container_width=True)

    st.warning("🎯 **Best Practice**: Place critical system instructions and crucial data either at the **very beginning** or at the **very end** (immediately preceding the generation prompt). Never bury critical constraints in the middle of retrieved documents!")

st.divider()

# ==============================================================================
# 4. Input vs Output Tokens Dynamics
# ==============================================================================
st.header("4. Input Tokens vs Output Tokens — Asymmetric Economics & Latency", divider="rainbow")

col_io1, col_io2 = st.columns(2)

with col_io1:
    st.markdown(
        r"""
### ⚖️ Architectural Asymmetry

In production LLM infrastructure, **Input Tokens** and **Output Tokens** have drastically different performance profiles:

#### 1. Input Processing (The "Prefill" Phase):
- **Highly Parallelizable**: All prompt tokens $(x_1, \dots, x_k)$ are known in advance.
- GPUs compute matrix multiplications across the entire prompt simultaneously in parallel.
- Fast compute throughput (thousands of tokens/sec).

#### 2. Output Generation (The "Decoding" Phase):
- **Strictly Sequential**: Token $y_{t}$ cannot be computed until token $y_{t-1}$ has been sampled and appended.
- Heavily **memory-bandwidth bound** rather than compute bound.
- Slower throughput (30–120 tokens/sec depending on model size).
- Directly governs user-facing latency.
"""
    )

with col_io2:
    st.markdown(
        """
### 💰 Pricing & Latency Disparity

Because output tokens monopolize GPU memory bandwidth for sequential decoding, cloud providers charge **3x to 5x more for output tokens than input tokens**:
"""
    )
    cost_df = pd.DataFrame(
        {
            "Model": ["Claude 3.5 Sonnet", "GPT-4o", "GPT-4o-mini", "DeepSeek-V3"],
            "Input ($ / 1M tokens)": ["$3.00", "$2.50", "$0.15", "$0.14"],
            "Output ($ / 1M tokens)": ["$15.00", "$10.00", "$0.60", "$0.28"],
            "Output/Input Ratio": ["5.0x", "4.0x", "4.0x", "2.0x"],
        }
    )
    st.dataframe(cost_df, hide_index=True, use_container_width=True)

    st.info(
        """
**Engineering Implications:**
* **TTFT (Time To First Token)**: Dictated by prompt token length (prefill time).
* **Total Latency**: Dictated by output length. If an app feels slow, reducing `max_tokens` or instructing the model to *'Be concise'* improves speed far more than trimming the input prompt!
        """
    )

st.divider()

# ==============================================================================
# 5. Model Temperature & Sampling Math
# ==============================================================================
st.header("5. Temperature, Top-P & Top-K — The Mathematics of Sampling", divider="rainbow")

st.markdown(
    r"""
### The Mathematical Formula: Softmax with Temperature
When an LLM evaluates the next token, the final linear layer outputs raw unnormalized log-odds called **Logits** $z_i$ for each vocabulary token.
To convert logits into a valid probability distribution $P(w_i)$ that sums to 1.0, the model applies **Softmax scaled by Temperature $T$**:

$$P(w_i) = \frac{\exp\left(\frac{z_i}{T}\right)}{\sum_{j} \exp\left(\frac{z_j}{T}\right)}$$
"""
)

col_t1, col_t2 = st.columns([2, 3])

with col_t1:
    st.markdown("#### 🎛️ Interactive Hyperparameter Explorer")
    temp = st.slider("Temperature (T)", min_value=0.01, max_value=2.0, value=0.7, step=0.05, key="temp_slider")
    top_p = st.slider("Top-P (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key="topp_slider")

    st.markdown(
        f"""
* **When $T \\to 0$ (e.g. 0.05)**: The highest logit dominates with probability $\\approx 1.0$. (Greedy Argmax Decoding).
* **When $T = 0.7$ (Balanced)**: Moderate variation while preserving coherence.
* **When $T > 1.5$ (High Entropy)**: Distribution flattens out, giving obscure low-probability tokens high likelihood (Nonsense / Hallucination).
* **Top-P = {top_p}**: Only the smallest subset of candidate tokens whose cumulative probability $\\ge {top_p}$ are considered for sampling.
"""
    )

with col_t2:
    candidate_tokens = ["Paris", "the", "Rome", "a", "France", "banana"]
    raw_logits = np.array([12.0, 10.5, 9.8, 8.5, 7.2, 2.0])

    scaled_logits = raw_logits / max(temp, 1e-4)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    df_probs = pd.DataFrame({"Token": candidate_tokens, "Probability": probs})
    fig_probs = px.bar(
        df_probs,
        x="Token",
        y="Probability",
        title=f"Next Token Probability Distribution (T = {temp})",
        color="Probability",
        color_continuous_scale="Viridis",
    )
    fig_probs.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_probs, use_container_width=True)

st.divider()

# ==============================================================================
# 6. Model Parameters & Quantization
# ==============================================================================
st.header("6. Model Parameters & Quantization — Dense vs MoE Architectures", divider="rainbow")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        """
### Dense vs Mixture-of-Experts (MoE)

* **Dense Models (e.g., Llama 3 70B, GPT-3 175B)**:
  - Every single parameter (all 70 billion weights) is activated for every single token processed.
  - Requires immense compute per token, but structurally straightforward.

* **Sparse MoE Models (e.g., Mixtral 8x7B, DeepSeek-V3, GPT-4)**:
  - The feed-forward network (FFN) layers are split into multiple specialized "expert" subnetworks.
  - A **Router Gate** dynamically routes each token to only the top-2 or top-8 experts.
  - *Example*: DeepSeek-V3 has **671B total parameters**, but only activates **37B parameters per token**, offering frontier-class intelligence at a fraction of the inference cost!
"""
    )

with col_p2:
    st.markdown(
        """
### Weight Precision & Quantization

A parameter is physically stored as a floating-point number. Quantization reduces the bit-width of weights to run large models on consumer GPUs:
"""
    )
    quant_df = pd.DataFrame(
        {
            "Precision": ["FP32 (Full)", "FP16 / BF16 (Half)", "INT8 (Quantized)", "INT4 (Quantized)"],
            "Bytes per Weight": ["4 Bytes", "2 Bytes", "1 Byte", "0.5 Bytes"],
            "VRAM for 70B Model": ["~280 GB", "~140 GB", "~70 GB", "~35–40 GB"],
            "Hardware Required": ["4x A100 (80GB)", "2x A100 (80GB)", "1x A100 (80GB)", "2x RTX 3090 / 4090"],
            "Quality Degradation": ["Baseline (0%)", "Negligible (<0.1%)", "Minimal (<0.5%)", "Slight (~1–2%)"],
        }
    )
    st.dataframe(quant_df, hide_index=True, use_container_width=True)

st.divider()

# ==============================================================================
# 7. Determinism & GPU Hardware
# ==============================================================================
st.header("7. Deterministic vs Probabilistic Behavior in Production", divider="rainbow")

st.markdown(
    """
### The Classic Interview Question:
> *"If I set Temperature = 0.0 and Top-P = 1.0, is the LLM guaranteed to produce identical outputs every single time?"*
"""
)

col_det1, col_det2 = st.columns(2)

with col_det1:
    st.error(
        """
#### ❌ The Common Naive Assumption:
*"Yes, because greedy decoding selects the single argmax token with highest probability every time, making the mathematical function deterministic."*
        """
    )

with col_det2:
    st.success(
        """
#### ✅ The Senior AI Engineer Reality:
*"In production cloud clusters (OpenAI, Anthropic), **No**. Even at Temperature = 0, responses can occasionally diverge due to GPU floating-point non-associativity and dynamic batching."*
        """
    )

st.markdown(
    """
#### 🔬 The 3 Reasons Why Temperature=0 is Non-Deterministic:
1. **Floating-Point Non-Associativity on GPUs**:
   - In computer architecture, $(a + b) + c \\ne a + (b + c)$ for 16-bit floating point numbers (FP16/BF16) due to rounding errors.
   - When matrix multiplications are split across hundreds of GPU tensor cores asynchronously, the order of additions changes slightly depending on hardware race conditions.
   - If two top candidate tokens have nearly identical logits (e.g. `14.500001` vs `14.500000`), a tiny rounding flip can swap the winner!
2. **Dynamic Request Batching & KV-Cache Sharing**:
   - Modern inference engines (vLLM, TensorRT-LLM) dynamically batch your prompt with other concurrent users' requests. Different batch alignments change kernel execution order.
3. **MoE Routing Tied-Score Perturbations**:
   - In Mixture of Experts, if routing tokens across experts has borderline scores, a micro-perturbation can send a token to an entirely different expert!

💡 **How to maximize determinism**: Use the API's `seed` parameter (OpenAI `seed=1234`), use greedy decoding (`temperature=0.0`), and lock system prompts to strict JSON schemas.
"""
)

st.divider()

# ==============================================================================
# 8. How Instructions Influence Model Behavior
# ==============================================================================
st.header("8. How Instructions Steer Latent Attention Weights", divider="rainbow")

col_st1, col_st2 = st.columns(2)

with col_st1:
    st.markdown(
        """
### 🎯 Attention Steering Mechanics
When you supply an instruction like *"You are a senior physician. Respond concisely in bullet points."*:
1. **Prefix Attention Priming**: The instruction tokens become the first elements in the Key-Value (KV) cache. Every subsequent token calculates dot-product attention against these tokens.
2. **Subnetwork Masking**: The tokens activate the medical semantic subnetwork learned during pre-training, suppressing informal or conversational probability distributions.
3. **Delimiter Hygiene**: Foundation models are trained heavily on Markdown headers (`### Instructions`), XML tags (`<context>...</context>`), and YAML boundaries. Using explicit delimiters prevents user inputs from leaking into task instructions.
"""
    )

with col_st2:
    st.markdown("#### ⚡ Negative Constraints vs Positive Commands")
    st.warning(
        """
**Why Negative Prompting Often Fails:**
* Prompt: *"Write a product review. Do NOT mention the price or shipping time."*
* What the model sees: The tokens `price` and `shipping` are explicitly introduced into the attention context, increasing the likelihood that attention heads attend to those semantic concepts!
        """
    )
    st.success(
        """
**The Engineered Alternative (Positive Framing):**
* Prompt: *"Focus exclusively on build quality, ergonomic design, and battery endurance."*
* Framing requirements positively conditions the probability mass onto desired target tokens without priming forbidden concepts.
        """
    )

st.divider()

# ==============================================================================
# Technical Interview Questions & Flashcards
# ==============================================================================
st.header("🎯 Technical Interview Preparation & Active Recall", divider="rainbow")

t_q1, t_q2, t_q3 = st.tabs([
    "❓ Technical Interview Questions",
    "⚠️ Interview Traps & Pitfalls",
    "📇 Active Recall Flashcards",
])

with t_q1:
    with st.expander("Q1: Explain the difference between BPE Tokenization and character-level encoding.", expanded=True):
        st.markdown(
            """
**Answer:**  
Byte-Pair Encoding (BPE) is a subword tokenization algorithm that begins with individual characters and iteratively merges the most frequent pairs of adjacent bytes or characters in a training corpus into unified vocabulary tokens. 
Character-level encoding treats every individual character as an atomic token, which avoids out-of-vocabulary (OOV) errors but inflates sequence lengths by 4x–5x, causing quadratic $O(N^2)$ attention computational blowups and limiting effective context windows. 
BPE strikes the optimal balance between sequence compression and vocabulary size, allowing common words to occupy a single token while breaking rare words into recognizable subwords.
"""
        )

    with st.expander("Q2: Why does an LLM struggle to count the number of specific letters in a word?", expanded=False):
        st.markdown(
            """
**Answer:**  
Large Language Models process text as discrete subword token vectors rather than sequences of raw characters. 
For example, the word "strawberry" is tokenized as atomic tokens (e.g. `["straw", "berry"]`), meaning the internal transformer attention heads see high-level semantic embeddings rather than individual 'r' characters. 
Unless instructed to decompose the string into separated characters or letters beforehand, the model must guess character frequencies from statistical co-occurrence rather than explicit inspection.
"""
        )

    with st.expander("Q3: What causes the 'Lost in the Middle' problem in large context window models?", expanded=False):
        st.markdown(
            """
**Answer:**  
The 'Lost in the Middle' phenomenon refers to the empirical finding that decoder-only models exhibit a U-shaped accuracy curve across long context windows, successfully retrieving facts placed at the beginning (primacy effect) or end (recency effect) while suffering up to a 50% drop in retrieval accuracy for facts located in the middle. 
This occurs because positional encodings (like RoPE) and autoregressive attention distributions tend to focus attention weights on recent tokens and initial system prefixes, allowing intermediate document chunks to experience attention dilution.
"""
        )

    with st.expander("Q4: Why are LLM output tokens significantly more expensive and slower than input tokens?", expanded=False):
        st.markdown(
            """
**Answer:**  
Input tokens are processed during the 'prefill' phase, which is highly parallelizable because all tokens are known simultaneously, enabling GPUs to saturate matrix multiplication tensor cores at maximum throughput. 
Output tokens are generated during the 'decoding' phase, which is strictly sequential and autoregressive—each token requires sampling before the next can begin. 
Because sequential decoding requires reloading gigabytes of model weights from High Bandwidth Memory (HBM) to GPU SRAM for every single generated token, it is memory-bandwidth bound, commanding higher latency and 3x–5x higher API pricing.
"""
        )

with t_q2:
    with st.expander("Trap 1: 'Does setting Temperature = 0 make an API call 100% deterministic?'", expanded=True):
        st.markdown(
            """
- **Naive Answer**: *"Yes, temperature 0 means greedy argmax decoding, so the model must always pick the highest probability token."*
- **Winning Response**: *"Mathematically yes, but in production distributed GPU environments, No. Due to non-associativity in parallel floating-point arithmetic (FP16/BF16), dynamic request batching, and mixture-of-experts routing boundary conditions, logit values can experience micro-perturbations on the order of $10^{-6}$. If two tokens are closely tied, this perturbation can flip the sampled token, creating cascading non-determinism. To maximize determinism, one should pair Temperature=0 with fixed seed parameters."*
"""
        )

    with st.expander("Trap 2: 'Can we fix context limits simply by increasing the context window to 10M tokens?'", expanded=False):
        st.markdown(
            """
- **Naive Answer**: *"Yes, once models have 10M token context windows, we will no longer need RAG or vector databases."*
- **Winning Response**: *"No, nominal context window length does not equal effective retrieval capability. Large context windows suffer from latency scaling, massive KV-cache VRAM memory footprints, high cost per call, and the 'lost-in-the-middle' retrieval degradation. RAG remains crucial for pinpoint precision, fact verification, privacy access control, and cost minimization."*
"""
        )

with t_q3:
    fc_cols = st.columns(3)
    with fc_cols[0]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**BPE Ratio**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** ~0.75 words per token (100 tokens $\\approx$ 75 words). Common words are 1 token; rare words and code are fragmented.")

    with fc_cols[1]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Temperature Math**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** $P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$. Lower T sharpens peak probabilities; Higher T flattens distribution toward uniform entropy.")

    with fc_cols[2]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prefill vs Decoding**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Prefill (Input) is compute-bound & parallel; Decoding (Output) is memory-bandwidth bound & strictly sequential.")

st.divider()

# ==============================================================================
# Summary & Self-Assessment Checklist
# ==============================================================================
st.header("📋 Lesson 1.2 Summary & Checklist", divider="rainbow")

col_sum1, col_sum2 = st.columns(2)
with col_sum1:
    st.markdown(
        """
### Key Takeaways:
1. **Tokens $\\ne$ Words**: BPE divides text into statistical chunks; tokenization boundaries dictate why models fail at letter counting and multi-digit arithmetic.
2. **Context Attention**: Self-attention is $O(N^2)$; facts in the middle suffer from the U-shaped recall degradation.
3. **Sampling Math**: Temperature scales logits prior to softmax; $T=0$ maximizes greedy selection but is vulnerable to GPU floating-point non-determinism.
4. **Input vs Output**: Output tokens are sequential and 3x–5x more expensive than parallel input tokens.
"""
    )

with col_sum2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain why LLMs fail at character counting due to BPE subwords.", value=True, key="chk_1_2_1")
    st.checkbox("I can write the mathematical formula for temperature-scaled softmax.", value=True, key="chk_1_2_2")
    st.checkbox("I can describe the 'Lost in the Middle' effect and how to structure prompts defensively.", value=True, key="chk_1_2_3")
    st.checkbox("I can explain why temperature 0 is not 100% deterministic on GPU clusters.", value=True, key="chk_1_2_4")
