"""
8.1 Language Model Fundamentals
===============================
Interactive Master Lesson exploring Sequence Probability, Conditional Probability,
Next-Token Prediction, and Autoregressive Vocabulary Distributions in Modern Foundation LLMs.
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
    page_title="8.1 Language Model Fundamentals - LLM Architecture",
    page_icon="🔮",
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
st.title("🔮 8.1 Language Model Fundamentals")
st.caption(
    "Chapter 8: Language Modeling | Module 1 of 4"
)

st.markdown(
    """
At its theoretical core, every modern decoder-only Large Language Model (from GPT-4 to Llama 3)
is an autoregressive probability estimator over discrete token sequences. By decomposing the joint
probability of text through the probabilistic chain rule, the task of complex language generation
reduces to iterative, conditional **next-token prediction** over an extensive vocabulary distribution.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE CORE FORMULATION
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Core Formulation", divider="rainbow")

st.markdown(
    """
### The Autoregressive Probability Pipeline
A decoder-only model takes a prompt sequence of token IDs, projects them into continuous embedding space,
processes them through stacked causal self-attention and MLP blocks, and projects the final hidden state
into a probability distribution over the vocabulary.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "lm_fundamentals_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("tokens", "Input Tokens\\n[x_1, x_2, ..., x_t]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("embed", "Token & Positional\\nEmbeddings", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("transformer", "Causal Transformer\\nL-Layers (Masked)", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("head", "LM Head Unembedding\\nLinear W_u in R^[d, V]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("softmax", "Softmax Normalizer\\nP(x_{t+1} | x_{<=t})", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("sampler", "Sampling / Search\\nTemperature, Top-p, Greedy", fillcolor="#7C3AED", fontcolor="#F8FAFC")

arch_dot.edge("tokens", "embed")
arch_dot.edge("embed", "transformer")
arch_dot.edge("transformer", "head", label="h_t in R^[d]")
arch_dot.edge("head", "softmax", label="Logits in R^[V]")
arch_dot.edge("softmax", "sampler", label="Distribution P")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Next-Token Prediction as Universal Pretraining**: Frontier models (GPT-4, Claude 3.5, Llama 3) rely on the exact same simple autoregressive cross-entropy objective on raw text. Emergent capabilities (reasoning, code synthesis, in-context translation) emerge directly from scaling compute and dataset size under this single objective.
* **Tied vs. Untied Embeddings**: While earlier models (GPT-2, T5) shared weights between input embeddings and the LM head (`W_emb = W_head^T`), modern 70B+ models frequently untie them or expand vocabulary sizes to 128k (Llama 3) and 150k (Qwen 2.5) to capture multilingual nuances without matrix parameter interference.
"""
)

# Contrastive Comparison Table
matrix_df = pd.DataFrame(
    {
        "Paradigm": ["N-gram Statistical LM", "Masked LM (BERT)", "Prefix LM (T5)", "Causal Decoder (Llama/GPT)"],
        "Sequence Probability": ["Markov Assumption P(w_t | w_{t-n+1:t-1})", "No True Joint Density P(W)", "Conditional P(Target | Source)", "Exact Joint Density P(x_1, ..., x_T)"],
        "Context Access": ["Fixed Window (n <= 5)", "Bidirectional Full Context", "Bidirectional Encoder + Causal Dec", "Strictly Causal Upper-Triangular Mask"],
        "Generation Mode": ["Stochastic Count Walking", "Iterative MCMC / Impractical", "Autoregressive Decoding", "Autoregressive Next-Token Sampling"],
        "Primary Use Case": ["Spellcheck, Simple Speech Rec", "Embedding Extraction, NLU", "Translation, Summarization", "Universal Foundation AI"],
    }
)
st.dataframe(matrix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: WHAT IS LANGUAGE MODELING?
# -----------------------------------------------------------------------------
st.header("2. What is Language Modeling? (Statistical to Neural Paradigm Shift)", divider="rainbow")

st.markdown(
    """
A **Language Model (LM)** is a statistical or neural system that assigns a probability distribution
over sequences of words or tokens. It answers fundamental queries:
1. *How natural or probable is a sequence?*
   `P("The astronaut landed on Mars") > P("Mars on landed astronaut The")`
2. *Given previous context, what token comes next?*
   `P("Paris" | "The capital of France is") >> P("Elephant" | "The capital of France is")`

### The Paradigm Shift
* **Statistical N-gram Era (1980s-2010s)**: Estimated probabilities through maximum likelihood frequency counts with smoothing (Kneser-Ney). Ssuffered from the curse of dimensionality: unseen 5-grams received zero probability.
* **Recurrent Neural Network Era (2014-2017)**: LSTMs and GRUs compressed arbitrary historical contexts into fixed hidden state vectors, eliminating Markov window limits but suffering from vanishing gradients over long spans.
* **Transformer Foundation Model Era (2017-Present)**: Decoupled context aggregation from recurrence using self-attention. Foundation models model joint probability distributions across millions of tokens with near-perfect in-context memory.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: PROBABILITY OF A SEQUENCE & THE CHAIN RULE
# -----------------------------------------------------------------------------
st.header("3. Joint Probability of a Sequence & The Chain Rule of Probability", divider="rainbow")

st.markdown(
    """
By the probabilistic chain rule, any joint probability distribution over `T` random variables
can be factored into an exact product of conditional probabilities without making any independence assumptions:

```text
P(x_1, x_2, ..., x_T) = P(x_1) * P(x_2 | x_1) * P(x_3 | x_1, x_2) ... P(x_T | x_1, ..., x_{T-1})
                      = PRODUCT_{t=1}^{T} P(x_t | x_{<t})
```

### Log-Space Computation for Numerical Stability
Multiplying hundreds of small decimal probabilities (`0.001 * 0.0004 * ...`) causes floating-point underflow.
In practice, all language models operate in log-probability space:

```text
log P(x_1, ..., x_T) = SUM_{t=1}^{T} log P(x_t | x_{<t})
```
Because `log(a * b) = log(a) + log(b)`, the joint sequence log-probability is the simple sum of individual conditional log-probabilities.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: CONDITIONAL PROBABILITY & CONTEXTUAL DEPENDENCIES
# -----------------------------------------------------------------------------
st.header("4. Conditional Probability & Contextual Dependencies", divider="rainbow")

st.markdown(
    """
The conditional probability `P(x_t | x_{<t})` models the likelihood of token `x_t` given all preceding context tokens `x_{<t} = [x_1, ..., x_{t-1}]`.

### Context Length & Information Accumulation
As context grows, conditional uncertainty diminishes:
* Given `x_1 = "The"`: Next token distribution is high-entropy (can be any noun or adjective: "dog", "company", "quick").
* Given `x_{<5} = "The capital of France is"`: Conditional distribution collapses into a sharp peak: `P("Paris" | x_{<5}) > 0.95`.

### Long-Range Syntactic and Semantic Agreement
The conditional distribution must enforce:
1. **Grammatical Agreement**: Subject-verb number agreement across dozens of intervening clauses (`"The keys that were left on the dusty wooden table [are / *is]"`).
2. **Coreference Resolution**: Pronoun gender and entity tracking across paragraphs.
3. **Factual and Algorithmic Consistency**: Variable bindings and mathematical computation.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: NEXT-TOKEN PREDICTION FORMULATION
# -----------------------------------------------------------------------------
st.header("5. Next-Token Prediction Formulation", divider="rainbow")

st.markdown(
    """
Next-token prediction formulates language generation as a multi-class classification problem.
At each position `t`:
1. The transformer processes the sequence `[x_1, ..., x_t]`.
2. The final hidden state vector `h_t in R^[d_model]` represents the contextual state at step `t`.
3. The linear LM head projects `h_t` to the vocabulary dimension `V`:
   ```text
   z_t = W_u * h_t + b_u,    where z_t in R^[V]
   ```
4. The unnormalized logits `z_t` are converted into normalized probabilities via Softmax:
   ```text
   P(x_{t+1} = w | x_{<=t}) = exp(z_t[w]) / SUM_{v=1}^{V} exp(z_t[v])
   ```
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: AUTOREGRESSIVE MODELING & CAUSAL MASKING
# -----------------------------------------------------------------------------
st.header("6. Autoregressive Modeling & Causal Masking", divider="rainbow")

st.markdown(
    """
**Autoregressive (AR)** modeling implies that the model's output at step `t` becomes part of the input conditioning context at step `t+1`.

### Training Efficiency: Parallel Pre-Computation
During training, we have the complete ground-truth target sequence. Because causal self-attention uses an **upper-triangular causal mask**:
* Token `x_1` attends only to `x_1`.
* Token `x_2` attends to `x_1, x_2`.
* Token `x_t` attends to `x_1, ..., x_t`.

This causal mask guarantees that computing predictions for **all `T` positions occurs simultaneously in a single forward pass**.
The model computes `T` separate classification losses in parallel without recurrent loops.

### Inference Bottleneck: Serial Autoregression
During generation (inference), the ground-truth future tokens are unknown.
Generating `K` tokens requires `K` sequential forward passes:
1. Pass prompt `[x_1, ..., x_t]` -> sample `x_{t+1}`.
2. Append `x_{t+1}` -> pass `[x_1, ..., x_{t+1}]` -> sample `x_{t+2}`.
3. Repeat until generating the `<|endoftext|>` token or reaching max length.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: TOKEN PROBABILITY & THE GEOMETRIC OUTPUT PROJECTION
# -----------------------------------------------------------------------------
st.header("7. Token Probability & The Geometric Output Projection", divider="rainbow")

st.markdown(
    """
### The Geometric Interpretation of Unembedding
The LM head weight matrix `W_u in R^[V, d_model]` can be viewed as an array of `V` distinct target vectors in the model's latent hidden space:

```text
z_t[v] = <W_u[v], h_t>
```

* Each row `W_u[v]` is the ideal prototype embedding for token `v`.
* The logit `z_t[v]` is the cosine-scaled inner product between the context representation `h_t` and prototype vector `W_u[v]`.
* When `h_t` aligns with a prototype vector, its logit surges, driving its probability toward 1.0.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: VOCABULARY PROBABILITY DISTRIBUTION & DENSE SOFTMAX DYNAMICS
# -----------------------------------------------------------------------------
st.header("8. Vocabulary Probability Distribution & Dense Softmax Dynamics", divider="rainbow")

st.markdown(
    """
The output of a language model is a dense probability vector of length `V` satisfying two core axioms:
1. `P(x = v) >= 0` for all `v in [1, V]`.
2. `SUM_{v=1}^{V} P(x = v) = 1.0`.

### Softmax Dynamics at Scale
In modern models, `V` is massive:
* GPT-2: `50,257`
* Llama 2: `32,000`
* Llama 3: `128,256`
* Gemma: `256,000`

Computing Softmax over a 256,000-dimensional vector for every token across batches of millions of tokens creates a severe GPU memory and memory bandwidth bottleneck.
Modern engines use online Softmax fusion (FlashInfer, vLLM) to perform Softmax normalization directly in GPU SRAM.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE AUTOREGRESSIVE SEQUENCE PREDICTOR IMPLEMENTATION
# -----------------------------------------------------------------------------
st.header("9. Production-Grade Autoregressive Sequence Predictor Implementation", divider="rainbow")

st.markdown(
    """
Vectorized PyTorch implementation demonstrating the causal next-token LM Head and sequence log-likelihood evaluation:
"""
)

code_snippet = """import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLMHead(nn.Module):
    # Causal LM Head projecting hidden states to vocabulary distributions
    def __init__(self, hidden_dim: int, vocab_size: int, tie_weights: bool = False, embedding_layer: nn.Embedding = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        if tie_weights and embedding_layer is not None:
            # Weight tying: reuse input embedding matrix
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.lm_head.weight = embedding_layer.weight
        else:
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # returns logits: [batch_size, seq_len, vocab_size]
        return self.lm_head(hidden_states)

def compute_sequence_log_likelihood(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    # Computes exact log P(x_1, ..., x_T) via cross-entropy sum
    # Shift logits and targets so token t predicts t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    
    # Compute per-token negative log likelihood
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    per_token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
    per_token_nll = per_token_nll.view(shift_targets.size())
    
    # Sequence log likelihood = - sum(NLL)
    seq_log_likelihood = -per_token_nll.sum(dim=-1)
    return seq_log_likelihood, per_token_nll
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE AUTOREGRESSIVE TOKEN PROBABILITY WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive Autoregressive Token Probability Workbench", divider="rainbow")

st.markdown(
    """
Simulate next-token vocabulary distributions and sequence joint log-likelihoods for candidate phrases.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    prompt_choice = st.selectbox(
        "Select Context Prompt",
        options=[
            "The capital of France is",
            "To be or not to be, that is the",
            "import numpy as",
            "The quick brown fox jumps over the",
        ],
        index=0,
    )
with col_ctrl2:
    temperature = st.slider("Sampling Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Toy candidate distributions
toy_distributions = {
    "The capital of France is": {
        "Paris": 8.5,
        "Lyon": 4.1,
        "Marseille": 3.8,
        "located": 3.2,
        "a": 2.5,
        "not": 1.2,
        "dog": -2.0,
    },
    "To be or not to be, that is the": {
        "question": 9.2,
        "answer": 4.0,
        "problem": 3.5,
        "truth": 2.1,
        "way": 1.8,
        "cat": -1.5,
    },
    "import numpy as": {
        "np": 10.4,
        "numpy": 3.2,
        "n": 2.1,
        "pd": 0.5,
        "math": -1.0,
    },
    "The quick brown fox jumps over the": {
        "lazy": 8.8,
        "sleeping": 5.2,
        "fence": 4.1,
        "dog": 3.5,
        "moon": 1.0,
    },
}

active_logits = toy_distributions[prompt_choice]
words = list(active_logits.keys())
raw_logits = np.array(list(active_logits.values()))

# Apply temperature
scaled_logits = raw_logits / temperature
exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
probs = exp_logits / np.sum(exp_logits)

dist_df = pd.DataFrame(
    {
        "Candidate Token": words,
        "Raw Logit": np.round(raw_logits, 2),
        "Scaled Logit (T)": np.round(scaled_logits, 2),
        "Probability P(x)": np.round(probs, 4),
        "Log Prob log P(x)": np.round(np.log(probs + 1e-9), 3),
    }
).sort_values(by="Probability P(x)", ascending=False)

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.subheader("Vocabulary Probability Distribution")
    st.dataframe(dist_df, use_container_width=True)
with col_d2:
    st.subheader("Top-1 Predicted Next Token")
    top_tok = dist_df.iloc[0]["Candidate Token"]
    top_prob = dist_df.iloc[0]["Probability P(x)"]
    st.metric("Predicted Token", f"'{top_tok}'", f"{top_prob * 100:.1f}% confidence")
    st.caption(f"Full sequence: '{prompt_choice} {top_tok}'")

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is joint sequence probability computed as a sum of log probabilities rather than a product of probabilities?"):
    st.markdown(
        """
        **Answer:**
        Probabilities lie in the range `[0, 1]`. When computing the joint probability of a 500-token sequence, multiplying 500 floating-point numbers smaller than 0.1 rapidly triggers arithmetic floating-point underflow (exceeding IEEE-754 precision).
        Converting to log-space transforms products into sums: `log P(X) = sum log P(x_t | x_{<t})`. Log probabilities are negative numbers that sum stably without underflow.
        """
    )

with st.expander("Q2: How does training an autoregressive model differ computationally from running inference?"):
    st.markdown(
        """
        **Answer:**
        * **Training (Parallel Teacher Forcing)**: The full ground-truth sequence is available upfront. Through causal self-attention masking, all `T` positions are computed concurrently in a single forward pass, yielding `O(1)` sequential steps on GPUs.
        * **Inference (Serial Autoregression)**: Future tokens are unknown. The model must run `T` distinct forward passes sequentially, each generating one token that is appended to the KV cache for the next step.
        """
    )

with st.expander("Q3: What is the effect of weight tying between input embeddings and the LM head?"):
    st.markdown(
        """
        **Answer:**
        Weight tying sets `W_head = W_emb^T`.
        * **Advantages**: Saves significant GPU parameter memory. For a 128k vocabulary and 4096 hidden dimension, each matrix consumes `128,000 * 4096 * 2 bytes = 1.05 GB`. Tying them saves 1 GB in memory and ensures input/output semantic consistency.
        * **Disadvantages**: Limits representational capacity. The optimal manifold for recognizing input tokens is not necessarily symmetric with the optimal manifold for generating token logits. Modern 70B+ models often untie them.
        """
    )

with st.expander("Q4: Can a Masked Language Model (BERT) compute the true joint probability P(x_1, ..., x_T)?"):
    st.markdown(
        """
        **Answer:**
        **No.** BERT models `P(x_mask | x_{unmasked})` under an assumed conditional independence between masked positions:
        `P(x_{mask1}, x_{mask2} | ...) = P(x_{mask1} | ...) * P(x_{mask2} | ...)`.
        Because BERT conditions on bidirectional context simultaneously, it violates the causal chain rule and cannot evaluate the true joint density of an arbitrary text sequence.
        """
    )

with st.expander("Q5: Why does next-token prediction lead to general reasoning abilities?"):
    st.markdown(
        """
        **Answer:**
        Predicting the next token across diverse internet text requires the model to implicitly compress the underlying physical, syntactic, and causal rules of the world that generated the text. To accurately predict the next token in a complex Python script or mathematical proof, the model must execute the algorithm or trace the proof internally.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Off-by-One Alignment Error in Training Loss**
        *Root Cause*: Passing unshifted logits `logits[:, :, :]` directly against `targets[:, :, :]` into cross-entropy loss, causing the model to learn the identity function (predicting token t given token t).
        *Production Fix*: Always shift tensors: `shift_logits = logits[:, :-1, :]` and `shift_targets = targets[:, 1:]`.
        """
    )
    st.warning(
        """
        **Trap 2: Logit Overflow in Float16 Softmax**
        *Root Cause*: If logits reach values > 65,504, `torch.exp(logits)` overflows in FP16 precision, producing `NaN` loss or garbage outputs.
        *Production Fix*: Always compute Softmax in FP32 precision, or subtract `max(logits)` prior to exponentiation (safe Softmax).
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Causal Mask Leakage in Custom Attention Kernels**
        *Root Cause*: Off-by-one error in causal mask indices that allows token `t` to attend to token `t+1`, allowing the model to cheat during training.
        *Production Fix*: Assert that the attention mask matrix has strictly `-inf` on all entries where `col > row`.
        """
    )
    st.warning(
        """
        **Trap 4: Vocabulary Size Incompatibility with Hardware Dimensions**
        *Root Cause*: Choosing a prime vocabulary size (e.g. 50,257 in GPT-2) prevents GPU Tensor Cores from aligning matrix multiplications along 8x or 64x boundaries.
        *Production Fix*: Pad vocabulary sizes to multiples of 64 or 128 (e.g., Llama 3 pads to 128,256).
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Chain Rule Exactness"):
        st.write("Does the chain rule of probability introduce any approximation error?")
        st.info("Zero error. The chain rule is an exact mathematical identity valid for any joint distribution.")

    with st.expander("⚡ Flashcard 2: Parallel Training"):
        st.write("Why can autoregressive models train in parallel despite generating sequentially?")
        st.info("Teacher forcing and upper-triangular causal attention masking allow all token positions to be evaluated concurrently.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Log-Likelihood Sign"):
        st.write("Why are sequence log-likelihoods always negative numbers?")
        st.info("Probabilities lie in (0, 1]. The natural logarithm of any number in this range is negative or zero.")

    with st.expander("⚡ Flashcard 4: Unembedding Matrix Shape"):
        st.write("What is the shape of the LM head projection matrix?")
        st.info("Shape: [hidden_dimension, vocab_size], mapping latent hidden vectors to vocabulary logits.")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the mathematical formulation of joint sequence probability via the chain rule.",
    "I can explain why next-token prediction acts as an effective universal pretraining objective.",
    "I understand the difference between parallel training via teacher forcing and serial inference.",
    "I know how the LM head un-embedding matrix projects hidden states into vocabulary logits.",
    "I can explain why log probabilities are used to prevent floating-point underflow.",
    "I understand how the causal attention mask prevents future information leakage during training.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Language Model Fundamentals and Autoregressive Probability Dynamics."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
