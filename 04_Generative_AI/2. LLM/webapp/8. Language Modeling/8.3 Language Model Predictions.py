"""
8.3 Language Model Predictions
==============================
Interactive Master Lesson exploring Logits, Softmax Normalization, Probability Distributions,
Top-k, Top-p (Nucleus) Sampling, Temperature Scaling, and Modern Decoding Strategies.
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
    page_title="8.3 Language Model Predictions - LLM Architecture",
    page_icon="🎲",
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
st.title("🎲 8.3 Language Model Predictions")
st.caption(
    "Chapter 8: Language Modeling | Module 3 of 4"
)

st.markdown(
    """
Once a causal language model computes final hidden states, transforming those vectors into
coherent, human-like text requires navigating the **decoding and prediction pipeline**.
From unnormalized **logits** to probability densities shaped by **temperature**, **Top-k**,
**Top-p (Nucleus)**, and modern **Min-p** sampling, the choice of sampling strategy dictates
whether a model hallucinates, loops in repetitive degeneration, or produces razor-sharp reasoning.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & DECODING PIPELINE
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & Generation Decoding Pipeline", divider="rainbow")

st.markdown(
    """
### The Logit-to-Token Transformation Lifecycle
During each autoregressive generation step, the model executes a deterministic forward pass to produce logits,
followed by stochastic or deterministic logit transformations before sampling the final token ID.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "decoding_pipeline_flow",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("h", "Context Vector\\nh_t in R^[d]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("lm_head", "Unembedding Layer\\nW_u in R^[V, d]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("raw_logits", "Raw Logits\\nz in R^[V]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("temp", "Temperature Scaling\\nz / T", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("filter", "Filtering Masks\\nTop-k, Top-p, Min-p", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("softmax", "Normalized Softmax\\nProbabilities P in [0, 1]", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("sample", "Sampler\\nMultinomial / Greedy", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("token", "Next Token ID\\nx_{t+1}", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("h", "lm_head")
arch_dot.edge("lm_head", "raw_logits")
arch_dot.edge("raw_logits", "temp")
arch_dot.edge("temp", "filter")
arch_dot.edge("filter", "softmax")
arch_dot.edge("softmax", "sample")
arch_dot.edge("sample", "token")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **The Death of Pure Greedy Decoding**: Greedy search (`argmax`) leads to repetitive loops ("I think that I think that...") and dry, robotic text because high-likelihood paths trap models in local probability sinks.
* **The Rise of Min-P Sampling**: While Top-p truncates based on cumulative probability mass, **Min-p** (introduced in 2024) dynamically filters tokens whose probability is below a fraction `p_min` of the top token's probability (`P(token) < p_min * P(top)`). When the top token has 95% confidence, only tokens > 4.75% survive; when the distribution is flat, hundreds of plausible tokens survive.
* **Focal Logit Soft-Capping**: Models like Gemma 2 and Grok apply `tanh` soft-capping to prevent runaway logit magnitudes before Softmax: `logits = cap * tanh(logits / cap)`.
"""
)

# Decoding Strategies Comparison Table
sampling_df = pd.DataFrame(
    {
        "Strategy": ["Greedy Search", "Pure Random Sampling", "Top-k Sampling", "Top-p (Nucleus)", "Min-P Sampling"],
        "Mechanism": ["argmax(logits)", "Sample from full Softmax", "Keep top K tokens, mask rest", "Keep smallest set with sum(P) >= p", "Keep tokens with P >= p_min * P_max"],
        "Repetition Risk": ["Extremely High (Loops)", "Zero", "Low (if k >= 40)", "Low", "Extremely Low"],
        "Hallucination Risk": ["Low (Factual/Rigid)", "Extremely High (Wild)", "Moderate (Tail leak)", "Low", "Minimal (Context-Adaptive)"],
        "Ideal Use Case": ["Code, Math, Exact Extraction", "Artistic brainstorming", "Balanced dialogue", "Standard conversational LLMs", "Modern frontier creative writing"],
    }
)
st.dataframe(sampling_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: LOGITS & UN-NORMALIZED OUTPUT PROJECTIONS
# -----------------------------------------------------------------------------
st.header("2. Logits & The Geometry of Un-normalized Output Projections", divider="rainbow")

st.markdown(
    """
A **logit** is an unnormalized scalar score assigned to each token in the vocabulary.
For a model with hidden vector `h_t` and unembedding matrix `W_u`:

```text
z_v = W_u[v] * h_t + b_v
```

### Properties of Logits
1. **Unbounded Real Numbers**: Logits range from `-infinity` to `+infinity`.
2. **Relative Distances Matter**: Adding a constant `c` to all logits does not change the resulting Softmax distribution:
   `Softmax(z + c) = Softmax(z)`.
   What governs token probabilities is strictly the relative differences `(z_i - z_j)`.
3. **Logit Magnitudes**: Typically, trained models produce logits with standard deviations between 5.0 and 25.0. If logits explode to > 100.0, the Softmax distribution degenerates into a one-hot delta function.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: SOFTMAX & NUMERICAL NORMALIZATION MECHANICS
# -----------------------------------------------------------------------------
st.header("3. Softmax & Numerical Normalization Mechanics", divider="rainbow")

st.markdown(
    """
The Softmax function exponentiates logits and normalizes them by their sum:

```text
P(x = v) = exp(z_v) / SUM_{j=1}^{V} exp(z_j)
```

### The Exponentiation Amplifier
Because `exp(x)` grows exponentially:
* A logit difference of `+1.0` between two tokens means the higher token is `e^1.0 = 2.72x` more likely.
* A logit difference of `+3.0` makes the token `e^3.0 = 20.1x` more likely.
* A logit difference of `+7.0` makes the token `e^7.0 = 1096.6x` more likely!

Small shifts in dot-product projection `W_u[v] * h_t` dramatically swing the probability distribution.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: PROBABILITY DISTRIBUTION PROPERTIES
# -----------------------------------------------------------------------------
st.header("4. Probability Distribution Properties (Entropy & Diversity)", divider="rainbow")

st.markdown(
    """
The shape of the vocabulary distribution reflects the model's contextual certainty:
"""
)

col_ent1, col_ent2 = st.columns(2)
with col_ent1:
    st.subheader("Low Entropy (High Certainty)")
    st.markdown(
        """
        * **Context**: `"The president of the United [States]"`
        * **Distribution**: A single towering peak (`P("States") = 0.98`), followed by an exponentially decaying flat tail.
        * **Information Content**: Nearly 0 nats of entropy. The model has almost zero doubt about the continuation.
        """
    )
with col_ent2:
    st.subheader("High Entropy (Open-Ended / Ambiguous)")
    st.markdown(
        """
        * **Context**: `"Suddenly, behind the old abandoned door, there appeared a [giant / ghost / shadow / light]"`
        * **Distribution**: Diffuse probability mass spread across dozens of plausible creative candidates.
        * **Information Content**: High Shannon entropy `H = - sum P log P`.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 5: TOP-K FILTERING & RANK TRUNCATION
# -----------------------------------------------------------------------------
st.header("5. Top-k Filtering & Rank Truncation (Fan et al., 2018)", divider="rainbow")

st.markdown(
    """
Top-k sampling filters the vocabulary to only the top `k` highest-probability candidates,
setting all other logits to `-infinity`:

```text
Logits_filtered[v] = z_v   if rank(v) <= k
Logits_filtered[v] = -inf  otherwise
```

### The Architectural Limitation of Fixed K
* **In Low-Entropy Contexts**: If `k = 50`, but there is only 1 valid continuation (`"Paris"` after `"capital of France is"`), the model is forced to keep 49 incorrect, nonsensical tail tokens in the candidate pool. If random sampling picks candidate #32, the model generates an absurd hallucination.
* **In High-Entropy Contexts**: If `k = 50`, but there are 200 equally valid descriptive adjectives, Top-k truncates 150 viable words, artificially restricting vocabulary diversity.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: TOP-P (NUCLEUS) SAMPLING & ADAPTIVE CUTOFFS
# -----------------------------------------------------------------------------
st.header("6. Top-p (Nucleus) Sampling & Adaptive Dynamic Cutoffs (Holtzman et al., 2019)", divider="rainbow")

st.markdown(
    """
**Top-p (Nucleus) sampling** solves the fixed-window flaw of Top-k by dynamically expanding
or contracting the candidate pool based on cumulative probability mass:

```text
V^(p) = smallest subset such that SUM_{v in V^(p)} P(v) >= p
```
Where `p in (0, 1]` (commonly set to `0.9` or `0.95`).

### Dynamic Context Adaptation
1. **When Certain (Low Entropy)**: If `P("Paris") = 0.96`, then `V^(p=0.9)` contains **only 1 token** (`{"Paris"}`). The model behaves like greedy search and avoids tail errors.
2. **When Uncertain (High Entropy)**: If 80 diverse tokens share small probabilities summing to `0.90`, all 80 tokens remain eligible for sampling.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: TEMPERATURE SCALING & DISTRIBUTION SHARPENING
# -----------------------------------------------------------------------------
st.header("7. Temperature Scaling & Distribution Sharpening Dynamics", divider="rainbow")

st.markdown(
    """
Temperature `T > 0` scales raw logits prior to Softmax normalization:

```text
P_T(v) = exp(z_v / T) / SUM_{j=1}^{V} exp(z_j / T)
```

### Mathematical Regimes of Temperature
* **T -> 0 (Freezing / Greedy)**: Logit differences are magnified toward infinity. The highest logit dominates with probability `1.0`. The output becomes fully deterministic.
* **T = 1.0 (Standard)**: The exact mathematical probability distribution output by the model.
* **T > 1.0 (Heating / High Entropy)**: Flattens logit differences. Rare tokens receive elevated probability.
* **T -> infinity (Pure Uniform)**: All tokens approach equal probability `1 / V`, resulting in completely random character salad.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: STOCHASTIC SAMPLING ALGORITHMS
# -----------------------------------------------------------------------------
st.header("8. Stochastic Sampling Algorithms (Greedy vs. Multinomial vs. Min-P)", divider="rainbow")

st.markdown(
    """
### Min-P Sampling: The Modern Contender
Introduced in 2024, **Min-p** sampling sets a dynamic floor relative to the top candidate:

```text
Threshold = p_min * max_v P(v)
Keep tokens where P(v) >= Threshold
```
* If the top token has probability `0.80` and `p_min = 0.05`, any token with `P < 0.04` is pruned.
* If the top token has probability `0.10` (uncertain), tokens down to `0.005` survive.
Min-p eliminates the unnatural cutoff cliff of Top-p and preserves creative expressiveness without risking degenerated low-probability tail artifacts.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: PRODUCTION-GRADE AUTOREGRESSIVE SAMPLING KERNEL
# -----------------------------------------------------------------------------
st.header("9. Production-Grade Autoregressive Sampling Kernel (LogitProcessor Pipeline)", divider="rainbow")

st.markdown(
    """
A complete, vectorized PyTorch implementation of temperature scaling, Top-k, Top-p, and Min-p logit filtering:
"""
)

code_snippet = """import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
) -> torch.Tensor:
    # logits shape: [batch_size, vocab_size]
    
    # 1. Greedy Shortcut
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)
        
    # 2. Temperature Scaling
    scaled_logits = logits / temperature

    # 3. Top-k Filtering
    if top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        # Find k-th largest value
        kth_vals = torch.topk(scaled_logits, top_k)[0][..., -1, None]
        scaled_logits = torch.where(scaled_logits < kth_vals, torch.full_like(scaled_logits, -float("Inf")), scaled_logits)

    # Convert to probabilities for top-p and min-p filtering
    probs = F.softmax(scaled_logits, dim=-1)

    # 4. Min-p Filtering
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True)[0]
        min_threshold = p_max * min_p
        scaled_logits = torch.where(probs < min_threshold, torch.full_like(scaled_logits, -float("Inf")), scaled_logits)
        probs = F.softmax(scaled_logits, dim=-1)

    # 5. Top-p (Nucleus) Filtering
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter removed indices back to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, -float("Inf"))
        probs = F.softmax(scaled_logits, dim=-1)

    # 6. Multinomial Sampling
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 10: INTERACTIVE SAMPLING WORKBENCH
# -----------------------------------------------------------------------------
st.header("10. Interactive Sampling Workbench: Temperature, Top-k, and Top-p Simulator", divider="rainbow")

st.markdown(
    """
Experiment with how different sampling parameters transform raw logits into active candidate distributions.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    workbench_temp = st.slider("Temperature (T)", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    workbench_top_k = st.slider("Top-k Threshold", min_value=1, max_value=8, value=5)
with col_ctrl2:
    workbench_top_p = st.slider("Top-p (Nucleus) Threshold", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
    workbench_min_p = st.slider("Min-p Factor", min_value=0.0, max_value=0.3, value=0.05, step=0.01)

# Toy sample vocabulary
sample_tokens = ["learning", "models", "intelligence", "systems", "networks", "training", "compute", "parameters"]
sample_logits = np.array([6.5, 5.8, 5.2, 4.3, 3.7, 3.1, 2.0, 1.2])

# Step 1: Temperature
t_logits = sample_logits / workbench_temp

# Step 2: Top-k
sorted_indices = np.argsort(t_logits)[::-1]
k_mask = np.zeros_like(t_logits, dtype=bool)
k_mask[sorted_indices[:workbench_top_k]] = True
t_logits = np.where(k_mask, t_logits, -1e9)

# Step 3: Probabilities
exp_l = np.exp(t_logits - np.max(t_logits))
probs = exp_l / np.sum(exp_l)

# Step 4: Min-P
p_max = np.max(probs)
min_p_mask = probs >= (p_max * workbench_min_p)
probs = np.where(min_p_mask, probs, 0.0)

# Step 5: Top-P
sorted_p_idx = np.argsort(probs)[::-1]
sorted_p = probs[sorted_p_idx]
cum_p = np.cumsum(sorted_p)
cutoff_idx = np.searchsorted(cum_p, workbench_top_p) + 1
keep_p_idx = sorted_p_idx[:cutoff_idx]

final_mask = np.zeros_like(probs, dtype=bool)
final_mask[keep_p_idx] = True
final_probs = np.where(final_mask, probs, 0.0)
if np.sum(final_probs) > 0:
    final_probs = final_probs / np.sum(final_probs)

sim_df = pd.DataFrame(
    {
        "Token": sample_tokens,
        "Raw Logit": sample_logits,
        "Temp Scaled": np.round(sample_logits / workbench_temp, 2),
        "Active Candidate": final_mask,
        "Final Probability": np.round(final_probs, 4),
    }
).sort_values(by="Final Probability", ascending=False)

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.subheader("Filtered Candidate Pool")
    st.dataframe(sim_df, use_container_width=True)
with col_s2:
    st.subheader("Active Sampling Distribution")
    active_df = sim_df[sim_df["Active Candidate"]].copy()
    st.bar_chart(active_df.set_index("Token")["Final Probability"])

# -----------------------------------------------------------------------------
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why does pure Greedy Decoding lead to degenerate repetitive loops in LLMs?"):
    st.markdown(
        """
        **Answer:**
        Greedy decoding always selects the token with the highest logit.
        If a model outputs a common phrase, the subsequent highest-probability token is often part of that exact same phrase pattern. Once a phrase repeats once, the self-attention mechanism attends heavily to the previous occurrence, artificially boosting the logits of the repeated tokens even higher. This creates a self-reinforcing positive feedback loop that the model cannot escape without stochastic sampling or repetition penalties.
        """
    )

with st.expander("Q2: What is the primary operational advantage of Top-p (Nucleus) over Top-k sampling?"):
    st.markdown(
        """
        **Answer:**
        Top-k uses a static token budget `k`.
        When the model is confident (low entropy), `k` is too large, retaining absurd tail tokens that cause hallucinations. When the model is uncertain (high entropy), `k` is too small, cutting off valid candidate words.
        Top-p adaptively resizes the candidate pool: it contracts to 1 or 2 tokens when the model is confident, and expands to dozens of tokens when the context is genuinely ambiguous.
        """
    )

with st.expander("Q3: How does Min-p sampling address the tail truncation problem in Top-p?"):
    st.markdown(
        """
        **Answer:**
        In Top-p, if cumulative mass reaches `0.9001` right before token #4, token #5 is cut off completely, even if its probability was virtually identical to token #4.
        Min-p sets a relative threshold: `P(v) >= p_min * P_max`.
        Tokens are evaluated against the leader token rather than an arbitrary cumulative sum cliff, preventing abrupt cutoffs while cleanly pruning truly irrelevant low-probability noise.
        """
    )

with st.expander("Q4: Why must Temperature be applied BEFORE Top-p filtering rather than after?"):
    st.markdown(
        """
        **Answer:**
        Top-p computes cumulative sums over the normalized probability distribution.
        Temperature alters the entropy of the probability distribution: low temperature concentrates probability mass into fewer tokens, causing the Top-p cumulative threshold to be reached much faster. High temperature flattens the distribution, causing Top-p to include more tokens. Applying temperature after Top-p would defeat this adaptive behavior.
        """
    )

with st.expander("Q5: What is repetition penalty and how is it implemented on logits?"):
    st.markdown(
        """
        **Answer:**
        A repetition penalty discounts logits for tokens that have already appeared in the generated context.
        For token `v` that appeared previously:
        * If `z_v > 0`: `z_v = z_v / penalty` (e.g., penalty = 1.15).
        * If `z_v < 0`: `z_v = z_v * penalty`.
        This gently suppresses repeated tokens without completely forbidding them (unlike strict n-gram blocking).
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Setting Temperature to Zero and Using Multinomial Sampling**
        *Root Cause*: Passing `temperature=0.0` into `torch.multinomial` causes division by zero (`z / 0 = inf`), resulting in NaNs and crashing the inference engine.
        *Production Fix*: Intercept `temperature <= 1e-4` and branch directly to deterministic `torch.argmax(logits)`.
        """
    )
    st.warning(
        """
        **Trap 2: Excessive Repetition Penalties Destroying Structured Formats**
        *Root Cause*: Setting `repetition_penalty > 1.3` when generating JSON or Markdown code.
        *Production Fix*: Severe repetition penalties penalize valid repeated syntax like `\"` or `}`. Use constrained grammars (Outlines/vLLM) instead of high repetition penalties.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Top-p Cumulative Mask Shift Omission**
        *Root Cause*: Masking tokens where `cumsum > top_p` without shifting the mask by 1 index to the right.
        *Production Fix*: Always shift the mask right: the first token that pushes cumulative probability *over* `top_p` must be included so the sum meets or exceeds `p`.
        """
    )
    st.warning(
        """
        **Trap 4: Missing Logit Soft-Capping in High-Temp Sampling**
        *Root Cause*: At temperatures > 1.2, extreme outlier logits can destabilize the Softmax calculation in FP16 precision.
        *Production Fix*: Clamp raw logits to `[-50.0, +50.0]` or utilize Gemma 2 `tanh` soft-capping before scaling.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Temperature = 0 Effect"):
        st.write("What decoding algorithm is mathematically equivalent to Temperature = 0?")
        st.info("Greedy decoding (argmax of logits).")

    with st.expander("⚡ Flashcard 2: Top-p Default Range"):
        st.write("What is the typical industry default value for Top-p?")
        st.info("0.90 to 0.95 (nucleus sampling).")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Min-P Core Formula"):
        st.write("What condition must a token satisfy to survive Min-p filtering?")
        st.info("P(token) >= p_min * P(top_token).")

    with st.expander("⚡ Flashcard 4: Softmax Shift Invariance"):
        st.write("Why can we subtract max(logits) before computing Softmax?")
        st.info("Softmax is translation invariant: Softmax(z - c) = Softmax(z) for any constant c.")

# -----------------------------------------------------------------------------
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand how raw logits are transformed into next-token predictions.",
    "I can mathematically explain why Temperature scales the sharpness of the Softmax distribution.",
    "I understand the difference between static Top-k and dynamic Top-p (Nucleus) candidate filtering.",
    "I know how Min-p sampling dynamically prunes tail tokens relative to the leader token.",
    "I can explain why greedy decoding frequently triggers degenerate repetitive loops.",
    "I know how to implement vectorized Top-k and Top-p filtering in PyTorch.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Language Model Predictions and Modern Decoding Strategies."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
