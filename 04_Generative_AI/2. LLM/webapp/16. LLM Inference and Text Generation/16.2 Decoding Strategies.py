import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.2 Decoding Strategies - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionDecodingEngine:
    """
    Self-contained educational decoding engine implementing:
    - Temperature scaling
    - Top-k truncation
    - Top-p (Nucleus) cumulative filtering
    - Min-p adaptive thresholding
    - Repetition penalty logit discounting
    """
    def __init__(self, vocab_size: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        self.vocab_tokens = [f"token_{i}" for i in range(vocab_size)]

    def apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_tokens: List[int],
        penalty: float = 1.2
    ) -> np.ndarray:
        """Multiplicatively penalizes tokens that have already been generated."""
        mod_logits = np.array(logits, dtype=np.float32)
        if penalty == 1.0 or not generated_tokens:
            return mod_logits

        for token_id in set(generated_tokens):
            if token_id < len(mod_logits):
                if mod_logits[token_id] > 0:
                    mod_logits[token_id] /= penalty
                else:
                    mod_logits[token_id] *= penalty
        return mod_logits

    def apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Scales logits by temperature factor."""
        if temperature <= 0.0:
            # Deterministic argmax indicator
            out = np.full_like(logits, -np.inf)
            out[np.argmax(logits)] = 0.0
            return out
        return logits / max(1e-4, temperature)

    def apply_top_k(self, logits: np.ndarray, k: int) -> np.ndarray:
        """Masks out all tokens below the top-k highest logits."""
        if k <= 0 or k >= len(logits):
            return logits
        mod_logits = np.array(logits, dtype=np.float32)
        # Find k-th largest value
        cutoff_idx = np.argsort(logits)[-k]
        cutoff_val = logits[cutoff_idx]
        mod_logits[logits < cutoff_val] = -np.inf
        return mod_logits

    def apply_top_p(self, probs: np.ndarray, p: float) -> np.ndarray:
        """Nucleus sampling: keeps smallest set of tokens whose cumulative prob exceeds p."""
        if p >= 1.0:
            return probs
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Determine cutoff
        cutoff_mask = cumulative_probs > p
        # Keep at least the top-1 token
        cutoff_mask[0] = False
        
        filtered_probs = np.array(probs, dtype=np.float32)
        filtered_indices_to_zero = sorted_indices[cutoff_mask]
        filtered_probs[filtered_indices_to_zero] = 0.0
        
        sum_p = np.sum(filtered_probs)
        return filtered_probs / sum_p if sum_p > 0 else probs

    def apply_min_p(self, probs: np.ndarray, min_p: float) -> np.ndarray:
        """Min-p sampling: scales cutoff relative to the top token probability."""
        if min_p <= 0.0:
            return probs
        p_top = np.max(probs)
        threshold = p_top * min_p
        
        filtered_probs = np.array(probs, dtype=np.float32)
        filtered_probs[filtered_probs < threshold] = 0.0
        sum_p = np.sum(filtered_probs)
        return filtered_probs / sum_p if sum_p > 0 else probs

    def sample_distribution(
        self,
        base_logits: np.ndarray,
        generated_tokens: List[int],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Runs the complete decoding filter pipeline and samples next token."""
        # Step 1: Repetition Penalty
        l_rep = self.apply_repetition_penalty(base_logits, generated_tokens, repetition_penalty)
        
        # Step 2: Temperature
        l_temp = self.apply_temperature(l_rep, temperature)
        
        if temperature <= 0.0:
            # Deterministic
            sampled_id = int(np.argmax(l_temp))
            probs = np.zeros_like(l_temp)
            probs[sampled_id] = 1.0
            return l_temp, probs, sampled_id

        # Step 3: Top-K
        l_k = self.apply_top_k(l_temp, top_k)

        # Softmax
        exp_l = np.exp(l_k - np.max(l_k))
        probs = exp_l / np.sum(exp_l)

        # Step 4: Top-P (Nucleus)
        probs_p = self.apply_top_p(probs, top_p)

        # Step 5: Min-P
        probs_final = self.apply_min_p(probs_p, min_p)

        # Step 6: Multinomial Sample
        sampled_id = int(self.rng.choice(len(probs_final), p=probs_final))
        return l_temp, probs_final, sampled_id

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("16.2 Decoding Strategies")
st.caption("Chapter 16: LLM Inference and Text Generation | Domain: 2. Large Language Models")

st.markdown("""
Once a language model's forward pass produces raw vocabulary logits, how does the system choose which token to emit?
The answer is defined by **Decoding Strategies**. 
Choosing the right decoding pipeline determines whether an LLM produces deterministic, syntax-perfect SQL code 
or imaginative, engaging creative writing.

In this lesson, you will master:
1. The mathematical hierarchy of sampling and search algorithms.
2. Greedy Decoding: deterministic argmax selection and its fatal degeneration loops.
3. Beam Search: breadth-first hypothesis tracking, beam width, and why modern generative models abandoned it.
4. Random Sampling: unconstrained multinomial selection and tail collapse.
5. Temperature Scaling: reshaping categorical distributions from uniform flat to delta peaks.
6. Top-K Truncation: restricting candidate token pools to a fixed integer bound.
7. Top-P (Nucleus) Sampling: dynamic probability mass thresholding.
8. Min-P Sampling: modern adaptive thresholding based on top token confidence.
9. Repetition Penalty: mathematically penalizing previously emitted token logits.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Probability Sampling & Decoding Topology", divider="rainbow")

st.markdown("""
Decoding transforms raw unnormalized logits through sequential filtering and scaling stages before token selection.
""")

decode_blueprint_dot = """
digraph DecodingTopology {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_logits {
        label = "Stage 1: Raw Logits";
        color = "#334155";
        fontcolor = "#94A3B8";
        RawLogits [label="Model Logits W_head @ h\\n[Vocab_Size = 128k]", fillcolor="#0F172A"];
        RepPenalty [label="Repetition Penalty\\nLogits / penalty on prior tokens", fillcolor="#0369A1"];
        TempScale [label="Temperature Scaling\\nLogits / T", fillcolor="#0369A1"];
        RawLogits -> RepPenalty -> TempScale;
    }

    subgraph cluster_filters {
        label = "Stage 2: Truncation Filters";
        color = "#334155";
        fontcolor = "#94A3B8";
        TopK [label="Top-K Filter\\nKeep top K logits, mask rest with -inf", fillcolor="#047857"];
        Softmax [label="Softmax Normalization\\nP = exp(l) / sum(exp(l))", fillcolor="#047857"];
        TopP [label="Top-P Nucleus\\nKeep cumulative mass <= p", fillcolor="#B45309"];
        MinP [label="Min-P Filter\\nDrop tokens < p_top * min_p", fillcolor="#B45309"];
        
        TempScale -> TopK -> Softmax -> TopP -> MinP;
    }

    subgraph cluster_sampling {
        label = "Stage 3: Token Emission";
        color = "#334155";
        fontcolor = "#94A3B8";
        SampledToken [label="Sample Next Token\\nMultinomial or Argmax", fillcolor="#7C3AED"];
        MinP -> SampledToken;
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(decode_blueprint_dot, use_container_width=True)

st.markdown("""
### The Sequential Filtering Pipeline
1. **Logit Transformation**: Repetition penalty and temperature scaling reshape relative confidence.
2. **Hard Truncation (Top-K)**: Prunes improbable tokens at the logit level.
3. **Probability Mass Truncation (Top-P & Min-P)**: Dynamically scales candidate set based on confidence.
4. **Sampling / Argmax**: Final token selection from the pruned, normalized probability distribution.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - GREEDY DECODING
# ==============================================================================
st.header("2. Greedy Decoding", divider="rainbow")

st.markdown("""
**Greedy Decoding** is the simplest decoding strategy: at each time step `t`, select the single token with the highest predicted probability:
```text
token_t = argmax_w P(w | x_<t)
```
""")

col_grd1, col_grd2 = st.columns(2)
with col_grd1:
    st.subheader("Advantages of Greedy Decoding")
    st.markdown("""
    * **100% Deterministic**: Given the exact same prompt, greedy decoding will always produce the exact same sequence.
    * **Zero Sampling Overhead**: Requires no random number generation or probability normalization.
    * **Ideal For Factuality**: The gold standard for math calculations, coding tasks, SQL generation, and structured JSON extraction.
    """)

with col_grd2:
    st.subheader("The Repetitive Degeneration Trap")
    st.markdown("""
    * **Shortsighted Myopia**: A high-probability token early on can lead into an inescapable low-probability sentence structure.
    * **Repetition Loops**: On open-ended creative generation, greedy decoding inevitably falls into repetitive cycles (*"I think that that that that..."*).
    * **Boring Prose**: Human speech does not use the #1 most probable word at every step; greedy text feels robotic and unnaturally bland.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - BEAM SEARCH
# ==============================================================================
st.header("3. Beam Search", divider="rainbow")

st.markdown("""
**Beam Search** is a heuristic breadth-first search algorithm that maintains a set of `B` most probable candidate sequences (the **Beam Width**).
At each step, it expands all `B` hypotheses by all vocabulary tokens, scores them by cumulative log probability, and prunes back to the top `B`.
```text
Score(y_1, ..., y_t) = Sum_{i=1}^t log P(y_i | y_<i, x) / (t^alpha)
```
Where `t^alpha` is a **length penalty** to prevent favoring overly short sequences.
""")

col_bm1, col_bm2 = st.columns(2)
with col_bm1:
    st.markdown("#### Where Beam Search Excels")
    st.markdown("""
    * **Machine Translation (Seq2Seq)**: Translating German to English where grammatical structure reorders words significantly.
    * **Automatic Speech Recognition (ASR)**: Whisper uses beam search to resolve acoustic ambiguities.
    * **Abstractive Summarization**: Finding optimal compression paths across short documents.
    """)

with col_bm2:
    st.markdown("#### Why Modern LLMs Abandoned Beam Search")
    st.markdown("""
    * **Memory & KV Cache Multiplier**: Running beam width `B = 4` quadruples KV Cache memory footprint!
    * **Degeneration on Long Text**: Holtzman et al. (2019) proved that beam search on large causal language models leads to extreme unnatural repetition on open-ended generation.
    * **Modern Consensus**: Generative conversational LLMs (GPT-4o, Llama-3) almost exclusively use **Nucleus / Min-P Sampling** rather than Beam Search.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - RANDOM SAMPLING
# ==============================================================================
st.header("4. Random Sampling", divider="rainbow")

st.markdown("""
**Random Sampling (Pure Multinomial Sampling)** samples each token randomly according to its raw probability distribution `P(y | x)`:
```text
token_t ~ Categorical(P(y | x))
```
""")

col_rnd1, col_rnd2 = st.columns(2)
with col_rnd1:
    st.subheader("The Problem with Pure Sampling")
    st.markdown("""
    * Even with a vocabulary of 128,000 tokens, thousands of bizarre, obscure, or nonsensical tokens have tiny non-zero probabilities (e.g. `0.00001`).
    * In a sequence of 500 tokens, pure random sampling will eventually roll the dice on an incoherent tail token.
    * Once an incoherent token is emitted, subsequent autoregressive steps derail completely into gibberish.
    """)

with col_rnd2:
    st.subheader("The Solution")
    st.markdown("""
    * Never use unconstrained random sampling.
    * Always combine sampling with **Temperature** and **Truncation Filters (Top-K / Top-P / Min-P)** to eliminate the long tail of low-probability noise.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - TEMPERATURE
# ==============================================================================
st.header("5. Temperature Scaling", divider="rainbow")

st.markdown("""
**Temperature (`T`)** is a hyperparameter borrowed from thermodynamics that controls the entropy of the probability distribution:
```text
P(token_i) = exp(Logits_i / T) / Sum_{j} exp(Logits_j / T)
```
""")

temp_df = pd.DataFrame({
    "Temperature Value": [
        "T = 0.0 (Greedy)",
        "T = 0.2 - 0.4 (Low)",
        "T = 0.7 - 0.8 (Standard)",
        "T = 1.0 (Neutral)",
        "T = 1.2 - 1.5 (High)",
        "T > 2.0 (Extreme)"
    ],
    "Distribution Shape": [
        "Dirac delta peak on argmax token",
        "Sharply peaked around top 2-3 tokens",
        "Smooth distribution balancing creativity and logic",
        "Raw unmodified model logits",
        "Flatter distribution elevating rare tokens",
        "Nearly uniform noise across all vocabulary"
    ],
    "Optimal Use Case": [
        "Code generation, mathematical proofs, JSON schemas",
        "Fact retrieval, technical QA, classification",
        "Chatbots, general writing, standard creative prose",
        "Theoretical baseline evaluation",
        "Brainstorming wild analogies, poetry, diverse ideas",
        "Unusable (produces word salad and hallucinated syntax)"
    ]
})
st.dataframe(temp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - TOP-K SAMPLING
# ==============================================================================
st.header("6. Top-K Sampling", divider="rainbow")

st.markdown("""
Introduced by Fan et al. (Facebook AI, 2018), **Top-K Sampling** sorts all logits and truncates the candidate pool 
strictly to the **top `K` highest-probability tokens**, setting all other logits to `-infinity`.
```text
V_top_k = { w in Vocab : rank(P(w)) <= K }
```
""")

col_tk1, col_tk2 = st.columns(2)
with col_tk1:
    st.markdown("#### How Top-K Prevents Derailment")
    st.markdown("""
    * If `K = 40`, the model is strictly forbidden from ever choosing any token ranked 41st or lower.
    * Completely eliminates the long tail of weird typos, obscure Unicode symbols, and toxic outliers.
    * Probabilities within the top `K` tokens are re-normalized so they sum to 1.0.
    """)

with col_tk2:
    st.markdown("#### The Inflexibility Flaw of Fixed K")
    st.markdown("""
    * **Flat Distribution Scenario**: When describing a color (*"The flower was [red/blue/yellow/purple/pink...]"*), 30 words might be valid. A small `K = 10` unnaturally clips valid choices!
    * **Peaked Distribution Scenario**: When continuing a common phrase (*"Barack [Obama]"*), only 1 token is valid (`P=0.99`). A large `K = 40` forces 39 implausible tokens into the candidate pool!
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - TOP-P (NUCLEUS) SAMPLING
# ==============================================================================
st.header("7. Top-P (Nucleus) Sampling", divider="rainbow")

st.markdown("""
Introduced by Ari Holtzman et al. (University of Washington, 2019), **Top-P (Nucleus) Sampling** dynamically adjusts 
the size of the candidate pool based on cumulative probability mass rather than a fixed integer count.
```text
V_top_p = Smallest set of tokens such that Sum_{w in V_top_p} P(w) >= p
```
""")

col_tp1, col_tp2 = st.columns(2)
with col_tp1:
    st.subheader("Dynamic Candidate Pool Adaptation")
    st.markdown("""
    * If `P = 0.9` (90% cumulative mass):
      * **When Confident**: If top token has `P = 0.95`, candidate pool contains **only 1 token** (`["Obama"]`).
      * **When Uncertain**: If probabilities are evenly distributed (`0.05` each), candidate pool dynamically expands to **18 tokens**!
    * Automatically adapts between strict precision and creative diversity.
    """)

with col_tp2:
    st.subheader("Industry Default Practice")
    st.markdown("""
    * Setting **`temperature = 0.7`** combined with **`top_p = 0.9`** is the universal standard for modern conversational assistants (ChatGPT, Claude, Llama).
    * Setting `top_p = 1.0` disables nucleus filtering.
    """)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - MIN-P SAMPLING
# ==============================================================================
st.header("8. Min-P Sampling", divider="rainbow")

st.markdown("""
Introduced by Grimmeisen (2023), **Min-P Sampling** is an advanced alternative to Top-P that has gained widespread adoption 
in open-source AI communities (llama.cpp, vLLM).
""")

col_mp1, col_mp2 = st.columns(2)
with col_mp1:
    st.markdown("#### The Min-P Mathematical Formulation")
    st.markdown("""
    Instead of summing cumulative mass, Min-P sets a cutoff threshold **relative to the probability of the most likely token**:
    ```text
    Threshold = P_top * min_p
    Candidate_Tokens = { w in Vocab : P(w) >= Threshold }
    ```
    Where `P_top` is the highest probability token at that step, and `min_p` is typically `0.05` to `0.10` (5% to 10%).
    """)

with col_mp2:
    st.markdown("#### Why Min-P Outperforms Top-P")
    st.markdown("""
    * **Prevents Low-Quality Inclusion**: In Top-P, if `p=0.9`, low-quality tokens can slip into the pool if the head tokens don't quite reach 0.9.
    * **Preserves High Creativity**: Min-P scales with confidence: if `P_top = 0.8`, threshold is `0.08` (strict); if `P_top = 0.1`, threshold is `0.01` (permits diverse valid alternatives).
    * Eliminates the need to aggressively tune `top_k` and `top_p` simultaneously.
    """)

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - REPETITION PENALTY
# ==============================================================================
st.header("9. Repetition Penalty", divider="rainbow")

st.markdown("""
Introduced by Nitish Shirish Keskar et al. (CTRL, Salesforce, 2019), **Repetition Penalty** discounts 
the logits of tokens that have already appeared in the prompt or generated sequence.
""")

st.markdown("""
#### The Mathematical Modification Rule
For each token `i` that has appeared in the generated context:
```text
Logit_i = Logit_i / penalty   if Logit_i > 0
Logit_i = Logit_i * penalty   if Logit_i < 0
```
Where `penalty >= 1.0` (typically `1.1` to `1.2`):
* If `penalty = 1.0`: No penalty applied.
* If `penalty = 1.2`: Emitted tokens have their logits suppressed, reducing their chance of re-appearing.
* **Caution**: Setting `penalty > 1.3` causes the model to refuse using common grammatical prepositions (*"the"*, *"and"*, *"in"*) or repeat character names!
""")

st.divider()

# ==============================================================================
# SECTION 10: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Multi-Strategy Decoding Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates the complete pipeline of modern decoding:
repetition penalty, temperature scaling, top-k, top-p, and min-p sampling.
""")

st.code("""
import numpy as np
from typing import List, Tuple

class ProductionDecodingEngine:
    \"\"\"
    Production-grade multi-strategy decoding engine.
    \"\"\"
    def __init__(self, vocab_size: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)

    def sample_distribution(
        self,
        base_logits: np.ndarray,
        generated_tokens: List[int],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # 1. Repetition penalty
        mod_logits = np.array(base_logits, dtype=np.float32)
        if repetition_penalty > 1.0 and generated_tokens:
            for tid in set(generated_tokens):
                if mod_logits[tid] > 0: mod_logits[tid] /= repetition_penalty
                else: mod_logits[tid] *= repetition_penalty

        # 2. Temperature
        if temperature <= 0.0:
            return mod_logits, mod_logits, int(np.argmax(mod_logits))
        mod_logits /= max(1e-4, temperature)

        # 3. Top-K
        if 0 < top_k < len(mod_logits):
            cutoff = np.sort(mod_logits)[-top_k]
            mod_logits[mod_logits < cutoff] = -np.inf

        # Softmax
        exp_l = np.exp(mod_logits - np.max(mod_logits))
        probs = exp_l / np.sum(exp_l)

        # 4. Top-P
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cum_p = np.cumsum(probs[sorted_idx])
            mask = cum_p > top_p
            mask[0] = False
            probs[sorted_idx[mask]] = 0.0
            probs /= np.sum(probs)

        # 5. Min-P
        if min_p > 0.0:
            probs[probs < (np.max(probs) * min_p)] = 0.0
            probs /= np.sum(probs)

        # 6. Sample
        return mod_logits, probs, int(self.rng.choice(len(probs), p=probs))
""", language="python")

st.divider()

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Interactive Sampling Visualizer & Logit Distribution Lab", divider="rainbow")

engine = ProductionDecodingEngine(vocab_size=12, seed=42)

# Synthetic base logits
np.random.seed(123)
synthetic_logits = np.array([4.2, 3.8, 3.1, 2.4, 2.0, 1.5, 1.1, 0.8, 0.4, -0.2, -1.0, -2.5], dtype=np.float32)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Decoding Hyperparameter Controls")
    u_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    u_top_k = st.slider("Top-K Truncation (0 = Disabled)", min_value=0, max_value=12, value=6)
    u_top_p = st.slider("Top-P Nucleus (1.0 = Disabled)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    u_min_p = st.slider("Min-P Threshold (0.0 = Disabled)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    u_rep_pen = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05)

    sim_past_tokens = [0, 1]  # simulate that token_0 and token_1 were previously generated
    st.caption("Previously generated tokens receiving repetition penalty: `token_0` and `token_1`.")

    l_scaled, probs_filtered, chosen_token = engine.sample_distribution(
        synthetic_logits,
        generated_tokens=sim_past_tokens,
        temperature=u_temp,
        top_k=u_top_k,
        top_p=u_top_p,
        min_p=u_min_p,
        repetition_penalty=u_rep_pen
    )

with wb_col2:
    st.subheader("Filtered Probability Distribution")
    st.metric("Sampled Token at this Step", f"token_{chosen_token}", f"Probability: {probs_filtered[chosen_token]*100:.2f}%")

    dist_df = pd.DataFrame({
        "Token": [f"token_{i}" for i in range(len(synthetic_logits))],
        "Filtered Probability": probs_filtered
    })
    st.bar_chart(dist_df.set_index("Token"))
    st.caption("Tokens filtered out by Top-K, Top-P, or Min-P have probability 0.0.")

st.divider()

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why does Top-P (Nucleus) sampling outperform Top-K on diverse prompts?"):
        st.write("""
        Top-K uses a static integer cutoff regardless of context uncertainty. 
        When the model is confident in a single completion (e.g. phrase completion), a large K includes dozens of absurd candidate words. 
        When the model has many valid synonym options (e.g. list of colors), a small K prematurely cuts off valid words.
        Top-P dynamically adapts: the candidate pool shrinks to 1 token when confident, and expands to dozens of tokens when uncertain, perfectly matching the distribution's entropy.
        """)
    with st.expander("Q2: What is Min-P sampling and why is it replacing Top-P in modern runtimes?"):
        st.write("""
        Min-P sets a cutoff threshold relative to the probability of the most likely token: `Threshold = P_top * min_p`.
        In Top-P, low-probability tail tokens can still enter the candidate set if high-probability tokens fail to reach the cumulative mass `p`.
        Min-P eliminates this artifact: it strictly discards any token that is not at least a certain fraction (e.g. 5%) as likely as the best candidate, producing more coherent and creative prose without tail noise.
        """)
    with st.expander("Q3: What happens mathematically when Temperature approaches 0 vs infinity?"):
        st.write("""
        As `T -> 0`: The logit differences `Logits / T` explode toward infinity. In the Softmax, the largest logit receives probability 1.0 while all other tokens receive probability 0.0 (exact Greedy Decoding).
        As `T -> infinity`: The scaled logits `Logits / T` approach 0.0. In the Softmax, `exp(0) = 1` for all tokens, collapsing the distribution into uniform random noise (`P = 1 / Vocab_Size`).
        """)

with q2:
    with st.expander("Q4: Why was Beam Search abandoned for open-ended conversational LLMs?"):
        st.write("""
        1. **Memory**: Beam width `B = 4` quadruples the KV Cache memory footprint on high-concurrency servers.
        2. **Perception of Quality**: Research (Holtzman et al., 2019) demonstrated that high-likelihood sequences found by beam search on open-ended text degenerate into repetitive, unnatural prose. Human language is naturally non-greedy.
        """)
    with st.expander("Q5: How does Repetition Penalty differ from Frequency Penalty?"):
        st.write("""
        **Repetition Penalty (Keskar et al.)** applies a **multiplicative divisor** (`Logits / penalty`) to any token that has appeared at least once.
        **Frequency Penalty (OpenAI)** applies a **linear subtraction** (`Logits - penalty * count`) proportional to the number of times the token has appeared in the sequence.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Setting Excessive Repetition Penalty (> 1.3)**
    * **Failure**: Setting `repetition_penalty = 1.5` to eliminate loops.
    * **Impact**: The model is penalized for using common grammatical conjunctions (*"the"*, *"and"*, *"is"*), resulting in distorted, broken English syntax.
    * **Remedy**: Keep repetition penalty strictly between `1.05` and `1.15`.
    """)
    st.error("""
    **🚨 Production Trap 2: Stacking Redundant Conflicting Filters**
    * **Failure**: Setting `top_k = 3`, `top_p = 0.5`, and `temperature = 1.5` simultaneously.
    * **Impact**: Top-K of 3 restricts candidates before Top-P can dynamically adapt, while high temperature creates chaotic swings inside the 3 tokens.
    * **Remedy**: Use `temperature + top_p` alone, or use `temperature + min_p`. Avoid over-constraining Top-K and Top-P together.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Using Temperature > 0 for Code and JSON Generation**
    * **Failure**: Setting `temperature = 0.7` for SQL query generation or Pydantic output.
    * **Impact**: Causes sporadic syntax errors, hallucinated schema keys, and invalid closing brackets.
    * **Remedy**: Always enforce `temperature = 0.0` for code, math, and JSON extraction.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Confusing Temperature Zero with Disabling Sampling**
    * **Failure**: Passing `temperature = 0.0` to frameworks that divide by temperature without a zero-check, triggering division-by-zero errors.
    * **Impact**: Runtime exception crashes serving containers.
    * **Remedy**: Always route `temperature = 0.0` to `argmax()` directly.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Greedy Decoding Formula"):
        st.write("What mathematical function defines greedy token selection?")
        st.info("`token_t = argmax_w P(w | x_<t)`.")
    with st.expander("⚡ Flashcard 2: Standard Production Sampling Pair"):
        st.write("What is the most common industry-standard sampling configuration for chat?")
        st.info("`temperature = 0.7` and `top_p = 0.9`.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Min-P Cutoff Calculation"):
        st.write("How does Min-P compute its threshold?")
        st.info("`Threshold = P_top * min_p`.")
    with st.expander("⚡ Flashcard 4: Beam Search Memory Cost"):
        st.write("How does beam search affect KV Cache memory usage?")
        st.info("Multiplies KV Cache memory by the beam width `B`.")

st.divider()

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Decoding Strategies before advancing to Lesson 16.3 (Generation Control):
""")

chk1 = st.checkbox("I understand why greedy decoding is deterministic but prone to repetition loops.", value=True)
chk2 = st.checkbox("I know why modern conversational models use sampling rather than Beam Search.", value=True)
chk3 = st.checkbox("I can explain how Temperature scales logits to sharpen or flatten the probability distribution.", value=True)
chk4 = st.checkbox("I understand the dynamic candidate pool sizing of Top-P (Nucleus) vs static Top-K.", value=True)
chk5 = st.checkbox("I understand how Min-P scales its cutoff relative to the top candidate token.", value=True)
chk6 = st.checkbox("I know how repetition penalties suppress previously generated tokens.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 16.3: Generation Control.")
