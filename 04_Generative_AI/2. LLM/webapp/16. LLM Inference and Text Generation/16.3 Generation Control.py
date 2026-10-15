import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.3 Generation Control - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionGenerationControlEngine:
    """
    Production-grade Generation Control Engine demonstrating:
    - Maximum token budget enforcement
    - Multi-character stop sequence buffering and early exit
    - Frequency penalty (proportional to count)
    - Presence penalty (binary flat penalty)
    - Seed-based deterministic PRNG tracking
    """
    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def apply_penalties(
        self,
        logits: np.ndarray,
        token_counts: Dict[int, int],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> np.ndarray:
        """
        Applies OpenAI-standard frequency and presence penalties:
        modified_logit = logit - (count * frequency_penalty) - (presence_flag * presence_penalty)
        """
        mod = np.array(logits, dtype=np.float32)
        for token_id, count in token_counts.items():
            if token_id < len(mod) and count > 0:
                mod[token_id] -= (count * frequency_penalty) + presence_penalty
        return mod

    def check_stop_sequences(self, current_text: str, stop_sequences: List[str]) -> Tuple[bool, Optional[str]]:
        """Checks if generated text ends with or contains any registered stop sequence."""
        for seq in stop_sequences:
            if seq and seq in current_text:
                return True, seq
        return False, None

    def simulate_generation(
        self,
        prompt: str,
        vocabulary: List[str],
        max_tokens: int = 15,
        stop_sequences: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """Simulates generation step-by-step applying penalties and stop checks."""
        stop_sequences = stop_sequences or []
        token_counts: Dict[int, int] = {}
        generated_tokens: List[str] = []
        full_text = prompt
        termination_reason = "max_tokens_reached"
        
        # Base vocabulary logits
        base_logits = np.linspace(3.0, 0.5, len(vocabulary))

        for step in range(max_tokens):
            # Apply frequency & presence penalties
            penalized_logits = self.apply_penalties(
                base_logits,
                token_counts,
                frequency_penalty,
                presence_penalty
            )
            # Softmax
            exp_l = np.exp(penalized_logits - np.max(penalized_logits))
            probs = exp_l / np.sum(exp_l)

            # Sample
            chosen_idx = int(self.rng.choice(len(vocabulary), p=probs))
            chosen_word = vocabulary[chosen_idx]

            # Update count
            token_counts[chosen_idx] = token_counts.get(chosen_idx, 0) + 1
            generated_tokens.append(chosen_word)
            full_text += " " + chosen_word

            # Check stop condition
            has_stopped, matched_seq = self.check_stop_sequences(full_text, stop_sequences)
            if has_stopped:
                termination_reason = f"stop_sequence_matched: '{matched_seq}'"
                break

        return {
            "full_text": full_text,
            "generated_tokens": generated_tokens,
            "token_count": len(generated_tokens),
            "termination_reason": termination_reason,
            "token_frequency_distribution": {vocabulary[k]: v for k, v in token_counts.items()}
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("16.3 Generation Control")
st.caption("Chapter 16: LLM Inference and Text Generation | Domain: 2. Large Language Models")

st.markdown("""
Left unconstrained, an autoregressive language model will continue streaming tokens indefinitely until 
it fills the physical context window or crashes GPU memory.
In production engineering, we control generation behavior through strict **stopping boundaries** 
(Maximum Tokens, Stop Sequences) and **dynamic logit penalties** (Frequency Penalty, Presence Penalty), 
while governing reproducibility through **deterministic seeding**.

In this lesson, you will master:
1. Maximum Tokens: Hard token budgets, truncation risks, and cost ceilings.
2. Stop Sequences: Multi-token delimiter buffers, early exit signals, and streaming flush mechanics.
3. Frequency Penalty: Linear logit suppression proportional to past token occurrence counts.
4. Presence Penalty: Binary logit suppression that encourages topical exploration.
5. Deterministic Generation: Zero-temperature decoding and eliminating GPU non-determinism.
6. Random Generation & Seed Reproducibility: Controlling pseudo-random number generator (PRNG) states.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Generation Termination & Logit Modification Pipeline", divider="rainbow")

st.markdown("""
At each iteration of the autoregressive loop, generation control intervenes both before token sampling 
(applying logit penalties) and after token emission (evaluating termination criteria).
""")

control_blueprint_dot = """
digraph GenerationControlPipeline {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_penalties {
        label = "Phase 1: Logit Penalty Transformation";
        color = "#334155";
        fontcolor = "#94A3B8";
        RawLogits [label="Raw Logits\\n[Vocab_Size]", fillcolor="#0F172A"];
        FreqPen [label="Frequency Penalty\\nSubtract: count * freq_pen", fillcolor="#0369A1"];
        PresPen [label="Presence Penalty\\nSubtract: 1(count > 0) * pres_pen", fillcolor="#0369A1"];
        SampleStep [label="Sample Token\\n(PRNG Seeded State)", fillcolor="#047857"];
        
        RawLogits -> FreqPen -> PresPen -> SampleStep;
    }

    subgraph cluster_checks {
        label = "Phase 2: Stopping Condition Evaluation";
        color = "#334155";
        fontcolor = "#94A3B8";
        CheckBudget [label="Step Count >= max_tokens?", fillcolor="#B45309"];
        CheckEOS [label="Token == <|eos|>?", fillcolor="#B45309"];
        CheckStopSeq [label="Buffer matches Stop Sequence?\\n(e.g. '\\n\\n' or 'User:')", fillcolor="#B45309"];
        
        SampleStep -> CheckBudget;
        CheckBudget -> CheckEOS [label="No"];
        CheckEOS -> CheckStopSeq [label="No"];
    }

    subgraph cluster_exit {
        label = "Phase 3: Execution Termination";
        color = "#334155";
        fontcolor = "#94A3B8";
        ContinueLoop [label="Append Token to Context\\nContinue Autoregressive Loop", fillcolor="#0284C7"];
        TerminateStream [label="Halt Generation & Flush Stream\\nReturn finish_reason", fillcolor="#7C3AED"];
    }

    CheckStopSeq -> ContinueLoop [label="No Match"];
    CheckBudget -> TerminateStream [label="Yes (max_tokens)"];
    CheckEOS -> TerminateStream [label="Yes (stop)"];
    CheckStopSeq -> TerminateStream [label="Yes (stop_sequence)"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(control_blueprint_dot, use_container_width=True)

st.markdown("""
### The Two-Stage Intervention Strategy
1. **Pre-Sampling Intervention**: Frequency and presence penalties modify logits to dynamically steer which tokens can be picked.
2. **Post-Sampling Intervention**: Step counters, EOS token comparisons, and string buffers check whether generation should halt immediately.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - MAXIMUM TOKENS
# ==============================================================================
st.header("2. Maximum Tokens", divider="rainbow")

st.markdown("""
The **Maximum Tokens** parameter imposes a hard upper bound on the number of tokens the model is permitted to generate:
* **`max_tokens` (OpenAI / Anthropic)**: Maximum tokens emitted in the completion.
* **`max_new_tokens` (Hugging Face)**: Explicitly defines the count of newly generated tokens (excluding prompt tokens).
""")

col_mt1, col_mt2 = st.columns(2)
with col_mt1:
    st.subheader("Why Maximum Tokens is Mandatory in Production")
    st.markdown("""
    * **Cost Cap Protection**: Without `max_tokens`, a single runaway prompt (e.g. generating an infinite code loop) can burn hundreds of dollars on cloud API billing.
    * **SLA Timeout Defense**: Serving clusters maintain timeout thresholds (e.g. 30 seconds). Uncapped generation causes requests to time out.
    * **Preventing Buffer Overflow**: Ensures the total sequence `Prompt_Tokens + Generated_Tokens` never exceeds the hardware KV Cache allocation.
    """)

with col_mt2:
    st.subheader("The Truncation Hazard (`finish_reason: length`)")
    st.markdown("""
    * If `max_tokens` is set too low (e.g. 50 tokens), generation stops mid-sentence or mid-JSON bracket:
      ```json
      {"status": "success", "results": [{"id": 1, "name": "Al
      ```
    * The downstream parser crashes with `JSONDecodeError` because the closing `}` was never emitted.
    * **Mitigation**: Always allocate generous `max_tokens` (e.g. 2,048) and rely on **Stop Sequences** or Structured Grammars for early termination!
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - STOP SEQUENCES
# ==============================================================================
st.header("3. Stop Sequences", divider="rainbow")

st.markdown("""
A **Stop Sequence** is a user-defined string or token sequence that instructs the inference engine 
to immediately halt text generation the moment that exact character sequence is produced.
""")

col_ss1, col_ss2 = st.columns(2)
with col_ss1:
    st.markdown("#### Common Production Stop Sequences")
    st.markdown("""
    * **Multi-Turn Chat**: `stop=["\nUser:", "<|im_start|>user", "<|eot_id|>"]`  
      Prevents the model from generating fake user queries once it finishes the assistant turn.
    * **Few-Shot Boundary**: `stop=["\n\nExample:", "###"]`  
      Stops the model from attempting to invent new demonstration examples.
    * **Code Execution**: `stop=["\n```\n", "def "]`  
      Terminates after generating the requested function block.
    """)

with col_ss2:
    st.markdown("#### The Multi-Token Buffer Problem in Streaming")
    st.markdown("""
    * What happens when the stop sequence is `"User:"`?
    * The model first emits `"Use"`, then `"r"`, then `":"`.
    * If the server immediately streams `"Use"` to the client WebSocket, the user briefly sees the stop sequence before it is pruned!
    * **Solution: Sliding Window Holdback**: Serving engines buffer tokens matching any prefix of a registered stop sequence, only flushing tokens once the buffer confirms no match.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - FREQUENCY PENALTY
# ==============================================================================
st.header("4. Frequency Penalty", divider="rainbow")

st.markdown("""
The **Frequency Penalty** (popularized by OpenAI) applies a linear discount to token logits 
proportional to the number of times that specific token has already appeared in the generated completion:
```text
Logit_i_new = Logit_i_raw - (c_i * Frequency_Penalty)
```
Where `c_i` is the frequency count of token `i` in the generated text, and `Frequency_Penalty in [-2.0, 2.0]` (typically `0.1` to `0.5`).
""")

col_fp1, col_fp2 = st.columns(2)
with col_fp1:
    st.subheader("Mechanics of Proportional Suppression")
    st.markdown("""
    * If a word has appeared **1 time**: logit is discounted by `1 * 0.2 = -0.2`.
    * If a word has appeared **5 times**: logit is discounted by `5 * 0.2 = -1.0`.
    * If a word has appeared **15 times**: logit is discounted by `15 * 0.2 = -3.0` (effectively banishing it from generation).
    """)

with col_fp2:
    st.subheader("When to Use Frequency Penalty")
    st.markdown("""
    * **Word Variety**: Eliminates repetitive overuse of pet phrases (*"furthermore"*, *"delve"*, *"crucial"*).
    * **Long-Form Articles**: Encourages broader vocabulary across multi-page essays.
    * **Warning**: Do not use on code or JSON! It penalizes mandatory syntax keywords (`return`, `class`, `"id"`).
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - PRESENCE PENALTY
# ==============================================================================
st.header("5. Presence Penalty", divider="rainbow")

st.markdown("""
The **Presence Penalty** applies a flat, one-time discount to token logits if the token has appeared **at least once** 
in the generated text, regardless of how many times it was repeated:
```text
Logit_i_new = Logit_i_raw - (1(c_i > 0) * Presence_Penalty)
```
Where `1(c_i > 0)` is a binary indicator function, and `Presence_Penalty in [-2.0, 2.0]` (typically `0.2` to `0.6`).
""")

col_pp1, col_pp2 = st.columns(2)
with col_pp1:
    st.markdown("#### Frequency vs. Presence Penalty")
    st.markdown("""
    * **Frequency Penalty**: Penalizes **repetition of specific words**. The more you say a word, the more it is penalized.
    * **Presence Penalty**: Penalizes **talking about the same concepts**. It encourages the model to introduce completely new ideas, topics, and vocabulary.
    """)

with col_pp2:
    st.markdown("#### Practical Example")
    st.markdown("""
    * In a brainstorming prompt about renewable energy:
    * High Presence Penalty forces the model to move beyond solar and wind to explore geothermal, tidal, and nuclear fusion.
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - DETERMINISTIC GENERATION
# ==============================================================================
st.header("6. Deterministic Generation", divider="rainbow")

st.markdown("""
In enterprise software engineering (CI test assertions, financial audits, medical reports), 
reproducibility is mandatory: the model must produce the exact same output given the exact same input.
""")

st.markdown("""
### The Two Requirements for Determinism
1. **Greedy Decoding (`temperature = 0.0`)**: Ensures the model always picks the argmax token.
2. **CUDA Non-Determinism Mitigation**:
   * Modern GPUs use parallel floating-point reductions (scatter-gather operations).
   * Floating-point addition is **non-associative**: `(a + b) + c != a + (b + c)` due to rounding.
   * If thread execution orders vary across GPU warps, tiny float differences (`1e-7`) can occasionally flip the argmax token!
   * In PyTorch, enforce deterministic algorithms:
     ```python
     torch.use_deterministic_algorithms(True)
     torch.backends.cudnn.deterministic = True
     ```
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - RANDOM GENERATION & SEEDS
# ==============================================================================
st.header("7. Random Generation & Seed Reproducibility", divider="rainbow")

st.markdown("""
When sampling with temperature (`T > 0`), generation is governed by a **Pseudo-Random Number Generator (PRNG)**.
By fixing the **Seed**, developers can reproduce creative sampling paths deterministically.
""")

col_rng1, col_rng2 = st.columns(2)
with col_rng1:
    st.subheader("Seed Control in APIs")
    st.markdown("""
    Modern APIs (OpenAI, Anthropic, vLLM) expose a `seed` parameter:
    ```python
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a poem"}],
        temperature=0.8,
        seed=42  # Locks PRNG state
    )
    ```
    * The API returns a `system_fingerprint`. If the seed and fingerprint match, outputs are deterministic.
    """)

with col_rng2:
    st.subheader("Why Seeds Matter in AI Development")
    st.markdown("""
    * **A/B Prompt Testing**: Fixing the seed ensures that differences in outputs are caused by your prompt changes, not random sampling luck.
    * **Debugging Edge Cases**: If a customer reports a hallucination on a specific prompt, capturing the seed allows developers to reproduce and diagnose the bug locally.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Generation Control Engine & Penalty Transformer in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates frequency/presence logit penalties, stop sequence buffering, 
and token budget enforcement.
""")

st.code("""
import numpy as np
from typing import Dict, List, Tuple, Optional

class ProductionGenerationControlEngine:
    \"\"\"
    Production-grade Generation Control Engine.
    \"\"\"
    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.RandomState(seed)

    def apply_penalties(
        self,
        logits: np.ndarray,
        token_counts: Dict[int, int],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> np.ndarray:
        mod = np.array(logits, dtype=np.float32)
        for token_id, count in token_counts.items():
            if token_id < len(mod) and count > 0:
                mod[token_id] -= (count * frequency_penalty) + presence_penalty
        return mod

    def check_stop_sequences(self, current_text: str, stop_sequences: List[str]) -> Tuple[bool, Optional[str]]:
        for seq in stop_sequences:
            if seq and seq in current_text:
                return True, seq
        return False, None
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Logit Penalty Simulator & Stopping Condition Workbench", divider="rainbow")

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("Generation Bounds & Penalties")
    user_max_tokens = st.slider("Max New Tokens Budget", min_value=3, max_value=20, value=8)
    user_stop_seq = st.text_input("Registered Stop Sequence", value="STOP")
    user_freq_pen = st.slider("Frequency Penalty", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    user_pres_pen = st.slider("Presence Penalty", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
    user_seed = st.number_input("PRNG Seed", value=42, step=1)

    toy_vocab = ["apple", "banana", "orange", "apple", "grape", "STOP", "cherry", "melon"]
    ctrl_engine = ProductionGenerationControlEngine(seed=int(user_seed))
    
    sim_res = ctrl_engine.simulate_generation(
        prompt="Basket:",
        vocabulary=toy_vocab,
        max_tokens=user_max_tokens,
        stop_sequences=[user_stop_seq],
        frequency_penalty=user_freq_pen,
        presence_penalty=user_pres_pen
    )

with wb_c2:
    st.subheader("Generation Trace & Termination Status")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Emitted Tokens", f"{sim_res['token_count']} Tokens")
    with col_t2:
        st.metric("Termination Reason", sim_res["termination_reason"])

    st.markdown("##### Resulting Text Stream:")
    st.code(sim_res["full_text"], language="text")

    st.markdown("##### Token Frequency Counts:")
    freq_series = pd.Series(sim_res["token_frequency_distribution"])
    st.bar_chart(freq_series)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: What is the mathematical difference between Frequency Penalty and Presence Penalty?"):
        st.write("""
        **Frequency Penalty** discounts logits proportional to the number of past occurrences: `Logits - c * freq_penalty`. A word that appeared 5 times is penalized 5x more than a word that appeared once.
        **Presence Penalty** discounts logits with a flat binary step if the token appeared at least once: `Logits - 1(c > 0) * pres_penalty`. It penalizes staying on the same topic rather than repeating the same word.
        """)
    with st.expander("Q2: Why does max_tokens truncate JSON outputs and how do you prevent it?"):
        st.write("""
        `max_tokens` is a hard generation cutoff. If the model is in the middle of producing an array or nested object when it hits the limit, generation halts immediately without emitting closing brackets.
        To prevent this:
        1. Set generous token budgets (2x-3x expected output length).
        2. Use grammar-constrained decoding (CFGs) which force the model to complete valid structures within budgets.
        3. Check `finish_reason == "length"` in API responses and retry with expanded limits.
        """)
    with st.expander("Q3: Why can temperature=0 still produce non-deterministic results on GPUs?"):
        st.write("""
        Modern GPU kernels execute parallel reduction operations (e.g. tree summations in FlashAttention and GEMM).
        Floating-point addition is non-associative. Because thread scheduling across GPU warps varies slightly depending on concurrent memory bus contention, the order of summation changes, producing minor floating-point rounding variations (`1e-7`).
        When two candidate tokens have virtually identical logits, this micro-noise can flip the argmax selection.
        """)

with q2:
    with st.expander("Q4: How does a streaming inference server handle multi-token stop sequences?"):
        st.write("""
        If a stop sequence spans multiple tokens (e.g. `\nUser:` which may be 2-3 tokens), the server cannot stream tokens immediately upon emission.
        It maintains a sliding-window character buffer. 
        Tokens that form a partial match with the stop sequence are held back in the buffer. 
        If subsequent tokens complete the match, generation halts and the buffer is discarded; if the match fails, the buffered tokens are flushed to the stream.
        """)
    with st.expander("Q5: Why should you avoid Frequency and Presence penalties during code generation?"):
        st.write("""
        Source code relies on high token repetition for correctness (e.g. repeating variable names, `return`, `self`, `def`, `class`).
        Applying penalties artificially degrades the logits of mandatory syntax tokens, forcing the model to rename variables mid-function or invent invalid syntax keywords.
        Keep both penalties at 0.0 for code and JSON generation.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Setting Positive Penalties on Code or JSON Tasks**
    * **Failure**: Applying `frequency_penalty = 0.5` on Python code generation.
    * **Impact**: The model invents bizarre synonyms for variables (*"val1"*, *"val2"*, *"datum"*) and forgets indentation keywords, breaking syntax.
    * **Remedy**: Keep frequency and presence penalties at `0.0` for code and structured extraction.
    """)
    st.error("""
    **🚨 Production Trap 2: Missing Stop Sequences in Few-Shot Prompts**
    * **Failure**: Omitting `stop=["\n\nExample"]` in few-shot prompting.
    * **Impact**: After answering the user query, the model continues generating fake examples 4, 5, and 6, wasting tokens.
    * **Remedy**: Always register the demonstration header as a stop sequence.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Ignoring finish_reason Telemetry**
    * **Failure**: Blindly parsing API response content without checking `finish_reason`.
    * **Impact**: Downstream parsers crash when `finish_reason == "length"` causes truncated outputs.
    * **Remedy**: Always check `if finish_reason == "length": handle_truncation()`.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Excessive Presence Penalty Causing Hallucination**
    * **Failure**: Setting `presence_penalty = 1.5` to encourage maximum creativity.
    * **Impact**: The model exhausts realistic vocabulary and begins hallucinating bizarre fabricated terminology.
    * **Remedy**: Keep presence penalty between `0.1` and `0.5` for creative tasks.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Frequency Penalty Equation"):
        st.write("How does Frequency Penalty scale with token count?")
        st.info("Linearly proportional: `Logits - (count * freq_penalty)`.")
    with st.expander("⚡ Flashcard 2: Presence Penalty Equation"):
        st.write("How does Presence Penalty scale with token count?")
        st.info("Flat binary step: `Logits - 1(count > 0) * pres_penalty`.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Truncation Finish Reason"):
        st.write("What finish reason indicates a maximum token budget cut off?")
        st.info("`finish_reason: length`.")
    with fc_col2:
        with st.expander("⚡ Flashcard 4: Code Generation Penalty Rule"):
            st.write("What should frequency and presence penalties be set to for coding?")
            st.info("Strictly 0.0.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Generation Control before advancing to Lesson 16.4 (KV Cache):
""")

chk1 = st.checkbox("I understand the difference between max_tokens and max_new_tokens.", value=True)
chk2 = st.checkbox("I know why stop sequences are required to prevent conversational runaway.", value=True)
chk3 = st.checkbox("I can derive the mathematical formulation of Frequency vs Presence penalties.", value=True)
chk4 = st.checkbox("I understand why penalties must be zero for code and JSON extraction tasks.", value=True)
chk5 = st.checkbox("I know why GPU parallel reductions can cause non-determinism even at temperature 0.", value=True)
chk6 = st.checkbox("I understand how multi-token stop sequences require sliding-window holdback buffers in streaming.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 16.4: KV Cache.")
