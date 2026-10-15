import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="16.1 LLM Generation - RAG Master Curriculum",
    page_icon="🤖",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class LLMGenerationEngine:
    """
    Production-grade LLM Generation Simulation Engine:
    1. Context payload injection and prompt template compilation.
    2. Context-aware conditioning and attention steering emulation.
    3. Autoregressive decoding simulation across greedy, top-p, and top-k modes.
    4. Temperature softmax logit scaling and sampling entropy calculation.
    5. Token budget headroom enforcement and truncation prevention.
    """
    def __init__(self, vocab_sample: Optional[Dict[str, float]] = None):
        self.vocab_sample = vocab_sample or {
            "Enterprise": 4.2,
            "SLA": 4.0,
            "guarantees": 3.8,
            "99.99%": 4.5,
            "uptime": 3.9,
            "for": 2.1,
            "Tier-1": 3.7,
            "banking": 3.4,
            "systems": 2.8,
            "according": 2.5,
            "to": 2.0,
            "policy": 3.1
        }

    def compile_prompt(self, system_instruction: str, context_chunks: List[str], user_query: str) -> str:
        """Assembles prompt payload with XML delimitation."""
        context_body = "\n".join([f"  <document id='DOC-{i+1}'>\n    {c}\n  </document>" for i, c in enumerate(context_chunks)])
        return (
            f"=== SYSTEM DIRECTIVE ===\n{system_instruction}\n\n"
            f"=== RETRIEVED CONTEXT ===\n<context>\n{context_body}\n</context>\n\n"
            f"=== USER QUERY ===\n<query>{user_query}</query>\n\n"
            f"=== ASSISTANT RESPONSE ==="
        )

    def apply_temperature(self, logits: Dict[str, float], temperature: float) -> Dict[str, float]:
        """
        Scales raw logits by temperature and calculates softmax probabilities:
        P(token) = exp(logit / T) / sum(exp(logit_j / T))
        When temperature approaches zero, applies greedy argmax.
        """
        tokens = list(logits.keys())
        raw_vals = np.array([logits[t] for t in tokens], dtype=np.float64)

        if temperature <= 0.01:
            # Deterministic greedy collapse
            probs = np.zeros_like(raw_vals)
            probs[np.argmax(raw_vals)] = 1.0
        else:
            scaled = raw_vals / max(temperature, 0.01)
            # Subtract max for numerical stability
            exp_vals = np.exp(scaled - np.max(scaled))
            probs = exp_vals / np.sum(exp_vals)

        return {tokens[i]: float(probs[i]) for i in range(len(tokens))}

    def calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculates Shannon entropy of the token distribution: H = -sum(p * log2(p))."""
        p_vals = np.array([p for p in probabilities.values() if p > 0.0], dtype=np.float64)
        return float(-np.sum(p_vals * np.log2(p_vals)))

    def simulate_token_generation(
        self,
        prompt_length_tokens: int,
        max_output_tokens: int,
        temperature: float,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """Simulates generation latency, token counts, and completion probability."""
        # Realistic hardware baseline: 35 tokens/sec generation, 500 tokens/sec prompt ingestion
        prompt_ingestion_latency_ms = (prompt_length_tokens / 500.0) * 1000.0
        # Simulated actual output length (capped by max_output_tokens)
        expected_output_tokens = min(max_output_tokens, int(45 + 15 * (1.0 - temperature)))
        generation_latency_ms = (expected_output_tokens / 35.0) * 1000.0
        total_latency_ms = prompt_ingestion_latency_ms + generation_latency_ms

        is_truncated = expected_output_tokens >= max_output_tokens

        return {
            "prompt_tokens": prompt_length_tokens,
            "generated_tokens": expected_output_tokens,
            "max_tokens_limit": max_output_tokens,
            "is_truncated": is_truncated,
            "ttft_ms": round(prompt_ingestion_latency_ms, 1),
            "total_latency_ms": round(total_latency_ms, 1),
            "tokens_per_second": 35.0
        }


# Instantiate engine at module level
engine = LLMGenerationEngine()

# --- HEADER SECTION ---
st.title("16.1 LLM Generation: Context Injection, Decoding Parameters, Temperature Dynamics, & Token Headroom")
st.caption("Chapter 16: Generation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In the Retrieval-Augmented Generation lifecycle, **Generation** represents the final synthesis stage where 
the autoregressive Transformer model consumes the compiled prompt payload (retrieved documents, user question, 
and system guardrails) and predicts the output sequence token by token.

While upstream retrieval and context engineering provide the factual substrate, **how the LLM produces tokens** 
is governed by generation parameters and decoding strategies. 

Misconfigured generation parameters trigger three catastrophic production failures:
1. **High-Temperature Hallucination**: Setting temperature too high (>0.7) forces the softmax distribution to flatten, causing the model to sample improbable tokens that stray from retrieved evidence.
2. **Abrupt Output Truncation**: Setting `max_tokens` too close to the estimated length cuts off sentences mid-thought, corrupting downstream JSON envelopes.
3. **Latency Explosion**: Bloated prompt context inflates Time-To-First-Token (TTFT), degrading interactive chat SLAs.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The LLM Generation Subsystem", divider="rainbow")

st.markdown("""
The generation subsystem operates as an autoregressive decoding loop. It ingests the combined prompt, 
populates the Key-Value (KV) cache, and iteratively computes logits, scales them via temperature, 
and applies sampling filters until a stop token is reached.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GenerationArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // INPUT ENVELOPE
        subgraph cluster_input {
            label="1. Prompt Payload Assembly";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            SystemPrompt [label="System Directive & Persona\\n(Closed-world epistemic rules)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            ContextPayload [label="XML Enveloped Context\\n(<document id='DOC-1'>...)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            UserQuery [label="User Question Envelope\\n(<query>What is the SLA?</query>)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // KV CACHING & PREFILL
        subgraph cluster_prefill {
            label="2. Prompt Prefill & KV Caching";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            Tokenizer [label="Tokenizer Encoding\\n(Maps text to token integer IDs)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            KVCache [label="Transformer KV Caching\\n(Parallel prefill of context tokens)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // AUTOREGRESSIVE DECODING LOOP
        subgraph cluster_decoding {
            label="3. Autoregressive Decoding Loop";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ForwardPass [label="Transformer Forward Pass\\n(Computes unnormalized vocabulary logits)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            TempScaling [label="Temperature Logit Scaling\\n(scaled_logits = logits / T)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            SamplingFilter [label="Top-P & Top-K Truncation\\n(Discards low-probability long tail)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            TokenSampler [label="Token Sampler / Argmax\\n(Selects next token ID)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
        }

        // STOP CRITERIA
        subgraph cluster_bounds {
            label="4. Boundary & Budget Validation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            BudgetCheck [label="Token Limit Auditor\\n(Generated < max_tokens limit)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
            StopCheck [label="Stop Sequence Detector\\n(Matches <eos> or </answer>)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
        }

        // FINAL OUTPUT
        FinalOutput [label="Completed Grounded Response\\n(Streamed or buffered to client)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];

        // EDGES
        SystemPrompt -> Tokenizer;
        ContextPayload -> Tokenizer;
        UserQuery -> Tokenizer;
        Tokenizer -> KVCache [label="Token IDs"];
        KVCache -> ForwardPass [label="Initial State"];
        ForwardPass -> TempScaling;
        TempScaling -> SamplingFilter;
        SamplingFilter -> TokenSampler;
        TokenSampler -> BudgetCheck;
        BudgetCheck -> StopCheck;
        StopCheck -> ForwardPass [label="Append Token (Loop)"];
        StopCheck -> FinalOutput [label="EOS or Max Reached"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Architectural Takeaways:
* **Prefill Phase vs Generation Phase**: Prompt processing is compute-bound and highly parallelized; token generation is memory-bandwidth bound and sequential (1 token per step).
* **KV Cache Dependency**: Fast generation relies entirely on reusing cached Key-Value tensors from earlier tokens without recalculating them.
* **Deterministic Decoding**: For enterprise factual grounding, temperature is held close to zero (0.0 to 0.2) to suppress generative variance.
""")

st.divider()

# ==============================================================================
# SECTION 2: SENDING RETRIEVED CONTEXT TO LLM
# ==============================================================================
st.header("2. Sending Retrieved Context to LLM (Context Injection, Latency Scaling & Attention Saturation)", divider="rainbow")

st.markdown("""
How retrieved documents are injected into the model determines both **computational cost** and **factual accuracy**. 
Injecting massive context payloads is not cost-free:
""")

inj_col1, inj_col2 = st.columns([1, 1])

with inj_col1:
    st.markdown("""
    ### 1. Time-To-First-Token (TTFT) Scaling:
    * **Prefill Latency**: Before the first output token can be generated, the model must execute a forward pass across every single prompt token.
    * **Empirical Latency Profile**: Ingesting 500 prompt tokens takes ~80 milliseconds, while ingesting 8,000 prompt tokens can take 800 to 1,500 milliseconds on enterprise GPUs.
    * **User Experience Impact**: Users perceive slow TTFT as system lag or unresponsiveness.
    
    ### 2. Context Window Saturation:
    * When context usage exceeds 70% of a model's window, cross-attention weights diffuse across competing keys.
    * Irrelevant tokens act as noise distractors, increasing hallucination rates by up to 3x.
    """)

with inj_col2:
    st.markdown("### TTFT Latency as a Function of Context Size")
    ctx_sizes = np.array([500, 1000, 2000, 4000, 8000, 16000])
    simulated_ttft = ctx_sizes * 0.12 + np.random.normal(0, 8, len(ctx_sizes))
    chart_data = pd.DataFrame({
        "Context Tokens": ctx_sizes,
        "TTFT Latency (ms)": np.clip(simulated_ttft, 20, 2500)
    }).set_index("Context Tokens")
    st.line_chart(chart_data)
    st.caption("Prompt prefill latency scales linearly with injected context tokens.")

st.divider()

# ==============================================================================
# SECTION 3: PROMPT CONSTRUCTION
# ==============================================================================
st.header("3. Prompt Construction (Template Assembly, Delimitation & Token Allocation)", divider="rainbow")

st.markdown("""
Prompt construction converts independent data streams into a cohesive input envelope. 
In production RAG, prompt assembly is executed by deterministic template compilers.
""")

pc_col1, pc_col2 = st.columns([1, 1])

with pc_col1:
    st.markdown("""
    ### Structural Delimiters:
    * **Boundary Envelopes**: XML containers (`<context>`, `<document id="...">`) separate factual content from operational directives.
    * **Token Space Partitioning**: 
      1. Fixed System Header (400 - 600 tokens)
      2. Dynamic Retrieved Context (capped at 50% - 60% of total window)
      3. User Question Envelope (100 - 300 tokens)
      4. Mandatory Headroom for Output (1,000 - 2,000 tokens)
    """)

with pc_col2:
    st.markdown("### Production Template Assembly Structure")
    st.code(r'''
=== SYSTEM DIRECTIVE ===
You are an enterprise knowledge assistant. Answer strictly from <context>.
Refuse if facts are missing. Never speculate.

=== RETRIEVED CONTEXT ===
<context>
  <document id="DOC-1" source="Runbook" updated="2024-03-01">
    Failover replication takes 120 seconds.
  </document>
</context>

=== USER QUERY ===
<query>How long does failover take?</query>

=== ASSISTANT RESPONSE ===
''', language="xml")

st.divider()

# ==============================================================================
# SECTION 4: CONTEXT-AWARE GENERATION
# ==============================================================================
st.header("4. Context-Aware Generation (Conditioned Probability Distributions & Attention Steering)", divider="rainbow")

st.markdown("""
During generation, the Transformer decoder computes conditional probability distributions:
`P(Token_t | Prompt, Token_1, ..., Token_t-1)`. 
In RAG, context-aware generation steers attention heads to prioritize keys derived from retrieved document blocks.
""")

cag_col1, cag_col2 = st.columns([1, 1])

with cag_col1:
    st.markdown("""
    ### Attention Steering Dynamics:
    * **Parametric vs Non-Parametric Competition**: Pre-trained weights favor generic, frequent token sequences (e.g., *"standard policy is 30 days"*). Injected context provides specific sequences (e.g., *"special pilot policy is 14 days"*).
    * **Attention Priming**: By placing the context within structured delimiters and referencing it in the system directive, self-attention query vectors compute high dot products with the document tokens, shifting the probability mass to the retrieved facts.
    * **Distractor Suppression**: When irrelevant chunks are pruned, attention heads focus exclusively on answer-bearing tokens.
    """)

with cag_col2:
    st.markdown("### Probability Mass Redistribution via Context Conditioning")
    prob_comp = pd.DataFrame([
        {"Token Candidate": "99.99% (From Context)", "Unconditioned Prior": "12.4%", "Conditioned with Context": "94.2%"},
        {"Token Candidate": "99.9% (Generic Common)", "Unconditioned Prior": "68.1%", "Conditioned with Context": "3.8%"},
        {"Token Candidate": "99.5% (Low SLA)", "Unconditioned Prior": "14.5%", "Conditioned with Context": "1.2%"},
        {"Token Candidate": "Unknown / Variable", "Unconditioned Prior": "5.0%", "Conditioned with Context": "0.8%"}
    ])
    st.dataframe(prob_comp, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 5: ANSWER GENERATION
# ==============================================================================
st.header("5. Answer Generation (Decoding Strategies, Greedy vs Sampling & TTFT vs Throughput)", divider="rainbow")

st.markdown("""
Once logits are computed for each step, a **decoding strategy** maps the vector of probabilities to a concrete token. 
Selecting the right strategy dictates whether the output is factual, diverse, or hallucinatory.
""")

dec_col1, dec_col2, dec_col3 = st.columns(3)

with dec_col1:
    with st.container(border=True):
        st.markdown("#### 1. Greedy Decoding (Argmax)")
        st.markdown("""
        * **Formula**: `Token = argmax P(token)`
        * **Characteristics**: 100% deterministic; always chooses the single highest-probability token.
        * **Best For**: Factual Q&A, structured JSON generation, citation compliance.
        * **Drawbacks**: Can produce repetitive loops in creative tasks, but ideal for RAG.
        """)

with dec_col2:
    with st.container(border=True):
        st.markdown("#### 2. Nucleus Sampling (Top-p)")
        st.markdown("""
        * **Formula**: Selects from smallest set of tokens where `Sum(P) >= p` (e.g., p = 0.90).
        * **Characteristics**: Dynamically adjusts candidate pool based on model confidence.
        * **Best For**: Conversational summaries requiring natural phrasing.
        * **Drawbacks**: Non-zero risk of sampling incorrect factual tokens.
        """)

with dec_col3:
    with st.container(border=True):
        st.markdown("#### 3. Top-k Sampling")
        st.markdown("""
        * **Formula**: Restricts candidate pool to fixed top `k` tokens (e.g., k = 40).
        * **Characteristics**: Hard cutoff regardless of probability distribution shape.
        * **Best For**: Creative writing.
        * **Drawbacks**: In factual RAG, hard cutoffs can either include bad tokens when confident or exclude valid alternatives.
        """)

st.divider()

# ==============================================================================
# SECTION 6: TEMPERATURE
# ==============================================================================
st.header("6. Temperature (Softmax Scaling, Sampling Entropy & Output Determinism)", divider="rainbow")

st.markdown("""
Temperature is a hyperparameter that scales the logit vector prior to softmax normalization:
`P(Token_i) = exp(Logit_i / Temperature) / Sum_j exp(Logit_j / Temperature)`.
It directly controls the **entropy** (randomness) of the predicted output distribution.
""")

tmp_col1, tmp_col2 = st.columns([1, 1])

with tmp_col1:
    st.markdown("""
    ### Temperature Operational Regimes:
    * **T = 0.0 (Greedy / Pseudo-Greedy)**: Probability mass collapses entirely onto the peak logit. The model becomes completely deterministic and repeatable. **Standard setting for enterprise RAG (compliance, data extraction, audit trails).**
    * **T = 0.1 to 0.3 (Low Temperature)**: Near-deterministic with minor linguistic variations. Ideal for customer-facing support bots where natural tone is desired without factual drift.
    * **T = 0.7 to 1.0 (High Temperature)**: Flattens the distribution, giving low-probability tokens an opportunity to be sampled. Produces creative metaphors, but causes severe factual hallucinations in RAG.
    """)

with tmp_col2:
    st.markdown("### Softmax Probability Distribution across Temperatures")
    test_logits = {"99.99%": 4.5, "99.9%": 3.8, "99.0%": 2.1, "Unknown": 1.0}
    t_0 = engine.apply_temperature(test_logits, 0.01)
    t_mid = engine.apply_temperature(test_logits, 0.5)
    t_high = engine.apply_temperature(test_logits, 1.2)

    dist_df = pd.DataFrame({
        "Token": list(test_logits.keys()),
        "T = 0.0 (Greedy)": [f"{t_0[k]*100:.1f}%" for k in test_logits],
        "T = 0.5 (Balanced)": [f"{t_mid[k]*100:.1f}%" for k in test_logits],
        "T = 1.2 (High Entropy)": [f"{t_high[k]*100:.1f}%" for k in test_logits]
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: MAX TOKENS
# ==============================================================================
st.header("7. Max Tokens (Generation Headroom, Truncation Protection & Cost Predictability)", divider="rainbow")

st.markdown("""
The `max_tokens` hyperparameter sets an absolute upper bound on the number of completion tokens generated. 
Misconfiguring `max_tokens` is one of the most common causes of silent production failures.
""")

mt_col1, mt_col2 = st.columns([1, 1])

with mt_col1:
    st.markdown("""
    ### Sizing `max_tokens` Correctly:
    * **Under-Allocation Danger**: If an LLM needs 350 tokens to produce a structured JSON response and `max_tokens` is set to 300, the output cuts off with an unclosed bracket (`"summary": "The...`), crashing downstream parsers.
    * **Over-Allocation & Cost Predictability**: While unused tokens are not billed by API providers, runaway conversational generations can consume excessive tokens if the model enters a loop.
    * **Stop Sequences as Safety Guards**: Always pair `max_tokens` with explicit stop sequences (e.g., `["</answer>", "\n\nUser:"]`) so generation halts as soon as the answer is complete.
    """)

with mt_col2:
    st.markdown("### Production Token Budget Allocation Model")
    budget_breakdown = pd.DataFrame([
        {"Zone": "System Directive & Rules", "Typical Allocation": "400 - 800 tokens", "Flexibility": "Fixed Envelope"},
        {"Zone": "Conversation History", "Typical Allocation": "1,000 - 2,000 tokens", "Flexibility": "Sliding Window / Pruned"},
        {"Zone": "Retrieved Document Context", "Typical Allocation": "2,000 - 4,500 tokens", "Flexibility": "Budget Capped Knapsack"},
        {"Zone": "Max Generation Output (`max_tokens`)", "Typical Allocation": "1,000 - 2,000 tokens", "Flexibility": "Non-Negotiable Reserve"}
    ])
    st.dataframe(budget_breakdown, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Enterprise LLM Generation Engine", divider="rainbow")

st.markdown("""
Below is the production implementation of `LLMGenerationEngine`. It manages prompt assembly, 
temperature scaling, entropy metrics, and token generation simulation in module scope.
""")

st.code(r'''
class LLMGenerationEngine:
    """
    Enterprise LLM Generation Orchestrator:
    - Assembles XML-delimited prompt payloads.
    - Applies temperature logit scaling and calculates distribution entropy.
    - Simulates autoregressive generation latency, TTFT, and throughput.
    - Enforces hard max_token limits and stop sequences.
    """
    def __init__(self, vocab_sample: Optional[Dict[str, float]] = None):
        self.vocab_sample = vocab_sample or {
            "Enterprise": 4.2, "SLA": 4.0, "guarantees": 3.8,
            "99.99%": 4.5, "uptime": 3.9, "for": 2.1, "Tier-1": 3.7
        }

    def compile_prompt(self, system_instruction: str, context_chunks: List[str], user_query: str) -> str:
        context_body = "\n".join([f"  <document id='DOC-{i+1}'>\n    {c}\n  </document>" for i, c in enumerate(context_chunks)])
        return (
            f"=== SYSTEM DIRECTIVE ===\n{system_instruction}\n\n"
            f"=== RETRIEVED CONTEXT ===\n<context>\n{context_body}\n</context>\n\n"
            f"=== USER QUERY ===\n<query>{user_query}</query>\n\n"
            f"=== ASSISTANT RESPONSE ==="
        )

    def apply_temperature(self, logits: Dict[str, float], temperature: float) -> Dict[str, float]:
        tokens = list(logits.keys())
        raw_vals = np.array([logits[t] for t in tokens], dtype=np.float64)
        if temperature <= 0.01:
            probs = np.zeros_like(raw_vals)
            probs[np.argmax(raw_vals)] = 1.0
        else:
            scaled = raw_vals / max(temperature, 0.01)
            exp_vals = np.exp(scaled - np.max(scaled))
            probs = exp_vals / np.sum(exp_vals)
        return {tokens[i]: float(probs[i]) for i in range(len(tokens))}

    def calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        p_vals = np.array([p for p in probabilities.values() if p > 0.0], dtype=np.float64)
        return float(-np.sum(p_vals * np.log2(p_vals)))

    def simulate_token_generation(
        self, prompt_length_tokens: int, max_output_tokens: int, temperature: float
    ) -> Dict[str, Any]:
        prompt_latency_ms = (prompt_length_tokens / 500.0) * 1000.0
        output_tokens = min(max_output_tokens, int(45 + 15 * (1.0 - temperature)))
        generation_latency_ms = (output_tokens / 35.0) * 1000.0
        return {
            "prompt_tokens": prompt_length_tokens,
            "generated_tokens": output_tokens,
            "is_truncated": output_tokens >= max_output_tokens,
            "ttft_ms": round(prompt_latency_ms, 1),
            "total_latency_ms": round(prompt_latency_ms + generation_latency_ms, 1)
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Generation Hyperparameter & Decoding Simulator", divider="rainbow")

st.markdown("""
Experiment with **Temperature**, **Max Output Tokens**, and **Prompt Length** to observe how decoding parameters 
alter token probability distributions, entropy, latency, and truncation risks in real time.
""")

sample_chunks = [
    "Enterprise Tier-1 SLA guarantees 99.99% uptime for core banking services.",
    "Failover to backup secondary region initiates automatically within 120 seconds of outage detection."
]

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Generation Hyperparameter Controls")
    sim_temperature = st.slider(
        "Temperature (Softmax Scaling):",
        min_value=0.0,
        max_value=1.5,
        value=0.2,
        step=0.05,
        help="0.0 = Deterministic Greedy. >0.7 = High Entropy / Hallucination Risk."
    )
    sim_max_tokens = st.slider(
        "Max Output Tokens (`max_tokens`):",
        min_value=20,
        max_value=200,
        value=60,
        step=5,
        help="Hard ceiling on generation length."
    )
    sim_prompt_tokens = st.slider(
        "Injected Prompt Size (Tokens):",
        min_value=500,
        max_value=8000,
        value=2500,
        step=250,
        help="Simulates prompt prefill token load."
    )
    sim_query = st.text_input("User Query:", value="What is the guaranteed uptime SLA?")

with wb_c2:
    st.markdown("### Live Decoding Metrics & Probabilities")
    # Step 1: Probability distribution calculation
    prob_dist = engine.apply_temperature(engine.vocab_sample, sim_temperature)
    dist_entropy = engine.calculate_entropy(prob_dist)

    # Step 2: Simulation metrics
    sim_res = engine.simulate_token_generation(
        prompt_length_tokens=sim_prompt_tokens,
        max_output_tokens=sim_max_tokens,
        temperature=sim_temperature
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Time-To-First-Token", f"{sim_res['ttft_ms']} ms")
    m2.metric("Total Latency", f"{sim_res['total_latency_ms']} ms")
    m3.metric("Distribution Entropy", f"{dist_entropy:.2f} bits")

    if sim_res["is_truncated"]:
        st.error(f"⚠️ Truncation Warning! Generation hit the {sim_max_tokens} max_tokens ceiling and was cut off.")
    else:
        st.success(f"✅ Generation completed cleanly within {sim_res['generated_tokens']} tokens.")

    st.markdown("#### Candidate Token Probabilities (Next Token Prediction):")
    token_chart_data = pd.DataFrame({
        "Token": list(prob_dist.keys()),
        "Probability (%)": [v * 100 for v in prob_dist.values()]
    }).sort_values(by="Probability (%)", ascending=False).head(6)
    st.bar_chart(token_chart_data.set_index("Token"))

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the mechanical difference between the Prompt Prefill Phase and the Token Generation Phase in LLM inference?**"):
    st.markdown("""
    **Answer**:
    * **Prefill Phase (Prompt Ingestion)**: The inference engine processes all prompt tokens in parallel in a single forward pass. This phase is **compute-bound** (matrix multiplication of large matrices) and scales with prompt token length. It populates the Key-Value (KV) cache.
    * **Generation Phase (Autoregressive Token Generation)**: The engine predicts one token at a time sequentially. Each step requires reading the entire KV cache and model weights from High Bandwidth Memory (HBM) to compute logits for a single token. This phase is **memory-bandwidth bound**.
    * **Production Implication**: Large context windows increase Time-To-First-Token (TTFT), but generation throughput (tokens per second) remains relatively constant.
    """)

with st.expander("**Q2: Why is temperature = 0.0 standard for enterprise RAG, and why does temperature > 0.7 cause hallucinations?**"):
    st.markdown("""
    **Answer**:
    Temperature scales unnormalized logits: `scaled_logits = logits / T`.
    * When `T = 0.0`, the softmax function collapses to an argmax operation. The model deterministically selects the single token with highest probability. Because the prompt conditions the model on retrieved facts, the highest-probability token is almost always the grounded evidence token.
    * When `T > 0.7`, the softmax denominator exponent flattens the distribution. Low-probability tokens receive artificially elevated sampling probabilities. The sampler occasionally selects a low-probability token (e.g., an incorrect number or date), after which subsequent autoregressive generation is forced to rationalize the erroneous token, producing an ungrounded hallucination.
    """)

with st.expander("**Q3: How does the Key-Value (KV) cache optimize autoregressive generation in RAG pipelines?**"):
    st.markdown("""
    **Answer**:
    In self-attention, token `t` attends to all prior tokens `1` to `t-1`. Without caching, computing step `t` would require recalculating the Key and Value projections for all preceding tokens, resulting in O(N^2) complexity per token.
    The KV cache stores the Key and Value activation tensors for all prior tokens in GPU memory. At each new generation step, the model only computes the Query, Key, and Value for the single newly generated token, and appends the new Key and Value to the cache. This reduces per-token computational complexity to O(N).
    """)

with st.expander("**Q4: What causes silent output truncation in RAG microservices, and how do you prevent it?**"):
    st.markdown("""
    **Answer**:
    Silent output truncation occurs when the generation loop reaches the `max_tokens` ceiling before emitting an end-of-sequence (`<eos>`) token.
    In structured JSON pipelines, this truncates unclosed braces, causing JSON parser exceptions.
    Prevention strategies:
    1. Allocate generous generation headroom (`max_tokens >= expected_length * 1.5`).
    2. Check the API response `finish_reason`. If `finish_reason == "length"`, trigger an automatic retry with expanded headroom or alert the system.
    3. Enforce stop sequences (e.g., `["</json>", "\n\nUser:"]`).
    """)

with st.expander("**Q5: Why is Top-p (Nucleus) sampling preferred over Top-k sampling when sampling is required in RAG?**"):
    st.markdown("""
    **Answer**:
    * **Top-k**: Always retains a fixed number `k` of candidate tokens regardless of the probability distribution shape. If the model is 99% confident in a single token, Top-k still forces 39 low-probability distractors into the sampling pool.
    * **Top-p (Nucleus)**: Retains only the dynamic set of tokens whose cumulative probability exceeds threshold `p` (e.g., 0.90). If the model is confident, the nucleus shrinks to 1 or 2 tokens. If the model is uncertain, the nucleus expands. This dynamically prevents sampling improbable tokens during factual generation.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Setting Temperature > 0.5 on Factual Inquiries**
    * **Failure Mode**: Setting high temperature to make answers sound *"creative"* causes the sampler to pick improbable numbers, creating hallucinations in legal or financial responses.
    * **Solution**: Lock temperature at 0.0 to 0.2 for all factual RAG workflows.
    """)

    st.error("""
    **Production Trap 2: Ignoring finish_reason == 'length'**
    * **Failure Mode**: Silently accepting API responses that terminated due to token length limits causes truncated answers and broken JSON objects to enter customer databases.
    * **Solution**: Always assert `finish_reason == 'stop'` in production gateway wrappers.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Context Saturation (>80% Window Fill)**
    * **Failure Mode**: Injecting 7,500 tokens of retrieved documents into an 8,192-token context window causes severe prefill latency (TTFT > 2s) and leaves inadequate headroom for answers.
    * **Solution**: Cap context injection at 50% to 60% of the total context window.
    """)

    st.warning("""
    **Production Trap 4: Missing Stop Sequences**
    * **Failure Mode**: Without explicit stop sequences, fine-tuned or open-source models can generate runaway commentary or start hallucinating follow-up user dialogues.
    * **Solution**: Explicitly configure stop tokens (e.g., `["</answer>", "\n\nUser:"]`).
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Time-To-First-Token (TTFT)")
        st.caption("What determines TTFT in RAG pipelines?")
        st.markdown("**Back**: Prompt prefill computation time, which scales with the total number of injected prompt context tokens.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Temperature Softmax Formula")
        st.caption("How does temperature mathematically scale logits?")
        st.markdown("**Back**: Divides unnormalized logits by T (`logits / T`) before exponentiation, sharpening distribution at T < 1 and flattening at T > 1.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Key-Value (KV) Caching")
        st.caption("Why is the KV cache critical for autoregressive decoding?")
        st.markdown("**Back**: Eliminates recomputing attention key and value projections for prior tokens, reducing per-token step complexity to O(N).")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Greedy Decoding (Argmax)")
        st.caption("When should greedy decoding be mandated in RAG?")
        st.markdown("**Back**: For compliance, auditing, structured JSON, and factual Q&A where determinism and repeatability are strictly required.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LLM generation dynamics before proceeding:")

c1 = st.checkbox("I understand the architectural distinction between the parallel prefill phase (TTFT) and sequential token generation.", value=True)
c2 = st.checkbox("I can explain how temperature mathematically scales logits and why T <= 0.2 is mandatory for factual RAG.", value=True)
c3 = st.checkbox("I understand the purpose and mechanics of the Key-Value (KV) cache in autoregressive decoding.", value=True)
c4 = st.checkbox("I know how to configure `max_tokens` headroom and inspect `finish_reason` to prevent silent output truncation.", value=True)
c5 = st.checkbox("I can differentiate between greedy decoding, nucleus (Top-p) sampling, and Top-k sampling.", value=True)
c6 = st.checkbox("I know how to partition the prompt token budget to avoid context window saturation and attention degradation.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **LLM Generation Mastery Achieved!** You are ready to configure high-performance, deterministic RAG generation pipelines.")
