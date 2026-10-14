import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.1 Why Evaluate LLMs - LLM Engineering",
    page_icon="📊",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class MultiDimensionalEvaluator:
    """
    Self-contained in-memory multi-dimensional LLM evaluator implementing:
    - Quality, Reliability, Accuracy, Robustness, Safety, Cost, and Latency scoring
    - Pareto trade-off composite index calculation
    - SLA threshold gating
    """
    
    MODELS = {
        "Frontier-Reasoner (o1-preview / Claude 3.5 Sonnet)": {
            "quality": 0.96, "reliability": 0.94, "accuracy": 0.95, "robustness": 0.91,
            "safety": 0.95, "cost_per_1m_tokens": 15.00, "ttft_ms": 1400, "tpot_ms": 35
        },
        "Mid-Tier Workhorse (GPT-4o-mini / Llama 3.3 70B)": {
            "quality": 0.88, "reliability": 0.89, "accuracy": 0.87, "robustness": 0.84,
            "safety": 0.92, "cost_per_1m_tokens": 0.60, "ttft_ms": 320, "tpot_ms": 14
        },
        "Edge-Optimized Small (Llama 3.2 3B / Gemma 2 2B)": {
            "quality": 0.72, "reliability": 0.75, "accuracy": 0.71, "robustness": 0.68,
            "safety": 0.85, "cost_per_1m_tokens": 0.10, "ttft_ms": 95, "tpot_ms": 6
        }
    }

    def evaluate_model(self, model_name: str, weights: Dict[str, float]) -> Dict[str, Any]:
        specs = self.MODELS[model_name]
        
        # Normalize cost and latency so higher is better ([0, 1])
        # Normalized Cost = 1.0 - (Cost / 20.0)
        norm_cost = max(0.0, 1.0 - (specs["cost_per_1m_tokens"] / 20.0))
        # Normalized Latency = 1.0 - (TTFT / 2000.0)
        norm_latency = max(0.0, 1.0 - (specs["ttft_ms"] / 2000.0))
        
        composite_score = (
            weights["quality"] * specs["quality"] +
            weights["reliability"] * specs["reliability"] +
            weights["accuracy"] * specs["accuracy"] +
            weights["robustness"] * specs["robustness"] +
            weights["safety"] * specs["safety"] +
            weights["cost"] * norm_cost +
            weights["latency"] * norm_latency
        ) / sum(weights.values())
        
        return {
            "model": model_name,
            "quality": specs["quality"],
            "reliability": specs["reliability"],
            "accuracy": specs["accuracy"],
            "robustness": specs["robustness"],
            "safety": specs["safety"],
            "cost_usd": specs["cost_per_1m_tokens"],
            "ttft_ms": specs["ttft_ms"],
            "composite_score": round(composite_score, 4)
        }

# --- TOP-LEVEL HEADER ---
st.title("21.1 Why Evaluate LLMs? The 7 Pillars of Production Model Evaluation")
st.caption("Chapter 21: LLM Evaluation | Master Architecture Specification")
st.markdown("""
Deploying foundation models in enterprise production requires rigorous, quantitative evaluation. 
Unlike classical software with deterministic assertion tests, LLM outputs are stochastic, generative, and open-ended. 
Evaluating an LLM demands a comprehensive multi-dimensional framework balancing **Quality**, **Reliability**, 
**Accuracy**, **Robustness**, **Safety**, **Cost**, and **Latency**.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: The Multi-Dimensional Evaluation Radar", divider="rainbow")

st.markdown("""
The architecture below outlines the enterprise evaluation matrix spanning statistical, linguistic, 
economic, and system-level performance vectors.
""")

dot_eval = """
digraph EvalBlueprint {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_cognitive {
        label = "Cognitive & Task Performance Pillars";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        PillarQuality [label="1. Model Quality\\n[Fluency, Coherence, Alignment]", fillcolor="#1E3A8A", color="#3B82F6"];
        PillarAccuracy [label="3. Accuracy & Factuality\\n[Factual Grounding, Hallucination Rate]", fillcolor="#1E3A8A", color="#3B82F6"];
        PillarReliability [label="2. Reliability\\n[Determinism, Semantic Variance]", fillcolor="#1E3A8A", color="#3B82F6"];
        PillarRobustness [label="4. Robustness\\n[Perturbation Invariance, OOD Resilience]", fillcolor="#1E3A8A", color="#3B82F6"];
    }

    subgraph cluster_governance {
        label = "Governance & System Performance Pillars";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        PillarSafety [label="5. Safety & Compliance\\n[Toxicity, Bias, PII Leakage, Red Teaming]", fillcolor="#78350F", color="#D97706"];
        PillarCost [label="6. Cost Economics\\n[Token Pricing, Cache Hit Rates, FLOPS]", fillcolor="#78350F", color="#D97706"];
        PillarLatency [label="7. Latency & Throughput\\n[TTFT, TPOT, P99 Concurrency Saturation]", fillcolor="#78350F", color="#D97706"];
    }

    subgraph cluster_output {
        label = "Decision & Deployment Gate";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#10B981";
        Scorecard [label="Composite Evaluation Scorecard\\n[Pareto Frontier & SLA Threshold Gate]", fillcolor="#064E3B", color="#10B981"];
        DeployGate [label="CI/CD Release Decision\\n[Canary Promotion or Regression Rollback]", fillcolor="#064E3B", color="#10B981"];
    }

    PillarQuality -> Scorecard;
    PillarAccuracy -> Scorecard;
    PillarReliability -> Scorecard;
    PillarRobustness -> Scorecard;
    PillarSafety -> Scorecard;
    PillarCost -> Scorecard;
    PillarLatency -> Scorecard;

    Scorecard -> DeployGate [label="Passes SLAs", color="#22C55E", penwidth=2];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_eval, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - MODEL QUALITY ---
st.header("2. Model Quality: Coherence, Fluency, & Instruction Following", divider="rainbow")

st.markdown("""
**Model Quality** measures the fundamental linguistic and structural competence of generated responses. 
It answers: *Does the model produce grammatically fluent, coherent, and instruction-aligned completions?*
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Key Dimensions of Quality")
    st.markdown("""
    - **Instruction Adherence (IFEval)**: Verifies whether the model complies with strict formatting constraints (e.g. *"Answer in exactly 3 bullet points, using fewer than 50 words, without mentioning the letter 'e'"*).
    - **Semantic Coherence**: Evaluates logical flow, absence of circular reasoning, and consistent topical depth.
    - **Style & Persona Consistency**: Assesses whether the model maintains the requested brand voice, tone, or technical register.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: How New Models Follow This
    Modern benchmarks (such as **IFEval - Instruction Following Evaluation**) move away from subjective scoring 
    toward verifiable programmatic assertions (e.g. regex counting of bullet points, JSON syntax verification). 
    Frontier models score >85% on IFEval, whereas early models scored below 40%.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - RELIABILITY ---
st.header("3. Reliability: Determinism, Semantic Variance, & Failure Rates", divider="rainbow")

st.markdown("""
**Reliability** measures output consistency across repeated invocations. 
In enterprise software, an API that produces an exceptional answer 80% of the time but crashes or hallucinating garbage 20% of the time is unusable.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Quantifying Semantic Variance")
    st.markdown("""
    - **Self-Consistency (Wang et al.)**: Sample `N = 10` responses at `temperature = 0.7`. Measure agreement rate across responses.
    - **Semantic Entropy (Farquhar et al., Nature 2024)**: Clusters sampled answers into semantic equivalence classes using an NLI cross-encoder. High entropy indicates unreliability and impending hallucination.
    - **Schema Conformance Rate**: Fraction of turns where structured JSON outputs conform to the strict Pydantic model without retry loops.
    """)

with col2:
    st.info("""
    **The Temperature Dilemma**:
    Setting `temperature = 0.0` provides greedy argmax decoding, maximizing consistency. 
    However, on multi-GPU tensor parallel clusters, subtle floating-point non-associativity (`(a + b) + c != a + (b + c)`) 
    can still introduce non-deterministic token differences unless deterministic kernels are strictly enforced.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - ACCURACY ---
st.header("4. Accuracy: Factuality, Domain Knowledge, & Hallucination Rates", divider="rainbow")

st.markdown("""
**Accuracy** measures the objective correctness of factual claims generated by the model. 
Unlike creative writing, enterprise workflows (financial analysis, medical diagnosis, legal review) require near-zero hallucination tolerance.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Standard Academic Benchmarks")
    st.markdown("""
    - **MMLU (Massive Multitask Language Understanding)**: 57 subjects spanning STEM, humanities, and social sciences.
    - **GSM8K & MATH**: Multi-step mathematical reasoning and symbolic derivation.
    - **HumanEval & MBPP**: Python code synthesis evaluated against automated unit tests (`pass@1`).
    - **HaluEval**: Domain-specific QA and summarization benchmark designed to induce and measure hallucination rates.
    """)

with col2:
    st.error("""
    **The Goodhart's Law Benchmark Trap**:
    *"When a measure becomes a target, it ceases to be a good measure."* 
    Modern foundation models frequently suffer from training set contamination (benchmark data present in web pre-training crawls). 
    Production systems must evaluate models against private, continuously refreshed internal evaluation suites.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - ROBUSTNESS ---
st.header("5. Robustness: Perturbation Invariance & Adversarial Resilience", divider="rainbow")

st.markdown("""
**Robustness** measures how gracefully a model handles noise, input perturbations, and out-of-distribution prompts. 
A fragile model produces radically different answers when a minor typo or whitespace change is introduced.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Robustness Perturbation Tests")
    st.markdown("""
    - **Character Perturbations**: Simulating common keyboard typos, character drops, and casing flips.
    - **Paraphrasing Invariance**: Rephrasing the user query in colloquial, formal, and broken English to verify that the core factual answer remains identical.
    - **Adversarial Distractors**: Injecting irrelevant sentences into context windows to test whether the model gets distracted by spurious correlations.
    """)

with col2:
    st.code("""
# Robustness Score Formula:
# Perturbation Invariance = Semantic_Similarity(Output(x), Output(Perturb(x)))
# Ideal Robustness: Perturbation Invariance > 0.95 across 100 random typo injections
    """, language="python")

st.divider()

# --- SECTION 6: TOPIC 5 - SAFETY ---
st.header("6. Safety: Toxicity, Bias, PII Redaction, & Red Teaming", divider="rainbow")

st.markdown("""
**Safety** evaluates whether a model produces harmful, toxic, legally hazardous, or socially discriminatory content. 
Safety evaluation forms the foundation of Responsible AI deployments.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Key Safety Dimensions")
    st.markdown("""
    - **Toxicity & Hate Speech**: Assessed using automated classifiers (Perspective API, Llama Guard).
    - **PII Leakage**: Verifying that the model refuses to regurgitate Personally Identifiable Information (SSNs, credit cards, emails).
    - **CBRN & Cyber Risk**: Evaluating model refusals against Chemical, Biological, Radiological, and Cyber-attack exploitation prompts.
    - **Automated Red Teaming**: Deploying adversarial attacker LLMs to iteratively discover prompt vulnerabilities.
    """)

with col2:
    st.warning("""
    **The Over-Refusal Penalty (False Positive Censorship)**:
    An overly paranoid safety alignment makes a model useless (e.g. refusing to answer *"How do I kill a Python process?"* because of the word 'kill'). 
    Safety benchmarks must measure both **True Positive Refusals** and **False Positive Refusal Rates**.
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - COST ---
st.header("7. Cost Economics: Token Pricing, Cache Hits, & FLOPS Budget", divider="rainbow")

st.markdown("""
In enterprise deployments serving millions of requests daily, financial viability dictates model selection. 
A model that is 3% more accurate but costs 30x more per token cannot be deployed at scale.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Economic Cost Modeling")
    st.markdown("""
    Total inference cost is governed by prompt token volume, completion token volume, and caching:
    
    `Total_Cost = (Tokens_Prompt * Price_Input) + (Tokens_Gen * Price_Output) - (Tokens_Cached * Discount_Cache)`
    
    - **Input vs Output Asymmetry**: Output tokens are typically 3x to 5x more expensive than input tokens due to auto-regressive memory bandwidth constraints.
    - **Context Caching Economics**: Anthropic and Google provide up to 75% discounts on prompt tokens that hit the persistent KV cache.
    """)

with col2:
    st.info("""
    **The Small Model Routing Architecture**:
    Production systems route 80% of low-complexity queries (classification, entity extraction) to small 3B-8B models (0.10 USD / 1M tokens) 
    and escalate only the 20% high-complexity queries to frontier reasoning models (15.00 USD / 1M tokens), saving 70%+ on monthly inference bills.
    """)

st.divider()

# --- SECTION 8: TOPIC 7 - LATENCY ---
st.header("8. Latency & Throughput: TTFT, TPOT, & Concurrency Saturation", divider="rainbow")

st.markdown("""
User engagement and real-time interactive agents are strictly bound by latency. 
Evaluating latency requires decomposing execution into two distinct hardware phases: **Prefill** and **Decode**.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Critical Latency Metrics")
    st.markdown("""
    - **Time-to-First-Token (TTFT)**: Elapsed time from HTTP request submission to the emission of the first completion token. Dominated by **Prompt Prefill compute (Compute-Bound)**.
    - **Time-Per-Output-Token (TPOT) / Inter-Token Latency (ITL)**: Elapsed time between consecutive tokens during generation. Dominated by **Memory Bandwidth (Memory-Bound)**.
    - **P99 Latency under Concurrency**: How latency degrades when 50 concurrent requests saturate the GPU KV cache.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: Speculative Decoding
    Modern serving runtimes (vLLM, TensorRT-LLM, SGLang) leverage **Speculative Decoding**: a draft model (e.g. 1B) emits 5 tokens 
    in parallel, and the target model (e.g. 70B) verifies all 5 tokens in a single forward pass, slashing TPOT by 2x to 3x!
    """)

st.divider()

# --- SECTION 9: PRODUCTION CODE IMPLEMENTATION ---
st.header("9. Production Implementation: Multi-Dimensional Evaluation Suite", divider="rainbow")

st.markdown("""
Below is a complete, production-ready Python evaluation harness that scores models across all 7 pillars, 
calculates a weighted composite score, and enforces hard SLA assertion gates.
""")

st.code('''
from typing import Dict, Any, List

class ProductionModelEvaluationHarness:
    """
    Automated evaluation harness for multi-dimensional LLM benchmarking.
    """
    def __init__(self, sla_thresholds: Dict[str, float]):
        self.sla_thresholds = sla_thresholds

    def evaluate_candidate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a candidate model against production SLA gates."""
        violations = []

        # Check Accuracy SLA
        if metrics.get("accuracy", 0.0) < self.sla_thresholds.get("min_accuracy", 0.85):
            violations.append(f"Accuracy {metrics.get('accuracy')} below SLA {self.sla_thresholds.get('min_accuracy')}")

        # Check Safety SLA
        if metrics.get("safety_score", 0.0) < self.sla_thresholds.get("min_safety", 0.95):
            violations.append(f"Safety {metrics.get('safety_score')} below SLA {self.sla_thresholds.get('min_safety')}")

        # Check Latency P95 SLA
        if metrics.get("ttft_ms", 9999) > self.sla_thresholds.get("max_ttft_ms", 500):
            violations.append(f"TTFT {metrics.get('ttft_ms')}ms exceeds SLA {self.sla_thresholds.get('max_ttft_ms')}ms")

        # Check Cost SLA
        if metrics.get("cost_per_1m", 999.0) > self.sla_thresholds.get("max_cost_usd", 2.0):
            violations.append(f"Cost {metrics.get('cost_per_1m')} USD exceeds SLA {self.sla_thresholds.get('max_cost_usd')} USD")

        verdict = "PASSED" if len(violations) == 0 else "FAILED_SLA"
        return {
            "verdict": verdict,
            "violations_count": len(violations),
            "violations": violations,
            "metrics": metrics
        }
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Evaluation Workbench: Pareto Trade-Off Simulator", divider="rainbow")

st.markdown("""
Explore the real-world trade-offs across the 7 evaluation pillars. Customize the importance weights of 
Quality, Accuracy, Reliability, Safety, Cost, and Latency, and observe which model emerges on the **Pareto Frontier**.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Set Organizational Priority Weights")
    w_qual = st.slider("Quality Weight:", 0.0, 1.0, 0.8, 0.1)
    w_acc = st.slider("Accuracy Weight:", 0.0, 1.0, 0.9, 0.1)
    w_rel = st.slider("Reliability Weight:", 0.0, 1.0, 0.7, 0.1)
    w_rob = st.slider("Robustness Weight:", 0.0, 1.0, 0.6, 0.1)
    w_safe = st.slider("Safety Weight:", 0.0, 1.0, 0.95, 0.05)
    w_cost = st.slider("Cost Weight (Cheap = High):", 0.0, 1.0, 0.5, 0.1)
    w_lat = st.slider("Latency Weight (Fast = High):", 0.0, 1.0, 0.5, 0.1)

with col2:
    evaluator = MultiDimensionalEvaluator()
    weights_dict = {
        "quality": w_qual, "accuracy": w_acc, "reliability": w_rel,
        "robustness": w_rob, "safety": w_safe, "cost": w_cost, "latency": w_lat
    }
    
    benchmark_results = []
    for model_key in evaluator.MODELS:
        benchmark_results.append(evaluator.evaluate_model(model_key, weights_dict))
        
    df_bench = pd.DataFrame(benchmark_results)
    df_bench = df_bench.sort_values(by="composite_score", ascending=False)
    
    st.subheader("Leaderboard: Composite Scorecard")
    st.dataframe(df_bench[["model", "composite_score", "accuracy", "safety", "cost_usd", "ttft_ms"]], width="stretch")
    
    best_model = df_bench.iloc[0]["model"]
    st.success(f"🏆 Top Pareto Candidate for Your Weights: **{best_model}**")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: What is the difference between Compute-Bound Prefill and Memory-Bound Decode during inference?"):
    st.markdown("""
    **Answer**:
    - **Prompt Prefill Phase**: The model processes all prompt tokens in parallel. Because large matrix multiplications occur simultaneously across all sequence tokens, the GPU tensor cores operate near peak FLOPS capacity. Prefill is **Compute-Bound**.
    - **Auto-Regressive Decode Phase**: Tokens are emitted strictly one at a time. Each generated token requires loading all model weights (e.g. 140GB for a 70B model) from High Bandwidth Memory (HBM) into SRAM just to process a single token vector. Compute cores sit idle waiting for memory transfer. Decode is **Memory-Bound**.
    """)

with st.expander("Q2: How do you detect and mitigate Benchmark Contamination in LLM evaluation?"):
    st.markdown("""
    **Answer**:
    Benchmark contamination occurs when evaluation questions or answers leak into web pre-training datasets. 
    Detection & mitigation techniques:
    1. **N-gram Shingling**: Compute 8-gram to 13-gram overlap between benchmark prompts and pre-training corpora.
    2. **Perplexity Probing**: Check whether the model's perplexity on the benchmark text is anomalously lower than on non-benchmark text of equivalent complexity.
    3. **Dynamic Canaries & Private Holdout Sets**: Continuously generate synthetic or private enterprise evaluation datasets that have never been exposed to the public internet.
    """)

with st.expander("Q3: What is Semantic Entropy and how does it quantify LLM uncertainty?"):
    st.markdown("""
    **Answer**:
    Standard token-level entropy measures vocabulary spread at each position. However, two completely different token sequences can express the exact same meaning (*"Paris"* vs *"The capital of France"*). 
    **Semantic Entropy** samples multiple completions at non-zero temperature, clusters them into semantic equivalence classes using bi-directional entailment (NLI), and computes Shannon entropy over the discrete semantic clusters. High semantic entropy indicates genuine factual uncertainty.
    """)

with st.expander("Q4: Why does setting temperature to 0.0 not guarantee 100% bit-exact deterministic output in production clusters?"):
    st.markdown("""
    **Answer**:
    On multi-GPU clusters running Tensor Parallelism or pipeline parallelism, reduction operations (`AllReduce`) across GPUs sum floating-point tensors in non-deterministic order. 
    Because 16-bit floating-point addition (`bfloat16`, `float16`) is non-associative, slight differences in summation order alter low-order bits, occasionally shifting the argmax token index when top logits are nearly identical.
    """)

with st.expander("Q5: What is the False Positive Refusal Rate and why is it a critical safety metric?"):
    st.markdown("""
    **Answer**:
    The False Positive Refusal Rate measures how often an aligned model incorrectly refuses benign, harmless prompts that contain sensitive keywords (e.g., refusing to answer *"How do I eliminate a software bug?"* because of the word 'eliminate'). 
    Evaluating safety without measuring false positive refusals leads to over-censored, unusable models.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Evaluating Solely on Public Academic Benchmarks**
    Relying exclusively on MMLU or GSM8K to choose a production model. High academic scores often fail to correlate with enterprise domain performance, instruction following, or tool calling accuracy.
    """)
    st.error("""
    **TRAP 2 (Error): Ignoring P99 Latency Under Concurrent Load**
    Benchmarking latency with single sequential requests (`concurrency = 1`). In production, 50 simultaneous requests cause KV cache memory paging, causing P99 latency to spike 10x higher than single-stream benchmarks.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Static Eval Suites Without Dynamic Drift Detection**
    Running a fixed 100-question eval set once before deployment. As real-world user queries evolve, the static evaluation suite ceases to represent production query distribution drift.
    """)
    st.warning("""
    **TRAP 4 (Warning): Ignoring Output Token Cost Asymmetry**
    Calculating budget assuming uniform token prices. Generating verbose 1,000-token answers costs 4x more than processing 1,000 prompt tokens. Enforce concise generation formatting to control costs.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **TTFT**  
    Time-to-First-Token: Prompt prefill latency from request to first emitted token.
    """)
with fc2:
    st.markdown("""
    **TPOT**  
    Time-Per-Output-Token: Inter-token latency governed by GPU memory bandwidth.
    """)
with fc3:
    st.markdown("""
    **Semantic Entropy**  
    Entropy computed over semantic equivalence clusters rather than raw tokens.
    """)
with fc4:
    st.markdown("""
    **Pareto Frontier**  
    The set of models that provide optimal trade-offs where no single metric can improve without degrading another.
    """)

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist: Why Evaluate LLMs Mastery", divider="rainbow")

st.markdown("Verify your understanding of enterprise LLM evaluation by reviewing each criterion:")

chk1 = st.checkbox("I can articulate the 7 fundamental pillars of production LLM evaluation.", key="why_chk1")
chk2 = st.checkbox("I understand the architectural distinction between Compute-Bound Prefill (TTFT) and Memory-Bound Decode (TPOT).", key="why_chk2")
chk3 = st.checkbox("I know how Semantic Entropy quantifies hallucination likelihood across equivalence clusters.", key="why_chk3")
chk4 = st.checkbox("I can identify benchmark contamination and design private holdout evaluation datasets.", key="why_chk4")
chk5 = st.checkbox("I understand how to measure True Positive vs False Positive Safety Refusals.", key="why_chk5")
chk6 = st.checkbox("I can construct a composite Pareto score balancing accuracy, safety, cost, and latency.", key="why_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of Multi-Dimensional LLM Evaluation Systems!")
