import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="13.4 Fine-Tuning Problems - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionFailureAuditor:
    """
    Diagnostic auditor analyzing LLM fine-tuning health:
    - Overfitting bifurcation detection (Train vs Val divergence)
    - Catastrophic forgetting tracking across baseline capabilities
    - Repetition penalty and length imbalance metrics
    """
    def __init__(self, baseline_mmlu: float = 68.5, baseline_gsm8k: float = 54.0):
        self.baseline_mmlu = baseline_mmlu
        self.baseline_gsm8k = baseline_gsm8k

    def calculate_overfitting_score(self, train_loss: float, val_loss: float) -> Tuple[float, str]:
        """Calculates loss divergence ratio and assigns risk level."""
        divergence = val_loss - train_loss
        ratio = val_loss / max(0.001, train_loss)
        if ratio > 2.0 or divergence > 1.2:
            return float(ratio), "Critical Overfitting (Severe Memorization)"
        elif ratio > 1.4 or divergence > 0.5:
            return float(ratio), "Moderate Overfitting (Early Stopping Needed)"
        else:
            return float(ratio), "Healthy Generalization"

    def audit_forgetting(self, current_mmlu: float, current_gsm8k: float) -> Dict[str, Any]:
        """Calculates degradation in pre-existing general reasoning benchmarks."""
        delta_mmlu = current_mmlu - self.baseline_mmlu
        delta_gsm8k = current_gsm8k - self.baseline_gsm8k
        
        is_forgetting = (delta_mmlu < -5.0) or (delta_gsm8k < -8.0)
        return {
            "delta_mmlu": float(delta_mmlu),
            "delta_gsm8k": float(delta_gsm8k),
            "severity": "High Catastrophic Forgetting" if is_forgetting else "Acceptable Drift"
        }

    def simulate_training_pathologies(
        self,
        epochs: int = 5,
        dataset_skew: float = 0.8,
        replay_ratio: float = 0.05
    ) -> pd.DataFrame:
        """Simulates train vs val curves and general knowledge retention."""
        steps = np.arange(1, epochs * 50 + 1)
        records = []
        for s in steps:
            # Overfitting curve
            t_loss = 3.5 * np.exp(-s / 50.0) + 0.3
            # Val loss bifurcates if epochs > 2 and replay is low
            v_penalty = max(0.0, (s - 75) * 0.015) if replay_ratio < 0.1 else max(0.0, (s - 120) * 0.005)
            v_loss = 3.5 * np.exp(-s / 65.0) + 0.6 + v_penalty
            
            # Capability retention
            retention = max(20.0, self.baseline_mmlu - (s * 0.18 * (1.0 - replay_ratio)))
            
            records.append({
                "Step": int(s),
                "Train Loss": float(t_loss),
                "Validation Loss": float(v_loss),
                "MMLU General Score": float(retention)
            })
        return pd.DataFrame(records)

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("13.4 Fine-Tuning Problems")
st.caption("Chapter 13: LLM Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
Fine-tuning an LLM is easy; fine-tuning an LLM **without breaking its reasoning, hallucinating false facts, 
or collapsing into repetitive loops** is where engineering expertise is proven.
Because fine-tuning shifts billions of interdependent parameters using only thousands of data points, 
models are exceptionally vulnerable to distribution collapse, catastrophic forgetting, sycophancy, and prompt leakage.

In this lesson, you will master:
1. Detecting and rectifying Overfitting and memorization bifurcation.
2. Catastrophic Forgetting: Why fine-tuning on SQL breaks grade-school math, and how replay buffers resolve it.
3. Data Leakage: Identifying sneaky contamination vectors between training prompts and evaluation benchmarks.
4. Poor Instruction Formatting: Eradicating double EOS tokens, missing role delimiters, and generation truncations.
5. Hallucination after tuning: The hazard of superficial alignment and sycophantic factual drift.
6. Dataset Imbalance: Addressing verb collapse, length bias, and task skew through stratified data balancing.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Post-Training Failure Landscape & Diagnostic Taxonomy", divider="rainbow")

st.markdown("""
When fine-tuning goes wrong, failures manifest across data pipelines, parameter dynamics, or downstream generation behavior.
""")

failure_dot = """
digraph FailureTaxonomy {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_data {
        label = "Layer 1: Data-Level Pathologies";
        color = "#334155";
        fontcolor = "#94A3B8";
        DataLeak [label="Data Leakage\\n(Train / Val Contamination)", fillcolor="#0F172A"];
        BadFormat [label="Malformed Templates\\n(Broken Headers / Delimiters)", fillcolor="#0F172A"];
        Skew [label="Dataset Imbalance\\n(Length & Task Skew)", fillcolor="#0F172A"];
    }

    subgraph cluster_optimization {
        label = "Layer 2: Optimization Failures";
        color = "#334155";
        fontcolor = "#94A3B8";
        Overfit [label="Severe Overfitting\\n(Loss Divergence / Memorization)", fillcolor="#B91C1C"];
        Forgetting [label="Catastrophic Forgetting\\n(General Reasoning Collapse)", fillcolor="#B91C1C"];
    }

    subgraph cluster_behavior {
        label = "Layer 3: Generation Pathologies";
        color = "#334155";
        fontcolor = "#94A3B8";
        Hallucinate [label="Factual Hallucination\\n(Sycophancy & Confabulation)", fillcolor="#B45309"];
        Repetition [label="Degenerate Repetition\\n(Endless Loop Generation)", fillcolor="#B45309"];
    }

    DataLeak -> Overfit [label="Inflated Validation Metrics"];
    BadFormat -> Repetition [label="Missing EOS / Role Markers"];
    Skew -> Forgetting [label="Narrow Domain Weight Shift"];
    Overfit -> Hallucinate [label="Superficial Token Memorization"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(failure_dot, use_container_width=True)

st.markdown("""
### The Diagnostic Triad
1. **Data Diagnostics**: Auditing token length histograms, MinHash similarity clusters, and role delimiter integrity.
2. **Loss Surface Diagnostics**: Tracking the generalization gap `Delta = Validation_Loss - Training_Loss` and gradient norms.
3. **Capability Auditing**: Running regression benchmarks (MMLU, ARC, GSM8K) to ensure specialized tuning has not eroded core reasoning.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - OVERFITTING
# ==============================================================================
st.header("2. Overfitting", divider="rainbow")

st.markdown("""
In language models, **overfitting** does not just mean higher validation loss—it produces distinct generative pathologies:
1. **Verbatim Memorization**: The model recites exact training paragraphs even when the user asks a completely different question with similar keywords.
2. **Loss Divergence**: Training loss continues driving toward zero (`< 0.5`), while validation loss curves upward aggressively.
3. **Repetition Loops**: The probability distribution collapses onto a few high-frequency training tokens, causing infinite loops (*"the the the the"*).
""")

col_of1, col_of2 = st.columns(2)
with col_of1:
    st.subheader("Root Causes in LLM SFT")
    st.markdown("""
    * **Too Many Epochs**: Training for 5 to 10 epochs over small datasets (< 5,000 samples). SFT should rarely exceed **1 to 3 epochs**.
    * **Learning Rate Too High**: An aggressive learning rate (`> 5e-5` for full fine-tuning) forces weights into sharp, overfitted local minima.
    * **Lack of Regularization**: Zero weight decay (`weight_decay = 0.0`) or missing dropout.
    * **Near-Duplicate Data**: Repeated samples in the training set artificially amplify token likelihoods.
    """)

with col_of2:
    st.subheader("Production Remedies")
    st.markdown("""
    * **Early Stopping with Checkpoints**: Monitor validation loss every 50 steps; restore the best checkpoint when validation loss rises for 3 consecutive evals.
    * **Weight Decay (`0.01` to `0.1`)**: Penalizes large parameter norms in AdamW.
    * **LoRA Rank Reduction**: Lowering LoRA rank `r` from 64 to 8 or 16 restricts adapter capacity, preventing verbatim memorization.
    * **Add General Replay Buffer**: Mix in 10-20% high-quality general dialogue data.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - CATASTROPHIC FORGETTING
# ==============================================================================
st.header("3. Catastrophic Forgetting", divider="rainbow")

st.markdown("""
**Catastrophic Forgetting** occurs when fine-tuning a model on a narrow domain or task causes its weights to drift, 
permanently erasing capabilities acquired during trillion-token pretraining.
""")

st.markdown("""
#### The SQL Fine-Tuning Parable
An enterprise team fine-tunes Llama-3-8B exclusively on 20,000 Text-to-SQL pairs for 3 epochs.
* **Result on SQL**: Text-to-SQL accuracy surges from 62% to 91%.
* **Result on General AI**: MMLU score plummets from 68% to 32%; GSM8K math accuracy collapses from 58% to 11%; the model forgets how to summarize text or hold a basic conversation!
""")

col_cf1, col_cf2 = st.columns(2)
with col_cf1:
    st.markdown("#### Mathematical Mechanism")
    st.markdown("""
    Gradient updates `Delta_W = - lr * Grad_W(L_task)` move parameters along directions optimal for `L_task`.
    These updates overwrite orthogonal weight projections that encoded general linguistic syntax and multi-step reasoning chains.
    """)

with col_cf2:
    st.markdown("#### Production Mitigation: The Replay Buffer")
    st.markdown("""
    * **Data Mixing (Replay Buffer)**: Always blend **15% to 25% general instruction data** (e.g. UltraChat, OpenOrca, LMSYS-Chat-1M) into your domain dataset.
    * **Parameter-Efficient Tuning (LoRA)**: Freezing base weights `W_0` guarantees the base model's world knowledge is mathematically immutable.
    * **Weight Merging & DARE**: Prune and merge fine-tuned weights back into the base model using DARE (Drop And REscale) or Ties-Merging.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - DATA LEAKAGE
# ==============================================================================
st.header("4. Data Leakage", divider="rainbow")

st.markdown("""
**Data Leakage** in LLM post-training creates a catastrophic illusion of competence: 
the model achieves near-perfect scores on internal benchmarks, but fails completely when deployed to production users.
""")

st.markdown("""
### Common Contamination Vectors
1. **Multi-Turn Conversation Splitting**: Putting Turn 1 and Turn 2 in the training set, and Turn 3 in the validation set. The model merely reads its own cached context!
2. **Template Duplication**: Having 5,000 samples generated from the exact same 3 synthetic prompt skeletons. A random split guarantees the validation set only tests memorized templates.
3. **Public Benchmark Contamination**: Ingesting open-source instruction mixtures that inadvertently contain questions from MMLU, HumanEval, or GSM8K.
""")

st.info("""
**Auditing Protocol**:
* Always split data by **Unique Document ID or User ID**, never by row index.
* Run MinHash shingle deduplication with Jaccard threshold `0.7` between Train and Validation sets.
* Enforce 13-gram decontamination against standard public benchmarks.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - POOR INSTRUCTION FORMATTING
# ==============================================================================
st.header("5. Poor Instruction Formatting", divider="rainbow")

st.markdown("""
Tokenization bugs and malformed chat templates are the most common source of silent fine-tuning failure.
""")

st.markdown("""
| Defect | Symptom in Production | Root Cause |
| :--- | :--- | :--- |
| **Missing `<|im_end|>` or `<|eot_id|>`** | Model generates assistant answers, then immediately continues generating fake user queries in an endless loop. | The training dataset omitted end-of-turn special tokens on assistant responses. |
| **Double EOS Tokens** | Generation abruptly stops after 1 or 2 tokens. | Both the dataset text and the tokenizer chat template injected an EOS token. |
| **Prompt Loss Leakage** | Model repeats user questions back or speaks in broken prompts. | Target labels for user prompts were not set to `ignore_index = -100`. |
| **Trailing Whitespace Shift** | Drastic drop in accuracy depending on whether prompt ends with a space. | Tokenizer generates different subword tokens for `"hello"` vs `" hello"`. |
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - HALLUCINATION AFTER TUNING
# ==============================================================================
st.header("6. Hallucination After Tuning", divider="rainbow")

st.markdown("""
A dangerous myth in LLM engineering is that fine-tuning is an effective way to inject new domain knowledge into a model.
In reality, **fine-tuning on new facts without adequate grounding frequently amplifies hallucination!**
""")

col_hal1, col_hal2 = st.columns(2)
with col_hal1:
    st.subheader("Superficial Alignment & Sycophancy")
    st.markdown("""
    * **Superficial Alignment**: SFT teaches the model the **style, tone, and grammar of an expert**, but cannot reliably update its underlying factual database.
    * **Sycophancy**: If training data rewards agreeing with user assumptions, the model will confirm false factual claims made by users (*"Yes, the earth is flat in quantum mechanics..."*).
    * **Factual Overconfidence**: Fine-tuned models often lose their calibration, answering confidently on topics where base models correctly expressed uncertainty.
    """)

with col_hal2:
    st.subheader("The Proper Architectural Division")
    st.markdown("""
    * **Do NOT use SFT for**: Storing changing product catalogs, internal knowledge bases, or recent news.
    * **DO use RAG for**: Factual grounding, citation verification, and real-time knowledge retrieval.
    * **DO use SFT for**: Formatting RAG answers, adhering to strict JSON schemas, adopting corporate brand voice, and learning complex reasoning styles.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - DATASET IMBALANCE
# ==============================================================================
st.header("7. Dataset Imbalance", divider="rainbow")

st.markdown("""
An unbalanced instruction dataset biases the model's generation length, formatting choices, and task distribution.
""")

col_imb1, col_imb2 = st.columns(2)
with col_imb1:
    st.markdown("#### The Three Types of Imbalance")
    st.markdown("""
    1. **Length Imbalance (Brevity Bias)**: If 80% of training responses are 1 sentence long, the model will refuse to produce detailed explanations.
    2. **Task Skew**: If 90% of samples are summarization and 10% are code generation, code generation quality degrades rapidly.
    3. **Tone & Verb Collapse**: If instructions almost exclusively start with *"Explain..."*, the model struggles with *"Extract..."*, *"Refactor..."*, or *"Audit..."*.
    """)

with col_imb2:
    st.markdown("#### Balancing Strategy")
    st.markdown("""
    * **Stratified Resampling**: Cap the maximum number of samples per task category (e.g. max 5,000 per domain).
    * **Instruction Paraphrasing**: Use synthetic LLM pipelines to rewrite prompt verbs across varied modalities.
    * **Length Stratification**: Ensure a balanced histogram across short (< 100 tokens), medium (100 - 500 tokens), and long (500+ tokens) responses.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Fine-Tuning Failure Detector and Diagnostic Auditor in Python", divider="rainbow")

st.markdown("""
The following self-contained class audits fine-tuning telemetry in production, detecting overfitting bifurcation, 
catastrophic forgetting, and length skew.
""")

st.code("""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

class ProductionFailureAuditor:
    \"\"\"
    Diagnostic auditor analyzing LLM fine-tuning health.
    \"\"\"
    def __init__(self, baseline_mmlu: float = 68.5, baseline_gsm8k: float = 54.0):
        self.baseline_mmlu = baseline_mmlu
        self.baseline_gsm8k = baseline_gsm8k

    def calculate_overfitting_score(self, train_loss: float, val_loss: float) -> Tuple[float, str]:
        divergence = val_loss - train_loss
        ratio = val_loss / max(0.001, train_loss)
        if ratio > 2.0 or divergence > 1.2:
            return float(ratio), "Critical Overfitting (Severe Memorization)"
        elif ratio > 1.4 or divergence > 0.5:
            return float(ratio), "Moderate Overfitting (Early Stopping Needed)"
        else:
            return float(ratio), "Healthy Generalization"

    def audit_forgetting(self, current_mmlu: float, current_gsm8k: float) -> Dict[str, Any]:
        delta_mmlu = current_mmlu - self.baseline_mmlu
        delta_gsm8k = current_gsm8k - self.baseline_gsm8k
        is_forgetting = (delta_mmlu < -5.0) or (delta_gsm8k < -8.0)
        return {
            "delta_mmlu": float(delta_mmlu),
            "delta_gsm8k": float(delta_gsm8k),
            "severity": "High Catastrophic Forgetting" if is_forgetting else "Acceptable Drift"
        }
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Fine-Tuning Pathology Simulator & Metric Auditor", divider="rainbow")

auditor = ProductionFailureAuditor()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Training Parameters & Replay Configuration")
    sim_epochs = st.slider("Training Epochs", min_value=1, max_value=8, value=4)
    sim_replay = st.slider("General Replay Buffer Ratio (%)", min_value=0, max_value=40, value=5, format="%d%%") / 100.0
    sim_skew = st.slider("Dataset Task Skew (Narrowness)", min_value=0.1, max_value=1.0, value=0.85)

    sim_df = auditor.simulate_training_pathologies(
        epochs=sim_epochs,
        dataset_skew=sim_skew,
        replay_ratio=sim_replay
    )

    latest_train_loss = sim_df["Train Loss"].iloc[-1]
    latest_val_loss = sim_df["Validation Loss"].iloc[-1]
    latest_mmlu = sim_df["MMLU General Score"].iloc[-1]

    ratio, status = auditor.calculate_overfitting_score(latest_train_loss, latest_val_loss)
    forgetting_audit = auditor.audit_forgetting(latest_mmlu, 45.0)

with wb_col2:
    st.subheader("Diagnostic Health Assessment")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Generalization Gap", f"{latest_val_loss - latest_train_loss:.2f}", f"Ratio: {ratio:.2f}x")
    with m_col2:
        st.metric("General MMLU Retention", f"{latest_mmlu:.1f}%", f"{forgetting_audit['delta_mmlu']:.1f}% vs baseline")

    if "Critical" in status:
        st.error(f"🚨 Status: {status}")
    elif "Moderate" in status:
        st.warning(f"⚠️ Status: {status}")
    else:
        st.success(f"✅ Status: {status}")

    if forgetting_audit["severity"] == "High Catastrophic Forgetting":
        st.error(f"🚨 Severe Catastrophic Forgetting! Replay buffer ({int(sim_replay*100)}%) is insufficient to protect reasoning.")
    else:
        st.success(f"✅ General Knowledge Protected (Drift: {forgetting_audit['delta_mmlu']:.1f}%).")

st.markdown("##### Loss Trajectory & Overfitting Bifurcation:")
st.line_chart(sim_df.set_index("Step")[["Train Loss", "Validation Loss"]])
st.caption("Notice how training loss continues plunging, while validation loss bifurcates and surges upwards when epochs > 2 and replay buffer is low.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: How do you mathematically detect when an LLM is overfitting during SFT?"):
        st.write("""
        Overfitting is detected when the generalization gap `Validation_Loss - Training_Loss` expands persistently across evaluation checkpoints, or when validation perplexity begins rising while training loss drops.
        Additionally, generation entropy drops sharply as the model converges onto rigid verbatim training outputs.
        """)
    with st.expander("Q2: Why does fine-tuning on a narrow task degrade general reasoning (catastrophic forgetting)?"):
        st.write("""
        Gradient updates minimize loss specifically on the narrow task, modifying shared feed-forward and attention weights that previously encoded broader multi-step logic.
        Without gradient signals reinforcing general skills, those parameter subspaces drift into configurations incompatible with general reasoning.
        """)
    with st.expander("Q3: What is the primary cause of infinite repetition loops in fine-tuned models?"):
        st.write("""
        The most common cause is missing or unlearned End-Of-Turn special tokens (`<|im_end|>` or `<|eot_id|>`).
        If the model never receives loss penalties for failing to emit EOS tokens at the end of assistant answers, it assigns low probability to stopping, continuing to generate until hitting `max_tokens`.
        """)

with q2:
    with st.expander("Q4: Why is fine-tuning an ineffective method for injecting new domain facts?"):
        st.write("""
        Fine-tuning adjusts behavioral style, tone, and output formatting. 
        Because fine-tuning datasets are small (thousands of pairs) compared to pretraining (trillions of tokens), the model memorizes superficial word associations rather than robust factual world models.
        When asked nuanced factual questions, it hallucinates plausible-sounding but completely fabricated statements. Factual retrieval should always be handled by RAG.
        """)
    with st.expander("Q5: What is Sycophancy in fine-tuned models and how do you prevent it?"):
        st.write("""
        Sycophancy is the tendency of an aligned model to agree with false premises or subjective biases in user queries 
        (e.g., agreeing with a user who claims that `1+1=3`).
        It occurs when training datasets contain answers that flatter or validate user prompts. 
        It is mitigated by including explicit disagreement and factual correction pairs in the instruction mix.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Training for Too Many Epochs**
    * **Failure**: Treating SFT like classical image classification and training for 10+ epochs.
    * **Impact**: Total representation collapse within 3 epochs; the model loses flexibility and memorizes training strings verbatim.
    * **Remedy**: Cap SFT at 1 to 3 epochs maximum, using validation loss checkpoints for early stopping.
    """)
    st.error("""
    **🚨 Production Trap 2: Neglecting the General Replay Buffer**
    * **Failure**: Training exclusively on 100% domain data without any general instruction samples.
    * **Impact**: The model achieves 95% domain accuracy but completely forgets how to summarize, code, or answer general questions.
    * **Remedy**: Always maintain a 15% to 25% general conversation replay buffer in every fine-tuning run.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Brevity Bias from Short Demonstrations**
    * **Failure**: Using training pairs where assistant responses average only 30 tokens.
    * **Impact**: When asked complex questions, the model outputs terse, unhelpful fragments.
    * **Remedy**: Audit and balance the length histogram of target responses across short, medium, and long categories.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Evaluating on Contaminated Benchmarks**
    * **Failure**: Reporting state-of-the-art results because training data contained paraphrased test questions.
    * **Impact**: Model fails disastrously in customer-facing production despite stellar internal metrics.
    * **Remedy**: Enforce strict 13-gram decontamination filtering before starting any training run.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Recommended SFT Epoch Range"):
        st.write("What is the standard number of epochs for fine-tuning an LLM on instruction data?")
        st.info("1 to 3 epochs maximum.")
    with fc_col1:
        with st.expander("⚡ Flashcard 2: Replay Buffer Ratio"):
            st.write("What percentage of general data should be mixed into domain fine-tuning to stop forgetting?")
            st.info("15% to 25% general conversational replay data.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Knowledge Injection Protocol"):
        st.write("Should you use SFT or RAG to give an LLM access to private product documentation?")
        st.info("RAG for factual knowledge; SFT for formatting, style, and tone.")
    with fc_col2:
        with st.expander("⚡ Flashcard 4: Primary Cause of Infinite Loops"):
            st.write("What missing token causes models to generate endless loops?")
            st.info("Missing or unmasked End-Of-Turn / EOS special tokens.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of LLM Fine-Tuning Problems before advancing to Chapter 14 (PEFT):
""")

chk1 = st.checkbox("I can recognize the symptoms of SFT overfitting: verbatim memorization and loss divergence.", value=True)
chk2 = st.checkbox("I understand why Catastrophic Forgetting occurs and how a 20% replay buffer prevents it.", value=True)
chk3 = st.checkbox("I know why fine-tuning is the wrong tool for factual knowledge injection and how it leads to hallucination.", value=True)
chk4 = st.checkbox("I understand the cause of repetition loops and missing end-of-turn delimiter tokens.", value=True)
chk5 = st.checkbox("I know how to identify and prevent brevity bias through response length stratification.", value=True)
chk6 = st.checkbox("I understand 13-gram decontamination to prevent train/test benchmark leakage.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! Chapter 13: LLM Fine-Tuning is fully completed. You are ready for Chapter 14: PEFT.")
