import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="3.1 Zero-Shot Basics | GenAI Roadmap",
    page_icon="🎯",
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
    .badge-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🎯 3.1 Zero-Shot Basics — Foundations, Mechanics & Boundaries")
st.markdown(
    """
Welcome to **Lesson 3.1: Zero-Shot Basics**.  
In classical machine learning, training a model to classify text requires hundreds or thousands of labeled data points. 
In modern Generative AI, modern Large Language Models possess the remarkable ability to perform complex downstream tasks with **zero task-specific training examples**. This paradigm is known as **Zero-Shot Prompting**.

Below, we examine the underlying mechanics of zero-shot inference, explore when to deploy it versus when to escalate to few-shot or fine-tuning, and inspect production failure modes.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is Zero-Shot Prompting & How It Works
# ==============================================================================
st.header("1. What is Zero-Shot Prompting & Internal Mechanics", divider="rainbow")

col_z1, col_z2 = st.columns([1, 1])

with col_z1:
    st.markdown(
        """
### Definition
**Zero-Shot Prompting** is the technique of presenting an LLM with a task description and an input payload **without providing any demonstrative input-output examples** in the prompt context.

### The Mechanics of Zero-Shot Generalization
How can an LLM perform a task it has never seen demonstrated in the prompt?
1. **Pre-Trained Latent Representations**: During self-supervised pre-training on petabytes of diverse text, the model builds rich semantic representations connecting concepts, vocabularies, and syntactic patterns.
2. **Instruction Fine-Tuning (IT / SFT)**: Models are trained on thousands of generic instruction-response pairs (FLAN, InstructGPT, UltraChat), teaching the attention layers to map natural language command verbs (*summarize*, *classify*, *translate*) directly to operational behavioral modes.
3. **RLHF Alignment Priors**: Reinforcement Learning from Human Feedback aligns the model's token distribution to prioritize helpful, direct solutions when an instruction is detected.
"""
    )

with col_z2:
    st.markdown("### Zero-Shot Attention Flow (ASCII)")
    st.code(
        """┌─────────────────────────────────────────────────────────┐
│ PROMPT CONTEXT (0 In-Context Examples)                  │
│                                                         │
│  [INSTRUCTION] "Classify review sentiment: [POS, NEG]"  │
│  [INPUT DATA]  "Fast delivery, but battery life is poor"│
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ AUTOREGRESSIVE INFERENCE                                │
│                                                         │
│ 1. Multi-Head Attention activates task command priors   │
│ 2. Queries parametric memory for semantic representations│
│ 3. Directly samples output token:                       │
│    P("NEGATIVE" | Context) = 0.88                       │
└─────────────────────────────────────────────────────────┘""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 2: When to Use Zero-Shot vs When to Escalate
# ==============================================================================
st.header("2. When to Use Zero-Shot vs When to Escalate", divider="rainbow")
st.markdown("Zero-shot prompting is the default starting point for any AI task. However, production workflows require clear escalation criteria:")

escalation_data = {
    "Evaluation Criterion": [
        "Task Standardness",
        "Token Budget & Cost",
        "Prompt Maintenance",
        "Complex Output Schemas",
        "Rare Domain Terminology",
        "Novel Reasoning Tasks"
    ],
    "Zero-Shot Prompting (Default)": [
        "Standard NLP tasks (sentiment, summarization, entity lookup)",
        "Lowest token consumption; minimum input latency",
        "Zero example curation needed; fast prototyping",
        "Prone to syntax drift without strict schema enforcement",
        "Struggles with proprietary acronyms or custom jargon",
        "Higher error rate on multi-step arithmetic/logic"
    ],
    "When to Escalate to Few-Shot / RAG": [
        "Highly specialized taxonomy or domain-specific classification",
        "Higher token cost justified by strict reliability needs",
        "Requires maintaining a curated exemplar library",
        "Few-shot exemplars anchor JSON/XML syntax reliably",
        "Few-shot injects domain jargon; RAG retrieves definitions",
        "Chain-of-Thought (CoT) few-shot required for step-by-step logic"
    ]
}
st.dataframe(pd.DataFrame(escalation_data), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Advantages vs Limitations
# ==============================================================================
st.header("3. Technical Tradeoffs: Advantages vs Limitations", divider="rainbow")

col_adv, col_lim = st.columns(2)

with col_adv:
    with st.container(border=True):
        st.subheader("✅ Advantages of Zero-Shot")
        st.markdown(
            """
- **Minimal Token Consumption**: Zero-shot prompts consume fewer prompt tokens, directly lowering cost per query and Time-To-First-Token (TTFT) latency.
- **Zero Cold-Start Overhead**: Engineers can deploy an initial baseline immediately without building, curating, or validating exemplar datasets.
- **Cache Friendly**: Static zero-shot system prompts maximize prefix caching efficiency in providers like Anthropic and OpenAI.
- **Unbiased Flexibility**: Avoids accidentally priming the model with biased or unrepresentative few-shot example patterns.
"""
        )

with col_lim:
    with st.container(border=True):
        st.subheader("⚠️ Limitations of Zero-Shot")
        st.markdown(
            """
- **Format Volatility**: Without explicit examples of the expected serialization format, outputs may fluctuate between markdown, lists, and conversational banter.
- **Majority-Class Bias**: In subjective classification tasks, zero-shot models default to their pre-training frequency priors (often skewing positive).
- **Domain Blindspots**: Models cannot infer proprietary internal business logic (e.g. what constitutes a 'Tier-1 Escalation' at your specific enterprise).
- **Vulnerability to Nuance**: Complex linguistic subtleties like dry sarcasm or double negatives frequently misclassify without exemplars.
"""
        )

st.divider()

# ==============================================================================
# Section 4: Production Code Patterns for Zero-Shot
# ==============================================================================
st.header("4. Production Python Implementation Patterns", divider="rainbow")
st.markdown("Below are three production-grade zero-shot patterns for classification, extraction, and transformation:")

st.code(
    '''import openai
from pydantic import BaseModel, Field
from typing import Literal

client = openai.OpenAI()

# -----------------------------------------------------------------------------
# Pattern 1: Zero-Shot Classification with Fallback Token
# -----------------------------------------------------------------------------
zero_shot_classification_prompt = """
INSTRUCTION:
Classify the intent of the incoming user message into exactly ONE category:
[BILLING_INQUIRY, BUG_REPORT, FEATURE_REQUEST, GENERAL_FEEDBACK].
If the intent cannot be determined, output: "UNKNOWN_INTENT".

CONSTRAINTS:
Output ONLY the category name. No markdown backticks, no punctuation.

USER MESSAGE:
"Can I export all active user tables as a Parquet file rather than CSV?"

INTENT:"""

# -----------------------------------------------------------------------------
# Pattern 2: Zero-Shot Structured Extraction with Pydantic
# -----------------------------------------------------------------------------
class ContactExtraction(BaseModel):
    full_name: str
    company_name: str
    email_address: str | None = None
    urgency_level: Literal["LOW", "MEDIUM", "HIGH"]

zero_shot_extraction_prompt = """
Extract contact details from the raw unformatted email text into the specified schema.
If an email is missing, set to null.

EMAIL TEXT:
"Hi team, this is Elena Rostova from QuantumDynamics Inc. We need urgent assistance with our license renewal today."
"""

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": zero_shot_extraction_prompt}],
    response_format=ContactExtraction
)

extracted_data = completion.choices[0].message.parsed
print(extracted_data.model_dump_json(indent=2))''',
    language="python"
)

st.divider()

# ==============================================================================
# Section 5: Interactive Zero-Shot Parameter & Temperature Lab
# ==============================================================================
st.header("5. Interactive Lab: Sampling Parameters in Zero-Shot Tasks", divider="rainbow")
st.markdown(
    """
In zero-shot tasks, model hyperparameters directly dictate deterministic consistency. 
Adjust the sampling controls below to observe how **Temperature** and **Top-P** impact zero-shot token probability spreads:
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    temp_val = st.slider("Model Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1, help="0.0 = greedy argmax; higher = stochastic exploration")
with col_ctrl2:
    topp_val = st.slider("Top-P (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=1.0, step=0.05, help="Restricts sampling to smallest set of tokens whose cumulative probability exceeds P")

# Visual probability simulation
st.markdown("#### Simulated Token Logit Distribution for Zero-Shot Classification:")
classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
base_logits = np.array([2.8, 1.2, 0.4])

if temp_val == 0.0:
    probs = np.zeros_like(base_logits)
    probs[np.argmax(base_logits)] = 1.0
else:
    scaled = base_logits / temp_val
    exp_scaled = np.exp(scaled - np.max(scaled))
    probs = exp_scaled / np.sum(exp_scaled)

chart_data = pd.DataFrame({"Class": classes, "Probability": probs})
st.bar_chart(chart_data.set_index("Class"))

if temp_val == 0.0:
    st.success("🎯 **Deterministic Mode (Temp 0.0)**: The argmax token (POSITIVE) is selected with 100% certainty. Ideal for zero-shot classification, entity extraction, and code generation.")
elif temp_val <= 0.4:
    st.info("⚖️ **Calibrated Mode (Temp 0.1 - 0.4)**: High confidence with minor variation allowed on edge cases. Ideal for zero-shot summarization and FAQ answering.")
else:
    st.warning("⚠️ **High Dispersion (Temp > 0.5)**: Token probability is flattened across candidates. High risk of format deviations and non-deterministic classifications in zero-shot prompts.")

st.divider()

# ==============================================================================
# Section 6: Architectural Decision Tree (Graphviz)
# ==============================================================================
st.header("6. Architectural Decision Tree: When to Use Zero-Shot", divider="rainbow")

col_dt1, col_dt2 = st.columns([1, 1])

with col_dt1:
    st.markdown("#### Escalation Flowchart")
    dot_tree = """
    digraph ZeroShotDecision {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Start [label="New Natural Language Task", fillcolor="#334155", fontcolor="#ffffff"];
        TestZero [label="Test Zero-Shot Prompt\\n(Temp = 0.0)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Eval [label="Accuracy >= 95% AND\\nFormat Consistent?", shape=diamond, fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        Deploy [label="Deploy to Production\\n(Lowest Token Cost)", fillcolor="#047857", fontcolor="#ffffff"];
        FewShot [label="Escalate to Few-Shot\\n(Add 3-5 Exemplars)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        RAG [label="Escalate to RAG\\n(External Retrieval)", fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];

        Start -> TestZero;
        TestZero -> Eval;
        Eval -> Deploy [label="Yes"];
        Eval -> FewShot [label="Format/Edge Failure"];
        Eval -> RAG [label="Domain Data Missing"];
    }
    """
    st.graphviz_chart(dot_tree, use_container_width=False)

with col_dt2:
    st.markdown("#### The Three Golden Rules of Zero-Shot Deployment")
    st.markdown(
        """
1. **Rule of Default**: Always start with Zero-Shot first. Never prematurely add few-shot examples without measuring a baseline failure rate.
2. **Rule of Framing**: Compensate for missing examples with crystal-clear constraints and delimiter tags (`<input>...</input>`).
3. **Rule of Fallback**: Always include a deterministic fallback condition (*"If not found, return NULL"*) to prevent generative hallucinations.
"""
    )

st.divider()

# ==============================================================================
# Section 7: Technical Interview Preparation
# ==============================================================================
st.header("7. Technical Interview Preparation", divider="rainbow")

z_tab1, z_tab2, z_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with z_tab1:
    with st.expander("Q1: What training process gives modern LLMs their strong zero-shot capabilities?", expanded=True):
        st.markdown(
            """
Zero-shot capabilities stem primarily from **Instruction Fine-Tuning (IFT/SFT)** on top of massive pre-training. 
While pre-trained base models (e.g. base LLaMA or GPT-3) are raw token predictors that often simply continue text rather than answer instructions, instruction tuning trains models on hundreds of thousands of structured `(Instruction, Input, Response)` tuples across thousands of distinct tasks. 
This aligns the model's cross-attention mechanisms to interpret imperative instructions as functional execution triggers rather than narrative text continuations.
"""
        )

    with st.expander("Q2: In what scenarios does Zero-Shot prompting completely fail?", expanded=False):
        st.markdown(
            """
1. **Proprietary Classification Taxonomies**: When class definitions differ from common public usage (e.g., custom enterprise ticket severity matrices).
2. **Strict Bespoke Formatting**: When a downstream system requires a non-standard JSON schema or obscure markup format that lacks public web training density.
3. **Complex Multi-Step Logic**: Tasks requiring intermediate state transitions (e.g., symbolic reasoning, nested arithmetic) where zero-shot attempts jump straight to an erroneous conclusion without intermediate scratchpad tokens.
"""
        )

    with st.expander("Q3: How does prompt caching interact with zero-shot vs few-shot prompts?", expanded=False):
        st.markdown(
            """
Zero-shot prompts feature shorter, highly standardized system instruction blocks. 
Because the prefix contains no dynamic or changing exemplars, the provider's Key-Value (KV) cache for the prompt prefix is hit with near 100% consistency across concurrent API calls, slashing latency by 50-80% and token costs by up to 50% compared to heavy few-shot prompts with varying examples.
"""
        )

with z_tab2:
    with st.expander("Trap 1: 'Is Zero-Shot prompting always cheaper than Few-Shot prompting?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, because zero-shot prompts have fewer input tokens."*
- **Winning Response**: *"Per single request, yes. But if a zero-shot prompt fails to adhere to schemas 15% of the time, your system will require retry loops, secondary validation LLM calls, and error-handling fallbacks. In high-throughput production systems, a robust 2-shot prompt that succeeds on the first attempt 99.8% of the time often incurs lower total compute and latency costs than a flaky zero-shot prompt requiring frequent retries."*
"""
        )

    with z_tab3:
        z_c1, z_c2, z_c3 = st.columns(3)
        with z_c1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Zero-Shot Prompting**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Executing a task via natural language instruction without providing in-context demonstrative input-output examples.")

        with z_c2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Instruction Tuning Role**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** SFT transforms a raw autoregressive text completer into an instruction-following operational assistant.")

        with z_c3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Escalation Trigger**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Escalate from zero-shot to few-shot when format volatility or edge-case classification accuracy falls below SLA.")

st.divider()

# ==============================================================================
# Section 8: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("8. Lesson 3.1 Summary & Self-Assessment", divider="rainbow")

col_zs1, col_zs2 = st.columns(2)
with col_zs1:
    st.markdown(
        """
### Key Takeaways:
1. **Default Paradigm**: Always test Zero-Shot as the architectural baseline before adding complexity.
2. **Powered by Instruction Tuning**: Pre-training provides knowledge; SFT and RLHF provide instruction-following reflexes.
3. **Cost & Latency Winner**: Zero-shot maximizes KV cache hits and minimizes input token consumption.
4. **Escalate with Data**: Move to Few-Shot or RAG only when format stability or domain specificity demands it.
"""
    )

with col_zs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain how instruction tuning enables zero-shot capabilities in LLMs.", value=True, key="chk_3_1_1")
    st.checkbox("I know the architectural criteria for when to escalate from zero-shot to few-shot or RAG.", value=True, key="chk_3_1_2")
    st.checkbox("I understand the effect of Temperature on zero-shot deterministic classification.", value=True, key="chk_3_1_3")
    st.checkbox("I can implement a production zero-shot extraction pipeline with fallback tokens in Python.", value=True, key="chk_3_1_4")
