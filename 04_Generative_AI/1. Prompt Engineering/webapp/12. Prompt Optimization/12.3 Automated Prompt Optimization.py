import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="12.3 Automated Prompt Optimization | GenAI Roadmap",
    page_icon="🤖",
    layout="wide",
)

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
    .dspy-chip {
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
st.title("🤖 12.3 Automated Prompt Optimization — DSPy, Search & Genetic Mutation")
st.markdown(
    """
Welcome to **Lesson 12.3: Automated Prompt Optimization (APO)**.  
Manual "prompt engineering"—spending hours tweaking individual words by hand—is rapidly being replaced by **algorithmic prompt compilation**.

Frameworks like **DSPy**, **MIPRO**, and **Evolutionary Prompt Mutators** treat prompts as trainable software parameters. Given a small training dataset and an evaluation metric, APO algorithms automatically search, mutate, test, and synthesize superior prompts that consistently outperform human-authored baselines.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Paradigm Shift: Manual Prompting vs. Prompt Compilation
# ==============================================================================
st.header("1. The Paradigm Shift: Prompts as Trainable Weights", divider="rainbow")

col_ps1, col_ps2 = st.columns(2)

with col_ps1:
    with st.container(border=True):
        st.subheader("❌ Traditional Manual Prompting (Brittle)")
        st.markdown(
            """
- **Manual Guesswork**: Engineers guess which words or role personas might improve model accuracy.
- **Model Lock-In**: Prompts handcrafted for GPT-4 often fail when migrated to Claude 3.5 or LLaMA 3.
- **No Global Gradient**: Hard to explore the combinatorial space of instructions, examples, and phrasing.
- **Fragile Maintenance**: Every business requirement change requires manual rewriting and retesting.
"""
        )

with col_ps2:
    with st.container(border=True):
        st.subheader("✅ Automated Prompt Optimization (Algorithmic)")
        st.markdown(
            """
- **Programmatic Signatures**: You define inputs, outputs, and validation metrics (e.g. F1, exact match, judge score).
- **Automated Exploration**: Algorithms generate candidates via mutation, paraphrasing, and few-shot mining.
- **Multi-Model Portability**: The same pipeline automatically re-compiles optimal prompts for any underlying LLM.
- **Reproducible Performance**: Prompts are optimized systematically against a quantitative objective function.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Core Optimization Paradigms
# ==============================================================================
st.header("2. Core Algorithmic Optimization Paradigms", divider="rainbow")
st.markdown("APO frameworks utilize distinct search and synthesis strategies across the discrete token space:")

paradigms_df = pd.DataFrame({
    "Framework / Paradigm": [
        "1. DSPy (Stanford)",
        "2. MIPRO (Multi-Prompt Instruction Optimization)",
        "3. APE (Automatic Prompt Engineer)",
        "4. Evolutionary / Genetic Mutators",
        "5. OPRO (Optimization by PROmpting)"
    ],
    "Core Search Mechanism": [
        "Teleprompters compile signatures into optimal few-shot examples and instruction prefixes",
        "Joint Bayesian optimization over instruction candidates and few-shot demonstration sets",
        "Uses an LLM to generate candidate prompts from input-output pairs, scoring them via beam search",
        "Generates prompt variations via crossover, mutation, and tournament survival selection",
        "Uses an LLM as the meta-optimizer, iteratively describing previous trajectory scores"
    ],
    "Key Benefit": [
        "Modular, code-first architecture; separates logic from prompt text entirely",
        "State-of-the-art accuracy across complex multi-hop reasoning tasks",
        "Completely automated zero-shot instruction synthesis without human priors",
        "Discovers unintuitive phrasing that human engineers would never invent",
        "Leverages meta-reasoning capabilities of frontier models for convergence"
    ]
})
st.dataframe(paradigms_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The DSPy / APO Compilation Pipeline", divider="rainbow")
st.markdown("How automated prompt compilers optimize pipelines against an evaluation metric:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9, color="white", fontcolor="white"]

        dataset [label="Training Dataset\n(50-100 Samples)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        metric_fn [label="Evaluation Metric\n(ExactMatch / LLM Judge)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        optimizer [label="APO Teleprompter\n(e.g., BootstrapFewShot)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        mutation_loop [label="Candidate Search\n& Mutation Trials", fillcolor="#EAB308", fontcolor="#000000"]
        compiled [label="Compiled Pipeline\n(Optimal Prompts & Shots)", fillcolor="#10B981", fontcolor="#FFFFFF"]

        dataset -> optimizer
        metric_fn -> optimizer
        optimizer -> mutation_loop -> optimizer
        optimizer -> compiled [label=" Best Score", color="#10B981"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Automated Prompt Optimizer Simulator
# ==============================================================================
st.header("4. Interactive Genetic Prompt Mutation Simulator", divider="rainbow")
st.markdown("Simulate how evolutionary algorithms mutate prompt instructions across multiple generations to improve accuracy:")

sim_task = st.selectbox(
    "Select Optimization Task",
    ["Legal Contract Extraction", "Code Security Vulnerability Flagging", "Clinical Sentiment Analysis"]
)

iterations_to_run = st.slider("Optimization Generations (Epochs)", min_value=1, max_value=4, value=3)

col_s1, col_s2 = st.columns([1, 1])

with col_s1:
    st.subheader("🧬 Genetic Mutation Trajectory")
    
    if sim_task == "Legal Contract Extraction":
        gen_data = [
            {"Gen": 0, "Prompt": "Extract contract liability details.", "Score": 62.4, "Action": "Initial Seed Baseline"},
            {"Gen": 1, "Prompt": "Extract liability clauses from the text. Return as JSON with caps and indemnities.", "Score": 74.8, "Action": "Mutated: Added schema cues"},
            {"Gen": 2, "Prompt": "Read contract. Extract liability caps, indemnification, and governing state. If uncapped, assign UNLIMITED.", "Score": 86.1, "Action": "Mutated: Edge-case boundary added"},
            {"Gen": 3, "Prompt": "<instructions>Extract liability terms.</instructions><output_format>{\"liability_cap_usd\": float, \"has_uncapped_clause\": bool}</output_format>", "Score": 94.3, "Action": "Compiled: XML + Typed JSON Schema"}
        ]
    elif sim_task == "Code Security Vulnerability Flagging":
        gen_data = [
            {"Gen": 0, "Prompt": "Find bugs in this code.", "Score": 55.0, "Action": "Initial Seed Baseline"},
            {"Gen": 1, "Prompt": "Find security vulnerabilities in code. Classify severity.", "Score": 71.2, "Action": "Mutated: Added severity taxonomy"},
            {"Gen": 2, "Prompt": "Audit code against OWASP Top 10. Flag SQLi, SSRF, XSS with exact line numbers.", "Score": 83.5, "Action": "Mutated: Added CWE criteria"},
            {"Gen": 3, "Prompt": "<system>Security auditor</system><task>Flag high-confidence exploits with CWE IDs.</task><constraints>Ignore stylistic lints.</constraints>", "Score": 92.8, "Action": "Compiled: Modular XML + Neg constraints"}
        ]
    else:
        gen_data = [
            {"Gen": 0, "Prompt": "Determine patient sentiment.", "Score": 68.0, "Action": "Initial Seed Baseline"},
            {"Gen": 1, "Prompt": "Analyze patient distress level. Rate 1-5.", "Score": 79.5, "Action": "Mutated: Categorical scale added"},
            {"Gen": 2, "Prompt": "Classify clinical emotional state into: ACUTE_PANIC, MILD_ANXIETY, CALM, UNRESPONSIVE.", "Score": 88.0, "Action": "Mutated: Strict medical enums"},
            {"Gen": 3, "Prompt": "<context>Clinical triage</context><instructions>Map patient voice to distress tier.</instructions><output>{\"tier\": enum, \"requires_stat_psych\": bool}</output>", "Score": 96.2, "Action": "Compiled: Production Clinical Schema"}
        ]

    active_gens = gen_data[:iterations_to_run + 1]
    trajectory_df = pd.DataFrame(active_gens)
    st.dataframe(trajectory_df[["Gen", "Score", "Action"]], use_container_width=True, hide_index=True)

with col_s2:
    st.subheader("📈 Performance Convergence")
    st.line_chart(trajectory_df.set_index("Gen")["Score"])
    
    best_candidate = active_gens[-1]
    st.success(f"🏆 Top Candidate (Gen {best_candidate['Gen']}): Score {best_candidate['Score']}%")
    st.code(best_candidate["Prompt"], language="text")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: DSPy Pipeline", divider="rainbow")
st.markdown("How to build and compile a self-optimizing prompt module using DSPy:")

st.code(
    '''import dspy

# 1. Define the Modular Signature (Declarative Task Specification)
class ContractAuditSignature(dspy.Signature):
    """Extract liability caps and identify uncapped indemnity clauses from contract text."""
    contract_text = dspy.InputField(desc="Raw legal agreement excerpt")
    liability_cap_usd = dspy.OutputField(desc="Numeric maximum liability cap or 'UNLIMITED'")
    has_indemnity_risk = dspy.OutputField(desc="Boolean indicating severe indemnification risk")

# 2. Define the Pipeline Module
class ContractAuditor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Chain of Thought reasoning wrapper
        self.prog = dspy.ChainOfThought(ContractAuditSignature)
        
    def forward(self, contract_text):
        return self.prog(contract_text=contract_text)

# 3. Define the Quantitative Evaluation Metric
def audit_metric(gold_example, prediction, trace=None):
    cap_match = gold_example.liability_cap_usd.strip() == prediction.liability_cap_usd.strip()
    risk_match = gold_example.has_indemnity_risk == prediction.has_indemnity_risk
    return cap_match and risk_match

# 4. Teleprompter Optimization (Automatic Compilation)
# In production, use BootstrapFewShot or MIPRO
# optimizer = dspy.teleprompt.BootstrapFewShot(metric=audit_metric, max_bootstrapped_demos=4)
# compiled_auditor = optimizer.compile(student=ContractAuditor(), trainset=train_examples)

print("DSPy Pipeline Definition Ready. Compilation systematically mines optimal few-shot exemplars.")
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: What is the core design philosophy of Stanford's DSPy framework?", expanded=True):
        st.markdown(
            """
DSPy separates **logic from prompt strings**. In traditional LLM development, changing a prompt string is tightly coupled with code. In DSPy:
1. You declare **Signatures** (`input -> output`) representing what needs to be done.
2. You compose signatures into **Modules** (like PyTorch neural network layers).
3. You run a **Teleprompter/Optimizer** that automatically compiles the prompt—generating optimal instructions, demonstrations, and reasoning steps against a defined validation dataset and metric function. When switching LLMs, you simply recompile.
"""
        )
    with st.expander("Q2: How does MIPRO (Multi-Prompt Instruction Optimization) work?"):
        st.markdown(
            """
MIPRO jointly optimizes both **instructions** and **few-shot demonstrations**:
1. It uses an LLM to generate diverse candidate instructions informed by dataset examples and reasoning traces.
2. It samples candidate demonstrations from successful pipeline executions.
3. It uses **Bayesian Optimization** (via surrogate models) to evaluate combinatorial pairings of instructions and examples, efficiently finding the global optimal prompt without brute-forcing every permutation.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can Automated Prompt Optimization replace high-quality datasets?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, with APO you don't need training data because the optimizer writes the prompt automatically."*
- **Winning Response**: *"Completely false. Automated Prompt Optimization is strictly **evaluation-driven**. It relies entirely on a curated ground-truth dataset (30-100 high-quality labeled examples) and a rigorous metric function. If your evaluation dataset has noisy labels or unrepresentative edge cases, the optimizer will overfit to faulty data, generating prompts that fail in production."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Automated Prompt Optimization (APO)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The practice of using algorithms, LLM evaluators, and search methods to automatically synthesize and tune prompts against a quantitative metric.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**DSPy Teleprompter**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** An optimizer in DSPy that takes a pipeline, training data, and a metric to compile the optimal prompt instructions and few-shot examples.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt Mutation**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Programmatically modifying candidate prompts through paraphrasing, re-ordering, and constraint injection to search the prompt space.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 12.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Prompts as Weights**: Treat prompts as optimizable software parameters rather than static prose.
2. **Evaluation-Driven**: Automated optimization requires a clean benchmark dataset and objective metric.
3. **Framework Adoption**: Tools like DSPy enable model-agnostic, portable prompt compilation.
4. **Superior Convergence**: Search algorithms systematically outperform human trial-and-error.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the conceptual difference between manual prompting and DSPy compilation.", value=True, key="chk_12_3_1")
    st.checkbox("I know how teleprompters (e.g. BootstrapFewShot, MIPRO) optimize pipelines.", value=True, key="chk_12_3_2")
    st.checkbox("I understand why high-quality evaluation datasets are mandatory for APO.", value=True, key="chk_12_3_3")
    st.checkbox("I can implement a basic DSPy signature and module in Python.", value=True, key="chk_12_3_4")
