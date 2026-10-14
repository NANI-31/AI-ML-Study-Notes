import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.2 Prompting Techniques - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionPromptTechniquesEngine:
    """
    Engine implementing core prompting techniques:
    - Zero-Shot: Direct instruction execution
    - One-Shot: Single structural anchor exemplar
    - Few-Shot: In-Context Learning with balanced exemplar distribution
    - Role Prompting: Explicit persona and expertise framing
    - Contextual Prompting: Strict reference document grounding
    """
    def __init__(self):
        self.default_exemplars = [
            {"input": "The battery life is phenomenal but the screen flickers.", "output": '{"sentiment": "Mixed", "aspects": ["battery", "display"]}'},
            {"input": "Worst customer service I have ever experienced.", "output": '{"sentiment": "Negative", "aspects": ["support"]}'},
            {"input": "Lightning fast shipping, item was pristine!", "output": '{"sentiment": "Positive", "aspects": ["delivery", "product"]}'}
        ]

    def build_zero_shot(self, task: str, user_input: str) -> str:
        """Constructs a direct zero-shot prompt."""
        return f"""<instruction>
{task.strip()}
</instruction>

<input_data>
{user_input.strip()}
</input_data>"""

    def build_few_shot(self, task: str, exemplars: List[Dict[str, str]], user_input: str) -> str:
        """Constructs an In-Context Learning prompt with balanced exemplars."""
        ex_str = ""
        for idx, ex in enumerate(exemplars, 1):
            ex_str += f"""<example_{idx}>
<input>
{ex['input']}
</input>
<output>
{ex['output']}
</output>
</example_{idx}>\n\n"""

        return f"""<instruction>
{task.strip()}
Follow the exact format shown in the demonstrations below.
</instruction>

<demonstrations>
{ex_str.strip()}
</demonstrations>

<input_data>
{user_input.strip()}
</input_data>"""

    def build_role_prompt(self, role: str, task: str, user_input: str) -> str:
        """Constructs a persona-conditioned prompt."""
        return f"""<role>
{role.strip()}
</role>

<instruction>
{task.strip()}
</instruction>

<input_data>
{user_input.strip()}
</input_data>"""

    def build_contextual_prompt(self, context: str, task: str, user_input: str) -> str:
        """Constructs a strictly grounded contextual prompt."""
        return f"""<context>
{context.strip()}
</context>

<instruction>
{task.strip()}
Answer strictly using facts established in <context>. If the answer cannot be deduced directly from <context>, state: "I do not possess sufficient verified context."
</instruction>

<query>
{user_input.strip()}
</query>"""

    def estimate_overhead(self, prompt: str) -> Dict[str, Any]:
        """Calculates token counts and costs across prompting strategies."""
        token_count = max(1, int(len(prompt) / 3.8))
        return {
            "token_count": token_count,
            "cost_gpt4o": (token_count / 1000) * 0.0025,
            "cost_llama3": (token_count / 1000) * 0.0002
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("15.2 Prompting Techniques")
st.caption("Chapter 15: Prompt Engineering | Domain: 2. Large Language Models")

st.markdown("""
How do we guide an LLM to reliably solve difficult tasks without updating a single weight parameter?
The secret lies in **In-Context Learning (ICL)**: providing prompt demonstrations that steer the model's autoregressive distribution.
From basic **Zero-Shot** instructions to structured **Few-Shot Exemplars**, **Role Persona Steering**, and **Strict Contextual Grounding**, 
choosing the right prompting strategy dictates whether an application succeeds with 99% reliability or fails on edge cases.

In this lesson, you will master:
1. The prompting spectrum: from zero demonstrations to structured in-context exemplars.
2. Zero-Shot Prompting: baseline task execution and limits on nuanced format adherence.
3. One-Shot Prompting: establishing output syntax and structure with minimal token overhead.
4. Few-Shot In-Context Learning: exemplar selection, label distribution balance, and demonstration ordering bias.
5. Role Prompting: calibrating vocabulary, tone, and technical precision through persona framing.
6. Instruction Prompting: imperative command structuring and negative boundary enforcement.
7. Contextual Prompting: grounding generations in external reference data to eliminate hallucination.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Prompting Technique Spectrum & In-Context Conditioning Gradient", divider="rainbow")

st.markdown("""
Prompting techniques form a continuous spectrum balancing token consumption against output determinism.
""")

technique_dot = """
digraph TechniqueSpectrum {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_zero {
        label = "Zero-Shot (Lowest Token Cost)";
        color = "#334155";
        fontcolor = "#94A3B8";
        ZeroShot [label="Zero-Shot\\nRaw Task Directive\\n(Relies on Pretraining Priors)", fillcolor="#0F172A"];
    }

    subgraph cluster_role {
        label = "Steering & Framing";
        color = "#334155";
        fontcolor = "#94A3B8";
        RolePrompt [label="Role Prompting\\nPersona Conditioning\\n(Narrows Latent Circuits)", fillcolor="#0369A1"];
        ContextPrompt [label="Contextual Prompting\\nReference Document Grounding\\n(RAG Fact Boundaries)", fillcolor="#0369A1"];
    }

    subgraph cluster_few {
        label = "In-Context Learning (Highest Determinism)";
        color = "#334155";
        fontcolor = "#94A3B8";
        OneShot [label="One-Shot\\n1 Anchor Example\\n(Locks Syntax Schema)", fillcolor="#047857"];
        FewShot [label="Few-Shot (3-5 Exemplars)\\nDiverse Examples + Edge Cases\\n(Calibrates Boundary Ambiguity)", fillcolor="#7C3AED"];
    }

    ZeroShot -> RolePrompt [label="+ Persona"];
    RolePrompt -> ContextPrompt [label="+ Grounding"];
    ContextPrompt -> OneShot [label="+ Syntax Anchor"];
    OneShot -> FewShot [label="+ In-Context Exemplars"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(technique_dot, use_container_width=True)

st.markdown("""
### The Engineering Trade-off Spectrum
* **Zero-Shot**: Lowest latency and cost, but highest formatting variance and susceptibility to ambiguity.
* **Role & Contextual**: Imparts expert perspective and factual grounding without consuming excessive tokens.
* **Few-Shot In-Context Learning**: Highest formatting reliability and schema adherence; consumes higher token context.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - ZERO-SHOT PROMPTING
# ==============================================================================
st.header("2. Zero-Shot Prompting", divider="rainbow")

st.markdown("""
In **Zero-Shot Prompting**, the model is presented with a task description and an input payload with **zero demonstration examples**.
```text
Classify the sentiment of the following movie review as Positive or Negative:
"The cinematography was breathtaking, but the plot was completely incoherent."
```
""")

col_zs1, col_zs2 = st.columns(2)
with col_zs1:
    st.subheader("How Zero-Shot Works Mechanistically")
    st.markdown("""
    * Relies entirely on the model's **pretraining priors** and **instruction-tuning alignment (SFT/RLHF)**.
    * The instruction activates the model's pre-existing concept of sentiment classification and movie reviews.
    * The model computes the next token directly from the conditional probability distribution.
    """)

with col_zs2:
    st.subheader("When Zero-Shot Fails in Production")
    st.markdown("""
    * **Complex or Non-Standard Schemas**: When the target output requires bespoke JSON keys or compound enum classifications (`"Partially_Positive_With_Reservation"`).
    * **Edge-Case Disambiguation**: The model defaults to its own subjective boundary definitions rather than your business logic.
    * **Formatting Inconsistency**: The model may output *"The sentiment is Negative"* instead of the raw requested string `"Negative"`.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - ONE-SHOT PROMPTING
# ==============================================================================
st.header("3. One-Shot Prompting", divider="rainbow")

st.markdown("""
**One-Shot Prompting** provides exactly **one complete demonstration example** before the actual target query.
```text
Task: Extract company names and stock tickers from financial headlines.

Example:
Headline: "Apple Inc. (AAPL) reported record services revenue, outpacing Microsoft (MSFT)."
Extraction: [{"company": "Apple Inc.", "ticker": "AAPL"}, {"company": "Microsoft", "ticker": "MSFT"}]

Headline: "Tesla surged after earnings while Amazon faced retail headwinds."
Extraction:
```
""")

col_os1, col_os2 = st.columns(2)
with col_os1:
    st.markdown("#### The Anchor Effect")
    st.markdown("""
    * **Syntax Locking**: A single example immediately communicates whether you expect JSON, bullet points, Markdown tables, or comma-separated values.
    * **Parser Compatibility**: Dramatically reduces the probability that downstream regex or JSON parsers crash due to unexpected preamble text (*"Sure! Here is the JSON..."*).
    """)

with col_os2:
    st.markdown("#### The Hazard: Single-Exemplar Bias")
    st.markdown("""
    * If the single example contains only 2 companies, the model may assume every headline contains exactly 2 companies!
    * If the example uses title-case keys, the model overfits to that specific casing style.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - FEW-SHOT PROMPTING
# ==============================================================================
st.header("4. Few-Shot Prompting (In-Context Learning)", divider="rainbow")

st.markdown("""
**Few-Shot Prompting** (popularized by Brown et al., GPT-3, 2020) provides **3 to 5 diverse demonstration exemplars**. 
It is the most reliable prompt engineering method for complex classification, entity extraction, and structured transformation.
""")

st.markdown("""
### Critical Principles of Few-Shot Exemplar Engineering
""")

fs_col1, fs_col2 = st.columns(2)
with fs_col1:
    st.markdown("#### 1. Label Distribution Balance")
    st.markdown("""
    * **The Majority Label Trap (Zhao et al., 2021)**: If you provide 4 Positive examples and 1 Negative example, the model develops an overwhelming **recency and frequency bias** toward predicting Positive regardless of input content!
    * **Rule**: Maintain strict 1:1 balance across all target classes in your few-shot exemplar set.
    """)

    st.markdown("#### 2. Demonstration Diversity")
    st.markdown("""
    * Include short inputs, long inputs, and ambiguous edge cases.
    * Show how to handle negative instances (e.g. what to output when an entity is missing: `[]` vs `null`).
    """)

with fs_col2:
    st.markdown("#### 3. Exemplar Ordering Bias (Recency Bias)")
    st.markdown("""
    * LLMs attend disproportionately to the **final exemplar** in the list.
    * If the last example belongs to Class A, predictions across test inputs skew toward Class A!
    * **Mitigation**: Randomize exemplar order in production, or test permutation invariance across test sets.
    """)

    st.markdown("#### 4. Formatting Uniformity")
    st.markdown("""
    * Use identical XML or Markdown delimiters across all examples.
    * Inconsistent delimiter usage between demonstrations destroys in-context pattern matching.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - ROLE PROMPTING
# ==============================================================================
st.header("5. Role Prompting", divider="rainbow")

st.markdown("""
**Role Prompting** explicitly assigns a persona, professional identity, and standard of rigor to the language model 
(e.g., *"You are a Principal Software Architect specialising in distributed systems"*).
""")

col_role1, col_role2 = st.columns(2)
with col_role1:
    st.subheader("Mechanisms of Latent Circuit Steering")
    st.markdown("""
    * Foundation models contain billions of diverse token associations, from elementary school blogs to peer-reviewed arXiv papers.
    * Assigning an expert role acts as a **soft semantic prior**, activating high-tier technical vocabulary, formal reasoning structures, and rigorous formatting conventions.
    * Research by Zheng et al. showed role prompting significantly boosts performance on medical, legal, and coding benchmarks.
    """)

with col_role2:
    st.subheader("Effective vs Ineffective Role Prompts")
    st.markdown("""
    * **Ineffective (Generic Fluff)**: *"You are the world's greatest smartest genius AI."* (Produces arrogant, hallucination-prone prose).
    * **Effective (Specific Operational Persona)**: *"You are a Staff Database Reliability Engineer. You evaluate SQL queries strictly on index selectivity, lock escalation risks, and buffer pool memory consumption. Provide concrete execution plan metrics."*
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - INSTRUCTION PROMPTING
# ==============================================================================
st.header("6. Instruction Prompting", divider="rainbow")

st.markdown("""
**Instruction Prompting** structures tasks around direct, unambiguous imperative directives and explicit operational boundaries.
""")

st.markdown("""
| Dimension | Weak Instruction | Strong Production Instruction |
| :--- | :--- | :--- |
| **Action Verb** | *"Can you help look at this text?"* | *"Extract all ISO-4217 currency codes and amounts."* |
| **Edge Cases** | *"Try to find dates."* | *"If no date is mentioned in the input, return null."* |
| **Output Bounds** | *"Make it brief."* | *"Summarize in exactly 3 bullet points, each under 15 words."* |
| **Negative Guardrail** | *"Don't guess."* | *"Rely strictly on provided data; do not extrapolate beyond stated facts."* |
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CONTEXTUAL PROMPTING
# ==============================================================================
st.header("7. Contextual Prompting", divider="rainbow")

st.markdown("""
**Contextual Prompting** is the foundational paradigm of **Retrieval-Augmented Generation (RAG)**: 
conditioning the model's generation strictly upon a verified external reference document rather than its internal parametric memory.
""")

col_cxt1, col_cxt2 = st.columns(2)
with col_cxt1:
    st.subheader("The Grounding Contract")
    st.markdown("""
    An effective contextual prompt establishes an explicit legalistic boundary:
    ```text
    <context>
    {{RETRIEVED_DOCUMENT_CHUNKS}}
    </context>
    
    Instruction: Answer the user question strictly using facts stated in <context>.
    If the context does not explicitly provide the answer, output:
    "The provided context does not contain sufficient information to answer."
    Do not use prior parametric knowledge to speculate.
    ```
    """)

with col_cxt2:
    st.subheader("Handling Contradictory Evidence")
    st.markdown("""
    * What happens when the context contradicts the model's pretraining knowledge (e.g., context says *"The Eiffel Tower was built in 1985"* vs true 1889)?
    * Without strict contextual instructions, base models frequently favor parametric pretraining facts over provided context.
    * Explicit contextual grounding overrides parametric biases, enforcing document fidelity.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Technique In-Context Prompt Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained module implements automated construction of Zero-Shot, Few-Shot, Role-Conditioned, 
and Contextually Grounded prompts with token budget auditing.
""")

st.code("""
from typing import List, Dict, Any

class ProductionPromptTechniquesEngine:
    \"\"\"
    Engine implementing core prompt techniques: Zero-Shot, Few-Shot, Role, and Contextual.
    \"\"\"
    def build_zero_shot(self, task: str, user_input: str) -> str:
        return f\"<instruction>\\n{task.strip()}\\n</instruction>\\n\\n<input_data>\\n{user_input.strip()}\\n</input_data>\"

    def build_few_shot(self, task: str, exemplars: List[Dict[str, str]], user_input: str) -> str:
        ex_str = ""
        for idx, ex in enumerate(exemplars, 1):
            ex_str += f\"<example_{idx}>\\n<input>\\n{ex['input']}\\n</input>\\n<output>\\n{ex['output']}\\n</output>\\n</example_{idx}>\\n\\n\"
        return f\"<instruction>\\n{task.strip()}\\nFollow the exact format shown in the demonstrations below.\\n</instruction>\\n\\n<demonstrations>\\n{ex_str.strip()}\\n</demonstrations>\\n\\n<input_data>\\n{user_input.strip()}\\n</input_data>\"

    def build_role_prompt(self, role: str, task: str, user_input: str) -> str:
        return f\"<role>\\n{role.strip()}\\n</role>\\n\\n<instruction>\\n{task.strip()}\\n</instruction>\\n\\n<input_data>\\n{user_input.strip()}\\n</input_data>\"
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Prompt Technique Comparator & Exemplar Selector", divider="rainbow")

engine = ProductionPromptTechniquesEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("Prompt Technique Configuration")
    technique_choice = st.selectbox(
        "Select Technique",
        ["Zero-Shot Prompting", "Few-Shot In-Context Learning (3 Exemplars)", "Role Prompting", "Strict Contextual Grounding (RAG)"]
    )
    user_task = st.text_area("Task Directive", value="Extract product sentiment and affected component aspects.")
    user_input = st.text_input("Input Query / Text", value="The camera is incredible, but battery drains in 3 hours.")

    if technique_choice == "Zero-Shot Prompting":
        rendered_prompt = engine.build_zero_shot(user_task, user_input)
    elif technique_choice == "Few-Shot In-Context Learning (3 Exemplars)":
        rendered_prompt = engine.build_few_shot(user_task, engine.default_exemplars, user_input)
    elif technique_choice == "Role Prompting":
        user_role = st.text_input("Role / Persona", value="You are a Principal Hardware Quality Assurance Engineer.")
        rendered_prompt = engine.build_role_prompt(user_role, user_task, user_input)
    else:
        sample_context = st.text_area("Reference Document Context", value="Warranty Policy: Battery degradation under 80% capacity within 12 months is eligible for replacement. Screen burn-in is not covered.")
        rendered_prompt = engine.build_contextual_prompt(sample_context, user_task, user_input)

    overhead = engine.estimate_overhead(rendered_prompt)

with wb_col2:
    st.subheader("Rendered Prompt & In-Context Telemetry")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Prompt Tokens", f"~{overhead['token_count']} Tokens")
    with col_m2:
        st.metric("Estimated Cost per Call", f"${overhead['cost_gpt4o']:.5f}", "GPT-4o Rate")

    st.markdown("##### Assembled In-Context Buffer:")
    st.code(rendered_prompt, language="xml")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: What is the 'Majority Label Bias' in Few-Shot In-Context Learning?"):
        st.write("""
        Majority Label Bias (Zhao et al., 2021) occurs when the distribution of labels across few-shot exemplars is imbalanced (e.g. 4 positive examples and 1 negative example).
        Language models attend heavily to token frequencies in the prompt prefix, causing them to disproportionately predict the majority class on ambiguous test samples regardless of true input content.
        Few-shot prompts must always maintain a strict 1:1 balanced distribution across target classes.
        """)
    with st.expander("Q2: How does Demonstration Ordering (Recency Bias) impact Few-Shot performance?"):
        st.write("""
        Due to causal self-attention and recency bias, the model pays disproportionate attention to the final exemplar presented before the test query.
        If the last exemplar in a 4-shot prompt belongs to Class A, model accuracy drops if the true test label is Class B.
        In production benchmarking, few-shot evaluations should test multiple random permutations of exemplar order to verify stability.
        """)
    with st.expander("Q3: When is One-Shot prompting sufficient over multi-exemplar Few-Shot?"):
        st.write("""
        One-Shot prompting is ideal when the primary objective is **syntax and schema locking** rather than complex boundary disambiguation.
        For example, demonstrating how a natural language request maps to a specific JSON schema or SQL table syntax only requires a single well-formed exemplar to guide output formatting.
        """)

with q2:
    with st.expander("Q4: Why does Role Prompting improve reasoning accuracy on specialized domains?"):
        st.write("""
        Role prompting acts as a semantic prior over the model's vast multi-domain representation space.
        Conditioning the prompt with a senior technical identity activates linguistic circuits associated with high-quality academic papers, architectural reviews, and formal proofs.
        It suppresses superficial, colloquial, or overly casual token generation paths.
        """)
    with st.expander("Q5: How do you enforce factual grounding in Contextual Prompting (RAG)?"):
        st.write("""
        Factual grounding requires:
        1. Encapsulating retrieved documents in clear structural delimiters (`<context>`).
        2. Explicit negative instructions: *"Answer strictly using facts established in <context>. If context does not contain the answer, state that information is missing."*
        3. Instructing the model to cite exact sentence references or quote excerpts from the context.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Imbalanced Few-Shot Exemplars**
    * **Failure**: Providing 3 Positive and 0 Negative examples in a sentiment analysis prompt.
    * **Impact**: Model develops severe positive prediction bias, failing completely on negative user feedback.
    * **Remedy**: Always maintain strict class parity in few-shot demonstration sets.
    """)
    st.error("""
    **🚨 Production Trap 2: Contradictory Exemplar Formats**
    * **Failure**: Example 1 outputs JSON; Example 2 outputs Markdown bullet points.
    * **Impact**: Confuses attention circuits; model outputs malformed hybrid text that crashes downstream parsers.
    * **Remedy**: Enforce rigorous formatting consistency across all demonstration samples.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: The 'Smartest AI' Role Anti-Pattern**
    * **Failure**: Setting role as *"You are the most brilliant genius AI that knows everything."*
    * **Impact**: Encourages extreme factual overconfidence, eliminating calibrated uncertainty and increasing hallucinations.
    * **Remedy**: Ground roles in specific operational responsibilities rather than exaggerated intellect.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Permitting Parametric Fallback in RAG Contexts**
    * **Failure**: Forgetting to explicitly forbid external knowledge when providing context.
    * **Impact**: Model hallucinates facts from pretraining when reference context has gaps.
    * **Remedy**: Add strict negative fallback: *"Do not speculate or extrapolate beyond the provided text."*
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Ideal Few-Shot Exemplar Count"):
        st.write("What is the typical recommended exemplar count for Few-Shot ICL?")
        st.info("3 to 5 diverse, balanced exemplars.")
    with st.expander("⚡ Flashcard 2: Recency Bias in Few-Shot"):
        st.write("Which exemplar exerts the strongest influence on model completions?")
        st.info("The final exemplar presented immediately before the test input.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: One-Shot Primary Function"):
        st.write("What is the main benefit of One-Shot prompting over Zero-Shot?")
        st.info("Locks down structural syntax and schema formatting with minimal token overhead.")
    with st.expander("⚡ Flashcard 4: Grounding Enforcement"):
        st.write("What phrase must be included in RAG contextual prompts to stop hallucination?")
        st.info("'Answer strictly using the provided context; state if context is insufficient.'")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Prompting Techniques before advancing to Lesson 15.3 (Reasoning-Oriented Prompting):
""")

chk1 = st.checkbox("I understand the trade-offs across the Zero-Shot to Few-Shot prompting spectrum.", value=True)
chk2 = st.checkbox("I know why Majority Label Bias occurs and how to balance few-shot exemplar distributions.", value=True)
chk3 = st.checkbox("I understand how Recency Bias affects demonstration ordering in in-context learning.", value=True)
chk4 = st.checkbox("I can construct effective operational Role prompts that steer latent vocabulary circuits.", value=True)
chk5 = st.checkbox("I know how to write strict contextual grounding instructions for RAG pipelines.", value=True)
chk6 = st.checkbox("I understand why One-Shot prompting is often sufficient for schema locking.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 15.3: Reasoning-Oriented Prompting.")
