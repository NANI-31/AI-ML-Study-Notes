import streamlit as st
import pandas as pd
import graphviz

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.1 What is Prompt Engineering? | GenAI Roadmap",
    page_icon="✍️",
    layout="wide",
)

# Custom Styling for Glassmorphism & High-Contrast Typography
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    /* Pin graphviz iframes to a fixed size — stops ResizeObserver loop during sidebar CSS transitions */
    .stGraphVizChart iframe {
        overflow: hidden !important;
        resize: none !important;
    }
    /* Hint the browser to composite the sidebar on its own GPU layer — prevents layout thrashing */
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("✍️ 1.1 What is Prompt Engineering? — Foundations, Anatomy & Comparative Architectures")
st.markdown(
    """
Welcome to **Lesson 1.1: What is Prompt Engineering?**  
This module establishes the core theoretical, algorithmic, and practical foundations of prompting. 
You will examine how natural language acts as a programming interface for autoregressive Large Language Models (LLMs), 
understand the difference between prompts and instructions, analyze critical trade-offs against traditional code, 
fine-tuning, and RAG, and test interactive architectural diagrams and interview scenarios.
"""
)

st.divider()

# ==============================================================================
# 1. Definition of Prompt Engineering
# ==============================================================================
st.header("1. Definition of Prompt Engineering", divider="rainbow")

col_def1, col_def2 = st.columns([3, 2])

with col_def1:
    st.markdown(
        r"""
### What is Prompt Engineering?

**Prompt Engineering (PE)** is the systematic discipline of designing, refining, structuring, and optimizing 
natural language inputs (and contextual framing) to reliably guide **Large Language Models (LLMs)** 
toward generating accurate, structured, safe, and contextually aligned completions—**without altering model weights**.

#### Formal Technical Definition:
In autoregressive transformer models, text generation is modeled as conditional probability over a vocabulary $V$:
$$P(w_{t} \mid w_{1}, w_{2}, \dots, w_{t-1})$$

A prompt $X = (x_1, x_2, \dots, x_k)$ acts as the initial **conditioning context prefix**. 
Prompt Engineering is the optimization of the sequence $X$ such that the generated continuation $Y = (y_1, y_2, \dots, y_m)$ 
maximizes the probability of satisfying a target objective function $f(Y)$, such as factual accuracy, schema compliance (JSON), 
or safety constraints:

$$\max_{X} \; \mathbb{E}_{Y \sim P(\cdot \mid X)} [f(Y)]$$

#### Core Pillars of Prompt Engineering:
1. **In-Context Conditioning**: Providing domain context, background knowledge, and boundary constraints directly inside the context window.
2. **Latent Space Steering**: Activating specific subnetworks and semantic representations pre-trained in the foundation model.
3. **Format & Schema Enforcement**: Forcing structured serializations (JSON, YAML, Markdown, XML) via delimiters and structural schemas.
4. **Reasoning Scaffolding**: Eliciting step-by-step cognitive traces through Chain-of-Thought (CoT), self-consistency, and ReAct paradigms.
"""
    )

with col_def2:
    st.markdown("#### 🏗️ The Prompt Conditioning Pipeline")
    def_dot = """
    digraph G {
        rankdir=TB;
        size="4,5!";
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#475569", fillcolor="#1E293B", penwidth=1.5];
        edge [color="#64748B", penwidth=1.5, fontname="Helvetica", fontcolor="#94A3B8"];

        A [label="User Objective / Raw Task", fillcolor="#0F172A", color="#3B82F6"];
        B [label="Structured Prompt Template\n(System + Context + Instruction + Delimiters)", fillcolor="#1E293B", color="#6366F1"];
        C [label="Foundation LLM\n(Attention Mechanism & KV Cache)", fillcolor="#1E293B", color="#10B981"];
        D [label="Target Completion\n(JSON, Classification, Plan, Summary)", fillcolor="#0F172A", color="#F59E0B"];

        A -> B [label=" Engineered Formatting"];
        B -> C [label=" Prefix Tokens Vector"];
        C -> D [label=" Autoregressive Decoding"];
    }
    """
    st.graphviz_chart(def_dot, use_container_width=False)

st.divider()

# ==============================================================================
# 2. What is a Prompt? vs 3. What is an Instruction?
# ==============================================================================
st.header("2. What is a Prompt? vs 3. What is an Instruction?", divider="rainbow")

st.markdown(
    """
In technical interviews, candidates often conflate **Prompt** and **Instruction**. 
Understanding their structural hierarchy is fundamental to designing production-grade LLM applications.
"""
)

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        """
### 📦 What is a Prompt?
A **Prompt** is the **entire aggregate input payload** supplied to the model's context window during inference. 
It encapsulates the full contextual environment, including:
- **System Persona / Instructions**: High-level behavioral rules and guardrails.
- **Few-Shot Exemplars**: Concrete pairs of input/output demonstrations.
- **External Retrieved Context**: Ground truth retrieved via RAG or SQL queries.
- **Dynamic User Input**: The runtime variable data being analyzed.
- **Task Directives / Constraints**: Specific rules, lengths, and format requirements.
- **Output Framing**: Trailing delimiters signaling the start of the assistant response.
"""
    )
    st.code(
        """# Example of a FULL PROMPT Payload:
\"\"\"
System: You are an expert data triage bot. Respond ONLY in valid JSON.
Context: Today's date is 2026-10-12. Server #4 is reporting CPU spike 98%.
Few-Shot Example:
  Input: Memory usage 40%
  Output: {"severity": "LOW", "action": "monitor"}
User Input: Server #4 CPU spike 98%.
Instruction: Categorize severity and recommend remediation action.
Output Format: {"severity": "...", "action": "..."}
\"\"\"""",
        language="text",
    )

with col_p2:
    st.markdown(
        """
### 🎯 What is an Instruction?
An **Instruction** is the **specific imperative task directive** located *inside* the prompt that commands the model on what transformation or computation to execute.
- It is the **action verb**: *"Classify"*, *"Summarize"*, *"Extract"*, *"Translate"*, *"Evaluate"*.
- While the context provides the *data*, the instruction dictates the *algorithm*.
- Vague instructions yield ambiguous, non-deterministic outputs; precise instructions with explicit constraints yield reliable production systems.
"""
    )
    st.code(
        """# Examples of Raw INSTRUCTIONS:
- "Summarize the customer review in exactly 3 bullet points."
- "Extract all monetary amounts, currency codes, and dates from the invoice."
- "Determine if the sentiment is POSITIVE, NEGATIVE, or NEUTRAL. Return only the label."
- "Translate the Python function into idiomatic Rust with error handling."
""",
        language="text",
    )

# Interactive Anatomy Inspector
st.subheader("🔬 Interactive Prompt Anatomy Inspector", divider="blue")
st.markdown("Toggle components on and off to observe how a structured production prompt is constructed:")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    inc_system = st.checkbox("Include System Persona", value=True)
    inc_context = st.checkbox("Include RAG Context", value=True)
with col_ctrl2:
    inc_fewshot = st.checkbox("Include Few-Shot Exemplar", value=True)
    inc_constraints = st.checkbox("Include Negative Constraints", value=True)
with col_ctrl3:
    inc_schema = st.checkbox("Include Structured JSON Schema", value=True)

constructed_prompt_parts = []
if inc_system:
    constructed_prompt_parts.append("[SYSTEM PERSONA]\nYou are an enterprise fraud analyst AI. You evaluate transaction logs with strict compliance standards.")
if inc_context:
    constructed_prompt_parts.append("[RETRIEVED CONTEXT]\nUser ID #9821 typically transacts from Dallas, TX (Avg: $45.00). Current transaction originated from Kiev, Ukraine.")
if inc_fewshot:
    constructed_prompt_parts.append("[FEW-SHOT DEMONSTRATION]\nInput: Dallas, TX -> $52.00\nOutput: {\"risk_score\": 0.05, \"decision\": \"ALLOW\"}")
constructed_prompt_parts.append("[USER INPUT DATA]\nLocation: Kiev, Ukraine | Amount: $4,950.00 | Merchant: CryptoBridge Exchange")
constructed_prompt_parts.append("[INSTRUCTION]\nEvaluate the risk score from 0.0 to 1.0 and determine whether to FLAG, BLOCK, or ALLOW this transaction.")
if inc_constraints:
    constructed_prompt_parts.append("[CONSTRAINTS & GUARDRAILS]\n- Do NOT include conversational filler, greetings, or markdown code blocks.\n- If location anomaly is detected, risk_score must be >= 0.85.")
if inc_schema:
    constructed_prompt_parts.append("[OUTPUT SCHEMA]\n{\n  \"risk_score\": float,\n  \"decision\": \"BLOCK\" | \"FLAG\" | \"ALLOW\",\n  \"rationale\": string\n}")

constructed_prompt = "\n\n".join(constructed_prompt_parts)

st.code(constructed_prompt, language="markdown")
st.caption(f"Estimated token footprint: ~{len(constructed_prompt.split()) * 1.3:.0f} tokens | Component count: {len(constructed_prompt_parts)}")

st.divider()

# ==============================================================================
# 4. Why Prompt Engineering is Important & 5. Role in Generative AI
# ==============================================================================
st.header("4. Why Prompt Engineering is Important & 5. Role in GenAI", divider="rainbow")

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown(
        """
### 💡 Why Prompt Engineering is Crucial in Production

1. **Massive Cost & Latency Reduction**:
   - Modifying a prompt costs **$0 in compute infrastructure**, whereas fine-tuning a 70B parameter model costs thousands of dollars in GPU clusters.
   - Prompt iteration takes **seconds**, while dataset preparation, training, and model evaluation cycles take days or weeks.

2. **Deterministic Control Over Probabilistic Models**:
   - LLMs are inherently non-deterministic next-token predictors.
   - Structured prompting (delimiters, schema enforcement, zero-shot CoT) turns stochastic token generation into a reliable, parseable software API.

3. **Hallucination Mitigation**:
   - Grounding prompts with retrieved context and explicit negative constraints (*"If the context does not contain the answer, reply 'UNKNOWN'"*) reduces hallucinations by up to 90%.

4. **Security & Guardrail Hardening**:
   - Defensive prompt architecture prevents **Prompt Injection**, **System Prompt Leakage**, and **Jailbreaking** attempts by untrusted users.
"""
    )

with col_imp2:
    st.markdown(
        """
### 🤖 Role of Prompts Across the Generative AI Stack

Prompts are not merely "queries"; they are the **runtime executable scripts** of Foundation Models.
"""
    )
    stack_dot = """
    digraph G {
        rankdir=TB;
        size="4,6!";
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#475569", fillcolor="#1E293B", penwidth=1.5];
        edge [color="#64748B", penwidth=1.5, fontname="Helvetica", fontcolor="#94A3B8"];

        App [label="End-User Application (Web / Mobile / Slack)", fillcolor="#0F172A", color="#3B82F6"];
        PE [label="Prompt Engineering & Orchestration\n(Templates • Few-Shot • Guardrails • LangChain)", fillcolor="#3B82F6", color="#60A5FA", fontcolor="#FFFFFF"];
        RAG [label="RAG & Knowledge Retrieval\n(Vector DBs • BM25 • Hybrid Search)", fillcolor="#1E293B", color="#8B5CF6"];
        Model [label="Foundation LLMs\n(Claude • GPT-4o • Llama 3 • Gemini)", fillcolor="#1E293B", color="#10B981"];
        Eval [label="LLM Evaluation & Observability\n(LangSmith • Promptfoo • DeepEval)", fillcolor="#1E293B", color="#F59E0B"];

        App -> PE [label=" User Intent"];
        PE -> RAG [label=" Query Context"];
        RAG -> PE [label=" Injected Knowledge"];
        PE -> Model [label=" Fully Conditioned Prompt"];
        Model -> PE [label=" Raw Tokens"];
        PE -> Eval [label=" Audit & Tracing"];
        PE -> App [label=" Verified Structured Response"];
    }
    """
    st.graphviz_chart(stack_dot, use_container_width=False)

st.divider()

# ==============================================================================
# 6. Prompt Engineering vs Traditional Programming
# ==============================================================================
st.header("6. Prompt Engineering vs Traditional Programming", divider="rainbow")

st.markdown(
    """
A classic interview question asks: *"How does software engineering differ when programming with prompts versus traditional deterministic code?"*
"""
)

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.markdown(
        """
#### 💻 Traditional Programming (Deterministic)
- **Input**: Explicit, typed variables and data structures.
- **Logic**: Deterministic algorithms, loops, if/else branches, ASTs.
- **Output**: Exactly predictable output given identical inputs ($f(x) = y$).
- **Failure Modes**: Syntax errors, runtime exceptions, NullPointerExceptions.
- **Testing**: Deterministic unit tests with binary assertions (`assert result == expected`).
- **Debugging**: Breakpoints, stack traces, step-through debuggers.
"""
    )
    st.code(
        r"""# Traditional Code: Deterministic Email Validation
import re

def is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

# 100% deterministic assertion
assert is_valid_email("user@example.com") is True
assert is_valid_email("invalid-email") is False""",
        language="python",
    )

with col_comp2:
    st.markdown(
        """
#### 🧠 Prompt Engineering (Probabilistic & Semantic)
- **Input**: Natural language prompt with fuzzy semantic instructions.
- **Logic**: Multi-billion parameter neural network matrix multiplications.
- **Output**: Probabilistic token sequences sampled from probability distributions.
- **Failure Modes**: Hallucinations, format drift, refusal to answer, prompt injection.
- **Testing**: LLM-as-a-judge evaluations, semantic similarity, precision/recall benchmarks.
- **Debugging**: Prompt perturbation, temperature tuning, few-shot injection.
"""
    )
    st.code(
        """# Prompt Engineering: Semantic Sentiment Classifier
prompt = \"\"\"
Classify the sentiment of the customer feedback into POSITIVE, NEUTRAL, or NEGATIVE.
Output only the category name.

Feedback: "The delivery arrived late, but the support rep refunded shipping immediately."
Category:\"\"\"

# Probabilistic output (Requires temperature=0.0 and eval suites for consistency)
# Output: NEUTRAL (or POSITIVE depending on token sampling)""",
        language="python",
    )

st.divider()

# ==============================================================================
# 7. Prompt Engineering vs Fine-Tuning & 8. Prompt Engineering vs RAG
# ==============================================================================
st.header("7. Prompt Engineering vs Fine-Tuning vs 8. RAG", divider="rainbow")

st.markdown(
    """
Every AI architect must determine whether a business requirement is solved via **Prompt Engineering**, 
**Retrieval-Augmented Generation (RAG)**, or **Fine-Tuning (PEFT/LoRA)**.
"""
)

tab_matrix, tab_finetune, tab_rag, tab_flow = st.tabs(
    [
        "📊 Master Architectural Comparison",
        "⚖️ Prompting vs Fine-Tuning",
        "⚖️ Prompting vs RAG",
        "🧭 Architectural Decision Flowchart",
    ]
)

with tab_matrix:
    st.markdown("### 🔍 Comparative Engineering Matrix")
    comp_df = pd.DataFrame(
        {
            "Dimension": [
                "Primary Purpose",
                "Weight Updates",
                "Data Requirements",
                "Speed to Deploy",
                "Hardware / Cost",
                "Knowledge Freshness",
                "Hallucination Risk",
                "Best Used For",
            ],
            "Prompt Engineering": [
                "Task steering, instructions, reasoning elicitation",
                "❌ Zero (Weights remain frozen)",
                "Zero-shot to few exemplars (1–10 examples)",
                "⚡ Minutes (Ultra rapid)",
                "Minimal (Standard API inference cost)",
                "Static (Limited to pre-training cutoff)",
                "Moderate (Requires grounding constraints)",
                "General tasks, rapid prototyping, workflows, chatbots",
            ],
            "RAG (Retrieval-Augmented)": [
                "Injecting proprietary external knowledge & truth",
                "❌ Zero (Weights remain frozen)",
                "Unstructured documents, PDFs, DBs (Chunks)",
                "Hours to Days (Vector DB pipeline)",
                "Moderate (Embedding API + Vector storage)",
                "🟢 Real-Time (Updates with newest docs)",
                "Low (Directly grounded in retrieved citations)",
                "Company wikis, document Q&A, legal/medical search",
            ],
            "Fine-Tuning (LoRA / SFT)": [
                "Adapting style, specialized syntax, or domain tone",
                "✅ Yes (Adapter weights or full weights updated)",
                "Thousands to tens of thousands of (prompt, completion) pairs",
                "Days to Weeks (Data curation + GPU runs)",
                "High (Requires GPU clusters: H100/A100)",
                "Static (Frozen at time of fine-tuning)",
                "Moderate to High (Cannot verify source citations)",
                "Unique domain jargon, specialized code syntax, latency optimization",
            ],
        }
    )
    st.dataframe(comp_df, width="stretch", hide_index=True)

with tab_finetune:
    st.markdown("### ⚖️ Prompt Engineering vs Fine-Tuning")
    col_ft1, col_ft2 = st.columns(2)
    with col_ft1:
        st.markdown(
            """
#### ✍️ When to Choose Prompt Engineering:
- **Fast Iteration**: You need to test an idea within hours.
- **Dynamic Workflows**: The task changes frequently based on user requests.
- **General Reasoning**: The base model already understands the language and logic (e.g., GPT-4o, Claude 3.5).
- **Zero GPU Management**: You do not have infrastructure engineers to manage training clusters.
- **Cost**: Eliminates training runs, checkpoint storage, and custom model hosting costs.
"""
        )
    with col_ft2:
        st.markdown(
            """
#### 🔧 When to Choose Fine-Tuning:
- **Style & Persona Enforcement**: Teaching an LLM a very distinctive voice or medical shorthand across millions of interactions.
- **Token Efficiency & Latency**: Eliminating 1,000-token system prompts and few-shot examples so you can query with a 50-token prompt.
- **Uncommon Syntax / Domain Code**: Emitting proprietary DSLs, custom SQL dialects, or rare hardware formats where base models fail.
- **Small Model Specialization**: Distilling a 70B model's performance into a fine-tuned 8B model to run on edge devices.
"""
        )
    st.info("💡 **Interview Rule of Thumb**: *Prompt first, retrieve second (RAG), fine-tune only when style, latency, or niche formatting cannot be achieved with in-context prompting.*")

with tab_rag:
    st.markdown("### ⚖️ Prompt Engineering vs RAG")
    col_rag1, col_rag2 = st.columns(2)
    with col_rag1:
        st.markdown(
            """
#### 🤝 The Symbiotic Relationship:
Candidates often mistakenly view RAG and Prompt Engineering as competing alternatives. 
**RAG fundamentally relies on Prompt Engineering to succeed.**

1. **RAG retrieves context**: The Vector DB finds the top-3 most relevant document chunks.
2. **Prompt Engineering synthesizes**: An engineered prompt binds the chunks, instructs the model to ignore hallucinated prior beliefs, and demands cited evidence.
"""
        )
    with col_rag2:
        st.code(
            """# How Prompt Engineering Bridges RAG:
rag_prompt = f\"\"\"
System: You are an internal compliance assistant. 
Answer the user question using ONLY the provided context snippets below.
If the answer cannot be determined from the context, state "I cannot find this in policy documentation."

[CONTEXT SNIPPETS]:
{retrieved_chunks}

[QUESTION]:
{user_query}

[ANSWER GUIDELINES]:
- Cite the section number for every claim.
- Do not assume facts not explicitly stated.
\"\"\"""",
            language="python",
        )

with tab_flow:
    st.markdown("### 🧭 Strategic Decision Flowchart")
    flow_dot = """
    digraph G {
        rankdir=TB;
        size="5,6!";
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#475569", fillcolor="#1E293B", penwidth=1.5];
        edge [color="#64748B", penwidth=1.5, fontname="Helvetica", fontcolor="#94A3B8"];

        Start [label="New GenAI Feature Request", fillcolor="#0F172A", color="#3B82F6"];
        Q1 [label="Does the model need private, dynamic,\nor real-time external knowledge?", shape=diamond, color="#F59E0B"];
        RAG_Node [label="Implement RAG\n(Vector Search + Grounded System Prompt)", fillcolor="#1E293B", color="#10B981"];
        Q2 [label="Does the model fail on style, tone,\nor require extreme latency/token reduction?", shape=diamond, color="#F59E0B"];
        FT_Node [label="Implement Fine-Tuning\n(LoRA / SFT Dataset Creation)", fillcolor="#1E293B", color="#8B5CF6"];
        PE_Node [label="Implement Pure Prompt Engineering\n(Few-Shot • System Prompt • Schema Constraints)", fillcolor="#1E293B", color="#3B82F6"];

        Start -> Q1;
        Q1 -> RAG_Node [label=" Yes"];
        Q1 -> Q2 [label=" No"];
        Q2 -> FT_Node [label=" Yes"];
        Q2 -> PE_Node [label=" No (Default)"];
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=False)

st.divider()

# ==============================================================================
# Technical Interview Questions & Flashcards
# ==============================================================================
st.header("🎯 Technical Interview Preparation: Core Questions & Scenarios", divider="rainbow")

qa_tab1, qa_tab2, qa_tab3 = st.tabs(
    [
        "❓ High-Frequency Interview Questions",
        "⚠️ Interview Traps & Naive Answers",
        "📇 Active Recall Flashcards",
    ]
)

with qa_tab1:
    with st.expander("Q1: What is the difference between In-Context Learning and Model Training?", expanded=True):
        st.markdown(
            """
**Answer:**  
In-context learning occurs entirely at inference time within the transformer's activation space without computing gradients or modifying model weights. 
The input prompt condition sets the attention matrix, activating pre-existing latent representations learned during pre-training. 
In contrast, model training (pre-training or fine-tuning) uses backpropagation and gradient descent to update the physical weight tensors stored on disk. 
Consequently, in-context learning is zero-cost in persistent infrastructure, immediately reversible, but temporary to the lifetime of the API call.
"""
        )

    with st.expander("Q2: What is the architectural difference between a Prompt and an Instruction?", expanded=False):
        st.markdown(
            """
**Answer:**  
A prompt is the holistic input payload provided to the LLM's context window, encompassing system prompts, few-shot demonstrations, grounding context, dynamic data, and formatting delimiters. 
An instruction is the specific imperative directive (e.g., "Summarize", "Classify", "Extract") embedded within that prompt dictating the computational task to perform. 
Confusing the two leads to poorly modularized prompt templates where instructions and context are blended without delimiters, causing instruction drift and injection vulnerability.
"""
        )

    with st.expander("Q3: Why can adding too much detail to a system prompt degrade LLM performance?", expanded=False):
        st.markdown(
            """
**Answer:**  
Overly verbose system prompts dilute the model's attention weights across hundreds of non-critical tokens, causing the "lost-in-the-middle" phenomenon and prompt fatigue. 
When conflicting constraints or excessive rules are introduced, the model's likelihood of contradicting itself increases as competing tokens pull probability mass in conflicting directions. 
Production prompt engineering emphasizes concise, modular constraints with structured delimiters (XML or Markdown tags) rather than unstructured prose.
"""
        )

    with st.expander("Q4: When should an enterprise choose Prompt Engineering over Fine-Tuning?", expanded=False):
        st.markdown(
            """
**Answer:**  
An enterprise should always exhaust prompt engineering before evaluating fine-tuning. 
Prompt engineering requires zero training compute, zero GPU cluster maintenance, operates on frozen foundation models, and allows immediate updates when business logic changes. 
Fine-tuning should only be reserved for situations where token latency must be minimized (stripping long instructions), specialized non-natural languages/DSLs must be learned, or distinct stylistic tone cannot be guided by few-shot exemplars.
"""
        )

with qa_tab2:
    st.markdown("### 🪤 Common Interview Traps & Winning Responses")
    
    with st.expander("Trap 1: 'Does adding polite words like Please and Thank You improve LLM responses?'", expanded=True):
        st.markdown(
            """
- **Naive Candidate Response**: *"Yes, it makes the AI happier and more likely to follow instructions because it was trained on human conversations."*
- **The Trap**: Anthropomorphizing the model and ignoring tokenization mechanics.
- **Winning Response**: *"Adding polite phrasing adds unnecessary tokens that dilute attention and can unintentionally steer the model into conversational, chatty modes when concise technical execution is required. However, empirical studies show that clear, professional, and authoritative framing reinforces persona adherence. Clean imperatives ('Extract the JSON...') are superior to polite requests in production systems."*
"""
        )

    with st.expander("Trap 2: 'Is Prompt Engineering obsolete because models are getting smarter?'", expanded=False):
        st.markdown(
            """
- **Naive Candidate Response**: *"Yes, newer models understand anything you type, so you don't need prompt engineering anymore."*
- **The Trap**: Equating conversational chatbot usage with production software engineering.
- **Winning Response**: *"No, as models become more capable, the role of Prompt Engineering shifts from basic syntax coaxing to complex system orchestration—such as schema enforcement, tool-use protocols (ReAct), multi-agent debate, security against indirect prompt injection, and continuous evaluation pipelines. Prompt engineering is the programming interface for non-deterministic AI systems."*
"""
        )

with qa_tab3:
    st.markdown("### 📇 Quick Recall Flashcards")
    fc_cols = st.columns(3)
    
    with fc_cols[0]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Prompt vs Instruction**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Prompt is the complete context payload (persona + context + examples + instruction). Instruction is the specific imperative action verb/task.")

    with fc_cols[1]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**In-Context Conditioning**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Steering model probabilities via prefix tokens without modifying weights, relying on the transformer attention mechanism.")

    with fc_cols[2]:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**When to use RAG vs Fine-Tuning**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Use RAG for dynamic external knowledge & factual grounding; use Fine-Tuning for style, tone, specialized syntax, or latency reduction.")

st.divider()

# ==============================================================================
# Summary & Checklist
# ==============================================================================
st.header("📋 Lesson 1.1 Summary & Review Checklist", divider="rainbow")

col_chk1, col_chk2 = st.columns(2)

with col_chk1:
    st.markdown(
        """
### Key Takeaways:
1. **Prompt Engineering** is the discipline of optimizing input context to steer autoregressive token probabilities toward desired outcomes without updating weights.
2. **Prompts** are holistic input packages; **Instructions** are task directives embedded inside them.
3. **Prompting vs Code**: Code is deterministic ($f(x)=y$); Prompts are probabilistic and require evaluation frameworks.
4. **Prompting vs RAG vs Fine-Tuning**: Prompt for task logic; RAG for fresh factual truth; Fine-tune for specialized style/syntax.
"""
    )

with col_chk2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can mathematically describe in-context conditioning via conditional token probabilities.", value=True, key="chk_eval_1")
    st.checkbox("I can articulate the exact architectural difference between a prompt and an instruction.", value=True, key="chk_eval_2")
    st.checkbox("I can defend when to use pure prompting vs RAG vs Fine-Tuning in an engineering interview.", value=True, key="chk_eval_3")
    st.checkbox("I understand why prompt engineering remains critical as foundation models scale.", value=True, key="chk_eval_4")
