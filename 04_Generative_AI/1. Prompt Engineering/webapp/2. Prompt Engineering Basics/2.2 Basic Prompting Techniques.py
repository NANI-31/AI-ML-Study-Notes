import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="2.2 Basic Prompting Techniques | GenAI Roadmap",
    page_icon="🛠️",
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
st.title("🛠️ 2.2 Basic Prompting Techniques — The Core Seven Archetypes")
st.markdown(
    """
Welcome to **Lesson 2.2: Basic Prompting Techniques**.  
In software engineering, before adopting advanced frameworks or microservices, developers master foundational control flow: loops, conditionals, and functions. 
Similarly, before deploying advanced strategies like Chain-of-Thought or Autonomous Agent loops, an AI engineer must master the **7 foundational prompting archetypes**.

Each technique conditions model probability distributions differently. Below, we dissect all 7 techniques with **production implementations, comparative code examples, failure modes, and an interactive side-by-side workbench**.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 7 Core Prompting Archetypes Overview
# ==============================================================================
st.header("1. Taxonomy of Foundational Prompting Archetypes", divider="rainbow")
st.markdown(
    """
Different business tasks demand different steering mechanisms. 
A naive prompt mixes queries, roles, and formats haphazardly, resulting in unpredictable tokens. 
Engineered prompts deliberately select from the **Seven Core Archetypes**:
"""
)

overview_data = {
    "Archetype": [
        "1. Direct Prompting",
        "2. Instruction Prompting",
        "3. Question-Based Prompting",
        "4. Context-Based Prompting",
        "5. Role-Based Prompting",
        "6. Constraint-Based Prompting",
        "7. Output-Format Prompting"
    ],
    "Primary Mechanism": [
        "Raw task query without preamble or decorative fluff",
        "Imperative command verbs specifying explicit actions",
        "Interrogative framing prompting explanatory or deductive reasoning",
        "Enclosing reference corpora to ground knowledge retrieval",
        "Semantic persona framing to condition tone, jargon, and viewpoint",
        "Hard negative/positive boundaries and deterministic fallback gates",
        "Strict syntactic schemas (JSON, Markdown, YAML, CSV contracts)"
    ],
    "Best Suited For": [
        "Simple zero-shot tasks, classification, rapid prototyping",
        "Multi-step algorithmic workflows, data transformation",
        "Customer support inquiries, tutoring, discovery workflows",
        "RAG pipelines, contract review, audit logs, compliance",
        "Brand voice calibration, domain expertise simulation",
        "Security guardrails, hallucination prevention, safety filtering",
        "API integrations, downstream database ingestion, microservices"
    ],
    "Risk Level": [
        "High variance on edge cases",
        "Moderate (requires clean ordering)",
        "Open to verbose conversational drift",
        "Low (strong ground truth anchor)",
        "Prone to sycophancy or superficial fluff",
        "Very low hallucination rate",
        "Schema parse errors if unconstrained"
    ]
}
st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Deep-Dive Into the 7 Techniques with Concrete Code
# ==============================================================================
st.header("2. Detailed Technical Breakdown of Each Archetype", divider="rainbow")

# Technique 1
with st.container(border=True):
    st.subheader("1. Direct Prompting")
    st.markdown(
        """
**Direct Prompting** feeds a plain statement or request directly to the model without framing, roles, or constraints.  
- **Attention Dynamics**: The model relies entirely on its pre-trained parametric knowledge and alignment priors to infer intent.
- **When to use**: Straightforward entity lookups, basic summarization, or low-stakes creative drafting.
- **Limitation**: Highly sensitive to model version updates; offers zero structural guarantees.
"""
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Naive / Casual Prompt:**")
        st.code("What are the key benefits of Docker containers?", language="text")
    with col_d2:
        st.markdown("**Production Python Payload:**")
        st.code(
            '''import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain container isolation in Docker in 3 sentences."}
    ],
    temperature=0.2
)
print(response.choices[0].message.content)''',
            language="python"
        )

# Technique 2
with st.container(border=True):
    st.subheader("2. Instruction Prompting")
    st.markdown(
        """
**Instruction Prompting** uses explicit imperative action verbs (*Extract*, *Categorize*, *Translate*, *Deduplicate*) rather than passive descriptions.
- **Attention Dynamics**: Imperative action verbs right at the beginning of the prompt establish high attention weights over subsequent tokens.
- **Rule of Thumb**: Start sentences with action verbs and break complex workflows into numbered operational stages.
"""
    )
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("**Instruction Pattern:**")
        st.code(
            '''Execute the following 3 steps sequentially:
1. Parse the provided incident report for root causes.
2. Filter out transient network glitches.
3. Output the primary software bug responsible.''',
            language="text"
        )
    with col_i2:
        st.markdown("**Production Python Payload:**")
        st.code(
            '''prompt = """
INSTRUCTION:
Classify the sentiment of the customer feedback into exactly one of: [POSITIVE, NEUTRAL, NEGATIVE].
Do not output punctuation, markdown, or explanation.

INPUT:
"The checkout process was seamless, but shipping was 2 days late."

SENTIMENT:"""
''',
            language="python"
        )

# Technique 3
with st.container(border=True):
    st.subheader("3. Question-Based Prompting")
    st.markdown(
        """
**Question-Based Prompting** poses interrogative queries (*Why*, *How*, *Under what circumstances*).  
- **Attention Dynamics**: Interrogatives trigger the model's exploratory retrieval pathways, eliciting reasoning and justification tokens.
- **Critical Risk**: Avoid *leading questions* (e.g. *"Isn't Python faster than Go for microservices?"*) as models exhibit **sycophancy**—agreeing with user presuppositions even when false.
"""
    )
    st.code(
        '''# Balanced Socratic Question Framing
prompt = """
Under what specific concurrency and memory constraints would a distributed key-value store 
choose an LSM-Tree over a B+ Tree index architecture? 
Analyze both read amplification and write amplification tradeoffs.
"""''',
        language="python"
    )

# Technique 4
with st.container(border=True):
    st.subheader("4. Context-Based Prompting (Grounding)")
    st.markdown(
        """
**Context-Based Prompting** supplies an explicit external corpus (e.g., knowledge base article, database record, API response) and commands the model to answer **exclusively** using that text.  
- **Attention Dynamics**: Forces the model's cross-attention mechanisms to bind answer tokens directly to the provided context spans.
- **Hallucination Defense**: Context-based prompting is the foundational building block of **Retrieval-Augmented Generation (RAG)**.
"""
    )
    st.code(
        '''prompt = """
You are an enterprise compliance auditor.
Answer the user question using ONLY the legal context enclosed within <context> tags below.
If the answer cannot be determined strictly from the context, respond with "INSUFFICIENT_CONTEXT".

<context>
Article 4.2: Data retention for EU customer records is strictly capped at 24 months post-account closure.
Archival backups older than 24 months must undergo automated cryptographic erasure within 7 business days.
</context>

QUESTION: Can backups of EU customer records be retained for 3 years if encrypted?
ANSWER:"""''',
        language="python"
    )

# Technique 5
with st.container(border=True):
    st.subheader("5. Role-Based Prompting (Personas)")
    st.markdown(
        """
**Role-Based Prompting** instructs the model to adopt a specific identity or domain authority (*"Act as a Principal Staff Site Reliability Engineer"*).  
- **Attention Dynamics**: Conditions the prior probability distribution toward domain-specific technical jargon, formal standards, and analytical rigor.
- **Engineering Reality**: Roles **calibrate style and perspective**; they do **not** magically inject factual knowledge the model does not possess.
"""
    )
    st.code(
        '''system_message = (
    "You are a Staff Security Architect conducting a strict threat modeling assessment. "
    "Evaluate all software designs according to the STRIDE framework. "
    "Be adversarial, direct, and highlight unmitigated attack vectors with severity ratings (P0-P3)."
)''',
        language="python"
    )

# Technique 6
with st.container(border=True):
    st.subheader("6. Constraint-Based Prompting (Guardrails & Boundaries)")
    st.markdown(
        """
**Constraint-Based Prompting** places strict guardrails around what the model is permitted and forbidden to generate.  
- **Positive vs. Negative Framing**: Positive constraints (*"Write exactly 3 bullet points, each under 15 words"*) outperform negative constraints (*"Do not write too much"*).
- **Fallback Mechanisms**: Always provide a deterministic exit token for out-of-distribution queries to prevent hallucinations.
"""
    )
    st.code(
        '''prompt = """
HARD CONSTRAINTS:
1. Output length must be between 50 and 80 words.
2. Must NOT include passive voice.
3. Must NOT reference competitor brand names.
4. If the product SKU is invalid, return exactly: {"error": "INVALID_SKU"}.
"""''',
        language="python"
    )

# Technique 7
with st.container(border=True):
    st.subheader("7. Output-Format Prompting (Syntactic Contracts)")
    st.markdown(
        """
**Output-Format Prompting** enforces a rigid syntax structure such as JSON, CSV, YAML, or Markdown tables.  
- **System Integration**: Essential when the LLM output is fed directly into downstream API endpoints, parsers, or SQL databases.
- **Best Practice**: Combine schema definitions with few-shot format examples or use modern Function Calling / Structured Outputs schemas.
"""
    )
    st.code(
        '''prompt = """
Extract customer details from the email and format strictly as a valid JSON object matching this schema:
{
  "customer_name": string,
  "account_id": integer,
  "urgent_priority": boolean,
  "detected_issues": list[string]
}

Do NOT wrap in markdown backticks or include introductory text.
Return ONLY raw parseable JSON.
"""''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Comparative Workbench
# ==============================================================================
st.header("3. Interactive Workbench: The Same Business Problem Across All 7 Techniques", divider="rainbow")
st.markdown(
    """
Witness how the **exact same real-world incident** (a high-priority checkout outage) is handled across each of the 7 archetypes.  
Select an archetype below to inspect its prompt structure, attention strategy, and expected output contract:
"""
)

technique_choice = st.selectbox(
    "Select Prompting Archetype to inspect:",
    [
        "1. Direct Prompting",
        "2. Instruction Prompting",
        "3. Question-Based Prompting",
        "4. Context-Based Prompting",
        "5. Role-Based Prompting",
        "6. Constraint-Based Prompting",
        "7. Output-Format Prompting"
    ],
    index=1
)

sample_incident = """Incident ID: INC-8821
Component: Payment Gateway / Stripe Webhook Dispatcher
Symptom: 504 Gateway Timeout during checkout spikes at 14:02 UTC.
Logs: Database connection pool exhausted (max_connections=100 reached). Idle connections holding locks for >30s.
Impact: 420 failed transactions, estimated revenue loss $18,400."""

if technique_choice == "1. Direct Prompting":
    st.info("Direct Prompting: Raw request without guidance.")
    st.code(f"Summarize what happened in this incident:\n\n{sample_incident}", language="text")
    st.markdown("**Expected Output Characteristics:** General narrative paragraph; varying length; might focus on revenue rather than root cause.")

elif technique_choice == "2. Instruction Prompting":
    st.info("Instruction Prompting: Numbered, imperative action directives.")
    st.code(f"""Analyze the incident report below following these exact instructions:
1. Identify the single technical root cause.
2. Calculate the average revenue loss per failed transaction.
3. State the exact configuration parameter that failed.

INCIDENT REPORT:
{sample_incident}""", language="text")
    st.markdown("**Expected Output Characteristics:** Clean 3-bullet breakdown directly addressing technical parameters and exact math.")

elif technique_choice == "3. Question-Based Prompting":
    st.info("Question-Based Prompting: Socratic inquiry prompting architectural reasoning.")
    st.code(f"""Based on this incident report:
1. Why did the database connection pool reach exhaustion despite traffic spikes being expected?
2. What architectural safeguard would prevent idle connections from holding database locks indefinitely?

INCIDENT DATA:
{sample_incident}""", language="text")
    st.markdown("**Expected Output Characteristics:** Deep architectural rationale discussing connection timeouts, pool sizing, and circuit breakers.")

elif technique_choice == "4. Context-Based Prompting":
    st.info("Context-Based Prompting: Strict grounding with hallucination fallback.")
    st.code(f"""You are a site reliability auditor.
Answer the audit query using ONLY the incident logs enclosed in <incident_context>.
Do not extrapolate, assume infrastructure not listed, or invent causes.
If the answer is not documented, respond with "DATA_NOT_IN_LOGS".

<incident_context>
{sample_incident}
</incident_context>

QUERY: What retry strategy was configured on the Stripe webhook dispatcher?
ANSWER:""", language="text")
    st.markdown("**Expected Output Characteristics:** Model outputs exactly `DATA_NOT_IN_LOGS` because retry strategy is not in the snippet—preventing hallucination!")

elif technique_choice == "5. Role-Based Prompting":
    st.info("Role-Based Prompting: Persona conditioning for audience calibration.")
    st.code(f"""Act as a Chief Technology Officer (CTO) briefing executive board members who do not have engineering backgrounds.
Explain this incident in terms of customer trust, business risk, and strategic capital allocation for remediation.
Do not use low-level database jargon like 'max_connections'.

INCIDENT:
{sample_incident}""", language="text")
    st.markdown("**Expected Output Characteristics:** Polished executive summary focusing on brand reputation, transaction impact, and preventive investment.")

elif technique_choice == "6. Constraint-Based Prompting":
    st.info("Constraint-Based Prompting: Strict boundaries and deterministic negative gates.")
    st.code(f"""Generate a Post-Mortem Action Item ticket from the incident data.

CONSTRAINTS:
- Maximum 3 actionable remediation steps.
- Each action item must start with an active engineering verb.
- Must include a specific numerical threshold or configuration change.
- Zero introductory or concluding sentences.

INCIDENT:
{sample_incident}""", language="text")
    st.markdown("**Expected Output Characteristics:** Exactly 3 crisp, imperative engineering tickets with quantitative thresholds.")

elif technique_choice == "7. Output-Format Prompting":
    st.info("Output-Format Prompting: Rigid machine-readable JSON schema.")
    st.code(f"""Parse the incident into a machine-readable JSON schema matching:
{{
  "incident_id": string,
  "affected_service": string,
  "root_cause_component": string,
  "financial_loss_usd": float,
  "failed_tx_count": integer,
  "severity": "P0" | "P1" | "P2"
}}

Return ONLY valid JSON. No markdown backticks.

INCIDENT:
{sample_incident}""", language="text")
    st.markdown("**Expected Output Characteristics:** Raw JSON payload ready for direct insertion into MongoDB or PostgreSQL via `json.loads()`.")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. Production Prompt Pipeline Synthesis", divider="rainbow")
st.markdown(
    """
In production AI systems, you never use just one archetype in isolation. 
Enterprise prompts blend these archetypes into an **engineered multi-layer assembly**:
"""
)

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("#### System Topology Diagram")
    # Fixed size Graphviz to avoid reflows
    dot_pipeline = """
    digraph PromptPipeline {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#94a3b8"];

        Role [label="5. Role / Persona\\n(System Tone & Prior)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Context [label="4. Context & Knowledge\\n(RAG Documents / Logs)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Instruction [label="2. Instruction & Task\\n(Numbered Directives)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Constraint [label="6. Hard Constraints\\n(Length, Safety, Fallback)", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        OutputFormat [label="7. Output Contract\\n(JSON Schema / Table)", fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];
        LLM [label="LLM Autoregressive Generation", fillcolor="#334155", fontcolor="#ffffff", shape=ellipse];

        Role -> LLM;
        Context -> LLM;
        Instruction -> LLM;
        Constraint -> LLM;
        OutputFormat -> LLM;
    }
    """
    st.graphviz_chart(dot_pipeline, use_container_width=False)

with col_arch2:
    st.markdown("#### Layered Token Stack (ASCII Anatomy)")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ [SYSTEM MESSAGE] Role Prompting (Identity & Authority)  │
├────────────────────────────────────────────────────────┤
│ [CONTEXT BLOCK] Context-Based Grounding (<docs>...</docs>)│
├────────────────────────────────────────────────────────┤
│ [TASK BLOCK] Instruction Prompting (Action Verbs)      │
├────────────────────────────────────────────────────────┤
│ [GUARDRAIL BLOCK] Constraint-Based Prompting (Boundaries)│
├────────────────────────────────────────────────────────┤
│ [SCHEMA BLOCK] Output-Format Prompting (JSON Contract) │
└───────────────────────────┬────────────────────────────┘
                            ▼
               [DETERMINISTIC INFERENCE]""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_tab1, qa_tab2, qa_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with qa_tab1:
    with st.expander("Q1: What is the core mechanical difference between Instruction Prompting and Direct Prompting?", expanded=True):
        st.markdown(
            """
- **Direct Prompting** relies on implicit inference from raw user input without structured steering.
- **Instruction Prompting** explicitly anchors the model's attention using imperative action verbs (*extract*, *synthesize*, *validate*) and provides structured operational sequences.
- **Production Impact**: Instruction prompting reduces prompt variance across different model checkpoints by up to 40% compared to direct prompting.
"""
        )

    with st.expander("Q2: Why do positive constraints consistently outperform negative constraints in LLMs?", expanded=False):
        st.markdown(
            """
Autoregressive language models predict each next word based on the words that came before it.
When you write *"Do NOT talk about pricing"*, the word **"pricing"** is placed directly into the prompt context, increasing attention activation around economic semantic clusters. 
Conversely, a positive constraint (*"Focus exclusively on technical architecture and system latency"*) conditions probability mass strictly onto target domain concepts without priming the forbidden concept.
"""
        )

    with st.expander("Q3: How do you guarantee reliable JSON output without utilizing provider-specific JSON mode or structured outputs?", expanded=False):
        st.markdown(
            r"""
1. **Provide a minimal valid schema example** in the prompt.
2. **Explicitly forbid markdown backticks** (`Do not wrap in ```json`).
3. **Set the first completion token prefill** (where supported, e.g. Anthropic assistant prefill with `{`).
4. **Use deterministic fallback instructions**: *"If fields are missing, populate with null"*.
5. **Implement client-side Pydantic / JSON schema validation** with a 1-shot retry loop for syntax corrections.
"""
        )

with qa_tab2:
    with st.expander("Trap 1: 'Does role prompting make a model smarter or more knowledgeable?'", expanded=True):
        st.markdown(
            """
- **Naive Answer**: *"Yes, assigning a role like 'world-class mathematician' unlocks advanced problem-solving intelligence."*
- **Winning Answer**: *"No. Pre-training weights and parametric knowledge are completely static at inference time. Role prompting functions as a semantic filter that shifts the tone, vocabulary, and stylistic conventions toward academic or technical registers. It can improve performance slightly on domain tasks by activating relevant latent concepts, but it cannot solve problems beyond the model's underlying reasoning boundaries."*
"""
        )

    with st.expander("Trap 2: 'Can output formatting prompts prevent bad or hallucinated data?'", expanded=False):
        st.markdown(
            """
- **Naive Answer**: *"Yes, asking for JSON ensures the data is clean and valid."*
- **Winning Answer**: *"No. Output formatting only guarantees syntactic structure, not semantic truth. A model can produce syntactically flawless JSON that contains complete hallucinations. Semantic accuracy requires Context-Based Prompting (grounding) or RAG; format prompting only standardizes serialization."*
"""
        )

with qa_tab3:
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Instruction vs Role**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Instruction specifies WHAT to execute; Role specifies HOW to communicate (tone/perspective).")

    with col_fc2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Sycophancy in Prompting**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The model's tendency to falsely agree with leading questions due to RLHF alignment bias.")

    with col_fc3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Deterministic Fallback Token**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A predefined token (e.g. `NOT_FOUND`) commanded when context is missing, stopping hallucinations.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 2.2 Summary & Self-Assessment", divider="rainbow")

col_sum1, col_sum2 = st.columns(2)
with col_sum1:
    st.markdown(
        """
### Key Takeaways:
1. **The 7 Core Archetypes**: Direct, Instruction, Question, Context, Role, Constraint, and Output-Format.
2. **Layered Assembly**: Enterprise prompts fuse multiple archetypes into a single coordinated pipeline.
3. **Positive Over Negative**: Positive operational boundaries condition attention far more reliably than negative prohibitions.
4. **Syntactic vs Semantic Validity**: Output formatting ensures serialization structure; context grounding ensures truth.
"""
    )

with col_sum2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can differentiate all 7 basic prompting archetypes and identify their optimal use cases.", value=True, key="chk_2_2_1")
    st.checkbox("I know why leading questions cause model sycophancy and how to formulate balanced inquiries.", value=True, key="chk_2_2_2")
    st.checkbox("I understand why positive constraints outperform negative constraints in autoregressive token prediction.", value=True, key="chk_2_2_3")
    st.checkbox("I can construct a multi-layered production prompt integrating Role, Context, Instruction, Constraint, and Schema.", value=True, key="chk_2_2_4")
