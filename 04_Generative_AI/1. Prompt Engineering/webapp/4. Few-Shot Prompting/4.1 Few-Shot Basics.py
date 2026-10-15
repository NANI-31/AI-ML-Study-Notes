import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="4.1 Few-Shot Basics | GenAI Roadmap",
    page_icon="🧬",
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
st.title("🧬 4.1 Few-Shot Basics — In-Context Learning & Exemplar Architecture")
st.markdown(
    """
Welcome to **Lesson 4.1: Few-Shot Basics**.  
While Zero-Shot prompting relies entirely on an LLM's pre-trained parametric memory, **Few-Shot Prompting** leverages one of the most powerful emergent capabilities of Transformer models: **In-Context Learning (ICL)**.

By supplying just a handful (1 to 5) of demonstrative input-output pairs inside the prompt, you steer model behavior without modifying a single weight parameter. 
Below, we dissect the mechanics of In-Context Learning, compare Zero-Shot vs One-Shot vs Few-Shot paradigms, and establish enterprise standards for exemplar engineering.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is In-Context Learning (ICL) & Few-Shot Prompting?
# ==============================================================================
st.header("1. In-Context Learning (ICL) Foundations", divider="rainbow")

col_icl1, col_icl2 = st.columns([1, 1])

with col_icl1:
    st.markdown(
        """
### What is Few-Shot Prompting?
**Few-Shot Prompting** is the practice of providing a small set of high-quality demonstrative input-output exemplars directly within the prompt context before posing the actual target task.

### How In-Context Learning Operates Internally:
- **Zero Gradient Updates**: The model weights remain 100% frozen. No backpropagation or optimizer steps occur.
- **Induction Heads**: Transformers utilize specialized two-layer attention circuits known as *Induction Heads*. These attention heads scan previous tokens in the context window for patterns like `[A] -> [B]`, and when `[A]` appears again, they copy or predict `[B]`.
- **Activation Shaping**: The exemplars act as an empirical prompt anchor, shaping intermediate latent activations toward the target output style, syntax, and reasoning trajectory.
"""
    )

with col_icl2:
    st.markdown("### The Induction Head Pattern (ASCII)")
    st.code(
        """Context Stream:
... [Input A]  ──(Pattern Match)──> [Output B]  <-- Exemplar 1
... [Input C]  ──(Pattern Match)──> [Output D]  <-- Exemplar 2
... [Target Input X] ──(Induction)─> [Predicted Y]

Transformer Attention Heads:
1. First head detects input-output delimiter tokens ("Input:" -> "Output:")
2. Second head binds the target query to the demonstration pattern
3. Autoregressively generates output matching the demonstrated format""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 2: Zero-Shot vs One-Shot vs Few-Shot Comparison
# ==============================================================================
st.header("2. Comparative Taxonomy: Zero-Shot vs One-Shot vs Few-Shot", divider="rainbow")
st.markdown("Selecting the right shot paradigm is an engineering tradeoff between token cost, latency, and output fidelity:")

shot_comparison = pd.DataFrame({
    "Paradigm": ["Zero-Shot (0 Examples)", "One-Shot (1 Example)", "Few-Shot (3–5 Examples)", "Many-Shot (20+ Examples)"],
    "Prompt Footprint": ["Minimal (20–100 tokens)", "Low (100–300 tokens)", "Moderate (300–1,200 tokens)", "Heavy (2,000–10,000 tokens)"],
    "Primary Role": [
        "Standard NLP tasks with natural language instructions",
        "The 'Rosetta Stone' for anchoring syntactic formatting & schema",
        "Disambiguating edge cases, teaching custom taxonomies & tone",
        "Complex domain adaptation, few-shot reasoning, specialized code"
    ],
    "Token Cost & TTFT": [
        "Lowest latency & minimum cost",
        "Negligible latency increase (+50ms)",
        "Moderate latency (+150ms); higher input token bill",
        "High cost; requires aggressive prefix caching"
    ],
    "Reliability on Custom Formats": [
        "Unstable (prone to markdown/chat drift)",
        "High (resolves 80% of formatting errors)",
        "Very High (>99% schema compliance)",
        "Near-perfect consistency across runs"
    ]
})
st.dataframe(shot_comparison, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Principles for Selecting and Designing Good Examples
# ==============================================================================
st.header("3. The 5 Golden Rules of Exemplar Engineering", divider="rainbow")
st.markdown(
    """
Providing bad examples is significantly worse than providing no examples at all. 
Faulty exemplars inject **spurious correlations, class imbalance, and format errors** directly into the attention stream.
Follow these 5 enterprise rules:
"""
)

col_r1, col_r2 = st.columns(2)

with col_r1:
    with st.container(border=True):
        st.subheader("Rule 1: Strict Label Balance")
        st.markdown(
            """
If demonstrating a 3-class classifier (*POSITIVE*, *NEUTRAL*, *NEGATIVE*), **never** provide 3 positive examples and only 1 negative example. 
LLMs exhibit severe **majority-class frequency bias**; they will disproportionately predict whichever label appeared most often in the context window.
- **Standard**: Always provide equal representation across all target classes (1:1:1 ratio).
"""
        )

    with st.container(border=True):
        st.subheader("Rule 2: Input Diversity & Length Variation")
        st.markdown(
            """
Do not make all your input examples 10 words long. 
Vary sentence structures, vocabularies, and length. 
If all examples are short, the model may truncate real-world long inputs or hallucinate summaries.
"""
        )

    with st.container(border=True):
        st.subheader("Rule 3: Consistent Delimiter Formatting")
        st.markdown(
            """
Keep the syntactic framing identical across all exemplars. 
If Exemplar 1 uses `Input: ... \nOutput: ...`, do not switch Exemplar 2 to `<query>...</query>`. 
Consistent syntax creates an unshakeable induction anchor for the attention heads.
"""
        )

with col_r2:
    with st.container(border=True):
        st.subheader("Rule 4: Realistic Production Domain Data")
        st.markdown(
            """
Avoid synthetic "toy" examples (*"The cat sat on the mat"*). 
Use sanitized, anonymized historical production snippets that contain real-world typos, technical acronyms, and formatting artifacts. 
This prepares the model for runtime noise.
"""
        )

    with st.container(border=True):
        st.subheader("Rule 5: Deliberate Edge-Case Inclusion")
        st.markdown(
            """
Always dedicate at least one exemplar to a negative or boundary condition. 
Show the model what to do when information is missing, ambiguous, or invalid (e.g. demonstrating the fallback token `"NOT_FOUND"`).
"""
        )

st.divider()

# ==============================================================================
# Section 4: Production Code Patterns
# ==============================================================================
st.header("4. Production Python Implementation Patterns", divider="rainbow")
st.markdown("Here is how production systems format few-shot prompts using structured XML tags and OpenAI Chat Message roles:")

st.code(
    '''import openai
from pydantic import BaseModel

client = openai.OpenAI()

# =============================================================================
# Pattern 1: Monolithic XML Few-Shot Prompt (Recommended for Completion & APIs)
# =============================================================================
few_shot_prompt = """
INSTRUCTION:
Classify customer support tickets into priority levels [P0, P1, P2].

### EXAMPLES:

<example>
<ticket>Database connection timed out for all EU users at 14:00 UTC. Revenue checkout is blocked.</ticket>
<priority>P0</priority>
<rationale>Critical service outage affecting payment flow.</rationale>
</example>

<example>
<ticket>Dark mode toggle in user profile settings resets to light mode upon browser refresh.</ticket>
<priority>P2</priority>
<rationale>Cosmetic UI bug with zero financial impact.</rationale>
</example>

<example>
<ticket>Nightly reporting batch job is running 45 minutes slower than typical baseline.</ticket>
<priority>P1</priority>
<rationale>Degraded performance with operational delay, but non-blocking.</rationale>
</example>

### TARGET TASK:
<ticket>Customer unable to update credit card billing address due to 400 Bad Request error.</ticket>
<priority>"""

# =============================================================================
# Pattern 2: Multi-Turn Message Pairing (Chat API Standard)
# =============================================================================
messages = [
    {
        "role": "system",
        "content": "You are a specialized legal clause classifier. Output strictly the clause category."
    },
    # Exemplar 1
    {"role": "user", "content": "Either party may terminate upon 30 days written notice."},
    {"role": "assistant", "content": "TERMINATION_CONVENIENCE"},
    # Exemplar 2
    {"role": "user", "content": "Neither party shall be liable for indirect or consequential damages."},
    {"role": "assistant", "content": "LIMITATION_LIABILITY"},
    # Exemplar 3 (Edge Case Fallback)
    {"role": "user", "content": "The party likes drinking sparkling water in meetings."},
    {"role": "assistant", "content": "NON_OPERATIONAL_CLAUSE"},
    # Target Request
    {"role": "user", "content": "In no event shall aggregate liability exceed total fees paid in the prior 12 months."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.0
)
print(response.choices[0].message.content)  # Outputs: LIMITATION_LIABILITY''',
    language="python"
)

st.divider()

# ==============================================================================
# Section 5: Interactive In-Context Learning Workbench
# ==============================================================================
st.header("5. Interactive Workbench: The In-Context Learning Effect", divider="rainbow")
st.markdown(
    """
Experience how adding exemplars transforms model output consistency on a non-standard task:  
**Task**: Converting non-standard company shorthand into structured incident tags.
"""
)

num_shots = st.radio(
    "Select Shot Configuration:",
    ["Zero-Shot (0 Examples)", "One-Shot (1 Example)", "Three-Shot (Balanced Exemplars)"],
    index=1,
    horizontal=True
)

target_shorthand = "db-pool-spike-p99-lax-s3-fail"

if num_shots == "Zero-Shot (0 Examples)":
    st.error("Zero-Shot Mode: Model has to guess taxonomy and casing.")
    st.code(
        f"""Parse this incident shorthand into tags:
"{target_shorthand}"
TAGS:""",
        language="text"
    )
    st.markdown("**Simulated Model Output:** `Database Spike, Port 99, Los Angeles, S3 Failure`  \n*(Notice: Inconsistent casing, guessed 'Port 99' instead of P99 percentile, unstandardized output).*")

elif num_shots == "One-Shot (1 Example)":
    st.info("One-Shot Mode: Anchors the serialization syntax and casing convention.")
    st.code(
        f"""Parse incident shorthand into standardized uppercase tags.

EXAMPLE:
Shorthand: redis-cache-miss-p95-fra
Tags: [INFRA:REDIS, METRIC:P95_LATENCY, REGION:FRA]

TARGET:
Shorthand: {target_shorthand}
Tags:""",
        language="text"
    )
    st.markdown("**Simulated Model Output:** `[INFRA:DATABASE, METRIC:P99_LATENCY, REGION:LAX, SERVICE:S3]`  \n*(Notice: Output instantly aligned to square brackets and uppercase key:value format!)*")

elif num_shots == "Three-Shot (Balanced Exemplars)":
    st.success("Three-Shot Mode: Full disambiguation of components, region codes, and severity.")
    st.code(
        f"""Parse incident shorthand into standardized structured tags.

EXAMPLE 1 (Hardware Outage):
Shorthand: ec2-node-oom-p0-iad
Tags: [SEV:P0, COMPONENT:COMPUTE, ERROR:OOM, REGION:US_EAST_IAD]

EXAMPLE 2 (Network Degradation):
Shorthand: cdn-edge-drop-p1-sin
Tags: [SEV:P1, COMPONENT:NETWORK, ERROR:PACKET_DROP, REGION:AP_SOUTH_SIN]

EXAMPLE 3 (Database Failure):
Shorthand: pgsql-deadlock-p0-fra
Tags: [SEV:P0, COMPONENT:DATABASE, ERROR:LOCK_CONTENTION, REGION:EU_CENTRAL_FRA]

TARGET:
Shorthand: {target_shorthand}
Tags:""",
        language="text"
    )
    st.markdown("**Simulated Model Output:** `[SEV:P0, COMPONENT:DATABASE, ERROR:CONNECTION_POOL_SPIKE, REGION:US_WEST_LAX, DEPENDENCY:S3]`  \n*(Notice: Perfect taxonomy alignment, accurate region decoding, and contextual inference).*")

st.divider()

# ==============================================================================
# Section 6: Architectural Attention Flowchart (Graphviz)
# ==============================================================================
st.header("6. Architectural Attention Topology", divider="rainbow")

col_top1, col_top2 = st.columns([1, 1])

with col_top1:
    st.markdown("#### In-Context Attention Induction")
    dot_icl = """
    digraph InductionHeads {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Ex1 [label="Exemplar 1\\nInput -> Output", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Ex2 [label="Exemplar 2\\nInput -> Output", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Ex3 [label="Exemplar 3 (Edge Case)\\nInput -> Fallback", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        Target [label="Target Runtime Input\\nQuery", fillcolor="#334155", fontcolor="#ffffff"];

        Induction [label="Transformer Induction Heads\\n(Cross-Attention Pattern Copying)", fillcolor="#047857", fontcolor="#ffffff"];
        Output [label="Calibrated Output Token Stream", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];

        Ex1 -> Induction;
        Ex2 -> Induction;
        Ex3 -> Induction;
        Target -> Induction;
        Induction -> Output;
    }
    """
    st.graphviz_chart(dot_icl, use_container_width=False)

with col_top2:
    st.markdown("#### Cost vs Accuracy Tradeoff Curve")
    chart_tradeoff = pd.DataFrame({
        "Shot Paradigm": ["0-Shot", "1-Shot", "3-Shot", "5-Shot", "10-Shot"],
        "Accuracy (%)": [52.0, 84.0, 96.5, 98.2, 98.6],
        "Input Tokens": [45, 180, 490, 820, 1650]
    })
    st.line_chart(chart_tradeoff.set_index("Shot Paradigm")[["Accuracy (%)", "Input Tokens"]])

st.divider()

# ==============================================================================
# Section 7: Technical Interview Preparation
# ==============================================================================
st.header("7. Technical Interview Preparation", divider="rainbow")

fs_tab1, fs_tab2, fs_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with fs_tab1:
    with st.expander("Q1: What are Transformer 'Induction Heads' and how do they explain Few-Shot learning?", expanded=True):
        st.markdown(
            """
Induction heads are two-layer attention circuits identified in Transformer mechanistical interpretability research (Elhage et al., Anthropic). 
The first head attends to the previous token, while the second head searches the earlier context for identical preceding tokens and copies what followed them. 
When few-shot exemplars provide repetitive patterns of `[Input] -> [Output]`, induction heads automatically activate, effectively implementing an in-context algorithm that predicts the corresponding `[Output]` token for the new query without any gradient weight changes.
"""
        )

    with st.expander("Q2: What is 'Recency Bias' in few-shot prompting and how do you mitigate it?", expanded=False):
        st.markdown(
            """
**Recency Bias** is the empirical phenomenon where an LLM is disproportionately likely to output the label or format of the **very last exemplar** in the prompt. 
For example, if the last few-shot example before the user query was labeled *"SPAM"*, the model has a higher probability of predicting *"SPAM"*.  
**Mitigation**:
1. Randomize or balance the order of exemplars across production requests.
2. Ensure the label distribution across exemplars is strictly balanced.
3. Separate the last exemplar from the real query with a strong structural delimiter (e.g. `### REAL QUERY:`).
"""
        )

    with fs_tab2:
        with st.expander("Trap 1: 'Can we improve few-shot accuracy indefinitely by adding 50 examples?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, more examples always give the model more data to learn from."*
- **Winning Response**: *"No. Few-shot accuracy experiences rapid logarithmic diminishing returns after 3 to 5 examples. Beyond 5 to 10 examples, the token cost and inference latency scale linearly while accuracy plateaus. Worse, exceeding optimal context depth can cause attention dilution (the 'Lost in the Middle' phenomenon). If 5 examples are insufficient to achieve target accuracy, the solution is not adding 50 examples in-context, but rather RAG (dynamic few-shot selection via vector search) or Parameter-Efficient Fine-Tuning (PEFT/LoRA)."*
"""
            )

    with fs_tab3:
        c_fs1, c_fs2, c_fs3 = st.columns(3)
        with c_fs1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**In-Context Learning (ICL)**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The ability of an LLM to adapt to new tasks at inference time using prompt exemplars with zero gradient updates.")

        with c_fs2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Majority Class Bias**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The tendency of an LLM to predict whichever class appeared most frequently in the few-shot examples.")

        with c_fs3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**One-Shot 'Rosetta Stone'**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using exactly 1 exemplar to anchor complex JSON schemas or output syntax with minimal token cost.")

st.divider()

# ==============================================================================
# Section 8: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("8. Lesson 4.1 Summary & Self-Assessment", divider="rainbow")

col_sum_a, col_sum_b = st.columns(2)
with col_sum_a:
    st.markdown(
        """
### Key Takeaways:
1. **In-Context Learning**: Operates via Transformer Induction Heads with zero gradient updates.
2. **The 1-Shot Sweet Spot**: A single example fixes 80% of formatting and syntax issues.
3. **The 3-to-5 Shot Sweet Spot**: Optimal balance between accuracy saturation and token economics.
4. **Exemplar Hygiene**: Strictly enforce label balance, input diversity, and edge-case inclusion.
"""
    )

with col_sum_b:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain how Transformer Induction Heads power in-context learning.", value=True, key="chk_4_1_1")
    st.checkbox("I know why class label imbalance in few-shot exemplars harms accuracy.", value=True, key="chk_4_1_2")
    st.checkbox("I understand when to use One-Shot for formatting vs Few-Shot for classification.", value=True, key="chk_4_1_3")
    st.checkbox("I can structure multi-turn few-shot prompts using OpenAI / Anthropic chat message schemas.", value=True, key="chk_4_1_4")
