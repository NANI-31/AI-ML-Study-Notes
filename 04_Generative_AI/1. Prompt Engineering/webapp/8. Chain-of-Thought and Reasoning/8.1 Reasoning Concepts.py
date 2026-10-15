import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="8.1 Reasoning Concepts | GenAI Roadmap",
    page_icon="🧠",
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
st.title("🧠 8.1 Reasoning Concepts — Test-Time Compute & Cognitive Scratchpads")
st.markdown(
    """
Welcome to **Lesson 8.1: Reasoning Concepts**.  
In standard zero-shot prompting, an LLM is expected to generate the final answer immediately on the very next token. 
For simple tasks like translation or sentiment classification, this works well. But for multi-step logic, complex business rules, and multi-variable arithmetic, demanding an immediate answer forces the model to jump to conclusions before performing the necessary computations.

**Reasoning Prompting** fundamentally changes this dynamic by allocating **Test-Time Compute**: allowing the model to generate intermediate reasoning tokens (a "mental scratchpad") before committing to a final conclusion.
Below, we master the foundational concepts of LLM reasoning architectures.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is Reasoning Prompting & Why Compute Matters
# ==============================================================================
st.header("1. What is Reasoning Prompting & Test-Time Compute", divider="rainbow")

col_r1, col_r2 = st.columns([1, 1])

with col_r1:
    st.markdown(
        """
### Definition
**Reasoning Prompting** is the architectural technique of structuring prompts to force an LLM to generate **explicit intermediate deduction steps** prior to emitting its final answer.

### Why Direct Answering Fails on Complex Tasks:
- **Autoregressive Mechanics**: A Transformer processes tokens sequentially. When asked *"What is the total cost?"*, the model has only the fixed computation of a single forward pass per token to compute the entire result.
- **The Scratchpad Effect**: By generating intermediate calculation words (*"First, we calculate shipping: 4 items x $5 = $20..."*), each newly generated token enters the context window and serves as input for subsequent attention heads.
- **Effective Parameter Expansion**: Generating intermediate tokens effectively increases the total sequential matrix operations available to solve the problem!
"""
    )

with col_r2:
    st.markdown("### Direct Jump vs Scratchpad Trajectory (ASCII)")
    st.code(
        """DIRECT JUMP (High Failure Rate):
[Problem Query] ───────────────────────────> [Final Answer: INCORRECT]
(Single forward pass attempted to solve all arithmetic in 1 token)

REASONING SCRATCHPAD (High Success Rate):
[Problem Query]
   │
   ▼
[Step 1: Parse variables]
   │
   ▼
[Step 2: Calculate sub-total]  <-- Each step feeds attention for next step
   │
   ▼
[Step 3: Apply discount rule]
   │
   ▼
[Final Output: VALIDATED CORRECT ANSWER]""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 2: Complex vs Simple Tasks: The Task Triage Matrix
# ==============================================================================
st.header("2. Complex vs Simple Tasks: When to Allocate Reasoning Compute", divider="rainbow")
st.markdown("Reasoning tokens consume API budget and increase latency. Always triage tasks appropriately:")

triage_df = pd.DataFrame({
    "Task Category": [
        "Simple / Surface Tasks (Direct Prompting)",
        "Moderate Tasks (Light Chain-of-Thought)",
        "Complex Multi-Step (Full Reasoning Decomposition)"
    ],
    "Characteristics": [
        "1-step lookup, sentiment, translation, grammar check, extraction",
        "Multi-factor classification, short summarization with constraints",
        "Multi-clause contract analysis, math puzzles, system architecture design, nested SQL"
    ],
    "Recommended Strategy": [
        "Direct Zero-Shot prompting with strict schema output (Temp 0.0)",
        "Light instruction: 'Briefly state the rationale before the answer'",
        "Full scratchpad prompting: 'Think step by step in <scratchpad> tags before outputting JSON'"
    ],
    "Token Overhead": [
        "0% extra tokens (fastest TTFT)",
        "20% - 40% extra tokens",
        "100% - 300% extra tokens (higher latency, maximum accuracy)"
    ],
    "Accuracy Differential": [
        "98% accuracy (Reasoning adds zero benefit)",
        "Reasoning improves accuracy from 75% -> 92%",
        "Reasoning improves accuracy from 35% -> 96%!"
    ]
})
st.dataframe(triage_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: The Error Cascade Phenomenon
# ==============================================================================
st.header("3. Multi-Step Problems & The Error Cascade Phenomenon", divider="rainbow")
st.markdown(
    """
In composite problems, each sub-calculation depends on the previous step. 
If an un-decomposed prompt makes a minor error at step 1, **all subsequent tokens compound the error** (The Error Cascade).
"""
)

col_ec1, col_ec2 = st.columns(2)

with col_ec1:
    with st.container(border=True):
        st.subheader("❌ Un-decomposed Error Cascade")
        st.markdown(
            """
**Task**: *"A store has 120 laptops. It sells 25% on Monday, receives 40 new laptops on Tuesday, and sells 1/3 of the remaining stock on Wednesday. How many are left?"*
- **Direct Output**: *"There are 85 laptops left."*
- **Why it failed**: The model attempted to mentally calculate `(120 * 0.75 + 40) * (2/3)` in a single autoregressive step, resulting in a hallucinated guess.
"""
        )

with col_ec2:
    with st.container(border=True):
        st.subheader("✅ Intermediate Reasoning Scratchpad")
        st.markdown(
            """
**Reasoning Breakdown**:
1. *Monday sales*: 25% of 120 = 30 laptops sold. Remaining: $120 - 30 = 90$.
2. *Tuesday arrivals*: $90 + 40 = 130$ laptops in inventory.
3. *Wednesday sales*: 1/3 of 130 is 43.33 laptops (cannot sell partial laptop; rounds to 43). Remaining: $130 - 43 = 87$.
- **Result**: Exactly 87 laptops left. Correct!
"""
        )

st.divider()

# ==============================================================================
# Section 4: Interactive Workbench: Direct vs Scratchpad Simulator
# ==============================================================================
st.header("4. Interactive Workbench: The Scratchpad Accuracy Boost", divider="rainbow")
st.markdown(
    """
Compare a **Direct Prompt** against a **Scratchpad-Enabled Prompt** on a multi-variable business allocation problem:  
**Problem**: *"We have 3 microservices with resource allocations: Service A uses 4 CPUs, Service B uses 8 CPUs, Service C uses 16 CPUs. We apply a 20% efficiency reduction to B and double A. What is total CPU usage?"*
"""
)

mode_choice = st.radio(
    "Choose Prompt Architecture:",
    ["1. Direct Output Prompt (Zero Reasoning Tokens)", "2. Scratchpad Decomposition Prompt (Intermediate Reasoning)"],
    index=1
)

if "Direct" in mode_choice:
    st.error("Direct Prompt: Model forced to guess immediately.")
    st.code(
        """PROMPT:
Calculate total CPU: Service A (4 CPU doubled), Service B (8 CPU - 20%), Service C (16 CPU).
Output ONLY the final number.

COMPLETION:
Total CPU: 28""",
        language="text"
    )
    st.markdown("🚨 **WRONG**: True answer is `(4 * 2) + (8 * 0.8) + 16 = 8 + 6.4 + 16 = 30.4`. Direct generation failed to execute the float math correctly!")

else:
    st.success("Scratchpad Prompt: Generates intermediate calculations step by step.")
    st.code(
        """PROMPT:
Calculate total CPU usage.
Think step by step in <scratchpad> tags before emitting the final JSON.

<scratchpad>
Step 1: Service A is doubled -> 4 * 2 = 8 CPUs.
Step 2: Service B reduced by 20% -> 8 * (1 - 0.20) = 8 * 0.80 = 6.4 CPUs.
Step 3: Service C unchanged -> 16 CPUs.
Step 4: Sum all services: 8 + 6.4 + 16 = 30.4 CPUs.
</scratchpad>

OUTPUT:
{"total_cpu": 30.4}""",
        language="text"
    )
    st.markdown("🎯 **EXACT MATCH**: Intermediate tokens enabled the model to compute sub-totals accurately!")

st.divider()

# ==============================================================================
# Section 5: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("5. The Test-Time Compute Allocation Spectrum", divider="rainbow")

col_tt1, col_tt2 = st.columns([1, 1])

with col_tt1:
    st.markdown("#### Test-Time Compute vs Problem Complexity")
    dot_comp = """
    digraph TestTimeCompute {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Query [label="Input Query Arrives", fillcolor="#334155", fontcolor="#ffffff"];
        Classifier [label="Task Complexity Router\\n(Is it multi-step?)", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        Direct [label="Direct Generation\\n(Lowest latency & cost)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Scratchpad [label="Scratchpad CoT Generation\\n(Intermediate reasoning tokens)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Output [label="Final Output Contract\\n(Validated Answer)", fillcolor="#047857", fontcolor="#ffffff"];

        Query -> Classifier;
        Classifier -> Direct [label="Single-step"];
        Classifier -> Scratchpad [label="Multi-step"];
        Direct -> Output;
        Scratchpad -> Output;
    }
    """
    st.graphviz_chart(dot_comp, use_container_width=False)

with col_tt2:
    st.markdown("#### The Modern Production Rationale")
    st.info(
        """
**Note for Production AI Systems**:  
In modern API systems, developers frequently instruct the model to write its chain of thought inside private `<scratchpad>...</scratchpad>` tags, and then parse and discard the scratchpad before returning the clean JSON payload to the user. 
This gives you all the accuracy benefits of reasoning without cluttering user interfaces!
"""
    )

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

rc_tab1, rc_tab2, rc_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with rc_tab1:
    with st.expander("Q1: Why does generating intermediate tokens improve an LLM's reasoning accuracy?", expanded=True):
        st.markdown(
            """
Transformers have a constant computational depth per token (fixed number of attention layers). 
When an LLM generates a single immediate answer token, it can only perform a fixed amount of parallel computation. 
By generating intermediate reasoning tokens, the model spreads the problem over multiple sequential forward passes. 
Crucially, each intermediate token is appended to the context window and attends back to earlier steps, effectively turning the context window into an external read-write working memory (scratchpad) that stores sub-solutions.
"""
        )

    with rc_tab2:
        with st.expander("Trap 1: 'Should we always enable Chain-of-Thought on every single prompt?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, reasoning makes the model smarter, so always use it."*
- **Winning Response**: *"No. On simple tasks (sentiment, language translation, binary classification), CoT adds unnecessary token costs (2x-5x higher bill) and increases latency by hundreds of milliseconds without improving accuracy. Furthermore, on structured API extraction tasks, verbose reasoning often leads to schema parse errors. Use task routing to deploy CoT strictly on multi-step, multi-clause, or arithmetic problems."*
"""
            )

    with rc_tab3:
        c_rc1, c_rc2, c_rc3 = st.columns(3)
        with c_rc1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Test-Time Compute**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Allocating computational FLOPs at inference time by allowing the model to generate intermediate reasoning tokens.")

        with c_rc2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**The Scratchpad Effect**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using generated tokens as working memory in context so future attention passes can condition on sub-results.")

        with c_rc3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Error Cascade**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** When an early mistake in an un-decomposed calculation ruins all subsequent downstream token probabilities.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 8.1 Summary & Self-Assessment", divider="rainbow")

col_rs1, col_rs2 = st.columns(2)
with col_rs1:
    st.markdown(
        """
### Key Takeaways:
1. **Test-Time Compute**: Reasoning tokens expand the computational depth available per problem.
2. **The Scratchpad**: Working memory in context prevents catastrophic arithmetic guesses.
3. **Task Triage**: Reserve reasoning prompts for multi-step logic; use direct prompts for lookups.
4. **Private Scratchpads**: Encapsulate reasoning in `<scratchpad>` tags to maintain clean output contracts.
"""
    )

with col_rs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the architectural reasons why direct token generation fails on multi-step problems.", value=True, key="chk_8_1_1")
    st.checkbox("I know when to triage a task toward Chain-of-Thought vs direct zero-shot prompting.", value=True, key="chk_8_1_2")
    st.checkbox("I can implement private scratchpad tags to separate reasoning tokens from clean JSON outputs.", value=True, key="chk_8_1_3")
    st.checkbox("I understand how intermediate tokens prevent error cascades in complex calculations.", value=True, key="chk_8_1_4")
