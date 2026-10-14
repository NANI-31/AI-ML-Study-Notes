import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="8.2 Reasoning Techniques | GenAI Roadmap",
    page_icon="⚙️",
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
st.title("⚙️ 8.2 Reasoning Techniques — Decomposition, Verification & Reflection")
st.markdown(
    """
Welcome to **Lesson 8.2: Reasoning Techniques**.  
While simple Chain-of-Thought prompting ("Think step by step") provides a solid starting point, complex enterprise challenges require **structured cognitive workflows**. 
When dealing with multi-constraint scheduling, compliance audits, or high-stakes financial validation, models must actively **decompose problems, plan subtasks, verify intermediate assumptions, and critique their own drafts**.

Below, we master the **6 Foundational Reasoning Techniques**: Step-by-step induction, Problem decomposition, Subtask planning, Self-verification, Critique and revision, and production reasoning protocols.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Core Reasoning Techniques Matrix
# ==============================================================================
st.header("1. The 6 Core Reasoning Techniques Matrix", divider="rainbow")
st.markdown("Each reasoning technique addresses a distinct cognitive failure mode in Transformer models:")

reasoning_matrix_df = pd.DataFrame({
    "Technique": [
        "1. Step-by-Step Prompting",
        "2. Problem Decomposition",
        "3. Subtask Planning",
        "4. Self-Verification",
        "5. Critique & Revision (Reflection)",
        "6. Private Scratchpad Isolation"
    ],
    "Operational Trigger": [
        "Single trigger phrase: 'Let's think step by step' (Zero-Shot CoT)",
        "Dividing a macro goal into independent, sequentially solved micro-questions",
        "Emitting an explicit plan DAG (Directed Acyclic Graph) before execution",
        "Double-checking math and constraints against raw inputs before finishing",
        "Two-phase loop: Draft initial solution -> Critique flaws -> Emit final revision",
        "Encapsulating thoughts in `<scratchpad>` to prevent JSON parse corruption"
    ],
    "Cognitive Benefit": [
        "Unlocks autoregressive intermediate token compute with zero manual prompt design",
        "Prevents cognitive overload and reduces working memory context contention",
        "Allows dependency tracking across complex multi-step workflows",
        "Catches calculation slips and hallucinated assumptions before outputting",
        "Drastically improves code quality, legal drafting, and logic rigor",
        "Maintains clean, machine-readable output schemas for downstream APIs"
    ]
})
st.dataframe(reasoning_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1. Step-by-Step
with st.container(border=True):
    st.subheader("1. Step-by-Step Prompting (Zero-Shot CoT)")
    st.markdown(
        """
Discovered by Kojima et al. (2022), simply appending the magic phrase **`"Let's think step by step"`** triggers a Transformer model to emit an internal deduction trail.
"""
    )
    st.code(
        '''prompt = """
Question:
A server rack has 8 compute blades. Each blade has 2 CPUs with 16 cores each.
If 4 blades are running at 50% capacity and the other 4 are running at 75% capacity,
how many active cores are currently utilized across the rack?

Let's think step by step:
"""''',
        language="text"
    )

# 2 & 3. Decomposition & Subtask Planning
with st.container(border=True):
    st.subheader("2 & 3. Problem Decomposition & Subtask Planning")
    st.markdown(
        """
Instructing the model to break a monolithic request into an explicit **Numbered Execution Plan** before executing any single action.
"""
    )
    st.code(
        '''prompt = """
TASK: Refactor a legacy Flask authentication endpoint to FastAPI with JWT.

INSTRUCTION:
Before writing any code, output an explicit 4-step execution plan:
1. Identify all legacy dependencies to remove.
2. Define the new Pydantic request/response schemas.
3. Outline the OAuth2 password bearer token dependency.
4. Draft the new async route handler.

Then, execute each step sequentially under explicit markdown headings.
"""''',
        language="text"
    )

# 4 & 5. Self-Verification & Critique (The Reflection Pattern)
with st.container(border=True):
    st.subheader("4 & 5. Self-Verification & The Reflection Pattern")
    st.markdown(
        """
The **Reflection Pattern** commands the model to critique its own intermediate response against hard constraints before producing the final output.
"""
    )
    st.code(
        '''prompt = """
Write a high-performance Python function to deduplicate a list of 1,000,000 dictionaries by 'id'.

EXECUTION PROTOCOL:
PHASE 1: Generate your initial candidate solution.
PHASE 2 (SELF-VERIFICATION):
- Critique time complexity (Is it O(N) or O(N^2)?).
- Critique memory overhead (Does it duplicate the entire list in memory?).
- Check if dictionary insertion order is preserved.
PHASE 3 (FINAL REVISION):
- Emit the refined, production-grade function based strictly on your critique findings.
"""''',
        language="python"
    )

# 6. Private Scratchpad Isolation
with st.container(border=True):
    st.subheader("6. Production Scratchpad Isolation")
    st.markdown(
        """
In production, your API caller expects clean JSON. 
Instruct the model to do its reasoning inside `<reasoning>` tags, then emit raw JSON at the end. 
Your backend regex parses out the JSON and discards the reasoning tokens!
"""
    )
    st.code(
        '''prompt = """
You are an automated logistics scheduler.
Calculate delivery fees based on weight, distance, and toll roads.

FIRST: Perform all step-by-step arithmetic inside <scratchpad>...</scratchpad> tags.
SECOND: Output ONLY the final result matching this JSON schema:
{
  "base_rate_usd": float,
  "distance_surcharge_usd": float,
  "toll_fee_usd": float,
  "total_fee_usd": float
}
"""''',
        language="text"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Reflection & Critique Workbench
# ==============================================================================
st.header("3. Interactive Workbench: The Multi-Phase Reflection Loop", divider="rainbow")
st.markdown("Watch how the **Reflection Pattern** self-corrects a subtle concurrency bug in software engineering:")

selected_phase = st.radio(
    "Select Workflow Phase:",
    ["Phase 1: Initial Draft", "Phase 2: Self-Critique & Verification", "Phase 3: Refined Production Output"],
    index=1,
    horizontal=True
)

if "Phase 1" in selected_phase:
    st.warning("⚠️ Phase 1: Candidate Draft (Contains subtle race condition)")
    st.code(
        '''# Naive Draft generated by model:
class Counter:
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1  # Not thread-safe!''',
        language="python"
    )
    st.markdown("Candidate draft looks clean at a glance, but fails in multi-threaded production environments.")

elif "Phase 2" in selected_phase:
    st.info("🔍 Phase 2: Self-Critique & Verification Audit")
    st.markdown(
        """
**Internal Critique Generated by Model:**
1. *Thread Safety Audit*: In Python (CPython GIL notwithstanding), `self.value += 1` compiles to multiple bytecode instructions (`LOAD_FAST`, `BINARY_ADD`, `STORE_FAST`). Under high concurrent threading, race conditions WILL cause dropped counts.
2. *Remediation Strategy*: Must introduce `threading.Lock()` or an atomic counter primitive.
3. *Context Management*: Wrap in `with self._lock:` to ensure deterministic lock release on exceptions.
"""
    )

elif "Phase 3" in selected_phase:
    st.success("✅ Phase 3: Refined Production Implementation")
    st.code(
        '''import threading

class ThreadSafeCounter:
    """Production-grade thread-safe counter with mutex lock synchronization."""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        with self._lock:
            self._value += delta
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value''',
        language="python"
    )
    st.markdown("🎯 Self-reflection successfully transformed a buggy prototype into hardened production code!")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. The Reflection & Verification Cycle Architecture", divider="rainbow")

col_rf1, col_rf2 = st.columns([1, 1])

with col_rf1:
    st.markdown("#### The Autonomous Reflection Loop")
    dot_ref = """
    digraph ReflectionLoop {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="Input Task Query", fillcolor="#334155", fontcolor="#ffffff"];
        Draft [label="1. Draft Generator\\n(Initial Solution)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Critique [label="2. Self-Critique Head\\n(Check constraints, math, safety)", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        PassGate [label="Flaws or Hallucinations\\nDetected?", shape=diamond, fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];
        Revise [label="3. Revision Engine\\n(Repair identified defects)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Output [label="Final Validated Output Contract", fillcolor="#047857", fontcolor="#ffffff"];

        Input -> Draft -> Critique -> PassGate;
        PassGate -> Revise [label="Yes"];
        Revise -> Output;
        PassGate -> Output [label="No (Flawless)"];
    }
    """
    st.graphviz_chart(dot_ref, use_container_width=False)

with col_rf2:
    st.markdown("#### Reflection Prompt Template Pattern")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ [PROMPT PROTOCOL]                                      │
│  1. Generate Initial Draft                             │
│  2. Execute Self-Audit:                                │
│     - Check: Did I violate any constraints?            │
│     - Check: Is my arithmetic verified?                │
│     - Check: Did I hallucinate unstated facts?         │
│  3. Emit Final Corrected Revision                      │
└────────────────────────────────────────────────────────┘""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

rt_tab1, rt_tab2, rt_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with rt_tab1:
    with st.expander("Q1: What is the mechanism behind Zero-Shot Chain-of-Thought ('Let's think step by step')?", expanded=True):
        st.markdown(
            """
During instruction pre-training and RLHF, model datasets contain massive amounts of high-quality mathematical proofs, code tutorials, and reasoning solutions that naturally commence with phrases like *"Let's think step by step"* or *"Step 1:"*. 
When this phrase is present in the prompt prefix, it acts as a strong attention trigger that activates the model's sequential reasoning pathways, conditioning the model to emit a step-by-step explanatory sequence rather than jumping to a low-confidence completion token.
"""
        )

    with rt_tab2:
        with st.expander("Trap 1: 'Does self-verification always catch 100% of errors?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, if you ask the model to check its work, it will catch its mistakes."*
- **Winning Response**: *"No. If the model made an error due to a fundamental lack of parametric knowledge (e.g. not knowing an obscure legal statute), its self-critique pass will often double down on the error with false confidence. Self-verification is highly effective for catching logic oversights, constraint omissions, and multi-step math slips where the model has the capability to recognize the error upon re-reading. For true factual verification, pair reflection with external tools (RAG, search, Python REPL)."*
"""
            )

    with rt_tab3:
        c_rt1, c_rt2, c_rt3 = st.columns(3)
        with c_rt1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Zero-Shot CoT Trigger**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The universal trigger phrase 'Let's think step by step' that unlocks intermediate reasoning compute.")

        with c_rt2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**The Reflection Pattern**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Prompt architecture where the model drafts a solution, critiques it for flaws, and revisions it before returning.")

        with c_rt3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Private Scratchpad**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Isolating thought tokens inside `<scratchpad>` tags so backend systems can strip them from JSON API outputs.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 8.2 Summary & Self-Assessment", divider="rainbow")

col_rst1, col_rst2 = st.columns(2)
with col_rst1:
    st.markdown(
        """
### Key Takeaways:
1. **Zero-Shot CoT**: "Let's think step by step" activates latent deduction pathways.
2. **Decomposition**: Divide complex monolithic tasks into ordered sub-plans.
3. **The Reflection Loop**: Draft -> Critique -> Revise dramatically improves code and logic.
4. **Scratchpad Cleanliness**: Keep internal reasoning separate from clean JSON outputs.
"""
    )

with col_rst2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can implement Zero-Shot CoT using standard step-by-step trigger phrases.", value=True, key="chk_8_2_1")
    st.checkbox("I know how to structure an automated Reflection (Draft -> Critique -> Revise) prompt.", value=True, key="chk_8_2_2")
    st.checkbox("I understand the limits of self-verification when parametric knowledge is absent.", value=True, key="chk_8_2_3")
    st.checkbox("I can implement private scratchpad tags to protect downstream JSON parsers.", value=True, key="chk_8_2_4")
