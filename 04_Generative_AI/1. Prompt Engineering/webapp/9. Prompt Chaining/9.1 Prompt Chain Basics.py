import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="9.1 Prompt Chain Basics | GenAI Roadmap",
    page_icon="🔗",
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
st.title("🔗 9.1 Prompt Chain Basics — Pipeline Architecture & State Passing")
st.markdown(
    """
Welcome to **Lesson 9.1: Prompt Chain Basics**.  
One of the most frequent beginner mistakes in AI development is the **Mega-Prompt Anti-Pattern**: attempting to extract entities, analyze tone, critique logic, translate languages, and format complex JSON all inside a single massive prompt. 
When a single prompt does too much, attention heads suffer from instruction interference, and failure at any step ruins the entire output.

**Prompt Chaining** adopts the classic Unix philosophy: *Build small, focused tools that do one thing well, and connect them together with clean data pipes*.
Below, we master the fundamentals of Prompt Chaining, data passing contracts, and modular workflow design.
"""
)

st.divider()

# ==============================================================================
# Section 1: Monolithic Single Prompts vs Modular Prompt Chains
# ==============================================================================
st.header("1. Monolithic Single Prompts vs Modular Chains", divider="rainbow")
st.markdown("Comparing the monolithic mega-prompt anti-pattern against chained pipelines:")

comp_chain_df = pd.DataFrame({
    "Dimension": [
        "Architectural Pattern",
        "Cognitive Load on Model",
        "Debugging & Observability",
        "Error Blast Radius",
        "Model Specialization",
        "Cost & Cache Efficiency"
    ],
    "Monolithic Mega-Prompt (Anti-Pattern)": [
        "1 massive prompt executing 5 distinct operational tasks",
        "High attention dilution; model forgets intermediate rules",
        "Opaque black-box; impossible to pinpoint which subtask failed",
        "Total failure: A single format bug ruins the entire run",
        "Must use expensive top-tier model (e.g. GPT-4o) for everything",
        "Zero prefix caching; massive token payload on every retry"
    ],
    "Modular Prompt Chain (Production Standard)": [
        "Sequential pipeline of atomic, single-responsibility prompts",
        "Laser-focused attention; 100% adherence to single task contract",
        "Transparent: Every intermediate JSON output is inspectable",
        "Isolated: Failed stage 3 can be retried without re-running stages 1 & 2",
        "Cost routing: Cheap mini-model for extraction; frontier model for reasoning",
        "High caching reuse; intermediate steps cache independently"
    ]
})
st.dataframe(comp_chain_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: The 5 Pillars of Prompt Chain Architecture
# ==============================================================================
st.header("2. The 5 Pillars of Prompt Chain Architecture", divider="rainbow")
st.markdown("Chaining breaks complex applications into manageable, typed pipeline stages:")

pillars_chain_df = pd.DataFrame({
    "Pillar": [
        "1. Single Responsibility Principle",
        "2. Typed Intermediate Contracts",
        "3. Sequential State Passing",
        "4. Isolated Failure Boundaries",
        "5. Heterogeneous Model Routing"
    ],
    "Engineering Standard": [
        "Each prompt executes exactly one task: Extract, Classify, Draft, or Validate",
        "All data passed between prompts must validate against a strict Pydantic model",
        "The output of Prompt A becomes a clean runtime variable in Prompt B's template",
        "If Step 2 fails schema validation, retry Step 2 only (never restart the whole chain)",
        "Use fast 8B models for Stage 1 (extraction), then switch to 70B+ for Stage 2 (synthesis)"
    ]
})
st.dataframe(pillars_chain_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Production Python Implementation Blueprint
# ==============================================================================
st.header("3. Production Python Implementation Blueprint", divider="rainbow")
st.markdown("Here is how production systems build a 3-stage chain using Pydantic validation between links:")

st.code(
    '''import openai
from pydantic import BaseModel, Field

client = openai.OpenAI()

# -----------------------------------------------------------------------------
# Intermediate Typed Schemas
# -----------------------------------------------------------------------------
class ExtractedEntities(BaseModel):
    customer_name: str
    order_id: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    reported_defects: list[str]

class TriageAction(BaseModel):
    priority_level: str
    compensation_usd: float
    recommended_action: str

# -----------------------------------------------------------------------------
# Stage 1: Extraction Prompt (Fast Mini-Model)
# -----------------------------------------------------------------------------
def stage_1_extract(raw_email: str) -> ExtractedEntities:
    prompt = f"""Extract customer details from the email into JSON.
EMAIL:
{raw_email}"""
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=ExtractedEntities
    )
    return resp.choices[0].message.parsed

# -----------------------------------------------------------------------------
# Stage 2: Triage Assessment (Business Policy Logic)
# -----------------------------------------------------------------------------
def stage_2_triage(entities: ExtractedEntities) -> TriageAction:
    prompt = f"""Determine triage action based on customer status:
Customer: {entities.customer_name}
Sentiment: {entities.sentiment_score}
Defects: {entities.reported_defects}

POLICY:
- If sentiment < -0.5 and defects exist: P0 escalation with $50 refund credit.
- Otherwise: P2 standard replacement."""
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=TriageAction
    )
    return resp.choices[0].message.parsed

# -----------------------------------------------------------------------------
# Stage 3: Customer Email Drafting (Frontier Persona Model)
# -----------------------------------------------------------------------------
def stage_3_draft_reply(entities: ExtractedEntities, triage: TriageAction) -> str:
    prompt = f"""ROLE: Head of Customer Experience.
Draft a warm, apologetic resolution email to {entities.customer_name}.
State that their compensation of ${triage.compensation_usd:.2f} has been processed.
Defects acknowledged: {', '.join(entities.reported_defects)}."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# -----------------------------------------------------------------------------
# Executing the Clean Pipeline
# -----------------------------------------------------------------------------
def run_support_chain(raw_email: str) -> str:
    entities = stage_1_extract(raw_email)       # Link 1
    triage = stage_2_triage(entities)           # Link 2
    final_email = stage_3_draft_reply(entities, triage) # Link 3
    return final_email''',
    language="python"
)

st.divider()

# ==============================================================================
# Section 4: Interactive Workbench: The 3-Stage Pipeline Runner
# ==============================================================================
st.header("4. Interactive Workbench: The 3-Stage Document Chain", divider="rainbow")
st.markdown("Inspect how data flows through each link of an automated customer incident chain:")

raw_email_input = st.text_area(
    "Raw Unstructured Inbound Email (Input to Stage 1):",
    "Hi team, I am Sarah Connor. Order #ORD-9921 arrived completely crushed and missing the charging cable! I am furious and need an immediate refund today.",
    height=80
)

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    with st.container(border=True):
        st.markdown("#### 🔗 Stage 1: Extraction")
        st.caption("Model: Fast Mini-LLM")
        stage_1_out = {
            "customer_name": "Sarah Connor",
            "order_id": "ORD-9921",
            "sentiment_score": -0.85,
            "reported_defects": ["Crushed packaging", "Missing charging cable"]
        }
        st.json(stage_1_out)
        st.success("Passed Schema Contract ✅")

with col_s2:
    with st.container(border=True):
        st.markdown("#### 🔗 Stage 2: Policy Triage")
        st.caption("Model: Business Logic Engine")
        stage_2_out = {
            "priority_level": "P0_ESCALATION",
            "compensation_usd": 50.00,
            "recommended_action": "EXPEDITE_REPLACEMENT"
        }
        st.json(stage_2_out)
        st.success("Policy Rules Evaluated ✅")

with col_s3:
    with st.container(border=True):
        st.markdown("#### 🔗 Stage 3: Email Drafting")
        st.caption("Model: Frontier Creative Model")
        st.markdown(
            f"""**Drafted Resolution:**  
*Dear {stage_1_out['customer_name']},*  
*We sincerely apologize for the condition of your order #{stage_1_out['order_id']}. We have immediately credited ${stage_2_out['compensation_usd']:.2f} to your account and dispatched a replacement charging cable via express courier.*"""
        )
        st.success("Final Deliverable Ready ✅")

st.divider()

# ==============================================================================
# Section 5: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("5. Chained Pipeline Topology & Error Boundaries", divider="rainbow")

col_cp1, col_cp2 = st.columns([1, 1])

with col_cp1:
    st.markdown("#### The Isolated Pipeline Architecture")
    dot_chain = """
    digraph PromptChain {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="Raw Unstructured Input", fillcolor="#334155", fontcolor="#ffffff"];
        P1 [label="Prompt 1: Information Extractor\\n(gpt-4o-mini, $0.15/1M tokens)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Schema1 [label="Pydantic Validation Link 1\\n(Typed JSON Object)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        P2 [label="Prompt 2: Decision Classifier\\n(Evaluates business SOP rules)", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        Schema2 [label="Pydantic Validation Link 2\\n(Action Directive)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        P3 [label="Prompt 3: Content Synthesizer\\n(gpt-4o, High-Elo Persona)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Output [label="Final Customer Deliverable", fillcolor="#047857", fontcolor="#ffffff"];

        Input -> P1 -> Schema1 -> P2 -> Schema2 -> P3 -> Output;
    }
    """
    st.graphviz_chart(dot_chain, use_container_width=False)

with col_cp2:
    st.markdown("#### The 3 Golden Rules of Prompt Chaining")
    st.markdown(
        """
1. **Never pass raw conversation strings**: Always serialize intermediate state into typed JSON dictionaries.
2. **Implement Retry Gating**: If Link 2 fails, retry Link 2 up to 3 times before raising an alert; never restart Link 1!
3. **Optimize Cost via Model Tiering**: Use lightweight mini models for extraction and classification; reserve frontier models strictly for final synthesis.
"""
    )

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

pc_tab1, pc_tab2, pc_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with pc_tab1:
    with st.expander("Q1: What is the primary advantage of Prompt Chaining over a single monolithic prompt?", expanded=True):
        st.markdown(
            """
1. **Cognitive Focus & Accuracy**: Forcing an LLM to do 5 things at once degrades performance across all 5 tasks due to attention contention. Breaking tasks into atomic prompts boosts individual task accuracy to >98%.
2. **Observability & Debuggability**: If a monolithic prompt fails, you cannot diagnose which instruction caused the error. In a chain, each intermediate JSON payload is logged and testable.
3. **Cost Optimization**: You can route 80% of pipeline stages (extraction, classification) to ultra-cheap, low-latency models (`gpt-4o-mini`, `gemini-1.5-flash`), reserving expensive frontier models only for final prose synthesis.
"""
        )

    with pc_tab2:
        with st.expander("Trap 1: 'Does Prompt Chaining always increase overall latency compared to a single prompt?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, calling an API 3 times sequentially is always slower than calling it once."*
- **Winning Response**: *"Not necessarily. A monolithic prompt with 5 tasks requires generating a massive, slow completion with high Time-to-First-Token (TTFT) and high token generation latency. More importantly, when a monolithic prompt fails, you must re-run the entire expensive prompt. In a chain, fast mini-models complete steps 1 and 2 in under 200ms each, and independent links can be run in parallel (fan-out). If a step fails, you retry only that 50-token step, resulting in lower total P99 latency and lower compute costs."*
"""
            )

    with pc_tab3:
        c_pc1, c_pc2, c_pc3 = st.columns(3)
        with c_pc1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Prompt Chaining**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Decomposing a complex task into sequential, atomic prompt stages where each output feeds the next prompt.")

        with c_pc2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Model Routing in Chains**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Assigning cheap mini-models to extraction stages and frontier models strictly to final synthesis.")

        with c_pc3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Typed Link Contracts**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using Pydantic schemas to validate data between chain stages to isolate failure boundaries.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 9.1 Summary & Self-Assessment", divider="rainbow")

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    st.markdown(
        """
### Key Takeaways:
1. **Unix Philosophy for AI**: Small, focused prompts connected by typed JSON data pipes.
2. **Stop the Mega-Prompt**: Decompose monolithic prompts to eliminate attention dilution.
3. **Isolate Failures**: Retry individual failed links without restarting the entire pipeline.
4. **Heterogeneous Model Routing**: Save 70%+ on API bills by pairing mini-models with frontier models.
"""
    )

with col_sc2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain the architectural disadvantages of monolithic mega-prompts.", value=True, key="chk_9_1_1")
    st.checkbox("I know how to implement a 3-stage prompt chain with Pydantic validation between links.", value=True, key="chk_9_1_2")
    st.checkbox("I understand how model routing within chains reduces cloud inference costs.", value=True, key="chk_9_1_3")
    st.checkbox("I can design isolated retry boundaries for intermediate chain failures.", value=True, key="chk_9_1_4")
