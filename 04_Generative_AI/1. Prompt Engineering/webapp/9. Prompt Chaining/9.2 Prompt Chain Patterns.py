import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="9.2 Prompt Chain Patterns | GenAI Roadmap",
    page_icon="🔀",
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
st.title("🔀 9.2 Prompt Chain Patterns — The 6 Foundational Workflows")
st.markdown(
    """
Welcome to **Lesson 9.2: Prompt Chain Patterns**.  
In software architecture, design patterns like *Factory*, *Singleton*, and *Observer* provide proven solutions to recurring engineering problems. 
In Generative AI application development, **Prompt Chain Patterns** provide proven pipeline topologies for decomposing complex multi-step objectives into reliable, production-tested workflows.

Below, we master the **6 Canonical Prompt Chain Patterns**: Generate-Review, Extract-Transform, Analyze-Summarize, Plan-Execute, Classify-Route, and Generate-Validate.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Canonical Chain Patterns Matrix
# ==============================================================================
st.header("1. The 6 Canonical Chain Patterns Matrix", divider="rainbow")
st.markdown("Comparing the 6 core prompt pipeline topologies:")

chain_patterns_df = pd.DataFrame({
    "Chain Pattern": [
        "1. Generate -> Review",
        "2. Extract -> Transform",
        "3. Analyze -> Summarize",
        "4. Plan -> Execute",
        "5. Classify -> Route",
        "6. Generate -> Validate"
    ],
    "Stage 1 (Upstream Link)": [
        "Drafter generates candidate solution or creative copy",
        "Extractor parses raw unstructured OCR/text into key entities",
        "Deep analytical prompt surfaces nuances, metrics, and risks",
        "Planner creates an explicit step-by-step dependency DAG",
        "Classifier categorizes incoming intent into a discrete enum",
        "Generator creates a structured payload (JSON / SQL / Code)"
    ],
    "Stage 2 (Downstream Link)": [
        "Critic reviews for compliance, tone, or hallucination errors",
        "Transformer normalizes fields to external API/SQL formats",
        "Summarizer compresses findings into an executive 3-bullet brief",
        "Executor carries out each planned subtask sequentially",
        "Router dispatches request to a specialized domain sub-prompt",
        "Validator (Pydantic / Linter) audits syntax; triggers retry if invalid"
    ],
    "Enterprise Use Case": [
        "Automated PR legal review, marketing copy generation",
        "Invoice OCR parsing, resume extraction into database columns",
        "Financial earnings call digestion, incident root-cause reports",
        "Autonomous code refactoring, multi-file software engineering",
        "Customer support triage bots, multi-skill voice assistants",
        "Automated database insertion, synthetic test data generation"
    ]
})
st.dataframe(chain_patterns_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1. Generate -> Review
with st.container(border=True):
    st.subheader("1. Generate -> Review Pattern (Dual-Model Quality Gate)")
    st.markdown(
        """
Separates the **creative/generative role** from the **critical/compliance role**. 
A single model struggle to be creative and strictly critical at the same moment; separating them into two prompt calls ensures rigorous quality control.
"""
    )
    st.code(
        '''# Stage 1: Drafter
draft_prompt = "Write a high-converting marketing email for our new B2B AI analytics tool."
draft_content = call_llm(draft_prompt)

# Stage 2: Compliance Reviewer
review_prompt = f"""
ROLE: Legal & Marketing Compliance Auditor.
Review the email draft below against FTC advertising guidelines.
Check:
1. Are there unsubstantiated guarantees (e.g. '100% guarantee')?
2. Is the pricing disclosure clear?
If flaws exist, rewrite the email to be compliant. Otherwise, approve it.

DRAFT:
{draft_content}"""
approved_email = call_llm(review_prompt)''',
        language="python"
    )

# 2. Extract -> Transform
with st.container(border=True):
    st.subheader("2. Extract -> Transform (AI ETL Pipeline)")
    st.markdown(
        """
Classic data engineering applied to LLMs: first extract raw data points from unstructured prose, then transform dates, currencies, and names into standardized schemas.
"""
    )
    st.code(
        '''# Stage 1: Extract entities
raw_note = "Dr. Alvarez examined John on 12/04/26. Prescribed 500mg Metformin twice daily."
entities = call_llm(f"Extract patient, doctor, date, and medication from: {raw_note}")

# Stage 2: Transform to FHIR / ISO Schema
fhir_json = call_llm(f"""
Normalize the extracted entities into a valid FHIR-compliant JSON object:
- Convert date to ISO-8601 (YYYY-MM-DD).
- Normalize dosage to standardized mg units.
DATA: {entities}""")''',
        language="python"
    )

# 5. Classify -> Route
with st.container(border=True):
    st.subheader("5. Classify -> Route (Dynamic Workflow Dispatcher)")
    st.markdown(
        """
A lightweight classifier acts as an intelligent switchboard, categorizing user intent and routing the conversation to a specialized domain sub-prompt.
"""
    )
    st.code(
        '''def triage_and_route(user_message: str):
    # Link 1: Lightweight classification
    intent = call_fast_model(f"""Classify into: [BILLING, TECH_SUPPORT, SALES_LEAD].
Input: '{user_message}'
Category:""")
    
    # Link 2: Dynamic Routing
    if intent == "BILLING":
        return call_billing_expert_prompt(user_message)
    elif intent == "TECH_SUPPORT":
        return call_sre_troubleshooter_prompt(user_message)
    else:
        return call_sales_qualifier_prompt(user_message)''',
        language="python"
    )

# 6. Generate -> Validate
with st.container(border=True):
    st.subheader("6. Generate -> Validate (Self-Healing JSON Pipeline)")
    st.markdown(
        """
Generates a structured payload, runs it through a deterministic client-side validator (Pydantic / JSON schema), and triggers a 1-shot self-correction loop if validation fails.
"""
    )
    st.code(
        '''from pydantic import BaseModel, ValidationError

class UserProfileSchema(BaseModel):
    user_id: int
    email: str
    active: bool

def generate_with_validation(raw_text: str, max_retries: int = 2) -> UserProfileSchema:
    prompt = f"Parse into JSON matching {UserProfileSchema.model_json_schema()}: {raw_text}"
    
    for attempt in range(max_retries):
        raw_json = call_llm(prompt)
        try:
            return UserProfileSchema.model_validate_json(raw_json)
        except ValidationError as err:
            # Self-healing loop: feed validation error back to model!
            prompt = f"Your previous JSON was invalid: {err}. Please correct and re-emit strictly valid JSON."
            
    raise RuntimeError("Failed to generate valid schema after retries.")''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Pattern Simulator Workbench
# ==============================================================================
st.header("3. Interactive Workbench: The Multi-Pattern Simulator", divider="rainbow")
st.markdown("Select a pattern to inspect how information flows between pipeline stages:")

selected_pattern = st.selectbox(
    "Choose Pattern to Simulate:",
    [
        "1. Generate -> Review (Compliance Quality Gate)",
        "2. Extract -> Transform (Invoice Standardization)",
        "3. Classify -> Route (Support Switchboard)",
        "4. Generate -> Validate (Self-Healing Code Linter)"
    ]
)

if "Generate -> Review" in selected_pattern:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Stage 1: Drafter Output (Raw Marketing Copy)**")
        st.code(
            '''"Our revolutionary AI engine guarantees 100% uptime and eliminates all human errors forever. Buy now for zero risk!"''',
            language="text"
        )
        st.error("Flagged by Auditor: Unsubstantiated claims ('guarantees 100%', 'eliminates all errors').")
    with col_p2:
        st.markdown("**Stage 2: Reviewer Output (Compliant Revision)**")
        st.code(
            '''"Designed for 99.9% high availability, our AI monitoring engine helps engineering teams detect and mitigate operational incidents faster."''',
            language="text"
        )
        st.success("Approved: Realistic claims, compliant terminology.")

elif "Extract -> Transform" in selected_pattern:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Stage 1: Extracted Entities**")
        st.json({
            "raw_vendor": "Stripe Payments UK",
            "raw_date": "12th of Oct, 2026",
            "raw_amount": "£1,420.50"
        })
    with col_p2:
        st.markdown("**Stage 2: Transformed Normalized Schema**")
        st.json({
            "vendor_id": "VEND-STRIPE-GB",
            "date_iso": "2026-10-12",
            "amount_minor_units": 142050,
            "currency": "GBP"
        })

elif "Classify -> Route" in selected_pattern:
    sample_q = st.text_input("Incoming User Query:", "I want to upgrade from 5 seats to 50 seats on an annual contract.")
    st.info("Stage 1 Classifier Result: `SALES_EXPANSION`")
    st.success("Stage 2 Dispatched: Routed to Enterprise Account Executive Prompt (`enterprise_sales_v2.jinja`).")

elif "Generate -> Validate" in selected_pattern:
    st.markdown("**Self-Healing Retry Loop Trace:**")
    st.code(
        """Attempt 1 -> Output: {"age": "twenty-five"} -> Pydantic ValidationError: age must be integer!
Attempt 2 (Correction Prompt with Error Diff) -> Output: {"age": 25} -> Pydantic Validation Passed!""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. Enterprise Pipeline Topologies", divider="rainbow")

col_topo1, col_topo2 = st.columns([1, 1])

with col_topo1:
    st.markdown("#### Classify -> Route -> Validate Topology")
    dot_patterns = """
    digraph PipelinePatterns {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="User Request", fillcolor="#334155", fontcolor="#ffffff"];
        Classifier [label="Link 1: Classifier\\n(Determine Route)", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];

        ExpertA [label="Link 2A: Tech Expert", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        ExpertB [label="Link 2B: Billing Expert", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];

        Validator [label="Link 3: Schema Validator\\n(Pydantic Gate)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Final [label="Verified Output Payload", fillcolor="#047857", fontcolor="#ffffff"];

        Input -> Classifier;
        Classifier -> ExpertA [label="Bug"];
        Classifier -> ExpertB [label="Invoice"];
        ExpertA -> Validator;
        ExpertB -> Validator;
        Validator -> Final;
    }
    """
    st.graphviz_chart(dot_patterns, use_container_width=False)

with col_topo2:
    st.markdown("#### The Pattern Selection Decision Tree")
    st.markdown(
        """
- Need safety/compliance? -> **Generate -> Review**
- Ingesting messy PDF/OCR? -> **Extract -> Transform**
- Long multi-topic transcript? -> **Analyze -> Summarize**
- Writing multi-file code? -> **Plan -> Execute**
- Handling multi-topic customer chat? -> **Classify -> Route**
- Emitting JSON for databases? -> **Generate -> Validate**
"""
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

pt_tab1, pt_tab2, pt_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with pt_tab1:
    with st.expander("Q1: How does the Generate -> Validate pattern implement automated self-healing?", expanded=True):
        st.markdown(
            """
In the Generate -> Validate pattern, when the model produces an output that fails deterministic schema parsing (e.g. invalid JSON, missing required key, wrong data type), the application does NOT crash. 
Instead, it captures the exact validation exception message (e.g. Pydantic error diff) and constructs a targeted correction prompt:  
*"Your previous completion failed with error: [Error Diff]. Please fix this specific field and re-emit strictly valid JSON."*  
Over 90% of syntax errors are resolved successfully on the first self-healing retry.
"""
        )

    with pt_tab2:
        with st.expander("Trap 1: 'Can we replace the Classify -> Route pattern by just putting all possible instructions in a giant system prompt?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, modern LLMs have huge context windows, so putting all company policies in one prompt is simpler than building a router."*
- **Winning Response**: *"No. Cramming dozens of orthogonal domain rules into a single prompt causes Instruction Interference, high token latency, and massive API costs on every user interaction. Furthermore, a billing specialist needs different tools (Stripe API) than a technical SRE (Kubernetes API). The Classify -> Route pattern allows each downstream prompt to remain razor-sharp, specialized, secure, and equipped with only the specific tool permissions it needs."*
"""
            )

    with pt_tab3:
        c_pt1, c_pt2, c_pt3 = st.columns(3)
        with c_pt1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Generate -> Review**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Decoupling creative generation from adversarial compliance review across two separate prompts.")

        with c_pt2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Self-Healing Retry**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Feeding deterministic Pydantic schema validation error messages back to the model for automatic correction.")

        with c_pt3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Classify -> Route**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using a lightweight upstream prompt to categorize user intent and dispatch to a specialized domain sub-prompt.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 9.2 Summary & Self-Assessment", divider="rainbow")

col_pts1, col_pts2 = st.columns(2)
with col_pts1:
    st.markdown(
        """
### Key Takeaways:
1. **The 6 Patterns**: Generate-Review, Extract-Transform, Analyze-Summarize, Plan-Execute, Classify-Route, Generate-Validate.
2. **Decouple Roles**: Separate creative generation from critical review.
3. **Self-Healing Pipelines**: Feed deterministic validator errors back to the model for automated recovery.
4. **Intelligent Routing**: Use fast classifiers to route users to specialized, least-privilege prompts.
"""
    )

with col_pts2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can identify the optimal prompt chain pattern for a given enterprise workload.", value=True, key="chk_9_2_1")
    st.checkbox("I know how to implement a self-healing Generate-Validate pipeline in Python.", value=True, key="chk_9_2_2")
    st.checkbox("I understand why Classify-Route outperforms monolithic multi-intent system prompts.", value=True, key="chk_9_2_3")
    st.checkbox("I can structure an AI ETL (Extract -> Transform) pipeline with strict schema typing.", value=True, key="chk_9_2_4")
