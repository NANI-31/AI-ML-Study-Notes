import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="5.2 Dynamic Prompt Construction | GenAI Roadmap",
    page_icon="🧩",
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
st.title("🧩 5.2 Dynamic Prompt Construction — Logic, Variables & Composition")
st.markdown(
    """
Welcome to **Lesson 5.2: Dynamic Prompt Construction**.  
While simple applications inject a single user string into a static prompt, enterprise AI workflows assemble prompts dynamically from multiple heterogeneous runtime sources: **user inputs, RAG document contexts, conversation histories, user permission tiers, and feature flags**.

Below, we master the architecture of **Dynamic Prompt Construction**, including conditional branching, multi-variable validation with Pydantic, composable prompt pipelines, and real-time template compilation.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Pillars of Dynamic Prompt Construction
# ==============================================================================
st.header("1. The 6 Pillars of Dynamic Construction", divider="rainbow")
st.markdown("Dynamic construction coordinates multiple runtime variables into an immutable, safe prompt payload:")

pillars_df = pd.DataFrame({
    "Construction Pillar": [
        "1. User Input Variables",
        "2. Context Variables",
        "3. System Directives vs Slots",
        "4. Conditional Logic Branching",
        "5. Multi-Variable Orchestration",
        "6. Reusable Composable Templates"
    ],
    "Runtime Source": [
        "End-user chat UI, form inputs, query parameters",
        "Vector database chunks, SQL metadata, Redis session caches",
        "Fixed security policy, role persona, compliance rules",
        "Feature flags, customer tier flags (`is_enterprise`), locale",
        "Strongly typed parameter dictionaries validated via Pydantic",
        "Componentized template partials (macros, inherited headers)"
    ],
    "Engineering Standard": [
        "Sanitize delimiter overrides; treat strictly as low-trust data",
        "Include chunk source IDs, timestamps, and confidence scores",
        "Static prefix position to guarantee provider KV cache reuse",
        "Use Jinja2 `{% if %}` blocks rather than nested Python f-strings",
        "Type validation with Pydantic; raise ValidationError before LLM call",
        "Store in Git repo as versioned `.jinja` modules"
    ]
})
st.dataframe(pillars_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Technical Deep-Dive & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1 & 2: User Input vs Context Variables
with st.container(border=True):
    st.subheader("1 & 2. Isolating User Inputs from Grounding Context")
    st.markdown(
        """
A dynamic prompt frequently combines **untrusted user queries** with **trusted reference context** retrieved from an internal database. 
They must be placed into distinct, explicit structural XML scopes.
"""
    )
    st.code(
        '''from jinja2 import Template

prompt_template = """
### SYSTEM POLICY:
You are an internal IT helpdesk agent.
Answer the user's inquiry using ONLY the verified runbook context below.

<verified_context>
{% for doc in context_documents %}
[DOC_ID: {{ doc.id }} | REVISION: {{ doc.updated_at }}]
{{ doc.content }}
{% endfor %}
</verified_context>

<user_query>
{{ user_input | trim }}
</user_query>

ANSWER:"""''',
        language="python"
    )

# 4: Conditional Prompts
with st.container(border=True):
    st.subheader("4. Conditional Branching in Templates (Jinja2 Logic)")
    st.markdown(
        """
Rather than maintaining 10 separate prompt strings for different user subscription tiers, use **conditional template blocks** (`{% if %}`). 
This centralizes prompt logic while dynamically adjusting instructions based on runtime permissions.
"""
    )
    st.code(
        '''conditional_template = """
### ROLE: Technical Customer Assistant
Customer Tier: {{ customer_tier }}

### PERMISSION SCOPES:
{% if customer_tier == 'ENTERPRISE' %}
- You are authorized to generate direct SSH remediation commands and debug scripts.
- Offer 24/7 dedicated engineering escalation contact numbers.
{% elif customer_tier == 'PRO' %}
- Suggest dashboard configuration steps and link to self-serve docs.
- Do NOT output raw root shell scripts.
{% else %}
- Provide general troubleshooting advice only.
{% endif %}

USER INQUIRY: {{ inquiry }}
"""''',
        language="python"
    )

# 5: Multi-Variable Orchestration with Pydantic
with st.container(border=True):
    st.subheader("5. Multi-Variable Validation with Pydantic")
    st.markdown(
        """
Before compiling a prompt template, validate all runtime variables through a Pydantic model. 
This catches missing keys, type mismatches, and empty strings **before** spending money on an LLM API call.
"""
    )
    st.code(
        '''from pydantic import BaseModel, Field, HttpUrl
from typing import Literal

class TriagePromptVariables(BaseModel):
    service_name: str = Field(min_length=2, max_length=50)
    error_code: int = Field(ge=100, le=599)
    environment: Literal["production", "staging", "development"]
    is_customer_facing: bool
    incident_url: HttpUrl | None = None

# Validation happens at application boundary:
try:
    vars_payload = TriagePromptVariables(
        service_name="PaymentService",
        error_code=504,
        environment="production",
        is_customer_facing=True
    )
    # Safely compile template using vars_payload.model_dump()
except ValueError as e:
    # Fail fast before invoking expensive LLM!
    print(f"Template Variable Error: {e}")''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Dynamic Prompt Assembly Workbench
# ==============================================================================
st.header("3. Interactive Workbench: Multi-Variable Compilation", divider="rainbow")
st.markdown("Configure runtime variables to observe dynamic Jinja2 conditional compilation in real time:")

col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    in_service = st.selectbox("`service`", ["AuthService", "PaymentGateway", "DataWarehouse"])
    in_tier = st.selectbox("`account_tier`", ["ENTERPRISE", "PRO", "FREE"])
with col_w2:
    in_env = st.selectbox("`env`", ["PRODUCTION", "STAGING"])
    in_strict_json = st.checkbox("`enforce_json_schema`", value=True)
with col_w3:
    in_include_docs = st.checkbox("`inject_rag_docs`", value=True)
    in_max_tokens = st.slider("`max_summary_words`", 30, 150, 60, 10)

raw_user_issue = st.text_input(
    "`user_issue_text`:",
    "Receiving HTTP 429 Too Many Requests when calling the batch sync endpoint."
)

st.markdown("#### Rendered Production Prompt:")

# Dynamic Python Rendering logic
rag_block = ""
if in_include_docs:
    rag_block = """\n<grounding_context>
[DOC-881]: Rate limits on PaymentGateway are capped at 50 req/sec for PRO and 500 req/sec for ENTERPRISE.
[DOC-882]: HTTP 429 triggers exponential backoff with jitter (initial delay 500ms).
</grounding_context>"""

tier_rule = ""
if in_tier == "ENTERPRISE":
    tier_rule = "- Priority: P0 SLA (4-hour resolution window). Direct escalation to Dedicated TAM."
elif in_tier == "PRO":
    tier_rule = "- Priority: Standard Business Hours Support."
else:
    tier_rule = "- Priority: Community Forum Support."

format_rule = "- Return strictly a JSON object: {\"cause\": string, \"remedy\": string, \"sla_tier\": string}." if in_strict_json else "- Format as crisp bullet points."

rendered_dyn_prompt = f"""### SYSTEM INSTRUCTION
You are an Automated Triage Engine for {in_service} ({in_env}).
Customer Tier: {in_tier}
SLA Policy:
{tier_rule}

HARD CONSTRAINTS:
1. Maximum length: {in_max_tokens} words.
2. {format_rule}
{rag_block}

<incoming_ticket>
{raw_user_issue}
</incoming_ticket>

### TRIAGE DECISION:"""

st.code(rendered_dyn_prompt, language="text")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. Dynamic Prompt Ingestion & Validation Architecture", divider="rainbow")

col_da1, col_da2 = st.columns([1, 1])

with col_da1:
    st.markdown("#### Variable Resolution Pipeline")
    dot_dyn = """
    digraph DynamicAssembly {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Pydantic [label="1. Pydantic Schema\\nValidate types & ranges", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        RAG [label="2. Vector Retrieval\\nFetch RAG context chunks", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Jinja [label="3. Jinja2 Engine\\nEvaluate {% if %} & {% for %}", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        Sanitize [label="4. Delimiter Fencing\\nWrap in <query> tags", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Final [label="Immutable Rendered Prompt\\n-> Dispatched to LLM API", fillcolor="#047857", fontcolor="#ffffff"];

        Pydantic -> Jinja;
        RAG -> Jinja;
        Jinja -> Sanitize -> Final;
    }
    """
    st.graphviz_chart(dot_dyn, use_container_width=False)

with col_da2:
    st.markdown("#### The Multi-Source Ingestion Stack")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ STATIC BASE TEMPLATE (Version-controlled in Git)       │
├────────────────────────────────────────────────────────┤
│ + [SYSTEM] Injected Role & Compliance Directives       │
│ + [SESSION] User Account Tier & Auth Roles (JWT token) │
│ + [RAG] Top-K Relevant Document Chunks (ChromaDB)      │
│ + [CHAT] Sliding Window Conversation History (Redis)   │
│ + [USER] Sanitized Input Query (Pydantic validated)    │
└───────────────────────────┬────────────────────────────┘
                            ▼
              [IMMUTABLE PROMPT DISPATCH]""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

dp_tab1, dp_tab2, dp_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with dp_tab1:
    with st.expander("Q1: How do you handle missing or optional variables in dynamic prompt templates?", expanded=True):
        st.markdown(
            """
1. **Pydantic Validation**: Define optional fields explicitly with default values (`notes: str | None = None`).
2. **Jinja2 Default Filters**: Use Jinja2 filters like `{{ notes | default('None provided', true) }}` to prevent runtime `UndefinedError` crashes.
3. **Conditional Template Blocks**: Wrap optional sections in `{% if notes %} ... {% endif %}` so empty slots do not waste prompt tokens.
"""
        )

    with dp_tab2:
        with st.expander("Trap 1: 'Can we build dynamic prompts using simple string replace (.replace)?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, `template.replace('{user_name}', name)` is simple and requires no dependencies."*
- **Winning Response**: *"No. `.replace()` lacks type validation, escaping, conditional branching, and loop handling. If a user's input contains `{user_name}` or other placeholders, naive string replacement can cause replacement recursion bugs or unintended variable poisoning. Production dynamic prompts require structured templating engines like Jinja2 combined with Pydantic schema validation."*
"""
            )

    with dp_tab3:
        c_dp1, c_dp2, c_dp3 = st.columns(3)
        with c_dp1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Dynamic Prompt Compilation**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Merging static template rules with multi-source runtime variables (RAG docs, user input, JWT claims).")

        with c_dp2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Pydantic Pre-Validation**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Type-checking template variables at the API boundary to fail fast before paying for LLM tokens.")

        with c_dp3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Jinja2 Conditional Branching**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using `{% if %}` blocks to dynamically activate rules based on permissions without duplicating templates.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 5.2 Summary & Self-Assessment", divider="rainbow")

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    st.markdown(
        """
### Key Takeaways:
1. **Dynamic Orchestration**: Fuses user inputs, RAG documents, and session permissions into a unified payload.
2. **Pydantic Validation**: Validate variable types and constraints before invoking LLM APIs.
3. **Conditional Logic**: Centralize enterprise logic with Jinja2 `{% if %}` blocks.
4. **Delimiter Fencing**: Quarantine runtime variables within XML tags (`<user_query>`).
"""
    )

with col_ds2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can implement a multi-variable dynamic prompt template using Jinja2.", value=True, key="chk_5_2_1")
    st.checkbox("I know how to validate template variables using Pydantic models.", value=True, key="chk_5_2_2")
    st.checkbox("I understand the architectural risk of naive `.replace()` string substitution.", value=True, key="chk_5_2_3")
    st.checkbox("I can structure conditional prompt blocks based on user subscription tiers.", value=True, key="chk_5_2_4")
