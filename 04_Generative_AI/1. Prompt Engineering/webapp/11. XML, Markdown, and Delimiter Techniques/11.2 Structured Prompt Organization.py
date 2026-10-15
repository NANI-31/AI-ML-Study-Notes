import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="11.2 Structured Prompt Organization | GenAI Roadmap",
    page_icon="🏗️",
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
    .tag-pill {
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
st.title("🏗️ 11.2 Structured Prompt Organization — Modular XML Architecture")
st.markdown(
    """
Welcome to **Lesson 11.2: Structured Prompt Organization**.  
As prompts grow from simple questions into complex enterprise operational specifications, monolithic blocks of text become brittle, unmaintainable, and prone to attention drift.

The gold standard for organizing production prompts is the **Modular XML Taxonomy**: dividing prompt components into dedicated functional tags—such as `<instructions>`, `<context>`, `<input>`, `<examples>`, `<constraints>`, and `<output_format>`.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Canonical 6-Tag Prompt Hierarchy
# ==============================================================================
st.header("1. The Canonical 6-Tag Prompt Hierarchy", divider="rainbow")
st.markdown("Each tag in the canonical prompt structure serves a unique semantic role for the transformer attention mechanism:")

tags_df = pd.DataFrame({
    "Tag Element": [
        "1. <instructions>",
        "2. <context>",
        "3. <input>",
        "4. <examples>",
        "5. <constraints>",
        "6. <output_format>"
    ],
    "Semantic Purpose": [
        "Primary objective and operational directive the model must accomplish",
        "Domain background, domain glossaries, system states, and reference documents",
        "Untrusted user query, inbound document, or runtime transaction payload",
        "Few-shot demonstrations showing input-to-output transformations",
        "Negative constraints, guardrails, policy restrictions, and what NOT to do",
        "Exact syntactic specification (JSON schema, table columns, markdown)"
    ],
    "Model Attention Benefit": [
        "Establishes initial high-level task framing",
        "Grounds reasoning in factual retrieval context",
        "Isolates dynamic input into an explicit data sandbox",
        "Calibrates tone, formatting nuance, and edge cases",
        "Acts as a late-stage filter against hallucination and policy violations",
        "Forces deterministic output generation"
    ]
})
st.dataframe(tags_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Complete Enterprise Prompt Blueprint
# ==============================================================================
st.header("2. Complete Production Prompt Blueprint", divider="rainbow")
st.markdown("Here is how an enterprise financial regulatory audit prompt is cleanly structured using the 6-tag architecture:")

st.code(
    '''<system>
You are an expert FinTech Regulatory Compliance Auditor.
</system>

<instructions>
Audit the provided loan application dossier for Anti-Money Laundering (AML) red flags.
Evaluate whether the applicant's declared income matches their verified tax transcripts,
and determine whether enhanced due diligence (EDD) is mandated.
</instructions>

<context>
Federal Bank Regulations mandate Enhanced Due Diligence (EDD) if:
1. Total unverified foreign wire deposits exceed $10,000 in a 30-day window.
2. The debt-to-income (DTI) ratio exceeds 43% with uncorroborated liquid assets.
3. The applicant is classified as a Politically Exposed Person (PEP).
</context>

<examples>
<example>
<input>Applicant received $15,000 from Cayman Islands shell account; DTI 38%; non-PEP.</input>
<output>{"edd_required": true, "trigger": "Foreign wire exceeding $10k threshold", "risk_level": "HIGH"}</output>
</example>
</examples>

<constraints>
- Do NOT provide legal advice or definitive financial verdicts; state regulatory triggers only.
- Do NOT assume missing data; if a field is absent, flag it as "UNVERIFIED".
- Never emit explanations outside the requested JSON format.
</constraints>

<input>
Applicant ID: APP-88201
Declared Annual Salary: $120,000
Verified Tax Return Income: $65,000
Wires: $12,500 incoming from offshore private account (Zurich) on 2026-09-15.
PEP Status: False
</input>

<output_format>
Return a single valid JSON object matching this schema:
{
  "applicant_id": str,
  "income_discrepancy_detected": bool,
  "edd_required": bool,
  "triggers_identified": list of str,
  "recommended_action": "APPROVE" | "ESCALATE_TO_COMPLIANCE" | "REJECT"
}
</output_format>''',
    language="xml",
)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Attention Segmentation via Modular XML", divider="rainbow")
st.markdown("Visualizing how modular XML partitions distinct prompt components for cleaner attention allocation:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=TB, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9, color="white", fontcolor="white"]

        inst [label="<instructions>\n(Task Framing)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        ctx [label="<context>\n(Factual Grounding)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        inp [label="<input>\n(Isolated Payload)", fillcolor="#EAB308", fontcolor="#000000"]
        cst [label="<constraints>\n(Guardrails)", fillcolor="#EF4444", fontcolor="#FFFFFF"]
        fmt [label="<output_format>\n(Schema Target)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="LLM Attention Engine", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        out [label="Deterministic Response", fillcolor="#10B981", fontcolor="#FFFFFF"]

        inst -> llm
        ctx -> llm
        inp -> llm
        cst -> llm
        fmt -> llm
        llm -> out
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Modular Prompt Builder
# ==============================================================================
st.header("4. Interactive Modular Prompt Builder", divider="rainbow")
st.markdown("Customize individual sections to see how clean XML encapsulation produces a production-ready prompt:")

col_b1, col_b2 = st.columns([1, 1])

with col_b1:
    st.subheader("🧩 Toggle & Customize Prompt Modules")
    inc_inst = st.checkbox("Include <instructions>", value=True)
    inc_ctx = st.checkbox("Include <context>", value=True)
    inc_ex = st.checkbox("Include <examples>", value=True)
    inc_cst = st.checkbox("Include <constraints>", value=True)
    inc_fmt = st.checkbox("Include <output_format>", value=True)

    task_name = st.selectbox(
        "Select Domain Task",
        ["Medical Triage Escalation", "Customer Retention Classification", "Code Vulnerability Audit"]
    )

with col_b2:
    st.subheader("📄 Live Assembled XML Prompt")
    
    sections = []
    if task_name == "Medical Triage Escalation":
        if inc_inst:
            sections.append("<instructions>\nTriage the incoming clinical note and classify the immediate urgency level.\n</instructions>")
        if inc_ctx:
            sections.append("<context>\nNormal adult vitals: HR 60-100 bpm, Systolic BP 90-120 mmHg, SpO2 >= 95%.\nCritical alert: SpO2 < 90% or Systolic BP > 180 mmHg.\n</context>")
        if inc_ex:
            sections.append("<examples>\n<example>\n<input>SpO2 88%, severe dyspnea.</input>\n<output>{\"urgency\": \"IMMEDIATE_TRAUMA\", \"action\": \"OXYGEN_STAT\"}</output>\n</example>\n</examples>")
        if inc_cst:
            sections.append("<constraints>\n- Never diagnose illness; report triage urgency only.\n- Output strictly JSON without conversational pleasantries.\n</constraints>")
        sections.append("<input>\nPatient reports acute chest tightness, SpO2 91%, HR 118 bpm, BP 160/100.\n</input>")
        if inc_fmt:
            sections.append("<output_format>\n{\n  \"urgency\": \"ROUTINE\" | \"URGENT\" | \"IMMEDIATE_TRAUMA\",\n  \"flagged_vitals\": [str],\n  \"recommended_unit\": str\n}\n</output_format>")
    
    elif task_name == "Customer Retention Classification":
        if inc_inst:
            sections.append("<instructions>\nClassify churn risk from the user's cancellation exit survey response.\n</instructions>")
        if inc_ctx:
            sections.append("<context>\nChurn tiers: HIGH (mentioning competitor pricing), MEDIUM (feature gap), LOW (temporary budget freeze).\n</context>")
        if inc_ex:
            sections.append("<examples>\n<example>\n<input>Switching to Linear because it is $10 cheaper.</input>\n<output>{\"churn_tier\": \"HIGH\", \"counter_offer\": \"MATCH_COMPETITOR_DISCOUNT\"}</output>\n</example>\n</examples>")
        if inc_cst:
            sections.append("<constraints>\n- Only suggest retention offers authorized in current quarter promotion.\n- Output strictly JSON.\n</constraints>")
        sections.append("<input>\nWe love the app, but leadership cut SaaS spend by 30% for Q4. Might return next year.\n</input>")
        if inc_fmt:
            sections.append("<output_format>\n{\n  \"churn_tier\": \"LOW\" | \"MEDIUM\" | \"HIGH\",\n  \"retention_eligible\": bool,\n  \"campaign_id\": str\n}\n</output_format>")
    
    else:
        if inc_inst:
            sections.append("<instructions>\nInspect the code snippet for Common Weakness Enumeration (CWE) security vulnerabilities.\n</instructions>")
        if inc_ctx:
            sections.append("<context>\nTarget standard: OWASP Top 10 (2025). Focus on Injection, Broken Authentication, and SSRF.\n</context>")
        if inc_ex:
            sections.append("<examples>\n<example>\n<input>cursor.execute(f'SELECT * FROM users WHERE id={uid}')</input>\n<output>{\"vuln\": \"SQL_INJECTION\", \"cwe\": \"CWE-89\", \"severity\": \"CRITICAL\"}</output>\n</example>\n</examples>")
        if inc_cst:
            sections.append("<constraints>\n- Do not flag stylistic issues (PEP8); focus exclusively on security vulnerabilities.\n</constraints>")
        sections.append("<input>\nimport requests\ndef fetch_user_avatar(url):\n    return requests.get(url).content\n</input>")
        if inc_fmt:
            sections.append("<output_format>\n{\n  \"vulnerability_found\": bool,\n  \"cwe_id\": str or null,\n  \"severity\": \"LOW\" | \"MEDIUM\" | \"HIGH\" | \"CRITICAL\",\n  \"remediation_snippet\": str\n}\n</output_format>")

    full_xml = "\n\n".join(sections)
    st.code(full_xml, language="xml")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Modular Prompt Builder", divider="rainbow")
st.markdown("Build an enterprise-grade `XMLPromptBuilder` class to standardize prompt formatting across your team:")

st.code(
    '''from typing import Optional, List, Dict
import html

class XMLPromptBuilder:
    """
    Production-grade builder for modular XML-formatted prompts.
    Enforces section separation, safe escaping, and strict schema tagging.
    """
    def __init__(self, system_role: str):
        self.system_role = system_role
        self.instructions: Optional[str] = None
        self.context: Optional[str] = None
        self.examples: List[Dict[str, str]] = []
        self.constraints: List[str] = []
        self.output_format: Optional[str] = None

    def set_instructions(self, instructions: str) -> "XMLPromptBuilder":
        self.instructions = instructions.strip()
        return self

    def set_context(self, context: str) -> "XMLPromptBuilder":
        self.context = context.strip()
        return self

    def add_example(self, input_val: str, output_val: str) -> "XMLPromptBuilder":
        self.examples.append({"input": input_val.strip(), "output": output_val.strip()})
        return self

    def add_constraint(self, constraint: str) -> "XMLPromptBuilder":
        self.constraints.append(constraint.strip())
        return self

    def set_output_format(self, schema_spec: str) -> "XMLPromptBuilder":
        self.output_format = schema_spec.strip()
        return self

    def build(self, user_input: str) -> str:
        parts = [f"<system>\\n{self.system_role}\\n</system>"]
        
        if self.instructions:
            parts.append(f"<instructions>\\n{self.instructions}\\n</instructions>")
        if self.context:
            parts.append(f"<context>\\n{self.context}\\n</context>")
        if self.examples:
            ex_blocks = ["<examples>"]
            for ex in self.examples:
                ex_blocks.append(f"  <example>\\n    <input>{ex['input']}</input>\\n    <output>{ex['output']}</output>\\n  </example>")
            ex_blocks.append("</examples>")
            parts.append("\\n".join(ex_blocks))
        if self.constraints:
            cst_lines = ["<constraints>"]
            for cst in self.constraints:
                cst_lines.append(f"- {cst}")
            cst_lines.append("</constraints>")
            parts.append("\\n".join(cst_lines))
            
        # Sanitize and enclose input
        clean_input = user_input.replace("</input>", "&lt;/input&gt;")
        parts.append(f"<input>\\n{clean_input}\\n</input>")
        
        if self.output_format:
            parts.append(f"<output_format>\\n{self.output_format}\\n</output_format>")

        return "\\n\\n".join(parts)

# Usage demonstration:
builder = (
    XMLPromptBuilder("You are a Senior Security Architect.")
    .set_instructions("Audit Python code for SSRF vulnerabilities.")
    .add_constraint("Do not execute code; static inspection only.")
    .set_output_format('{"has_vuln": bool, "cwe": str}')
)
prompt = builder.build("def download(url): return urllib.request.urlopen(url).read()")
print(prompt)
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
    with st.expander("Q1: Why is separating `<instructions>` from `<constraints>` superior to combining them in one paragraph?", expanded=True):
        st.markdown(
            """
1. **Cognitive Load & Salience**: Positive directives ("Do X, Y, Z") require goal-seeking attention, while negative constraints ("Never do A, B") require boundary filtering. Grouping them forces the transformer to parse contrasting logical operators in a single clause.
2. **Post-Generation Verification**: In automated evaluation pipelines (LLM-as-a-Judge), having distinct `<constraints>` tags enables dedicated assertions where a judge model audits compliance against the exact list of forbidden behaviors.
"""
        )
    with st.expander("Q2: Where in the prompt should `<output_format>` be positioned for maximum adherence?"):
        st.markdown(
            """
Always position `<output_format>` **at the very end** of the prompt (right before or immediately after the `<input>` tag). 
Due to **Recency Bias** in autoregressive transformers, instructions placed closest to the output generation point exert the strongest conditioning over the first generated tokens (e.g. producing the opening `{` for JSON).
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we put the `<input>` data before the `<instructions>`?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"It doesn't matter; attention is bidirectional across the prompt, so the model sees all tokens."*
- **Winning Response**: *"Placing `<input>` before `<instructions>` is an anti-pattern. If a 10,000-token document appears first, the model processes all those tokens without knowing the task objective or reading criteria, resulting in diluted attention. Providing `<instructions>` first establishes task framing, priming the self-attention heads to actively highlight relevant entities as the document tokens are ingested."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Modular Prompt Taxonomy**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Structuring prompts into dedicated semantic tags (<instructions>, <context>, <input>, etc.) to maximize attention allocation and maintainability.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Recency Bias in Formatting**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The phenomenon where prompts place <output_format> at the very end so formatting constraints have highest salience when generation starts.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Task Priming**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Stating <instructions> before <input> so the LLM's attention heads know what to look for while reading the input payload.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 11.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Modular Architecture**: Decompose prompts into the canonical 6 tags: instructions, context, input, examples, constraints, format.
2. **Task Framing**: Always declare instructions before input to prime attention heads.
3. **Recency Advantage**: Place output format specifications at the bottom to maximize syntactic adherence.
4. **Programmatic Builders**: Use OOP builder patterns in Python to eliminate string formatting bugs across engineering teams.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can structure an enterprise prompt using the canonical 6-tag XML schema.", value=True, key="chk_11_2_1")
    st.checkbox("I understand why instructions must precede input payloads.", value=True, key="chk_11_2_2")
    st.checkbox("I know why output format belongs at the prompt's conclusion.", value=True, key="chk_11_2_3")
    st.checkbox("I can implement an XMLPromptBuilder class in Python.", value=True, key="chk_11_2_4")
