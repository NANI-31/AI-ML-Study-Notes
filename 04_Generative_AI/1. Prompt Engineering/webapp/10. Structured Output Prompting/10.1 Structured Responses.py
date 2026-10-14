import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="10.1 Structured Responses | GenAI Roadmap",
    page_icon="🧱",
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
    .format-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
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
st.title("🧱 10.1 Structured Responses — JSON, Tables, Lists & Key-Value Schemas")
st.markdown(
    """
Welcome to **Lesson 10.1: Structured Responses**.  
Natural language is expressive and human-friendly, but software architectures, microservices, databases, and automated pipelines require **deterministic, machine-readable syntax**. 

Structured output prompting is the discipline of constraining an LLM to generate data conforming strictly to predictable specifications—such as **JSON**, **Markdown Tables**, **Strict Lists**, or **Key-Value Pairs**—enabling seamless ingestion without brittle regex parsing.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is Structured Output & Why It Is Mission-Critical
# ==============================================================================
st.header("1. What is Structured Output & Why It Matters", divider="rainbow")

col_w1, col_w2 = st.columns(2)

with col_w1:
    with st.container(border=True):
        st.subheader("❌ Unstructured Output (Free-Form Prose)")
        st.markdown(
            """
**User Prompt:**
```text
Extract invoice details from: Invoice #9482 paid on Oct 12 by Acme Corp for $1,250.00.
```

**LLM Response:**
```text
Sure! Here are the details you requested:
The invoice number is #9482, paid by Acme Corp on October 12. 
The total amount charged was $1,250.00. Let me know if you need anything else!
```
**Enterprise Failure Points:**
- Unpredictable conversational filler ("Sure! Here are the details...")
- Varying date formats ("October 12" vs "2026-10-12")
- Inconsistent key names across calls
- Impossible to load directly into SQL or send over REST APIs without custom NLP parsers
"""
        )

with col_w2:
    with st.container(border=True):
        st.subheader("✅ Structured Output (Deterministic JSON)")
        st.markdown(
            """
**User Prompt:**
```text
Extract invoice details into JSON conforming to:
{"invoice_id": str, "vendor": str, "amount_usd": float, "date": "YYYY-MM-DD"}
Return ONLY valid JSON.
```

**LLM Response:**
```json
{
  "invoice_id": "9482",
  "vendor": "Acme Corp",
  "amount_usd": 1250.00,
  "date": "2026-10-12"
}
```
**Enterprise Advantages:**
- **Zero Hallucinated Filler**: No chit-chat or introductory pleasantries
- **Direct Serializability**: Ingestible directly via `json.loads()`
- **Type Safety**: Enforces numeric floats, ISO-8601 timestamps, booleans
- **Downstream Composability**: Directly binds to Pydantic models and database rows
"""
        )

st.divider()

# ==============================================================================
# Section 2: Core Formats Matrix
# ==============================================================================
st.header("2. Core Structured Output Formats Matrix", divider="rainbow")
st.markdown("Choosing the right output format balances token consumption, parsing reliability, and human readability:")

format_comparison_df = pd.DataFrame({
    "Format": [
        "1. JSON (JavaScript Object Notation)",
        "2. Markdown Tables",
        "3. Delimited Lists (Bullets / Numbered)",
        "4. Key-Value Pairs (YAML / INI style)",
        "5. CSV (Comma Separated Values)"
    ],
    "Token Efficiency": [
        "Moderate (schema keys repeated in arrays)",
        "Moderate-High (pipes and alignment spaces)",
        "High (minimal syntax overhead)",
        "High (concise key-value syntax)",
        "Very High (header line + raw values)"
    ],
    "Parsability Reliability": [
        "Highest (`json.loads()`, Pydantic)",
        "Moderate (requires markdown table parser)",
        "Moderate (string splitting / regex)",
        "High (`yaml.safe_load()`)",
        "Moderate-High (`csv.reader`)"
    ],
    "Best Production Use Case": [
        "API integrations, microservices, tool-calling payloads, database records",
        "Human review dashboards, documentation generators, executive reports",
        "Checklists, ranking recommendations, sequential action items",
        "Configuration files, prompt metadata, logging annotations",
        "Bulk tabular data export, batch dataset generation"
    ]
})
st.dataframe(format_comparison_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Unstructured Ingestion to Validated Ingestion", divider="rainbow")
st.markdown("In production pipelines, structured outputs must pass through automated parsing and schema validation gates before reaching databases:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        raw_input [label="Raw Text Input\n(Emails, Tickets, Docs)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        prompt [label="Prompt with\nTarget Schema", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        llm [label="LLM Inference\n(Constrained Gen)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        validator [label="Parser & Validator\n(Pydantic / json.loads)", fillcolor="#EAB308", fontcolor="#000000"]
        db [label="Database / API\n(Safe Ingestion)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        fallback [label="Retry or Fallback\nRepair Loop", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        raw_input -> prompt -> llm -> validator
        validator -> db [label=" Valid Schema", color="#10B981"]
        validator -> fallback [label=" Parse Error", color="#EF4444"]
        fallback -> prompt [label=" Error Feedback", style="dashed"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Structured Output Workbench
# ==============================================================================
st.header("4. Interactive Structured Output Workbench", divider="rainbow")
st.markdown("Test how raw customer support logs are structured into different enterprise formats:")

raw_ticket_sample = """Customer Ticket #40291:
From: alex.turner@example.com
Date: October 12, 2026, 14:32 EST
Priority: Urgent
Product: CloudSync Enterprise v3.2
Issue: Users cannot authenticate via Okta SSO since this morning's patch. 
Error code thrown is 403-FORBIDDEN_RELAY. We have 450 engineers blocked across London and Singapore offices."""

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.subheader("📥 Input Raw Telemetry")
    ticket_input = st.text_area("Support Ticket Body", value=raw_ticket_sample, height=200)
    target_format = st.selectbox(
        "Select Target Output Format",
        ["JSON Object", "Markdown Table", "Key-Value Pairs (YAML)", "Bullet Action Items"],
    )

with col_sb2:
    st.subheader("📤 Extracted Structured Representation")
    
    if target_format == "JSON Object":
        sample_json = {
            "ticket_id": 40291,
            "sender_email": "alex.turner@example.com",
            "timestamp": "2026-10-12T14:32:00-05:00",
            "priority": "URGENT",
            "product": "CloudSync Enterprise",
            "product_version": "3.2",
            "error_code": "403-FORBIDDEN_RELAY",
            "affected_users_count": 450,
            "impacted_locations": ["London", "Singapore"],
            "root_cause_suspect": "Morning patch Okta SSO integration"
        }
        st.json(sample_json)
        is_valid = True
        try:
            json.dumps(sample_json)
            st.success("✅ JSON Schema Validation: Valid & Ingestible")
        except Exception as e:
            st.error(f"❌ Syntax Error: {e}")

    elif target_format == "Markdown Table":
        sample_table = pd.DataFrame({
            "Attribute": ["Ticket ID", "Sender", "Priority", "Product", "Error Code", "Users Blocked", "Locations"],
            "Extracted Value": ["#40291", "alex.turner@example.com", "Urgent", "CloudSync Enterprise v3.2", "403-FORBIDDEN_RELAY", "450", "London, Singapore"]
        })
        st.table(sample_table)
        st.info("ℹ️ Rendered as markdown table with strict columns.")

    elif target_format == "Key-Value Pairs (YAML)":
        yaml_content = """ticket_id: 40291
client:
  email: alex.turner@example.com
  regions:
    - London
    - Singapore
incident:
  severity: P1-CRITICAL
  error: 403-FORBIDDEN_RELAY
  blocked_headcount: 450
  affected_subsystem: Okta SSO Relay"""
        st.code(yaml_content, language="yaml")
        st.success("✅ Clean key-value hierarchy parsed without conversational overhead.")

    else:
        checklist = """- [ ] Identify SSO patch deployed on 2026-10-12 morning
- [ ] Investigate SAML/Okta relay assertion header mapping (error 403-FORBIDDEN_RELAY)
- [ ] Notify London and Singapore incident command centers (450 engineers impacted)
- [ ] Roll back patch or reissue Okta client secret certificate"""
        st.markdown(checklist)
        st.success("✅ Deterministic action item checklist.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Schema Enforcement", divider="rainbow")
st.markdown("In production systems, combine system prompt schema definition with automated Python validation:")

st.code(
    '''from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json

# 1. Define the Expected Schema as a Pydantic Model
class SupportTicketSchema(BaseModel):
    ticket_id: int = Field(description="Unique numeric ticket ID")
    sender_email: str = Field(description="Email address of the reporter")
    priority: str = Field(description="Priority rating: LOW, MEDIUM, HIGH, URGENT")
    error_code: Optional[str] = Field(default=None, description="System error code if present")
    affected_users: int = Field(default=0, description="Estimated count of blocked users")
    locations: List[str] = Field(default_factory=list, description="List of affected offices")

# 2. Production Prompt Template with Schema Definition
SYSTEM_PROMPT = """You are an automated telemetry parser.
Extract the ticket details into a single valid JSON object matching this schema:
{
  "ticket_id": int,
  "sender_email": str,
  "priority": "LOW" | "MEDIUM" | "HIGH" | "URGENT",
  "error_code": str or null,
  "affected_users": int,
  "locations": list of str
}
Return ONLY the raw JSON string inside triple backticks. Do not include introductory text.
"""

def parse_and_validate_llm_response(raw_llm_output: str) -> SupportTicketSchema:
    # Strip markdown backticks if present
    cleaned_output = raw_llm_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]
    
    # Parse JSON
    parsed_dict = json.loads(cleaned_output.strip())
    
    # Validate against Pydantic schema
    validated_record = SupportTicketSchema(**parsed_dict)
    return validated_record
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
    with st.expander("Q1: Why does raw prompt-based JSON generation sometimes fail even with frontier models?", expanded=True):
        st.markdown(
            """
Frontier models are probabilistic token predictors, not compilers. Without constrained decoding (like OpenAI Structured Outputs or Grammar-based sampling), raw prompting can fail due to:
1. **Preamble Hallucination**: Emitting conversational text ("Here is your JSON:") before the opening brace `{`.
2. **Trailing Commas**: Emitting trailing commas `[1, 2, 3,]`, which is valid JavaScript but invalid RFC 8259 JSON.
3. **Escaping Mishaps**: Forgetting to escape inner double quotes inside string values (`"description": "User said "hello""`).
4. **Token Truncation**: Cutting off before the closing brace `}` if the `max_tokens` threshold is reached.
"""
        )
    with st.expander("Q2: How do Native Structured Outputs (JSON Schema mode) differ from naive prompt-based JSON instructions?"):
        st.markdown(
            """
- **Prompt-Based JSON**: The model is asked in natural language to generate JSON. Token generation is unconstrained, so the model may theoretically emit invalid tokens at any point.
- **Native Structured Outputs (Grammar Masking)**: The LLM decoding engine dynamically masks token logits during inference. At each generation step, only tokens that can validly continue the context-free grammar (CFG) of the specified JSON schema have non-zero probability. This mathematically guarantees 100% valid JSON conforming to the schema.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we rely purely on prompt instructions like \"Respond only in JSON\" for production microservices?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, if you tell GPT-4 to output only JSON, it always does."*
- **Winning Response**: *"No. In production microservices, prompt instructions alone have an empirical failure rate of 0.5% to 3% due to edge cases, unescaped quotes, or conversational preambles. Production architectures MUST implement defense-in-depth: use native JSON schema enforcement where supported, wrap extraction in a Pydantic parser with try/except, and implement an automated self-correction repair loop on validation errors."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Structured Output**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Constraining an LLM's response to conform to machine-readable specifications (JSON, YAML, tables) for automated ingestion.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Constrained Decoding**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Masking invalid token logits during sampling so the LLM cannot emit characters that violate the target grammar or schema.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Schema Drift**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** When an LLM unpredictably changes key names or nesting structures across model versions or system updates.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 10.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Machine-Readability**: Structured outputs bridge conversational LLMs and deterministic backend code.
2. **Format Selection**: Use JSON for APIs, Markdown tables for human review, and Key-Value pairs for configuration.
3. **Defense-in-Depth**: Always combine prompt schema definitions with programmatic parsers (e.g. Pydantic) and error fallbacks.
4. **Constrained Decoding**: Use native JSON Schema mode whenever the provider supports it to guarantee 100% syntactic compliance.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why downstream applications fail on unstructured text.", value=True, key="chk_10_1_1")
    st.checkbox("I can design explicit JSON and table schemas in prompt templates.", value=True, key="chk_10_1_2")
    st.checkbox("I can implement Pydantic validation and markdown-delimiter stripping.", value=True, key="chk_10_1_3")
    st.checkbox("I know the difference between prompt-based JSON and constrained decoding.", value=True, key="chk_10_1_4")
