import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="17.2 Designing Tools | GenAI Roadmap",
    page_icon="📐",
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
    .design-badge {
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
st.title("📐 17.2 Designing Tools — Naming, Descriptions & Defensive Validation")
st.markdown(
    """
Welcome to **Lesson 17.2: Designing Tools**.  
A model's ability to call external functions successfully depends almost entirely on how clearly those tools are designed and documented. Vague names, missing descriptions, and untyped parameters lead directly to **tool hallucinations**, **wrong function selections**, and **runtime crashes**.

In this lesson, we explore the core principles of production tool design: **semantic naming conventions**, **informative docstrings**, **strict typing and enums**, **client-side input validation**, and **self-healing error feedback**.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Principles of Robust Tool Design
# ==============================================================================
st.header("1. The 6 Principles of Robust Tool Design", divider="rainbow")
st.markdown("Engineering standards to ensure reliable tool selection and zero runtime exceptions:")

principles_df = pd.DataFrame({
    "Design Principle": [
        "1. Precise Semantic Naming",
        "2. Comprehensive Descriptions",
        "3. Strict Type Constraints & Enums",
        "4. Minimal Required Parameters",
        "5. Defensive Client-Side Validation",
        "6. Informative Error Feedback"
    ],
    "Implementation Standard": [
        "Use action-oriented, domain-specific names: `cancel_flight_reservation` instead of `cancel`",
        "Explain WHAT the tool does, WHEN to use it, and WHEN NOT to use it in 2-3 sentences",
        "Use strict enums (`'USD' | 'EUR'`) and regex patterns instead of unbounded strings",
        "Only mark strictly necessary fields as required; provide sensible defaults for optional fields",
        "Validate inputs via Pydantic before calling backend databases or microservices",
        "Return descriptive JSON error messages so the LLM can understand what failed and retry"
    ],
    "Prevents What Failure?": [
        "Tool ambiguity and collision when multiple similar functions exist",
        "Model guessing or invoking tools for inappropriate tasks",
        "Invalid arguments, out-of-range dates, or malformed enum values",
        "Execution deadlocks when the model cannot infer optional values",
        "Unhandled 500 exceptions and database corruption",
        "Silent failure or infinite retry loops where the model repeats the same mistake"
    ]
})
st.dataframe(principles_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Poor Tool Design vs. Production Tool Design
# ==============================================================================
st.header("2. Contrasting Tool Designs: Fragile vs. Production", divider="rainbow")

col_td1, col_td2 = st.columns(2)

with col_td1:
    with st.container(border=True):
        st.subheader("❌ Fragile Tool Schema (Ambiguous)")
        st.code(
            '''{
  "name": "lookup",
  "description": "Looks up data.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "type": {"type": "string"}
    },
    "required": ["query", "type"]
  }
}''',
            language="json",
        )
        st.error("🚨 Critical Defects:")
        st.markdown(
            """
- Generic name `lookup` collides with search, user, and inventory tools.
- Useless description gives zero guidance on when to invoke.
- `type` is an unconstrained string—the model has no idea what values are valid (`"user"`, `"order"`, `"invoice"`?).
"""
        )

with col_td2:
    with st.container(border=True):
        st.subheader("✅ Hardened Production Tool Schema")
        st.code(
            '''{
  "name": "fetch_order_status_by_id",
  "description": "Retrieves real-time shipping and fulfillment status for an e-commerce order. Use when a customer asks about delivery or tracking. For returns, use initiate_return instead.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "description": "Alphanumeric order ID formatted like ORD-XXXXX"
      },
      "include_item_details": {
        "type": "boolean",
        "description": "Whether to return individual SKU line items. Default is false."
      }
    },
    "required": ["order_id"]
  }
}''',
            language="json",
        )
        st.success("🟢 Production Quality:")
        st.markdown(
            """
- Explicit domain name and boundary explanation
- Contrastive guidance ("For returns, use initiate_return instead")
- Formatted description hints (`ORD-XXXXX`) and optional boolean with default
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Defensive Tool Validation & Self-Healing Loop", divider="rainbow")
st.markdown("How structured errors returned in tool messages enable the LLM to self-heal invalid arguments:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        call [label="LLM Tool Call\n{'amount': -50.0}", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        validator [label="Pydantic Validator\n(ge=0.0 constraint)", fillcolor="#EAB308", fontcolor="#000000"]
        exec_ok [label="Execute API\n(Safe Transfer)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        err_msg [label="Tool Error Message\n{'error': 'amount must be >= 0'}", fillcolor="#EF4444", fontcolor="#FFFFFF"]
        retry [label="LLM Self-Correction\n{'amount': 50.0}", fillcolor="#6366F1", fontcolor="#FFFFFF"]

        call -> validator
        validator -> exec_ok [label=" Valid Args", color="#10B981"]
        validator -> err_msg [label=" ValidationError", color="#EF4444"]
        err_msg -> retry [label=" Feedback Loop"]
        retry -> validator
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Tool Design Auditor
# ==============================================================================
st.header("4. Interactive Tool Design Auditor", divider="rainbow")
st.markdown("Audit a candidate tool design and observe automated linting recommendations:")

selected_tool_candidate = st.selectbox(
    "Select Candidate Tool to Audit",
    [
        "Candidate A: Vague Calculator Tool",
        "Candidate B: Unbounded SQL Query Tool",
        "Candidate C: Production Hardened Flight Booking Tool"
    ]
)

col_aud1, col_aud2 = st.columns([1, 1])

with col_aud1:
    st.subheader("📋 Candidate Tool JSON Schema")
    if selected_tool_candidate == "Candidate A: Vague Calculator Tool":
        schema_sample = {
            "name": "calc",
            "description": "Calculates stuff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    elif selected_tool_candidate == "Candidate B: Unbounded SQL Query Tool":
        schema_sample = {
            "name": "execute_query",
            "description": "Runs raw SQL on the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_string": {"type": "string"}
                },
                "required": ["sql_string"]
            }
        }
    else:
        schema_sample = {
            "name": "book_commercial_flight",
            "description": "Reserves a seat on a commercial flight. Requires passenger passport and flight SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_sku": {"type": "string", "description": "Flight code formatted like UA102"},
                    "seat_class": {"type": "string", "enum": ["ECONOMY", "BUSINESS", "FIRST"]},
                    "passenger_name": {"type": "string"}
                },
                "required": ["flight_sku", "seat_class", "passenger_name"]
            }
        }
    st.json(schema_sample)

with col_aud2:
    st.subheader("🔍 Linter Audit & Quality Score")
    if selected_tool_candidate == "Candidate A: Vague Calculator Tool":
        st.error("🔴 Tool Quality Score: 32/100 (HIGH RISK)")
        st.markdown(
            """
**Linter Flags:**
- ❌ **Name Too Terse**: `calc` is ambiguous. Rename to `evaluate_mathematical_expression`.
- ❌ **Uninformative Description**: "Calculates stuff" fails to specify supported operators, syntax, or constraints.
- ❌ **Security Risk**: Free-form expression evaluation risks arbitrary code execution (e.g. `eval('__import__("os").system("...")')`).
"""
        )
    elif selected_tool_candidate == "Candidate B: Unbounded SQL Query Tool":
        st.warning("🟠 Tool Quality Score: 55/100 (SECURITY VULNERABILITY)")
        st.markdown(
            """
**Linter Flags:**
- ❌ **High Security Hazard**: Allowing raw unparameterized SQL strings exposes the database to prompt-injected `DROP TABLE` attacks.
- ⚠️ **Missing Table Scope**: The tool gives no context on accessible tables or columns.
- **Recommended Fix**: Constrain parameters to read-only views or parameterized filter objects (`{"table": "orders", "customer_id": 402}`).
"""
        )
    else:
        st.success("🟢 Tool Quality Score: 96/100 (PRODUCTION GRADE)")
        st.markdown(
            """
**Linter Flags:**
- ✅ **Descriptive Snake Case**: `book_commercial_flight` is unambiguous.
- ✅ **Strict Enums**: `seat_class` is bounded to valid tiers (`ECONOMY`, `BUSINESS`, `FIRST`).
- ✅ **Helpful Format Guidance**: `flight_sku` includes format examples (`UA102`).
"""
        )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Defensive Pydantic Tool Wrapper", divider="rainbow")
st.markdown("Python pattern that validates arguments and formats informative self-healing error traces:")

st.code(
    '''from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any
import json

class TransferFundsSchema(BaseModel):
    recipient_account: str = Field(pattern=r"^ACCT-[0-9]{5}$", description="Account ID formatted as ACCT-XXXXX")
    amount_usd: float = Field(gt=0.0, le=10000.0, description="Transfer amount between $0.01 and $10,000.00")
    memo: str = Field(default="Standard Transfer", max_length=100)

def safe_tool_executor(raw_arguments_json: str) -> str:
    """
    Safely parses and validates LLM tool arguments using Pydantic.
    If validation fails, returns a formatted JSON error for the LLM to self-heal.
    """
    try:
        args_dict = json.loads(raw_arguments_json)
        # Strict validation check
        validated = TransferFundsSchema(**args_dict)
        
        # In production: execute actual atomic bank transfer
        return json.dumps({
            "status": "SUCCESS",
            "tx_id": "TX-99482",
            "transferred_amount": validated.amount_usd,
            "recipient": validated.recipient_account
        })
        
    except (ValidationError, json.JSONDecodeError) as exc:
        # Return structured error back to the model as tool message
        return json.dumps({
            "status": "VALIDATION_ERROR",
            "message": "Invalid tool arguments supplied. Please correct and retry.",
            "details": str(exc)
        })

# Demonstration of self-healing error payload
invalid_payload = json.dumps({"recipient_account": "invalid_id", "amount_usd": -50.0})
print("Validation Error Response for LLM:")
print(safe_tool_executor(invalid_payload))
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
    with st.expander("Q1: Why should tool descriptions include contrastive instructions ('Use tool X for A, but use tool Y for B')?", expanded=True):
        st.markdown(
            """
When an application has multiple related tools (e.g. `create_calendar_invite` vs `update_calendar_invite`), the model's self-attention heads can experience semantic overlap. 
Contrastive guidance in the docstring establishes clear decision boundaries:
- *"Use `update_calendar_invite` when modifying an existing meeting ID. If the meeting does not yet exist, use `create_calendar_invite` instead."*
This eliminates tool flipping and ensures deterministic function selection.
"""
        )
    with st.expander("Q2: How does informative error feedback in tool messages enable agentic self-healing?"):
        st.markdown(
            """
If a tool execution fails and the server returns a generic string like `{"error": "Failed"}`, the LLM has no idea what went wrong and will either give up or repeat the identical invalid call.
When the server returns structured, actionable errors (e.g. `{"error": "Date '2026-02-30' is invalid. February has 28 days."}`), the LLM reads the tool output, realizes its mistake, and issues a corrected tool call (`2026-02-28`) automatically on the next turn.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we rely on the LLM to format dates and phone numbers correctly without regex patterns in the schema?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, if you put 'ISO date' in the description, the model always outputs YYYY-MM-DD."*
- **Winning Response**: *"No. LLMs will frequently emit local formats like 'MM/DD/YYYY', '12th of October', or relative dates ('yesterday'). In production, you must enforce JSON Schema `pattern` regex constraints (e.g. `^[0-9]{4}-[0-9]{2}-[0-9]{2}$`) or validate the string with Pydantic and return a clear self-healing format error if the syntax deviates."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Contrastive Docstring**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A tool description that clarifies boundary conditions by specifying when to use the tool and when to choose an alternative.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Self-Healing Error Feedback**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Returning structured validation failure details in a tool message so the LLM can understand its mistake and emit corrected arguments.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Schema Regex Constraints**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Using JSON Schema `pattern` properties to enforce exact string syntax (like ISO dates or SKU prefixes) at decoding time.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 17.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Clear Semantic Naming**: Use unambiguous, domain-specific names to prevent tool selection collisions.
2. **Contrastive Descriptions**: Explain both what the tool does and when to choose alternative tools.
3. **Strict Parameter Bounds**: Enforce enums, regex patterns, and numeric ranges via JSON Schema.
4. **Informative Error Messages**: Return detailed validation feedback in tool messages to enable self-healing.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to write effective tool descriptions with contrastive guidance.", value=True, key="chk_17_2_1")
    st.checkbox("I understand why unbounded string parameters cause tool hallucination.", value=True, key="chk_17_2_2")
    st.checkbox("I can implement defensive client-side validation using Pydantic.", value=True, key="chk_17_2_3")
    st.checkbox("I understand how structured error messages enable LLMs to self-heal arguments.", value=True, key="chk_17_2_4")
