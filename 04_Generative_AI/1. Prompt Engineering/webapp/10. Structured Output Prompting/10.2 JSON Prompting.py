import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="10.2 JSON Prompting | GenAI Roadmap",
    page_icon="📦",
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
    .schema-token {
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
st.title("📦 10.2 JSON Prompting — Schema Design, Nested Structures & Validation")
st.markdown(
    """
Welcome to **Lesson 10.2: JSON Prompting**.  
JSON is the undisputed lingua franca of web services and distributed systems. 
However, prompting an LLM to generate JSON is far more nuanced than simply saying *"Output JSON"*.

In enterprise production, reliable JSON generation requires mastering **JSON Schema specifications**, differentiating **required vs. optional fields**, managing **deeply nested objects and arrays**, and applying **strict syntax validation guards**.
"""
)

st.divider()

# ==============================================================================
# Section 1: JSON Schema Essentials in Prompt Engineering
# ==============================================================================
st.header("1. Core Pillars of Production JSON Prompting", divider="rainbow")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    with st.container(border=True):
        st.subheader("1. Explicit Typing")
        st.markdown(
            """
Never let the model infer types. Explicitly declare primitive types:
- `string`: Names, ISO dates, IDs
- `number / integer`: Prices, counts, HTTP status
- `boolean`: Flags, booleans (`true`/`false`)
- `array`: Collections of homogeneous items
- `null`: Missing or inapplicable optional fields
"""
        )

with col_p2:
    with st.container(border=True):
        st.subheader("2. Required vs Optional")
        st.markdown(
            """
Always declare what MUST exist versus what can be `null`:
- **Required**: Critical primary keys and status fields
- **Optional**: Secondary metadata or error messages
- **Default Fallback**: Explicit instruction to use `null` instead of omitting the key entirely, preserving fixed key order
"""
        )

with col_p3:
    with st.container(border=True):
        st.subheader("3. Negative Constraints")
        st.markdown(
            """
State clear anti-hallucination rules:
- **NO comments**: Comments (`//` or `/* */`) violate RFC 8259 JSON
- **NO trailing commas**: `{"a": 1,}` breaks strict parsers
- **NO single quotes**: Enforce double quotes `"` on all keys and strings
- **NO backtick wrappers**: If raw JSON is needed directly
"""
        )

st.divider()

# ==============================================================================
# Section 2: Complex Hierarchies — Nested Objects & Arrays
# ==============================================================================
st.header("2. Handling Complex Hierarchies: Nested Objects & Arrays", divider="rainbow")
st.markdown("Real-world data structures (e-commerce orders, medical charts, cloud infrastructure) are inherently hierarchical:")

col_c1, col_c2 = st.columns(2)

with col_c1:
    with st.container(border=True):
        st.subheader("📋 JSON Schema Definition (System Prompt)")
        st.code(
            '''{
  "order_id": "string (format: ORD-XXXX)",
  "customer": {
    "user_id": "string",
    "email": "string",
    "tier": "STANDARD" | "GOLD" | "PLATINUM"
  },
  "items": [
    {
      "sku": "string",
      "quantity": "integer (>= 1)",
      "unit_price_usd": "float",
      "discounts_applied": ["string"]
    }
  ],
  "payment": {
    "method": "CREDIT_CARD" | "PAYPAL" | "WIRE",
    "is_successful": "boolean",
    "transaction_ref": "string or null"
  }
}''',
            language="json",
        )

with col_c2:
    with st.container(border=True):
        st.subheader("🎯 Validated Concrete Model Output")
        st.code(
            '''{
  "order_id": "ORD-98211",
  "customer": {
    "user_id": "usr_4021",
    "email": "sarah.chen@techcorp.io",
    "tier": "PLATINUM"
  },
  "items": [
    {
      "sku": "GPU-RTX4090",
      "quantity": 2,
      "unit_price_usd": 1599.99,
      "discounts_applied": ["VIP_PARTNER_10"]
    },
    {
      "sku": "CABLE-DP21",
      "quantity": 4,
      "unit_price_usd": 19.50,
      "discounts_applied": []
    }
  ],
  "payment": {
    "method": "CREDIT_CARD",
    "is_successful": true,
    "transaction_ref": "ch_3N1K9x2eZvKYlo2C"
  }
}''',
            language="json",
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Schema-Driven Generation & Parsing Pipeline", divider="rainbow")
st.markdown("How structured JSON flows from prompt engineering to backend consumption:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        schema [label="JSON Schema / Pydantic\nClass Definition", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        prompt [label="Prompt Engineering\n(Type Specs + Examples)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        llm [label="LLM Generation\n(JSON Mode / Grammar)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        cleaner [label="Regex & Backtick\nSanitizer", fillcolor="#EAB308", fontcolor="#000000"]
        validator [label="JSON Schema / Pydantic\nValidation Check", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        dest [label="Enterprise Service\n(SQL, Kafka, REST)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        repair [label="Schema Repair Prompt\n(Pass error trace)", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        schema -> prompt -> llm -> cleaner -> validator
        validator -> dest [label=" Passes", color="#10B981"]
        validator -> repair [label=" ValidationError", color="#EF4444"]
        repair -> llm [label=" Retry with Trace", style="dashed"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive JSON Schema Generator & Validation Workbench
# ==============================================================================
st.header("4. Interactive JSON Schema & Validation Workbench", divider="rainbow")
st.markdown("Configure an e-commerce order extraction payload and test real-time validation:")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Schema Configuration")
    include_customer = st.checkbox("Include Customer Profile (Nested Object)", value=True)
    include_discounts = st.checkbox("Include Discount Array (List of Strings)", value=True)
    allow_null_notes = st.checkbox("Allow Optional 'notes' field (Nullable)", value=True)
    
    st.subheader("🧪 Simulated Raw LLM Response")
    simulate_error = st.selectbox(
        "Simulate Generation Quality",
        [
            "Perfect Clean JSON",
            "Markdown Wrapped (```json ... ```)",
            "Trailing Comma Error",
            "Missing Required Key ('order_id')",
            "Type Mismatch (quantity as string)"
        ]
    )

with wb_col2:
    st.subheader("🔍 Parsing & Validation Output")
    
    # Construct base valid dict
    payload_dict = {
        "order_id": "ORD-5542",
        "total_usd": 249.50,
        "is_fulfilled": False
    }
    if include_customer:
        payload_dict["customer"] = {
            "name": "Jordan Lee",
            "email": "jordan@example.com"
        }
    if include_discounts:
        payload_dict["discounts"] = ["WELCOME10", "FREESHIP"]
    if allow_null_notes:
        payload_dict["notes"] = None

    raw_text = ""
    if simulate_error == "Perfect Clean JSON":
        raw_text = json.dumps(payload_dict, indent=2)
    elif simulate_error == "Markdown Wrapped (```json ... ```)":
        raw_text = f"```json\n{json.dumps(payload_dict, indent=2)}\n```"
    elif simulate_error == "Trailing Comma Error":
        formatted = json.dumps(payload_dict, indent=2)
        raw_text = formatted[:-2] + ",\n}"
    elif simulate_error == "Missing Required Key ('order_id')":
        broken_dict = dict(payload_dict)
        broken_dict.pop("order_id", None)
        raw_text = json.dumps(broken_dict, indent=2)
    elif simulate_error == "Type Mismatch (quantity as string)":
        broken_dict = dict(payload_dict)
        broken_dict["total_usd"] = "TWO HUNDRED DOLLARS"
        raw_text = json.dumps(broken_dict, indent=2)

    st.code(raw_text, language="json" if not raw_text.startswith("```") else "text")

    # Validate output
    st.markdown("#### Validation Report")
    
    # Sanitize markdown backticks
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        parsed_json = json.loads(clean_text)
        # Check required fields
        if "order_id" not in parsed_json:
            st.error("❌ Schema Validation Error: Missing required key 'order_id'")
        elif not isinstance(parsed_json.get("total_usd"), (int, float)):
            st.error(f"❌ Type Validation Error: 'total_usd' expected float, got {type(parsed_json.get('total_usd')).__name__}")
        else:
            st.success("✅ Passed JSON Syntax & Type Validation!")
            st.json(parsed_json)
    except json.JSONDecodeError as jde:
        st.error(f"❌ JSON Syntax Error: {jde.msg} at line {jde.lineno}, col {jde.colno}")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Native JSON Schema Enforcement", divider="rainbow")
st.markdown("Compare OpenAI's legacy JSON Mode with modern Strict Structured Outputs via Pydantic:")

st.code(
    '''from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import openai

# 1. Production Pydantic Model with Field Descriptions
class LineItem(BaseModel):
    sku: str = Field(description="Unique inventory SKU code")
    quantity: int = Field(ge=1, description="Number of units purchased")
    unit_price: float = Field(ge=0.0, description="Unit price in USD")

class OrderExtraction(BaseModel):
    order_id: str = Field(description="Order identifier prefixed with ORD-")
    customer_email: EmailStr = Field(description="Validated email address")
    items: List[LineItem] = Field(description="List of purchased line items")
    discount_codes: List[str] = Field(default_factory=list, description="Promo codes applied")
    special_notes: Optional[str] = Field(default=None, description="Delivery or packaging notes")

# 2. Native Structured Outputs via OpenAI SDK (Strict Schema Mode)
client = openai.OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a customer order extraction service. Extract order details accurately."},
        {"role": "user", "content": "Order ORD-9921 for user dev@acme.corp: bought 2x LAPTOP-PRO at 1200.00 each using code HOLIDAY20."}
    ],
    response_format=OrderExtraction,  # Passes JSON Schema with strict=True
)

order: OrderExtraction = completion.choices[0].message.parsed
print(f"Extracted Order: {order.order_id}, Items: {len(order.items)}, Email: {order.customer_email}")
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
    with st.expander("Q1: What is the technical difference between OpenAI's `response_format={'type': 'json_object'}` and `response_format={'type': 'json_schema'}`?", expanded=True):
        st.markdown(
            """
1. **JSON Mode (`json_object`)**: Guarantees that the generated string is valid syntactically as JSON. However, it does **NOT** guarantee that specific keys, nested objects, or field types exist. The model can hallucinate arbitrary key names or omit mandatory fields.
2. **Structured Outputs (`json_schema`)**: Uses **Grammar-Constrained Decoding**. It transforms the provided Pydantic or JSON Schema into a Context-Free Grammar (CFG) at the inference engine level. Any token that would violate the specified keys, types, or nesting is masked to 0 logit probability. It guarantees 100% adherence to both JSON syntax AND your specific schema.
"""
        )
    with st.expander("Q2: How do you handle schema validation errors in prompt pipelines that do not support native constrained decoding?"):
        st.markdown(
            """
Implement an automated **Schema Repair Loop**:
1. Intercept the raw string output from the LLM.
2. Attempt validation using `pydantic.ValidationError` or `jsonschema.validate()`.
3. If an error is raised, extract the exact error message and field path (e.g. `items.0.quantity: value is not a valid integer`).
4. Re-prompt the model with the original text, the malformed output, and the exact validation error trace, instructing it to repair only the invalid fields.
5. Limit the repair loop to a maximum of 2 retries before escalating to human review or a fallback rule.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we use comments in JSON schemas passed to LLMs?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, writing `// field explanation` inside the JSON example helps the model understand what the field means."*
- **Winning Response**: *"No. While the LLM may read comments during input context parsing, it frequently mimics that pattern by generating `//` comments in its own response. Because RFC 8259 JSON forbids comments, this causes standard Python parsers (`json.loads()`) to immediately crash with a JSONDecodeError. Always place field descriptions inside schema descriptions or separate Markdown instructions, never as pseudo-comments inside JSON payloads."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Strict Schema Mode**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Constraining token sampling logits so the model mathematically cannot emit tokens outside the predefined JSON schema.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**JSON Mode vs Schema Mode**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** JSON mode guarantees valid JSON syntax; Schema mode guarantees adherence to your specific keys and data types.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Schema Repair Loop**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Feeding validation error traces back to the model in an automated retry loop to fix broken fields without re-running the entire pipeline.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 10.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Explicit Schemas**: Always specify types, required fields, and nullable constraints in system prompts.
2. **Grammar-Constrained Decoding**: Use native JSON Schema mode (`strict=True`) for enterprise microservices.
3. **No Pseudo-Comments**: Avoid comments in prompt examples to prevent invalid output generation.
4. **Self-Healing Retries**: Capture Pydantic validation errors and re-feed them to the LLM for automated repair.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can differentiate between JSON Mode and Structured Outputs (Schema Mode).", value=True, key="chk_10_2_1")
    st.checkbox("I know how to structure nested objects and arrays in prompt schemas.", value=True, key="chk_10_2_2")
    st.checkbox("I can implement an automated Pydantic schema validation and repair loop.", value=True, key="chk_10_2_3")
    st.checkbox("I understand why JSON comments and trailing commas crash production parsers.", value=True, key="chk_10_2_4")
