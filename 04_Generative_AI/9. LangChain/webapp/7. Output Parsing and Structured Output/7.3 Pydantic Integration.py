import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="7.3 Pydantic Integration - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("7.3 Pydantic Integration: Type Safety, Field Constraints & Schema Validation")
st.caption("Chapter 7: Output Parsing and Structured Output | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Pydantic is the standard data validation and settings management library for modern Python. 
In the LangChain ecosystem, Pydantic acts as the **universal schema contract language**:
- It dictates prompt formatting instructions sent to LLMs.
- It validates, coerces, and deserializes JSON strings into typed Python objects.
- It powers function calling and tool definitions across major model providers.

In this lesson, you will master the 7 core pillars of Pydantic integration:
1. **Pydantic Models**: Defining schemas by subclassing `BaseModel`.
2. **Field Definitions**: Using `Field()` to inject semantic descriptions that guide LLM token generation.
3. **Validation**: Enforcing runtime constraints using `@field_validator` and `@model_validator`.
4. **Nested Models**: Structuring complex hierarchical object trees.
5. **Optional Fields**: Safely handling nullable and omitted attributes.
6. **Enum Fields**: Restricting LLM categorical selections to strict enumeration constants.
7. **Error Handling**: Intercepting `ValidationError` exceptions and implementing auto-correction loops.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockPydanticValidationResult:
    """Represents the outcome of a Pydantic schema validation simulation in module scope."""
    def __init__(self, is_valid: bool, validated_dict: Optional[Dict[str, Any]], errors: List[str]):
        self.is_valid = is_valid
        self.validated_dict = validated_dict
        self.errors = errors

class MockPydanticValidatorLab:
    """Simulates real-world Pydantic model validation with field boundaries and enums in module scope."""
    def __init__(self):
        self.valid_priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def validate_ticket_payload(self, raw_dict: Dict[str, Any]) -> MockPydanticValidationResult:
        errors = []
        validated = {}

        # 1. ticket_id validation (int, positive)
        if "ticket_id" not in raw_dict:
            errors.append("Field 'ticket_id' is required.")
        else:
            try:
                t_id = int(raw_dict["ticket_id"])
                if t_id <= 0:
                    errors.append("Field 'ticket_id' must be a positive integer > 0.")
                else:
                    validated["ticket_id"] = t_id
            except (ValueError, TypeError):
                errors.append(f"Field 'ticket_id' must be an integer, got '{raw_dict['ticket_id']}'.")

        # 2. title validation (str, min_length=5)
        if "title" not in raw_dict:
            errors.append("Field 'title' is required.")
        else:
            t_title = str(raw_dict["title"]).strip()
            if len(t_title) < 5:
                errors.append(f"Field 'title' must be at least 5 characters long (got {len(t_title)}).")
            else:
                validated["title"] = t_title

        # 3. priority (Enum check)
        if "priority" not in raw_dict:
            errors.append("Field 'priority' is required.")
        else:
            p_val = str(raw_dict["priority"]).upper().strip()
            if p_val not in self.valid_priorities:
                errors.append(f"Field 'priority' must be one of {self.valid_priorities}, got '{p_val}'.")
            else:
                validated["priority"] = p_val

        # 4. Optional tags (list[str])
        tags = raw_dict.get("tags", [])
        if isinstance(tags, list):
            validated["tags"] = [str(t).strip() for t in tags]
        else:
            errors.append("Field 'tags' must be a list of strings if provided.")

        # 5. Nested customer object
        customer = raw_dict.get("customer")
        if not customer or not isinstance(customer, dict):
            errors.append("Nested field 'customer' is required and must be an object.")
        else:
            cust_errors = []
            cust_dict = {}
            if "email" not in customer or "@" not in str(customer["email"]):
                cust_errors.append("customer.email must be a valid email address containing '@'.")
            else:
                cust_dict["email"] = str(customer["email"]).strip()

            if "tier" not in customer:
                cust_dict["tier"] = "STANDARD"  # Default
            else:
                cust_dict["tier"] = str(customer["tier"]).upper().strip()

            if cust_errors:
                errors.extend(cust_errors)
            else:
                validated["customer"] = cust_dict

        if errors:
            return MockPydanticValidationResult(False, None, errors)
        return MockPydanticValidationResult(True, validated, [])

# --- SECTION 1: ARCHITECTURE BLUEPRINT & VALIDATION ENGINE ---
st.header("1. Architecture Blueprint: The Pydantic Type-Safety Contract & Validation Engine", divider="rainbow")

st.markdown("""
The architecture below depicts how Pydantic bridges prompt engineering and deterministic code execution. 
Pydantic models perform **dual-direction translation**: inspecting Python classes into JSON Schema specifications 
for the prompt, and parsing unstructured tokens back into verified typed object graphs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Tier 1: Schema Definition
        subgraph cluster_schema {
            label = "1. SCHEMA SPECIFICATION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            PydanticClass [label="Pydantic BaseModel Definition\\nclass Ticket(BaseModel):\\n  id: int\\n  priority: PriorityEnum\\n  customer: CustomerProfile", color="#10B981"];
            JsonSchemaGen [label="JSON Schema Introspection\\nmodel_json_schema()\\nGenerates OpenAPI-compatible contract"];
        }

        // Tier 2: Prompt Injection
        subgraph cluster_prompting {
            label = "2. PROMPT INJECTION &amp; CONTEXT TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            FormatInstructions [label="parser.get_format_instructions()\\nInjects JSON Schema &amp; Instructions into Prompt"];
            LLMGeneration [label="Chat Model (GPT-4o / Claude 3.5)\\nGuided Next-Token Generation"];
        }

        // Tier 3: Validation Pipeline
        subgraph cluster_validation_pipe {
            label = "3. VALIDATION &amp; TYPE-COERCION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            RawJsonPayload [label="Raw LLM Response String\\ncontent='{\"id\": 101, ...}'"];
            PydanticParserNode [label="PydanticOutputParser\\nCalls Ticket.model_validate_json()"];
            FieldValidators [label="@field_validator &amp; @model_validator\\nCustom business logic guards"];
        }

        // Tier 4: Output States
        subgraph cluster_outcomes {
            label = "4. RESOLUTION &amp; EXECUTION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            LiveInstance [label="Validated Instance: Ticket\\nAccess via ticket.customer.email\\nFull IDE Autocomplete", color="#10B981"];
            ValidationErrorNode [label="pydantic.ValidationError\\nTriggers Error Handler / Retry Chain", color="#EF4444"];
        }

        // Connections
        PydanticClass -> JsonSchemaGen;
        JsonSchemaGen -> FormatInstructions;
        FormatInstructions -> LLMGeneration;
        LLMGeneration -> RawJsonPayload;
        RawJsonPayload -> PydanticParserNode;
        PydanticParserNode -> FieldValidators;
        
        FieldValidators -> LiveInstance [label="Constraints Satisfied", color="#10B981"];
        FieldValidators -> ValidationErrorNode [label="Validation Failure", color="#EF4444", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: PYDANTIC MODELS ---
st.header("2. Pydantic Models (`BaseModel` Foundation)", divider="rainbow")

st.markdown("""
At the core of Pydantic is `BaseModel`. When you define a class inheriting from `BaseModel`, 
Python type annotations are automatically converted into runtime data parsing and validation rules.
""")

st.subheader("Fundamental Model Definition in LangChain")
st.code('''from pydantic import BaseModel
from typing import List

class CodeReviewSummary(BaseModel):
    repository_name: str
    total_files_reviewed: int
    vulnerabilities_detected: bool
    tags: List[str]

# Inspecting the automatic JSON Schema generated by Pydantic:
print(CodeReviewSummary.model_json_schema())
''', language="python")

st.info("""
💡 **Pydantic v2 Notice**: Modern LangChain uses **Pydantic v2**. 
Methods such as `.dict()` and `.parse_raw()` are deprecated in favor of **`.model_dump()`** and **`.model_validate_json()`**.
""")

st.divider()

# --- SECTION 3: FIELD DEFINITIONS ---
st.header("3. Field Definitions (`Field()` as Semantic Anchors)", divider="rainbow")

st.markdown("""
While standard type annotations (e.g. `title: str`) declare variable types, LLMs need semantic guidance. 
Using `Field()` allows you to provide:
1. **`description`**: Explains the exact semantic meaning of the field. This description is injected directly into the LLM prompt!
2. **`default` / `default_factory`**: Provides safe fallback values if the model omits the attribute.
3. **Numerical boundaries**: `ge` (greater than or equal), `le` (less than or equal).
4. **String constraints**: `min_length`, `max_length`, `pattern` (regex).
""")

st.code('''from pydantic import BaseModel, Field

class EmployeeProfile(BaseModel):
    employee_id: int = Field(
        ...,  # Ellipsis means the field is strictly required
        description="The unique 6-digit corporate employee badge ID",
        ge=100000,
        le=999999
    )
    full_name: str = Field(
        ...,
        description="Legal first and last name of employee",
        min_length=2,
        max_length=100
    )
    department: str = Field(
        default="Engineering",
        description="Organizational business unit"
    )
    salary_usd: float = Field(
        ...,
        description="Annual compensation in USD",
        ge=0.0
    )
''', language="python")

st.divider()

# --- SECTION 4: VALIDATION ---
st.header("4. Validation Hooks (`@field_validator` & `@model_validator`)", divider="rainbow")

st.markdown("""
Type annotations verify that a value is a string or number, but **business rules** require custom logic. 
Pydantic provides decorators to intercept and validate fields during instantiation:
- **`@field_validator`**: Evaluates individual attributes (e.g., verifying an email format or checking allowed prefixes).
- **`@model_validator(mode='after')`**: Evaluates multi-field cross-validation logic across the entire object.
""")

st.code('''from pydantic import BaseModel, Field, field_validator, model_validator

class ServerDeploymentRequest(BaseModel):
    cluster_name: str = Field(description="Target Kubernetes cluster")
    min_replicas: int = Field(ge=1, description="Minimum worker pods")
    max_replicas: int = Field(ge=1, description="Maximum autoscaled pods")

    @field_validator("cluster_name")
    @classmethod
    def validate_cluster_name(cls, v: str) -> str:
        valid_regions = ["us-east", "us-west", "eu-central", "ap-southeast"]
        if not any(v.startswith(reg) for reg in valid_regions):
            raise ValueError(f"Cluster name must start with one of {valid_regions}")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_replica_bounds(self) -> "ServerDeploymentRequest":
        if self.min_replicas > self.max_replicas:
            raise ValueError(
                f"min_replicas ({self.min_replicas}) cannot exceed max_replicas ({self.max_replicas})"
            )
        return self
''', language="python")

st.divider()

# --- SECTION 5: NESTED MODELS ---
st.header("5. Nested Models (Hierarchical Schemas & Composition)", divider="rainbow")

st.markdown("""
Real-world systems rarely operate on flat dictionaries. Pydantic allows you to nest models within models, 
enabling LLMs to generate richly structured hierarchical data trees.
""")

nested_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    OrderModel [label="Root: PurchaseOrder\\norder_id: str\\ntimestamp: str\\ncustomer: CustomerInfo\\nline_items: List[LineItem]\\nbilling: BillingAddress", color="#10B981"];
    
    subgraph cluster_nested {
        label = "Nested Child Models";
        style = "dashed";
        color = "#334155";
        fontcolor = "#94A3B8";

        CustomerNode [label="Child 1: CustomerInfo\\nname: str\\nemail: EmailStr\\nloyalty_tier: str", fillcolor="#0F172A"];
        LineItemNode [label="Child 2: LineItem\\nsku: str\\nquantity: int\\nunit_price: float", fillcolor="#0F172A"];
        BillingNode [label="Child 3: BillingAddress\\nstreet: str\\ncity: str\\npostal_code: str", fillcolor="#0F172A"];
    }

    OrderModel -> CustomerNode [label="has one"];
    OrderModel -> LineItemNode [label="has many (List)"];
    OrderModel -> BillingNode [label="has one"];
}
"""
st.graphviz_chart(nested_dot, use_container_width=True)

st.code('''from pydantic import BaseModel, Field
from typing import List

class LineItem(BaseModel):
    sku: str = Field(description="Product SKU identifier")
    quantity: int = Field(ge=1, description="Number of units purchased")
    unit_price: float = Field(ge=0.0, description="Price per unit in USD")

class CustomerInfo(BaseModel):
    name: str = Field(description="Customer full name")
    email: str = Field(description="Customer primary email")

class PurchaseOrder(BaseModel):
    order_id: str = Field(description="Order UUID string")
    customer: CustomerInfo = Field(description="Customer identity record")
    items: List[LineItem] = Field(description="List of ordered items")
''', language="python")

st.divider()

# --- SECTION 6: OPTIONAL FIELDS ---
st.header("6. Optional Fields (`Optional[T]` & Default Handling)", divider="rainbow")

st.markdown("""
LLMs frequently encounter inputs where certain information is unstated or ambiguous. 
If an attribute is declared as strictly required, the model may either **hallucinate** a fictitious value or crash the parser.

Declaring attributes as `Optional[T] = None` explicitly communicates to both the model and the parser that the 
field may be omitted safely.
""")

st.code('''from pydantic import BaseModel, Field
from typing import Optional, List

class BugReport(BaseModel):
    title: str = Field(description="Summary title of the defect")
    stack_trace: Optional[str] = Field(
        default=None,
        description="Raw error stack trace if present in the user report"
    )
    reproduction_steps: Optional[List[str]] = Field(
        default_factory=list,
        description="Numbered steps to reproduce bug if provided"
    )
    browser_version: Optional[str] = Field(
        default=None,
        description="Reported client browser version or OS"
    )
''', language="python")

st.divider()

# --- SECTION 7: ENUM FIELDS ---
st.header("7. Enum Fields (Restricted Categorical Constants)", divider="rainbow")

st.markdown("""
When a field must take only one of a predefined set of values (e.g. ticket priority, user role, cloud provider), 
using open-ended string types allows models to invent synonyms (e.g. returning `"URGENT"` or `"IMMEDIATE"` when the database requires `"CRITICAL"`).

Binding a standard Python `Enum` to a Pydantic model restricts valid outputs to exact enumeration constants.
""")

st.code('''from enum import Enum
from pydantic import BaseModel, Field

class CloudProvider(str, Enum):
    AWS = "AWS"
    GCP = "GCP"
    AZURE = "AZURE"
    OCI = "OCI"

class EnvironmentTier(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class InfrastructureTarget(BaseModel):
    provider: CloudProvider = Field(description="Target public cloud infrastructure provider")
    environment: EnvironmentTier = Field(description="Deployment target tier")
    region: str = Field(description="Deployment region code e.g. us-east-1")
''', language="python")

st.divider()

# --- SECTION 8: ERROR HANDLING ---
st.header("8. Error Handling (`ValidationError` & Schema Recovery)", divider="rainbow")

st.markdown("""
When an LLM produces output violating a Pydantic contract, Pydantic raises a `ValidationError` detailing 
the exact path, invalid value, and rule that failed.

Rather than terminating the request, resilient systems inspect the error and choose one of three recovery strategies:
1. **Fallback Recovery**: Return a default safe degraded payload.
2. **OutputFixingParser**: Pass the failed JSON and the Pydantic error trace to an LLM with instructions to repair the syntax.
3. **Prompt Loop (Self-Correction)**: Reprompt the primary model with the validation error attached.
""")

st.code('''from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=InfrastructureTarget)

try:
    # Attempting to parse invalid payload missing required enum values
    result = parser.parse(\'\'\'{"provider": "DIGITAL_OCEAN", "environment": "production", "region": "nyc3"}\'\'\')
except OutputParserException as exc:
    print("Caught parser error:")
    print(exc)
    # Trigger secondary repair chain or return structured error response
''', language="python")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise Pydantic-Enforced API Integration Pipeline", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates an end-to-end Pydantic validation pipeline 
featuring nested child models, enum restrictions, custom field validators, and robust exception recovery.
""")

st.code('''from enum import Enum
from typing import List, Optional, Dict, Any
import logging
from pydantic import BaseModel, Field, field_validator, EmailStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

logger = logging.getLogger("EnterprisePydanticPipeline")

class PriorityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class CustomerMetadata(BaseModel):
    email: str = Field(description="Corporate customer email address")
    company_name: str = Field(description="Organization name")
    account_tier: str = Field(default="ENTERPRISE", description="SLA service tier")

class SupportTicketSchema(BaseModel):
    ticket_id: int = Field(description="Unique numeric incident ID", ge=1000)
    title: str = Field(description="Brief summary title", min_length=5, max_length=150)
    priority: PriorityLevel = Field(description="Severity classification")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    customer: CustomerMetadata = Field(description="Customer identity and metadata")

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        return [tag.lower().strip() for tag in v if tag.strip()]

class SupportTicketParserService:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SupportTicketSchema)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an automated support ticket triage parser.
Parse the incoming support request into the required structured schema.
{format_instructions}"""),
            ("human", "Raw Support Request:\\n{raw_request}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.model | self.parser

    def triage_request(self, raw_request: str) -> Dict[str, Any]:
        try:
            ticket: SupportTicketSchema = self.chain.invoke({"raw_request": raw_request})
            return {
                "status": "SUCCESS",
                "ticket": ticket.model_dump(),
                "priority_code": ticket.priority.value
            }
        except OutputParserException as e:
            logger.error(f"Pydantic parsing failed: {e}")
            return {
                "status": "VALIDATION_FAILED",
                "error_message": str(e),
                "raw_input_snippet": raw_request[:100]
            }
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Live Pydantic Model Validator & Schema Generator", divider="rainbow")

st.markdown("""
Test how Pydantic models validate runtime dictionary payloads. 
Inspect field validations (positive ID, minimum title length, enum constants, nested customer emails) 
and review the generated JSON Schema.
""")

validator_lab = MockPydanticValidatorLab()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("📝 Input Ticket Payload (JSON)")
    default_payload = """{
  "ticket_id": 4820,
  "title": "Database connection pool timeout in prod",
  "priority": "CRITICAL",
  "tags": ["database", "postgres", "timeout"],
  "customer": {
    "email": "ops-lead@acme-corp.com",
    "tier": "ENTERPRISE"
  }
}"""
    user_json = st.text_area("JSON Payload to Validate", value=default_payload, height=220)

with col_wb2:
    st.subheader("📋 Active Pydantic Schema Rules")
    st.info("""
    **Validation Constraints:**
    - `ticket_id`: Required `int`, must be > 0.
    - `title`: Required `str`, minimum 5 characters.
    - `priority`: Required `Enum` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
    - `tags`: Optional `List[str]`.
    - `customer`: Required nested object with valid `email` (must contain `@`).
    """)

if st.button("🔍 Validate against Pydantic Contract", type="primary"):
    import json
    st.subheader("📊 Validation Output")
    
    try:
        parsed_dict = json.loads(user_json)
        res = validator_lab.validate_ticket_payload(parsed_dict)
        
        if res.is_valid:
            st.success("✅ Pydantic Validation Passed! Object instantiated successfully.")
            st.write("#### Validated Object (`model_dump()`):")
            st.json(res.validated_dict)
        else:
            st.error(f"❌ Pydantic ValidationError ({len(res.errors)} violations):")
            for err in res.errors:
                st.write(f"- 🔴 `{err}`")
    except Exception as e:
        st.error(f"JSON Syntax Error before Pydantic parsing: {e}")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is Pydantic v2 significantly faster than Pydantic v1 in high-throughput LLM pipelines?",
     "Pydantic v2 has its core validation engine written entirely in Rust (pydantic-core). JSON serialization and deserialization happen directly in native compiled Rust code rather than interpreted Python loops, resulting in 5x to 20x performance improvements when parsing large LLM responses."),
    ("How does Field(description=...) assist the LLM in generating accurate structured data?",
     "When PydanticOutputParser generates format instructions, it serializes the Pydantic model into a JSON Schema. The description attributes are included directly in the schema's 'description' fields, effectively serving as granular, field-level prompt instructions that guide the LLM's token generation."),
    ("What is the difference between @field_validator and @model_validator in Pydantic v2?",
     "@field_validator operates on a single specific field in isolation and can run before or after Pydantic's default type coercion. @model_validator(mode='after') operates on the entire model instance after all individual fields have been validated, allowing cross-field comparisons (e.g. verifying end_date > start_date)."),
    ("How should you handle an LLM that consistently fails enum validation by generating synonyms?",
     "Three approaches: 1) Add explicit descriptions to each Enum member explaining acceptable values, 2) Use a pre-processing field validator that maps common synonyms to the canonical enum value, or 3) Use provider-native structured output (OpenAI/Anthropic tool calling) which enforces exact enum schemas at the decoding level."),
    ("Can Pydantic models validate streaming LLM outputs chunk by chunk?",
     "Standard Pydantic models require a complete JSON object to validate. However, libraries like partialjson or LangChain's partial streaming parser can parse incomplete JSON trees during streaming, allowing progressive UI rendering before final Pydantic validation on the completed stream.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Deprecated Pydantic v1 Syntax in v2 Environments. Calling `.dict()` or `@validator` in Pydantic v2 code will raise deprecation warnings or AttributeError. Always use `.model_dump()`, `.model_validate()`, and `@field_validator`.")
    st.error("**Trap 2 (Breaking Error)**: Circular Model References. Defining recursive models (e.g. Node referencing Node) without proper `ForwardRef` or `model_rebuild()` causes JSON schema generation to crash with RecursionError during prompt assembly.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Overly Deep Nesting in Schemas. Defining schemas with 5+ layers of nested models generates massive JSON schema instructions (often 1000+ prompt tokens) and increases LLM syntax error rates. Keep schemas flat when possible.")
    st.warning("**Trap 4 (Quality Warning)**: Missing Default Values for Optional Fields. Declaring a field as `Optional[str]` without `= None` means the field is still strictly required to be provided as either a string or explicit null. Always assign `= None` for optional attributes.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Pydantic v2 Serialization"):
        st.markdown("**Use .model_dump() for dicts and .model_dump_json() for JSON strings in Pydantic v2.**")
    with st.expander("🎴 Flashcard 2: Field(description=...)"):
        st.markdown("**Descriptions are injected into the JSON Schema prompt to guide the LLM's understanding.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Enum Constraints"):
        st.markdown("**Inheriting from (str, Enum) guarantees outputs match predefined constant strings.**")
    with st.expander("🎴 Flashcard 4: @model_validator(mode='after')"):
        st.markdown("**Validates relationships between multiple fields across the completed model instance.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can define Pydantic models using BaseModel and type annotations.", value=True)
chk2 = st.checkbox("I know how to use Field(description=...) to guide LLM token generation.", value=True)
chk3 = st.checkbox("I can implement @field_validator and @model_validator for custom business rules.", value=True)
chk4 = st.checkbox("I understand how to build hierarchical schemas using nested Pydantic models.", value=True)
chk5 = st.checkbox("I know how to restrict categorical outputs using Python Enum classes.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 Pydantic integration interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 7.3 Pydantic Integration. Ready for 7.4 Structured Output!")
