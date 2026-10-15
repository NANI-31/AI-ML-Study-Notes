import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="16.3 Structured Generation - RAG Master Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class StructuredRAGEngine:
    """
    Production-grade Structured Generation Engine:
    1. Schema definition and validation for downstream microservices.
    2. Constrained decoding and grammar-based JSON enforcement.
    3. Structured answer decomposition with step-by-step reasoning fields.
    4. Machine-readable source metadata attachment.
    5. Calibrated confidence scoring and uncertainty quantification.
    """
    def __init__(self):
        pass

    def build_pydantic_schema_definition(self) -> str:
        """Returns standard Python schema contract representation."""
        return (
            "class RAGStructuredResponse(BaseModel):\n"
            "    query: str = Field(description='Original user question')\n"
            "    answer: str = Field(description='Synthesized grounded response')\n"
            "    is_grounded: bool = Field(description='True if 100% supported by context')\n"
            "    confidence_score: float = Field(ge=0.0, le=1.0, description='Calibrated score')\n"
            "    confidence_band: str = Field(description='HIGH, MEDIUM, or LOW')\n"
            "    cited_sources: List[Dict[str, Any]] = Field(description='Metadata provenance')\n"
            "    missing_facts: Optional[str] = Field(default=None, description='Information gaps')"
        )

    def validate_structured_payload(self, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates payload against the enterprise schema contract."""
        required_keys = ["query", "answer", "is_grounded", "confidence_score", "cited_sources"]
        errors = []

        for k in required_keys:
            if k not in payload:
                errors.append(f"Missing required field: '{k}'")

        if "confidence_score" in payload:
            score = payload["confidence_score"]
            if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
                errors.append("Field 'confidence_score' must be a float between 0.0 and 1.0")

        if "is_grounded" in payload and not isinstance(payload["is_grounded"], bool):
            errors.append("Field 'is_grounded' must be a boolean")

        if "cited_sources" in payload and not isinstance(payload["cited_sources"], list):
            errors.append("Field 'cited_sources' must be a list")

        return len(errors) == 0, errors

    def generate_sample_structured_response(
        self,
        query: str,
        answer: str,
        is_grounded: bool,
        confidence: float,
        sources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Constructs a validated structured payload."""
        band = "HIGH" if confidence >= 0.85 else ("MEDIUM" if confidence >= 0.60 else "LOW")
        return {
            "query": query,
            "answer": answer,
            "is_grounded": is_grounded,
            "confidence_score": round(confidence, 2),
            "confidence_band": band,
            "cited_sources": sources,
            "missing_facts": None if is_grounded else "Specific SLA penalty figures absent from documentation."
        }


# Instantiate engine at module level
struct_engine = StructuredRAGEngine()

# --- HEADER SECTION ---
st.title("16.3 Structured Generation: Pydantic Schemas, JSON Mode, Source Metadata, & Calibrated Confidence")
st.caption("Chapter 16: Generation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In early RAG prototypes, developers treated the generation output as an unstructured raw text blob: 
`response_text = llm.generate(prompt)`.

In modern enterprise architectures, however, **unstructured text is a major technical liability**:
1. **Frontend UI Fragility**: Web applications require rich interactive components—such as clickable citation pills, expandable source sidebars, confidence meters, and suggested follow-ups. Regex parsing of raw markdown is brittle and breaks on formatting changes.
2. **Microservice Workflow Automation**: Downstream systems (e.g., Salesforce, Jira, automated remediation bots) require strict, type-safe data objects. A single unescaped markdown tag crashes JSON deserialization.
3. **Auditability & Telemetry**: Regulated enterprises must record exact confidence scores, timestamped source IDs, and citation mappings in observability databases.

**Structured Generation** constrains the model output to guaranteed, machine-readable schemas, 
bundling answers, source metadata, citations, and calibrated confidence metrics into a single type-safe payload.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Structured & Schema-Constrained RAG Pipeline", divider="rainbow")

st.markdown("""
Structured generation interposes a formal grammar compiler between the decoding loop and downstream services. 
Token prediction is restricted to tokens that satisfy the target schema grammar at every autoregressive step.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StructuredGenerationArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // SCHEMA DEFINITION
        subgraph cluster_schema {
            label="1. Schema Definition & Grammar Compilation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            PydanticModel [label="Pydantic Model Contract\\n(Field types, validations, defaults)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            JsonGrammar [label="CFG Grammar / Logit Mask\\n(Restricts tokens to valid JSON transitions)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // CONSTRAINED GENERATION
        subgraph cluster_constrained {
            label="2. Constrained Token Generation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            LLMDecoder [label="Transformer Inference Loop\\n(Sampling logits with temperature <= 0.2)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            LogitMasker [label="Grammar Logit Masker\\n(Zeroes probability of invalid tokens)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            ValidJsonStream [label="Guaranteed Valid JSON Output\\n(Zero syntax errors or unclosed braces)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // DECONSTRUCTED RESPONSE
        subgraph cluster_payload {
            label="3. Deconstructed Structured Payload";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            AnswerField [label="Answer & Reasoning\\n(Factual text & step-by-step logic)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            ProvenanceField [label="Source Metadata Envelopes\\n(Doc IDs, titles, timestamps, offsets)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            ConfidenceField [label="Calibrated Confidence\\n(Numerical score 0-1 & qualitative band)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // DISPATCH
        subgraph cluster_delivery {
            label="4. Downstream Integration";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            UIRenderer [label="UI Component Renderer\\n(Citation pills, sidebars, confidence badge)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            ApiWebhook [label="Automated Microservice Queue\\n(Zero parser failures in production)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        }

        // PIPELINE EDGES
        PydanticModel -> JsonGrammar;
        JsonGrammar -> LogitMasker;
        LLMDecoder -> LogitMasker [label="Raw Logits"];
        LogitMasker -> LLMDecoder [label="Constrained Mask"];
        LLMDecoder -> ValidJsonStream;
        ValidJsonStream -> AnswerField;
        ValidJsonStream -> ProvenanceField;
        ValidJsonStream -> ConfidenceField;
        AnswerField -> UIRenderer;
        ProvenanceField -> UIRenderer;
        ConfidenceField -> UIRenderer;
        AnswerField -> ApiWebhook [label="Type-Safe JSON"];
        ProvenanceField -> ApiWebhook;
        ConfidenceField -> ApiWebhook;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Architectural Guarantees:
* **Zero Syntax Errors**: Grammar-constrained decoding (e.g., Guidance, Outlines, or OpenAI Structured Outputs) prevents invalid JSON at the tokenizer level.
* **Deterministic Contract Enforcement**: Fields cannot be omitted or assigned invalid data types.
* **Separation of Concerns**: Natural language reasoning is decoupled from metadata IDs and numerical confidence ratings.
""")

st.divider()

# ==============================================================================
# SECTION 2: JSON RESPONSES
# ==============================================================================
st.header("2. JSON Responses (Constrained Decoding, Grammar Masks & Raw JSON Contracts)", divider="rainbow")

st.markdown("""
There are two primary paradigms for obtaining JSON responses from foundation models:
""")

jr_col1, jr_col2 = st.columns([1, 1])

with jr_col1:
    st.markdown("""
    ### 1. Prompt-Engineered JSON vs Grammar-Constrained JSON:
    * **Prompted JSON (Legacy)**: Instructing the model *"Output valid JSON only"*. Risks markdown code-fence encapsulation (```` ```json ````), trailing commas, or truncated braces that break `json.loads()`.
    * **Grammar-Constrained Decoding (Modern)**: Compiles a Context-Free Grammar (CFG) from the JSON schema. At each step, a logit mask sets the probability of any token that would violate the JSON grammar to `-infinity`. The model is mathematically incapable of generating invalid JSON.
    """)

with jr_col2:
    st.markdown("### Grammar Masking Mechanics")
    grammar_table = pd.DataFrame([
        {"Current Output State": '{"answer": "', "Valid Next Tokens": "Any string character, \\\"", "Masked / Forbidden Tokens": "{, }, :, numbers, true/false"},
        {"Current Output State": '{"is_grounded": ', "Valid Next Tokens": "true, false", "Masked / Forbidden Tokens": "strings, numbers, {, }"},
        {"Current Output State": '{"confidence": 0.', "Valid Next Tokens": "0-9 digits", "Masked / Forbidden Tokens": "letters, commas, quotes, braces"},
        {"Current Output State": '{"confidence": 0.95', "Valid Next Tokens": ',, }', "Masked / Forbidden Tokens": "digits, letters, quotes"}
    ])
    st.dataframe(grammar_table, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: PYDANTIC SCHEMAS
# ==============================================================================
st.header("3. Pydantic Schemas (Type Validation, Field Constraints & Serialization Pipelines)", divider="rainbow")

st.markdown("""
In Python enterprise stacks, Pydantic acts as the canonical data-contract standard. 
Defining RAG outputs via Pydantic models enables automatic type coercion, regex validation, and range checks.
""")

pyd_col1, pyd_col2 = st.columns([1, 1])

with pyd_col1:
    st.markdown("""
    ### Production Contract Features:
    * **Strict Field Boundaries**: Numerical bounds (`ge=0.0, le=1.0`) ensure confidence metrics remain calibrated.
    * **Enum Constraints**: Constrains categorical outputs (`"HIGH"`, `"MEDIUM"`, `"LOW"`) preventing arbitrary string drift.
    * **Nested Model Hierarchies**: Embeds discrete citation arrays and metadata dictionaries inside the root response object.
    """)

with pyd_col2:
    st.markdown("### Reference Pydantic Contract")
    st.code(r'''
from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    document_id: str = Field(description="Unique source ID")
    verbatim_quote: str = Field(description="Direct supporting excerpt")

class RAGResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: List[Citation]
    missing_facts: Optional[str] = None
''', language="python")

st.divider()

# ==============================================================================
# SECTION 4: STRUCTURED ANSWERS
# ==============================================================================
st.header("4. Structured Answers (Deconstructed Evidence, Step-by-Step Reasoning Fields & Separation of Concerns)", divider="rainbow")

st.markdown("""
Rather than generating an opaque paragraph, structured answers separate the **internal reasoning chain** 
from the **user-facing answer** and **actionable recommendations**.
""")

sa_col1, sa_col2, sa_col3 = st.columns(3)

with sa_col1:
    with st.container(border=True):
        st.markdown("#### 1. Internal Reasoning Field")
        st.markdown("""
        * **Field**: `thinking_process` (str)
        * **Purpose**: Records intermediate cross-document bridge deductions.
        * **Visibility**: Logged in telemetry, hidden from end-user UI.
        """)

with sa_col2:
    with st.container(border=True):
        st.markdown("#### 2. User-Facing Answer")
        st.markdown("""
        * **Field**: `executive_summary` (str)
        * **Purpose**: Direct, polished synthesis of the factual answer.
        * **Visibility**: Rendered in the main user viewport.
        """)

with sa_col3:
    with st.container(border=True):
        st.markdown("#### 3. Actionable Next Steps")
        st.markdown("""
        * **Field**: `recommended_actions` (List[str])
        * **Purpose**: Explicit operational follow-ups derived from policy.
        * **Visibility**: Rendered as interactive task chips.
        """)

st.divider()

# ==============================================================================
# SECTION 5: SOURCE METADATA
# ==============================================================================
st.header("5. Source Metadata (Machine-Readable Provenance, Document IDs & Chunk Offsets)", divider="rainbow")

st.markdown("""
Source metadata embeds machine-readable provenance directly into the JSON response object. 
This allows frontend applications to build interactive citation sidebars and verify document recency.
""")

sm_col1, sm_col2 = st.columns([1, 1])

with sm_col1:
    st.markdown("""
    ### Mandatory Provenance Fields:
    1. **`document_id`**: Immutable identifier (e.g., `DOC-SEC-402`).
    2. **`source_uri`**: Clickable URL pointing directly to the Confluence, Jira, or S3 object.
    3. **`last_updated`**: ISO-8601 timestamp for customer-facing freshness indicators.
    4. **`authority_tier`**: Certified classification (`Canonical`, `Operational`, or `Informal`).
    """)

with sm_col2:
    st.markdown("### JSON Source Metadata Envelope Example")
    st.code(r'''
"cited_sources": [
  {
    "id": "SEC-402",
    "title": "Corporate Cryptography Standards",
    "uri": "https://wiki.internal/crypto/sec-402",
    "last_updated": "2024-02-15T09:30:00Z",
    "authority_tier": "Canonical",
    "page_offset": 4
  }
]
''', language="json")

st.divider()

# ==============================================================================
# SECTION 6: CONFIDENCE INFORMATION
# ==============================================================================
st.header("6. Confidence Information (Self-Reflective Scoring, Logit Calibration & Uncertainty Quantification)", divider="rainbow")

st.markdown("""
Generative models frequently state incorrect facts with total linguistic confidence. 
Production RAG systems incorporate **calibrated confidence reporting** to quantify uncertainty.
""")

ci_col1, ci_col2 = st.columns([1, 1])

with ci_col1:
    st.markdown("""
    ### 3 Approaches to Confidence Quantification:
    * **1. Self-Reflective LLM Scoring**: Asking the model to evaluate the completeness of context on a scale of 0.0 to 1.0. Fast, but requires prompt calibration to avoid overconfidence.
    * **2. Token Log-Probability Averaging**: Calculating the mean log-probability across generated answer tokens. Low average log-prob indicates the model sampled uncertain tokens.
    * **3. Verification Ratio Metric**: Number of verified claims divided by total generated claims. The most rigorous and auditable metric.
    """)

with ci_col2:
    st.markdown("### Confidence Tiers & Operational Actions")
    conf_actions = pd.DataFrame([
        {"Confidence Band": "HIGH (0.85 - 1.0)", "Operational Action": "Directly render to user. Zero disclaimers required.", "UI Indication": "Green Verified Pill"},
        {"Confidence Band": "MEDIUM (0.60 - 0.84)", "Operational Action": "Render with advisory notice suggesting user review.", "UI Indication": "Amber Review Banner"},
        {"Confidence Band": "LOW (0.00 - 0.59)", "Operational Action": "Suppress answer. Trigger automated human escalation.", "UI Indication": "Red Escalation Alert"}
    ])
    st.dataframe(conf_actions, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise Structured RAG Output Engine", divider="rainbow")

st.markdown("""
Below is the production implementation of `StructuredRAGEngine`. It runs in module scope, 
validating schemas and generating type-safe JSON objects for downstream consumers.
""")

st.code(r'''
class StructuredRAGEngine:
    """
    Enterprise Structured RAG Generation Engine:
    - Enforces Pydantic-aligned schema contracts.
    - Validates required fields, numerical confidence ranges, and data types.
    - Packs verified answers, source provenance envelopes, and confidence ratings.
    """
    def validate_structured_payload(self, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["query", "answer", "is_grounded", "confidence_score", "cited_sources"]
        errors = []
        for k in required:
            if k not in payload:
                errors.append(f"Missing required field: '{k}'")

        if "confidence_score" in payload:
            s = payload["confidence_score"]
            if not isinstance(s, (int, float)) or s < 0.0 or s > 1.0:
                errors.append("confidence_score must be a float between 0.0 and 1.0")

        if "is_grounded" in payload and not isinstance(payload["is_grounded"], bool):
            errors.append("is_grounded must be a boolean")

        return len(errors) == 0, errors

    def generate_sample_structured_response(
        self, query: str, answer: str, is_grounded: bool, confidence: float, sources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        band = "HIGH" if confidence >= 0.85 else ("MEDIUM" if confidence >= 0.60 else "LOW")
        return {
            "query": query,
            "answer": answer,
            "is_grounded": is_grounded,
            "confidence_score": round(confidence, 2),
            "confidence_band": band,
            "cited_sources": sources,
            "missing_facts": None if is_grounded else "Information incomplete in retrieved corpus."
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Schema Validator & Structured Response Inspector", divider="rainbow")

st.markdown("""
Inspect and validate production structured RAG payloads. 
Adjust generation parameters, simulate confidence scores, and verify how the schema validator enforces data contracts.
""")

sample_metadata_sources = [
    {
        "id": "SEC-BitLocker-01",
        "title": "Endpoint Disk Encryption Standards",
        "uri": "https://sec.internal/standards/disk-enc",
        "last_updated": "2024-02-10",
        "authority": "Canonical Tier 1"
    },
    {
        "id": "HR-Remote-02",
        "title": "Global Workplace Flexibility Policy",
        "uri": "https://hr.internal/policies/remote-work",
        "last_updated": "2024-01-20",
        "authority": "Canonical Tier 1"
    }
]

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Payload Configuration Controls")
    in_query = st.text_input("User Question:", value="What encryption is required for employee laptops?")
    in_answer = st.text_area(
        "Synthesized Grounded Answer:",
        value="All employee laptop hard drives must be encrypted using AES-256 BitLocker encryption.",
        height=100
    )
    in_grounded = st.toggle("Mark as 100% Grounded in Context", value=True)
    in_confidence = st.slider("Calibrated Confidence Score:", min_value=0.0, max_value=1.0, value=0.95, step=0.05)

with wb_c2:
    st.markdown("### Generated Structured JSON Payload")
    payload = struct_engine.generate_sample_structured_response(
        query=in_query,
        answer=in_answer,
        is_grounded=in_grounded,
        confidence=in_confidence,
        sources=sample_metadata_sources
    )

    is_valid, validation_errors = struct_engine.validate_structured_payload(payload)

    if is_valid:
        st.success(f"✅ Schema Valid: Conforms to Enterprise Contract ({payload['confidence_band']} Confidence)")
    else:
        st.error(f"❌ Schema Invalid: {len(validation_errors)} error(s) detected.")

    st.json(payload)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is Grammar-Constrained Decoding, and how does it guarantee valid JSON at the tokenizer level?**"):
    st.markdown("""
    **Answer**:
    Grammar-constrained decoding constructs a finite state machine (FSM) or Context-Free Grammar (CFG) from a target schema.
    At every forward pass of the Transformer model:
    1. The FSM examines the tokens generated so far and determines the set of allowed next tokens that maintain valid grammar.
    2. A binary logit mask is applied to the vocabulary logits before the softmax operation, setting the logits of all invalid tokens to `-infinity`.
    3. The model cannot sample tokens that violate the schema. The generation is mathematically guaranteed to output valid, parseable JSON without post-processing or retry loops.
    """)

with st.expander("**Q2: Why should downstream microservices consume Pydantic objects rather than raw JSON strings in RAG pipelines?**"):
    st.markdown("""
    **Answer**:
    Raw JSON strings only validate syntax (matching brackets and quotes). They do not validate **semantics or types**.
    Pydantic objects enforce:
    1. **Type Safety**: Ensuring `confidence_score` is a float, not a string `"high"`.
    2. **Value Bounds**: Ensuring confidence is bounded `[0.0, 1.0]` and preventing out-of-range anomalies.
    3. **Schema Evolution**: Providing default values and backward compatibility when new metadata fields are introduced.
    """)

with st.expander("**Q3: What causes LLM overconfidence in self-reported confidence scores, and how can it be calibrated?**"):
    st.markdown("""
    **Answer**:
    Language models trained with Reinforcement Learning from Human Feedback (RLHF) are incentivized to provide confident, helpful-sounding responses. Consequently, when asked *"How confident are you from 0 to 100?"*, models routinely report 95%+ even when hallucinating.
    Calibration techniques:
    1. **Verbatim Grounding Ratio**: Calculate confidence empirically as `Verified Claims / Total Claims` rather than relying on self-reporting.
    2. **Logit Probability Extraction**: Measure the average token log-probability of generated entity names.
    3. **Consistency Sampling**: Generate 3 responses at temperature 0.4 and measure cross-generation consensus.
    """)

with st.expander("**Q4: How does structured generation simplify frontend UI rendering of citations?**"):
    st.markdown("""
    **Answer**:
    With raw text, frontends must run regex to parse citations like `[DOC-1]`, match them with external lists, and handle edge cases where the LLM writes `[Doc 1]` or `(source: 1)`.
    With structured generation, citations are returned as a first-class array of objects:
    `[{"doc_id": "SEC-01", "url": "https://...", "quote": "..."}]`
    The frontend simply iterates through the array to render interactive citation pills, clickable modals, and sidebar links cleanly.
    """)

with st.expander("**Q5: What are the latency tradeoffs of using grammar-constrained decoding vs standard unconstrained generation?**"):
    st.markdown("""
    **Answer**:
    Grammar-constrained decoding incurs a slight computational overhead:
    * **Initialization Overhead**: 5 to 50 milliseconds to compile the JSON schema into an FSM or CFG parser.
    * **Per-Token Masking Overhead**: 0.5 to 2 milliseconds per token to update the FSM state and apply the logit mask.
    * **Net Latency Advantage**: Because the model is prevented from generating conversational pleasantries or runaway prose, the total token count is significantly lower, resulting in a net **faster** end-to-end response time.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Relying on Markdown Code Fences for JSON**
    * **Failure Mode**: When asking for JSON, models often output ```` ```json { ... } ``` ````. Passing this directly to `json.loads()` raises a `JSONDecodeError`.
    * **Solution**: Use constrained decoding modes or strip markdown fences programmatically before parsing.
    """)

    st.error("""
    **Production Trap 2: Omitting Missing-Information Fields**
    * **Failure Mode**: Forcing the model to output a standard answer field when information is missing forces it to fill the field with speculative text.
    * **Solution**: Include an explicit `missing_facts` or `is_supported` field in the schema to allow clean abstention.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Uncalibrated Self-Reported Confidence**
    * **Failure Mode**: Blindly routing queries based on the model's self-reported confidence without verification.
    * **Solution**: Combine self-reported confidence with empirical claim-level NLI verification.
    """)

    st.warning("""
    **Production Trap 4: Overly Complex Deeply Nested Schemas**
    * **Failure Mode**: Schemas with 8+ levels of nesting increase token consumption and can degrade reasoning quality in compact models.
    * **Solution**: Keep schemas flat and modular, restricting nesting to 2-3 levels maximum.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Grammar-Constrained Decoding")
        st.caption("How does CFG decoding prevent JSON parsing errors?")
        st.markdown("**Back**: Applies a dynamic logit mask at each step, setting the probability of syntax-violating tokens to negative infinity.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Pydantic Field Validation")
        st.caption("Why use Pydantic models for RAG output ingestion?")
        st.markdown("**Back**: Enforces strict types, numerical bounds, enum categories, and automated serialization across microservices.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Structured Provenance Envelope")
        st.caption("What attributes must accompany cited sources in structured JSON?")
        st.markdown("**Back**: Unique doc ID, source URI, last updated timestamp, authority tier, and verbatim supporting quote.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Calibrated Confidence Tiers")
        st.caption("What operational actions correspond to High, Medium, and Low confidence?")
        st.markdown("**Back**: High = direct render; Medium = render with user review banner; Low = suppress output and escalate to human.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of structured RAG generation before proceeding:")

c1 = st.checkbox("I understand the mechanics of grammar-constrained decoding and how logit masking guarantees valid JSON.", value=True)
c2 = st.checkbox("I can define Pydantic schema contracts enforcing types, numerical ranges, and required fields.", value=True)
c3 = st.checkbox("I know how to structure RAG responses to separate internal reasoning, user answers, and action items.", value=True)
c4 = st.checkbox("I can attach machine-readable source provenance envelopes to support interactive UI citations.", value=True)
c5 = st.checkbox("I understand why self-reported confidence is uncalibrated and how to combine it with empirical NLI verification.", value=True)
c6 = st.checkbox("I know how to design structured schemas that enable clean abstention when evidence is absent.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **Structured Generation Mastery Achieved!** You are ready to integrate type-safe RAG outputs into enterprise production microservices.")
