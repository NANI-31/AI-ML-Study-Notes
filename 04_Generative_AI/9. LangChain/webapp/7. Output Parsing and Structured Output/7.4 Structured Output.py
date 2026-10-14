import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="7.4 Structured Output - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("7.4 Structured Output: Native Constrained Decoding, Tool Calling & Agent Schemas")
st.caption("Chapter 7: Output Parsing and Structured Output | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In early Generative AI applications, obtaining structured data required complex prompt formatting instructions 
and fragile post-processing output parsers. Today, modern frontier models (OpenAI, Anthropic, Google Gemini) 
support **Native Structured Output** and **Tool-Based Extraction** directly at the inference engine layer.

With LangChain's unified `ChatModel.with_structured_output(schema)` interface, obtaining structured data 
is now guaranteed, type-safe, and requires zero prompt format instruction bloat.

In this lesson, you will master:
- **Provider-Native Structured Output**: Leveraging constrained decoding and JSON Schema enforcement at the model API level.
- **Tool-Based Extraction**: Using function calling to extract complex entity schemas across providers lacking native JSON mode.
- **Engine-Level Schema Validation**: How constrained grammars prevent syntax errors before tokens are even sampled.
- **Structured Agent Responses**: Forcing autonomous agents to return strongly-typed Pydantic objects as their final deliverable.
- **Handling Validation Failures**: Managing edge cases, fallback schemas, and structural repairs.
- **Multiple Structured Outputs**: Dynamically routing between distinct schemas using typed unions (`Union[SchemaA, SchemaB]`).
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockExtractionRecord:
    """Represents a simulated structured extraction run in module scope."""
    def __init__(self, mode: str, schema_name: str, payload: Dict[str, Any], token_overhead: int, latency_ms: int):
        self.mode = mode
        self.schema_name = schema_name
        self.payload = payload
        self.token_overhead = token_overhead
        self.latency_ms = latency_ms

class MockStructuredOutputSimulator:
    """Simulates native vs tool-based structured output extraction in module scope."""
    def __init__(self):
        pass

    def extract_entity(self, text: str, extraction_mode: str, target_schema: str) -> MockExtractionRecord:
        text_lower = text.lower()

        # Simulated Extraction Logic
        if target_schema == "SecurityIncident":
            severity = "CRITICAL" if any(w in text_lower for w in ["breach", "zero-day", "ransomware", "urgent"]) else "HIGH"
            cve_id = "CVE-2026-9812" if "cve" in text_lower or "vulnerability" in text_lower else "CVE-PENDING"
            impacted = ["auth-service", "user-db"] if "database" in text_lower or "auth" in text_lower else ["api-gateway"]
            
            payload = {
                "schema": "SecurityIncident",
                "cve_id": cve_id,
                "severity": severity,
                "impacted_services": impacted,
                "requires_disclosure": severity == "CRITICAL"
            }
        else:  # CustomerFeedback
            sentiment = "NEGATIVE" if any(w in text_lower for w in ["bad", "terrible", "crash", "broken", "fail"]) else "POSITIVE"
            score = 1 if sentiment == "NEGATIVE" else 5
            category = "PERFORMANCE" if "slow" in text_lower or "latency" in text_lower else "PRODUCT_FEATURE"
            
            payload = {
                "schema": "CustomerFeedback",
                "sentiment": sentiment,
                "satisfaction_score": score,
                "category": category,
                "churn_risk": sentiment == "NEGATIVE"
            }

        # Token overhead: Native structured output adds 0 prompt tokens (handled via API schema parameter)
        # Tool-based adds minimal tool schema tokens
        token_overhead = 0 if "Native" in extraction_mode else 85
        latency_ms = 220 if "Native" in extraction_mode else 310

        return MockExtractionRecord(
            mode=extraction_mode,
            schema_name=target_schema,
            payload=payload,
            token_overhead=token_overhead,
            latency_ms=latency_ms
        )

# --- SECTION 1: ARCHITECTURE BLUEPRINT & CONSTRAINED DECODING ---
st.header("1. Architecture Blueprint: Modern Structured Output & Provider-Native Constrained Decoding", divider="rainbow")

st.markdown("""
The diagram below contrasts **Legacy Output Parsing** with modern **Native Structured Output**. 
In the modern paradigm, schema constraints are pushed directly into the model's token sampler or function calling engine, 
guaranteeing 100% syntactic conformance with zero parsing boilerplate.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Schema Definition
        subgraph cluster_schema {
            label = "1. SCHEMA SPECIFICATION LAYER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserSchema [label="Pydantic Model / TypedDict\\nclass ExtractedEntity(BaseModel)", color="#10B981"];
            StructuredCall [label="model.with_structured_output(schema)\\nUnified LangChain Factory Interface"];
        }

        // Level 2: Provider Dispatch
        subgraph cluster_dispatch {
            label = "2. STRATEGY SELECTION &amp; COMPILATION LAYER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            MethodSelector [label="method='json_schema' vs 'function_calling'\\nAutomatic capability negotiation"];
            EngineGrammar [label="Grammar-Constrained Token Sampler\\nMasks invalid tokens at each step", color="#38BDF8"];
            ToolDef [label="Synthetic Tool Schema Object\\nname='extract_data', parameters=json_schema"];
        }

        // Level 3: Model Execution
        subgraph cluster_execution {
            label = "3. INFERENCE &amp; CONSTRAINED SAMPLING LAYER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            NativeLLM [label="OpenAI / Claude / Gemini API\\nConstrained Decoding Mode\\nZero invalid syntax possible", color="#10B981"];
            ToolCallingLLM [label="Tool Calling Compatible Model\\nEmits tool_calls=[{'name': 'extract_data'}]"];
        }

        // Level 4: Delivery
        subgraph cluster_delivery {
            label = "4. RESOLUTION &amp; INSTANTIATION LAYER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            PydanticInstance [label="Direct Pydantic Model Instance\\nExtractedEntity(...) with 100% type safety", color="#10B981"];
        }

        // Connections
        UserSchema -> StructuredCall;
        StructuredCall -> MethodSelector;
        
        MethodSelector -> EngineGrammar [label="method='json_schema' (Native)"];
        MethodSelector -> ToolDef [label="method='function_calling' (Tools)"];

        EngineGrammar -> NativeLLM;
        ToolDef -> ToolCallingLLM;

        NativeLLM -> PydanticInstance [label="Direct JSON payload parsed to schema"];
        ToolCallingLLM -> PydanticInstance [label="Tool arguments parsed to schema"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Evolutionary Breakthrough:**
- **Legacy Output Parsing**: Model generates text; parser tries to parse it after generation; failures require re-prompting.
- **Native Structured Output**: The inference engine constrains generation at token sampling time, guaranteeing that syntax errors are mathematically impossible.
""")

st.divider()

# --- SECTION 2: PROVIDER-NATIVE STRUCTURED OUTPUT ---
st.header("2. Provider-Native Structured Output (`with_structured_output`)", divider="rainbow")

st.markdown("""
Modern frontier models offer native JSON Schema adherence (e.g. OpenAI Structured Outputs). 
In LangChain, you activate this capability via a single method call: `model.with_structured_output(schema)`.

When using native structured output:
1. **Zero Prompt Token Overhead**: You do not need to inject format instructions into the prompt. The schema is passed directly as an API parameter.
2. **Guaranteed Syntax**: The API guarantees that the output matches the provided JSON Schema.
3. **Direct Return Type**: Invoking the returned Runnable directly returns an instantiated Pydantic object or dictionary.
""")

st.subheader("Implementing Native Structured Output in LCEL")
st.code('''from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. Define the Schema Contract
class SecurityAuditLog(BaseModel):
    threat_detected: bool = Field(description="Whether a security threat was identified")
    threat_level: str = Field(description="Severity: LOW, MEDIUM, HIGH, CRITICAL")
    cve_reference: str = Field(description="Associated CVE identifier if known")
    remediation_steps: list[str] = Field(description="Ordered containment steps")

# 2. Bind the schema to the model using native structured output
model = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = model.with_structured_output(SecurityAuditLog, method="json_schema")

# 3. Invocation: Directly returns a SecurityAuditLog instance!
result: SecurityAuditLog = structured_llm.invoke(
    "Warning: SQL injection vulnerability detected on /api/v1/auth endpoint. Immediate sanitization required."
)

print(type(result))          # <class '__main__.SecurityAuditLog'>
print(result.threat_level)   # 'HIGH' or 'CRITICAL'
print(result.threat_detected) # True
''', language="python")

st.divider()

# --- SECTION 3: TOOL-BASED STRUCTURED OUTPUT ---
st.header("3. Tool-Based Structured Output (`method='function_calling'`)", divider="rainbow")

st.markdown("""
Not all model providers support native JSON Schema constrained decoding. For providers or local open-source models 
that support **Tool Calling** (Function Calling), LangChain emulates structured output by creating a synthetic tool.

The model is instructed to call a function whose parameters match your schema, and LangChain intercepts the 
tool call arguments, parsing them directly into your Pydantic object.
""")

st.code('''from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

class TranslationAnalysis(BaseModel):
    detected_language: str = Field(description="Source language detected")
    translated_text: str = Field(description="English translation")
    sentiment: str = Field(description="Sentiment tone: POSITIVE, NEUTRAL, NEGATIVE")

# Using Anthropic Claude via Tool Calling Strategy
claude_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)

# Specify method='function_calling' (default across tool-enabled models)
structured_claude = claude_model.with_structured_output(
    TranslationAnalysis,
    method="function_calling"
)

result: TranslationAnalysis = structured_claude.invoke(
    "El servicio al cliente fue excepcional y resolvieron mi problema rápidamente."
)

print(result.detected_language)  # 'Spanish'
print(result.translated_text)    # 'Customer service was exceptional...'
print(result.sentiment)          # 'POSITIVE'
''', language="python")

st.divider()

# --- SECTION 4: SCHEMA VALIDATION AT THE ENGINE LAYER ---
st.header("4. Schema Validation at the Engine Layer (Constrained Decoding)", divider="rainbow")

st.markdown("""
How does constrained decoding achieve guaranteed schema conformance? 
During next-token autoregressive sampling, the model's vocabulary logit distribution is filtered through a **Context-Free Grammar (CFG)**:
""")

engine_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    GeneratedPrefix [label="Current Token State: '{\"age\": '"];
    EngineCheck [label="Grammar Parser:\\nSchema requires Integer for 'age'"];
    
    subgraph cluster_vocab {
        label = "Token Logit Masking";
        style = "dashed";
        color = "#334155";
        fontcolor = "#94A3B8";

        AllowedTokens [label="Digits [0-9]\\nLogits Preserved", fillcolor="#0F172A", color="#10B981"];
        MaskedTokens [label="Quotes, Letters, Brackets, Whitespace\\nLogits set to -Infinity", fillcolor="#0F172A", color="#EF4444"];
    }

    SampledToken [label="Next Sampled Token: '2'\\nZero Syntax Divergence Possible", fillcolor="#0284C7"];

    GeneratedPrefix -> EngineCheck;
    EngineCheck -> AllowedTokens;
    EngineCheck -> MaskedTokens;
    AllowedTokens -> SampledToken;
}
"""
st.graphviz_chart(engine_dot, use_container_width=True)

st.markdown("""
Because invalid syntax tokens are masked to negative infinity before sampling, **syntax errors, unescaped quotes, 
and truncated keys are eliminated at the physical sampling layer**.
""")

st.divider()

# --- SECTION 5: STRUCTURED AGENT RESPONSES ---
st.header("5. Structured Agent Responses (Enforcing Typed Outputs from Agents)", divider="rainbow")

st.markdown("""
In autonomous agent architectures, agents cycle through multi-step reasoning and tool calls. 
However, the agent's **final output** often needs to be consumed by an API rather than displayed as free-form chat.

You can enforce structured agent outputs by defining a dedicated `FinalResponse` schema and binding it as the 
terminal response contract.
""")

st.code('''from pydantic import BaseModel, Field
from typing import List

class AgentResearchResult(BaseModel):
    query: str = Field(description="The original user research inquiry")
    key_findings: List[str] = Field(description="Bullet points of verified factual discoveries")
    sources_consulted: List[str] = Field(description="URLs or document IDs referenced")
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0")

# In agent orchestration (e.g. LangGraph or LCEL Agent):
# The final node binds the structured model to produce the terminal payload:
terminal_node = (
    prompt 
    | model.with_structured_output(AgentResearchResult)
)
''', language="python")

st.divider()

# --- SECTION 6: HANDLING VALIDATION FAILURES ---
st.header("6. Handling Validation Failures (Retries, Schema Relaxation & Resiliency)", divider="rainbow")

st.markdown("""
While native structured output eliminates syntactic errors, semantic validation failures can still occur 
if custom `@field_validator` constraints fail (e.g., regex checks or cross-field validation rules).

To ensure production resiliency:
- Use `.with_retry()` to retry on validation failures.
- Provide clear `description` attributes on fields that frequently fail.
- Use `.with_fallbacks()` with a relaxed schema if the primary schema fails repeatedly.
""")

st.code('''from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

# Primary model with strict validation
strict_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(StrictSchema)

# Fallback model with relaxed schema
relaxed_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RelaxedSchema)

# Combine using LCEL fallbacks
resilient_structured_pipeline = strict_llm.with_fallbacks([relaxed_llm])
''', language="python")

st.divider()

# --- SECTION 7: MULTIPLE STRUCTURED OUTPUTS ---
st.header("7. Multiple Structured Outputs (Typed Unions & Dynamic Schemas)", divider="rainbow")

st.markdown("""
Often an application must classify and extract one of **several different structured types** depending on input context. 
LangChain supports passing a `Union[SchemaA, SchemaB]` to `with_structured_output()`. The model will select the 
appropriate schema and return the matching instance!
""")

st.code('''from typing import Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class BugReport(BaseModel):
    type: str = Field(default="bug", description="Discriminator field")
    error_message: str = Field(description="The error or exception text")
    severity: str = Field(description="LOW, HIGH, CRITICAL")

class FeatureRequest(BaseModel):
    type: str = Field(default="feature", description="Discriminator field")
    feature_name: str = Field(description="Name of the proposed capability")
    user_benefit: str = Field(description="Value proposition for users")

# Union of multiple schemas
MultiIntentSchema = Union[BugReport, FeatureRequest]

model = ChatOpenAI(model="gpt-4o", temperature=0)
multi_router_llm = model.with_structured_output(MultiIntentSchema)

# Test 1: Model automatically returns BugReport instance
result1 = multi_router_llm.invoke("Database connection pool threw TimeoutException in auth pod")
print(type(result1))  # <class '__main__.BugReport'>

# Test 2: Model automatically returns FeatureRequest instance
result2 = multi_router_llm.invoke("We need dark mode support in the dashboard settings")
print(type(result2))  # <class '__main__.FeatureRequest'>
''', language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Universal Structured Output Service", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates a hardened enterprise extraction microservice 
using `with_structured_output`, multi-provider support, fallback models, and full Pydantic typing.
""")

st.code('''import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

logger = logging.getLogger("StructuredOutputService")

class EnterpriseDocumentSummary(BaseModel):
    title: str = Field(description="Synthesized canonical document title")
    author: Optional[str] = Field(default=None, description="Author or organization if identified")
    key_takeaways: List[str] = Field(description="3 to 5 core strategic takeaways")
    action_items: List[str] = Field(description="Concrete operational next steps")
    sentiment: str = Field(description="Overall tone: POSITIVE, NEUTRAL, URGENT")

class ProductionExtractionPipeline:
    """Production structured extraction pipeline with cross-provider failover."""

    def __init__(self):
        # Primary: OpenAI GPT-4o with Native JSON Schema Structured Output
        primary_model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.primary_structured = primary_model.with_structured_output(
            EnterpriseDocumentSummary,
            method="json_schema"
        )

        # Fallback: Anthropic Claude 3.5 Sonnet with Tool-Based Structured Output
        fallback_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)
        self.fallback_structured = fallback_model.with_structured_output(
            EnterpriseDocumentSummary,
            method="function_calling"
        )

        # Resilient Model Pipeline
        self.resilient_model = self.primary_structured.with_fallbacks([self.fallback_structured])

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an executive document intelligence system. Extract structured strategic summaries."),
            ("human", "Raw Document Text:\\n{document_text}")
        ])

        self.pipeline: RunnableSequence = self.prompt | self.resilient_model

    def extract(self, document_text: str) -> Dict[str, Any]:
        logger.info(f"Extracting structured summary for text length {len(document_text)} chars...")
        try:
            summary: EnterpriseDocumentSummary = self.pipeline.invoke({"document_text": document_text})
            return {
                "status": "SUCCESS",
                "summary": summary.model_dump(),
                "key_takeaways_count": len(summary.key_takeaways)
            }
        except Exception as e:
            logger.error(f"Structured extraction failed across all providers: {e}", exc_info=True)
            return {
                "status": "FAILED",
                "error": str(e)
            }
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Universal Structured Output Simulator", divider="rainbow")

st.markdown("""
Test how **Native Structured Output (`json_schema`)** and **Tool-Based Extraction (`function_calling`)** 
extract structured entities from raw natural language input.
""")

sim = MockStructuredOutputSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("📥 Input Raw Unstructured Text")
    sample_text_presets = {
        "Security Breach Alert": "CRITICAL: A remote code execution zero-day vulnerability (CVE-2026-9812) was exploited in the auth-service database layer. Immediate public disclosure required.",
        "Frustrated Customer Feedback": "Your platform is terribly slow! The dashboard crashes constantly when loading reports. Fix this immediately or I am canceling my subscription.",
        "Routine Performance Audit": "The user-db cluster experienced a mild latency spike of 210ms during weekly backup execution. All systems remained operational."
    }
    
    preset_choice = st.selectbox("Load Sample Incident / Feedback", list(sample_text_presets.keys()))
    user_input_text = st.text_area("Unstructured Input Text", value=sample_text_presets[preset_choice], height=130)

with col_wb2:
    st.subheader("⚙️ Extraction Configuration")
    chosen_schema = st.selectbox("Target Pydantic Schema", ["SecurityIncident", "CustomerFeedback"])
    chosen_mode = st.radio("Structured Output Mode", ["Native Constrained Decoding (json_schema)", "Tool-Based Calling (function_calling)"])

if st.button("⚡ Extract Structured Data", type="primary"):
    res = sim.extract_entity(user_input_text, chosen_mode, chosen_schema)
    
    st.subheader("📊 Structured Extraction Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Schema", res.schema_name)
    m2.metric("Prompt Token Overhead", f"{res.token_overhead} tokens")
    m3.metric("Extraction Latency", f"{res.latency_ms} ms")

    st.write("#### Resulting Strongly-Typed Pydantic Object (`model_dump()`):")
    st.json(res.payload)

    if res.token_overhead == 0:
        st.success("🎯 **Native Structured Output Advantage**: 0 prompt token overhead! The schema was enforced directly at the token sampling layer.")
    else:
        st.info("ℹ️ **Tool-Based Advantage**: Compatible across any model supporting tool calling, with minimal schema parameter overhead.")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does model.with_structured_output() improve upon legacy PydanticOutputParser?",
     "PydanticOutputParser injects format instructions into the prompt (costing tokens) and parses free-form text post-hoc, which can still fail if the model emits markdown or invalid syntax. model.with_structured_output() uses provider-native constrained decoding (OpenAI/Anthropic tool calling or JSON schema mode), ensuring 100% syntactic compliance at the engine level with zero prompt token waste."),
    ("What is the difference between method='json_schema' and method='function_calling'?",
     "method='json_schema' uses provider-native constrained sampling where grammar rules mask invalid tokens during sampling. method='function_calling' translates the schema into a synthetic tool definition and forces the model to emit a tool_call argument dictionary. 'json_schema' is preferred when supported, while 'function_calling' offers broader model compatibility."),
    ("Can with_structured_output() return a TypedDict instead of a Pydantic BaseModel?",
     "Yes. with_structured_output() supports Pydantic BaseModels, Python TypedDict classes, and raw JSON Schema dictionaries. If a TypedDict or raw schema is passed, it returns a standard Python dictionary conforming to that schema."),
    ("How does LangChain handle multiple possible schemas using Union types?",
     "When passing Union[SchemaA, SchemaB], LangChain translates both schemas into tool definitions (or a combined JSON schema with anyOf/oneOf). The model evaluates the input and selects the most appropriate schema, returning the corresponding instantiated Pydantic object."),
    ("What happens if a model receives a schema with fields that are not supported by constrained decoding?",
     "Certain complex JSON Schema features (such as patternProperties or unbounded recursion) are not supported by native constrained decoding engines. In such cases, the API will raise an error, and LangChain will recommend falling back to method='function_calling' or PydanticOutputParser.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Assuming All Models Support Native JSON Schema. Trying to use method='json_schema' on older or open-source models that only support basic text completion will raise an UnsupportedFeature exception. Always use method='function_calling' or test model compatibility.")
    st.error("**Trap 2 (Breaking Error)**: Forgetting Optional/Nullable in Constrained Schemas. In native structured outputs (like OpenAI), all fields are strictly required in the schema unless marked as Optional with a null type. Omitting a field from the required list in strict mode causes an API validation error.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Neglecting System Prompts in Structured Calls. Relying solely on the schema without an informative system prompt can result in valid but semantically low-quality extractions. Always combine with_structured_output() with clear contextual system instructions.")
    st.warning("**Trap 4 (Latency Warning)**: Giant Schemas in Tool Calling Mode. In method='function_calling', very large schemas with dozens of nested fields must be serialized into the tool definition parameter on every request, adding significant latency. Keep extraction schemas focused.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: with_structured_output()"):
        st.markdown("**The unified LangChain interface for extracting typed Pydantic models from LLMs.**")
    with st.expander("🎴 Flashcard 2: json_schema vs function_calling"):
        st.markdown("**json_schema uses native grammar constraints; function_calling uses tool call emulation.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Zero Prompt Token Overhead"):
        st.markdown("**Native structured outputs enforce schemas via API parameters without polluting the prompt.**")
    with st.expander("🎴 Flashcard 4: Typed Unions in Structured Output"):
        st.markdown("**Passing Union[SchemaA, SchemaB] allows the model to dynamically choose the appropriate output schema.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can implement model.with_structured_output() with Pydantic schemas.", value=True)
chk2 = st.checkbox("I understand the difference between native JSON mode and tool-based extraction.", value=True)
chk3 = st.checkbox("I know how constrained decoding eliminates syntax and delimiter errors at the engine layer.", value=True)
chk4 = st.checkbox("I can configure autonomous agents to return strongly-typed Pydantic responses.", value=True)
chk5 = st.checkbox("I understand how to handle multiple schemas using Union types.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 structured output interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 7.4 Structured Output and completed all of Chapter 7: Output Parsing and Structured Output!")
