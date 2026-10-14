import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="7.2 Output Parsers - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("7.2 Output Parsers: The Core Parsers, Pydantic Models & Custom Parsers")
st.caption("Chapter 7: Output Parsing and Structured Output | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In LangChain, **Output Parsers** are first-class LCEL Runnables that transform model generation outputs into 
strongly-typed Python data structures. Rather than writing ad-hoc string splitting or regex logic, LangChain 
provides an extensible hierarchy of parsers tailored for every engineering use-case:

In this lesson, you will master:
- **`StrOutputParser`**: High-performance string extraction with streaming token support.
- **`JsonOutputParser`**: Structural JSON deserialization with markdown fence stripping.
- **`PydanticOutputParser`**: Type-safe validation using Pydantic data schemas.
- **`StructuredOutputParser`**: Lightweight multi-attribute extraction using `ResponseSchema`.
- **Custom Parsers**: Building specialized parsers by subclassing `BaseOutputParser[T]`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockParserOutcome:
    """Represents parser evaluation outcome in module scope."""
    def __init__(self, parser_name: str, target_type: str, output_repr: str, success: bool, message: str):
        self.parser_name = parser_name
        self.target_type = target_type
        self.output_repr = output_repr
        self.success = success
        self.message = message

class MockMultiParserLab:
    """Simulates the behavior of 4 primary LangChain parsers in module scope."""
    def __init__(self):
        pass

    def run_parser_test(self, parser_choice: str, raw_text: str) -> MockParserOutcome:
        raw_clean = raw_text.strip()

        if parser_choice == "StrOutputParser":
            return MockParserOutcome(
                "StrOutputParser",
                "str",
                repr(raw_clean),
                True,
                f"Successfully extracted string of length {len(raw_clean)} chars."
            )

        elif parser_choice == "JsonOutputParser":
            # Attempt to strip fences
            cleaned = raw_clean
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            try:
                import json
                data = json.loads(cleaned)
                return MockParserOutcome(
                    "JsonOutputParser",
                    type(data).__name__,
                    json.dumps(data, indent=2),
                    True,
                    "JSON successfully parsed into native Python dictionary."
                )
            except Exception as e:
                return MockParserOutcome(
                    "JsonOutputParser",
                    "None",
                    f"Error: {str(e)}",
                    False,
                    f"JSONDecodeError: Failed to parse into valid JSON."
                )

        elif parser_choice == "PydanticOutputParser (UserModel: id, name, email)":
            cleaned = raw_clean
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            try:
                import json
                data = json.loads(cleaned)
                if not isinstance(data, dict):
                    raise ValueError("Target must be a JSON object")
                
                # Verify required fields
                if "id" not in data or "name" not in data or "email" not in data:
                    missing = [f for f in ["id", "name", "email"] if f not in data]
                    raise ValueError(f"Missing required fields: {missing}")

                # Type check
                if not isinstance(data["id"], int):
                    raise ValueError(f"Field 'id' must be integer, got {type(data['id']).__name__}")
                if "@" not in str(data["email"]):
                    raise ValueError(f"Field 'email' failed email validation pattern: {data['email']}")

                repr_str = f"UserModel(id={data['id']}, name='{data['name']}', email='{data['email']}')"
                return MockParserOutcome(
                    "PydanticOutputParser",
                    "UserModel (Pydantic)",
                    repr_str,
                    True,
                    "Schema validated and instantiated into UserModel instance."
                )
            except Exception as e:
                return MockParserOutcome(
                    "PydanticOutputParser",
                    "ValidationError",
                    f"Validation Failure: {str(e)}",
                    False,
                    f"Pydantic schema validation rejected output."
                )

        elif parser_choice == "Custom CSV/List Parser (Line Delimited)":
            lines = [line.strip().lstrip("-*123456789. ") for line in raw_clean.splitlines() if line.strip()]
            return MockParserOutcome(
                "CustomListParser",
                "List[str]",
                str(lines),
                True,
                f"Parsed {len(lines)} distinct item entries from line-delimited text."
            )

        return MockParserOutcome("Unknown", "None", "", False, "Parser not recognized")

# --- SECTION 1: ARCHITECTURE BLUEPRINT & CLASS ECOSYSTEM ---
st.header("1. Architecture Blueprint: The Output Parser Hierarchy & Class Ecosystem", divider="rainbow")

st.markdown("""
LangChain's output parser framework is built upon the unified `BaseOutputParser[T]` abstract class, 
which extends `RunnableSerializable[Union[str, BaseMessage], T]`. This ensures every parser is 
a fully functional LCEL component supporting invocation, batching, and streaming.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Root Interface
        subgraph cluster_root {
            label = "1. LCEL BASE INTERFACE TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            Runnable [label="Runnable[Input, Output]\\ninvoke(), batch(), stream(), astream()"];
            BaseOutputParser [label="BaseOutputParser[T]\\nabstract parse(text: str) -> T\\nabstract get_format_instructions() -> str", color="#10B981"];
        }

        // Standard Builtin Parsers
        subgraph cluster_builtins {
            label = "2. CORE STANDARD IMPLEMENTATIONS TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            StrParser [label="StrOutputParser\\nOutput: str\\nExtracts AIMessage.content cleanly"];
            JsonParser [label="JsonOutputParser\\nOutput: Union[dict, list]\\nStrips ```json &amp; decodes JSON"];
            PydanticParser [label="PydanticOutputParser[T]\\nOutput: BaseModel Instance\\nStrict type assertion &amp; coercion", color="#38BDF8"];
            StructuredParser [label="StructuredOutputParser\\nOutput: Dict[str, Any]\\nDriven by List[ResponseSchema]"];
        }

        // Custom Extensibility Tier
        subgraph cluster_custom {
            label = "3. CUSTOM EXTENSION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            CustomBase [label="Subclass BaseOutputParser[T]\\nOverride .parse() &amp; .get_format_instructions()"];
            CustomCSV [label="CommaSeparatedListOutputParser\\nOutput: List[str]"];
            CustomDomain [label="DomainSpecificParser (e.g. SQL / Diff / AST)\\nOutput: Domain AST Object", color="#F59E0B"];
        }

        // Connections
        Runnable -> BaseOutputParser [label="inherits from"];
        BaseOutputParser -> StrParser;
        BaseOutputParser -> JsonParser;
        BaseOutputParser -> PydanticParser;
        BaseOutputParser -> StructuredParser;
        BaseOutputParser -> CustomBase;
        
        CustomBase -> CustomCSV;
        CustomBase -> CustomDomain;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Architecture Insights:**
- **Generic Type Parametrization**: `BaseOutputParser[T]` uses Python generics (`T`), guaranteeing IDE type hinting and static type checker verification.
- **Bi-Directional Contract**: Every parser provides both parsing logic (`.parse(text)`) and format instructions (`.get_format_instructions()`) for prompt injection.
""")

st.divider()

# --- SECTION 2: STRING OUTPUT PARSER ---
st.header("2. String Output Parser (`StrOutputParser`)", divider="rainbow")

st.markdown("""
`StrOutputParser` is the most widely utilized output parser in LangChain. 
It extracts raw strings from `AIMessage`, `HumanMessage`, `ChatMessage`, or `str` inputs.
""")

st.subheader("Key Capabilities of `StrOutputParser`")
st.markdown("""
1. **Streaming Token Passthrough**: When used with `.stream()` or `.astream()`, `StrOutputParser` directly streams token strings without buffering the entire completion.
2. **Type Simplification**: Eliminates the need for manual attribute lookups (`response.content`), allowing subsequent runnables in LCEL to consume clean text.
3. **Payload Sanitization**: Discards model metadata (token usage, finish reason, logprobs) when only the semantic response is needed.
""")

st.code('''from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

parser = StrOutputParser()

# 1. Parsing a completed AIMessage
msg = AIMessage(content="Microservices decouple deployment cycles.")
output_str = parser.invoke(msg)
print(type(output_str), output_str)
# <class 'str'> 'Microservices decouple deployment cycles.'

# 2. Piping inside LCEL
chain = prompt | model | parser
''', language="python")

st.divider()

# --- SECTION 3: JSON OUTPUT PARSER ---
st.header("3. JSON Output Parser (`JsonOutputParser`)", divider="rainbow")

st.markdown("""
`JsonOutputParser` deserializes LLM responses into Python dictionaries or lists. 
Unlike standard Python `json.loads()`, `JsonOutputParser` includes built-in tolerance for LLM syntax patterns:
- It automatically strips markdown code fences (` ```json ... ``` `).
- It ignores introductory text before the first opening brace `{` or bracket `[`.
- It supports partial streaming parsing via `parse_result()`.
""")

st.code('''from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

json_parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract server telemetry. {format_instructions}"),
    ("human", "{log_entry}")
]).partial(format_instructions=json_parser.get_format_instructions())

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
telemetry_chain = prompt | model | json_parser

result = telemetry_chain.invoke({
    "log_entry": "2026-10-13 14:02:11 [WARN] CPU utilization on worker-node-04 reached 91.2% with memory at 64%."
})

print(type(result))  # <class 'dict'>
print(result)
# Output: {'timestamp': '2026-10-13 14:02:11', 'level': 'WARN', 'host': 'worker-node-04', 'cpu_percent': 91.2, 'memory_percent': 64.0}
''', language="python")

st.divider()

# --- SECTION 4: PYDANTIC OUTPUT PARSER ---
st.header("4. Pydantic Output Parser (`PydanticOutputParser`)", divider="rainbow")

st.markdown("""
`PydanticOutputParser` is the industry standard for type-safe LLM outputs. 
By binding a Pydantic `BaseModel` class to the parser, you guarantee that:
1. Every field in the generated output adheres to expected data types (e.g., `int`, `float`, `List[str]`).
2. Custom field validation rules (e.g., regex patterns, numerical bounds) are strictly evaluated.
3. The parser returns a live Pydantic instance with dot-notation field access (`result.user_id`).
""")

st.code('''from pydantic import BaseModel, Field, EmailStr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Define the Pydantic Schema Contract
class UserProfile(BaseModel):
    user_id: int = Field(description="Unique numeric database user ID")
    full_name: str = Field(description="Full name of user")
    email: str = Field(description="Valid email address")
    roles: list[str] = Field(default_factory=list, description="Assigned authorization roles")
    is_active: bool = Field(default=True, description="Account active status")

# 2. Instantiate PydanticOutputParser
pydantic_parser = PydanticOutputParser(pydantic_object=UserProfile)

# 3. Inject Format Instructions into System Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated user directory parser.\\n{format_instructions}"),
    ("human", "Profile record: {raw_text}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
user_chain = prompt | model | pydantic_parser

# 4. Invocation returns strongly-typed UserProfile instance
profile: UserProfile = user_chain.invoke({
    "raw_text": "Create record for Sarah Connor (sarah@resistance.org), assigned to admin and secops roles with ID 849."
})

print(profile.full_name)   # 'Sarah Connor'
print(profile.roles)       # ['admin', 'secops']
print(type(profile))       # <class 'UserProfile'>
''', language="python")

st.divider()

# --- SECTION 5: STRUCTURED OUTPUT PARSER ---
st.header("5. Structured Output Parser (`StructuredOutputParser` & `ResponseSchema`)", divider="rainbow")

st.markdown("""
When you need structured multi-field outputs without defining formal Pydantic classes (or in lightweight 
environments where Pydantic is not a dependency), LangChain provides `StructuredOutputParser` combined with `ResponseSchema`.
""")

st.code('''from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Define Response Schemas for individual target fields
response_schemas = [
    ResponseSchema(name="sentiment", description="The tone of user message: Positive, Neutral, or Negative"),
    ResponseSchema(name="urgency_score", description="An integer from 1 to 5 indicating urgency"),
    ResponseSchema(name="suggested_action", description="A concrete 1-sentence recommendation for support agent")
]

# 2. Instantiate StructuredOutputParser
structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. Inject Format Instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the incoming customer support ticket.\\n{format_instructions}"),
    ("human", "Ticket Message: {ticket_text}")
]).partial(format_instructions=structured_parser.get_format_instructions())

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
support_chain = prompt | model | structured_parser

result = support_chain.invoke({
    "ticket_text": "I was billed twice for my subscription this morning and need an immediate refund!"
})

print(result)
# Output: {'sentiment': 'Negative', 'urgency_score': '5', 'suggested_action': 'Issue immediate refund and verify payment gateway webhook.'}
''', language="python")

st.divider()

# --- SECTION 6: CUSTOM PARSERS ---
st.header("6. Custom Parsers (Subclassing `BaseOutputParser[T]`)", divider="rainbow")

st.markdown("""
When none of the built-in parsers meet your specific domain requirements (e.g., parsing raw diff patches, 
validating SQL ASTs, or extracting custom delimiter syntax), you can create a custom parser by subclassing `BaseOutputParser[T]`.
""")

st.subheader("Implementing a Custom Comma-Separated Key-Value Parser")
st.code('''from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any

class KeyValueOutputParser(BaseOutputParser[Dict[str, str]]):
    """Custom parser that transforms 'key=value' line-delimited outputs into Python dictionaries."""

    def parse(self, text: str) -> Dict[str, str]:
        results = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                results[k.strip()] = v.strip().strip("\'\"")
        return results

    def get_format_instructions(self) -> str:
        return (
            "Return the output as a list of key=value pairs, one per line.\\n"
            "Do NOT include JSON, markdown fences, or conversational text.\\n"
            "Example:\\n"
            "status=SUCCESS\\n"
            "latency_ms=45\\n"
            "region=us-east-1"
        )

# Direct LCEL usage
custom_parser = KeyValueOutputParser()
chain = prompt | model | custom_parser
''', language="python")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Multi-Parser Factory & Fallback Engine", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates a **Dynamic Multi-Parser Engine** that 
validates incoming user submissions with Pydantic, intercepts formatting exceptions, and automatically 
falls back to a secondary repair strategy.
""")

st.code('''import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

logger = logging.getLogger("ParserEngine")

class SecurityIncidentReport(BaseModel):
    incident_id: str = Field(description="Unique identifier e.g. INC-9481")
    severity: str = Field(description="One of: LOW, MEDIUM, HIGH, CRITICAL")
    affected_services: list[str] = Field(description="List of impacted microservices")
    mitigation_steps: list[str] = Field(description="Actionable steps to contain threat")

class EnterpriseParserPipeline:
    """Production pipeline featuring strict Pydantic validation and graceful fallback handling."""
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.pydantic_parser = PydanticOutputParser(pydantic_object=SecurityIncidentReport)
        self.fallback_json_parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a cybersecurity incident triage engineer.
Extract security incident report details from the incident transcript.
{format_instructions}"""),
            ("human", "Incident Raw Log:\\n{raw_log}")
        ]).partial(format_instructions=self.pydantic_parser.get_format_instructions())

        self.primary_chain = self.prompt | self.model | self.pydantic_parser

    def process_incident(self, raw_log: str) -> Dict[str, Any]:
        try:
            # Primary path: Returns validated Pydantic model
            report: SecurityIncidentReport = self.primary_chain.invoke({"raw_log": raw_log})
            logger.info(f"Successfully parsed incident: {report.incident_id}")
            return {
                "status": "VALIDATED",
                "data": report.model_dump(),
                "validation_engine": "Pydantic"
            }
        except (OutputParserException, ValidationError) as err:
            logger.warning(f"Pydantic parsing failed: {err}. Attempting unvalidated JSON fallback...")
            
            # Secondary fallback: Extract raw JSON without crashing application
            try:
                raw_json = self.fallback_json_parser.invoke(raw_log)
                return {
                    "status": "PARTIAL_RECOVERY",
                    "data": raw_json,
                    "validation_engine": "JsonFallback",
                    "warning": str(err)
                }
            except Exception as fallback_err:
                logger.error(f"Fallback parsing also failed: {fallback_err}")
                return {
                    "status": "UNPARSEABLE",
                    "data": None,
                    "error": str(fallback_err)
                }
''', language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Interactive Multi-Parser Testing Lab", divider="rainbow")

st.markdown("""
Test how the four primary parsing architectures handle different raw output string structures. 
Experiment with well-formed, fenced, and invalid payloads to observe parser error handling in action.
""")

lab = MockMultiParserLab()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🧪 Select Parser & Input Text")
    parser_types = [
        "StrOutputParser",
        "JsonOutputParser",
        "PydanticOutputParser (UserModel: id, name, email)",
        "Custom CSV/List Parser (Line Delimited)"
    ]
    chosen_parser = st.selectbox("Choose Target Parser Engine", parser_types)

    default_presets = {
        "StrOutputParser": "Microservices architectures enable autonomous deployments across engineering squads.",
        "JsonOutputParser": '```json\n{\n  "status": "SUCCESS",\n  "cluster": "us-east-prod",\n  "active_nodes": 16\n}\n```',
        "PydanticOutputParser (UserModel: id, name, email)": '{\n  "id": 402,\n  "name": "Alex Mercer",\n  "email": "alex@prototype.dev"\n}',
        "Custom CSV/List Parser (Line Delimited)": "- Python\n- Rust\n- Go\n- TypeScript\n- Elixir"
    }

    test_input = st.text_area("Raw Generation Text to Parse", value=default_presets[chosen_parser], height=160)

with col_wb2:
    st.subheader("⚙️ Parser Capabilities Overview")
    if chosen_parser == "StrOutputParser":
        st.info("**StrOutputParser**: Extracts `.content` as plain string. Zero schema validation. Full streaming support.")
    elif chosen_parser == "JsonOutputParser":
        st.info("**JsonOutputParser**: Deserializes text into native `dict` or `list`. Automatically strips markdown fences.")
    elif chosen_parser.startswith("Pydantic"):
        st.info("**PydanticOutputParser**: Enforces strict types (`id: int`, `email: str`). Rejects missing fields or invalid types.")
    else:
        st.info("**Custom Parser**: Parses line-delimited bullet points into a clean `List[str]`.")

if st.button("🚀 Run Parser Execution", type="primary"):
    res = lab.run_parser_test(chosen_parser, test_input)
    
    st.subheader("📊 Parser Execution Results")
    m1, m2 = st.columns(2)
    m1.metric("Parser Execution Status", "PASSED" if res.success else "FAILED")
    m2.metric("Target Output Type", res.target_type)

    if res.success:
        st.success(res.message)
        st.write("#### Parsed Representation:")
        st.code(res.output_repr, language="python" if "UserModel" in res.output_repr or "List" in res.target_type else "json")
    else:
        st.error(res.message)
        st.code(res.output_repr, language="text")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does PydanticOutputParser differ fundamentally from JsonOutputParser?",
     "JsonOutputParser only checks syntactic validity (is the text valid JSON?) and returns an untyped Python dictionary. PydanticOutputParser enforces both syntactic validity AND strict semantic schema validation (type checking, field boundary constraints, default values), and returns a validated Pydantic model instance with dot-notation field access."),
    ("Why is StructuredOutputParser less commonly used in modern LangChain compared to PydanticOutputParser?",
     "StructuredOutputParser relies on manual ResponseSchema definitions and basic dictionary parsing. PydanticOutputParser integrates with Python's modern typing system, supports complex nested schemas, custom validators, and enum types, and has first-class IDE autocomplete support."),
    ("What methods must be implemented when creating a custom parser by subclassing BaseOutputParser[T]?",
     "At minimum, you must implement the abstract method parse(self, text: str) -> T. To allow format instruction injection into prompts, you should also override get_format_instructions(self) -> str."),
    ("Can output parsers be used in streaming workflows?",
     "Yes. BaseOutputParser implements the Runnable interface. Parsers like StrOutputParser stream individual string chunks as they arrive. More advanced parsers like JsonOutputParser support partial parsing, yielding partial JSON structures during streaming."),
    ("What is OutputFixingParser and how does it prevent pipeline failures?",
     "OutputFixingParser wraps a base parser and a chat model. If the base parser fails due to malformed output or a schema violation, OutputFixingParser automatically sends the malformed text and the error message to an LLM with a correction prompt, giving the model a second chance to fix the format.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Instantiating Pydantic Models with Raw Dictionaries Without Validation. Bypassing the parser and manually calling Model(**json_dict) can crash your API if the LLM omitted an optional field. Always use PydanticOutputParser or model_validate().")
    st.error("**Trap 2 (Breaking Error)**: Unhandled Validation Errors in Synchronous Endpoints. When Pydantic validation fails, PydanticOutputParser raises OutputParserException. If uncaught, this results in an unhandled HTTP 500 error for clients. Always wrap parsers with exception handlers or fallbacks.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant Pydantic Format Instructions. Complex Pydantic schemas can generate format instructions exceeding 500 tokens. When using modern models supporting native JSON mode or function calling, consider model.with_structured_output() to eliminate prompt token overhead.")
    st.warning("**Trap 4 (Quality Warning)**: LLM Escaping Markdown Inside JSON Strings. Models sometimes generate valid JSON where string fields contain raw newline or quote characters that break parser tokenizers. Always sanitize escaped characters before passing to strict parsers.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: PydanticOutputParser Return Type"):
        st.markdown("**Returns an instantiated, strongly-typed Pydantic BaseModel object with dot notation.**")
    with st.expander("🎴 Flashcard 2: BaseOutputParser[T] Core Method"):
        st.markdown("**The required abstract method to implement is .parse(text: str) -> T.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: StructuredOutputParser Limitation"):
        st.markdown("**Returns plain dictionaries; lacks strong typing and complex nested field validation.**")
    with st.expander("🎴 Flashcard 4: OutputFixingParser"):
        st.markdown("**Uses an auxiliary LLM call to automatically repair malformed generation outputs.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I know when to choose between StrOutputParser, JsonOutputParser, and PydanticOutputParser.", value=True)
chk2 = st.checkbox("I can bind a Pydantic BaseModel to PydanticOutputParser and inject format instructions.", value=True)
chk3 = st.checkbox("I understand how StructuredOutputParser uses ResponseSchema objects.", value=True)
chk4 = st.checkbox("I can build a custom parser by subclassing BaseOutputParser[T].", value=True)
chk5 = st.checkbox("I understand how to implement fallback and recovery mechanisms for malformed outputs.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 output parser interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 7.2 Output Parsers. Ready for 7.3 Pydantic Integration!")
