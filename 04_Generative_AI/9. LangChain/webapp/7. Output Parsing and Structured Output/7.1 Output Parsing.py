import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="7.1 Output Parsing - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("7.1 Output Parsing: Bridges from Probabilistic Tokens to Deterministic Types")
st.caption("Chapter 7: Output Parsing and Structured Output | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Large Language Models are probabilistic autoregressive token generators. By default, they return 
free-form strings wrapped inside message containers (`AIMessage.content`). However, downstream enterprise 
software systems—databases, REST APIs, microservices, and frontend web applications—require **deterministic, 
strictly validated, strongly-typed data structures**.

**Output Parsing** is the critical boundary layer that deserializes, validates, and transforms raw generative 
tokens into reliable application payloads.

In this lesson, you will master:
- **The Core Problem**: Why downstream automation fails when relying on raw LLM string generation.
- **String Parsing**: Extracting raw string content cleanly and streaming tokens through LCEL pipelines.
- **JSON Parsing**: Stripping markdown fences, repairing truncated brackets, and decoding into Python dictionaries.
- **Structured Data Transformation**: Mapping probabilistic answers into predictable schemas.
- **Output Validation**: Implementing runtime schema assertions and handling structural decoding failures.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockParseResult:
    """Represents the outcome of a parsing and validation execution in module scope."""
    def __init__(self, raw_input: str, parsed_data: Any, is_valid: bool, error_msg: Optional[str] = None):
        self.raw_input = raw_input
        self.parsed_data = parsed_data
        self.is_valid = is_valid
        self.error_msg = error_msg

class MockOutputParsingSimulator:
    """Simulates multi-format output parsing and schema validation in module scope."""
    def __init__(self):
        pass

    def parse_string(self, text: str) -> MockParseResult:
        cleaned = text.strip()
        # Remove common markdown artifact quotes if present
        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1]).strip()
        return MockParseResult(text, cleaned, True)

    def parse_json(self, text: str, required_keys: List[str]) -> MockParseResult:
        cleaned = text.strip()
        # Remove markdown fences like ```json ... ```
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                cleaned = parts[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) > 1:
                cleaned = parts[1].split("```")[0].strip()

        # Simple json parsing simulator
        try:
            import json
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                return MockParseResult(text, None, False, f"Parsed JSON must be an object/dict, got {type(data).__name__}")
            
            # Check required keys
            missing = [k for k in required_keys if k not in data]
            if missing:
                return MockParseResult(text, data, False, f"Schema validation failed: missing required keys: {missing}")
            
            return MockParseResult(text, data, True)
        except Exception as e:
            return MockParseResult(text, None, False, f"JSONDecodeError: {str(e)}")

# --- SECTION 1: ARCHITECTURE BLUEPRINT & DESERIALIZATION FLOW ---
st.header("1. Architecture Blueprint: The Output Parsing Pipeline & Deserialization Flow", divider="rainbow")

st.markdown("""
The output parsing layer sits immediately downstream of the model invocation. It intercepts raw 
generation artifacts (`ChatResult` or `AIMessage`), cleans delimiter syntax (such as markdown code blocks), 
deserializes text into target memory structures, and enforces schema conformance contracts.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Tier 1: Probabilistic Model Generation
        subgraph cluster_model {
            label = "1. PROBABILISTIC MODEL TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserPrompt [label="ChatPromptTemplate\\nFormatted Prompt Instructions"];
            ChatModel [label="ChatModel (e.g., GPT-4o / Claude 3.5)\\nAutoregressive Next-Token Sampling"];
            RawAIMessage [label="AIMessage Object\\ncontent='```json\\n{\"name\": \"Alice\"}\\n```'"];
        }

        // Tier 2: Extraction & Sanitization
        subgraph cluster_sanitization {
            label = "2. EXTRACTION &amp; SANITIZATION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            ContentExtractor [label="Extract .content string\\nDiscard token metadata"];
            MarkdownStripper [label="Fence Stripper &amp; Whitespace Trimmer\\nRemove ```json and trailing whitespace"];
        }

        // Tier 3: Deserialization Engine
        subgraph cluster_deserialization {
            label = "3. DESERIALIZATION ENGINE (RunnableParser)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ParserRouter [label="Parser Implementation Decision\\nStr | Json | Pydantic | Custom"];
            JsonDecoder [label="JSON Syntax Deserializer\\njson.loads() / partial_json_parser"];
        }

        // Tier 4: Validation & Downstream
        subgraph cluster_validation {
            label = "4. VALIDATION &amp; CONSUMPTION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            SchemaValidator [label="Schema Conformance Guard\\nAssert required keys and data types"];
            SuccessState [label="Validated Output Object\\nTyped Dict / Pydantic Model", color="#10B981"];
            FailureHandler [label="OutputParserException\\nAuto-Retry / Formatting Fix Prompt", color="#EF4444"];
        }

        // Connections
        UserPrompt -> ChatModel;
        ChatModel -> RawAIMessage;
        RawAIMessage -> ContentExtractor;
        ContentExtractor -> MarkdownStripper;
        MarkdownStripper -> ParserRouter;
        
        ParserRouter -> JsonDecoder [label="JSON / Schema Mode"];
        JsonDecoder -> SchemaValidator [label="Syntax Valid"];
        JsonDecoder -> FailureHandler [label="Syntax Error (Malformed)", color="#EF4444", style=dashed];
        
        SchemaValidator -> SuccessState [label="Passed Schema Contract", color="#10B981"];
        SchemaValidator -> FailureHandler [label="Missing Required Keys", color="#EF4444", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Key Pipeline Axioms:**
1. **Separation of Concerns**: The LLM handles semantic reasoning; the parser handles syntactic validation.
2. **Defensive Parsing**: Real parsers anticipate common LLM quirks, such as conversational conversational prefixes ("Here is your JSON:").
3. **Fail-Fast Semantics**: If model output violates a critical schema, the parser raises an explicit `OutputParserException` instead of allowing silent downstream data corruption.
""")

st.divider()

# --- SECTION 2: WHY OUTPUT PARSING IS REQUIRED ---
st.header("2. Why Output Parsing Is Required", divider="rainbow")

st.markdown("""
To understand the necessity of output parsers, consider what happens when a software engineer attempts to 
integrate an LLM into an automated system without one:
""")

col_prob1, col_prob2 = st.columns(2)
with col_prob1:
    st.error("❌ Without Output Parsing (Fragile)")
    st.markdown("""
    - **Conversational Bloat**: The model prepends *'Sure! Here is the data you requested:'* before the payload.
    - **Markdown Code Fences**: The text is wrapped in ```json ... ``` which causes `json.loads()` to crash immediately.
    - **Inconsistent Keys**: The model uses `"user_age"` in one call and `"age_years"` in the next.
    - **Silent Failures**: Downstream microservices crash with `KeyError` or `TypeError` during peak production traffic.
    """)

with col_prob2:
    st.success("✅ With Output Parsing (Robust)")
    st.markdown("""
    - **Deterministic Type Contracts**: Guarantees output type is always a Python `str`, `dict`, `list`, or Pydantic instance.
    - **Format Instructions Injection**: Parsers can automatically inject precise structural format instructions into prompts.
    - **Automatic Stream Support**: LCEL parsers support streaming chunks as they arrive from the model.
    - **Standardized Retry Hooks**: Seamlessly integrates with `RetryWithErrorOutputParser` to fix malformed outputs automatically.
    """)

st.markdown("""
### The Fundamental Impedance Mismatch
In classical software engineering, functions obey strict static typing (`Callable[[InputType], OutputType]`). 
LLMs, however, operate in the domain of probability distributions over token vocabularies. 
Output parsers serve as the **type-safety bridge**, translating probabilistic natural language into deterministic runtime objects.
""")

st.divider()

# --- SECTION 3: STRING OUTPUT ---
st.header("3. String Output (`StrOutputParser`)", divider="rainbow")

st.markdown("""
The most fundamental parser in modern LangChain is `StrOutputParser`. While seemingly simple, it performs 
two vital architectural functions in LCEL:
1. **Content Extraction**: It extracts the `.content` attribute from `AIMessage` or `ChatResult` objects, converting complex objects into clean `str` instances.
2. **Streaming Passthrough**: Unlike manual string conversions (`str(response)`), `StrOutputParser` implements LCEL streaming protocols. As the LLM generates tokens chunk-by-chunk, `StrOutputParser` immediately yields each text chunk downstream.
""")

st.subheader("Implementation & Pipe Composition")
st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Initialize Model and Parser
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Compose Chain
prompt = ChatPromptTemplate.from_template("Explain the concept of idempotency in API design in 2 sentences.")
chain = prompt | model | parser

# Standard Invocation (Returns clean str, not AIMessage)
result = chain.invoke({})
print(type(result))  # <class 'str'>
print(result)

# Streaming Invocation (Streams tokens cleanly without AIMessageChunk wrappers)
for chunk in chain.stream({}):
    print(chunk, end="", flush=True)
''', language="python")

st.info("""
🔍 **When to use `StrOutputParser`**:
Always use `StrOutputParser` for end-user chat interfaces, raw text summarization, content rewriting, 
or whenever downstream steps expect a plain Python string rather than an `AIMessage` object.
""")

st.divider()

# --- SECTION 4: JSON OUTPUT ---
st.header("4. JSON Output (`JsonOutputParser`)", divider="rainbow")

st.markdown("""
When building microservices or AI agent workflows, applications almost universally communicate via JSON. 
However, prompting an LLM with *"Return JSON"* frequently leads to parsing errors due to markdown code block wrappers 
(```json ... ```) or conversational preamble.

`JsonOutputParser` solves this by:
- Automatically stripping markdown code fences.
- Extracting the first valid JSON object or array found within the response text.
- Parsing the string into native Python dictionaries or lists.
- Providing standardized format instructions via `parser.get_format_instructions()`.
""")

st.subheader("JSON Output Pipeline Example")
st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 1. Initialize JsonOutputParser
json_parser = JsonOutputParser()

# 2. Extract standard format instructions
instructions = json_parser.get_format_instructions()

# 3. Inject instructions into the system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated extraction assistant.\\n{format_instructions}"),
    ("human", "Extract the user details: {raw_text}")
]).partial(format_instructions=instructions)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Construct LCEL Chain
json_chain = prompt | model | json_parser

# 5. Invocation returns native Python dict
result = json_chain.invoke({
    "raw_text": "Customer John Doe (email: john@acme.com) reported ticket #9481 concerning database latency."
})

print(type(result))  # <class 'dict'>
print(result)
# Output: {'name': 'John Doe', 'email': 'john@acme.com', 'ticket_id': 9481, 'issue': 'database latency'}
''', language="python")

st.divider()

# --- SECTION 5: STRUCTURED DATA ---
st.header("5. Structured Data (Schemas & Tabular Representations)", divider="rainbow")

st.markdown("""
Beyond basic key-value dictionaries, enterprise LLM pipelines often require complex structured data 
representations, such as lists of tabular records, hierarchical trees, or typed entities.
""")

data_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    RawInput [label="Input Document:\\nUnstructured Server Incident Log"];
    ParserNode [label="JsonOutputParser with Schema Contract\\nExpects: List[IncidentRecord]"];
    
    subgraph cluster_records {
        label = "Structured Python Objects in Memory";
        style = "dashed";
        color = "#334155";
        fontcolor = "#94A3B8";

        Record1 [label="Record 1:\\n{service: 'auth', error_rate: 0.12, severity: 'HIGH'}", fillcolor="#0F172A"];
        Record2 [label="Record 2:\\n{service: 'billing', error_rate: 0.01, severity: 'LOW'}", fillcolor="#0F172A"];
        Record3 [label="Record 3:\\n{service: 'db-proxy', error_rate: 0.45, severity: 'CRITICAL'}", fillcolor="#0F172A"];
    }

    RawInput -> ParserNode;
    ParserNode -> Record1;
    ParserNode -> Record2;
    ParserNode -> Record3;
}
"""
st.graphviz_chart(data_dot, use_container_width=True)

st.markdown("""
### Handling Structured Arrays and Tabular Records
When instructing models to return arrays of structured records, specify the exact list-of-objects schema. 
This allows downstream analytics pipelines (such as `pandas.DataFrame(result)`) to consume the parsed output directly 
without intermediate transformation steps.
""")

st.code('''# Direct conversion from parsed JSON to pandas DataFrame
import pandas as pd

raw_llm_json_response = [
    {"service": "auth-service", "latency_ms": 142, "status": "HEALTHY"},
    {"service": "payment-gateway", "latency_ms": 890, "status": "DEGRADED"},
    {"service": "notification-hub", "latency_ms": 65, "status": "HEALTHY"}
]

# Immediate tabular integration
df = pd.DataFrame(raw_llm_json_response)
print(df.describe())
''', language="python")

st.divider()

# --- SECTION 6: OUTPUT VALIDATION ---
st.header("6. Output Validation (Schema Conformance & Exception Handling)", divider="rainbow")

st.markdown("""
Parsing is only half the battle. Syntactically valid JSON may still be **semantically invalid** if:
- Required fields are missing.
- Values have incorrect data types (e.g., string `"twenty"` instead of integer `20`).
- Numerical values violate business domain constraints (e.g., negative price or probability > 1.0).

Robust production architectures wrap the parsing stage with schema assertions and error-recovery hooks.
""")

st.subheader("Validation Strategy Comparison")
comp_df = pd.DataFrame({
    "Parsing Strategy": ["Raw String Split", "Regex Extraction", "JsonOutputParser", "PydanticOutputParser"],
    "Syntax Tolerance": ["Zero", "Moderate", "High (Strips markdown)", "High (Strips markdown)"],
    "Type Coercion": ["None", "None", "Basic Python types", "Strict / Coerced"],
    "Schema Validation": ["None", "None", "Key checking only", "Exhaustive field validation"],
    "Error Recovery": ["Manual try/except", "Manual try/except", "OutputParserException", "Self-Correcting Retry Parser"]
})
st.dataframe(comp_df, use_container_width=True)

st.subheader("Defensive Validation Pattern in LCEL")
st.code('''from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

def enforce_schema_constraints(parsed_data: dict) -> dict:
    """Validates presence and boundaries of business critical fields."""
    required_fields = ["transaction_id", "amount_usd", "risk_score"]
    
    # 1. Existence check
    for field in required_fields:
        if field not in parsed_data:
            raise OutputParserException(f"Schema violation: missing required field '{field}'")
            
    # 2. Type & Boundary checks
    if not isinstance(parsed_data["amount_usd"], (int, float)) or parsed_data["amount_usd"] < 0:
        raise OutputParserException("Schema violation: 'amount_usd' must be a non-negative number.")
        
    if not (0.0 <= parsed_data["risk_score"] <= 1.0):
        raise OutputParserException("Schema violation: 'risk_score' must be between 0.0 and 1.0.")
        
    return parsed_data

# Integrate validation directly into LCEL chain
validated_pipeline = prompt | model | json_parser | RunnableLambda(enforce_schema_constraints)
''', language="python")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Safe Parsing & Validation Pipeline", divider="rainbow")

st.markdown("""
The following complete production script illustrates a hardened output parsing pipeline featuring 
prompt format instructions injection, markdown fence stripping, PII token sanitization, 
and automated schema enforcement.
""")

st.code('''import json
import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

logger = logging.getLogger("ProductionParser")

class EnterpriseFinancialRecordParser:
    """Production-grade parser ensuring strict JSON conformance and financial validation."""
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.raw_parser = JsonOutputParser()
        
        # Build prompt with explicit format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial audit parser.
Extract the transaction audit log from the provided transcript.
{format_instructions}"""),
            ("human", "Audit Transcript: {transcript}")
        ]).partial(format_instructions=self.raw_parser.get_format_instructions())

        # Construct Master Pipeline
        self.pipeline: RunnableSequence = (
            self.prompt
            | self.model
            | self.raw_parser
            | RunnableLambda(self._validate_contract)
        )

    def _validate_contract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validates payload keys and coerces numerical values."""
        expected = ["transaction_id", "vendor", "total_amount", "currency", "approved"]
        missing = [key for key in expected if key not in payload]
        
        if missing:
            logger.error(f"Validation failed: missing fields {missing}")
            raise OutputParserException(f"Missing required financial fields: {missing}")

        # Coerce currency formatting
        payload["currency"] = str(payload["currency"]).upper().strip()
        payload["total_amount"] = float(payload["total_amount"])
        payload["approved"] = bool(payload["approved"])
        
        return payload

    def execute(self, transcript: str) -> Dict[str, Any]:
        try:
            return self.pipeline.invoke({"transcript": transcript})
        except OutputParserException as ope:
            logger.warning(f"Output parsing rejected malformed generation: {ope}")
            return {
                "status": "PARSE_ERROR",
                "error_details": str(ope),
                "raw_transcript_preview": transcript[:100]
            }
''', language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Live Output Parsing & Validation Simulator", divider="rainbow")

st.markdown("""
Test how raw model outputs (including markdown fences, conversational preambles, and malformed JSON) 
behave under different parsing modes. Inspect syntax cleanup and schema validation in real time.
""")

simulator = MockOutputParsingSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("📥 Input Raw Model Generation")
    sample_options = {
        "Clean JSON": '{"user_id": 1042, "role": "admin", "active": true, "tokens_used": 450}',
        "Markdown Fenced JSON": '```json\n{\n  "user_id": 1042,\n  "role": "admin",\n  "active": true,\n  "tokens_used": 450\n}\n```',
        "Conversational Preamble + Fenced": 'Here is the JSON data you asked for:\n```json\n{\n  "user_id": 1042,\n  "role": "admin",\n  "active": true\n}\n```\nLet me know if you need anything else!',
        "Malformed Syntax (Trailing Comma)": '{\n  "user_id": 1042,\n  "role": "admin",\n  "active": true,\n}',
        "Missing Required Key ('tokens_used')": '{\n  "user_id": 1042,\n  "role": "admin",\n  "active": true\n}'
    }
    
    selected_sample = st.selectbox("Load Sample Generation Pattern", list(sample_options.keys()))
    user_raw_input = st.text_area("Raw Text to Parse", value=sample_options[selected_sample], height=180)

with col_wb2:
    st.subheader("⚙️ Parser Configuration")
    parser_mode = st.radio("Target Parser Mode", ["JsonOutputParser (with Schema Validation)", "StrOutputParser (String Extraction)"])
    required_keys_input = st.text_input("Required Schema Keys (comma-separated)", value="user_id, role, active, tokens_used")
    req_keys = [k.strip() for k in required_keys_input.split(",") if k.strip()]

if st.button("⚡ Execute Parser", type="primary"):
    st.subheader("📊 Parsing Results")
    
    if parser_mode.startswith("StrOutputParser"):
        res = simulator.parse_string(user_raw_input)
        st.info("Mode: **StrOutputParser**")
        st.write("#### Extracted Clean String Output:")
        st.code(res.parsed_data, language="text")
    else:
        res = simulator.parse_json(user_raw_input, req_keys)
        st.info("Mode: **JsonOutputParser**")
        
        m1, m2 = st.columns(2)
        m1.metric("Syntax Parsing", "PASSED" if res.parsed_data is not None else "FAILED")
        m2.metric("Schema Validation", "VALID" if res.is_valid else "INVALID")
        
        if res.is_valid:
            st.success("✅ Output successfully parsed and validated against schema!")
            st.write("#### Parsed Python Object (`dict`):")
            st.json(res.parsed_data)
        else:
            st.error(f"❌ {res.error_msg}")
            if res.parsed_data:
                st.warning("Partial/Unvalidated Payload:")
                st.json(res.parsed_data)

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why should you avoid using basic regex to parse JSON from LLM outputs?",
     "Regex parsing is brittle against nested JSON structures, escaped quotation marks, Unicode sequences, and variable whitespace. Furthermore, regex cannot validate structural data types or schema constraints. Dedicated parsers like JsonOutputParser or PydanticOutputParser handle tokenizer edge cases, bracket matching, and structural validation reliably."),
    ("What is the primary difference in execution behavior between StrOutputParser and simply calling str(response)?",
     "StrOutputParser implements the full LCEL Runnable interface, including support for streaming (.stream() and .astream()). As tokens arrive from the LLM, StrOutputParser immediately extracts and yields text strings chunk-by-chunk. In contrast, calling str(response) requires waiting for the full response to complete and converts the entire AIMessage wrapper object into a string rather than just its content."),
    ("How does JsonOutputParser handle markdown code fences like ```json ... ```?",
     "JsonOutputParser contains built-in preprocessing logic that strips surrounding markdown code block fences (```json and ```) as well as conversational preambles or postscripts before invoking the JSON deserializer."),
    ("What happens when an LLM hallucinates an unexpected field in a JSON response?",
     "In basic JsonOutputParser, unexpected keys are simply preserved in the resulting Python dictionary. However, in strict validation frameworks (or Pydantic models configured with extra='forbid'), unexpected fields trigger a validation error, allowing the application to reject unauthorized fields."),
    ("How does format instruction injection improve parsing success rates?",
     "By calling parser.get_format_instructions() and injecting the generated text into the prompt, the model is provided with explicit syntactic constraints, schema specifications, and instructions to omit conversational prose. This primes the model to output syntactically valid JSON directly.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Crashing on Unescaped Quotes. If a model generates JSON containing unescaped double quotes inside a string value (e.g. {\"comment\": \"He said \"hello\"\"}), standard json.loads() fails with JSONDecodeError. Always sanitize or use robust partial JSON parsers.")
    st.error("**Trap 2 (Breaking Error)**: Incomplete JSON in Streamed Responses. Calling json.loads() on intermediate token chunks during streaming raises immediate decoding exceptions because the JSON syntax is incomplete. Use specialized streaming JSON parsers or accumulate the full message.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Bloated Format Instructions. Default Pydantic format instructions for complex nested schemas can consume 300 to 600 prompt tokens on every request. Consider using provider-native structured outputs (JSON schema mode) when available to eliminate format instruction overhead.")
    st.warning("**Trap 4 (Silent Failure Warning)**: Assuming Dict Keys are Uniform Across Models. Different models (or even the same model across temperatures) may oscillate between camelCase and snake_case keys. Always implement schema normalization or strict Pydantic field aliases.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: StrOutputParser Role"):
        st.markdown("**Extracts the .content string from AIMessage objects and enables chunk streaming.**")
    with st.expander("🎴 Flashcard 2: JsonOutputParser Cleaning"):
        st.markdown("**Automatically strips ```json fences and parses text into native Python dicts or lists.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Format Instructions"):
        st.markdown("**parser.get_format_instructions() generates prompt text specifying the required output format.**")
    with st.expander("🎴 Flashcard 4: OutputParserException"):
        st.markdown("**Specialized exception raised when model generation violates syntax or schema constraints.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why raw LLM outputs must be parsed before consumption by APIs.", value=True)
chk2 = st.checkbox("I can use StrOutputParser to stream text chunks cleanly through LCEL chains.", value=True)
chk3 = st.checkbox("I can implement JsonOutputParser and inject format instructions into prompts.", value=True)
chk4 = st.checkbox("I know how to strip markdown fences and validate JSON schema conformance.", value=True)
chk5 = st.checkbox("I understand the difference between syntactic parsing and semantic validation.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 output parsing interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 7.1 Output Parsing. Ready for 7.2 Output Parsers!")
