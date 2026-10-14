import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.4 Structured Outputs - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionStructuredOutputEngine:
    """
    Production-grade Structured Output and Tool Calling Engine.
    Demonstrates:
    - JSON extraction and markdown fence stripping
    - Schema validation against required field definitions
    - Simulated Function / Tool calling dispatch loop
    - Parsing repair for trailing commas and unquoted keys
    """
    def __init__(self):
        self.registered_tools = {
            "fetch_stock_quote": {
                "description": "Retrieves real-time equity pricing and trading volume.",
                "parameters": {"ticker": "str", "include_extended_hours": "bool"}
            },
            "execute_trade_order": {
                "description": "Executes a market or limit order on the brokerage exchange.",
                "parameters": {"ticker": "str", "shares": "int", "order_type": "str"}
            }
        }

    def strip_markdown_blocks(self, raw_text: str) -> str:
        """Strips ```json ... ``` code fence wrappers from raw model text."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def validate_schema(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validates dictionary data against required fields."""
        missing = [f for f in required_fields if f not in data]
        return len(missing) == 0, missing

    def simulate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates tool execution on backend API."""
        if tool_name not in self.registered_tools:
            return {"status": "error", "message": f"Tool '{tool_name}' not recognized."}
        
        if tool_name == "fetch_stock_quote":
            ticker = arguments.get("ticker", "UNKNOWN").upper()
            return {"status": "success", "ticker": ticker, "price": 182.45, "currency": "USD", "volume": "54.2M"}
        elif tool_name == "execute_trade_order":
            ticker = arguments.get("ticker", "UNKNOWN").upper()
            shares = arguments.get("shares", 0)
            order_type = arguments.get("order_type", "market")
            return {"status": "executed", "order_id": "ORD-98234", "ticker": ticker, "shares": shares, "order_type": order_type}
        
        return {"status": "error", "message": "Execution error"}

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("15.4 Structured Outputs")
st.caption("Chapter 15: Prompt Engineering | Domain: 2. Large Language Models")

st.markdown("""
Integrating Large Language Models into enterprise software architectures requires **deterministic, machine-readable outputs**.
A production backend API cannot parse prose essays; it requires validated JSON conforming strictly to database models and schemas.
However, autoregressive models naturally default to conversational chatter, markdown backticks, and trailing commentary.

This lesson covers the modern engineering stack for **Structured Outputs**: JSON schema enforcement, 
Grammar-Constrained Decoding (CFGs), Function Calling / Tool Use, and robust output validation pipelines.

In this lesson, you will master:
1. The limitations of naive prompt-based JSON requests and common failure modes.
2. Schema-Based Outputs: Pydantic specifications and Context-Free Grammar (CFG) decoding.
3. Function Calling: Transforming natural language directives into structured RPC payloads.
4. Tool Calling: Multi-turn tool execution loops and tool response assimilation.
5. Output Validation: Schema verification, automated retry loops, and error feedback.
6. Parsing Model Outputs: Stripping code fences, repairing broken JSON syntax, and defensive deserialization.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Structured Generation & Grammar-Guided Decoding Stack", divider="rainbow")

st.markdown("""
Production systems ensure structural compliance through prompt-level schemas, constrained logit masking, 
or post-generation validation loops.
""")

struct_dot = """
digraph StructuredStack {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_spec {
        label = "Layer 1: Schema Definition";
        color = "#334155";
        fontcolor = "#94A3B8";
        Pydantic [label="Pydantic Model / JSON Schema\\n(e.g. OrderPayload, ToolSpec)", fillcolor="#0F172A"];
        FSM [label="Grammar FSM Compiler\\n(Generates Allowed Token Masks)", fillcolor="#0369A1"];
        Pydantic -> FSM;
    }

    subgraph cluster_decode {
        label = "Layer 2: Constrained Decoding (Logit Masking)";
        color = "#334155";
        fontcolor = "#94A3B8";
        Logits [label="Raw Model Logits\\n[Vocab_Size]", fillcolor="#1E293B"];
        LogitMask [label="Mask Invalid Tokens\\nLogits[invalid] = -Infinity", fillcolor="#047857"];
        Sample [label="Sample Allowed Token\\n(Guarantees 100% Valid Syntax)", fillcolor="#047857"];
        Logits -> LogitMask -> Sample;
    }

    subgraph cluster_validation {
        label = "Layer 3: Execution & Tool Dispatch";
        color = "#334155";
        fontcolor = "#94A3B8";
        ParsedJSON [label="Verified JSON Payload", fillcolor="#7C3AED"];
        ToolExec [label="Backend Tool Execution\\nAPI / Database RPC", fillcolor="#B45309"];
        ParsedJSON -> ToolExec;
    }

    FSM -> LogitMask [label="FSM State Transitions"];
    Sample -> ParsedJSON [label="Deterministic Output"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(struct_dot, use_container_width=True)

st.markdown("""
### The Three Layers of Output Reliability
1. **Schema Definition**: Formal Pydantic schemas compiled into JSON Schema specifications.
2. **Grammar-Constrained Decoding (CFG)**: Masking illegal tokens at the logit level so the model cannot physically emit invalid syntax (Outlines, SGLang, OpenAI Structured Outputs).
3. **Validation & Tool Dispatch**: Deserializing verified payloads into typed runtime objects and dispatching backend API calls.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - JSON OUTPUTS
# ==============================================================================
st.header("2. JSON Outputs", divider="rainbow")

st.markdown("""
Prompting a base model with *"Output strictly in JSON"* fails approximately 10% to 20% of the time in production.
""")

col_js1, col_js2 = st.columns(2)
with col_js1:
    st.subheader("Common Failure Modes of Naive JSON Prompts")
    st.markdown("""
    * **Conversational Politeness**: Prepending phrases like *"Sure, here is your JSON:"* or appending *"Let me know if you need changes!"*.
    * **Markdown Fence Enclosures**: Wrapping valid JSON in ```` ```json ... ``` ```` blocks that cause `json.loads()` to throw `JSONDecodeError`.
    * **Trailing Commas**: Emitting `{"items": [1, 2, 3,]}` which is valid in JavaScript/Python but strictly illegal in standard RFC 8259 JSON.
    * **Unquoted Keys**: Emitting `{name: "Alice"}` instead of `{"name": "Alice"}`.
    """)

with col_js2:
    st.subheader("Prompt-Level Mitigation Strategies")
    st.markdown("""
    * **Negative Boundary Enforcements**: *"Do not output conversational preamble. Do not wrap in markdown fences. Output raw JSON only."*
    * **Few-Shot Exemplar Anchoring**: Providing 1 or 2 examples demonstrating exact JSON formatting.
    * **JSON Mode Parameter**: Setting `response_format={"type": "json_object"}` in OpenAI/Anthropic APIs (guarantees syntactically valid JSON, but does not guarantee schema adherence).
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - SCHEMA-BASED OUTPUTS
# ==============================================================================
st.header("3. Schema-Based Outputs (Grammar-Constrained Decoding)", divider="rainbow")

st.markdown("""
**Schema-Based Outputs** guarantee that generation not only adheres to valid JSON syntax, 
but conforms **100% deterministically to a specified Pydantic or JSON Schema definition**.
""")

st.markdown("""
#### How Grammar-Constrained Decoding (CFG) Works
Libraries like **Outlines**, **SGLang**, and **OpenAI Structured Outputs** operate at the **token sampling level**:
1. The target Pydantic schema is compiled into a **Finite State Machine (FSM)** or Context-Free Grammar.
2. At step `t`, the FSM inspects the tokens generated so far and determines which vocabulary tokens are mathematically valid next.
3. For example, if the model has generated `{"age": `, the FSM only permits integer digit tokens (`0-9`); all letter and quote tokens receive logit value `-infinity`!
4. The model physically cannot hallucinate invalid keys, wrong types, or missing commas.
""")

st.code("""
from pydantic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    name: str = Field(description="Full name of user")
    age: int = Field(ge=0, le=120)
    roles: List[str] = Field(description="Assigned RBAC roles")

# OpenAI Structured Outputs API
client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract user: Alice is 34, lead dev."}],
    response_format=UserProfile
)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - FUNCTION CALLING
# ==============================================================================
st.header("4. Function Calling", divider="rainbow")

st.markdown("""
**Function Calling** bridges the gap between natural language understanding and programmatic remote procedure calls (RPC).
Instead of generating free-form answers, the model chooses to call an external function and formats the required parameters as JSON.
""")

col_fn1, col_fn2 = st.columns(2)
with col_fn1:
    st.markdown("#### Tool Specification Schema")
    st.markdown("""
    Tools are provided to the model in the API payload as JSON Schema definitions:
    ```json
    {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
      }
    }
    ```
    """)

with col_fn2:
    st.markdown("#### Model Decision Logic")
    st.markdown("""
    * When the user asks: *"What's the weather in Seattle?"*, the model evaluates its tool list.
    * It recognizes that it lacks real-time weather data.
    * Rather than generating a text answer, it emits a `tool_calls` payload:
      ```json
      {"name": "get_weather", "arguments": "{\"location\": \"Seattle, WA\"}"}
      ```
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - TOOL CALLING
# ==============================================================================
st.header("5. Tool Calling (The Multi-Turn Execution Loop)", divider="rainbow")

st.markdown("""
A complete tool use workflow is a multi-turn conversation loop coordinated between the client application and the LLM.
""")

tool_loop_dot = """
digraph ToolLoop {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    Turn1 [label="Turn 1: User Query\\n'What is Apple's stock price?'", fillcolor="#0F172A"];
    Turn2 [label="Turn 2: LLM Emits Tool Call\\nget_stock_quote(ticker='AAPL')", fillcolor="#0369A1"];
    Turn3 [label="Turn 3: Application Executes Tool\\nFetches API: {'price': 182.45, 'currency': 'USD'}", fillcolor="#B45309"];
    Turn4 [label="Turn 4: LLM Synthesizes Answer\\n'Apple (AAPL) is currently trading at $182.45.'", fillcolor="#047857"];

    Turn1 -> Turn2 [label="Model Infers Tool Needed"];
    Turn2 -> Turn3 [label="Client App Runs Python Code"];
    Turn3 -> Turn4 [label="Returns tool result to LLM"];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(tool_loop_dot, use_container_width=True)

st.markdown("""
### Critical Architecture Rules for Tool Calling
1. **The LLM Never Runs Code Directly**: The LLM merely emits a string describing which function to call and what arguments to pass. Execution is strictly handled by the host application environment.
2. **Deterministic Role Segregation**: In OpenAI/Anthropic APIs, the tool execution result is appended as a message with role `"tool"` referencing the specific `tool_call_id`.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - OUTPUT VALIDATION
# ==============================================================================
st.header("6. Output Validation", divider="rainbow")

st.markdown("""
Even when using advanced prompting, defensive software engineering demands **rigorous validation** before passing model outputs to production databases.
""")

col_val1, col_val2 = st.columns(2)
with col_val1:
    st.subheader("Validation with Pydantic & Instructor")
    st.markdown("""
    * **Pydantic Field Constraints**: Enforce regex patterns, range limits (`gt=0`), and custom validators.
    * **The Automated Retry Loop**: If validation fails:
      ```python
      try:
          data = MySchema.model_validate_json(raw_output)
      except ValidationError as e:
          # Feed validation error back into the model!
          retry_prompt = f"Your output had validation errors: {e}. Please fix and regenerate."
      ```
    * Libraries like **Instructor** (Jason Liu) automate this retry loop seamlessly.
    """)

with col_val2:
    st.subheader("Business Rule Auditing")
    st.markdown("""
    * Type validity does not guarantee business logic validity.
    * Cross-reference extracted values against external databases (e.g. verifying that an extracted customer ID actually exists in PostgreSQL before proceeding).
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - PARSING MODEL OUTPUTS
# ==============================================================================
st.header("7. Parsing Model Outputs", divider="rainbow")

st.markdown("""
When grammar-constrained decoding is unavailable (e.g. legacy APIs, self-hosted models without Outlines), 
robust post-processing pipelines must handle common model syntax glitches.
""")

st.markdown("""
### The 3-Stage Defensive Parsing Pipeline
""")

st.code("""
import json
import re

def parse_llm_json(raw_text: str) -> dict:
    # Stage 1: Strip markdown code blocks
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = re.search(r"```json\\s*(.*?)\\s*```", cleaned, re.DOTALL).group(1)
    elif "```" in cleaned:
        cleaned = re.search(r"```\\s*(.*?)\\s*```", cleaned, re.DOTALL).group(1)
        
    # Stage 2: Attempt standard JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
        
    # Stage 3: Fallback to heuristic JSON repair (e.g. json_repair library)
    # Fixes unquoted keys, trailing commas, single quotes, unclosed brackets
    import json_repair
    return json_repair.repair_json(cleaned, return_objects=True)
""", language="python")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Structured Output Generator & Pydantic Validator in Python", divider="rainbow")

st.markdown("""
The following self-contained class implements fence stripping, schema validation, and tool call dispatch simulation.
""")

st.code("""
from typing import Dict, Any, List, Tuple

class ProductionStructuredOutputEngine:
    \"\"\"
    Production-grade Structured Output and Tool Calling Engine.
    \"\"\"
    def __init__(self):
        self.registered_tools = {
            "fetch_stock_quote": {"parameters": ["ticker"]},
            "execute_trade_order": {"parameters": ["ticker", "shares", "order_type"]}
        }

    def strip_markdown_blocks(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            return "\\n".join(lines).strip()
        return text

    def validate_schema(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        missing = [f for f in required_fields if f not in data]
        return len(missing) == 0, missing

    def simulate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "fetch_stock_quote":
            return {"status": "success", "ticker": arguments.get("ticker", "").upper(), "price": 182.45}
        return {"status": "executed", "order_id": "ORD-98234", "details": arguments}
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Pydantic Schema Validator & Tool Dispatch Simulator", divider="rainbow")

engine = ProductionStructuredOutputEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("1. Raw Model Output Sanitizer & Validator")
    sample_raw = st.text_area(
        "Simulate Raw Model Output (Notice Markdown Fences)",
        value="""```json
{
  "ticker": "AAPL",
  "shares": 50,
  "order_type": "limit",
  "limit_price": 180.50
}
```"""
    )
    req_fields = st.multiselect(
        "Required Schema Fields",
        ["ticker", "shares", "order_type", "account_id"],
        default=["ticker", "shares", "order_type"]
    )

    stripped = engine.strip_markdown_blocks(sample_raw)
    
    # Try parsing
    import json
    try:
        parsed_dict = json.loads(stripped)
        is_valid, missing = engine.validate_schema(parsed_dict, req_fields)
        parse_success = True
    except Exception as e:
        parsed_dict = {}
        is_valid = False
        missing = [f"JSON Syntax Error: {e}"]
        parse_success = False

with wb_col2:
    st.subheader("2. Validation & Tool Dispatch Tracer")
    
    if parse_success:
        if is_valid:
            st.success("✅ Schema Validation PASSED: All required fields present.")
            st.json(parsed_dict)

            st.markdown("##### Simulated Tool Call Execution:")
            dispatch_result = engine.simulate_tool_call("execute_trade_order", parsed_dict)
            st.info(f"API Dispatch Result: {dispatch_result}")
        else:
            st.error(f"❌ Schema Validation FAILED! Missing fields: {missing}")
    else:
        st.error(f"🚨 JSON Parsing Crashed: {missing[0]}")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: How does Grammar-Constrained Decoding (CFG) guarantee 100% schema adherence?"):
        st.write("""
        Grammar-constrained decoding compiles a target JSON Schema into a Finite State Machine (FSM).
        During token generation, before the model samples the next token, the FSM masks the logit distribution: any token that would violate the grammar receives a logit score of -infinity.
        Because invalid tokens have zero probability, the model is physically prevented from outputting unquoted keys, wrong data types, or syntax errors.
        """)
    with st.expander("Q2: What is the difference between Function Calling and direct JSON prompting?"):
        st.write("""
        In direct JSON prompting, the user instructs the model in natural language to emit JSON. The model may include markdown fences, conversational preambles, or hallucinate keys.
        Function Calling provides formal API schema definitions. Frontier models are fine-tuned to emit dedicated `tool_calls` payloads that bypass conversational tokens and integrate directly into RPC client libraries.
        """)
    with st.expander("Q3: Does the LLM execute code when performing Function Calling?"):
        st.write("""
        No. The LLM never executes code.
        The LLM only acts as a reasoning router: it examines user intent and emits a JSON string specifying which function to call and what arguments to supply.
        The host application receives this JSON, verifies permissions, runs the actual Python/Node function, and sends the return value back to the model.
        """)

with q2:
    with st.expander("Q4: How should a production pipeline handle Pydantic ValidationError exceptions?"):
        st.write("""
        Production pipelines should implement an automated reflection retry loop:
        1. Catch the `ValidationError` exception.
        2. Format the exact validation error message into a follow-up user turn: *"Your output failed validation: field 'age' must be >= 0. Please fix this field."*
        3. Re-prompt the model (up to 2-3 retries). 
        This achieves over 99.5% final schema compliance without human intervention.
        """)
    with st.expander("Q5: Why does naive json.loads() frequently fail on raw LLM completions?"):
        st.write("""
        Models frequently enclose JSON in markdown code blocks (` ```json ... ``` `), include conversational pleasantries, or emit trailing commas (`[1, 2,]`).
        Standard Python `json.loads()` strictly rejects markdown fences and trailing commas. Production code must strip code fences and use defensive repair libraries like `json_repair`.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Passing Raw LLM Text to json.loads()**
    * **Failure**: Writing `data = json.loads(response.choices[0].message.content)` without stripping markdown fences.
    * **Impact**: Crashes production APIs whenever the model decides to wrap output in ` ```json `.
    * **Remedy**: Always strip code fences using regex before calling `json.loads()`.
    """)
    st.error("""
    **🚨 Production Trap 2: Omitting Tool Argument Type Validations**
    * **Failure**: Blindly unpacking tool arguments into Python functions: `my_func(**tool_call.arguments)`.
    * **Impact**: Vulnerable to type mismatch errors, SQL injections, or unvalidated parameter attacks.
    * **Remedy**: Always parse tool arguments through a strict Pydantic model before function invocation.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Confusing JSON Mode with Structured Outputs**
    * **Failure**: Assuming `response_format={"type": "json_object"}` guarantees correct fields.
    * **Impact**: JSON Mode only guarantees valid JSON syntax; keys can still be completely hallucinated or omitted!
    * **Remedy**: Use OpenAI Structured Outputs (`json_schema`) or Outlines for deterministic schema enforcement.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Excessive Schema Complexity in Tools**
    * **Failure**: Passing a 50-field nested JSON Schema into a tool definition.
    * **Impact**: Model context fills up with schema definitions; argument hallucination increases rapidly.
    * **Remedy**: Keep tool schemas minimal, flat, and modular.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Constrained Decoding Tool"):
        st.write("Name a popular open-source library for grammar-constrained decoding:")
        st.info("Outlines (or SGLang / Guidance).")
    with st.expander("⚡ Flashcard 2: RFC 8259 Trailing Comma Rule"):
        st.write("Are trailing commas valid in standard RFC 8259 JSON?")
        st.info("No. Trailing commas violate standard JSON syntax.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Who Executes Tool Code?"):
        st.write("Does the LLM execute tool code directly?")
        st.info("No. The client application executes code; the LLM only generates the call specification.")
    with st.expander("⚡ Flashcard 4: JSON Mode vs Structured Outputs"):
        st.write("What is the difference between JSON Mode and Structured Outputs?")
        st.info("JSON Mode guarantees valid JSON syntax; Structured Outputs guarantees exact schema adherence.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Structured Outputs before advancing to Lesson 15.5 (Prompt Optimization):
""")

chk1 = st.checkbox("I understand why naive prompt-based JSON enforcement fails in production.", value=True)
chk2 = st.checkbox("I know how grammar-constrained decoding (CFGs) uses logit masking to guarantee valid schemas.", value=True)
chk3 = st.checkbox("I understand the multi-turn execution loop of Function Calling and Tool Use.", value=True)
chk4 = st.checkbox("I know why client applications must execute code and validate arguments with Pydantic.", value=True)
chk5 = st.checkbox("I can implement an automated reflection retry loop for Pydantic ValidationError exceptions.", value=True)
chk6 = st.checkbox("I know how to sanitize model outputs by stripping markdown code fences and repairing syntax.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 15.5: Prompt Optimization.")
