import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="8.1 Introduction to Tools - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("8.1 Introduction to Tools: Interfaces, Capabilities, Schemas & Grounding")
st.caption("Chapter 8: Tools | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
By default, Large Language Models are closed-world cognitive engines: their knowledge is bounded by their 
training cutoff date, they cannot execute external computation, and they cannot interact with external systems.

**Tools** are the sensory organs and actuator limbs of AI agents. In LangChain, a Tool is an executable 
component with a defined input schema, a descriptive purpose, and a programmatic execution function. 
When equipped with tools, LLMs evolve from text generators into **autonomous problem solvers**.

In this lesson, you will master:
- **What is a Tool?**: The anatomy of LangChain `BaseTool` objects.
- **Why Agents Need Tools**: Grounding models with real-time data, computational rigor, and system side-effects.
- **Tool Inputs & Arguments**: Validating arguments via strict schemas before execution.
- **Tool Outputs**: Returning textual observations, structured data, and error traces back to the model context.
- **Tool Descriptions**: Crafting semantic descriptions that guide the LLM's tool selection decisions.
- **Tool Schemas**: Exposing JSON Schema contracts to provider function calling engines.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockToolInspectionResult:
    """Represents the schema introspection of a tool in module scope."""
    def __init__(self, tool_name: str, description: str, args_schema: Dict[str, Any], return_direct: bool):
        self.tool_name = tool_name
        self.description = description
        self.args_schema = args_schema
        self.return_direct = return_direct

class MockToolDefinitionSimulator:
    """Simulates tool schema generation and execution in module scope."""
    def __init__(self):
        self.catalog = {
            "get_stock_price": MockToolInspectionResult(
                tool_name="get_stock_price",
                description="Retrieve current real-time market price for a given stock ticker symbol. Use when user asks for stock quotes.",
                args_schema={
                    "ticker": {"type": "string", "description": "Stock symbol (e.g. AAPL, MSFT)", "required": True},
                    "currency": {"type": "string", "description": "Currency code (e.g. USD, EUR)", "default": "USD"}
                },
                return_direct=False
            ),
            "execute_sql_query": MockToolInspectionResult(
                tool_name="execute_sql_query",
                description="Execute a read-only SELECT query against the analytics database. Always limit results to 50 rows.",
                args_schema={
                    "query": {"type": "string", "description": "Valid PostgreSQL SELECT statement", "required": True},
                    "timeout_seconds": {"type": "integer", "description": "Execution timeout limit", "default": 10}
                },
                return_direct=False
            ),
            "send_slack_alert": MockToolInspectionResult(
                tool_name="send_slack_alert",
                description="Send an operational alert message to a specific Slack channel. Produces external side-effects.",
                args_schema={
                    "channel": {"type": "string", "description": "Slack channel name (e.g. #ops-alerts)", "required": True},
                    "message": {"type": "string", "description": "Alert body text", "required": True},
                    "urgency": {"type": "string", "enum": ["INFO", "WARNING", "CRITICAL"], "default": "INFO"}
                },
                return_direct=True
            )
        }

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.catalog:
            return {"status": "ERROR", "error": f"Tool '{tool_name}' not found in registry."}

        tool = self.catalog[tool_name]
        # Validate required arguments
        for arg_name, arg_meta in tool.args_schema.items():
            if arg_meta.get("required", False) and arg_name not in arguments:
                return {
                    "status": "VALIDATION_FAILED",
                    "error": f"Missing required parameter '{arg_name}' for tool '{tool_name}'."
                }

        # Simulated Tool Execution Payloads
        if tool_name == "get_stock_price":
            ticker = arguments.get("ticker", "").upper()
            return {
                "status": "SUCCESS",
                "result": f"Current market price of {ticker}: 218.45 {arguments.get('currency', 'USD')}. Volume: 48.2M shares.",
                "execution_time_ms": 14
            }
        elif tool_name == "execute_sql_query":
            return {
                "status": "SUCCESS",
                "result": [{"user_id": 101, "total_orders": 14, "lifetime_value": 4820.50}],
                "execution_time_ms": 68
            }
        elif tool_name == "send_slack_alert":
            return {
                "status": "SUCCESS",
                "result": f"Dispatched alert to {arguments.get('channel')} with urgency [{arguments.get('urgency', 'INFO')}].",
                "execution_time_ms": 95
            }
        return {"status": "SUCCESS", "result": "Executed"}

# --- SECTION 1: ARCHITECTURE BLUEPRINT & AGENT INTERACTION LIFECYCLE ---
st.header("1. Architecture Blueprint: The Tool Interface & Agent Interaction Lifecycle", divider="rainbow")

st.markdown("""
In the LangChain framework, a **Tool** is not merely a Python function; it is a formal wrapper adhering 
to the `BaseTool` specification. It defines a JSON Schema contract presented to the model during prompting 
and handles parameter deserialization and output capture during execution.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Tool Registry & Schema Introspection
        subgraph cluster_registration {
            label = "1. REGISTRATION &amp; SCHEMA EXPOSURE TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ToolObject [label="BaseTool Instance\\nname='get_weather'\\ndescription='Fetch live weather'\\nargs_schema=WeatherInput", color="#10B981"];
            JsonSchemaGen [label="JSON Schema Compilation\\nname, description, parameters\\nOpenAPI / Tool Definition Spec"];
        }

        // Level 2: Model Binding & Decision
        subgraph cluster_binding {
            label = "2. MODEL BINDING &amp; SELECTION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            BoundModel [label="ChatModel.bind_tools([tool])\\nInjects tool schemas into API payload"];
            ModelInference [label="Model Autoregressive Generation\\nDetermines tool need from prompt &amp; description"];
            ToolCallMessage [label="AIMessage with tool_calls\\n[ToolCall(name='get_weather', args={'city': 'Tokyo'})]"];
        }

        // Level 3: Validation & Execution
        subgraph cluster_execution {
            label = "3. VALIDATION &amp; DISPATCH TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ArgsValidator [label="Pydantic Args Validation\\nEnsures types and boundaries match schema"];
            PhysicalExecutor [label="Physical Function Invocation\\ntool._run(**args) or tool._arun(**args)"];
        }

        // Level 4: Observation & Grounding
        subgraph cluster_observation {
            label = "4. OBSERVATION &amp; GROUNDING TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ToolMessageNode [label="ToolMessage Object\\ncontent='22°C, Sunny'\\ntool_call_id='call_9481'", color="#10B981"];
            AgentSynthesis [label="Model Second Turn\\nGrounds final answer using ToolMessage observation"];
        }

        // Wiring
        ToolObject -> JsonSchemaGen;
        JsonSchemaGen -> BoundModel;
        BoundModel -> ModelInference;
        ModelInference -> ToolCallMessage;
        
        ToolCallMessage -> ArgsValidator [label="Dispatch tool_call"];
        ArgsValidator -> PhysicalExecutor [label="Validation OK"];
        PhysicalExecutor -> ToolMessageNode [label="Capture return value"];
        
        ToolMessageNode -> AgentSynthesis [label="Feed observation back to LLM"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Lifecycle Axioms:**
1. **The Model Decides; The Host Executes**: The LLM *never* runs the tool code directly. It emits structured JSON expressing its intent to run the tool. The hosting application validates and executes the code locally.
2. **Deterministic Sandboxing**: Tools must enforce strict argument validation and timeout limits to prevent malicious input injections or infinite hanging operations.
""")

st.divider()

# --- SECTION 2: WHAT IS A TOOL? ---
st.header("2. What Is a Tool? (The Fundamental Concept)", divider="rainbow")

st.markdown("""
In LangChain, a **Tool** is an abstraction that bundles three critical components:
1. **A Name (`name: str`)**: A unique identifier used by the model to refer to the tool (e.g. `web_search_duckduckgo`).
2. **A Description (`description: str`)**: Natural language text explaining *what* the tool does and *when* the model should invoke it.
3. **An Argument Schema (`args_schema: Type[BaseModel]`)**: A Pydantic model defining the expected input types, required parameters, and constraints.
4. **An Execution Callable (`_run` and `_arun`)**: The synchronous and asynchronous Python functions executed when the tool is called.
""")

st.code('''from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate e.g. '2 * 3 + 14'")

class CustomCalculatorTool(BaseTool):
    name: str = "custom_calculator"
    description: str = "Perform exact arithmetic calculations. Use when precision is required."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Synchronous execution handler."""
        try:
            # Safe evaluated math calculation
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Calculation error: {e}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous execution handler."""
        return self._run(expression)
''', language="python")

st.divider()

# --- SECTION 3: WHY AGENTS NEED TOOLS ---
st.header("3. Why Agents Need Tools (Overcoming LLM Limitations)", divider="rainbow")

st.markdown("""
LLMs possess remarkable reasoning and semantic synthesis capabilities, but suffer from three fundamental limitations:
""")

col_lim1, col_lim2, col_lim3 = st.columns(3)
with col_lim1:
    st.error("1. Knowledge Cutoff & Staleness")
    st.markdown("""
    Pre-trained weights are static snapshots of the internet. 
    **Tools provide real-time grounding**:
    - Weather forecasts
    - Live financial quotes
    - Today's company news
    - Active server logs
    """)

with col_lim2:
    st.error("2. Computational Blindspots")
    st.markdown("""
    Autoregressive decoders predict tokens; they do not calculate math.
    **Tools guarantee algorithmic rigor**:
    - Precise arithmetic & floating point math
    - SQL database aggregations
    - Complex physics simulations
    - Exact regex pattern searching
    """)

with col_lim3:
    st.error("3. Zero External Actuation")
    st.markdown("""
    Raw LLMs cannot affect external system state.
    **Tools execute side-effects**:
    - Sending emails or Slack messages
    - Writing records to PostgreSQL
    - Triggering CI/CD pipelines
    - Controlling robotics / IoT devices
    """)

st.divider()

# --- SECTION 4: TOOL INPUTS ---
st.header("4. Tool Inputs (Signatures, Strongly-Typed Parameters & Coercion)", divider="rainbow")

st.markdown("""
A tool's input contract specifies how the LLM should format the tool call arguments. 
Inputs can range from a single scalar primitive to complex nested dictionaries.
""")

st.subheader("Input Contract Taxonomy")
input_taxonomy_df = pd.DataFrame({
    "Input Style": ["Single Primitive (String)", "Multi-Argument Function", "Pydantic Schema (Complex)", "Nested Object Tree"],
    "Definition Method": ["@tool def search(query: str)", "@tool def book(room: str, days: int)", "args_schema=CustomerProfile", "args_schema=OrderTree"],
    "Type Safety": ["Basic", "Strong", "Enterprise Grade", "Enterprise Grade"],
    "Validation Hooks": ["None", "Type hints", "@field_validator", "Full recursive validation"]
})
st.dataframe(input_taxonomy_df, use_container_width=True)

st.code('''from langchain_core.tools import tool
from typing import Optional

@tool
def fetch_user_record(user_id: int, include_billing: bool = False, region: Optional[str] = "us-east") -> dict:
    """Fetch user record from primary database.
    
    Args:
        user_id: The unique numeric identifier of the user.
        include_billing: If True, attaches credit card and invoice metadata.
        region: The geographical cluster housing the user shard.
    """
    return {
        "user_id": user_id,
        "name": "Jane Smith",
        "region": region,
        "billing_attached": include_billing
    }

# Inspecting tool's compiled arguments:
print(fetch_user_record.args)
# Output: {'user_id': {'title': 'User Id', 'type': 'integer'}, ...}
''', language="python")

st.divider()

# --- SECTION 5: TOOL OUTPUTS ---
st.header("5. Tool Outputs (Strings, Observations & Return Direct)", divider="rainbow")

st.markdown("""
When a tool finishes execution, it returns an observation. In LangChain, tool outputs are typically converted 
into a **`ToolMessage`** that is appended to the message history and sent back to the model.

Key considerations for tool outputs:
1. **String Conversion**: Even if a tool returns a dictionary or list, LangChain serializes it into a string (e.g. JSON string) so the LLM tokenizer can ingest it.
2. **Context Window Management**: If a tool returns a 50,000-row SQL dump, it will blow past the model's context limit. Tools should summarize, truncate, or paginate large payloads.
3. **`return_direct=True`**: An optional configuration flag. When set to `True`, the tool's output is returned directly to the end user, bypassing the model's second synthesis turn.
""")

st.code('''from langchain_core.tools import tool

@tool(return_direct=True)
def download_large_report_url(report_id: str) -> str:
    """Generate a pre-signed S3 download URL for an executive report.
    This tool returns directly to the user without LLM rewriting.
    """
    return f"https://reports.internal.corp/s3/download/{report_id}?token=exp9481"
''', language="python")

st.divider()

# --- SECTION 6: TOOL DESCRIPTIONS ---
st.header("6. Tool Descriptions (The Semantic Prompting Bridge)", divider="rainbow")

st.markdown("""
The **description** of a tool is its most critical parameter. 
The LLM reads this description to decide *whether*, *when*, and *how* to invoke the tool. 
A poorly written description leads to tool hallucination, false positives, or the model failing to call the tool entirely.
""")

st.subheader("Rules for Writing High-Quality Tool Descriptions")
st.markdown("""
1. **State the Specific Objective**: Clearly specify the exact capability (e.g., *"Search the internal corporate HR wiki"* vs *"Search"*).
2. **Define When to Use (and When NOT to Use)**: Provide clear guardrails (e.g., *"Use this for company policies regarding maternity leave, health insurance, and PTO. Do NOT use for engineering technical documentation."*).
3. **Specify Expected Input Formats**: Mention required formats (e.g., *"Date must be formatted as YYYY-MM-DD"*).
4. **Clarify Side Effects**: Signal whether the operation is read-only or mutates external state.
""")

st.code('''# Example of a Production-Grade Tool Description
@tool
def cancel_flight_reservation(booking_reference: str, reason: str) -> str:
    """Irrevocably cancel an active airline passenger flight reservation.
    
    CRITICAL: Only invoke this tool when the user has EXPLICITLY confirmed they want to cancel 
    their booking. Do NOT invoke for fare inquiries or flight rescheduling.
    
    Args:
        booking_reference: The 6-character alphanumeric PNR code (e.g. 'XK982A').
        reason: Mandatory explanation for passenger cancellation request.
    """
    return f"Reservation {booking_reference} successfully cancelled. Cancellation fee: $50 USD."
''', language="python")

st.divider()

# --- SECTION 7: TOOL SCHEMAS ---
st.header("7. Tool Schemas (JSON Schema Specification & OpenAPI Conformance)", divider="rainbow")

st.markdown("""
Modern LLMs do not inspect Python source code directly. When tools are bound to a chat model via `bind_tools()`, 
LangChain automatically translates the tool definition into a standardized **OpenAPI / JSON Schema** dictionary.
""")

st.subheader("Generated JSON Schema Structure")
st.code('''# Inspecting the exact JSON Schema sent to provider APIs:
from langchain_core.utils.function_calling import convert_to_openai_tool

openai_tool_schema = convert_to_openai_tool(fetch_user_record)
print(openai_tool_schema)

# Structure:
# {
#   "type": "function",
#   "function": {
#     "name": "fetch_user_record",
#     "description": "Fetch user record from primary database...",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "user_id": {"type": "integer", "description": "The unique numeric identifier..."},
#         "include_billing": {"type": "boolean", "default": false},
#         "region": {"type": "string", "default": "us-east"}
#       },
#       "required": ["user_id"]
#     }
#   }
# }
''', language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Tool Definition & Validation Service", divider="rainbow")

st.markdown("""
The following complete production pattern illustrates an enterprise tool registry that defines, 
validates, and executes secure tools with argument sanitization and telemetry tracking.
""")

st.code('''import logging
from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger("EnterpriseToolRegistry")

class KubernetesScaleInput(BaseModel):
    deployment_name: str = Field(description="Name of Kubernetes deployment to scale")
    namespace: str = Field(default="production", description="Target Kubernetes namespace")
    target_replicas: int = Field(description="Target pod replica count", ge=1, le=100)

class KubernetesScalerTool(BaseTool):
    name: str = "scale_kubernetes_deployment"
    description: str = """Scale pod replicas for a Kubernetes deployment. 
Use when an operator requests autoscaling adjustment or cluster capacity changes.
Only permitted for replica counts between 1 and 100."""
    args_schema: Type[BaseModel] = KubernetesScaleInput

    def _run(self, deployment_name: str, target_replicas: int, namespace: str = "production") -> str:
        logger.info(f"Scaling deployment '{deployment_name}' in '{namespace}' to {target_replicas} replicas.")
        # In real production: call Kubernetes API client
        return f"Successfully updated deployment '{deployment_name}' in namespace '{namespace}' to {target_replicas} pods."

    async def _arun(self, deployment_name: str, target_replicas: int, namespace: str = "production") -> str:
        return self._run(deployment_name, target_replicas, namespace)

class EnterpriseToolManager:
    """Central registry and dispatch engine for enterprise AI tools."""
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.register_tool(KubernetesScalerTool())

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def list_schemas(self) -> List[Dict[str, Any]]:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        return [convert_to_openai_tool(t) for t in self.tools.values()]

    def invoke(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self.tools[name].invoke(args)
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Tool Definition Inspector & Execution Playground", divider="rainbow")

st.markdown("""
Inspect real tool schemas generated for provider function calling, test input parameter validation, 
and observe live simulated tool executions.
""")

sim = MockToolDefinitionSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🛠️ Select Tool from Catalog")
    chosen_tool_name = st.selectbox("Registered Tool", list(sim.catalog.keys()))
    tool_meta = sim.catalog[chosen_tool_name]

    st.write(f"**Description**: *{tool_meta.description}*")
    st.write(f"**Direct Return to User (`return_direct`)**: `{tool_meta.return_direct}`")

    st.markdown("#### Configure Invocation Arguments")
    args_input = {}
    if chosen_tool_name == "get_stock_price":
        ticker_val = st.text_input("Ticker Symbol", value="AAPL")
        currency_val = st.selectbox("Currency", ["USD", "EUR", "GBP"])
        args_input = {"ticker": ticker_val, "currency": currency_val}
    elif chosen_tool_name == "execute_sql_query":
        query_val = st.text_area("SQL Statement", value="SELECT user_id, total_orders FROM users WHERE total_orders > 10;")
        args_input = {"query": query_val, "timeout_seconds": 10}
    elif chosen_tool_name == "send_slack_alert":
        chan_val = st.text_input("Slack Channel", value="#ops-alerts")
        msg_val = st.text_input("Alert Message", value="Prod DB latency exceeds SLA threshold (>500ms)")
        urgency_val = st.selectbox("Urgency", ["INFO", "WARNING", "CRITICAL"])
        args_input = {"channel": chan_val, "message": msg_val, "urgency": urgency_val}

with col_wb2:
    st.subheader("📋 Generated JSON Schema (OpenAPI)")
    st.json({
        "type": "function",
        "function": {
            "name": tool_meta.tool_name,
            "description": tool_meta.description,
            "parameters": {
                "type": "object",
                "properties": tool_meta.args_schema,
                "required": [k for k, v in tool_meta.args_schema.items() if v.get("required", False)]
            }
        }
    })

if st.button("🚀 Invoke Tool Simulation", type="primary"):
    res = sim.execute_tool_call(chosen_tool_name, args_input)
    st.subheader("📊 Tool Execution Output")
    
    if res["status"] == "SUCCESS":
        st.success(f"Status: **{res['status']}** (Latency: {res['execution_time_ms']} ms)")
        st.write("#### Returned Observation (`ToolMessage.content`):")
        st.write(res["result"])
    else:
        st.error(f"Status: **{res['status']}**")
        st.write(res["error"])

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Does the LLM execute the tool code inside its own runtime environment?",
     "No. The LLM never executes code directly. The model merely generates a structured text payload (a tool call) requesting the host application to run the tool. The hosting LangChain Python runtime executes the function locally, captures the return value, and sends the result back to the model as a ToolMessage."),
    ("Why is a tool's natural language description as important as its Python code implementation?",
     "The description is the sole semantic clue the model receives to understand the purpose, scope, and preconditions of the tool. If the description is vague or ambiguous, the model may fail to call it, call it with inappropriate arguments, or hallucinate calling a non-existent tool."),
    ("What is the purpose of the return_direct parameter in BaseTool?",
     "When return_direct=True, LangChain immediately terminates the agent reasoning loop upon tool completion and returns the tool's raw output directly to the end user, bypassing the second model turn. This is ideal for tools generating pre-signed download URLs, rendered charts, or binary file payloads."),
    ("How do Pydantic schemas prevent security vulnerabilities in tools?",
     "Pydantic schemas enforce strict type validation, regex pattern matching, and numerical boundaries on all tool arguments before the Python execution function is invoked. This prevents arbitrary code execution, SQL injections, and buffer overflows caused by malformed or adversarial model outputs."),
    ("What is the difference between _run() and _arun() in LangChain BaseTool?",
     "_run() is the synchronous execution method used in standard blocking chains. _arun() is the asynchronous coroutine method used in async web frameworks (FastAPI) and streaming agents. If _arun() is not implemented, calling ainvoke() will raise a NotImplementedError unless configured to run in a thread executor.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Unhandled Exceptions Inside Tools. If your tool throws an unhandled exception (e.g. database connection timeout), the entire agent pipeline crashes immediately. Always handle exceptions inside the tool and return informative error strings so the LLM can self-correct.")
    st.error("**Trap 2 (Breaking Error)**: Mutable Default Arguments in Tool Signatures. Defining a tool with `def tool(tags: list = [])` causes mutable state leakage across distinct user sessions. Always use Pydantic models with `default_factory=list`.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Massive Tool Output Dumps. Returning multi-megabyte API responses or raw HTML dumps inside a tool observation blows through token budgets and causes context window overflow. Always summarize or filter tool outputs before returning them.")
    st.warning("**Trap 4 (Quality Warning)**: Overlapping Tool Descriptions. Having two tools with similar descriptions (e.g. 'search_docs' and 'lookup_documentation') causes model indecision and hallucinated tool calls. Keep tool descriptions mutually exclusive.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: BaseTool Core Attributes"):
        st.markdown("**name, description, args_schema, and the execution methods _run() / _arun().**")
    with st.expander("🎴 Flashcard 2: Execution Boundary"):
        st.markdown("**LLMs only emit tool_calls intent; the host Python runtime executes the code.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: return_direct=True"):
        st.markdown("**Immediately returns tool output to user without a second model synthesis turn.**")
    with st.expander("🎴 Flashcard 4: ToolMessage Role"):
        st.markdown("**Carries the tool's execution result back to the model history matched by tool_call_id.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural definition of a Tool in LangChain.", value=True)
chk2 = st.checkbox("I know why AI agents require tools for grounding, math, and external side-effects.", value=True)
chk3 = st.checkbox("I can define tool input schemas using Pydantic BaseModel classes.", value=True)
chk4 = st.checkbox("I understand how tool descriptions guide LLM tool selection decisions.", value=True)
chk5 = st.checkbox("I know how LangChain translates tools into standard OpenAPI JSON schemas.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 introduction to tools interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 8.1 Introduction to Tools. Ready for 8.2 Creating Tools!")
