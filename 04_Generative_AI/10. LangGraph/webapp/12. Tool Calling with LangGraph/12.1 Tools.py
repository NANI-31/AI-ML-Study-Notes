import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="12.1 Tools - LangGraph Curriculum",
    page_icon="🛠️",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-tool {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-schema {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #9333EA;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class ToolSimulationEngine:
    """
    Simulates tool registration, JSON schema extraction, argument validation,
    execution, and error handling.
    """
    def __init__(self):
        self.registered_tools = {
            "fetch_customer_record": {
                "description": "Fetch customer account status and balance from the core CRM.",
                "parameters": {
                    "customer_id": {"type": "string", "description": "Unique alphanumeric customer identifier (e.g. CUST-1049)"},
                    "include_history": {"type": "boolean", "description": "Whether to return the last 90 days transaction audit"}
                },
                "required": ["customer_id"]
            },
            "calculate_tax_liability": {
                "description": "Calculate municipal and federal tax liability for a specified transaction amount.",
                "parameters": {
                    "amount_usd": {"type": "number", "description": "Transaction amount in USD"},
                    "jurisdiction": {"type": "string", "description": "Two-letter state or province code (e.g. NY, CA, TX)"}
                },
                "required": ["amount_usd", "jurisdiction"]
            },
            "execute_database_rollback": {
                "description": "Trigger an automated transactional rollback on a stalled distributed transaction.",
                "parameters": {
                    "transaction_id": {"type": "string", "description": "Distributed transaction UUID"},
                    "force": {"type": "boolean", "description": "Bypass graceful shutdown checks"}
                },
                "required": ["transaction_id"]
            }
        }

    def execute_tool(self, tool_name: str, args: Dict[str, Any], simulate_error: bool) -> Dict[str, Any]:
        if simulate_error:
            return {
                "status": "ERROR",
                "error_type": "ToolExecutionError",
                "message": f"External service unavailable: Connection timed out while executing {tool_name} with parameters {args}."
            }
        
        if tool_name == "fetch_customer_record":
            cid = args.get("customer_id", "UNKNOWN")
            return {
                "status": "SUCCESS",
                "data": {
                    "customer_id": cid,
                    "tier": "Enterprise Gold",
                    "balance_usd": 84250.75,
                    "risk_flag": False
                }
            }
        elif tool_name == "calculate_tax_liability":
            amt = float(args.get("amount_usd", 1000.0))
            jur = args.get("jurisdiction", "CA").upper()
            rate = 0.0825 if jur == "CA" else (0.08875 if jur == "NY" else 0.0625)
            return {
                "status": "SUCCESS",
                "data": {
                    "base_amount": amt,
                    "jurisdiction": jur,
                    "effective_tax_rate": rate,
                    "tax_liability_usd": round(amt * rate, 2)
                }
            }
        else:
            tx = args.get("transaction_id", "tx_9982")
            return {
                "status": "SUCCESS",
                "data": {
                    "transaction_id": tx,
                    "rollback_status": "COMMITTED",
                    "locks_released": 14
                }
            }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Tool Calling Protocol and JSON Schema Binding", divider="rainbow")

st.markdown("""
A **Tool** in LangGraph is a formalized interface bridging probabilistic LLMs to deterministic computing environments.
Tools provide the model with actionable capabilities: querying SQL databases, calculating tax figures,
calling microservice APIs, or mutating enterprise state.
When you bind tools to a model via `model.bind_tools(tools)`, LangChain inspects the function signatures,
extracts OpenAPI-compliant **JSON schemas**, and transmits them to the LLM's native tool calling endpoint.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    tool_blueprint_dot = """
    digraph ToolArchitectureBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_authoring {
            label = "1. Tool Definition Layer (@tool / StructuredTool)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            PYTHON_FN [label="Python Function with Type Hints & Docstring\\ndef query_database(query: str, limit: int = 10) -> str: ...", fillcolor="#0F172A", color="#38BDF8"];
            PYDANTIC_ARGS [label="Pydantic ArgsSchema (Optional for Complex Payloads)\\nclass QueryInput(BaseModel): ...", fillcolor="#1E1E38", color="#8B5CF6"];
            JSON_SCHEMA [label="OpenAPI / JSON Schema Extractor\\n{name: 'query_database', description: '...', parameters: {...}}", fillcolor="#2E1065", color="#C084FC"];

            PYTHON_FN -> JSON_SCHEMA;
            PYDANTIC_ARGS -> JSON_SCHEMA;
        }

        subgraph cluster_binding {
            label = "2. LLM Model Binding Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            BIND_CALL [label="model.bind_tools([query_database])\\nInjects JSON schemas into LLM API payload", fillcolor="#0F172A", color="#38BDF8"];
            LLM_REASON [label="LLM Ingestion & Reasoning\\nEvaluates user intent against tool schema descriptions", fillcolor="#064E3B", color="#34D399"];

            JSON_SCHEMA -> BIND_CALL;
            BIND_CALL -> LLM_REASON;
        }

        subgraph cluster_execution {
            label = "3. Tool Invocation & State Feedback";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            AI_CALL [label="AIMessage with tool_calls\\n[{'name': 'query_database', 'args': {...}, 'id': 'call_1'}]", fillcolor="#0F172A", color="#F59E0B"];
            TOOL_EXEC [label="ToolNode Execution\\nRuns function, captures return or handles exceptions", fillcolor="#431407", color="#FB923C"];
            TOOL_MSG [label="ToolMessage Committed to Channel\\ncontent = '{result: ...}', tool_call_id = 'call_1'", fillcolor="#0F172A", color="#34D399"];

            LLM_REASON -> AI_CALL;
            AI_CALL -> TOOL_EXEC;
            TOOL_EXEC -> TOOL_MSG;
        }
    }
    """
    st.graphviz_chart(tool_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Rule:</b> The quality of tool execution depends directly on the quality of your
docstring and parameter descriptions. The LLM reads the description to decide <i>when</i> to call the tool
and the parameter types to decide <i>what</i> arguments to supply.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: WHAT IS A TOOL?
# ==============================================================================
st.header("2. What is a Tool?: Deterministic Capabilities and External API Gateways", divider="rainbow")

st.markdown("""
An LLM alone is a closed, probabilistic text generator limited by its training cutoff and lack of private context.
A **Tool** gives the LLM deterministic, real-time agency:
1. **Fact Grounding**: Fetching live database records, real-time stock prices, or vector embeddings.
2. **Accurate Computation**: Delegating symbolic math and arithmetic to Python or calculator modules.
3. **Action Execution**: Sending emails, writing database rows, or triggering Kubernetes deployments.
""")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("#### Without Tools (Probabilistic)")
    st.error("User: What is 93821 multiplied by 88219?\\nLLM: It is approximately 8,276,800,000... (often hallucinates last digits)")

with col_t2:
    st.markdown("#### With Tools (Deterministic)")
    st.success("User: What is 93821 multiplied by 88219?\\nTool Call: `multiply(a=93821, b=88219)` -> Returns `8276704899`\\nLLM: The exact product is 8,276,704,899.")

# ==============================================================================
# SECTION 3: CREATING TOOLS
# ==============================================================================
st.header("3. Creating Tools: The @tool Decorator and StructuredTool Primitives", divider="rainbow")

st.markdown("""
LangChain provides the `@tool` decorator to convert standard Python functions into
fully validated `BaseTool` instances:
""")

st.code("""
from langchain_core.tools import tool

@tool
def get_weather_forecast(city: str, units: str = "celsius") -> str:
    \"\"\"Fetch the current weather forecast for a specified city.
    
    Args:
        city: The name of the city (e.g. 'London', 'Tokyo').
        units: Temperature scale ('celsius' or 'fahrenheit'). Defaults to celsius.
    \"\"\"
    # Production implementation calling external weather API
    return f"Weather in {city}: 18 degrees {units}, partly cloudy."
""", language="python")

st.markdown("""
For tools requiring custom initialization, stateful clients, or asynchronous batching,
use `StructuredTool.from_function()`:
""")

st.code("""
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="Sanitized SQL query to execute")
    read_only: bool = Field(default=True, description="Enforce SELECT queries only")

db_tool = StructuredTool.from_function(
    func=execute_sql_query,
    name="sql_analytics_query",
    description="Execute read-only SQL queries against the analytical warehouse.",
    args_schema=DatabaseQueryInput
)
""", language="python")

# ==============================================================================
# SECTION 4: TOOL SCHEMAS
# ==============================================================================
st.header("4. Tool Schemas: Pydantic Validation, Docstring Extraction, and Schema Binding", divider="rainbow")

st.markdown("""
When `@tool` is applied, LangChain analyzes:
1. **Function Name**: Becomes the tool name sent to the API.
2. **Docstring Summary**: Becomes the tool description explaining *purpose*.
3. **Type Annotations**: Translated into JSON Schema types (`str` -> `string`, `int` -> `integer`).
4. **Parameter Docstrings**: Extracted into parameter-level descriptions.
""")

st.code("""
# Inspecting the auto-generated JSON schema:
print(get_weather_forecast.name)
# 'get_weather_forecast'

print(get_weather_forecast.description)
# 'Fetch the current weather forecast for a specified city.'

print(get_weather_forecast.args_schema.schema())
# {
#   "title": "get_weather_forecastSchema",
#   "type": "object",
#   "properties": {
#     "city": {"title": "City", "type": "string", "description": "The name of the city..."},
#     "units": {"title": "Units", "type": "string", "default": "celsius"}
#   },
#   "required": ["city"]
# }
""", language="python")

# ==============================================================================
# SECTION 5: TOOL ARGUMENTS
# ==============================================================================
st.header("5. Tool Arguments: Type Coercion, Complex Payloads, and Default Parameters", divider="rainbow")

st.markdown("""
Frontier LLMs emit tool arguments as JSON text strings.
LangChain automatically deserializes and validates them against the tool's Pydantic schema before invocation.
""")

arg_df = pd.DataFrame([
    {
        "Python Type": "str",
        "JSON Schema Type": "string",
        "Validation Behavior": "Direct string validation"
    },
    {
        "Python Type": "int, float",
        "JSON Schema Type": "integer, number",
        "Validation Behavior": "Rejects non-numeric strings; coerces valid numbers"
    },
    {
        "Python Type": "List[str]",
        "JSON Schema Type": "array of strings",
        "Validation Behavior": "Ensures valid JSON list structure"
    },
    {
        "Python Type": "Literal['A', 'B']",
        "JSON Schema Type": "string (enum)",
        "Validation Behavior": "Restricts argument to predefined constant options"
    }
])
st.table(arg_df)

# ==============================================================================
# SECTION 6: TOOL RESULTS
# ==============================================================================
st.header("6. Tool Results: Return Types, JSON Serialization, and State Artifact Extraction", divider="rainbow")

st.markdown("""
What should a tool return?
- **String / Dict / Pydantic Model**: Converted to string and wrapped in `ToolMessage(content=...)`.
- **Tool Artifacts (`response_format='content_and_artifact'`)**: Enables returning raw data (like DataFrames or images) directly to the graph state without passing heavy tokens into the LLM context window!
""")

st.code("""
@tool(response_format="content_and_artifact")
def generate_customer_report(account_id: str) -> tuple[str, dict]:
    \"\"\"Generates audit report for account.\"\"\"
    raw_data = {"account_id": account_id, "transactions": [102.5, 94.0, 510.2]}
    summary_message = f"Report generated: 3 transactions found for {account_id}."
    
    # Returns (content_for_llm, heavy_artifact_for_state)
    return summary_message, raw_data
""", language="python")

# ==============================================================================
# SECTION 7: TOOL ERRORS
# ==============================================================================
st.header("7. Tool Errors: Exception Handling, Retry Policies, and Fallback Reporting", divider="rainbow")

st.markdown("""
In production, external APIs fail (network timeouts, invalid IDs, database errors).
If an exception crashes the tool, the graph must not crash.
Instead, catch exceptions and return an informative error message inside `ToolMessage`
so the LLM can self-correct or notify the user gracefully.
""")

st.code("""
@tool
def fetch_api_data(endpoint: str) -> str:
    \"\"\"Query external microservice endpoint.\"\"\"
    try:
        response = call_external_api(endpoint)
        return str(response)
    except ConnectionError:
        # Return error message directly to model context
        return "Error: External API timed out. Please retry with a different endpoint or report outage."
    except Exception as e:
        return f"Error executing tool: {type(e).__name__} - {str(e)}"
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Enterprise Production Tool Suite with Error Resilience", divider="rainbow")

st.markdown("""
Here is an end-to-end executable tool suite demonstrating strict Pydantic schemas,
custom argument validation, and error-tolerant execution:
""")

st.code("""
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. Pydantic Input Schemas
class CreditTransferInput(BaseModel):
    source_account: str = Field(description="Source account ID (format ACC-XXXX)")
    destination_account: str = Field(description="Destination account ID (format ACC-XXXX)")
    amount_usd: float = Field(gt=0, description="Positive transaction amount in USD")
    priority: Literal["STANDARD", "EXPRESS"] = Field(default="STANDARD", description="Transfer SLA speed")

# 2. Production Tool Definition
@tool(args_schema=CreditTransferInput)
def execute_wire_transfer(
    source_account: str,
    destination_account: str,
    amount_usd: float,
    priority: str = "STANDARD"
) -> str:
    \"\"\"Execute a secure wire transfer between two accounts with ledger validation.\"\"\"
    try:
        # Enforce validation bounds
        if amount_usd > 25000.0:
            return f"Error: Transfers over 25,000 USD require human supervisor approval."
        
        return f"SUCCESS: Transferred {amount_usd:.2f} USD from {source_account} to {destination_account} via {priority} transfer."
    except Exception as e:
        return f"System Error during transfer: {str(e)}"

# 3. Model Binding
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
model_with_tools = model.bind_tools([execute_wire_transfer])

# 4. Invocation Demonstration
messages = [HumanMessage(content="Transfer 1400 USD from ACC-1092 to ACC-9941 right now.")]
response: AIMessage = model_with_tools.invoke(messages)

print("Emitted Tool Calls:", response.tool_calls)
# Result: [{'name': 'execute_wire_transfer', 'args': {'amount_usd': 1400.0, 'destination_account': 'ACC-9941', 'source_account': 'ACC-1092', 'priority': 'STANDARD'}, 'id': 'call_1'}]
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Tool Definition, Schema Inspector & Execution Sandbox", divider="rainbow")

st.markdown("""
Inspect how Python tools translate into JSON schemas and test simulated tool execution
under both normal conditions and simulated external API failure modes.
""")

eng = ToolSimulationEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Select & Configure Tool")
    selected_tool = st.selectbox(
        "Select Enterprise Tool:",
        list(eng.registered_tools.keys())
    )
    
    meta = eng.registered_tools[selected_tool]
    st.info(f"**Description**: {meta['description']}")

    # Dynamic argument inputs
    input_args = {}
    if selected_tool == "fetch_customer_record":
        input_args["customer_id"] = st.text_input("Customer ID:", value="CUST-1049")
        input_args["include_history"] = st.checkbox("Include 90-Day Audit History", value=True)
    elif selected_tool == "calculate_tax_liability":
        input_args["amount_usd"] = st.number_input("Transaction Amount (USD):", value=15000.0, step=1000.0)
        input_args["jurisdiction"] = st.selectbox("State Jurisdiction:", ["CA", "NY", "TX", "WA"])
    else:
        input_args["transaction_id"] = st.text_input("Transaction UUID:", value="tx_88a92b")
        input_args["force"] = st.checkbox("Force Rollback", value=False)

    simulate_err = st.checkbox("🔥 Simulate External API Exception / Timeout", value=False)
    run_btn = st.button("🚀 Execute Tool Sandbox", type="primary")

with wb_col2:
    st.subheader("📋 JSON Schema & Execution Telemetry")
    st.markdown("#### OpenAPI / JSON Schema Generated for LLM")
    st.json({
        "name": selected_tool,
        "description": meta["description"],
        "parameters": {
            "type": "object",
            "properties": meta["parameters"],
            "required": meta["required"]
        }
    })

    if run_btn:
        st.markdown("#### Execution Result Payload")
        result = eng.execute_tool(selected_tool, input_args, simulate_err)
        if result["status"] == "SUCCESS":
            st.success("Tool executed successfully!")
            st.json(result["data"])
        else:
            st.error(f"Captured Exception: {result['error_type']}")
            st.warning(result["message"])

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: How does model.bind_tools() communicate tool definitions to LLM providers?", expanded=False):
    st.markdown("""
    **Answer**:
    `model.bind_tools()` serializes each tool's name, docstring, and Pydantic args schema into an OpenAPI-compliant
    JSON schema dictionary. It attaches this schema to the provider's API request payload (e.g. the `tools` parameter in OpenAI or Anthropic).
    The model uses this schema to evaluate whether to answer directly or generate structured `tool_calls`.
    """)

with st.expander("Q2: What is the purpose of response_format='content_and_artifact' in LangChain tools?", expanded=False):
    st.markdown("""
    **Answer**:
    By default, everything a tool returns is converted to a string and passed into the LLM context via `ToolMessage(content=...)`.
    For heavy objects (e.g. Pandas DataFrames, binary files, 10,000 raw JSON records), this wastes context tokens.
    `response_format="content_and_artifact"` allows the tool to return a 2-tuple: `(summary_str_for_llm, raw_object_for_state)`,
    keeping the context window lean while saving raw data into graph channels.
    """)

with st.expander("Q3: Why should a tool never raise an unhandled exception in production?", expanded=False):
    st.markdown("""
    **Answer**:
    An unhandled exception in a tool crashes the entire superstep and terminates the graph run.
    Instead, tools should catch exceptions and return an error string: `"Error: Database timeout..."`.
    This enables the LLM to inspect the error in a `ToolMessage`, re-try with different arguments, or explain the failure gracefully to the user.
    """)

with st.expander("Q4: How do docstrings impact tool selection accuracy?", expanded=False):
    st.markdown("""
    **Answer**:
    Frontier models use semantic matching between user intent and tool docstrings.
    A vague docstring (`"Does calculations"`) results in erratic tool selection.
    A precise docstring (`"Calculates municipal and federal tax liability for USD transactions in CA, NY, TX"`) provides
    exact semantic triggers, maximizing routing precision.
    """)

with st.expander("Q5: What is the difference between @tool and StructuredTool.from_function()?", expanded=False):
    st.markdown("""
    **Answer**:
    `@tool` is a convenience decorator that automatically infers schemas from function type hints and docstrings.
    `StructuredTool.from_function()` provides explicit programmatic control: you can inject custom Pydantic schemas,
    separate sync and async execution functions (`coroutine=...`), and configure custom error handlers dynamically.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Unhandled Type Coercion Crashes in Tool Functions")
    st.markdown("""
    *Mechanism*: Assuming tool arguments will always arrive as valid types without Pydantic schema validation.  
    *Consequence*: Python raises `TypeError` or `ValueError` at runtime, crashing the entire agent workflow.  
    *Fix*: Use explicit Pydantic schemas with strict type hints to validate arguments before function execution.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing Parameter Descriptions in Tool Schemas")
    st.markdown("""
    *Mechanism*: Writing tool functions with type hints but omitting parameter descriptions in docstrings.  
    *Consequence*: The LLM hallucinates incorrect parameter formats (e.g. passing a full state name 'California' instead of 2-letter 'CA').  
    *Fix*: Always document argument formats and valid options explicitly in Google or Sphinx docstring format.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Overloading Tools with Massive Return Payloads")
    st.markdown("""
    *Mechanism*: Returning a 500KB JSON payload directly into `ToolMessage.content`.  
    *Consequence*: Exhausts LLM context windows and drastically increases latency and billing.  
    *Fix*: Truncate payloads, return summaries, or use `response_format='content_and_artifact'`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Side-Effect Tools Without Confirmation")
    st.markdown("""
    *Mechanism*: Giving an LLM direct access to destructive tools (`delete_database`, `transfer_funds`) without human-in-the-loop approvals.  
    *Consequence*: Hallucinated arguments cause irreversible data corruption or unintended financial transactions.  
    *Fix*: Gate destructive tools behind LangGraph `interrupt()` checkpoints.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**@tool**\\nDecorator converting standard Python functions into validated LangChain tools with auto-generated schemas.")
with col_fc2:
    st.info("**bind_tools()**\\nMethod attaching JSON tool schemas to ChatOpenAI/ChatAnthropic models for native tool calling.")
with col_fc3:
    st.info("**StructuredTool**\\nCore tool abstraction allowing explicit Pydantic schema binding and sync/async function routing.")
with col_fc4:
    st.info("**Tool Artifact**\\nA mechanism to return heavy data objects to graph state without bloating the LLM prompt window.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to define tools using the @tool decorator and type hints.", value=True)
c2 = st.checkbox("I understand how LangChain generates OpenAPI JSON schemas from docstrings.", value=True)
c3 = st.checkbox("I know how to bind tools to chat models using model.bind_tools().", value=True)
c4 = st.checkbox("I understand how to handle tool errors gracefully to allow LLM self-correction.", value=True)
c5 = st.checkbox("I know how to return heavy data as artifacts using content_and_artifact format.", value=True)
c6 = st.checkbox("I understand how to guard destructive tools with human approval checks.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 12.1 'Tools' with full architectural and practical comprehension!")
