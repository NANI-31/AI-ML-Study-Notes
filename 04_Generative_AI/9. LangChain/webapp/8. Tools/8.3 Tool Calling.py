import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="8.3 Tool Calling - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("8.3 Tool Calling: Binding, Invocations, Parallel Execution & Tool Messages")
st.caption("Chapter 8: Tools | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Modern frontier chat models (OpenAI, Anthropic Claude, Google Gemini) feature **native tool calling capabilities**. 
Rather than forcing models to generate unstructured text and hoping a regex matches function calls, models are 
explicitly trained to emit structured `tool_calls` payloads whenever a query requires external action.

In this lesson, you will master the end-to-end tool calling lifecycle:
1. **Binding Tools to Models**: Attaching tool definitions via `model.bind_tools(tools)`.
2. **Tool Calls Inspection**: Deconstructing `AIMessage.tool_calls` (`id`, `name`, `args`).
3. **Argument Validation**: Safely deserializing arguments into validated Python parameters.
4. **Tool Results**: Returning observations via `ToolMessage` paired with matching `tool_call_id`.
5. **Multiple Tool Calls**: Managing multiple consecutive tool dispatches across dialogue turns.
6. **Parallel Tool Calls**: Concurrently executing independent tool calls emitted in a single model turn.
7. **Tool Error Interception**: Formatting tool failure messages to allow LLM self-correction.
8. **Tool Retries**: Implementing automated retry policies to handle transient tool failures.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockToolCallRecord:
    """Represents a simulated tool invocation event in module scope."""
    def __init__(self, call_id: str, tool_name: str, args: Dict[str, Any], status: str, result_preview: str, duration_ms: int):
        self.call_id = call_id
        self.tool_name = tool_name
        self.args = args
        self.status = status
        self.result_preview = result_preview
        self.duration_ms = duration_ms

class MockToolCallingSimulator:
    """Simulates tool calling, parallel dispatch, and error handling in module scope."""
    def __init__(self):
        pass

    def simulate_tool_execution(
        self,
        query: str,
        enable_parallel: bool,
        simulate_tool_error: bool
    ) -> Dict[str, Any]:
        records: List[MockToolCallRecord] = []
        total_time_ms = 0

        # Model Step 1: Tool Decision
        query_lower = query.lower()
        tool_calls = []

        if "weather" in query_lower:
            tool_calls.append(("get_weather", {"city": "Tokyo", "units": "celsius"}, "call_wth_910"))
        if "stock" in query_lower or "price" in query_lower:
            tool_calls.append(("get_stock_quote", {"ticker": "NVDA"}, "call_stk_482"))
        if not tool_calls:
            # Default multiple tools
            tool_calls = [
                ("get_weather", {"city": "Paris", "units": "celsius"}, "call_wth_101"),
                ("get_stock_quote", {"ticker": "AAPL"}, "call_stk_102")
            ]

        # Execute Tools
        if enable_parallel and len(tool_calls) > 1:
            # Parallel execution: latency bounded by max
            durations = [140, 195]
            max_lat = max(durations)
            total_time_ms += max_lat
            
            for idx, (t_name, t_args, c_id) in enumerate(tool_calls):
                if simulate_tool_error and idx == 0:
                    status = "ERROR"
                    res = f"ToolException: Connection timeout accessing weather provider for {t_args.get('city')}."
                else:
                    status = "SUCCESS"
                    res = f"Weather in {t_args.get('city')}: 21°C, Sunny." if "weather" in t_name else f"Stock {t_args.get('ticker')}: $128.45 USD (+3.2%)."
                
                records.append(MockToolCallRecord(c_id, t_name, t_args, status, res, durations[idx % len(durations)]))
        else:
            # Sequential execution: latency sum
            for idx, (t_name, t_args, c_id) in enumerate(tool_calls):
                dur = 150
                total_time_ms += dur
                if simulate_tool_error and idx == 0:
                    status = "ERROR"
                    res = f"ToolException: Database error processing request."
                else:
                    status = "SUCCESS"
                    res = f"Result payload for {t_name} with args {t_args}."
                records.append(MockToolCallRecord(c_id, t_name, t_args, status, res, dur))

        return {
            "total_calls": len(records),
            "execution_mode": "Parallel (asyncio.gather)" if enable_parallel else "Sequential (for-loop)",
            "total_latency_ms": total_time_ms,
            "records": records
        }

# --- SECTION 1: ARCHITECTURE BLUEPRINT & THE TOOL CALLING PROTOCOL ---
st.header("1. Architecture Blueprint: The Modern Tool Calling Protocol & ReAct Loop", divider="rainbow")

st.markdown("""
The architecture below depicts the modern two-turn **Tool Calling Protocol** standardized across 
OpenAI, Anthropic, and Google Gemini APIs. The model acts as the reasoning engine; the host application acts as the tool executor.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Tier 1: Ingestion & Model Binding
        subgraph cluster_binding_tier {
            label = "1. INTAKE &amp; BINDING TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserMsg [label="HumanMessage\\n'What is weather in Tokyo &amp; stock price of NVDA?'"];
            ModelWithTools [label="model.bind_tools([get_weather, get_stock])\\nInjected tool schemas into API payload"];
        }

        // Tier 2: Model Turn 1 (Tool Emission)
        subgraph cluster_turn1 {
            label = "2. MODEL TURN 1 (DECISION &amp; TOOL CALL EMISSION)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            AIMessageOut [label="AIMessage with tool_calls\\ntool_calls=[\\n  {id: 'call_1', name: 'get_weather', args: {'city': 'Tokyo'}},\\n  {id: 'call_2', name: 'get_stock', args: {'ticker': 'NVDA'}}\\n]"];
        }

        // Tier 3: Host Execution Tier
        subgraph cluster_host_exec {
            label = "3. HOST EXECUTION &amp; PARALLEL DISPATCH TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ParallelDispatcher [label="asyncio.gather() / ToolNode\\nDispatches tool calls concurrently", color="#10B981"];
            Tool1Exec [label="Execute get_weather(city='Tokyo')\\nLatency: 140ms"];
            Tool2Exec [label="Execute get_stock(ticker='NVDA')\\nLatency: 180ms"];
        }

        // Tier 4: Observation Tier
        subgraph cluster_obs {
            label = "4. OBSERVATION TIER (ToolMessage Packaging)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ToolMsg1 [label="ToolMessage(tool_call_id='call_1', content='Tokyo: 22°C')"];
            ToolMsg2 [label="ToolMessage(tool_call_id='call_2', content='NVDA: $128.45')"];
            MsgHistory [label="Consolidated Conversation History\\n[HumanMessage, AIMessage, ToolMessage1, ToolMessage2]"];
        }

        // Tier 5: Model Turn 2 (Final Synthesis)
        subgraph cluster_turn2 {
            label = "5. MODEL TURN 2 (FINAL SYNTHESIS)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            FinalAIMessage [label="Final AIMessage (content='In Tokyo it is 22°C, and NVDA is trading at $128.45...')", color="#10B981"];
        }

        // Wiring
        UserMsg -> ModelWithTools;
        ModelWithTools -> AIMessageOut;
        
        AIMessageOut -> ParallelDispatcher [label="Extract .tool_calls"];
        ParallelDispatcher -> Tool1Exec;
        ParallelDispatcher -> Tool2Exec;

        Tool1Exec -> ToolMsg1;
        Tool2Exec -> ToolMsg2;

        ToolMsg1 -> MsgHistory;
        ToolMsg2 -> MsgHistory;
        AIMessageOut -> MsgHistory [style=dashed];

        MsgHistory -> FinalAIMessage [label="Invoke model second turn"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**The Crucial `tool_call_id` Invariant:**
Every `ToolMessage` **must** carry the exact `tool_call_id` emitted by the model's `AIMessage`. 
Provider APIs enforce this invariant strictly. If an API receives a `ToolMessage` with a mismatched or missing ID, 
it throws a fatal HTTP 400 Bad Request error.
""")

st.divider()

# --- SECTION 2: BINDING TOOLS TO MODELS ---
st.header("2. Binding Tools to Models (`model.bind_tools`)", divider="rainbow")

st.markdown("""
In LangChain, attaching tools to a chat model is accomplished via `.bind_tools()`. 
This returns a new Runnable that automatically passes the tools' JSON Schemas in every API request.
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 22°C, Clear skies."

@tool
def get_stock_price(ticker: str) -> str:
    """Get real-time stock price for a ticker."""
    return f"{ticker}: $182.40 USD."

tools = [get_weather, get_stock_price]

# Bind tools to the model
model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

# Invocation: model inspects user query and decides if tools are needed
response = model_with_tools.invoke("What is the current weather in Tokyo?")
print(response.tool_calls)
# Output: [{'name': 'get_weather', 'args': {'city': 'Tokyo'}, 'id': 'call_abc123'}]
''', language="python")

st.divider()

# --- SECTION 3: TOOL CALLS ---
st.header("3. Tool Calls (`AIMessage.tool_calls` Anatomy)", divider="rainbow")

st.markdown("""
When a tool-bound model decides to call one or more tools, the resulting `AIMessage` contains a `tool_calls` attribute. 
LangChain standardizes this across all providers (OpenAI, Anthropic, Google) into a clean list of dictionaries.
""")

st.subheader("Structure of a LangChain `ToolCall` Object")
st.code('''# Anatomy of a ToolCall dictionary:
{
    "name": "get_weather",          # str: The exact tool name to execute
    "args": {"city": "Tokyo"},       # dict: Validated, parsed Python arguments
    "id": "call_9481ad8291"          # str: Unique invocation ID assigned by model
}
''', language="python")

st.markdown("""
Notice that `args` is **already parsed** into a native Python dictionary! 
LangChain handles raw JSON deserialization automatically, sparing you from manual `json.loads()` calls.
""")

st.divider()

# --- SECTION 4: TOOL CALL ARGUMENTS ---
st.header("4. Tool Call Arguments (Validation & Unpacking)", divider="rainbow")

st.markdown("""
While the model outputs dictionary arguments, safe execution requires matching tool names to their 
physical Python callables and unpacking the arguments dictionary safely: `tool.invoke(tool_call['args'])`.
""")

st.code('''# Dynamic tool lookup and invocation pattern
tool_map = {tool.name: tool for tool in tools}

for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    if tool_name in tool_map:
        target_tool = tool_map[tool_name]
        # Execute tool with validated arguments
        observation = target_tool.invoke(tool_args)
        print(f"Observation from {tool_name}: {observation}")
    else:
        print(f"Warning: Model requested unregistered tool: {tool_name}")
''', language="python")

st.divider()

# --- SECTION 5: TOOL RESULTS ---
st.header("5. Tool Results (`ToolMessage` & Conversation Grounding)", divider="rainbow")

st.markdown("""
After executing a tool, you must package the result into a `ToolMessage` and pass it back to the model. 
This provides the LLM with the factual observation needed to synthesize the final user-facing response.
""")

st.code('''from langchain_core.messages import HumanMessage, ToolMessage

# Complete Two-Turn Conversation History
messages = [
    HumanMessage(content="What is the weather in Tokyo?"),
    response  # The AIMessage containing the tool_call
]

# Create matching ToolMessage
for tool_call in response.tool_calls:
    observation = tool_map[tool_call["name"]].invoke(tool_call["args"])
    messages.append(
        ToolMessage(
            tool_call_id=tool_call["id"],  # MUST MATCH the model's call id
            content=str(observation)
        )
    )

# Invoke model for Turn 2: Synthesizes final answer grounded in observation
final_response = model_with_tools.invoke(messages)
print(final_response.content)
# Output: "The current weather in Tokyo is 22°C with clear skies."
''', language="python")

st.divider()

# --- SECTION 6: MULTIPLE TOOL CALLS ---
st.header("6. Multiple Tool Calls (Sequential Multi-Turn Workflows)", divider="rainbow")

st.markdown("""
Complex reasoning often requires a sequence of tool calls across multiple turns:
1. Turn 1: Call `search_flights(destination="London")` ➔ returns flight options.
2. Turn 2: Model inspects results, asks to call `convert_currency(amount=450, from_curr="GBP", to_curr="USD")`.
3. Turn 3: Model receives currency conversion and delivers final itinerary.

This iterative loop of `Model ➔ Tool ➔ Model ➔ Tool ➔ Final Answer` forms the core of the **ReAct (Reason + Act)** pattern.
""")

st.divider()

# --- SECTION 7: PARALLEL TOOL CALLS ---
st.header("7. Parallel Tool Calls (Concurrent Multi-Tool Dispatch)", divider="rainbow")

st.markdown("""
When a user prompt contains multiple independent queries (e.g. *"Compare the weather in Tokyo and Paris"*), 
modern models emit **multiple tool calls in a single `AIMessage`**.
""")

st.subheader("Sequential vs Parallel Latency Timeline")
st.markdown("""
- **Sequential Execution**: `time = t(Tokyo) + t(Paris) = 200ms + 200ms = 400ms`.
- **Parallel Concurrency (`asyncio.gather`)**: `time = max(t(Tokyo), t(Paris)) = ~205ms` (50% latency reduction).
""")

st.code('''import asyncio
from langchain_core.messages import ToolMessage

async def execute_parallel_tool_calls(aimessage, tool_map):
    """Executes all tool calls emitted in an AIMessage concurrently."""
    async def run_single_call(call):
        tool = tool_map[call["name"]]
        # Asynchronously invoke tool
        result = await tool.ainvoke(call["args"])
        return ToolMessage(tool_call_id=call["id"], content=str(result))

    # Run all tools concurrently across non-blocking coroutines
    tool_messages = await asyncio.gather(*[
        run_single_call(call) for call in aimessage.tool_calls
    ])
    return tool_messages
''', language="python")

st.divider()

# --- SECTION 8: TOOL ERRORS ---
st.header("8. Tool Errors (Exception Interception & Self-Correction)", divider="rainbow")

st.markdown("""
External systems fail: databases throw connection errors, APIs rate limit, and third-party servers go down. 
If a tool raises an unhandled exception, the entire agent pipeline crashes.

**Best Practice**: Catch exceptions inside tools or tool runners, and return the error message inside the `ToolMessage`. 
Frontier LLMs can read the error message, diagnose the failure, and attempt an alternative action or notify the user.
""")

st.code('''from langchain_core.messages import ToolMessage

def safe_tool_executor(tool_call, tool_map):
    tool_name = tool_call["name"]
    tool = tool_map.get(tool_name)
    
    if not tool:
        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Error: Tool '{tool_name}' does not exist. Available tools: {list(tool_map.keys())}"
        )
        
    try:
        result = tool.invoke(tool_call["args"])
        return ToolMessage(tool_call_id=tool_call["id"], content=str(result))
    except Exception as exc:
        # Pass exception message back to LLM to allow self-correction
        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Error executing tool '{tool_name}': {str(exc)}. Please check your arguments and try again."
        )
''', language="python")

st.divider()

# --- SECTION 9: TOOL RETRIES ---
st.header("9. Tool Retries (Automated Recovery & Exponential Backoff)", divider="rainbow")

st.markdown("""
For transient tool failures (HTTP 503, rate limits, socket resets), wrap tool execution in automated 
retry policies using tenacity or LangChain's `.with_retry()`.
""")

st.code('''from langchain_core.tools import tool

# Defining a tool with retry resilience
@tool
def fetch_financial_filings(ticker: str) -> str:
    """Fetch SEC 10-K filings with retry resilience."""
    # Production API call wrapped with retry backoff
    return f"Retrieved 10-K filing for {ticker}."

resilient_tool = fetch_financial_filings.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
''', language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Resilient Multi-Tool Execution Loop", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates a hardened tool execution loop 
featuring parallel asynchronous dispatch, error containment, and conversational grounding.
""")

st.code('''import asyncio
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

logger = logging.getLogger("EnterpriseToolLoop")

@tool
def check_order_status(order_id: str) -> str:
    """Check fulfillment status of an e-commerce order."""
    if order_id == "FAIL-99":
        raise ConnectionResetError("Warehouse database replica unreachable.")
    return f"Order {order_id}: Shipped via FedEx (Tracking #94819283)."

@tool
def check_inventory_level(sku: str) -> str:
    """Check remaining warehouse inventory stock for a product SKU."""
    return f"SKU {sku}: 420 units available across regional fulfillment centers."

class EnterpriseToolExecutionEngine:
    """Production two-turn tool execution pipeline with parallel execution and error handling."""
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [check_order_status, check_inventory_level]
        self.tool_map = {t.name: t for t in self.tools}
        self.bound_model = self.model.bind_tools(self.tools)

    async def _execute_single_call(self, tool_call: Dict[str, Any]) -> ToolMessage:
        call_id = tool_call["id"]
        name = tool_call["name"]
        args = tool_call["args"]

        if name not in self.tool_map:
            return ToolMessage(tool_call_id=call_id, content=f"Unknown tool '{name}'.")

        try:
            logger.info(f"Executing tool '{name}' with args {args}")
            tool_instance = self.tool_map[name]
            result = await tool_instance.ainvoke(args)
            return ToolMessage(tool_call_id=call_id, content=str(result))
        except Exception as exc:
            logger.error(f"Tool '{name}' execution failed: {exc}")
            return ToolMessage(
                tool_call_id=call_id,
                content=f"ExecutionError in '{name}': {str(exc)}"
            )

    async def run(self, user_query: str) -> str:
        messages: List[BaseMessage] = [HumanMessage(content=user_query)]
        
        # Turn 1: Model determines tools to run
        ai_response: AIMessage = await self.bound_model.ainvoke(messages)
        messages.append(ai_response)

        # If model called no tools, return direct answer
        if not ai_response.tool_calls:
            return ai_response.content

        # Turn 2: Dispatch all tools concurrently in parallel
        tool_observations = await asyncio.gather(*[
            self._execute_single_call(call) for call in ai_response.tool_calls
        ])
        messages.extend(tool_observations)

        # Turn 3: Second model invocation grounded on tool observations
        final_answer = await self.bound_model.ainvoke(messages)
        return final_answer.content
''', language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Tool Calling & Parallel Dispatch Simulator", divider="rainbow")

st.markdown("""
Test how modern models emit structured `tool_calls`, compare sequential vs parallel execution latency, 
and observe how error interception feeds diagnostic feedback back to the LLM.
""")

sim = MockToolCallingSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("📥 Input Complex User Query")
    sample_queries = {
        "Multi-Tool Parallel Query": "Compare the current weather in Tokyo with the stock price of NVDA.",
        "Single Weather Query": "What is the weather in Tokyo right now?",
        "Single Financial Query": "Can you check the current stock price for NVDA?"
    }
    
    selected_preset = st.selectbox("Load Sample Query", list(sample_queries.keys()))
    user_query_input = st.text_input("User Prompt", value=sample_queries[selected_preset])

    st.markdown("#### Execution Parameters")
    parallel_toggle = st.checkbox("Enable Parallel Concurrency (asyncio.gather)", value=True)
    error_simulation = st.checkbox("Simulate Upstream Tool Exception (Timeout/Error)", value=False)

with col_wb2:
    st.subheader("⚙️ Configured Tools in Model Binding")
    st.info("""
    **Bound Tools in Environment:**
    1. `get_weather(city: str, units: str)`
    2. `get_stock_quote(ticker: str)`
    
    **Execution Strategy:**
    - Tool calls extracted from `AIMessage.tool_calls`
    - Packaged into `ToolMessage` with matching `tool_call_id`
    """)

if st.button("⚡ Execute Tool Calling Pipeline", type="primary"):
    res = sim.simulate_tool_execution(user_query_input, parallel_toggle, error_simulation)
    
    st.subheader("📊 Execution Diagnostics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tools Invoked", res["total_calls"])
    m2.metric("Dispatch Strategy", res["execution_mode"])
    m3.metric("Total Tool Latency", f"{res['total_latency_ms']} ms")

    st.write("#### ⏱️ Tool Execution Spans & Observations:")
    for rec in res["records"]:
        icon = "🟢" if rec.status == "SUCCESS" else "🔴"
        with st.expander(f"{icon} {rec.tool_name} (ID: `{rec.call_id}`) - {rec.duration_ms} ms", expanded=True):
            st.caption(f"Status: **{rec.status}**")
            st.json({"tool_name": rec.tool_name, "arguments": rec.args, "tool_call_id": rec.call_id})
            st.write("**Returned Tool Observation (`ToolMessage.content`)**:")
            if rec.status == "SUCCESS":
                st.success(rec.result_preview)
            else:
                st.error(rec.result_preview)

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is model.bind_tools() superior to manually prompting an LLM with 'Output JSON to call a tool'?",
     "model.bind_tools() leverages provider-native function calling APIs trained specifically on tool calling tokens. The model emits structured tool_calls dictionaries at the token generation layer, guaranteeing valid JSON formatting, accurate argument schema alignment, and distinct tool call IDs, completely eliminating brittle regex parsing."),
    ("Why must ToolMessage.tool_call_id strictly match AIMessage.tool_calls[i]['id']?",
     "Provider APIs (OpenAI, Anthropic) use tool_call_id as a cryptographic foreign key to map specific tool observations back to their corresponding function calls in conversation memory. If an ID is missing, truncated, or mismatched, the API rejects the request with an HTTP 400 error."),
    ("How does parallel tool calling improve latency in multi-step AI agent workflows?",
     "When an agent needs multiple independent observations (e.g. searching 3 distinct documents or querying 2 databases), parallel tool calling emits all tool calls in a single turn. Executing them concurrently via asyncio.gather() bounds overall latency to max(t_i) rather than the sum of all latencies."),
    ("What should your tool execution runner do when a tool raises an unhandled exception?",
     "The runner should intercept the exception, log the traceback, and construct a ToolMessage containing a concise error description. Passing the error back into conversation history allows the LLM to understand what failed, adjust its parameters, and self-correct on the next turn."),
    ("What is the difference between tool_choice='auto', tool_choice='required', and tool_choice='none'?",
     "'auto' (default) allows the model to decide whether to call a tool or reply with normal text. 'required' forces the model to call at least one tool before generating a conversational response. 'none' disables tool calling entirely, forcing pure text generation.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Omission of ToolMessage in Conversational Turns. If a model emits tool_calls and your application invokes the model again without appending the corresponding ToolMessage objects, the API crashes with an invalid conversation history error.")
    st.error("**Trap 2 (Breaking Error)**: Modifying Tool Call IDs. Attempting to generate custom IDs (e.g. id=1, 2) instead of preserving the exact random hash string provided by the model breaks the provider's internal state machine.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Infinite Tool Calling Loops. An agent that repeatedly receives failed tool observations may retry calling the exact same failing tool infinitely. Always implement maximum turn limits (e.g. max_iterations=5).")
    st.warning("**Trap 4 (Quality Warning)**: Hallucinating Tool Names. When models are prompted with many tools with similar names, they may invent hybrid tool names. Always validate tool_call['name'] against a registered tool whitelist.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: model.bind_tools()"):
        st.markdown("**Injects tool JSON schemas into the model API payload and returns a Runnable.**")
    with st.expander("🎴 Flashcard 2: AIMessage.tool_calls"):
        st.markdown("**Contains list of dictionaries with name, parsed args, and unique id.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: ToolMessage Invariant"):
        st.markdown("**Carries the tool observation and MUST include tool_call_id matching the request.**")
    with st.expander("🎴 Flashcard 4: Parallel Tool Execution"):
        st.markdown("**asyncio.gather() executes multiple tool calls concurrently to minimize latency.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand how to bind tools to chat models using model.bind_tools().", value=True)
chk2 = st.checkbox("I can inspect and parse AIMessage.tool_calls dictionaries.", value=True)
chk3 = st.checkbox("I know how to construct ToolMessage objects with matching tool_call_id values.", value=True)
chk4 = st.checkbox("I can execute multiple tool calls concurrently using asyncio.gather().", value=True)
chk5 = st.checkbox("I know how to intercept tool exceptions and feed error observations back to the LLM.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 tool calling interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 8.3 Tool Calling. Ready for 8.4 Real-World Tools!")
