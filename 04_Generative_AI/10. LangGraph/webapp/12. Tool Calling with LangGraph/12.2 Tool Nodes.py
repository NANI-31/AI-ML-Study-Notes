import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="12.2 Tool Nodes - LangGraph Curriculum",
    page_icon="⚙️",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-tool {
        background-color: #431407;
        color: #FB923C;
        border: 1px solid #EA580C;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-loop {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class ToolNodeExecutionSimulator:
    """
    Simulates ToolNode batch execution, parallel tool calls, cyclic routing,
    and recursion limit termination.
    """
    def __init__(self):
        self.available_tools = {
            "search_customer_db": "SELECT customer data from internal CRM",
            "verify_credit_score": "Calculate FICO risk metric",
            "execute_refund": "Trigger Stripe refund transaction"
        }

    def simulate_agent_turn(
        self,
        query: str,
        max_iterations: int,
        inject_invalid_args: bool
    ) -> Dict[str, Any]:
        trace = []
        iteration = 0
        current_state = "LLM_REASONING"
        completed = False

        trace.append(f"Turn started with query: '{query[:40]}...'")

        while iteration < max_iterations and not completed:
            iteration += 1
            trace.append(f"--- Iteration {iteration}: LLM Evaluates Intent ---")
            
            if "refund" in query.lower() and iteration == 1:
                tool_call = {
                    "name": "search_customer_db",
                    "args": {"query": "customer_id_992"},
                    "id": f"call_{iteration}_1"
                }
                trace.append(f"LLM decided to call tool: [{tool_call['name']}] with args {tool_call['args']}")
                trace.append(f"ToolNode executed [{tool_call['name']}] -> Returned: 'Customer record located: Alice M.'")
                trace.append("Routing edge: Tool executed -> Loop back to LLM reasoning node.")
            elif "refund" in query.lower() and iteration == 2:
                if inject_invalid_args:
                    tool_call = {
                        "name": "execute_refund",
                        "args": {"amount": -50.0}, # Invalid negative
                        "id": f"call_{iteration}_2"
                    }
                    trace.append(f"LLM emitted tool call with INVALID ARGS: {tool_call['args']}")
                    trace.append(f"ToolNode Pre-Flight Validation REJECTED call: 'ValidationError: amount must be > 0'")
                    trace.append("ToolNode emitted ToolMessage with validation error -> LLM will self-correct.")
                else:
                    tool_call = {
                        "name": "execute_refund",
                        "args": {"amount": 50.0, "reason": "Customer request"},
                        "id": f"call_{iteration}_2"
                    }
                    trace.append(f"LLM decided to call tool: [{tool_call['name']}] with args {tool_call['args']}")
                    trace.append(f"ToolNode executed [{tool_call['name']}] -> Returned: 'Refund of 50.00 USD committed successfully.'")
                    trace.append("Routing edge: Tool executed -> Loop back to LLM reasoning node.")
            else:
                trace.append("LLM decided: No further tools required -> Emitted final natural language response.")
                completed = True
                break

        if not completed and iteration >= max_iterations:
            trace.append(f"🚨 GraphTerminated: Recursion limit reached ({max_iterations} iterations). Loop halted.")

        return {
            "iterations_used": iteration,
            "completed": completed,
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: ToolNode Lifecycle and Graph Routing Cycles", divider="rainbow")

st.markdown("""
In LangGraph, executing tools is not an ad-hoc inline step inside the LLM node.
Instead, it is modeled as an independent graph node: **`ToolNode`**.
The LLM node emits an `AIMessage` with `tool_calls`. A **conditional edge** (`tools_condition`) inspects
the message: if tools are requested, it routes to `ToolNode`; otherwise, it routes to `END`.
After executing tools, `ToolNode` loops directly back to the LLM node, completing the foundational **ReAct cycle**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    tool_node_blueprint_dot = """
    digraph ToolNodeArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        START_NODE [label="START (User Request Ingestion)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_react_loop {
            label = "The Canonical ReAct Execution Loop";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            LLM_NODE [label="LLM Node: call_model(state)\\n- Ingests full message history\\n- Emits AIMessage(content, tool_calls)", fillcolor="#064E3B", color="#34D399"];
            ROUTER [label="tools_condition(state)\\nAre there tool_calls on latest AIMessage?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
            TOOL_NODE [label="ToolNode(tools=[tool_1, tool_2, ...])\\n- Parses tool_calls list\\n- Executes tools (Parallel or Serial)\\n- Emits ToolMessage for each call", fillcolor="#431407", color="#FB923C"];

            LLM_NODE -> ROUTER [label="Emits AIMessage"];
            ROUTER -> TOOL_NODE [label="Yes (tool_calls present)"];
            TOOL_NODE -> LLM_NODE [label="Returns ToolMessages (Loop Back)"];
        }

        END_NODE [label="END (Final Answer Emitted to User)", fillcolor="#0F172A", color="#64748B"];

        START_NODE -> LLM_NODE;
        ROUTER -> END_NODE [label="No (tools completed / text response)"];
    }
    """
    st.graphviz_chart(tool_node_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Key Architectural Takeaway:</b> By decoupling tool execution into a dedicated <code>ToolNode</code>,
LangGraph enables parallel tool execution, sandboxed error recovery, fine-grained telemetry tracking,
and human-in-the-loop approval interrupts directly at the tool boundary.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: TOOL EXECUTION NODES
# ==============================================================================
st.header("2. Tool Execution Nodes: The Prebuilt ToolNode Primitive", divider="rainbow")

st.markdown("""
LangGraph provides `ToolNode` inside `langgraph.prebuilt`.
Instead of manually writing a loop that iterates through `message.tool_calls`, `ToolNode` handles:
1. Extracting `tool_calls` from the latest `AIMessage`.
2. Validating arguments against each tool's Pydantic schema.
3. Executing the functions (concurrently if multiple tools were requested).
4. Wrapping outputs into `ToolMessage` objects with exact `tool_call_id` linkage.
""")

st.code("""
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> float:
    \"\"\"Fetch current equity trading price.\"\"\"
    return 182.45

@tool
def get_forex_rate(pair: str) -> float:
    \"\"\"Fetch foreign exchange rate.\"\"\"
    return 1.085

# Instantiate prebuilt ToolNode with tool list
tool_node = ToolNode(tools=[get_stock_price, get_forex_rate])
""", language="python")

# ==============================================================================
# SECTION 3: TOOL ROUTING
# ==============================================================================
st.header("3. Tool Routing: Conditional Edges and the tools_condition Function", divider="rainbow")

st.markdown("""
How does the graph know whether to invoke `ToolNode` or transition to `END`?
LangGraph includes `tools_condition`: a prebuilt conditional routing function:
""")

st.code("""
from langgraph.prebuilt import tools_condition

# Source definition of tools_condition:
def tools_condition(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "__end__"

# Wire conditional edge into graph
builder.add_conditional_edges("agent", tools_condition)
""", language="python")

# ==============================================================================
# SECTION 4: MULTIPLE TOOLS
# ==============================================================================
st.header("4. Multiple Tools: Managing Large Toolsets and Concurrent Invocations", divider="rainbow")

st.markdown("""
When an LLM issues multiple tool calls in a single turn (e.g. searching 3 stocks simultaneously),
`ToolNode` executes them concurrently by default:
""")

col_mt1, col_mt2 = st.columns(2)
with col_mt1:
    st.markdown("#### Parallel Tool Emission")
    st.code("""
# LLM emits parallel calls in one turn:
AIMessage(
    content="",
    tool_calls=[
        {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "c1"},
        {"name": "get_stock_price", "args": {"ticker": "MSFT"}, "id": "c2"},
        {"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": "c3"}
    ]
)
""", language="python")

with col_mt2:
    st.markdown("#### ToolNode Parallel Execution")
    st.markdown("""
    1. `ToolNode` evaluates `c1`, `c2`, and `c3` asynchronously in parallel.
    2. Emits 3 matching `ToolMessage` objects in a single superstep.
    3. The state channel receives all 3 outputs simultaneously, reducing turn latency from 1500ms to 500ms.
    """)

# ==============================================================================
# SECTION 5: DYNAMIC TOOL SELECTION
# ==============================================================================
st.header("5. Dynamic Tool Selection: Contextual Tool Filtering and Permission Scopes", divider="rainbow")

st.markdown("""
Providing 50+ tools simultaneously degrades LLM reasoning and inflates prompt token overhead.
In enterprise graphs, dynamic tool selection filters tools contextually based on user role or active sub-task:
""")

st.code("""
def agent_node(state: MessagesState, config: RunnableConfig) -> dict:
    user_role = config["configurable"].get("user_role", "viewer")
    
    # Restrict destructive tools to administrators
    if user_role == "admin":
        active_tools = [view_data, edit_data, delete_data]
    else:
        active_tools = [view_data]
        
    model_with_dynamic_tools = base_model.bind_tools(active_tools)
    response = model_with_dynamic_tools.invoke(state["messages"])
    return {"messages": [response]}
""", language="python")

# ==============================================================================
# SECTION 6: TOOL LOOPS
# ==============================================================================
st.header("6. Tool Loops: Cyclic ReAct Execution and Termination Conditions", divider="rainbow")

st.markdown("""
In a cyclic agent, the loop continues until:
1. The LLM decides it has gathered sufficient facts and emits natural language text without `tool_calls`.
2. The graph hits its **recursion limit** (default: 25 supersteps).
""")

with st.expander("🔍 View Tool Loop Termination States", expanded=True):
    loop_dot = """
    digraph LoopTermination {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        STEP_EVAL [label="Superstep Evaluation: step_count < recursion_limit?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
        CHECK_CALLS [label="Did LLM return tool_calls?", shape=diamond, fillcolor="#1E1E38", color="#8B5CF6"];

        RUN_TOOLS [label="Execute ToolNode\\nLoop back to LLM", fillcolor="#431407", color="#FB923C"];
        NORMAL_FINISH [label="Normal Termination: Route to END\\nUser receives natural answer", fillcolor="#064E3B", color="#34D399"];
        ABORT_LIMIT [label="Graph RecursionError Raised\\nPrevent infinite loops", fillcolor="#3B1A1A", color="#F87171"];

        STEP_EVAL -> CHECK_CALLS [label="Yes (< 25)"];
        STEP_EVAL -> ABORT_LIMIT [label="No (>= 25)"];
        CHECK_CALLS -> RUN_TOOLS [label="Yes (tool_calls > 0)"];
        CHECK_CALLS -> NORMAL_FINISH [label="No (empty tool_calls)"];
        RUN_TOOLS -> STEP_EVAL [label="Next superstep"];
    }
    """
    st.graphviz_chart(loop_dot, use_container_width=True)

# ==============================================================================
# SECTION 7: TOOL VALIDATION
# ==============================================================================
st.header("7. Tool Validation: Pre-Flight Argument Checks and Guardrails", divider="rainbow")

st.markdown("""
Pre-flight argument validation ensures arguments satisfy domain business rules before
executing side-effects. You can configure `handle_tool_errors=True` on `ToolNode`:
""")

st.code("""
# ToolNode with built-in error handling
tool_node = ToolNode(
    tools=[wire_transfer_tool],
    handle_tool_errors=True  # Catches ToolException and returns as ToolMessage
)
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Tool Cyclic ReAct Graph with Validation", divider="rainbow")

st.markdown("""
Below is the complete, runnable implementation of a multi-tool ReAct graph featuring
prebuilt `ToolNode`, `tools_condition` routing, and recursion limit protection:
""")

st.code("""
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define Tools
@tool
def calculate_mortgage_payment(principal: float, annual_interest_rate: float, years: int) -> float:
    \"\"\"Calculate monthly mortgage installment given principal, rate, and term.\"\"\"
    monthly_rate = (annual_interest_rate / 100) / 12
    num_payments = years * 12
    monthly_payment = principal * (monthly_rate * ((1 + monthly_rate) ** num_payments)) / (((1 + monthly_rate) ** num_payments) - 1)
    return round(monthly_payment, 2)

@tool
def check_property_tax(zip_code: str) -> float:
    \"\"\"Estimate average annual property tax rate for a specified zip code.\"\"\"
    tax_rates = {"94102": 1.18, "10001": 1.25, "75001": 2.15}
    return tax_rates.get(zip_code, 1.20)

tools = [calculate_mortgage_payment, check_property_tax]

# 2. Configure Model
base_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
model_with_tools = base_model.bind_tools(tools)

# 3. Node Definitions
def reasoning_node(state: MessagesState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

# 4. Assemble Graph
builder = StateGraph(MessagesState)
builder.add_node("agent", reasoning_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # ReAct Cycle

# 5. Compile with Recursion Guard
agent_graph = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Tool Routing and Cyclic Loop Execution Sandbox", divider="rainbow")

st.markdown("""
Step through a simulated multi-iteration ReAct loop. Observe how `ToolNode` executes tool requests,
detects validation issues, and feeds results back to the LLM until the final answer is reached.
""")

sim_engine = ToolNodeExecutionSimulator()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Simulation Controls")
    query_scenario = st.selectbox(
        "Select User Inquiry Scenario:",
        [
            "Can you process a refund of 50.00 USD for customer ID 992?",
            "What is the company policy regarding software reimbursements?"
        ]
    )
    max_steps = st.slider("Max Loop Iterations (Recursion Limit):", min_value=1, max_value=5, value=3)
    inject_bad_args = st.checkbox("Inject Invalid Argument (Trigger ToolNode Validation)", value=False)
    
    execute_sim = st.button("🚀 Run ReAct Cycle", type="primary")

with wb_col2:
    st.subheader("📋 ReAct Execution Trace")
    if execute_sim:
        res = sim_engine.simulate_agent_turn(query_scenario, max_steps, inject_bad_args)
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Iterations Used", res["iterations_used"])
        with m_c2:
            st.metric("Status", "COMPLETED" if res["completed"] else "RECURSION LIMIT EXCEEDED")

        st.markdown("#### Superstep Audit Log")
        for step in res["trace"]:
            if "Iteration" in step:
                st.markdown(f"**{step}**")
            elif "INVALID ARGS" in step or "REJECTED" in step:
                st.error(step)
            elif "ToolNode executed" in step:
                st.info(step)
            elif "Routing edge" in step:
                st.warning(step)
            else:
                st.write(f"- {step}")
    else:
        st.info("Configure the test parameters on the left and click 'Run ReAct Cycle' to observe the ReAct loop execution.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What are the primary responsibilities of LangGraph's prebuilt ToolNode?", expanded=False):
    st.markdown("""
    **Answer**:
    `ToolNode` abstracts tool execution boilerplate:
    1. Extracts `tool_calls` from the latest `AIMessage`.
    2. Maps the tool name to the registered Python callable.
    3. Validates and deserializes JSON argument strings into typed inputs.
    4. Executes the functions (concurrently across multiple calls).
    5. Wraps each result into a `ToolMessage` with the exact corresponding `tool_call_id`.
    """)

with st.expander("Q2: How does tools_condition decide whether to branch to 'tools' or '__end__'?", expanded=False):
    st.markdown("""
    **Answer**:
    `tools_condition` inspects `state["messages"][-1]`.
    If the message possesses a non-empty `tool_calls` attribute (`len(message.tool_calls) > 0`), it returns `"tools"`.
    If `tool_calls` is empty or absent (the model generated a direct conversational reply), it returns `"__end__"`.
    """)

with st.expander("Q3: What causes an infinite loop in a ReAct graph and how is it prevented?", expanded=False):
    st.markdown("""
    **Answer**:
    An infinite loop occurs when:
    1. A tool returns an error that the model fails to understand, causing it to repeatedly emit the same failing tool call.
    2. The model gets stuck in an oscillation between two tools.
    Prevention:
    1. Configure `recursion_limit` in invocation config: `app.invoke(input, {"recursion_limit": 15})`.
    2. Add state counters (`retry_count`) to short-circuit repeated failures.
    """)

with st.expander("Q4: Why should ToolNode loop back to the agent node rather than going directly to END?", expanded=False):
    st.markdown("""
    **Answer**:
    A tool only outputs raw data (e.g. JSON strings or numbers). The user expects a natural, polite synthesis.
    Looping back to the agent gives the LLM the opportunity to observe the `ToolMessage` result, verify that
    the user's inquiry was satisfied, and formulate a clear final explanation.
    """)

with st.expander("Q5: Can ToolNode execute custom asynchronous tools without blocking the event loop?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes. `ToolNode` natively supports asynchronous coroutines (`async def`).
    When invoked via `app.ainvoke()`, `ToolNode` executes all concurrent tool calls using `asyncio.gather()`,
    maximizing I/O throughput.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting the Return Edge from tools to agent")
    st.markdown("""
    *Mechanism*: Connecting `builder.add_edge(START, "agent")` and `builder.add_conditional_edges("agent", tools_condition)` but omitting `builder.add_edge("tools", "agent")`.  
    *Consequence*: After executing tools, graph execution halts abruptly without emitting a final answer to the user.  
    *Fix*: Always close the ReAct cycle with `builder.add_edge("tools", "agent")`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Tool Exceptions Crashing the State Machine")
    st.markdown("""
    *Mechanism*: Using `ToolNode` with default settings while tools raise unhandled database or HTTP exceptions.  
    *Consequence*: The entire graph execution crashes, terminating the user session ungracefully.  
    *Fix*: Initialize `ToolNode(tools=..., handle_tool_errors=True)` to convert exceptions into informative `ToolMessage` payloads.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Exceeding Default Recursion Limits on Complex Queries")
    st.markdown("""
    *Mechanism*: Complex multi-step tasks requiring 30+ tool steps failing with `GraphRecursionError`.  
    *Consequence*: Long-running research workflows abort prematurely.  
    *Fix*: Explicitly increase `config={"recursion_limit": 50}` during graph compilation or invocation.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Binding Too Many Tools Simultaneously")
    st.markdown("""
    *Mechanism*: Binding 40+ diverse tools to a single generalist LLM node.  
    *Consequence*: Degradation in tool selection accuracy and high prompt token costs.  
    *Fix*: Implement dynamic tool filtering or decompose into specialized multi-agent subgraphs.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**ToolNode**\\nThe prebuilt LangGraph node that extracts tool_calls, runs functions, and emits ToolMessages.")
with col_fc2:
    st.info("**tools_condition**\\nPrebuilt conditional edge routing to 'tools' if tool_calls exist, or '__end__' otherwise.")
with col_fc2:
    st.info("**ReAct Cycle**\\nThe cyclic pattern: LLM calls Tool -> ToolNode returns data -> LLM reasons over data -> Final answer.")
with col_fc4:
    st.info("**recursion_limit**\\nThe maximum number of supersteps a graph is permitted to execute before aborting.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to instantiate and attach ToolNode to a LangGraph graph.", value=True)
c2 = st.checkbox("I understand how tools_condition routes execution between tools and END.", value=True)
c3 = st.checkbox("I can wire the cyclic return edge from ToolNode back to the agent node.", value=True)
c4 = st.checkbox("I know how to configure handle_tool_errors=True on ToolNode.", value=True)
c5 = st.checkbox("I understand how ToolNode handles parallel tool calls concurrently.", value=True)
c6 = st.checkbox("I can configure recursion_limit to prevent infinite loops in cyclic agents.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 12.2 'Tool Nodes' with full architectural and practical comprehension!")
