import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="12.3 AI Agent Tool Workflow - LangGraph Curriculum",
    page_icon="🔄",
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
    .badge-stage {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-done {
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
class SixStageWorkflowEngine:
    """
    Deconstructs and simulates the exact 6-stage tool workflow:
    1. LLM Decides -> 2. Tool Call -> 3. Tool Execution ->
    4. Tool Result -> 5. LLM Reasoning -> 6. Final Response
    """
    def __init__(self):
        self.stages = [
            "1. LLM Decides",
            "2. Tool Call",
            "3. Tool Execution",
            "4. Tool Result",
            "5. LLM Reasoning",
            "6. Final Response"
        ]

    def run_step_by_step(self, user_inquiry: str) -> List[Dict[str, Any]]:
        trace = []
        
        # Stage 1
        trace.append({
            "stage": "1. LLM Decides",
            "description": "LLM ingests user inquiry, detects need for factual verification, and elects to call tool.",
            "data": {"intent": "Account Balance Lookup", "selected_tool": "get_account_balance", "confidence": 0.98}
        })
        
        # Stage 2
        trace.append({
            "stage": "2. Tool Call",
            "description": "LLM emits AIMessage with structured tool_calls payload matching tool schema.",
            "data": {"tool_name": "get_account_balance", "args": {"account_id": "ACC-8831"}, "tool_call_id": "call_982bca1"}
        })
        
        # Stage 3
        trace.append({
            "stage": "3. Tool Execution",
            "description": "ToolNode intercepts call, executes Python function against production database.",
            "data": {"execution_time_ms": 42, "status": "200_OK"}
        })
        
        # Stage 4
        trace.append({
            "stage": "4. Tool Result",
            "description": "ToolNode encapsulates return value into ToolMessage with matching tool_call_id.",
            "data": {"content": '{"account_id": "ACC-8831", "balance_usd": 128450.00, "currency": "USD"}', "tool_call_id": "call_982bca1"}
        })
        
        # Stage 5
        trace.append({
            "stage": "5. LLM Reasoning",
            "description": "LLM evaluates observational ToolMessage, verifies factual sufficiency, and prepares response.",
            "data": {"sufficient_data": True, "additional_tools_needed": False}
        })
        
        # Stage 6
        trace.append({
            "stage": "6. Final Response",
            "description": "LLM emits final AIMessage with natural language answer and transitions to END.",
            "data": {"final_answer": "Account ACC-8831 currently holds a verified balance of 128,450.00 USD."}
        })
        
        return trace


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The 6-Stage AI Agent Tool Workflow", divider="rainbow")

st.markdown("""
Every tool invocation in LangGraph unfolds across an invariant **6-stage chronological lifecycle**:
1. **LLM Decides**: The model processes context and decides whether external capabilities are required.
2. **Tool Call**: The model emits a structured tool invocation request (`AIMessage.tool_calls`).
3. **Tool Execution**: `ToolNode` invokes the corresponding Python function in a sandboxed environment.
4. **Tool Result**: The output is formatted into a `ToolMessage` linked via `tool_call_id`.
5. **LLM Reasoning**: The model observes the factual data and determines whether further steps are needed.
6. **Final Response**: The model synthesizes the gathered data into a final answer for the user.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    stages_dot = """
    digraph SixStageToolLifecycle {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        USER_PROMPT [label="User Prompt Ingested\\n'What is the balance of ACC-8831?'", fillcolor="#0F172A", color="#38BDF8"];

        STAGE_1 [label="Stage 1: LLM Decides\\nModel reasons: 'I lack balance data; I must query CRM.'", fillcolor="#1E1E38", color="#8B5CF6"];
        STAGE_2 [label="Stage 2: Tool Call\\nEmits AIMessage(tool_calls=[{name, args, id}])", fillcolor="#0F172A", color="#F59E0B"];
        STAGE_3 [label="Stage 3: Tool Execution\\nToolNode runs get_account_balance(account_id='ACC-8831')", fillcolor="#431407", color="#FB923C"];
        STAGE_4 [label="Stage 4: Tool Result\\nWraps JSON in ToolMessage(tool_call_id='call_...')", fillcolor="#1E293B", color="#34D399"];
        STAGE_5 [label="Stage 5: LLM Reasoning\\nModel reviews ToolMessage: 'Fact verified. Ready to reply.'", fillcolor="#064E3B", color="#34D399"];
        STAGE_6 [label="Stage 6: Final Response\\nEmits final conversational text to user. Routes to END.", fillcolor="#0F172A", color="#38BDF8"];

        USER_PROMPT -> STAGE_1;
        STAGE_1 -> STAGE_2 [label="Triggers call"];
        STAGE_2 -> STAGE_3 [label="tools_condition -> ToolNode"];
        STAGE_3 -> STAGE_4 [label="Function returns payload"];
        STAGE_4 -> STAGE_5 [label="Loop back to agent node"];
        STAGE_5 -> STAGE_6 [label="tools_condition -> END"];
    }
    """
    st.graphviz_chart(stages_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Core Design Principle:</b> The agent never hallucinates actions.
Stages 2 through 4 are strictly deterministic and verifiable, while Stages 1, 5, and 6 leverage
probabilistic LLM reasoning for planning and synthesis.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: STAGE 1: LLM DECIDES
# ==============================================================================
st.header("2. Stage 1: LLM Decides (Intent Recognition & Tool Triggering)", divider="rainbow")

st.markdown("""
In Stage 1, the LLM evaluates the user prompt against the descriptions of all bound tools.
It performs an internal classification:
- **Direct Answer**: If the query is conversational or within parametric memory, emit text directly.
- **Tool Trigger**: If the query requires real-time facts or external computation, proceed to tool emission.
""")

st.code("""
# The model receives:
messages = [
    SystemMessage(content="You are a banking agent. Use get_account_balance for balance queries."),
    HumanMessage(content="What is the balance of ACC-8831?")
]

# Model internally recognizes intent and prepares structured parameters
""", language="python")

# ==============================================================================
# SECTION 3: STAGE 2: TOOL CALL
# ==============================================================================
st.header("3. Stage 2: Tool Call (Structured Argument Assembly & Schema Conformance)", divider="rainbow")

st.markdown("""
In Stage 2, the LLM emits an `AIMessage` with `tool_calls`.
The payload conforms strictly to the tool's JSON schema:
""")

st.code("""
# Emitted AIMessage:
AIMessage(
    content="",  # Content is typically empty during tool calling
    tool_calls=[
        {
            "name": "get_account_balance",
            "args": {"account_id": "ACC-8831"},
            "id": "call_982bca1"
        }
    ]
)
""", language="python")

# ==============================================================================
# SECTION 4: STAGE 3: TOOL EXECUTION
# ==============================================================================
st.header("4. Stage 3: Tool Execution (Sandboxed Runtime Dispatch & I/O Isolation)", divider="rainbow")

st.markdown("""
In Stage 3, the graph's `ToolNode` executes the Python callable.
Execution is isolated: network errors or database timeouts are trapped and handled cleanly.
""")

st.code("""
# Inside ToolNode:
def run_tool(call):
    tool_fn = tool_registry[call["name"]]
    # Validates args via Pydantic and executes
    return tool_fn.invoke(call["args"])
""", language="python")

# ==============================================================================
# SECTION 5: STAGE 4: TOOL RESULT
# ==============================================================================
st.header("5. Stage 4: Tool Result (Observational Payload Wrapping & State Merge)", divider="rainbow")

st.markdown("""
In Stage 4, the output from the tool function is wrapped in a `ToolMessage`
and linked back via `tool_call_id`:
""")

st.code("""
# Generated ToolMessage:
ToolMessage(
    content='{"account_id": "ACC-8831", "balance_usd": 128450.00, "status": "ACTIVE"}',
    name="get_account_balance",
    tool_call_id="call_982bca1"  # Matches Stage 2 ID!
)
""", language="python")

# ==============================================================================
# SECTION 6: STAGE 5: LLM REASONING
# ==============================================================================
st.header("6. Stage 5: LLM Reasoning (Fact Evaluation & Multi-Step Planning)", divider="rainbow")

st.markdown("""
In Stage 5, the LLM is invoked again with the updated message history (including the `ToolMessage`).
It reasons over the observed factual data:
- Does this data answer the user's inquiry?
- Do I need to call a secondary tool (e.g. currency conversion)?
- Are there errors or discrepancies to resolve?
""")

# ==============================================================================
# SECTION 7: STAGE 6: FINAL RESPONSE
# ==============================================================================
st.header("7. Stage 6: Final Response (Contextual Synthesis & Executive Delivery)", divider="rainbow")

st.markdown("""
In Stage 6, having confirmed that all necessary data has been collected, the LLM emits a final
natural language answer with empty `tool_calls`.
The `tools_condition` router detects zero tool calls and transitions execution to `END`.
""")

st.code("""
# Final AIMessage emitted to user:
AIMessage(
    content="The verified balance for account ACC-8831 is 128,450.00 USD as of today.",
    tool_calls=[]  # Empty -> routes to END!
)
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Enterprise Financial Auditor Agent", divider="rainbow")

st.markdown("""
Below is the complete production implementation tracing all 6 stages through a compiled LangGraph workflow:
""")

st.code("""
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define Production Tool
@tool
def query_account_ledger(account_id: str) -> str:
    \"\"\"Query core enterprise ledger for verified balance and account status.\"\"\"
    database = {
        "ACC-8831": {"balance": 128450.00, "status": "ACTIVE", "tier": "Enterprise"},
        "ACC-4412": {"balance": 1520.50, "status": "RESTRICTED", "tier": "Standard"}
    }
    record = database.get(account_id)
    if record:
        return f"Account {account_id}: Balance={record['balance']} USD, Status={record['status']}"
    return f"Error: Account {account_id} not found."

tools = [query_account_ledger]

# 2. Bind Model
model = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(tools)

# 3. Nodes
def agent_reasoning_node(state: MessagesState) -> dict:
    \"\"\"Handles Stage 1 (Decide), Stage 5 (Reason), and Stage 6 (Final Response).\"\"\"
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

# 4. Assemble Graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_reasoning_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

financial_agent = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: 6-Stage Stepper Simulation & Telemetry Inspector", divider="rainbow")

st.markdown("""
Step through the 6 stages of an AI agent tool workflow.
Observe how messages, structured payloads, and decisions transition from user input to final delivery.
""")

workflow_eng = SixStageWorkflowEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Simulation Trigger")
    sample_query = st.selectbox(
        "Select User Inquiry:",
        [
            "What is the balance of account ACC-8831?",
            "Can you verify the transaction history for ACC-4412?"
        ]
    )
    start_sim = st.button("🚀 Step Through 6 Stages", type="primary")

with wb_col2:
    st.subheader("📋 6-Stage Execution Stepper")
    if start_sim:
        trace_data = workflow_eng.run_step_by_step(sample_query)
        st.success("Complete 6-stage lifecycle completed successfully!")
        
        for item in trace_data:
            st.markdown(f"#### <span class='badge-stage'>{item['stage']}</span>", unsafe_allow_html=True)
            st.write(item["description"])
            st.json(item["data"])
            st.divider()
    else:
        st.info("Click 'Step Through 6 Stages' on the left to inspect the telemetry payload across each stage.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What prevents the LLM from answering immediately in Stage 1 without calling a tool?", expanded=False):
    st.markdown("""
    **Answer**:
    The system prompt and tool definitions establish that the agent does not possess private runtime data.
    When frontier models are presented with an explicit question requiring private state (such as an account balance)
    and provided a matching tool schema, their instruction-tuning strongly favors emitting `tool_calls` over hallucinating numbers.
    """)

with st.expander("Q2: Why is Stage 5 (Reasoning over Tool Result) necessary instead of returning the raw tool output?", expanded=False):
    st.markdown("""
    **Answer**:
    Tool outputs are raw computational artifacts: JSON dictionaries, SQL rows, or status codes.
    The user expects a contextual, natural language answer.
    Stage 5 allows the LLM to verify that the tool succeeded, handle error messages gracefully, and translate raw data into an executive answer.
    """)

with st.expander("Q3: How does the graph transition from Stage 5 to Stage 6?", expanded=False):
    st.markdown("""
    **Answer**:
    Stage 5 and Stage 6 occur within the same node invocation.
    During Stage 5, the model reasons over the `ToolMessage`.
    During Stage 6, it emits an `AIMessage` containing conversational text with `tool_calls=[]`.
    The `tools_condition` router detects zero tool calls and exits the loop to `END`.
    """)

with st.expander("Q4: What occurs if Stage 3 (Tool Execution) raises a network timeout?", expanded=False):
    st.markdown("""
    **Answer**:
    If `handle_tool_errors=True` is enabled on `ToolNode`, the exception is caught and wrapped into a `ToolMessage`
    containing `"Error: Network timeout"`. In Stage 5, the model sees this error message and can either retry with a backup tool
    or formulate an apology explaining the outage.
    """)

with st.expander("Q5: Can an agent invoke multiple tools simultaneously during Stage 2?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes. Frontier models support parallel tool calling. In Stage 2, the `AIMessage` can contain multiple tool call dictionaries
    in its `tool_calls` list. In Stage 3, `ToolNode` executes all of them concurrently and returns multiple `ToolMessage` payloads.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Returning Raw Tool Output Directly to Users")
    st.markdown("""
    *Mechanism*: Routing `ToolNode` directly to `END` instead of back to the LLM agent node.  
    *Consequence*: Users receive unformatted raw JSON strings instead of polite, contextual responses.  
    *Fix*: Always route `ToolNode` back to the reasoning agent node.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Hallucinated Tool Calls from Missing Parameters")
    st.markdown("""
    *Mechanism*: User prompt omits a required parameter (e.g. "Check my balance" without specifying account ID).  
    *Consequence*: The LLM hallucinates an account ID or crashes during argument schema validation.  
    *Fix*: Instruct the agent in the system prompt to ask clarifying questions when required arguments are missing.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Redundant Reasoning Steps on Trivial Queries")
    st.markdown("""
    *Mechanism*: Forcing every user turn through tools even when answering basic greetings.  
    *Consequence*: Adds 600ms latency and unnecessary tool schema prompt tokens.  
    *Fix*: Allow the model to answer conversational queries directly without invoking tools.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Silent Execution Failures Without Retries")
    st.markdown("""
    *Mechanism*: Tool encounters a transient 503 error and returns an empty string without an error explanation.  
    *Consequence*: The LLM hallucinates that no data exists for the user.  
    *Fix*: Ensure tools return explicit error descriptions (`"Error 503: Service temporarily unavailable"`).
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Stage 1: Decide**\\nThe phase where the LLM evaluates user intent and decides whether external tools are needed.")
with col_fc2:
    st.info("**Stage 2: Call**\\nThe structured payload emitted by the model containing function name, arguments, and unique ID.")
with col_fc3:
    st.info("**Stage 4: Result**\\nThe ToolMessage returning deterministic factual data back to the conversation channel.")
with col_fc4:
    st.info("**Stage 6: Final Response**\\nThe synthesized natural language answer terminating the ReAct cycle.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know all 6 chronological stages of the AI agent tool workflow.", value=True)
c2 = st.checkbox("I understand how the LLM decides between direct conversational replies and tool calls.", value=True)
c3 = st.checkbox("I know why ToolNode must loop back to the agent node before terminating at END.", value=True)
c4 = st.checkbox("I understand how tool_call_id correlates Stage 2 calls with Stage 4 results.", value=True)
c5 = st.checkbox("I know how to handle tool errors during Stage 3 to allow self-correction in Stage 5.", value=True)
c6 = st.checkbox("I understand how tools_condition identifies the transition to Stage 6 final responses.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 12.3 'AI Agent Tool Workflow' with full architectural and practical comprehension!")
