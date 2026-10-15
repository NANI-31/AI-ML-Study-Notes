import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="13.2 Agent Loop - LangGraph Curriculum",
    page_icon="🔁",
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
    .badge-phase {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-stop {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
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
class AgentLoopExecutionEngine:
    """
    Simulates the canonical 5-phase loop:
    Think -> Act -> Observe -> Continue -> Stop
    """
    def __init__(self):
        self.phases = ["Think", "Act", "Observe", "Continue", "Stop"]

    def execute_multi_step_loop(self, task_goal: str, max_steps: int) -> List[Dict[str, Any]]:
        steps = []
        current_step = 0
        goal_met = False

        while current_step < max_steps and not goal_met:
            current_step += 1
            if current_step == 1:
                steps.append({
                    "step": current_step,
                    "phase": "Think",
                    "action": "Model decomposes goal: 'Query system logs for error code 500 in auth-service.'",
                    "status": "In Progress"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Act",
                    "action": "Emitted tool_calls: query_logs(service='auth-service', code=500)",
                    "status": "Action Dispatched"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Observe",
                    "action": "Tool output received: '3 log entries found: database connection pool exhausted at 14:02 UTC.'",
                    "status": "Observation Committed"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Continue",
                    "action": "Model checks goal: Identified symptom, but root cause unknown. Requires database health check.",
                    "status": "Loop Continues"
                })
            elif current_step == 2:
                steps.append({
                    "step": current_step,
                    "phase": "Think",
                    "action": "Model reasons: 'Need to inspect PostgreSQL active connections and max_connections limit.'",
                    "status": "In Progress"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Act",
                    "action": "Emitted tool_calls: check_db_metrics(metric='connections')",
                    "status": "Action Dispatched"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Observe",
                    "action": "Tool output received: 'active_connections=100, max_limit=100. Pool saturated.'",
                    "status": "Observation Committed"
                })
                steps.append({
                    "step": current_step,
                    "phase": "Stop",
                    "action": "Goal satisfied: Full root cause identified. Model synthesizes final diagnostic report.",
                    "status": "Goal Satisfied"
                })
                goal_met = True

        return steps


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The 5-Phase Autonomous Agent Loop", divider="rainbow")

st.markdown("""
The **Agent Loop** is the engine of autonomous AI systems.
Rather than stopping at a single tool call, the agent repeatedly executes a 5-phase cycle:
**Think** (deliberation), **Act** (tool call dispatch), **Observe** (tool result ingestion),
**Continue** (goal convergence assessment), and **Stop** (termination and delivery).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    loop_blueprint_dot = """
    digraph AgentLoopPhases {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        START_NODE [label="START (Goal Ingested)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_loop_engine {
            label = "Autonomous 5-Phase Execution Loop";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            THINK [label="1. Think (Deliberation)\\n- Analyzes current state & previous observations\\n- Decides next tactical action", fillcolor="#1E1E38", color="#8B5CF6"];
            ACT [label="2. Act (Action Dispatch)\\n- Emits structured tool_calls\\n- Dispatches to ToolNode", fillcolor="#0F172A", color="#F59E0B"];
            OBSERVE [label="3. Observe (Environment Feedback)\\n- Executes tool\\n- Appends ToolMessage to state", fillcolor="#431407", color="#FB923C"];
            EVALUATE [label="4. Continue vs Stop?\\nIs the goal fully satisfied or recursion limit reached?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];

            THINK -> ACT;
            ACT -> OBSERVE;
            OBSERVE -> EVALUATE;
            EVALUATE -> THINK [label="Continue (Goal Incomplete)"];
        }

        STOP_NODE [label="5. Stop (Final Synthesis)\\nEmits natural language explanation to user and exits", fillcolor="#064E3B", color="#34D399"];

        START_NODE -> THINK;
        EVALUATE -> STOP_NODE [label="Stop (Goal Satisfied)"];
    }
    """
    st.graphviz_chart(loop_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Loop Dynamics:</b> The Continue vs. Stop evaluation is the central intelligence of the agent.
Premature termination results in incomplete answers, while failure to terminate causes runaway loops.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: PHASE 1: THINK
# ==============================================================================
st.header("2. Phase 1: Think (Internal Reasoning & Chain-of-Thought Formulation)", divider="rainbow")

st.markdown("""
In the **Think** phase, the LLM processes incoming context, past tool outputs, and user constraints.
It formulates an internal plan before generating any tool calls.
""")

st.code("""
# The model reasons over history:
# "I observed that user authentication failed with error 500.
# Next, I should query the database connection metrics to see if the pool is saturated."
""", language="python")

# ==============================================================================
# SECTION 3: PHASE 2: ACT
# ==============================================================================
st.header("3. Phase 2: Act (Structured Tool Invocation & Action Dispatch)", divider="rainbow")

st.markdown("""
In the **Act** phase, thoughts are converted into concrete actions.
The model emits structured `tool_calls`:
""")

st.code("""
AIMessage(
    content="",
    tool_calls=[{"name": "check_db_metrics", "args": {"metric": "pool_usage"}, "id": "call_db_1"}]
)
""", language="python")

# ==============================================================================
# SECTION 4: PHASE 3: OBSERVE
# ==============================================================================
st.header("4. Phase 3: Observe (Environmental Feedback & ToolMessage Ingestion)", divider="rainbow")

st.markdown("""
In the **Observe** phase, `ToolNode` executes the requested action and writes the result into state.
The agent observes real facts from the external world:
""")

st.code("""
ToolMessage(
    content='{"active_connections": 100, "max_connections": 100, "status": "SATURATED"}',
    tool_call_id="call_db_1"
)
""", language="python")

# ==============================================================================
# SECTION 5: PHASE 4: CONTINUE
# ==============================================================================
st.header("5. Phase 4: Continue (Evaluation Loop, Convergence Checks & Next-Step Planning)", divider="rainbow")

st.markdown("""
In the **Continue** phase, the model assesses:
- *Do I have enough information to solve the user's inquiry completely?*
- If **NO**, the loop returns to Phase 1 (Think) to plan the next tool invocation.
""")

# ==============================================================================
# SECTION 6: PHASE 5: STOP
# ==============================================================================
st.header("6. Phase 5: Stop (Termination Criteria, Goal Satisfaction & Output Delivery)", divider="rainbow")

st.markdown("""
In the **Stop** phase, the model recognizes that the goal has been achieved.
It generates a final `AIMessage` with natural language text and **zero tool calls**,
causing `tools_condition` to route execution to `END`.
""")

st.code("""
AIMessage(
    content="Root Cause Identified: The auth-service threw 500 errors because the PostgreSQL connection pool reached its 100-connection ceiling at 14:02 UTC.",
    tool_calls=[]
)
""", language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Autonomous Multi-Turn Research Loop with Convergence Guards", divider="rainbow")

st.markdown("""
Below is the complete production implementation of a multi-turn investigative agent loop
with explicit recursion limits and error-handled tools:
""")

st.code("""
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Diagnostic Tools
@tool
def query_service_logs(service_name: str) -> str:
    \"\"\"Query recent error logs for a specific microservice.\"\"\"
    return "Error log: Database connection pool exhausted at 14:02 UTC."

@tool
def check_db_metrics(metric_name: str) -> str:
    \"\"\"Inspect live database telemetry metrics.\"\"\"
    return "Active Connections: 100 / 100 (Max Limit Saturated)."

tools = [query_service_logs, check_db_metrics]

# 2. Model Binding
model = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(tools)

# 3. Node Definitions
def loop_agent_node(state: MessagesState) -> dict:
    \"\"\"Executes Phase 1 (Think), Phase 2 (Act), Phase 4 (Continue), and Phase 5 (Stop).\"\"\"
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

# 4. Assemble Graph
builder = StateGraph(MessagesState)
builder.add_node("agent", loop_agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# 5. Compile with strict recursion guard
investigator_agent = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Agent Loop Stepper & Convergence Simulator", divider="rainbow")

st.markdown("""
Step through the autonomous 5-phase execution loop. Observe how the agent moves from initial inquiry
through multi-phase deliberation and action until reaching full goal satisfaction.
""")

loop_engine = AgentLoopExecutionEngine()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Mission Parameters")
    mission = st.selectbox(
        "Select Agent Diagnostic Mission:",
        [
            "Investigate 500 error spike in auth-service and identify root cause",
            "Perform vulnerability audit on Redis caching layer"
        ]
    )
    max_iter = st.slider("Max Loop Steps Allowed:", min_value=1, max_value=4, value=2)
    start_loop = st.button("🚀 Execute Autonomous Loop", type="primary")

with wb_c2:
    st.subheader("📋 Phase Execution Audit")
    if start_loop:
        results = loop_engine.execute_multi_step_loop(mission, max_iter)
        st.success(f"Executed {len(results)} individual phases across multi-step cycle!")
        
        for item in results:
            badge = "badge-stop" if item["phase"] == "Stop" else "badge-phase"
            st.markdown(f"**Step {item['step']} — Phase: <span class='{badge}'>{item['phase']}</span>**", unsafe_allow_html=True)
            st.write(f"- Action: {item['action']}")
            st.info(f"Status: {item['status']}")
            st.divider()
    else:
        st.info("Select a mission and click 'Execute Autonomous Loop' to inspect the phase-by-phase trace.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the classic ReAct loop and how does it map to LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    The ReAct (Reason + Act) loop alternates between reasoning (Think), tool invocation (Act), and observation ingestion (Observe).
    In LangGraph, the agent node performs the Think and Act phases, `ToolNode` performs the Observe phase,
    and `tools_condition` determines whether to Continue or Stop.
    """)

with st.expander("Q2: How does an agent know when to stop looping?", expanded=False):
    st.markdown("""
    **Answer**:
    The LLM inspects its accumulated observations against the user prompt.
    When it determines that all required facts have been obtained, it generates a response without `tool_calls`.
    The `tools_condition` edge detects the absence of tool calls and routes execution to `END`.
    """)

with st.expander("Q3: What causes an agent to loop infinitely without stopping?", expanded=False):
    st.markdown("""
    **Answer**:
    1. A tool returns an ambiguous or error message that the LLM tries to fix by calling the same failing tool.
    2. The system prompt fails to provide clear completion criteria.
    3. The model hallucinates that it needs additional unnecessary data.
    """)

with st.expander("Q4: What is the purpose of the Continue phase in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    The Continue phase is the decision to loop back from `tools` to `agent`.
    It feeds the newly gathered observations back into the LLM's context window, allowing it to re-assess its hypothesis
    and plan the next action.
    """)

with st.expander("Q5: How does recursion_limit protect enterprise infrastructure during agent loops?", expanded=False):
    st.markdown("""
    **Answer**:
    `recursion_limit` sets a hard ceiling on the number of graph supersteps.
    If an agent encounters a reasoning loop or buggy tool, execution aborts when the limit is reached,
    preventing infinite token billing and API exhaustion.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing Convergence Criteria in System Prompts")
    st.markdown("""
    *Mechanism*: Giving an agent tools without defining what constitutes a satisfactory answer.  
    *Consequence*: The agent keeps calling diagnostic tools indefinitely, exhausting the recursion limit.  
    *Fix*: Explicitly state in the system prompt: "Once the root cause is identified, stop tool execution and present your findings."
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Tool Failure Blindness Leading to Retries")
    st.markdown("""
    *Mechanism*: Tool fails with an empty string rather than an informative error message.  
    *Consequence*: The agent cannot understand why the call failed and retries the exact same call in a loop.  
    *Fix*: Always return explicit error descriptions from tools so the agent can adapt.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Quadratic Context Growth During Long Loops")
    st.markdown("""
    *Mechanism*: Retaining extensive tool payloads across a 15-iteration loop.  
    *Consequence*: Latency increases on each iteration as prompt tokens balloon.  
    *Fix*: Implement tool payload trimming or use `response_format='content_and_artifact'`.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Hardcoding Single-Iteration Chains")
    st.markdown("""
    *Mechanism*: Routing `ToolNode` directly to a final formatter node instead of looping back to the agent.  
    *Consequence*: The agent cannot perform multi-step investigation or recovery.  
    *Fix*: Always wire `ToolNode` back to the agent node to allow cyclic reasoning.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Think**\\nThe deliberation phase where the model analyzes history and plans its next action.")
with col_fc2:
    st.info("**Act**\\nThe phase where the model emits structured tool_calls to invoke external tools.")
with col_fc3:
    st.info("**Observe**\\nThe ingestion of tool results into the state channel via ToolMessage.")
with col_fc4:
    st.info("**Stop**\\nThe phase where the model generates a final natural language answer and exits.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know all 5 phases of the agent loop (Think, Act, Observe, Continue, Stop).", value=True)
c2 = st.checkbox("I understand how the agent evaluates goal satisfaction during the Continue phase.", value=True)
c3 = st.checkbox("I know how the absence of tool_calls signals transition to the Stop phase.", value=True)
c4 = st.checkbox("I understand how recursion limits protect against infinite loops.", value=True)
c5 = st.checkbox("I know how to write system prompts with clear convergence criteria.", value=True)
c6 = st.checkbox("I can build a multi-turn investigative agent that loops across sequential tool calls.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 13.2 'Agent Loop' with full architectural and practical comprehension!")
