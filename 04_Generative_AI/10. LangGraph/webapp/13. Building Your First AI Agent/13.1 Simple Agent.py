import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="13.1 Simple Agent - LangGraph Curriculum",
    page_icon="🤖",
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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class SimpleAgentEngine:
    """
    Simulates the 5 foundational stages of a simple agent:
    1. User Input -> 2. LLM Reasoning -> 3. Tool Selection ->
    4. Tool Execution -> 5. Final Response
    """
    def __init__(self):
        self.mock_tools = {
            "calculator": lambda a, b, op: a * b if op == "multiply" else a + b,
            "system_info": lambda: "CloudScale Cluster v2.4 - US-East-1 - 99.98 percent Uptime"
        }

    def execute_simple_agent(self, user_prompt: str) -> Dict[str, Any]:
        trace = []
        trace.append(f"Stage 1 (User Input): Received '{user_prompt}'")

        if any(w in user_prompt.lower() for w in ["multiply", "calculate", "times", "*", "+"]):
            trace.append("Stage 2 (LLM Reasoning): Model determines arithmetic calculation is needed.")
            trace.append("Stage 3 (Tool Selection): Selected tool 'calculator' with args {'a': 125, 'b': 8, 'op': 'multiply'}.")
            res = self.mock_tools["calculator"](125, 8, "multiply")
            trace.append(f"Stage 4 (Tool Execution): Calculator evaluated to: {res}")
            final_ans = f"The result of 125 multiplied by 8 is {res}."
            trace.append(f"Stage 5 (Final Response): Synthesized answer: '{final_ans}'")
        elif "status" in user_prompt.lower() or "cluster" in user_prompt.lower():
            trace.append("Stage 2 (LLM Reasoning): Model identifies system monitoring inquiry.")
            trace.append("Stage 3 (Tool Selection): Selected tool 'system_info' with args {}.")
            res = self.mock_tools["system_info"]()
            trace.append(f"Stage 4 (Tool Execution): Telemetry fetched: '{res}'")
            final_ans = f"Current system telemetry: {res}."
            trace.append(f"Stage 5 (Final Response): Synthesized answer: '{final_ans}'")
        else:
            trace.append("Stage 2 (LLM Reasoning): General conversational inquiry; no tool required.")
            trace.append("Stage 3 (Tool Selection): Bypassed (direct answering).")
            trace.append("Stage 4 (Tool Execution): Bypassed.")
            final_ans = f"I am your assistant. Regarding '{user_prompt}', everything is functioning normally."
            trace.append(f"Stage 5 (Final Response): Emitted answer: '{final_ans}'")

        return {
            "prompt": user_prompt,
            "final_answer": final_ans,
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Minimalist ReAct Agent Architecture", divider="rainbow")

st.markdown("""
A **Simple AI Agent** is the cleanest embodiment of an autonomous reasoning loop.
Consisting of only two nodes (`agent` and `tools`) and one conditional edge (`tools_condition`),
it ingests user input, reasons over available tools, executes selected functions, and synthesizes the final reply.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    simple_agent_dot = """
    digraph SimpleAgentBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        START_IN [label="START: User Input Ingestion\\n(MessagesState)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_agent_core {
            label = "Simple ReAct State Machine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            AGENT_NODE [label="Node: 'agent' (LLM Reasoning)\\n- Ingests messages\\n- Formulates hypothesis\\n- Emits AIMessage(content, tool_calls)", fillcolor="#064E3B", color="#34D399"];
            ROUTER [label="tools_condition(state)\\nAre tool_calls present?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
            TOOL_NODE [label="Node: 'tools' (Tool Execution)\\n- Executes selected tool\\n- Emits ToolMessage", fillcolor="#431407", color="#FB923C"];

            AGENT_NODE -> ROUTER [label="Emits turn"];
            ROUTER -> TOOL_NODE [label="Yes (tool requested)"];
            TOOL_NODE -> AGENT_NODE [label="Returns tool observation (Loop)"];
        }

        END_OUT [label="END: Final Synthesized Response Delivered", fillcolor="#0F172A", color="#64748B"];

        START_IN -> AGENT_NODE;
        ROUTER -> END_OUT [label="No (answer complete)"];
    }
    """
    st.graphviz_chart(simple_agent_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Simplicity by Design:</b> Despite having only two nodes, this pattern handles 90 percent of general-purpose
agentic automation tasks. The agent autonomously decides whether to answer from memory or call tools.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: USER INPUT
# ==============================================================================
st.header("2. User Input: Ingestion, Validation, and Thread Framing", divider="rainbow")

st.markdown("""
User input arrives as a `HumanMessage` or raw string dictionary: `{"messages": [HumanMessage(content="...")]}`.
Best practice frames the input with a persistent `SystemMessage` establishing boundaries and identity.
""")

st.code("""
from langchain_core.messages import HumanMessage, SystemMessage

initial_state = {
    "messages": [
        SystemMessage(content="You are a helpful calculation assistant. Always verify calculations using the math tool."),
        HumanMessage(content="What is 125 multiplied by 8?")
    ]
}
""", language="python")

# ==============================================================================
# SECTION 3: LLM REASONING
# ==============================================================================
st.header("3. LLM Reasoning: Prompt Evaluation and Hypothesis Formation", divider="rainbow")

st.markdown("""
When the LLM receives the messages, it processes the prompt against the schema of bound tools.
It determines whether it already knows the answer or requires external computational assistance.
""")

# ==============================================================================
# SECTION 4: TOOL SELECTION
# ==============================================================================
st.header("4. Tool Selection: Intent Mapping and Schema Adherence", divider="rainbow")

st.markdown("""
If tool execution is required, the LLM selects the appropriate tool and formats its parameters
into a valid JSON payload matching the tool's Pydantic schema:
""")

st.code("""
# Generated AIMessage payload:
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "multiply",
            "args": {"a": 125, "b": 8},
            "id": "call_125_mult"
        }
    ]
)
""", language="python")

# ==============================================================================
# SECTION 5: TOOL EXECUTION
# ==============================================================================
st.header("5. Tool Execution: Invocation Runtime and Result Ingestion", divider="rainbow")

st.markdown("""
`ToolNode` catches the `tool_calls` request, executes the underlying Python function,
and appends a `ToolMessage` with the exact `tool_call_id`:
""")

st.code("""
# ToolNode emits:
ToolMessage(
    content="1000",
    name="multiply",
    tool_call_id="call_125_mult"
)
""", language="python")

# ==============================================================================
# SECTION 6: FINAL RESPONSE
# ==============================================================================
st.header("6. Final Response: Synthesis, Hallucination Verification, and Delivery", divider="rainbow")

st.markdown("""
The conversation loops back to the agent node.
Observing the verified factual result (`1000`), the LLM synthesizes a polite, natural explanation:
""")

st.code("""
# Final response emitted to user:
AIMessage(
    content="The result of 125 multiplied by 8 is 1,000.",
    tool_calls=[]
)
""", language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Minimalist ReAct Agent in 35 Lines", divider="rainbow")

st.markdown("""
Here is the complete, self-contained, and runnable implementation of a simple ReAct agent:
""")

st.code("""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define tools
@tool
def calculate(expression: str) -> str:
    \"\"\"Evaluate a basic mathematical expression safely.\"\"\"
    try:
        # Restricted eval for safe arithmetic
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [calculate]

# 2. Model & binding
model = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(tools)

# 3. Node definitions
def agent_node(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 4. Assemble Graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

simple_agent = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Simple Agent Execution Tracer & Live Step Simulator", divider="rainbow")

st.markdown("""
Test the 5 stages of a simple agent. Observe how the agent dynamically selects tools for arithmetic
and monitoring queries, while bypassing tools for standard conversational prompts.
""")

eng = SimpleAgentEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Agent Input")
    user_prompt = st.text_input(
        "Enter User Prompt:",
        value="What is 125 multiplied by 8?"
    )
    run_sim = st.button("🚀 Run Agent Loop", type="primary")

with wb_col2:
    st.subheader("📊 Execution Trace")
    if run_sim and user_prompt.strip():
        report = eng.execute_simple_agent(user_prompt)
        st.success(f"Final Answer: {report['final_answer']}")
        
        st.markdown("#### Chronological Step Audit")
        for step in report["trace"]:
            st.write(f"- {step}")
    else:
        st.info("Enter a prompt and click 'Run Agent Loop' to inspect the execution trace.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What makes a simple agent fundamentally different from an LCEL chain?", expanded=False):
    st.markdown("""
    **Answer**:
    A chain is a rigid, linear Directed Acyclic Graph (DAG) with hardcoded execution paths (`prompt | model | parser`).
    A simple agent is a **cyclic state machine** where the LLM dynamically decides at runtime whether to execute tools,
    how many times to loop, and when to conclude execution.
    """)

with st.expander("Q2: Why is tools_condition called a conditional edge rather than a node?", expanded=False):
    st.markdown("""
    **Answer**:
    A node mutates or returns state deltas.
    A conditional edge is a pure routing function: it takes the current state as input, returns a string destination key
    (`"tools"` or `"__end__"`), and does not alter the graph state.
    """)

with st.expander("Q3: What happens if the LLM decides no tools are needed for a question?", expanded=False):
    st.markdown("""
    **Answer**:
    The LLM emits an `AIMessage` with direct conversational text and `tool_calls=[]`.
    `tools_condition` inspects the message, sees zero tool calls, and routes directly to `__end__`.
    `ToolNode` is completely bypassed.
    """)

with st.expander("Q4: Can a simple agent call multiple tools across multiple sequential turns?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Because `ToolNode` loops back to `agent`, the LLM can evaluate the result of Tool A, decide it needs additional information,
    call Tool B, observe Tool B's result, and repeat until it has enough data to formulate a final answer.
    """)

with st.expander("Q5: How does LangGraph prevent infinite loops in simple agents?", expanded=False):
    st.markdown("""
    **Answer**:
    Via the built-in `recursion_limit` parameter in the execution configuration.
    If the agent exceeds the allowed number of supersteps (default: 25), execution halts with `GraphRecursionError`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting to bind tools to the LLM model instance")
    st.markdown("""
    *Mechanism*: Passing tools to `ToolNode(tools)` but passing the unbound `model` to `agent_node`.  
    *Consequence*: The LLM never receives tool schemas and attempts to answer everything from hallucinated memory.  
    *Fix*: Always invoke `model = base_model.bind_tools(tools)`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing return edge from tools to agent")
    st.markdown("""
    *Mechanism*: Omitting `builder.add_edge("tools", "agent")`.  
    *Consequence*: Execution dead-ends at `ToolNode` without giving the LLM a chance to synthesize the final answer.  
    *Fix*: Always close the cyclic loop.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Allocating Tools for Simple Agents")
    st.markdown("""
    *Mechanism*: Giving a simple agent 30+ disparate tools.  
    *Consequence*: Increases prompt latency and risks tool confusion.  
    *Fix*: Keep toolsets focused (3-5 tools per simple agent).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Non-Idempotent Tool Loops")
    st.markdown("""
    *Mechanism*: A tool modifies database records on each iteration during a multi-loop query.  
    *Consequence*: If the model loops 3 times, the side-effect executes 3 times.  
    *Fix*: Separate read-only query tools from write-heavy action tools.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**ReAct**\\nReasoning + Acting: The core agent loop alternating between thought and tool execution.")
with col_fc2:
    st.info("**tools_condition**\\nThe conditional edge routing to 'tools' if tool_calls exist or '__end__' if finished.")
with col_fc3:
    st.info("**bind_tools()**\\nThe method that transmits OpenAPI tool schemas to the chat model.")
with col_fc4:
    st.info("**recursion_limit**\\nSafety threshold preventing infinite loops in cyclic graphs.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know all 5 stages of a simple agent (Input, Reason, Select, Execute, Respond).", value=True)
c2 = st.checkbox("I know how to build a 2-node ReAct agent in LangGraph using MessagesState.", value=True)
c3 = st.checkbox("I understand why bind_tools() is mandatory for tool awareness.", value=True)
c4 = st.checkbox("I know why ToolNode must loop back to the agent node.", value=True)
c5 = st.checkbox("I understand how tools_condition identifies completion to route to END.", value=True)
c6 = st.checkbox("I can configure recursion limits to guard against infinite loops.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 13.1 'Simple Agent' with full architectural and practical comprehension!")
