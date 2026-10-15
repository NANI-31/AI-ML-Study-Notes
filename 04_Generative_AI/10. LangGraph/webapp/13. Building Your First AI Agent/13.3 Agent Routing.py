import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="13.3 Agent Routing - LangGraph Curriculum",
    page_icon="🚦",
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
    .badge-route {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-escalate {
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
class AgentRoutingEngine:
    """
    Simulates multi-lane intent routing across 5 distinct inquiry categories:
    1. Tool-Required
    2. Knowledge Questions
    3. Calculation Questions
    4. Search Questions
    5. Human Escalation
    """
    def __init__(self):
        self.routes = {
            "KNOWLEDGE": {"lane": "Direct Parametric Reply", "tools": "None", "cost": "Lowest"},
            "CALCULATION": {"lane": "Math Calculator Tool", "tools": "python_eval_tool", "cost": "Low"},
            "SEARCH": {"lane": "Web/Vector Search Tool", "tools": "vector_retrieval_tool", "cost": "Medium"},
            "ACTION": {"lane": "Core Action Tool", "tools": "crm_mutation_tool", "cost": "Medium"},
            "ESCALATE": {"lane": "Human-in-the-Loop Escalation", "tools": "pager_supervisor", "cost": "High (Human Time)"}
        }

    def classify_and_route(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        if any(w in p for w in ["lawsuit", "refund 10000", "fraud", "unauthorized", "legal", "breach"]):
            category = "ESCALATE"
            reason = "Sensitive compliance or high-liability threshold detected -> Routing to human supervisor."
        elif any(w in p for w in ["calculate", "compound", "sqrt", "*", "/", "formula"]):
            category = "CALCULATION"
            reason = "Mathematical expression detected -> Routing to deterministic Python math tool."
        elif any(w in p for w in ["who is", "latest news", "2025", "current weather", "search"]):
            category = "SEARCH"
            reason = "Real-time or external factual lookup detected -> Routing to external search tool."
        elif any(w in p for w in ["cancel order", "transfer", "update address", "delete"]):
            category = "ACTION"
            reason = "State-modifying action requested -> Routing to authenticated CRM tool."
        else:
            category = "KNOWLEDGE"
            reason = "General conceptual knowledge question -> Routing to direct zero-tool LLM response."

        route_info = self.routes[category]
        return {
            "category": category,
            "assigned_lane": route_info["lane"],
            "tools_assigned": route_info["tools"],
            "relative_cost": route_info["cost"],
            "rationale": reason
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Multi-Lane Intent Classification and Agent Routing Matrix", divider="rainbow")

st.markdown("""
Not all user queries require heavy tool calling.
Sending every query through a tool loop incurs unnecessary latency and costs.
**Agent Routing** classifies incoming prompts into dedicated specialized execution lanes:
**Direct Knowledge** (no tools), **Calculations** (symbolic math tools), **Search** (retrieval tools),
**State Actions** (authenticated API tools), or **Human Escalation** (supervisor review).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    routing_blueprint_dot = """
    digraph MultiLaneAgentRouting {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        USER_IN [label="User Query Ingestion\\n(MessagesState)", fillcolor="#0F172A", color="#38BDF8"];

        ROUTER_NODE [label="Router Evaluator Node: classify_intent(state)\\nExtracts category: [KNOWLEDGE, CALC, SEARCH, ACTION, ESCALATE]", fillcolor="#1E1E38", color="#8B5CF6"];
        ROUTE_EDGE [label="Conditional Edge\\nroute_query(state)", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_lanes {
            label = "Specialized Execution Lanes";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            LANE_KNOWLEDGE [label="Lane 1: Direct Knowledge\\nDirect LLM reply without tools (Zero Tool Overhead)", fillcolor="#064E3B", color="#34D399"];
            LANE_CALC [label="Lane 2: Calculation Tool\\nDelegates to Python math engine", fillcolor="#0F172A", color="#38BDF8"];
            LANE_SEARCH [label="Lane 3: Web / Vector Search\\nRetrieves external ground truth", fillcolor="#1E1E38", color="#8B5CF6"];
            LANE_ACTION [label="Lane 4: CRM Action Tool\\nExecutes authenticated mutation", fillcolor="#431407", color="#FB923C"];
            LANE_ESCALATE [label="Lane 5: Human Escalation\\nTriggers interrupt() checkpoint for supervisor approval", fillcolor="#3B1A1A", color="#F87171"];
        }

        END_OUT [label="END (Delivery / Superstep Committed)", fillcolor="#0F172A", color="#64748B"];

        USER_IN -> ROUTER_NODE;
        ROUTER_NODE -> ROUTE_EDGE;

        ROUTE_EDGE -> LANE_KNOWLEDGE [label="'knowledge'"];
        ROUTE_EDGE -> LANE_CALC [label="'calc'"];
        ROUTE_EDGE -> LANE_SEARCH [label="'search'"];
        ROUTE_EDGE -> LANE_ACTION [label="'action'"];
        ROUTE_EDGE -> LANE_ESCALATE [label="'escalate'"];

        LANE_KNOWLEDGE -> END_OUT;
        LANE_CALC -> END_OUT;
        LANE_SEARCH -> END_OUT;
        LANE_ACTION -> END_OUT;
        LANE_ESCALATE -> END_OUT;
    }
    """
    st.graphviz_chart(routing_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Strategic Takeaway:</b> Routing is the first line of defense for production AI agents.
It ensures that simple conceptual questions never touch tools, expensive external queries are throttled,
and high-liability requests are safely escalated to humans.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: TOOL-REQUIRED QUESTIONS
# ==============================================================================
st.header("2. Tool-Required Questions: Differentiating Parametric Knowledge from Actionable Tasks", divider="rainbow")

st.markdown("""
A question requires a tool when its resolution demands:
1. **Private Enterprise Data**: ("What is the balance of my account ACC-994?")
2. **Real-Time Timestamps**: ("What is the current price of Ethereum right now?")
3. **State Side-Effects**: ("Cancel my recurring subscription starting next month.")
""")

# ==============================================================================
# SECTION 3: KNOWLEDGE QUESTIONS
# ==============================================================================
st.header("3. Knowledge Questions: Direct Zero-Tool Answering from Parametric Memory", divider="rainbow")

st.markdown("""
Queries such as *"What is the difference between supervised and unsupervised learning?"*
require no tools. Routing these directly to the LLM avoids tool schema bloat and reduces latency by 500ms.
""")

st.code("""
def direct_knowledge_node(state: MessagesState) -> dict:
    # Invokes the model without tool bindings
    response = base_model.invoke(state["messages"])
    return {"messages": [response]}
""", language="python")

# ==============================================================================
# SECTION 4: CALCULATION QUESTIONS
# ==============================================================================
st.header("4. Calculation Questions: Mathematical Routing to Symbolic Computation Tools", divider="rainbow")

st.markdown("""
LLMs frequently hallucinate arithmetic on large numbers.
Routing calculation questions to a dedicated math node guarantees 100% precision:
""")

st.code("""
@tool
def calculate_expression(expression: str) -> str:
    \"\"\"Perform exact symbolic and floating point mathematics.\"\"\"
    import math
    return str(eval(expression, {"__builtins__": None, "math": math}, {}))
""", language="python")

# ==============================================================================
# SECTION 5: SEARCH QUESTIONS
# ==============================================================================
st.header("5. Search Questions: Ingesting Real-Time External Facts via Web & Vector Search", divider="rainbow")

st.markdown("""
Search-required inquiries route to retrieval tools that fetch real-time news, documentation chunks,
or vector embeddings from ChromaDB or Milvus before synthesizing an answer.
""")

# ==============================================================================
# SECTION 6: HUMAN ESCALATION
# ==============================================================================
st.header("6. Human Escalation: Routing Ambiguous, Critical, or High-Stakes Inquiries to Supervisors", divider="rainbow")

st.markdown("""
In enterprise settings, questions involving legal threats, high-value refunds (> 5,000 USD),
or data deletion must **escalate to a human**.
In LangGraph, this is modeled via the `interrupt()` function:
""")

st.code("""
from langgraph.types import interrupt

def escalation_node(state: MessagesState) -> dict:
    # Pauses the graph and registers an approval checkpoint in checkpointer
    supervisor_response = interrupt({
        "reason": "High-value transaction or legal escalation required.",
        "user_query": state["messages"][-1].content
    })
    
    # Execution resumes here once approved by supervisor!
    return {
        "messages": [AIMessage(content=f"Supervisor Approved: {supervisor_response['decision']}")]
    }
""", language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Intent Enterprise Dispatcher Graph", divider="rainbow")

st.markdown("""
Below is the complete implementation of a production multi-intent routing graph:
""")

st.code("""
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# 1. State Definition
class RouterState(MessagesState):
    classification: Literal["knowledge", "calc", "search", "escalate"]

# 2. Router Model
classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def classify_node(state: RouterState) -> dict:
    last_msg = state["messages"][-1].content.lower()
    
    if any(k in last_msg for k in ["lawsuit", "refund 10000", "fraud"]):
        category = "escalate"
    elif any(k in last_msg for k in ["*", "/", "calculate"]):
        category = "calc"
    elif any(k in last_msg for k in ["search", "current news", "who is"]):
        category = "search"
    else:
        category = "knowledge"
        
    return {"classification": category}

def route_decision(state: RouterState) -> str:
    return state["classification"]

# 3. Execution Lanes
def knowledge_lane(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[Knowledge Lane]: Direct answer from parametric memory.")]}

def calc_lane(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[Calculation Lane]: Exact arithmetic evaluated.")]}

def search_lane(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[Search Lane]: External facts retrieved and summarized.")]}

def escalate_lane(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[Escalation Lane]: Request forwarded to Tier-3 human specialist.")]}

# 4. Assemble Graph
builder = StateGraph(RouterState)
builder.add_node("classifier", classify_node)
builder.add_node("knowledge", knowledge_lane)
builder.add_node("calc", calc_lane)
builder.add_node("search", search_lane)
builder.add_node("escalate", escalate_lane)

builder.add_edge(START, "classifier")
builder.add_conditional_edges("classifier", route_decision)

for lane in ["knowledge", "calc", "search", "escalate"]:
    builder.add_edge(lane, END)

routing_graph = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Intent Classifier, Multi-Lane Router & Escalation Simulator", divider="rainbow")

st.markdown("""
Test how incoming user prompts are categorized into the 5 core routing lanes.
Observe the rationale, assigned toolset, and cost profiles in real time.
""")

router_eng = AgentRoutingEngine()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Query Ingestion")
    test_prompt = st.selectbox(
        "Select Test Prompt Scenario:",
        [
            "What is the mathematical formulation of cross-entropy loss?",
            "Calculate 4820 divided by 16 multiplied by 1.15.",
            "Search the latest breaking news regarding AI regulations.",
            "I demand an immediate refund of 10,000 USD or I will file a lawsuit for fraud!",
            "Cancel my subscription and update my billing address."
        ]
    )
    custom_inquiry = st.text_input("Or enter custom query:", value="")
    final_query = custom_inquiry if custom_inquiry.strip() else test_prompt
    
    evaluate_btn = st.button("🚀 Route Query", type="primary")

with wb_c2:
    st.subheader("📊 Router Dispatch Telemetry")
    if evaluate_btn:
        dispatch = router_eng.classify_and_route(final_query)
        
        badge_style = "badge-escalate" if dispatch["category"] == "ESCALATE" else "badge-route"
        st.markdown(f"### Category: <span class='{badge_style}'>{dispatch['category']}</span>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Assigned Execution Lane", dispatch["assigned_lane"])
        with col_m2:
            st.metric("Relative Cost Profile", dispatch["relative_cost"])

        st.info(f"**Tools Assigned**: `{dispatch['tools_assigned']}`")
        st.write(f"**Classification Rationale**: {dispatch['rationale']}")
    else:
        st.info("Select a query and click 'Route Query' to observe lane assignment.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: Why is intent routing preferable to binding all tools to a single generalist model?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Token Cost**: Binding 20 tools adds thousands of prompt tokens on every turn.
    2. **Accuracy**: Models with too many tools suffer from tool confusion and hallucinated parameters.
    3. **Latency**: Routing trivial queries to zero-tool lanes saves hundreds of milliseconds.
    """)

with st.expander("Q2: How does LangGraph handle Human Escalation?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph provides the `interrupt()` primitive.
    When an escalation condition is met, `interrupt()` pauses execution and saves a checkpoint.
    The host application notifies a human operator. Once the human reviews and approves the request,
    the graph is resumed with `Command(resume=supervisor_data)`.
    """)

with st.expander("Q3: When should you use a fast LLM vs regex heuristics for routing?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Regex / Keywords**: Best for deterministic triggers (e.g. keywords like `"lawsuit"`, `"refund > 5000"`, `"calc"`). Latency is sub-1ms.
    - **Fast LLM (gpt-4o-mini)**: Best for nuanced semantic intent where phrasing varies widely. Latency is 150-250ms.
    """)

with st.expander("Q4: What happens if an escalation node is reached but no checkpointer is attached to the graph?", expanded=False):
    st.markdown("""
    **Answer**:
    The graph will raise an error at runtime.
    `interrupt()` requires a persistent checkpointer (e.g. `MemorySaver`, `SqliteSaver`) to save the state snapshot while waiting for human input.
    """)

with st.expander("Q5: Can multiple routing lanes be executed in parallel?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! If a query requires both real-time search and mathematical computation, the router can return multiple branch keys,
    executing both lanes concurrently in the same superstep before merging results.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Calling interrupt() Without a Compiled Checkpointer")
    st.markdown("""
    *Mechanism*: Using `interrupt()` in an escalation node on a graph compiled without `checkpointer=...`.  
    *Consequence*: Graph crashes with `ValueError: interrupt() requires a checkpointer`.  
    *Fix*: Always pass a checkpointer instance during graph compilation (`builder.compile(checkpointer=checkpointer)`).
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Slow Classifier Bottleneck")
    st.markdown("""
    *Mechanism*: Using a slow 800ms frontier model to classify intent before routing.  
    *Consequence*: The classification overhead negates any speed benefits of lightweight execution lanes.  
    *Fix*: Use small fast models (`gpt-4o-mini`) or embedding classifiers for routing.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Misrouting Knowledge Queries to Search Tools")
    st.markdown("""
    *Mechanism*: Overly sensitive search classifiers triggering external web searches for basic definitions.  
    *Consequence*: Unnecessary external API billing and 1500ms latency delays.  
    *Fix*: Tune classification prompts to prefer parametric knowledge for evergreen topics.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Escalation Deadlocks Without Notification")
    st.markdown("""
    *Mechanism*: Graph interrupts for human approval without an alerting webhook to page the supervisor.  
    *Consequence*: The workflow remains stalled indefinitely in the database.  
    *Fix*: Pair `interrupt()` with automated Slack, PagerDuty, or email webhooks.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Intent Routing**\\nDirecting incoming queries to specialized graph lanes based on complexity and required tools.")
with col_fc2:
    st.info("**interrupt()**\\nThe LangGraph function that pauses execution and awaits human supervisor approval.")
with col_fc3:
    st.info("**Zero-Tool Lane**\\nAn execution path for conceptual questions that bypasses tool schemas to minimize latency.")
with col_fc4:
    st.info("**Human-in-the-Loop**\\nA design pattern where high-risk actions require human sign-off before committing.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the 5 core routing categories (Knowledge, Calc, Search, Action, Escalate).", value=True)
c2 = st.checkbox("I understand why zero-tool lanes reduce latency and costs for conceptual queries.", value=True)
c3 = st.checkbox("I can implement human escalation in LangGraph using the interrupt() primitive.", value=True)
c4 = st.checkbox("I know why interrupt() strictly requires a persistent checkpointer.", value=True)
c5 = st.checkbox("I understand the performance trade-offs between fast LLM classifiers and keyword heuristics.", value=True)
c6 = st.checkbox("I can assemble a multi-lane state machine with conditional routing in LangGraph.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 13.3 'Agent Routing' with full architectural and practical comprehension!")
