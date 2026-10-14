import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="20.4 Multi-Agent State - LangGraph Curriculum",
    page_icon="🗄️",
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
    .badge-state {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-priv {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #7E22CE;
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
class MultiAgentStateSimulator:
    """
    Simulates multi-agent state architectures:
    - Shared blackboard state
    - Private scratchpads
    - Handoff payloads and agent memory stores
    """
    def __init__(self):
        pass

    def simulate_state_handoff(
        self,
        customer_query: str,
        account_tier: str,
        use_private_scratchpad: bool
    ) -> Dict[str, Any]:
        # 1. Triage Agent Phase
        shared_state = {
            "customer_id": "cust_8821",
            "tier": account_tier,
            "query": customer_query,
            "active_agent": "TriageAgent"
        }
        
        triage_scratchpad = {
            "raw_auth_tokens": "jwt_token_secret_xyz",
            "risk_score": 0.12,
            "routing_confidence": 0.99
        } if use_private_scratchpad else None

        # 2. Handoff event
        handoff_packet = {
            "from_agent": "TriageAgent",
            "to_agent": "BillingSpecialist",
            "reason": "Dispute on invoice INV-2024-09",
            "timestamp": "2026-10-13T10:00:00Z"
        }

        # 3. Billing Agent Phase
        billing_state_update = {
            "active_agent": "BillingSpecialist",
            "disputed_invoice_id": "INV-2024-09",
            "resolution": "Credit issued of $45.00 USD based on VIP Tier terms."
        }
        shared_state.update(billing_state_update)

        return {
            "shared_state": shared_state,
            "private_scratchpad": triage_scratchpad,
            "handoff_packet": handoff_packet
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Multi-Agent State Topologies, Memory Partitions, and Handoff Dynamics", divider="rainbow")

st.markdown("""
Managing state across multiple cooperating agents is significantly more nuanced than managing a single agent's memory.
Without careful state design:
- Private variables from one agent leak into others, bloating prompts.
- Agents overwrite each other's outputs during concurrent supersteps.
- Handoffs between agents drop critical context, forcing users to repeat themselves.

LangGraph solves multi-agent state through **hybrid state architectures**:
- **Shared Blackboard**: Global channels visible to all participating agents (e.g. conversational chat history, final deliverables).
- **Private Scratchpads**: Dedicated sub-dictionaries or isolated child subgraph channels where specialists store intermediate variables.
- **Structured Handoff Protocols**: Explicit state transitions containing reason tokens and contextual summaries.

The architectural blueprint below illustrates the partition between shared global channels and private specialist state channels.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiAgentStateTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_global_blackboard {
            label="1. Global Shared State Blackboard (Visible to All Agents)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            bb_messages [label="messages: Annotated[list, add_messages]\\n(Full Conversational History)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            bb_context [label="active_agent: str | customer_profile: dict\\n(Universal Session Context)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_agent_a {
            label="2. Agent A: Customer Support Specialist";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            agent_a_node [label="Support Node Execution", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            priv_a [label="Private Scratchpad A\\n(CRM raw dumps, session tokens)", fillcolor="#2E1065", fontcolor="#C084FC", color="#7E22CE"];
        }

        subgraph cluster_agent_b {
            label="3. Agent B: Billing Dispute Specialist";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            agent_b_node [label="Billing Node Execution", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            priv_b [label="Private Scratchpad B\\n(Stripe API payloads, invoice tables)", fillcolor="#2E1065", fontcolor="#C084FC", color="#7E22CE"];
        }

        subgraph cluster_memory_store {
            label="4. Multi-Tier Memory Store (Short-Term vs Long-Term)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            short_term_ckpt [label="Short-Term Memory\\n(Thread Checkpointer: Current Run)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            long_term_store [label="Long-Term Cross-Thread Store\\n(BaseStore: User Preferences)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        bb_messages -> agent_a_node [label="Reads History"];
        agent_a_node -> priv_a [label="Writes Private State"];
        agent_a_node -> bb_messages [label="Handoff: transfer_to_billing", color="#FBBF24"];
        
        bb_messages -> agent_b_node [label="Receives Handoff & History"];
        agent_b_node -> priv_b [label="Writes Private State"];
        agent_b_node -> bb_messages [label="Appends Final Solution"];

        agent_b_node -> short_term_ckpt [label="Saved in Thread Checkpoint"];
        agent_b_node -> long_term_store [label="Updates User Memory Profile"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: SHARED STATE
# ==============================================================================
st.header("2. Shared State: The Global Blackboard Pattern and Annotated Channel Reducers", divider="rainbow")

st.markdown("""
In the **Global Blackboard** pattern, agents collaborate by reading and writing to common state channels.
To prevent concurrent worker collisions, LangGraph utilizes **Annotated Channel Reducers**:
- `Annotated[List[BaseMessage], add_messages]`: Appends incoming message deltas.
- `Annotated[List[dict], operator.add]`: Concatenates list items from multiple parallel agents.
- Custom Reducers: Define domain-specific conflict resolution logic (e.g. merging dictionaries by timestamp).
""")

st.code('''from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class BlackboardState(TypedDict):
    # Appends new messages from any agent cleanly
    messages: Annotated[List[BaseMessage], add_messages]
    # Concatenates findings from parallel specialist agents
    aggregated_findings: Annotated[List[str], operator.add]
    # Replaces current active assignment
    active_agent: str''', language="python")

# ==============================================================================
# SECTION 3: PRIVATE STATE
# ==============================================================================
st.header("3. Private State: Encapsulated Scratchpads, Subgraph Isolation, and Channel Masking", divider="rainbow")

st.markdown("""
Not all data belongs on the global blackboard.
A Python Coding Agent running unit tests generates large compiler tracebacks that have zero value to an Executive Writing Agent.
LangGraph provides two ways to enforce **Private State**:
1. **Subgraph State Isolation**: Embedding the specialist as a compiled child graph with its own private `TypedDict`.
2. **Channel Namespaces**: Using dedicated dictionary keys (e.g. `agent_scratchpads: Dict[str, dict]`) keyed by agent name.
""")

# ==============================================================================
# SECTION 4: AGENT MEMORY
# ==============================================================================
st.header("4. Agent Memory: Short-Term Thread Checkpoints vs Long-Term Cross-Session Stores", divider="rainbow")

st.markdown("""
Multi-agent systems operate across two distinct memory horizons:
""")

col_mem1, col_mem2 = st.columns(2)
with col_mem1:
    st.markdown("#### Short-Term Memory (Thread Level)")
    st.markdown("""
    - Managed by the checkpointer (`PostgresSaver`, `MemorySaver`).
    - Retains messages, pending tasks, and intermediate calculations for a single session (`thread_id`).
    - Discarded or archived when the workflow finishes.
    """)

with col_mem2:
    st.markdown("#### Long-Term Memory (Cross-Thread Level)")
    st.markdown("""
    - Managed by LangGraph's `BaseStore` (e.g. `InMemoryStore`, `PostgresStore`).
    - Stored under semantic namespaces: `("users", user_id, "memories")`.
    - Persists user preferences, historical patterns, and learned organizational guidelines across all future conversations.
    """)

# ==============================================================================
# SECTION 5: AGENT HANDOFFS
# ==============================================================================
st.header("5. Agent Handoffs: Transfer Tools, Authority Delegation, and Context Bundling", divider="rainbow")

st.markdown("""
When an agent determines that a user's request exceeds its mandate, it initiates a **handoff**.
The modern pattern in LangGraph is **Transfer Tool Calling**:
""")

st.code('''# Defining explicit transfer tools
def transfer_to_fraud_analyst(suspicion_reason: str) -> str:
    """Transfer to Fraud Specialist when suspicious activity is detected."""
    return f"Delegating to fraud team: {suspicion_reason}"

# The router edge inspects tool calls to determine the next node
def route_handoff(state: State) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_name = last_msg.tool_calls[0]["name"]
        if tool_name == "transfer_to_fraud_analyst":
            return "fraud_specialist_node"
    return END''', language="python")

# ==============================================================================
# SECTION 6: AGENT-TO-AGENT COMMUNICATION
# ==============================================================================
st.header("6. Agent-to-Agent Communication: Point-to-Point Messaging, Broadcasts, and Event Busses", divider="rainbow")

st.markdown("""
In LangGraph, communication between agents is decoupled through the state graph:
- **Point-to-Point**: Agent A yields an update with `next_agent="agent_b"`, routing directly to Agent B.
- **Broadcast (Fan-Out)**: A supervisor node returns edges to multiple specialist nodes, causing them to execute in parallel in the same superstep.
- **Event Bus**: Agents emit events to an audit channel while continuing their reasoning.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Agent State Graph with Private Scratchpads and Handoffs", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern demonstrating shared state, private scratchpad isolation, and transfer tool handoffs.
""")

st.code('''from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class EnterpriseMultiAgentState(TypedDict):
    # Global Shared Channels
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    active_agent: str
    # Private Scratchpad Channels (isolated per agent)
    support_scratchpad: Optional[dict]
    billing_scratchpad: Optional[dict]

def support_agent_node(state: EnterpriseMultiAgentState) -> dict:
    query = state["messages"][-1].content
    # Simulates handoff decision
    if "invoice" in query.lower() or "refund" in query.lower():
        return {
            "active_agent": "billing_agent",
            "support_scratchpad": {"triage_note": "Identified billing keyword. Delegating."},
            "messages": [AIMessage(content="Transferring you to our billing specialist...", name="support_agent")]
        }
    return {
        "messages": [AIMessage(content="I can answer your general support query.", name="support_agent")]
    }

def billing_agent_node(state: EnterpriseMultiAgentState) -> dict:
    return {
        "active_agent": "billing_agent",
        "billing_scratchpad": {"refund_processed": True, "amount": 25.0},
        "messages": [AIMessage(content="Billing Specialist here: Your $25.00 refund has been issued.", name="billing_agent")]
    }

def route_agent_handoff(state: EnterpriseMultiAgentState) -> str:
    if state.get("active_agent") == "billing_agent":
        # If active agent was switched to billing, run billing
        last_name = getattr(state["messages"][-1], "name", "")
        if last_name == "billing_agent":
            return END
        return "billing_agent"
    return END

builder = StateGraph(EnterpriseMultiAgentState)
builder.add_node("support_agent", support_agent_node)
builder.add_node("billing_agent", billing_agent_node)

builder.add_edge(START, "support_agent")
builder.add_conditional_edges("support_agent", route_agent_handoff, {"billing_agent": "billing_agent", END: END})
builder.add_edge("billing_agent", END)

app = builder.compile(checkpointer=MemorySaver())

result = app.invoke(
    {
        "messages": [HumanMessage(content="I need a refund on my last invoice please.")],
        "customer_id": "cust_101",
        "active_agent": "support_agent"
    },
    config={"configurable": {"thread_id": "handoff_demo_01"}}
)
print("Conversation turns:", len(result["messages"]))
print("Final Message:", result["messages"][-1].content)''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: State Boundary & Handoff Protocol Simulator", divider="rainbow")

st.markdown("""
Simulate how state channels, private scratchpads, and handoff packets transition between agents during a live conversation.
""")

sim = MultiAgentStateSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Handoff Parameters")
    cust_query = st.text_input("User Message", value="I was overbilled on invoice INV-2024-09")
    tier_choice = st.selectbox("Account Tier", ["Standard Tier", "VIP Enterprise Tier"])
    enable_scratchpad = st.toggle("Enable Isolated Private Scratchpads", value=True)

    if st.button("🚀 Execute Multi-Agent State Handoff"):
        h_res = sim.simulate_state_handoff(cust_query, tier_choice, enable_scratchpad)
        st.session_state["state_handoff_res"] = h_res

with col_sim2:
    st.subheader("📋 State Channel Inspection")
    if "state_handoff_res" in st.session_state:
        res = st.session_state["state_handoff_res"]
        
        st.markdown("#### 1. Global Shared State (Blackboard)")
        st.json(res["shared_state"])

        st.markdown("#### 2. Explicit Handoff Packet")
        st.json(res["handoff_packet"])

        if res["private_scratchpad"]:
            st.markdown("#### 3. Agent A Private Scratchpad (Hidden from Agent B)")
            st.json(res["private_scratchpad"])
            st.success(" Clean Data Encapsulation: Sensitive tokens remained private!")
        else:
            st.warning("⚠️ Private scratchpad disabled: all data leaked onto global state.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the Global Blackboard pattern in multi-agent state design?", expanded=False):
    st.markdown("""
    **Answer**:
    The Global Blackboard is a centralized state dictionary visible to all agents.
    Agents post findings, messages, and deliverables to shared channels configured with reducers (such as `add_messages`), allowing other agents to inspect progress and build upon peer contributions.
    """)

with st.expander("Q2: Why should agents maintain private scratchpads instead of storing everything on the blackboard?", expanded=False):
    st.markdown("""
    **Answer**:
    Private scratchpads prevent prompt bloat and context window degradation.
    Intermediate variables (like raw web scrapes or compiler debug traces) only matter to the specialist generating them and would confuse other agents if placed on the global state.
    """)

with st.expander("Q3: How does LangGraph handle concurrent updates to the same state channel?", expanded=False):
    st.markdown("""
    **Answer**:
    Channels must be declared with a **Reducer** function (e.g. `Annotated[list, operator.add]`).
    When multiple nodes execute in parallel during a superstep, the Pregel engine collects their updates and executes the reducer to merge the values deterministically.
    """)

with st.expander("Q4: How does LangGraph distinguish between short-term memory and long-term memory across agents?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Short-Term Memory** is thread-scoped, managed by checkpointers (`PostgresSaver`) to persist the active multi-agent conversation.
    - **Long-Term Memory** is cross-thread, managed by `BaseStore` to persist shared agent knowledge, user preferences, and institutional guidelines across independent sessions.
    """)

with st.expander("Q5: What is the modern mechanism for implementing agent handoffs in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Using transfer tools (e.g. `transfer_to_agent_b()`).
    When an LLM invokes a transfer tool, conditional routing edges inspect the tool call name and route execution directly to the designated specialist node.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Invalid Channel Update Collisions Without Reducers")
    st.markdown("""
    *Mechanism*: Two parallel agents returning `{"status": "..."}` for a channel without a reducer.  
    *Consequence*: Raises `InvalidUpdateError` because LangGraph cannot determine which update wins.  
    *Fix*: Always use an `Annotated` channel reducer or write a custom conflict resolution function.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Dropping Context During Agent Handoffs")
    st.markdown("""
    *Mechanism*: Re-initializing state with an empty `messages` list when delegating to another agent.  
    *Consequence*: The receiving agent has no conversational history and asks the user to repeat everything.  
    *Fix*: Maintain the shared `messages` channel across handoffs.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Privacy Warning): Leaking Sensitive API Tokens into Shared History")
    st.markdown("""
    *Mechanism*: Storing database credentials or raw auth headers inside `AIMessage` content.  
    *Consequence*: Tokens are saved to the checkpointer and forwarded to all subsequent LLM calls.  
    *Fix*: Keep authentication tokens in private state scratchpads or environment variables.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Cost Warning): Unbounded Growth of the Shared Blackboard")
    st.markdown("""
    *Mechanism*: Letting `messages` grow past 100 turns in multi-agent loops without summarization or trimming.  
    *Consequence*: Token costs scale quadratically on every turn.  
    *Fix*: Implement a periodic message trimming node (`trim_messages`).
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Global Blackboard**\\nA shared state dictionary with channel reducers visible to all participating agents.")
with col_fc2:
    st.info("**Private Scratchpad**\\nIsolated state channels storing intermediate data specific to one agent.")
with col_fc3:
    st.info("**Transfer Tool**\\nA tool function invoked by an agent to delegate control to a peer specialist.")
with col_fc4:
    st.info("**BaseStore**\\nLangGraph cross-thread key-value storage for long-term agent memory.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the Global Blackboard pattern and how channel reducers resolve updates.", value=True)
c2 = st.checkbox("I know how to implement private scratchpads to protect the shared context window.", value=True)
c3 = st.checkbox("I understand the difference between thread checkpointers and BaseStore long-term memory.", value=True)
c4 = st.checkbox("I can implement agent handoffs using transfer tools and conditional edges.", value=True)
c5 = st.checkbox("I know how to avoid InvalidUpdateError collisions during parallel agent execution.", value=True)
c6 = st.checkbox("I understand why sensitive authentication credentials must be kept out of message channels.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 20.4 'Multi-Agent State' with full architectural and practical comprehension!")
