import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="10.1 Understanding State - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("10.1 Understanding State: Schemas, Message Channels, Reducers & Context")
st.caption("Chapter 10: Agent State and Memory | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In classical stateless software, functions receive arguments, compute an output, and release all memory. 
In contrast, an autonomous AI Agent is a **stateful cyclic system**: it must accumulate observations, 
track intermediate reasoning turns, update operational flags, and maintain context across extended executions.

In modern LangChain and LangGraph, **State** is the central nervous system of the agent. 
It defines the explicit, type-safe data schema that every node in the graph reads from and writes to.

In this lesson, you will master:
- **What is Agent State?**: The shared memory backbone connecting reasoning nodes and tool actuators.
- **Message State**: The foundational `MessagesState` schema and the `add_messages` reducer.
- **Custom State**: Extending schemas with custom domain variables, counters, and metadata.
- **State Schemas**: Deciding between Python `TypedDict` and Pydantic `BaseModel` for state contracts.
- **State Updates & Reducers**: How nodes return partial updates that merge seamlessly into global state.
- **Runtime Context**: Passing execution settings (such as `thread_id` and auth tokens) via `RunnableConfig`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockStateSnapshot:
    """Represents a frozen snapshot of agent state in module scope."""
    def __init__(self, step_name: str, messages_count: int, current_status: str, retry_count: int, variables: Dict[str, Any]):
        self.step_name = step_name
        self.messages_count = messages_count
        self.current_status = current_status
        self.retry_count = retry_count
        self.variables = variables

class MockStateChannelSimulator:
    """Simulates state accumulation, channel reducers, and custom keys in module scope."""
    def __init__(self):
        pass

    def run_state_lifecycle(self, user_prompt: str, tenant_tier: str, max_retries: int) -> List[MockStateSnapshot]:
        history: List[MockStateSnapshot] = []

        # State 0: Initial Ingestion
        s0 = MockStateSnapshot(
            step_name="0. Graph Initialization (Ingress)",
            messages_count=1,
            current_status="INITIALIZING",
            retry_count=0,
            variables={"tenant_tier": tenant_tier, "user_prompt": user_prompt}
        )
        history.append(s0)

        # State 1: Model Turn 1 (Tool Called)
        s1 = MockStateSnapshot(
            step_name="1. Agent Node (Emits Tool Call)",
            messages_count=2,
            current_status="TOOL_PENDING",
            retry_count=0,
            variables={"tenant_tier": tenant_tier, "active_tool": "query_database"}
        )
        history.append(s1)

        # State 2: Tool Execution (Observation Appended via add_messages)
        s2 = MockStateSnapshot(
            step_name="2. ToolNode (Observation Appended via Reducer)",
            messages_count=3,
            current_status="OBSERVATION_INGESTED",
            retry_count=0,
            variables={"tenant_tier": tenant_tier, "db_records_found": 42}
        )
        history.append(s2)

        # State 3: Final Synthesis Turn
        s3 = MockStateSnapshot(
            step_name="3. Agent Node (Final Synthesis & Exit)",
            messages_count=4,
            current_status="COMPLETED",
            retry_count=0,
            variables={"tenant_tier": tenant_tier, "task_completed": True, "delivered_payload": "Analytics report synthesized."}
        )
        history.append(s3)

        return history

# --- SECTION 1: ARCHITECTURE BLUEPRINT & CHANNEL ARCHITECTURE ---
st.header("1. Architecture Blueprint: The Agent State Machine & Channel Architecture", divider="rainbow")

st.markdown("""
The architecture below details how **State Channels** and **Reducers** govern communication inside modern agents. 
Nodes do not pass data directly to each other; instead, they inspect the central State, 
compute partial modifications, and publish updates back through typed channels.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Central State Hub
        subgraph cluster_state_hub {
            label = "1. CENTRAL AGENT STATE STORE (TypedDict / BaseModel)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            MessagesChannel [label="Channel: messages\\nType: Annotated[List[BaseMessage], add_messages]\\nReducer: Appends or updates messages by ID", color="#10B981"];
            CustomKeysChannel [label="Channel: custom_context\\nKeys: user_id, auth_tier, step_count\\nDefault Reducer: Last-Write-Wins (Overwrite)"];
        }

        // Level 2: Graph Nodes (Readers & Writers)
        subgraph cluster_nodes {
            label = "2. GRAPH EXECUTION NODES";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            AgentReasoningNode [label="Node: 'agent' (LLM)\\nReads state['messages']\\nReturns: {'messages': [AIMessage(...)]}"];
            ToolExecutionNode [label="Node: 'tools' (ToolNode)\\nReads latest AIMessage.tool_calls\\nReturns: {'messages': [ToolMessage(...)]}", color="#38BDF8"];
            GuardrailNode [label="Node: 'guardrail'\\nReads state &amp; verifies safety\\nReturns: {'safety_cleared': True}"];
        }

        // Level 3: Runtime Context & Config
        subgraph cluster_config {
            label = "3. RUNTIME CONTEXT (RunnableConfig)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            RunnableConfigNode [label="RunnableConfig\\nthread_id='session_9481'\\nuser_token='Bearer eyJ...'\\ncallbacks=[TraceCollector]"];
        }

        // Connections
        MessagesChannel -> AgentReasoningNode [label="Read history"];
        CustomKeysChannel -> AgentReasoningNode [label="Read context"];
        
        AgentReasoningNode -> MessagesChannel [label="Publish partial update\\n{'messages': [AIMessage]}", color="#10B981"];
        
        MessagesChannel -> ToolExecutionNode [label="Read tool_calls"];
        ToolExecutionNode -> MessagesChannel [label="Publish partial update\\n{'messages': [ToolMessage]}", color="#38BDF8"];

        RunnableConfigNode -> AgentReasoningNode [label="Inject runtime context"];
        RunnableConfigNode -> ToolExecutionNode [label="Inject credentials"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Key Architectural Principles:**
1. **Decoupled Nodes**: Nodes are pure functions `Callable[[State], Partial[State]]`. They have zero direct knowledge of other nodes.
2. **Channel-Specific Reducers**: Each field in the state schema can define its own reducer function (e.g. `add_messages` appends messages, while integers can increment or overwrite).
3. **Immutability by Default**: Nodes do not mutate state in-place; they return a dictionary containing only the fields they wish to modify.
""")

st.divider()

# --- SECTION 2: WHAT IS AGENT STATE? ---
st.header("2. What Is Agent State? (The Central Nervous System)", divider="rainbow")

st.markdown("""
In LangGraph, **State** is a shared data structure that represents the current snapshot of an agent's reality. 
As an agent executes, it moves through a sequence of discrete state transitions:
`State_0 ➔ Node A ➔ State_1 ➔ Node B ➔ State_2 ... ➔ State_Final`.

Without state, an agent could not remember what tools it already called, what errors occurred, 
or what user question it was trying to answer.
""")

st.divider()

# --- SECTION 3: MESSAGE STATE ---
st.header("3. Message State (`MessagesState` & `add_messages` Reducer)", divider="rainbow")

st.markdown("""
Because 90% of agents are conversational, LangChain provides a built-in schema called `MessagesState`. 
It consists of a single channel: a list of `BaseMessage` objects governed by the **`add_messages` reducer**.
""")

st.subheader("The `add_messages` Reducer Algorithm")
st.markdown("""
In classical Python, returning `{"messages": [new_msg]}` would overwrite the entire list. 
The `add_messages` reducer modifies this behavior:
1. **Append**: If `new_msg.id` is not present in the current message list, it is appended to the end.
2. **Update/Overwrite**: If `new_msg.id` matches an existing message ID in the list, the existing message is updated in-place (useful for streaming edits or error corrections).
3. **Delete**: If a message has `RemoveMessage(id=...)`, the corresponding message is purged from the list.
""")

st.code('''from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Builtin MessagesState is defined as:
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
''', language="python")

st.divider()

# --- SECTION 4: CUSTOM STATE ---
st.header("4. Custom State (Extending Schemas with Domain Context)", divider="rainbow")

st.markdown("""
Real-world enterprise agents require more than just raw messages. You often need to track:
- Authenticated user ID and role (`user_id`, `role`).
- Accumulated intermediate data (e.g. extracted entities, raw database rows).
- Operational flags (e.g. `is_authenticated`, `requires_supervisor_review`).
- Retry and iteration counters (`iteration_count: Annotated[int, operator.add]`).
""")

st.code('''import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class EnterpriseIncidentState(TypedDict):
    # Channel 1: Message history (appends via add_messages)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Channel 2: Domain Context (overwritten on update)
    incident_id: str
    tenant_id: str
    severity: str
    
    # Channel 3: Intermediate Structured Findings
    impacted_services: List[str]
    
    # Channel 4: Counter with addition reducer
    total_tool_calls_executed: Annotated[int, operator.add]
''', language="python")

st.divider()

# --- SECTION 5: STATE SCHEMAS ---
st.header("5. State Schemas (`TypedDict` vs Pydantic `BaseModel`)", divider="rainbow")

st.markdown("""
LangGraph supports defining state schemas using either Python's standard `TypedDict` or Pydantic's `BaseModel`.
""")

st.subheader("Comparison: `TypedDict` vs Pydantic `BaseModel`")
schema_cmp_df = pd.DataFrame({
    "Criterion": ["Runtime Validation", "Performance Overhead", "Syntax & Usage", "Best Used For"],
    "TypedDict": ["None (Static typing only)", "Zero (Pure native dict)", "Standard dict access: state['key']", "Internal state performance (Recommended)"],
    "Pydantic BaseModel": ["Strict (Evaluated on every update)", "Slight (Rust validation)", "Dot notation: state.key", "Ingress/Egress boundary validation"]
})
st.dataframe(schema_cmp_df, use_container_width=True)

st.info("""
💡 **Industry Recommendation**: Use **`TypedDict`** for the internal agent state graph to avoid unnecessary validation overhead on high-frequency state updates, and use **Pydantic** for input/output boundary payloads and tool arguments.
""")

st.divider()

# --- SECTION 6: STATE UPDATES ---
st.header("6. State Updates (Partial Returns & Reducer Fusion)", divider="rainbow")

st.markdown("""
A common beginner mistake is assuming a node must return the entire state dictionary. 
In LangGraph, **nodes only return the partial dictionary of fields they want to change**.
""")

st.code('''# Node definition: only returns the keys it updates
def incident_classifier_node(state: EnterpriseIncidentState) -> Dict[str, Any]:
    # Read from existing state:
    messages = state["messages"]
    last_msg = messages[-1].content
    
    # Compute updates:
    detected_severity = "CRITICAL" if "database down" in last_msg.lower() else "MEDIUM"
    
    # Return ONLY the changed keys!
    return {
        "severity": detected_severity,
        "total_tool_calls_executed": 1  # operator.add will increment this by 1!
    }
''', language="python")

st.divider()

# --- SECTION 7: RUNTIME CONTEXT ---
st.header("7. Runtime Context (`RunnableConfig` & Execution Isolation)", divider="rainbow")

st.markdown("""
Not all information belongs in the agent's semantic state. 
Sensitive configuration parameters—such as user authentication tokens, database connection pools, 
callbacks, and session thread identifiers—are passed via **`RunnableConfig`**.
""")

st.subheader("Key Parameters in `RunnableConfig`")
st.markdown("""
- **`configurable.thread_id`**: Identifies the unique conversation thread for state persistence.
- **`configurable.user_id`**: Identifies the calling tenant or user.
- **`callbacks`**: Attaches tracing observers (e.g. LangSmith, OpenTelemetry).
- **`tags` / `metadata`**: Attached to trace spans for filtering and billing analysis.
""")

st.code('''from langchain_core.runnables import RunnableConfig

def secure_tool_executor(state: dict, config: RunnableConfig):
    # Extract runtime context securely from config:
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    user_token = config.get("configurable", {}).get("user_jwt")
    
    print(f"Executing request for thread '{thread_id}' with bearer token.")
    return {"messages": ["Observation verified"]}
''', language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Type-Safe Agent State Engine", divider="rainbow")

st.markdown("""
The following complete production pattern illustrates an enterprise agent state definition 
featuring custom typed channels, addition reducers, message accumulation, and config extraction.
""")

st.code('''import operator
import logging
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

logger = logging.getLogger("EnterpriseStateEngine")

# 1. Define Strict Enterprise State Contract
class ITSupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_tier: str
    active_incident_code: Optional[str]
    steps_taken: Annotated[int, operator.add]
    remediation_status: str

# 2. Define Stateful Graph Nodes
def intake_node(state: ITSupportState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Intake processing for thread: {thread_id}")
    return {
        "steps_taken": 1,
        "remediation_status": "ANALYZING"
    }

def triage_node(state: ITSupportState) -> Dict[str, Any]:
    tier = state.get("customer_tier", "STANDARD")
    last_message = state["messages"][-1].content
    
    if "urgent" in last_message.lower() and tier == "ENTERPRISE":
        code = "INC-P1-CRITICAL"
    else:
        code = "INC-P3-NORMAL"

    logger.info(f"Assigned incident code: {code}")
    return {
        "active_incident_code": code,
        "steps_taken": 1,
        "remediation_status": "TRIAGED",
        "messages": [AIMessage(content=f"Incident triaged with priority classification {code}.")]
    }

# 3. Compile the State Machine
def build_governed_state_machine():
    workflow = StateGraph(ITSupportState)
    
    workflow.add_node("intake", intake_node)
    workflow.add_node("triage", triage_node)
    
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "triage")
    workflow.add_edge("triage", END)
    
    return workflow.compile()
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Agent State Channel & Reducer Simulator", divider="rainbow")

st.markdown("""
Observe how state evolves across a multi-stage agent pipeline. 
Inspect how the `add_messages` reducer appends messages and how custom context keys update at each node transition.
""")

state_sim = MockStateChannelSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("⚙️ Configure Session Parameters")
    user_prompt_input = st.text_input("User Initial Message", value="Our payment gateway is returning HTTP 504 timeouts.")
    tier_input = st.selectbox("Customer SLA Tier", ["ENTERPRISE", "GROWTH", "STANDARD"])
    retry_bound = st.slider("Max Allowed Retries", min_value=1, max_value=5, value=3)

with col_wb2:
    st.subheader("📋 Active State Schema Contract")
    st.code('''class ITSupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tenant_tier: str
    active_tool: Optional[str]
    db_records_found: int
    task_completed: bool''', language="python")

if st.button("🚀 Step Through State Lifecycle", type="primary"):
    snapshots = state_sim.run_state_lifecycle(user_prompt_input, tier_input, retry_bound)
    
    st.subheader("📊 State Progression Across Node Transitions")
    for snap in snapshots:
        with st.expander(f"📍 {snap.step_name} - Messages Count: {snap.messages_count}", expanded=True):
            m1, m2 = st.columns(2)
            m1.metric("Execution Status", snap.current_status)
            m2.metric("Total Messages in State", snap.messages_count)
            
            st.write("#### Channel Variables (`state` keys):")
            st.json(snap.variables)

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does the add_messages reducer handle messages with matching IDs?",
     "When a message with an existing ID is passed to add_messages, the reducer updates the existing message in-place rather than appending a duplicate. This enables in-place editing of messages, streaming updates, and replacing placeholder messages with completed outputs."),
    ("What happens if a node returns a dictionary containing a key that is NOT defined in the State schema?",
     "In LangGraph, if a node returns an undeclared key, a KeyError or InvalidUpdateError is raised during graph execution. Every piece of data modified by nodes must be explicitly defined in the state schema contract."),
    ("Why should operational counters in state schemas use Annotated[int, operator.add] instead of a plain int?",
     "By default, state keys use last-write-wins semantics (the returned value overwrites the old value). By using Annotated[int, operator.add], returning {'step_count': 1} automatically adds 1 to the existing count, enabling concurrent nodes to update counters safely without race conditions."),
    ("What is the difference between data stored in State versus data passed in RunnableConfig?",
     "State stores mutable, domain-specific data that the agent reasons over and modifies (messages, findings, flags). RunnableConfig passes immutable, runtime execution context that nodes need to access but should not modify (thread_id, user JWT tokens, trace callbacks)."),
    ("Why is TypedDict generally preferred over Pydantic BaseModel for internal LangGraph state?",
     "TypedDict offers zero runtime overhead because it operates as a native Python dictionary while providing static type checking during development. In high-frequency cyclic agent graphs executing dozens of node transitions, TypedDict avoids Pydantic's per-step validation overhead.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Mutating State in Place Inside Nodes. Doing `state['messages'].append(new_msg)` directly inside a node causes subtle caching bugs and breaks time-travel checkpointers. Always return a new dictionary `return {'messages': [new_msg]}`.")
    st.error("**Trap 2 (Breaking Error)**: Forgetting Reducer Annotations on List Channels. Declaring `items: List[str]` without a reducer means returning `{'items': ['c']}` will completely overwrite `['a', 'b']` with `['c']`. Use `Annotated[List[str], operator.add]` to append.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Giant Binary Blobs in State. Storing multi-megabyte image buffers or raw PDFs directly inside the state dictionary bloats checkpoint databases and slows state deserialization. Store blobs in S3 and store pre-signed URLs in state.")
    st.warning("**Trap 4 (Quality Warning)**: Storing Unredacted Auth Tokens in State. Putting raw JWT tokens into state causes them to be logged in trace stores and serialized into persistent checkpoints. Pass credentials through RunnableConfig.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: add_messages Invariant"):
        st.markdown("**Appends new messages, updates messages with existing IDs, and supports RemoveMessage.**")
    with st.expander("🎴 Flashcard 2: Partial Node Returns"):
        st.markdown("**Nodes only return the specific dictionary keys they want to update, not the whole state.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Annotated Reducers"):
        st.markdown("**Annotated[T, reducer_func] specifies custom merge logic for individual state channels.**")
    with st.expander("🎴 Flashcard 4: RunnableConfig Role"):
        st.markdown("**Carries runtime execution metadata (thread_id, auth tokens) isolated from state.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the role of State as the shared data backbone of agent graphs.", value=True)
chk2 = st.checkbox("I know how the add_messages reducer manages message appending and updates.", value=True)
chk3 = st.checkbox("I can define custom state schemas with custom channels and operator reducers.", value=True)
chk4 = st.checkbox("I know when to use TypedDict vs Pydantic BaseModel for state definitions.", value=True)
chk5 = st.checkbox("I understand why nodes return partial dictionaries rather than mutating state in-place.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 agent state interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 10.1 Understanding State. Ready for 10.2 Conversation Memory!")
