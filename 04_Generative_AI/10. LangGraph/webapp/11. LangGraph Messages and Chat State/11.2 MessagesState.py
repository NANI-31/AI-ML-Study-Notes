import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="11.2 MessagesState - LangGraph Curriculum",
    page_icon="📦",
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
    .badge-state {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-replace {
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
class ReducerSimulatorEngine:
    """
    Simulates the add_messages reducer mechanics:
    - Appending new message IDs
    - Updating existing messages in-place upon ID match
    - Deleting messages on RemoveMessage sentinel
    """
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.state_messages = [
            {"id": "msg_001", "role": "system", "content": "You are a customer service assistant."},
            {"id": "msg_002", "role": "human", "content": "I need help with order 4492."},
            {"id": "msg_003", "role": "ai", "content": "Looking up order 4492 right now."}
        ]

    def apply_delta(self, operation: str, msg_id: str, content: str, role: str) -> Dict[str, Any]:
        before_state = list(self.state_messages)
        action_taken = ""

        if operation == "APPEND":
            new_msg = {"id": msg_id, "role": role, "content": content}
            self.state_messages.append(new_msg)
            action_taken = f"Appended new message [{msg_id}] to end of sequence."
        elif operation == "REPLACE":
            replaced = False
            for idx, m in enumerate(self.state_messages):
                if m["id"] == msg_id:
                    self.state_messages[idx] = {"id": msg_id, "role": role, "content": content}
                    replaced = True
                    action_taken = f"Replaced existing message [{msg_id}] in-place."
                    break
            if not replaced:
                new_msg = {"id": msg_id, "role": role, "content": content}
                self.state_messages.append(new_msg)
                action_taken = f"ID [{msg_id}] not found; appended as new message."
        elif operation == "REMOVE":
            original_len = len(self.state_messages)
            self.state_messages = [m for m in self.state_messages if m["id"] != msg_id]
            if len(self.state_messages) < original_len:
                action_taken = f"Deleted message [{msg_id}] via RemoveMessage sentinel."
            else:
                action_taken = f"RemoveMessage target [{msg_id}] was not present in state."

        return {
            "before_count": len(before_state),
            "after_count": len(self.state_messages),
            "action_taken": action_taken,
            "current_state": self.state_messages
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The MessagesState Schema and Channel Reducer Mechanics", divider="rainbow")

st.markdown("""
`MessagesState` is LangGraph's standard built-in TypedDict designed specifically for conversational workflows.
Instead of maintaining a generic unstructured list, `MessagesState` binds the channel to the
`add_messages` reducer function.
This gives conversational graphs automatic support for **monotonic accumulation**, **ID-based message updates**,
and **surgical message pruning** (`RemoveMessage`).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MessagesStateArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_schema {
            label = "1. MessagesState Class Definition";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            BASE_SCHEMA [label="class MessagesState(TypedDict):\\n  messages: Annotated[List[BaseMessage], add_messages]", fillcolor="#0F172A", color="#38BDF8"];
            EXTENDED_SCHEMA [label="class CustomEnterpriseState(MessagesState):\\n  user_id: str\\n  current_intent: str\\n  retry_count: int", fillcolor="#1E1E38", color="#8B5CF6"];

            BASE_SCHEMA -> EXTENDED_SCHEMA [label="Inheritance / Subclassing"];
        }

        subgraph cluster_nodes {
            label = "2. Graph Nodes Emitting Deltas";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            NODE_A [label="Node A (Append)\\nreturn {'messages': [AIMessage(id='x3', ...)]}", fillcolor="#064E3B", color="#34D399"];
            NODE_B [label="Node B (Replace)\\nreturn {'messages': [AIMessage(id='x3', ...)]}", fillcolor="#431407", color="#FB923C"];
            NODE_C [label="Node C (Prune)\\nreturn {'messages': [RemoveMessage(id='x1')]}", fillcolor="#3B1A1A", color="#F87171"];
        }

        subgraph cluster_reducer {
            label = "3. add_messages Reducer Reconciliation";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            REDUCER [label="add_messages(current_list, new_messages)\\n- Scans message.id\\n- New ID -> Append\\n- Matching ID -> Replace In-Place\\n- RemoveMessage -> Delete Target", fillcolor="#0F172A", color="#F59E0B"];
            FINAL_STATE [label="Resulting Persisted State\\n[Msg1, Msg2, Msg3_Updated]", fillcolor="#0F172A", color="#34D399"];

            REDUCER -> FINAL_STATE [label="Committed to thread checkpoint"];
        }

        EXTENDED_SCHEMA -> NODE_A [label="State passed to nodes"];
        NODE_A -> REDUCER [label="Delta in"];
        NODE_B -> REDUCER [label="Delta in"];
        NODE_C -> REDUCER [label="Delta in"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Core Design Rule:</b> In LangGraph, you rarely need to build message lists from scratch.
Importing <code>from langgraph.graph import MessagesState</code> provides the standard channel with
production-tested de-duplication and replacement semantics out of the box.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: UNDERSTANDING MESSAGESSTATE
# ==============================================================================
st.header("2. Understanding MessagesState: The Standard Conversational Container", divider="rainbow")

st.markdown("""
What is `MessagesState`? Under the hood, it is simply a pre-defined `TypedDict` included in LangGraph:
""")

st.code("""
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The actual source definition in langgraph.graph.state:
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
""", language="python")

st.markdown("""
Using `MessagesState` gives you three immediate guarantees:
1. **Type Safety**: IDEs and linters recognize `state["messages"]` as a list of `BaseMessage` objects.
2. **Channel Reducer**: Returning `{"messages": [new_message]}` from any node automatically appends to the conversation without overwriting historical turns.
3. **Prebuilt Compatibility**: Prebuilt components like `ToolNode` and `tools_condition` expect this exact state schema.
""")

# ==============================================================================
# SECTION 3: EXTENDING MESSAGESSTATE
# ==============================================================================
st.header("3. Extending MessagesState: Subclassing and Domain Field Composition", divider="rainbow")

st.markdown("""
In real-world enterprise applications, conversational agents require more than just messages.
They need user IDs, customer entitlement tiers, database session tokens, and routing flags.
You can **extend `MessagesState` directly via subclassing**:
""")

st.code("""
from langgraph.graph import MessagesState
from typing import Optional, Dict, Any, Literal

# Subclassing MessagesState preserves the 'messages' channel while adding custom fields
class EnterpriseAgentState(MessagesState):
    user_id: str
    account_tier: Literal["Standard", "Pro", "Enterprise"]
    session_metadata: Dict[str, Any]
    requires_human_approval: bool
    escalation_reason: Optional[str]
""", language="python")

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.info("**Inherited Channel**\\n`messages` channel retains the full `add_messages` reducer behavior without re-declaring `Annotated`.")
with col_ex2:
    st.info("**Standard Key Behavior**\\nAdded fields (e.g. `account_tier`) follow default LangGraph behavior: returning a key in a node updates/overwrites its value.")

# ==============================================================================
# SECTION 4: ADDING CUSTOM STATE
# ==============================================================================
st.header("4. Adding Custom State: Managing Domain Keys Alongside Conversation Channels", divider="rainbow")

st.markdown("""
When combining custom fields with message channels, you can apply custom reducers to non-message keys as well:
""")

st.code("""
import operator
from typing import Annotated, List
from langgraph.graph import MessagesState

class AnalyticsAgentState(MessagesState):
    # Scalar key: overwrites by default
    query_intent: str
    
    # Custom reducer: appends items to a list
    visited_nodes: Annotated[List[str], operator.add]
    
    # Custom reducer: tracks numeric increments
    total_token_cost: Annotated[float, operator.add]
""", language="python")

# ==============================================================================
# SECTION 5: MESSAGE REDUCERS
# ==============================================================================
st.header("5. Message Reducers: Custom Reducers vs Built-in add_messages", divider="rainbow")

st.markdown("""
A **reducer** is a function that takes two arguments: `(current_state_value, incoming_update_value)`
and returns the merged result.
""")

comp_df = pd.DataFrame([
    {
        "Reducer Type": "operator.add",
        "Append Logic": "Concatenates lists directly: a + b",
        "Duplicate Handling": "Creates duplicates; cannot replace messages",
        "Pruning Support": "None"
    },
    {
        "Reducer Type": "add_messages (LangGraph Built-in)",
        "Append Logic": "Appends messages if ID is new",
        "Duplicate Handling": "Replaces message in-place if ID matches existing message",
        "Pruning Support": "Full support via RemoveMessage(id=...)"
    },
    {
        "Reducer Type": "Custom Window Reducer",
        "Append Logic": "Appends and automatically slices to last N messages",
        "Duplicate Handling": "Custom logic required",
        "Pruning Support": "Automated tail trimming"
    }
])
st.table(comp_df)

# ==============================================================================
# SECTION 6: MESSAGE ACCUMULATION
# ==============================================================================
st.header("6. Message Accumulation: Monotonic Growth and Concurrent Superstep Merges", divider="rainbow")

st.markdown("""
During a multi-turn conversation or multi-branch execution, the `add_messages` reducer
ensures that concurrent branches reconcile safely:
""")

with st.expander("🔍 View Message Accumulation and Branch Merging Flow", expanded=True):
    accum_dot = """
    digraph MessageAccumulation {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CURRENT [label="Superstep N State\\n[Msg1, Msg2]", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_parallel {
            label = "Concurrent Branch Execution";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            BRANCH_1 [label="Worker 1 returns\\n{'messages': [Msg3(id='c')]}", fillcolor="#064E3B", color="#34D399"];
            BRANCH_2 [label="Worker 2 returns\\n{'messages': [Msg4(id='d')]}", fillcolor="#1E1E38", color="#8B5CF6"];
        }

        MERGED [label="Superstep N+1 Merged State\\n[Msg1, Msg2, Msg3, Msg4]", fillcolor="#0F172A", color="#34D399"];

        CURRENT -> BRANCH_1;
        CURRENT -> BRANCH_2;
        BRANCH_1 -> MERGED [label="Channel merge"];
        BRANCH_2 -> MERGED [label="Channel merge"];
    }
    """
    st.graphviz_chart(accum_dot, use_container_width=True)

# ==============================================================================
# SECTION 7: MESSAGE REPLACEMENT
# ==============================================================================
st.header("7. Message Replacement: ID-Based Mutations and RemoveMessage Pruning", divider="rainbow")

st.markdown("""
LangGraph supports surgical message mutation without destroying historical state integrity.
""")

col_mr1, col_mr2 = st.columns(2)
with col_mr1:
    st.markdown("#### Scenario A: In-Place Message Update")
    st.code("""
# Correct an earlier message by emitting same ID
def edit_message_node(state: MessagesState) -> dict:
    corrected_msg = AIMessage(
        id="target_ai_msg_004",  # Matches existing ID!
        content="Corrected factual figure: 18,200.00 USD"
    )
    return {"messages": [corrected_msg]}
""", language="python")

with col_mr2:
    st.markdown("#### Scenario B: Surgical Message Deletion")
    st.code("""
from langchain_core.messages import RemoveMessage

# Delete an obsolete or sensitive message from history
def purge_sensitive_info_node(state: MessagesState) -> dict:
    return {
        "messages": [RemoveMessage(id="target_ai_msg_004")]
    }
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Enterprise Support Graph with Extended MessagesState", divider="rainbow")

st.markdown("""
Here is a complete, production-grade example illustrating an extended `MessagesState`
with custom domain fields, message mutation, and `RemoveMessage` cleanup:
""")

st.code("""
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# 1. Extend MessagesState with domain-specific properties
class EnterpriseTicketState(MessagesState):
    ticket_id: str
    priority: str
    audit_notes: List[str]

# 2. Node Definitions
def ingest_and_label_node(state: EnterpriseTicketState) -> dict:
    # Update custom scalar keys and add acknowledgment message
    ack = AIMessage(
        id=f"ack_{state['ticket_id']}",
        content=f"Ticket {state['ticket_id']} logged with priority {state['priority']}."
    )
    return {
        "audit_notes": [f"Ticket registered at priority {state['priority']}"],
        "messages": [ack]
    }

def sanitize_pii_node(state: EnterpriseTicketState) -> dict:
    # Example: If any human message contained raw credit cards, replace or purge
    messages = state["messages"]
    updates = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and "card:" in msg.content.lower():
            # Replace with redacted version using same ID
            redacted = HumanMessage(
                id=msg.id,
                content="[REDACTED SENSITIVE FINANCIAL PII]"
            )
            updates.append(redacted)
    return {"messages": updates}

# 3. Assemble Graph
builder = StateGraph(EnterpriseTicketState)
builder.add_node("label", ingest_and_label_node)
builder.add_node("sanitize", sanitize_pii_node)

builder.add_edge(START, "label")
builder.add_edge("label", "sanitize")
builder.add_edge("sanitize", END)

ticket_workflow = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: State Reducer & Message Mutation Laboratory", divider="rainbow")

st.markdown("""
Test how the `add_messages` reducer processes incoming node deltas.
Simulate **Appending** new messages, **Replacing** existing messages in-place via identical IDs,
and **Pruning** messages using `RemoveMessage`.
""")

if "reducer_engine" not in st.session_state:
    st.session_state.reducer_engine = ReducerSimulatorEngine()

eng: ReducerSimulatorEngine = st.session_state.reducer_engine

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Emit State Delta Operation")
    op_type = st.selectbox(
        "Select Delta Operation Type:",
        ["APPEND", "REPLACE", "REMOVE"]
    )
    
    if op_type == "APPEND":
        target_id = st.text_input("New Message ID:", value=f"msg_{len(eng.state_messages)+1:03d}")
        target_role = st.selectbox("Role:", ["human", "ai", "system"])
        target_content = st.text_area("Message Content:", value="Can you confirm delivery status?")
    elif op_type == "REPLACE":
        target_id = st.selectbox("Select Existing Message ID to Replace In-Place:", [m["id"] for m in eng.state_messages])
        target_role = st.selectbox("Role:", ["ai", "human", "system"])
        target_content = st.text_area("Replacement Content:", value="Order 4492 has shipped via FedEx Priority!")
    else:
        target_id = st.selectbox("Select Message ID to Purge via RemoveMessage:", [m["id"] for m in eng.state_messages])
        target_role = "sentinel"
        target_content = "[RemoveMessage Sentinel Payload]"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        apply_btn = st.button("🚀 Apply State Delta", type="primary")
    with col_btn2:
        reset_btn = st.button("🔄 Reset Initial State")

    if reset_btn:
        eng.reset_state()
        st.rerun()

with wb_c2:
    st.subheader("📦 Live MessagesState Channel View")
    if apply_btn:
        res = eng.apply_delta(op_type, target_id, target_content, target_role)
        st.success(res["action_taken"])

    st.markdown(f"**Total Messages in Channel:** `{len(eng.state_messages)}`")
    for idx, msg in enumerate(eng.state_messages):
        badge = "badge-replace" if op_type == "REPLACE" and msg["id"] == target_id else "badge-state"
        st.markdown(f"**[{idx+1}]** <span class='{badge}'>{msg['role'].upper()}</span> `ID: {msg['id']}`", unsafe_allow_html=True)
        st.text(msg["content"])
        st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the underlying implementation of MessagesState in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    `MessagesState` is a standard `TypedDict` containing a single key:
    `messages: Annotated[Sequence[BaseMessage], add_messages]`.
    It couples the `Sequence[BaseMessage]` type annotation with the `add_messages` reducer function,
    enabling automated de-duplication, list append, and ID-based in-place updates.
    """)

with st.expander("Q2: Why should you subclass MessagesState instead of creating a fresh TypedDict with a raw list?", expanded=False):
    st.markdown("""
    **Answer**:
    If you create a custom state with `messages: List[BaseMessage]` without specifying `add_messages`,
    LangGraph treats the field as a default scalar key. Any node returning `{"messages": [response]}` will
    **overwrite the entire message history**, deleting all earlier human and system turns!
    Subclassing `MessagesState` guarantees that the reducer is inherited automatically.
    """)

with st.expander("Q3: How does add_messages reconcile parallel updates during fan-out branches?", expanded=False):
    st.markdown("""
    **Answer**:
    When two nodes execute in parallel during the same superstep and both emit `{"messages": [...]}`,
    the `add_messages` reducer merges the deltas deterministically at the superstep synchronization barrier.
    If each branch emits unique IDs, both are appended. If both emit identical IDs, one overwrites the other based
    on node completion order.
    """)

with st.expander("Q4: Can RemoveMessage be used to clear the entire message history at once?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes. You can return `{"messages": [RemoveMessage(id=m.id) for m in state["messages"]]}`.
    The `add_messages` reducer will process each deletion in sequence, resetting the channel to an empty list `[]`.
    """)

with st.expander("Q5: What happens if an extended state key has no reducer defined and multiple parallel nodes write to it?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph will raise an `InvalidUpdateError` at runtime during graph execution because multiple parallel branches
    attempted to write conflicting updates to a channel without an aggregating reducer function (e.g. `operator.add`).
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Accidental Message History Overwrites")
    st.markdown("""
    *Mechanism*: Defining `class AgentState(TypedDict)` and listing `messages: List[BaseMessage]` without the `add_messages` reducer.  
    *Consequence*: Nodes returning `{"messages": [response]}` wipe out all previous conversation turns.  
    *Fix*: Always subclass `MessagesState` or explicitly annotate with `add_messages`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Non-Deterministic Parallel Writes to Unreduced State Keys")
    st.markdown("""
    *Mechanism*: Having two parallel branches write to an unreduced custom field (e.g., `status: str`).  
    *Consequence*: The graph crashes with an unresolvable channel collision exception.  
    *Fix*: Use custom reducers or structure the graph so parallel nodes write to distinct state keys.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Pruning Message Context")
    st.markdown("""
    *Mechanism*: Aggressively removing earlier messages via `RemoveMessage` without leaving a summary.  
    *Consequence*: The agent loses context about earlier user constraints, causing amnesia.  
    *Fix*: Combine `RemoveMessage` with an updated `SystemMessage` summarizing pruned context.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Mutating Custom Dict State Objects In-Place")
    st.markdown("""
    *Mechanism*: Directly doing `state['session_metadata']['token'] = 'new_val'` inside a node function.  
    *Consequence*: Bypasses checkpointer snapshot isolation and breaks checkpoint rollback capabilities.  
    *Fix*: Always return new or updated dictionary keys in the returned state delta dictionary.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**MessagesState**\\nThe built-in LangGraph TypedDict providing pre-configured add_messages channel reducer behavior.")
with col_fc2:
    st.info("**Subclassing MessagesState**\\nThe recommended method for adding custom domain keys while preserving message accumulation.")
with col_fc3:
    st.info("**add_messages Reducer**\\nChannel function that appends new IDs and replaces matching IDs in-place.")
with col_fc4:
    st.info("**RemoveMessage**\\nA sentinel message object that commands the add_messages reducer to delete an item by ID.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the exact TypedDict definition of MessagesState in LangGraph.", value=True)
c2 = st.checkbox("I know how to extend MessagesState to add custom domain attributes.", value=True)
c3 = st.checkbox("I understand how add_messages handles ID collisions for in-place message updates.", value=True)
c4 = st.checkbox("I can use RemoveMessage to delete specific historical messages by ID.", value=True)
c5 = st.checkbox("I understand how parallel branches merge their message deltas into the state channel.", value=True)
c6 = st.checkbox("I know how to prevent accidental history wipes by ensuring reducers are present.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 11.2 'MessagesState' with full architectural and practical comprehension!")
