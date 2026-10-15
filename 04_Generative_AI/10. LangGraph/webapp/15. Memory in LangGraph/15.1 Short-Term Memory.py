import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="15.1 Short-Term Memory - LangGraph Curriculum",
    page_icon="🧠",
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
    .badge-mem {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-sum {
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
class ShortTermMemorySimulator:
    """
    Simulates thread-scoped working memory, context window budgets,
    and automatic summarization compaction.
    """
    def __init__(self):
        self.thread_memories: Dict[str, List[Dict[str, str]]] = {
            "thread_01": [
                {"role": "user", "content": "My account number is ACC-1092."},
                {"role": "assistant", "content": "Logged account ACC-1092."}
            ]
        }

    def add_message(self, thread_id: str, content: str, enable_compaction: bool) -> Dict[str, Any]:
        if thread_id not in self.thread_memories:
            self.thread_memories[thread_id] = []

        history = self.thread_memories[thread_id]
        history.append({"role": "user", "content": content})
        history.append({"role": "assistant", "content": f"Understood: '{content[:25]}...'. Updated working memory."})

        compacted = False
        if enable_compaction and len(history) > 4:
            # Compact older turns
            compacted = True
            history = [
                {"role": "system", "content": "Summary of earlier turns: User referenced account ACC-1092 and requested service status."},
                history[-2],
                history[-1]
            ]
            self.thread_memories[thread_id] = history

        return {
            "thread_id": thread_id,
            "total_messages": len(history),
            "compacted": compacted,
            "messages": history
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Short-Term Memory Flow and Thread Isolation", divider="rainbow")

st.markdown("""
In LangGraph, **Short-Term Memory** (working memory) is strictly **thread-scoped**.
It encapsulates the conversation history, working variables, and tool execution outputs accumulated
within a single `thread_id`.
When a conversation terminates or reaches context window limits, short-term memory is managed
via **sliding windows** (`trim_messages`) or **rolling summarization**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    stm_blueprint_dot = """
    digraph ShortTermMemoryArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        USER_MSG [label="Incoming User Turn\\nthread_id = 'session_921'", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_stm {
            label = "Short-Term Working Memory Tier (Thread-Scoped)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            LOAD_THREAD [label="Load Thread Checkpoint\\n(Restores MessagesState for session_921)", fillcolor="#2E1065", color="#C084FC"];
            BUDGET_CHECK [label="Context Window Evaluation\\nDoes thread exceed token limit?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
            COMPACT_NODE [label="Compaction / Summarization Node\\nPrunes older turns into SystemMessage summary", fillcolor="#431407", color="#FB923C"];
            AGENT_REASON [label="Agent Reasoning Node (call_model)\\nExecutes with bounded working context", fillcolor="#064E3B", color="#34D399"];

            LOAD_THREAD -> BUDGET_CHECK;
            BUDGET_CHECK -> COMPACT_NODE [label="Yes (> 2000 tokens)"];
            BUDGET_CHECK -> AGENT_REASON [label="No (within budget)"];
            COMPACT_NODE -> AGENT_REASON;
        }

        PERSIST_STORE [label="Checkpointer Database\\nCommits updated short-term memory snapshot", fillcolor="#0F172A", color="#38BDF8"];

        USER_MSG -> LOAD_THREAD;
        AGENT_REASON -> PERSIST_STORE [label="Superstep committed"];
    }
    """
    st.graphviz_chart(stm_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Memory Boundary Rule:</b> Short-term memory is ephemeral to the conversation thread.
It does not leak into other threads or users. Once a thread is completed or archived, its working state is sealed.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: THREAD-SCOPED MEMORY
# ==============================================================================
st.header("2. Thread-Scoped Memory: Scoping Context to Single Conversation Threads", divider="rainbow")

st.markdown("""
Every piece of state in `MessagesState` belongs to a specific thread.
If User A discusses their order in `thread_1` and User B inquires about billing in `thread_2`,
LangGraph guarantees that their working memories remain strictly isolated:
""")

st.code("""
# Thread 1: User A
app.invoke({"messages": [HumanMessage(content="My name is Alice")]}, {"configurable": {"thread_id": "t1"}})

# Thread 2: User B
app.invoke({"messages": [HumanMessage(content="My name is Bob")]}, {"configurable": {"thread_id": "t2"}})

# Asking Thread 1: "What is my name?" returns "Alice"
# Asking Thread 2: "What is my name?" returns "Bob"
""", language="python")

# ==============================================================================
# SECTION 3: CONVERSATION HISTORY
# ==============================================================================
st.header("3. Conversation History: Multi-Turn Dialogues and Message Retention", divider="rainbow")

st.markdown("""
Short-term memory accumulates multi-turn dialogue history through the `add_messages` reducer.
Historical turns provide the necessary context for the agent to resolve co-references ("it", "the previous order", "that invoice").
""")

# ==============================================================================
# SECTION 4: STATE-BASED MEMORY
# ==============================================================================
st.header("4. State-Based Memory: Custom TypedDict Domain State as Working Memory", divider="rainbow")

st.markdown("""
Short-term memory is not limited to chat messages. Custom fields in your state TypedDict
act as **structured working scratchpads** for scratch calculations, temporary flags, and intermediate artifacts:
""")

st.code("""
class WorkingScratchpadState(MessagesState):
    extracted_entities: Dict[str, str]  # e.g. {"account": "ACC-1092", "currency": "USD"}
    validation_status: bool
    intermediate_calculation: float
""", language="python")

# ==============================================================================
# SECTION 5: MESSAGE PERSISTENCE
# ==============================================================================
st.header("5. Message Persistence: Checkpointer Serialization and Storage Commit", divider="rainbow")

st.markdown("""
At the end of every superstep, the checkpointer serializes all short-term messages and custom state values
into binary blobs committed to disk or database storage.
""")

# ==============================================================================
# SECTION 6: CONTEXT WINDOWS
# ==============================================================================
st.header("6. Context Windows: Guarding Context Limits with Token Budgets", divider="rainbow")

st.markdown("""
LLMs have finite context windows.
If short-term memory grows unbounded, inference costs increase quadratically and the model
eventually throws a `ContextWindowExceeded` exception.
Production graphs must implement automated sliding window trimming:
""")

st.code("""
from langchain_core.messages import trim_messages

trimmed_history = trim_messages(
    state["messages"],
    max_tokens=2500,
    strategy="last",
    token_counter=model,
    include_system=True
)
""", language="python")

# ==============================================================================
# SECTION 7: MESSAGE SUMMARIZATION
# ==============================================================================
st.header("7. Message Summarization: Rolling Summaries and Context Compaction", divider="rainbow")

st.markdown("""
When working memory exceeds safe bounds, a rolling summarization node compacts earlier dialogue
into a concise summary and deletes raw messages using `RemoveMessage`:
""")

st.code("""
from langchain_core.messages import RemoveMessage, SystemMessage

def compaction_node(state: MessagesState) -> dict:
    if len(state["messages"]) > 10:
        older_turns = state["messages"][:-4]
        summary = model.invoke(f"Summarize: {older_turns}").content
        
        prune_deltas = [RemoveMessage(id=m.id) for m in older_turns]
        new_summary = SystemMessage(content=f"Summary of earlier conversation: {summary}")
        return {"messages": prune_deltas + [new_summary]}
    return {"messages": []}
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Short-Term Memory Pipeline", divider="rainbow")

st.markdown("""
Below is the complete, runnable implementation of a stateful, thread-scoped conversational agent:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class ShortTermChatState(MessagesState):
    session_topic: str

# 2. Model & Checkpointer
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
checkpointer = MemorySaver()

# 3. Node Definitions
def conversation_node(state: ShortTermChatState) -> dict:
    messages = state["messages"]
    topic = state.get("session_topic", "General")
    framed = [SystemMessage(content=f"You are a helpful assistant discussing {topic}.")] + messages
    response = model.invoke(framed)
    return {"messages": [response]}

# 4. Assemble Graph
builder = StateGraph(ShortTermChatState)
builder.add_node("agent", conversation_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Thread-Scoped Short-Term Memory Simulator", divider="rainbow")

st.markdown("""
Interact with simulated short-term memory across isolated threads.
Toggle automated context compaction to observe how older messages are pruned and summarized in real time.
""")

stm_engine = ShortTermMemorySimulator()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Memory Simulation Controls")
    active_th = st.selectbox("Select Active Thread:", ["thread_01", "thread_02"])
    new_text = st.text_input("Enter New Message:", value="Can you confirm my subscription plan?")
    enable_comp = st.toggle("Enable Rolling Compaction (> 4 messages)", value=True)
    send_msg = st.button("🚀 Send to Working Memory", type="primary")

with wb_c2:
    st.subheader("📋 Active Thread Working Memory")
    if send_msg and new_text.strip():
        report = stm_engine.add_message(active_th, new_text, enable_comp)
        if report["compacted"]:
            st.warning("⚡ Rolling compaction triggered: older turns synthesized into summary!")
        else:
            st.success("Message appended to working memory.")

    current_mem = stm_engine.thread_memories.get(active_th, [])
    st.markdown(f"**Total Messages in `{active_th}`:** `{len(current_mem)}`")

    for idx, msg in enumerate(current_mem):
        badge = "badge-sum" if msg["role"] == "system" else "badge-mem"
        st.markdown(f"**[{idx+1}]** <span class='{badge}'>{msg['role'].upper()}</span>", unsafe_allow_html=True)
        st.text(msg["content"])
        st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What defines short-term memory in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Short-term memory is the working context scoped strictly to a single `thread_id`.
    It includes the `MessagesState` channel and any custom TypedDict fields committed to checkpoints during that thread's lifecycle.
    It does not persist across separate thread IDs.
    """)

with st.expander("Q2: How do you prevent short-term memory from causing context window overflows?", expanded=False):
    st.markdown("""
    **Answer**:
    1. Using `trim_messages()` to enforce a hard token ceiling while retaining the head `SystemMessage`.
    2. Using rolling summarization nodes that condense older turns into an ongoing summary and prune raw messages with `RemoveMessage`.
    """)

with st.expander("Q3: Can short-term memory be shared between two different users?", expanded=False):
    st.markdown("""
    **Answer**:
    No. In multi-tenant systems, each user must be assigned an isolated `thread_id`.
    Sharing a `thread_id` across users would leak private conversation history and state variables.
    """)

with st.expander("Q4: Where does short-term memory reside between consecutive HTTP turns?", expanded=False):
    st.markdown("""
    **Answer**:
    In the checkpointer database (e.g. PostgreSQL `checkpoints` table).
    When an HTTP request arrives, LangGraph queries the checkpointer using `thread_id`, re-hydrates the state into memory,
    executes the turn, and writes the updated checkpoint back to the database.
    """)

with st.expander("Q5: What is the role of custom TypedDict fields in short-term memory?", expanded=False):
    st.markdown("""
    **Answer**:
    They act as a structured scratchpad. Instead of storing complex data inside conversational text messages,
    custom state keys store verified flags, extracted entity dictionaries, and numeric metrics cleanly.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Cross-User Thread ID Collisions")
    st.markdown("""
    *Mechanism*: Using static thread IDs like `"user_session"` across multiple concurrent web users.  
    *Consequence*: Users see each other's messages and private data, causing catastrophic security leaks.  
    *Fix*: Always generate unique UUIDs or user-scoped thread IDs (`f"usr_{user_id}_sess_{session_id}"`).
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Unmanaged Quadratic History Growth")
    st.markdown("""
    *Mechanism*: Allowing a multi-turn conversation to grow past 50 turns without trimming.  
    *Consequence*: Token costs increase quadratically and calls fail with context limit errors.  
    *Fix*: Implement automated `trim_messages()` or rolling summarization.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Re-Summarizing on Every Turn")
    st.markdown("""
    *Mechanism*: Calling the summarization LLM on every single turn instead of checking a threshold.  
    *Consequence*: Doubles API latency and incurs unnecessary token expenses.  
    *Fix*: Only trigger summarization when message count exceeds a threshold (e.g. > 10 messages).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Loss of Co-References During Aggressive Trimming")
    st.markdown("""
    *Mechanism*: Trimming history too aggressively (keeping only the last 1 turn).  
    *Consequence*: The agent cannot resolve pronouns ("it", "the first one") referring to previous turns.  
    *Fix*: Maintain a minimum window of 4 to 6 recent turns.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Short-Term Memory**\\nWorking state scoped to a single thread_id, managing conversation context and scratchpad fields.")
with col_fc2:
    st.info("**trim_messages()**\\nUtility pruning message lists to fit strictly within model token budgets.")
with col_fc3:
    st.info("**Thread Isolation**\\nGuarantee that state in one thread_id never leaks into other threads.")
with col_fc4:
    st.info("**Rolling Summarization**\\nCompacting older conversation turns into an ongoing summary to preserve long-term context.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the exact scope and lifecycle of short-term memory in LangGraph.", value=True)
c2 = st.checkbox("I know how thread_id guarantees privacy and isolation between different users.", value=True)
c3 = st.checkbox("I can implement token budget protection using trim_messages().", value=True)
c4 = st.checkbox("I know how to implement rolling summarization nodes with RemoveMessage.", value=True)
c5 = st.checkbox("I understand how custom TypedDict fields act as structured working memory.", value=True)
c6 = st.checkbox("I know how to prevent cross-user thread ID collisions in production APIs.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 15.1 'Short-Term Memory' with full architectural and practical comprehension!")
