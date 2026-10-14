import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="11.3 Conversational Workflows - LangGraph Curriculum",
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
    .badge-user {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-bot {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-summary {
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
class ContextManagementEngine:
    """
    Simulates multi-turn conversations, context token estimation,
    sliding window message trimming, and rolling summarization compaction.
    """
    def __init__(self):
        self.max_tokens_threshold = 400
        self.history: List[Dict[str, str]] = []
        self.summary_text: str = ""

    def add_turn(self, user_text: str, apply_strategy: str) -> Dict[str, Any]:
        # Add user turn
        self.history.append({"role": "user", "content": user_text, "tokens": len(user_text.split()) * 2})
        
        # Simulated assistant response
        bot_reply = f"Acknowledged: '{user_text[:28]}...'. Processing request against current state context."
        self.history.append({"role": "assistant", "content": bot_reply, "tokens": len(bot_reply.split()) * 2})
        
        total_tokens_before = sum(t["tokens"] for t in self.history)
        action_log = []

        if apply_strategy == "Sliding Window Trimming" and total_tokens_before > self.max_tokens_threshold:
            # Keep latest 4 messages
            pruned_count = max(0, len(self.history) - 4)
            self.history = self.history[-4:]
            action_log.append(f"Trimmed {pruned_count} earlier messages to keep token window under {self.max_tokens_threshold}.")
        elif apply_strategy == "Rolling Summarization" and total_tokens_before > self.max_tokens_threshold:
            # Compact older messages into summary
            older_msgs = self.history[:-2]
            compacted_summary = "User inquired about system operations and credentials. Assistant provided verification guidance."
            self.summary_text = compacted_summary
            self.history = self.history[-2:]
            action_log.append("Synthesized rolling summary of older conversation history; retained latest 2 active turns.")

        total_tokens_after = sum(t["tokens"] for t in self.history) + (len(self.summary_text.split()) * 2 if self.summary_text else 0)

        return {
            "total_turns": len(self.history) // 2,
            "raw_messages_count": len(self.history),
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "summary": self.summary_text,
            "action_log": action_log,
            "active_messages": self.history
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Conversational State Machine with Automated Context Optimization", divider="rainbow")

st.markdown("""
A production-ready conversational workflow in LangGraph consists of more than just a single prompt loop.
It incorporates a **State Machine Lifecycle** that actively balances:
1. Multi-turn thread continuity using checkpointer backends.
2. Token budget enforcement using **Message Trimming** (`trim_messages`).
3. Knowledge preservation using **Rolling Summarization Nodes**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    convo_blueprint_dot = """
    digraph ConversationalWorkflow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        USER_MSG [label="New User Message Ingestion\\n(thread_id='session_9281')", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_context_mgmt {
            label = "1. Context Management Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            EVAL_BUDGET [label="Token Budget Monitor: should_compact_context()\\nInspects message length vs token ceiling", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
            TRIM_NODE [label="Sliding Window Pruning: trim_messages()\\nKeeps last N tokens or turns intact", fillcolor="#431407", color="#FB923C"];
            SUMMARIZE_NODE [label="Rolling Summarizer Node\\nSynthesizes earlier turns into SystemMessage summary", fillcolor="#2E1065", color="#C084FC"];
            PASSTHROUGH [label="Passthrough (Tokens within Budget)", fillcolor="#0F172A", color="#64748B"];

            EVAL_BUDGET -> TRIM_NODE [label="Option A: Fast Window"];
            EVAL_BUDGET -> SUMMARIZE_NODE [label="Option B: Semantic Summary"];
            EVAL_BUDGET -> PASSTHROUGH [label="Under threshold"];
        }

        subgraph cluster_agent_core {
            label = "2. LLM Reasoning & Response";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            AGENT_NODE [label="Chat Model Node (call_model)\\nInjects (Summary + Window + New Turn)", fillcolor="#064E3B", color="#34D399"];
            TOOL_NODE [label="ToolNode (Optional)\\nExecutes domain tools if requested", fillcolor="#1E1E38", color="#8B5CF6"];
            AGENT_NODE -> TOOL_NODE [label="tool_calls present"];
            TOOL_NODE -> AGENT_NODE [label="Returns ToolMessage"];
        }

        subgraph cluster_persistence {
            label = "3. Checkpointer Persistence";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            THREAD_STORE [label="Thread Checkpointer (SQLite / Postgres)\\nPersists latest snapshot with thread_id index", fillcolor="#0F172A", color="#F59E0B"];
        }

        USER_MSG -> EVAL_BUDGET [label="START"];
        TRIM_NODE -> AGENT_NODE;
        SUMMARIZE_NODE -> AGENT_NODE;
        PASSTHROUGH -> AGENT_NODE;
        AGENT_NODE -> THREAD_STORE [label="END / Checkpoint committed"];
    }
    """
    st.graphviz_chart(convo_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Rule:</b> A conversation without context management is a time bomb.
As conversations exceed 20 to 30 turns, raw context bloat leads to quadratic cost increases and
context window overflow errors. Production graphs must either prune messages sliding-window style or
compact them with rolling summaries.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: CHATBOT GRAPH
# ==============================================================================
st.header("2. Chatbot Graph: Building Multi-Turn Stateful Agent Loops", divider="rainbow")

st.markdown("""
A chatbot graph connects user inputs to an LLM node and optionally tools, compiled with a checkpointer
so the graph remembers the conversation across multiple turns.
""")

st.code("""
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 1. Instantiate model and checkpointer
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
checkpointer = MemorySaver()

# 2. Define node
def chatbot_node(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. Assemble graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 4. Compile with checkpointer
chat_app = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 3: MULTI-TURN CONVERSATION
# ==============================================================================
st.header("3. Multi-Turn Conversation: Thread Persistence and Turn-Taking Protocols", divider="rainbow")

st.markdown("""
In LangGraph, multi-turn conversations are partitioned by `thread_id`.
Passing the same `thread_id` in the `config` dictionary causes the checkpointer to load
all previous messages before executing the new turn.
""")

st.code("""
config = {"configurable": {"thread_id": "client_session_104"}}

# Turn 1
input_1 = {"messages": [HumanMessage(content="Hi, my name is Alex.")]}
out_1 = chat_app.invoke(input_1, config=config)

# Turn 2 (State is automatically reloaded from checkpointer!)
input_2 = {"messages": [HumanMessage(content="What is my name?")]}
out_2 = chat_app.invoke(input_2, config=config)

print(out_2["messages"][-1].content)
# Output: "Your name is Alex!"
""", language="python")

# ==============================================================================
# SECTION 4: CONVERSATION HISTORY
# ==============================================================================
st.header("4. Conversation History: State Channel Longevity and Checkpoint Queries", divider="rainbow")

st.markdown("""
You can inspect, query, or audit conversation history at any point in time using `get_state()`:
""")

st.code("""
# Inspect the current state snapshot of a conversation thread
snapshot = chat_app.get_state(config)

print("Thread Current Step:", snapshot.metadata["step"])
print("Total Messages in History:", len(snapshot.values["messages"]))

# Iterate and inspect historical turns
for msg in snapshot.values["messages"]:
    print(f"[{msg.type.upper()}]: {msg.content}")
""", language="python")

# ==============================================================================
# SECTION 5: CONTEXT MANAGEMENT
# ==============================================================================
st.header("5. Context Management: Guarding Token Windows and Latency Ceilings", divider="rainbow")

st.markdown("""
Enterprise conversational agents encounter two primary failure modes if context is left unmanaged:
1. **Context Window Overflow**: The combined message history exceeds the maximum tokens supported by the LLM.
2. **Degraded Instruction Following ('Needle in a Haystack' Problem)**: The LLM gets confused by old, irrelevant instructions buried deep in 40 turns of history.
""")

context_df = pd.DataFrame([
    {
        "Management Strategy": "Unmanaged Append (Default)",
        "Token Growth": "Linear / Quadratic with turns",
        "Cost Impact": "High (pays for all turns repeatedly)",
        "Context Retention": "100 percent verbatim",
        "Best Used For": "Short sessions (< 8 turns)"
    },
    {
        "Management Strategy": "Sliding Window Trimming",
        "Token Growth": "Bounded to strict maximum",
        "Cost Impact": "Low and constant",
        "Context Retention": "Recent turns only; drops early turns",
        "Best Used For": "Transactional workflows, customer help desks"
    },
    {
        "Management Strategy": "Rolling Summarization",
        "Token Growth": "Logarithmic / bounded",
        "Cost Impact": "Medium (incurs periodic summary LLM call)",
        "Context Retention": "Semantic essence preserved permanently",
        "Best Used For": "Long-lived advisory, therapy, research agents"
    }
])
st.table(context_df)

# ==============================================================================
# SECTION 6: MESSAGE TRIMMING
# ==============================================================================
st.header("6. Message Trimming: Implementing trim_messages Sliding Windows", divider="rainbow")

st.markdown("""
LangChain provides `trim_messages()`, a utility to slice message sequences
to a strict token budget while preserving system instructions and tool-calling invariants.
""")

st.code("""
from langchain_core.messages import trim_messages

# Slices messages to keep at most 2000 tokens while preserving SystemMessage
trimmed = trim_messages(
    state["messages"],
    max_tokens=2000,
    strategy="last",
    token_counter=model,
    include_system=True,  # SystemMessage at index 0 is NEVER dropped!
    allow_partial=False,
    start_on="human"      # Ensures valid alternating dialogue starting on a Human turn
)

response = model.invoke(trimmed)
""", language="python")

# ==============================================================================
# SECTION 7: MESSAGE SUMMARIZATION
# ==============================================================================
st.header("7. Message Summarization: Rolling Summaries and Context Compaction Nodes", divider="rainbow")

st.markdown("""
Rather than discarding older messages, a **Summarization Node** periodically condenses
the older conversation into an executive summary stored in a custom state field or head `SystemMessage`.
""")

st.code("""
from langchain_core.messages import RemoveMessage, SystemMessage

def summarize_conversation_node(state: MessagesState) -> dict:
    messages = state["messages"]
    
    # If messages exceed 10 turns, summarize the oldest 6 turns
    if len(messages) > 10:
        to_summarize = messages[:-4]
        summary_prompt = f"Summarize the key facts from this conversation excerpt:\\n{to_summarize}"
        summary_response = model.invoke(summary_prompt)
        
        # Prune summarized messages from state
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize]
        new_system_msg = SystemMessage(content=f"Summary of previous discussion: {summary_response.content}")
        
        return {
            "messages": delete_messages + [new_system_msg]
        }
    return {"messages": []}
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Self-Trimming & Summarizing Chatbot Graph", divider="rainbow")

st.markdown("""
Below is an end-to-end production implementation of an enterprise chatbot featuring
thread checkpointer memory, conditional context evaluation, and automated rolling summarization:
""")

st.code("""
from typing import Annotated, TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class ResilientChatState(MessagesState):
    summary: str

# 2. Model & Checkpointer
chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)
checkpointer = MemorySaver()

# 3. Nodes
def conversation_node(state: ResilientChatState) -> dict:
    # Inject summary if present
    messages = state["messages"]
    if state.get("summary"):
        messages = [SystemMessage(content=f"Context Summary: {state['summary']}")] + messages
    response = chat_model.invoke(messages)
    return {"messages": [response]}

def should_summarize(state: ResilientChatState) -> Literal["summarize", "end"]:
    if len(state["messages"]) > 8:
        return "summarize"
    return "end"

def summarize_node(state: ResilientChatState) -> dict:
    # Summarize all messages except latest 2
    messages_to_compact = state["messages"][:-2]
    summary_prompt = f"Existing Summary: {state.get('summary', 'None')}\\nCompact these turns:\\n{messages_to_compact}"
    new_summary = chat_model.invoke(summary_prompt).content
    
    # Delete compacted messages
    prune_deltas = [RemoveMessage(id=m.id) for m in messages_to_compact]
    return {
        "summary": new_summary,
        "messages": prune_deltas
    }

# 4. Graph Construction
builder = StateGraph(ResilientChatState)
builder.add_node("conversation", conversation_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "conversation")
builder.add_conditional_edges("conversation", should_summarize, {"summarize": "summarize", "end": END})
builder.add_edge("summarize", END)

chatbot_pipeline = builder.compile(checkpointer=checkpointer)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Multi-Turn Chatbot with Dynamic Context Management", divider="rainbow")

st.markdown("""
Engage in a live simulated multi-turn conversation and observe how different context management strategies
control token consumption, prune history, and generate rolling summaries in real time.
""")

if "convo_engine" not in st.session_state:
    st.session_state.convo_engine = ContextManagementEngine()

c_eng: ContextManagementEngine = st.session_state.convo_engine

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("💬 Send Multi-Turn Message")
    strategy = st.radio(
        "Select Context Management Strategy:",
        ["Sliding Window Trimming", "Rolling Summarization", "Unmanaged Append"]
    )
    user_input = st.text_input(
        "Enter User Message:",
        value="What are the port requirements for deploying Redis in cluster mode?"
    )
    
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        send_btn = st.button("🚀 Send Message Turn", type="primary")
    with col_sub2:
        clear_btn = st.button("🧹 Clear Conversation")

    if clear_btn:
        c_eng.history = []
        c_eng.summary_text = ""
        st.rerun()

with wb_c2:
    st.subheader("📊 Conversation Telemetry & Context State")
    if send_btn and user_input.strip():
        report = c_eng.add_turn(user_input, strategy)
        
        m_t1, m_t2, m_t3 = st.columns(3)
        with m_t1:
            st.metric("Total Turns", report["total_turns"])
        with m_t2:
            st.metric("Raw Messages", report["raw_messages_count"])
        with m_t3:
            st.metric("Estimated Tokens", report["tokens_after"])

        if report["action_log"]:
            for action in report["action_log"]:
                st.warning(f"⚡ {action}")

        if report["summary"]:
            st.markdown("#### 📜 Rolling Context Summary")
            st.info(report["summary"])

    st.markdown("#### Active Conversation Turns")
    if c_eng.history:
        for idx, item in enumerate(c_eng.history):
            badge = "badge-user" if item["role"] == "user" else "badge-bot"
            st.markdown(f"**[{idx+1}]** <span class='{badge}'>{item['role'].upper()}</span> `({item['tokens']} tokens)`", unsafe_allow_html=True)
            st.text(item["content"])
    else:
        st.info("No active messages in conversation thread. Enter a prompt on the left to start.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: How does LangGraph maintain conversation history across independent HTTP requests?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph utilizes **Checkpointers** (such as `MemorySaver`, `SqliteSaver`, or `PostgresSaver`).
    When an HTTP request arrives, the client sends a `thread_id` inside `config={"configurable": {"thread_id": "..."}}`.
    The checkpointer loads the latest persisted state snapshot for that thread, merges incoming messages, executes the graph,
    and commits the updated state back to the database at the end of the turn.
    """)

with st.expander("Q2: Why is trim_messages preferable to raw list slicing (e.g. state['messages'][-5:])?", expanded=False):
    st.markdown("""
    **Answer**:
    Raw Python slicing can accidentally:
    1. Cut off a `SystemMessage` at index 0 containing critical safety rules.
    2. Slice away an `AIMessage` that initiated a `ToolMessage`, causing API validation failures.
    3. Start the prompt with an `AIMessage` or `ToolMessage` rather than a `HumanMessage`.
    `trim_messages()` is schema-aware: it preserves system prompts, respects tool call/response pairs, and ensures valid dialogue boundaries.
    """)

with st.expander("Q3: What are the latency trade-offs of Rolling Summarization versus Sliding Window Trimming?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Sliding Window**: Zero LLM overhead. Slicing is an instant O(1) memory operation. However, earlier context is completely forgotten.
    - **Rolling Summarization**: Preserves semantic continuity across long conversations. However, triggering summarization incurs an extra LLM call (adding 400-800ms of latency) and minor token costs.
    """)

with st.expander("Q4: Where should an executive summary be stored: in a state key or a SystemMessage?", expanded=False):
    st.markdown("""
    **Answer**:
    Best practice is to store the summary in a dedicated state field (e.g. `state["summary"] = "..."`).
    Then, the LLM invocation node dynamically constructs a `SystemMessage(content=f"Context: {state['summary']}")`
    at execution time. This keeps state channels clean and modular.
    """)

with st.expander("Q5: How can a user inspect earlier turns of a conversation that were saved in a persistent checkpointer?", expanded=False):
    st.markdown("""
    **Answer**:
    By calling `graph.get_state_history(config)`.
    This yields an iterator of historical state snapshots for that specific `thread_id`, enabling time-travel debugging,
    audit logging, and historical turn replay.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Dropping System Prompts During Manual List Slicing")
    st.markdown("""
    *Mechanism*: Using naive list slicing like `messages = state["messages"][-6:]` before invoking the model.  
    *Consequence*: The `SystemMessage` at index 0 is dropped, causing the agent to forget its persona, rules, and security bounds.  
    *Fix*: Always pass `include_system=True` to `trim_messages()` or prepend the system prompt explicitly.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Orphaned ToolMessage After History Trimming")
    st.markdown("""
    *Mechanism*: Trimming drops an `AIMessage` containing a tool call but retains the corresponding `ToolMessage`.  
    *Consequence*: Cloud model APIs reject the prompt with HTTP 400 because `ToolMessage` has no preceding parent tool call.  
    *Fix*: Use `trim_messages()` which enforces tool call atomicity.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Frequent Summarization Calls")
    st.markdown("""
    *Mechanism*: Running the summarization node on every single turn rather than setting a threshold.  
    *Consequence*: Doubles API latency and incurs unnecessary token costs on each user interaction.  
    *Fix*: Only trigger summarization when message count or token budget crosses a threshold (e.g. > 10 messages).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Missing thread_id in Production API Calls")
    st.markdown("""
    *Mechanism*: Calling `graph.invoke(input)` without specifying `thread_id` in `config`.  
    *Consequence*: The checkpointer cannot index the session, causing the graph to act as a stateless single-turn assistant.  
    *Fix*: Always pass `config={"configurable": {"thread_id": user_session_id}}`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**thread_id**\\nThe unique identifier passed in config that indexes and loads conversation state from checkpointers.")
with col_fc2:
    st.info("**trim_messages()**\\nA LangChain utility for pruning message lists while preserving system instructions and tool pairs.")
with col_fc3:
    st.info("**Rolling Summarization**\\nA pattern where older messages are synthesized into an ongoing summary and pruned from state.")
with col_fc4:
    st.info("**get_state()**\\nThe LangGraph method used to inspect the current state snapshot and message channel of a thread.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to build a multi-turn chatbot graph using MessagesState.", value=True)
c2 = st.checkbox("I understand how thread_id enables persistent multi-turn conversations.", value=True)
c3 = st.checkbox("I can use trim_messages to enforce strict token budgets without dropping system prompts.", value=True)
c4 = st.checkbox("I know how to implement a rolling summarization node to compact older history.", value=True)
c5 = st.checkbox("I understand how to query historical checkpoints using get_state() and get_state_history().", value=True)
c6 = st.checkbox("I know how to prevent orphaned ToolMessage validation errors during history trimming.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 11.3 'Conversational Workflows' with full architectural and practical comprehension!")
