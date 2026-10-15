import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="11.1 Message Concepts - LangGraph Curriculum",
    page_icon="💬",
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
    .badge-human {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-ai {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-system {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #9333EA;
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
class MessageTaxonomyEngine:
    """
    Manages and simulates message sequences across the core LangChain/LangGraph
    message hierarchy: SystemMessage, HumanMessage, AIMessage, and ToolMessage.
    """
    def __init__(self):
        self.message_types = ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"]

    def create_message(
        self,
        msg_type: str,
        content: str,
        msg_id: str,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return {
            "type": msg_type,
            "id": msg_id,
            "content": content,
            "tool_call_id": tool_call_id,
            "tool_calls": tool_calls or []
        }

    def simulate_conversation_flow(self, user_inquiry: str) -> List[Dict[str, Any]]:
        thread = [
            self.create_message(
                "SystemMessage",
                "You are an enterprise financial operations agent. Adhere to audit standards.",
                "msg_001"
            ),
            self.create_message(
                "HumanMessage",
                user_inquiry,
                "msg_002"
            )
        ]
        
        # Check if query needs a tool
        if any(w in user_inquiry.lower() for w in ["balance", "price", "stock", "rate", "audit"]):
            tool_call_id = "call_abc123"
            thread.append(self.create_message(
                "AIMessage",
                "",
                "msg_003",
                tool_calls=[{"name": "fetch_account_balance", "args": {"account_id": "ACC-9921"}, "id": tool_call_id}]
            ))
            thread.append(self.create_message(
                "ToolMessage",
                '{"account_id": "ACC-9921", "balance_usd": 142500.00, "status": "ACTIVE"}',
                "msg_004",
                tool_call_id=tool_call_id
            ))
            thread.append(self.create_message(
                "AIMessage",
                "Your account ACC-9921 balance is confirmed at 142,500.00 USD in active status.",
                "msg_005"
            ))
        else:
            thread.append(self.create_message(
                "AIMessage",
                f"I have received your request ('{user_inquiry}'). How else may I assist with your audit records?",
                "msg_003"
            ))
        return thread


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Core Message Taxonomy and Communication Topologies", divider="rainbow")

st.markdown("""
In LangGraph, conversations are modeled as **immutable message streams**.
Every interaction conforms to the standardized LangChain `BaseMessage` inheritance hierarchy:
**SystemMessage**, **HumanMessage**, **AIMessage**, and **ToolMessage**.
Understanding how these four message classes interact is the foundational prerequisite for building robust agent loops.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    message_blueprint_dot = """
    digraph MessageTaxonomyBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        BASE_MSG [label="BaseMessage (LangChain Core Abstraction)\\n- id: Optional[str]\\n- content: Union[str, List[Union[str, Dict]]]\\n- additional_kwargs: Dict[str, Any]\\n- response_metadata: Dict[str, Any]", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_subclasses {
            label = "Concrete Message Hierarchy";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            SYS_MSG [label="SystemMessage\\n- Role: 'system'\\n- Governs agent persona, rules & guardrails\\n- Position: Index 0", fillcolor="#2E1065", color="#C084FC"];
            HUMAN_MSG [label="HumanMessage\\n- Role: 'user'\\n- Ingests external user prompts & multimodal inputs", fillcolor="#1E293B", color="#38BDF8"];
            AI_MSG [label="AIMessage\\n- Role: 'assistant'\\n- Generated by LLM\\n- Emits content and/or tool_calls", fillcolor="#064E3B", color="#34D399"];
            TOOL_MSG [label="ToolMessage\\n- Role: 'tool'\\n- Tool output payload\\n- Mandatory: tool_call_id matching AIMessage", fillcolor="#431407", color="#FB923C"];
        }

        subgraph cluster_state_channel {
            label = "LangGraph Channel Flow";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            CHANNEL [label="State Channel: MessagesState\\nmessages: Annotated[List[BaseMessage], add_messages]", fillcolor="#0F172A", color="#F59E0B"];
            CHECKPOINTER [label="Thread Checkpointer (MemorySaver / Postgres)\\nPersists message history indexed by thread_id", fillcolor="#0F172A", color="#34D399"];

            CHANNEL -> CHECKPOINTER [label="Committed after superstep"];
        }

        BASE_MSG -> SYS_MSG;
        BASE_MSG -> HUMAN_MSG;
        BASE_MSG -> AI_MSG;
        BASE_MSG -> TOOL_MSG;

        SYS_MSG -> CHANNEL [label="Prepend frame"];
        HUMAN_MSG -> CHANNEL [label="User input"];
        AI_MSG -> CHANNEL [label="Model generation"];
        TOOL_MSG -> CHANNEL [label="Tool result"];
    }
    """
    st.graphviz_chart(message_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Rule:</b> Every message in a LangGraph graph possesses an optional unique <code>id</code>.
When new messages are added, the channel reducer uses this ID to detect whether to append a new entry
or perform an in-place replacement (essential for editing history, human intervention, and streaming updates).
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: HUMAN MESSAGES
# ==============================================================================
st.header("2. Human Messages: User Intent, Prompt Wrapping, and Multimodal Payloads", divider="rainbow")

st.markdown("""
A `HumanMessage` represents input from a human user or an upstream client application.
It can encapsulate raw textual instructions, structured metadata, or multimodal payloads (images, audio, PDFs).
""")

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.markdown("#### Textual Ingestion")
    st.code("""
from langchain_core.messages import HumanMessage

# Standard text message
user_msg = HumanMessage(
    content="What was our EBITDA margin for Q3 2024?",
    name="auditor_jane",
    id="usr_001"
)
""", language="python")

with col_h2:
    st.markdown("#### Multimodal Document Ingestion")
    st.code("""
# High-resolution invoice inspection
multimodal_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Extract all line-item totals from this receipt:"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/receipt.jpg"}
        }
    ],
    id="usr_002"
)
""", language="python")

# ==============================================================================
# SECTION 3: AI MESSAGES
# ==============================================================================
st.header("3. AI Messages: Model Generation, Reasoning Thoughts, and Tool Calls", divider="rainbow")

st.markdown("""
An `AIMessage` is returned when a chat model executes. It encapsulates:
1. **`content`**: Natural language response generated for the human.
2. **`tool_calls`**: Structured tool invocation requests intended for execution nodes.
3. **`usage_metadata`**: Exact token accounting (`input_tokens`, `output_tokens`, `total_tokens`).
""")

st.code("""
from langchain_core.messages import AIMessage

# Scenario A: Direct text response
direct_reply = AIMessage(
    content="The EBITDA margin for Q3 2024 was 28.4 percent.",
    id="ai_001"
)

# Scenario B: Tool invocation response (content is often empty)
tool_calling_reply = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "query_financial_database",
            "args": {"metric": "EBITDA", "quarter": "Q3", "year": 2024},
            "id": "call_982bf71a"
        }
    ],
    id="ai_002"
)
""", language="python")

# ==============================================================================
# SECTION 4: SYSTEM MESSAGES
# ==============================================================================
st.header("4. System Messages: Agent Steering, System Directives, and Behavioral Guardrails", divider="rainbow")

st.markdown("""
A `SystemMessage` instructs the LLM on how to behave, what tone to adopt, and which constraints to honor.
In LangGraph, system prompts are positioned at **index 0** of the message history.
""")

sys_table = pd.DataFrame([
    {
        "Component": "Persona & Role",
        "Purpose": "Establishes institutional domain authority and expected perspective.",
        "Example Snippet": "'You are a FINRA-certified compliance compliance examiner.'"
    },
    {
        "Component": "Operational Bounds",
        "Purpose": "Prevents unauthorized tool execution and out-of-scope speculation.",
        "Example Snippet": "'Never execute wire transfers exceeding 50,000 USD without supervisor confirmation.'"
    },
    {
        "Component": "Formatting Mandates",
        "Purpose": "Forces the model to adhere to deterministic output templates.",
        "Example Snippet": "'Format all financial figures in ISO standard with currency symbols.'"
    }
])
st.table(sys_table)

# ==============================================================================
# SECTION 5: TOOL MESSAGES
# ==============================================================================
st.header("5. Tool Messages: Execution Output Ingestion and Tool Call ID Linkage", divider="rainbow")

st.markdown("""
A `ToolMessage` carries the output of an executed tool back into the conversation thread.
**Mandatory Rule:** Every `ToolMessage` must include a `tool_call_id` that **exactly matches**
the `id` emitted by the upstream `AIMessage.tool_calls`. Omitting or mismatching this ID causes
immediate API exceptions from OpenAI and Anthropic.
""")

with st.expander("🔍 View Tool Call Linkage Protocol", expanded=True):
    tool_link_dot = """
    digraph ToolLinkage {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        AI_NODE [label="LLM Node Emits AIMessage\\ntool_calls = [{'name': 'calc', 'id': 'call_xyz789', 'args': {...}}]", fillcolor="#064E3B", color="#34D399"];
        TOOL_NODE [label="ToolNode Executes Function\\nresult = {'tax_rate': 0.15}", fillcolor="#1E293B", color="#F59E0B"];
        TOOL_MSG [label="ToolNode Returns ToolMessage\\ncontent = '{tax_rate: 0.15}'\\ntool_call_id = 'call_xyz789'", fillcolor="#431407", color="#FB923C"];
        LLM_REASON [label="LLM Synthesizes Final Answer\\nSees AIMessage followed by matching ToolMessage", fillcolor="#0F172A", color="#38BDF8"];

        AI_NODE -> TOOL_NODE [label="Passes tool_calls"];
        TOOL_NODE -> TOOL_MSG [label="Wraps raw output"];
        TOOL_MSG -> LLM_REASON [label="Matches call_xyz789"];
    }
    """
    st.graphviz_chart(tool_link_dot, use_container_width=True)

st.code("""
from langchain_core.messages import ToolMessage

# Tool execution result
tool_output = ToolMessage(
    content='{"status": "CONFIRMED", "tx_hash": "0x4e8a..."}',
    name="transfer_funds",
    tool_call_id="call_982bf71a",  # MUST match AIMessage tool_calls ID!
    id="tool_001"
)
""", language="python")

# ==============================================================================
# SECTION 6: MESSAGE HISTORY
# ==============================================================================
st.header("6. Message History: Ordered Conversation Sequences and Chronological Threads", divider="rainbow")

st.markdown("""
A conversation thread in LangGraph is an ordered sequence of messages that tells a coherent story:
1. System context defines the universe of rules.
2. User provides prompt or query.
3. Assistant contemplates actions and triggers tool calls.
4. Tool returns observational facts.
5. Assistant generates final answer.
""")

st.code("""
# Canonical multi-turn thread structure
conversation_thread = [
    SystemMessage(content="You are a legal contract reviewer."),
    HumanMessage(content="Is clause 4.2 in compliance with GDPR?"),
    AIMessage(content="", tool_calls=[{"name": "fetch_gdpr_article", "args": {"article": 17}, "id": "call_1"}]),
    ToolMessage(content="Article 17: Right to Erasure...", tool_call_id="call_1"),
    AIMessage(content="Clause 4.2 complies with GDPR Article 17 regarding data deletion obligations.")
]
""", language="python")

# ==============================================================================
# SECTION 7: MESSAGE STATE
# ==============================================================================
st.header("7. Message State: Graph Channels, Serializability, and Persistence Protocols", divider="rainbow")

st.markdown("""
In LangGraph, message state is declared via `MessagesState` or custom schemas wrapping
`Annotated[List[BaseMessage], add_messages]`.
Because state is persisted to checkpointers (Postgres, SQLite, Redis), all messages must be
**JSON-serializable** via LangChain's internal serializer.
""")

col_ms1, col_ms2 = st.columns(2)
with col_ms1:
    st.markdown("#### Message State Serialization Lifecycle")
    st.markdown("""
    1. **In-Memory**: Standard Python objects (`HumanMessage`, `AIMessage`).
    2. **Checkpointer Commit**: Serialized into JSON blobs `{type: "human", data: {content: "...", id: "..."}}`.
    3. **State Reload**: De-serialized back into first-class message class instances upon graph resumption.
    """)

with col_ms2:
    st.markdown("#### Channel Reducer Invariants")
    st.markdown("""
    - **Monotonic Growth**: Without trimming, message channels grow monotonically per turn.
    - **De-duplication**: Repeated emissions of identical message IDs update in-place rather than duplicating.
    - **Pruning**: Emitting `RemoveMessage(id="msg_old")` purges target messages from memory.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Chat Thread with Full Message Taxonomy", divider="rainbow")

st.markdown("""
Below is an end-to-end executable implementation showing how all four core message types
are constructed, updated, and verified inside a complete LangGraph agent:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages

# 1. State Definition
class ProductionChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str

# 2. Model Configuration
model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 3. Node Definitions
def system_framing_node(state: ProductionChatState) -> dict:
    \"\"\"Prepend enterprise system instructions if not already present.\"\"\"
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            content="You are an enterprise support intelligence node. Ground all replies in factual verification."
        )
        return {"messages": [system_msg]}
    return {"messages": []}

def model_call_node(state: ProductionChatState) -> dict:
    \"\"\"Invoke LLM with complete conversation context.\"\"\"
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 4. Graph Construction
builder = StateGraph(ProductionChatState)
builder.add_node("framing", system_framing_node)
builder.add_node("llm", model_call_node)

builder.add_edge(START, "framing")
builder.add_edge("framing", "llm")
builder.add_edge("llm", END)

chat_graph = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Message Sequence Builder and Inspector", divider="rainbow")

st.markdown("""
Experiment with assembling heterogeneous conversation threads. Test how user queries trigger
tool call sequences and inspect the resulting message taxonomy in real time.
""")

engine = MessageTaxonomyEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Thread Parameters")
    inquiry_choice = st.selectbox(
        "Select User Inquiry Type:",
        [
            "What is the current balance of account ACC-9921?",
            "Can you explain the audit retention policy for tax records?",
            "Check current exchange rate for EUR to USD conversion.",
            "Summarize our latest security compliance guidelines."
        ]
    )
    custom_text = st.text_input("Or enter custom query:", value="")
    final_query = custom_text if custom_text.strip() else inquiry_choice
    
    run_sim = st.button("🚀 Generate Thread Sequence", type="primary")

with wb_col2:
    st.subheader("📋 Message Sequence Inspector")
    if run_sim:
        thread_messages = engine.simulate_conversation_flow(final_query)
        st.success(f"Generated {len(thread_messages)} messages in conversation sequence!")
        
        for idx, m in enumerate(thread_messages):
            role_type = m["type"]
            badge_class = "badge-system" if role_type == "SystemMessage" else (
                "badge-human" if role_type == "HumanMessage" else (
                    "badge-ai" if role_type == "AIMessage" else "badge-tool"
                )
            )
            
            st.markdown(f"**Step {idx+1}**: <span class='{badge_class}'>{role_type}</span> `[ID: {m['id']}]`", unsafe_allow_html=True)
            if m["content"]:
                st.text(m["content"])
            if m["tool_calls"]:
                st.info(f"Tool Invocations Requested: {m['tool_calls']}")
            if m["tool_call_id"]:
                st.warning(f"Linked Tool Call ID: {m['tool_call_id']}")
            st.divider()
    else:
        st.info("Select a query and click 'Generate Thread Sequence' to view the message sequence and linkage.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the fundamental difference between AIMessage.content and AIMessage.tool_calls?", expanded=False):
    st.markdown("""
    **Answer**:
    - `content`: Represents human-readable natural language output generated by the LLM.
    - `tool_calls`: Contains structured invocations intended for programmatic execution by `ToolNode`. When frontier models invoke tools, `content` is frequently empty (`""`), meaning downstream logging systems must inspect `tool_calls` rather than relying solely on `content`.
    """)

with st.expander("Q2: Why must ToolMessage always contain a tool_call_id?", expanded=False):
    st.markdown("""
    **Answer**:
    In parallel tool-calling workflows, an LLM might emit 3 separate tool calls in a single turn.
    The `tool_call_id` provides the deterministic foreign key linking each `ToolMessage` back to its exact
    counterpart in `AIMessage.tool_calls`. Without this ID, frontier APIs (OpenAI, Anthropic) reject the turn with HTTP 400 Bad Request.
    """)

with st.expander("Q3: How does LangGraph handle message ID collisions during channel merging?", expanded=False):
    st.markdown("""
    **Answer**:
    The `add_messages` reducer treats message `id` as a primary key:
    1. If the incoming message ID matches an existing message, it **replaces** the existing message in-place.
    2. If the ID is new, it **appends** the message to the end of the sequence.
    This enables seamless human edits and message rewriting.
    """)

with st.expander("Q4: Can a SystemMessage be injected in the middle of a multi-turn conversation thread?", expanded=False):
    st.markdown("""
    **Answer**:
    While technically possible in some API schemas, injecting `SystemMessage` midway through a conversation thread is an anti-pattern.
    It invalidates prompt KV cache prefixes and degrades instruction-following adherence.
    Dynamic steering should instead be injected via structured tool results or framing instructions in `HumanMessage`.
    """)

with st.expander("Q5: What is RemoveMessage and how does it interact with message state?", expanded=False):
    st.markdown("""
    **Answer**:
    `RemoveMessage(id="target_id")` is a sentinel message class in LangChain/LangGraph.
    When returned by a node into an `add_messages` channel, the reducer deletes the message matching `target_id`
    from the persistent state history, enabling memory pruning and context window optimization.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing tool_call_id in Custom Tool Execution Nodes")
    st.markdown("""
    *Mechanism*: Manually constructing `ToolMessage(content=result)` without assigning `tool_call_id=tool_call['id']`.  
    *Consequence*: Cloud model APIs reject subsequent conversation turns with `InvalidRequestError: Tool call ID not found`.  
    *Fix*: Always pass `tool_call_id=call['id']` when creating `ToolMessage` instances.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Mutating Message Lists In-Place Inside Nodes")
    st.markdown("""
    *Mechanism*: Executing `state['messages'].append(new_msg)` directly inside node code.  
    *Consequence*: Bypasses channel reducers, corrupts state immutability, and breaks checkpoint rollback capabilities.  
    *Fix*: Return a delta dictionary `return {'messages': [new_msg]}` to let `add_messages` manage state updates.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Unbounded Conversation History Bloat")
    st.markdown("""
    *Mechanism*: Retaining 50+ turns of raw chat history without token trimming or summarization.  
    *Consequence*: Latency increases linearly and inference costs escalate on every single turn.  
    *Fix*: Apply `trim_messages()` or periodic rolling summarization nodes.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Non-Serializable Custom Message Attributes")
    st.markdown("""
    *Mechanism*: Attaching non-serializable objects (database connections, lambda functions) into `additional_kwargs`.  
    *Consequence*: Checkpointers crash with `TypeError: Object of type ... is not JSON serializable`.  
    *Fix*: Store only JSON-compliant primitives (strings, ints, dicts, lists) inside message metadata.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**BaseMessage**\\nThe foundational LangChain class from which System, Human, AI, and Tool messages inherit.")
with col_fc2:
    st.info("**tool_call_id**\\nThe required correlation token connecting an executed ToolMessage with its parent AIMessage.")
with col_fc3:
    st.info("**RemoveMessage**\\nA sentinel class used in add_messages channels to delete historical messages by ID.")
with col_fc4:
    st.info("**MessagesState**\\nThe standard LangGraph state container encapsulating a managed message channel.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the distinct roles of SystemMessage, HumanMessage, AIMessage, and ToolMessage.", value=True)
c2 = st.checkbox("I understand why every ToolMessage must include a matching tool_call_id.", value=True)
c3 = st.checkbox("I know how the add_messages reducer handles ID matching for in-place updates.", value=True)
c4 = st.checkbox("I can use RemoveMessage to prune messages from persistent state.", value=True)
c5 = st.checkbox("I understand why node functions must return state deltas rather than mutating state lists.", value=True)
c6 = st.checkbox("I know how to structure multimodal messages with mixed text and image payloads.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 11.1 'Message Concepts' with full architectural and practical comprehension!")
