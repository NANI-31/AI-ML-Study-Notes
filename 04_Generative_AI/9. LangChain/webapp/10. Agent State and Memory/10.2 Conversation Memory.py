import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="10.2 Conversation Memory - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("10.2 Conversation Memory: Working Context, Persistence, Trimming & Summarization")
st.caption("Chapter 10: Agent State and Memory | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Every LLM call is stateless by nature. When a user asks a follow-up question (*"What did I just ask you?"*), 
the model only knows the answer if the previous conversational turns are re-injected into the prompt.

However, unconstrained conversation growth causes two critical problems:
1. **Context Window Exhaustion**: Hitting hard token limits and causing API errors.
2. **Attention Degradation (The Lost-in-the-Middle Problem)**: Models lose reasoning precision when buried under dozens of obsolete tool outputs.

In this lesson, you will master the 6 pillars of conversation memory engineering:
- **Short-Term Memory**: Managing immediate session working memory.
- **Conversation History**: Structuring multi-turn dialogues with message objects.
- **Message Persistence**: Storing and restoring threads using durable checkpointers.
- **Context Management**: Enforcing token budgets to optimize latency and costs.
- **Message Trimming**: Programmatically trimming old messages using `trim_messages`.
- **Message Summarization**: Condensing long dialogue histories into running summaries.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockMemoryTrimReport:
    """Represents a simulated memory pruning operation in module scope."""
    def __init__(self, original_count: int, final_count: int, tokens_saved: int, retained_messages: List[Dict[str, str]], summary_text: Optional[str] = None):
        self.original_count = original_count
        self.final_count = final_count
        self.tokens_saved = tokens_saved
        self.retained_messages = retained_messages
        self.summary_text = summary_text

class MockConversationMemorySimulator:
    """Simulates message trimming, token budgeting, and running summarization in module scope."""
    def __init__(self):
        pass

    def apply_memory_strategy(
        self,
        strategy: str,
        max_tokens_budget: int,
        messages_pool: List[Dict[str, str]]
    ) -> MockMemoryTrimReport:
        orig_count = len(messages_pool)
        
        if strategy == "Window Trimming (Keep Last N Messages)":
            keep_count = min(3, orig_count)
            # Ensure system prompt preserved
            sys_msg = [m for m in messages_pool if m["role"] == "System"]
            non_sys = [m for m in messages_pool if m["role"] != "System"]
            retained = sys_msg + non_sys[-keep_count:]
            tokens_saved = max(0, (orig_count - len(retained)) * 45)
            
            return MockMemoryTrimReport(
                original_count=orig_count,
                final_count=len(retained),
                tokens_saved=tokens_saved,
                retained_messages=retained,
                summary_text=None
            )

        elif strategy == "Progressive Summarization (Compress Older Turns)":
            sys_msg = [m for m in messages_pool if m["role"] == "System"]
            non_sys = [m for m in messages_pool if m["role"] != "System"]
            
            summary = "Summary of previous turns: User experienced database latency on worker node-04; agent verified connection pool exhaustion."
            summary_msg = {"role": "SystemSummary", "content": summary}
            retained = sys_msg + [summary_msg] + non_sys[-2:]
            tokens_saved = max(0, (orig_count - len(retained)) * 52)

            return MockMemoryTrimReport(
                original_count=orig_count,
                final_count=len(retained),
                tokens_saved=tokens_saved,
                retained_messages=retained,
                summary_text=summary
            )

        else:  # Raw Unbounded
            return MockMemoryTrimReport(
                original_count=orig_count,
                final_count=orig_count,
                tokens_saved=0,
                retained_messages=messages_pool,
                summary_text=None
            )

# --- SECTION 1: ARCHITECTURE BLUEPRINT & MEMORY MANAGEMENT ---
st.header("1. Architecture Blueprint: The Conversation Memory Management & Pruning Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates how memory is actively governed in production. 
Before conversation history reaches the chat model, it passes through **Pruning and Summarization Middleware** 
to guarantee that the token count fits within the model's optimal attention budget.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Persistent Storage
        subgraph cluster_storage {
            label = "1. PERSISTENT STORAGE TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            DatabaseStore [label="PostgreSQL / SQLite Checkpointer\\nStores complete un-pruned history (100+ turns)", color="#38BDF8"];
        }

        // Level 2: Retrieval & Ingress
        subgraph cluster_retrieval {
            label = "2. RETRIEVAL &amp; THREAD RESTORATION";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            ThreadLoader [label="Load thread_id='session_4820'\\nUnpacks complete message list"];
            TokenAuditor [label="Token Counter (tiktoken)\\nCalculates total tokens across history"];
        }

        // Level 3: Pruning Middleware
        subgraph cluster_pruning {
            label = "3. CONTEXT MANAGEMENT &amp; PRUNING MIDDLEWARE";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            DecisionBranch [label="Is Token Count &gt; Budget (e.g. 4,000 tokens)?"];
            MessageTrimmer [label="trim_messages()\\nKeeps SystemMessage + last N turns", color="#10B981"];
            SummarizerNode [label="Auxiliary LLM Summarizer\\nCompresses old turns into SystemMessage summary", color="#F59E0B"];
        }

        // Level 4: Model Ingestion
        subgraph cluster_model {
            label = "4. OPTIMIZED PROMPT INGESTION";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            OptimizedPrompt [label="Pruned Prompt Context\\nOptimal attention density &amp; low token cost", color="#10B981"];
            FrontierModel [label="Chat Model (GPT-4o / Claude 3.5)\\nHigh reasoning fidelity without memory bloat"];
        }

        // Wiring
        DatabaseStore -> ThreadLoader;
        ThreadLoader -> TokenAuditor;
        TokenAuditor -> DecisionBranch;

        DecisionBranch -> OptimizedPrompt [label="Within Budget (No Pruning)"];
        DecisionBranch -> MessageTrimmer [label="Over Budget (Trim Strategy)"];
        DecisionBranch -> SummarizerNode [label="Over Budget (Summary Strategy)"];

        MessageTrimmer -> OptimizedPrompt;
        SummarizerNode -> OptimizedPrompt;
        
        OptimizedPrompt -> FrontierModel;
        FrontierModel -> DatabaseStore [label="Append new turns to DB (Preserve complete archive)", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Key Invariant: Archive vs Context**
Production architectures maintain a separation between the **Durable Archive** (the complete, un-pruned history stored in Postgres for compliance) 
and the **Working Context Window** (the pruned, token-budgeted prompt passed to the model for active reasoning).
""")

st.divider()

# --- SECTION 2: SHORT-TERM MEMORY ---
st.header("2. Short-Term Memory (In-Session Working Context)", divider="rainbow")

st.markdown("""
**Short-Term Memory** refers to the working memory active during a single operational session. 
It holds the immediate back-and-forth between the user and the agent, tracking:
- The user's active goal.
- Current clarifications and constraints.
- Intermediate tool calls and observations from the ongoing turn.
""")

st.divider()

# --- SECTION 3: CONVERSATION HISTORY ---
st.header("3. Conversation History (Structured Message Arrays)", divider="rainbow")

st.markdown("""
In LangChain, conversation history is represented not as raw text, but as a strongly-typed sequence 
of **BaseMessage** objects:
- `SystemMessage`: Establishes the agent's identity and operational constraints.
- `HumanMessage`: Represents user inputs.
- `AIMessage`: Contains model responses and requested `tool_calls`.
- `ToolMessage`: Contains tool execution observations, linked by `tool_call_id`.
""")

st.code('''from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# A complete structured conversation history trace:
conversation_history = [
    SystemMessage(content="You are a customer support billing assistant."),
    HumanMessage(content="Why was my card billed $49 on Oct 12?"),
    AIMessage(
        content="",
        tool_calls=[{"name": "lookup_invoice", "args": {"date": "2026-10-12"}, "id": "call_inv_101"}]
    ),
    ToolMessage(
        tool_call_id="call_inv_101",
        content="Invoice #9481: Pro Subscription Renewal ($49.00 USD)."
    ),
    AIMessage(content="The $49 charge on October 12 was for your monthly Pro Subscription renewal.")
]
''', language="python")

st.divider()

# --- SECTION 4: MESSAGE PERSISTENCE ---
st.header("4. Message Persistence (Durable Thread Checkpointers)", divider="rainbow")

st.markdown("""
In LangGraph, message persistence is handled by **Checkpointers**. 
By associating a conversation with a `thread_id`, the agent automatically persists and restores history 
across server restarts and multi-day user sessions.
""")

st.code('''from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# In-memory checkpointer for development (use PostgresSaver in production)
checkpointer = MemorySaver()

agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)

# Session A: Thread 101
config_user_a = {"configurable": {"thread_id": "thread_user_101"}}
agent.invoke({"messages": [("human", "My preferred region is us-east-1")]}, config=config_user_a)

# Session B: Thread 102 (Completely isolated memory)
config_user_b = {"configurable": {"thread_id": "thread_user_102"}}
agent.invoke({"messages": [("human", "My preferred region is eu-west-1")]}, config=config_user_b)
''', language="python")

st.divider()

# --- SECTION 5: CONTEXT MANAGEMENT ---
st.header("5. Context Management (Token Budgets & Attention Degradation)", divider="rainbow")

st.markdown("""
Modern models boast 128k to 1M token context windows, but stuffing hundreds of thousands of tokens into prompt memory 
carries severe penalties:
- **Cost Multipliers**: Paying input token fees on 50,000 tokens on *every single turn* multiplies API bills exponentially.
- **Latency Penalties**: Time-to-First-Token (TTFT) scales with prompt size.
- **Lost-in-the-Middle Phenomenon**: Studies consistently show models overlook factual details positioned in the middle of long contexts.
""")

st.divider()

# --- SECTION 6: MESSAGE TRIMMING ---
st.header("6. Message Trimming (`trim_messages` & Token-Based Pruning)", divider="rainbow")

st.markdown("""
LangChain provides the native **`trim_messages`** utility to programmatically prune message arrays 
before model invocation based on token counts or turn counts.
""")

st.subheader("Key Parameters of `trim_messages`")
st.markdown("""
- **`max_tokens`**: Maximum token ceiling allowed for the trimmed message list.
- **`strategy="last"`**: Keeps the most recent messages (sliding window).
- **`token_counter`**: Pass a model instance or token counting function (e.g. `tiktoken`).
- **`include_system=True`**: Ensures the critical `SystemMessage` is **never discarded** during trimming.
- **`start_on="human"`**: Guarantees the trimmed conversation begins with a clean user message rather than an orphaned `ToolMessage`.
""")

st.code('''from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Define automated trimming pre-processor
trimmed_history = trim_messages(
    messages=conversation_history,
    max_tokens=500,               # Enforce token ceiling
    strategy="last",             # Retain most recent turns
    token_counter=model,         # Accurate token estimation
    include_system=True,         # NEVER discard system instructions!
    start_on="human"             # Ensure first message is from human
)
''', language="python")

st.divider()

# --- SECTION 7: MESSAGE SUMMARIZATION ---
st.header("7. Message Summarization (Progressive Memory Compression)", divider="rainbow")

st.markdown("""
While trimming simply discards older turns, **Message Summarization** preserves essential long-term facts 
by compressing older dialogue turns into a concise running summary.
""")

st.code('''from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Pattern: Progressive Running Summary
def compress_history_with_summary(messages: list, summary_model) -> list:
    if len(messages) < 6:
        return messages

    old_turns = messages[:-3]  # Older turns to compress
    recent_turns = messages[-3:]  # Recent turns to preserve verbatim

    # Generate running summary of older turns
    summary_prompt = "Summarize the key facts established in this conversation: " + str(old_turns)
    summary_text = summary_model.invoke(summary_prompt).content

    return [
        SystemMessage(content=f"Summary of prior conversation: {summary_text}")
    ] + recent_turns
''', language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Memory-Managed Agent Pipeline", divider="rainbow")

st.markdown("""
The following complete production pattern illustrates an enterprise memory management pipeline 
that loads persisted thread history, calculates token consumption, applies automated message trimming, 
and invokes the model within a strict token budget.
""")

st.code('''import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    trim_messages
)

logger = logging.getLogger("EnterpriseMemoryPipeline")

class GovernedMemoryAgentPipeline:
    """Production pipeline enforcing strict token budgeting and message trimming."""

    def __init__(self, token_budget: int = 1500):
        self.token_budget = token_budget
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.system_message = SystemMessage(
            content="You are an enterprise technical support specialist. Provide concise resolutions."
        )

    def prepare_context(self, raw_history: List[BaseMessage]) -> List[BaseMessage]:
        logger.info(f"Auditing conversation history containing {len(raw_history)} messages...")
        
        # Ensure system message is included
        full_list = [self.system_message] + [m for m in raw_history if not isinstance(m, SystemMessage)]

        # Enforce automated token trimming
        trimmed: List[BaseMessage] = trim_messages(
            messages=full_list,
            max_tokens=self.token_budget,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            start_on="human"
        )

        logger.info(f"Trimmed history from {len(full_list)} to {len(trimmed)} messages.")
        return trimmed

    def invoke_with_memory(self, history: List[BaseMessage], user_input: str) -> str:
        history.append(HumanMessage(content=user_input))
        safe_context = self.prepare_context(history)
        
        response: AIMessage = self.model.invoke(safe_context)
        history.append(response)
        return response.content
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Interactive Message Trimming & Summarization Simulator", divider="rainbow")

st.markdown("""
Experiment with different memory management strategies on an extended multi-turn conversation. 
Observe how **Window Trimming** and **Progressive Summarization** reduce token usage while preserving critical context.
""")

mem_sim = MockConversationMemorySimulator()

# Mock extended conversation pool
mock_pool = [
    {"role": "System", "content": "You are a cloud infrastructure SRE assistant."},
    {"role": "User", "content": "We are seeing high memory consumption on worker-node-04."},
    {"role": "Assistant", "content": "I recommend checking top memory-consuming processes using the telemetry tool."},
    {"role": "ToolObservation", "content": "Telemetry output: PID 8492 (redis-proxy) consuming 8.4GB RAM (91%)."},
    {"role": "Assistant", "content": "The redis-proxy service is leaking memory. Have you applied the patch from ticket ENG-9102?"},
    {"role": "User", "content": "No, we have not applied ENG-9102 yet. Can we restart the daemon safely?"},
    {"role": "Assistant", "content": "Yes, restarting the daemon will drop connections briefly for 2 seconds. A maintenance window is advised."},
    {"role": "User", "content": "Understood. Please initiate the restart now."}
]

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("⚙️ Memory Management Strategy")
    strategy_choice = st.selectbox(
        "Pruning Strategy",
        ["Window Trimming (Keep Last N Messages)", "Progressive Summarization (Compress Older Turns)", "Unbounded (No Pruning)"]
    )
    budget_slider = st.slider("Maximum Token Budget Ceiling", min_value=150, max_value=800, value=300)

with col_wb2:
    st.subheader("📋 Unpruned Message Archive")
    st.caption(f"Total raw dialogue turns in archive: **{len(mock_pool)} turns** (~420 tokens)")

if st.button("✂️ Execute Memory Pruning", type="primary"):
    report = mem_sim.apply_memory_strategy(strategy_choice, budget_slider, mock_pool)
    
    st.subheader("📊 Memory Audit & Token Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("Original Messages", report.original_count)
    m2.metric("Retained Messages", report.final_count)
    m3.metric("Tokens Saved", f"{report.tokens_saved} tokens")

    if report.summary_text:
        st.info(f"**Synthesized Running Summary**: *{report.summary_text}*")

    st.write("#### 📝 Final Context Window Ingested by Model:")
    for msg in report.retained_messages:
        with st.expander(f"[{msg['role']}]", expanded=True):
            st.write(msg["content"])

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the 'Lost-in-the-Middle' phenomenon and how does memory trimming mitigate it?",
     "Research shows that language models recall information located at the very beginning (primacy effect) and very end (recency effect) of prompts with high accuracy, but frequently overlook factual details buried in the middle of long contexts. Trimming removes obsolete intermediate observations, keeping the context concise and focused on relevant working memory."),
    ("Why is include_system=True critical when using trim_messages?",
     "If include_system=False, a sliding window trimming strategy will discard the SystemMessage located at the beginning of the list once the token budget is reached. This causes the model to lose its identity, operational guardrails, and tool guidelines. include_system=True guarantees the SystemMessage is always preserved."),
    ("What does the start_on parameter in trim_messages do?",
     "When trimming from the end of a message history, the first retained message might arbitrarily be a ToolMessage or an AIMessage without context. Setting start_on='human' ensures the trimmed history always begins cleanly with a HumanMessage, maintaining valid conversation syntax across API providers."),
    ("How does LangGraph distinguish between ephemeral session state and persistent memory?",
     "Ephemeral state exists only in working memory during a single graph execution. Persistent memory is enabled by attaching a Checkpointer (such as PostgresSaver or SqliteSaver). Passing a thread_id in RunnableConfig instructs the checkpointer to load and save the conversation state to an external database."),
    ("When should you choose Message Summarization over Window Trimming?",
     "Window trimming is fast, zero-cost, and deterministic, but completely deletes historical context. Message summarization uses an auxiliary LLM call to compress older turns, preserving key factual agreements and user preferences from 20 turns ago while still keeping token counts within budget.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Orphaned ToolMessages After Naive Slicing. Slicing messages with `messages[-5:]` can retain a ToolMessage while cutting off the preceding AIMessage containing the tool_call. Provider APIs reject this with an HTTP 400 error. Always use `trim_messages()` with `start_on='human'`.")
    st.error("**Trap 2 (Breaking Error)**: Shared thread_id Across Distinct Users. Using a static or default `thread_id` across different user sessions causes conversation histories to collide, leaking private user data across sessions. Always derive thread_id from authenticated session tokens.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running Summarizers on Every Single Turn. Invoking an LLM summarizer on every turn doubles API call volume and latency. Trigger summarization only when accumulated tokens exceed 75% of the allocated budget.")
    st.warning("**Trap 4 (Quality Warning)**: Trimming Messages Inside Tool Calls. Discarding the user question that initiated a tool call leaves the model unable to synthesize the final answer. Always keep the root user query intact.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: trim_messages() Utility"):
        st.markdown("**Programmatically truncates message lists based on token budgets while preserving SystemMessage.**")
    with st.expander("🎴 Flashcard 2: start_on='human'"):
        st.markdown("**Guarantees the trimmed conversation begins with a clean user message, preventing orphaned ToolMessages.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: thread_id Parameter"):
        st.markdown("**Config key in RunnableConfig that partitions and restores persistent conversation sessions.**")
    with st.expander("🎴 Flashcard 4: Lost-in-the-Middle Effect"):
        st.markdown("**Degradation in model recall for facts positioned in the middle of long context windows.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural role of short-term conversation memory.", value=True)
chk2 = st.checkbox("I know how to represent multi-turn history using BaseMessage subclasses.", value=True)
chk3 = st.checkbox("I can persist and restore conversation threads using checkpointers and thread_id.", value=True)
chk4 = st.checkbox("I understand why context management and token budgets are necessary to prevent degradation.", value=True)
chk5 = st.checkbox("I know how to configure trim_messages with start_on='human' and include_system=True.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 conversation memory interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 10.2 Conversation Memory. Ready for 10.3 Long-Term Memory!")
