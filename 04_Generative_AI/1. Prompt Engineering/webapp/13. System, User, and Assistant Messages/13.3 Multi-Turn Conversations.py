import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="13.3 Multi-Turn Conversations | GenAI Roadmap",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .memory-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("💬 13.3 Multi-Turn Conversations — Context Management, Truncation & Memory")
st.markdown(
    """
Welcome to **Lesson 13.3: Multi-Turn Conversations**.  
All modern LLMs are fundamentally **stateless**: every API call is completely independent, and the model remembers nothing about previous interactions unless the client sends the conversation history back with every request.

In production, naive history accumulation causes exponential token growth, skyrocketing API costs, and context window overflow. Mastering multi-turn conversational engineering requires **sliding windows**, **hierarchical summarization**, and **structured entity memory**.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Mechanics of Stateless Conversational AI
# ==============================================================================
st.header("1. Stateless Execution & Token Accumulation", divider="rainbow")

col_st1, col_st2 = st.columns(2)

with col_st1:
    with st.container(border=True):
        st.subheader("⚙️ Stateless HTTP API Reality")
        st.markdown(
            """
- The LLM server has zero session state or persistent memory of your user.
- To maintain conversational coherence, the client application must re-transmit the **entire historical transcript** on Turn $N$:
  - Turn 1: `[System, User1] -> Asst1` (100 tokens)
  - Turn 2: `[System, User1, Asst1, User2] -> Asst2` (250 tokens)
  - Turn 3: `[System, User1, Asst1, User2, Asst2, User3] -> Asst3` (450 tokens)
- The cumulative tokens billed grow quadratically with conversation length.
"""
        )

with col_st2:
    with st.container(border=True):
        st.subheader("⚠️ Enterprise Pain Points of Naive History")
        st.markdown(
            """
1. **Cost Explosion**: In a 20-turn customer service chat, early messages are billed 20 separate times.
2. **Context Window Saturation**: Long conversations exceed maximum token limits (e.g. 8k, 32k, 128k).
3. **Attention Dispersion**: As irrelevant early turns accumulate, the model experiences "Lost in the Middle" degradation, forgetting recent user corrections.
4. **Latency Inflation**: Prefill processing time scales linearly with total prompt token count.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Enterprise Memory Strategies Compared
# ==============================================================================
st.header("2. Enterprise Conversation Memory Strategies", divider="rainbow")
st.markdown("Architectural comparison of production memory management patterns:")

strategies_df = pd.DataFrame({
    "Memory Pattern": [
        "1. Full Transcript (Naive)",
        "2. Sliding Window (BufferWindow)",
        "3. Hierarchical Summarization (SummaryBuffer)",
        "4. Entity / Key-Value Extraction",
        "5. Vector-Backed Semantic Episodic Memory"
    ],
    "How It Works": [
        "Appends all turns unconditionally until the API crashes or limits out",
        "Retains only the latest K turns (e.g., last 4 user-assistant pairs); discards older turns",
        "When token budget is exceeded, a secondary LLM summarizes older turns into a concise briefing",
        "Extracts structured user facts (e.g. name, preferences, account ID) into a persistent JSON profile",
        "Stores past user utterances in a vector DB; retrieves only top-k semantically relevant memories"
    ],
    "Token Efficiency": [
        "Worst (O(N^2) cumulative tokens)",
        "Highest (Constant O(K) token ceiling)",
        "High (Constant summary length + K turns)",
        "Extremely High (Minimal JSON metadata overhead)",
        "High (Retrieves only 2-3 relevant historical snippets)"
    ],
    "Best Production Use Case": [
        "Short 1-2 turn diagnostic workflows only",
        "Customer support triage where only immediate task context matters",
        "Long-running consultative advisory, enterprise coaching, complex problem solving",
        "Personalized copilots, e-commerce shopping assistants, user profile tracking",
        "Multi-session agents, lifelong companion bots, knowledge-base search"
    ]
})
st.dataframe(strategies_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Summary-Buffered Context Pipeline", divider="rainbow")
st.markdown("How the Summary-Buffer Memory pattern balances long-term memory with low token consumption:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        new_turn [label="New User Turn N", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        buffer [label="Recent K Turns\n(Last 3 Messages)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        summary [label="Rolling Summary\n(Compressed Past)", fillcolor="#EAB308", fontcolor="#000000"]
        system_msg [label="System Prompt\n(Core Persona)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="LLM Generation Engine", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        resp [label="Assistant Output", fillcolor="#10B981", fontcolor="#FFFFFF"]

        system_msg -> llm
        summary -> llm
        buffer -> llm
        new_turn -> llm
        llm -> resp
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Memory & Context Window Simulator
# ==============================================================================
st.header("4. Interactive Memory Strategy Simulator", divider="rainbow")
st.markdown("Simulate a multi-turn conversation and observe how different memory architectures manage context:")

selected_strategy = st.radio(
    "Choose Active Memory Architecture",
    ["Full Transcript (Naive)", "Sliding Window (Last 2 Turns)", "Summary + Recent Buffer"],
    horizontal=True
)

# Simulated conversation turns
CONVERSATION_TURNS = [
    ("user", "Hi, my name is Alex and I'm a developer building a fintech app on AWS."),
    ("assistant", "Nice to meet you, Alex! How can I assist with your fintech app on AWS today?"),
    ("user", "We are migrating from PostgreSQL to DynamoDB for our transaction ledger."),
    ("assistant", "DynamoDB is great for scale, but ensure you model access patterns carefully since single-table design requires strict partition keys."),
    ("user", "What were the two databases I mentioned earlier, and who am I?")
]

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("📜 Client-Side Ingestion Stream")
    for role, text in CONVERSATION_TURNS:
        with st.chat_message(role):
            st.write(text)

with col_sim2:
    st.subheader("🧠 Context Payload Sent to LLM (Turn 5)")
    
    if selected_strategy == "Full Transcript (Naive)":
        st.warning("⚠️ High Token Consumption: All 5 turns re-transmitted")
        payload = [
            {"role": "system", "content": "You are a helpful cloud architecture assistant."},
            {"role": "user", "content": CONVERSATION_TURNS[0][1]},
            {"role": "assistant", "content": CONVERSATION_TURNS[1][1]},
            {"role": "user", "content": CONVERSATION_TURNS[2][1]},
            {"role": "assistant", "content": CONVERSATION_TURNS[3][1]},
            {"role": "user", "content": CONVERSATION_TURNS[4][1]}
        ]
        st.json(payload)
        st.markdown("**Model Knowledge:** Remembers name (Alex) and both databases (Postgres, DynamoDB).")

    elif selected_strategy == "Sliding Window (Last 2 Turns)":
        st.error("❌ Context Amnesia: Turns 1 & 2 dropped!")
        payload = [
            {"role": "system", "content": "You are a helpful cloud architecture assistant."},
            {"role": "user", "content": CONVERSATION_TURNS[2][1]},
            {"role": "assistant", "content": CONVERSATION_TURNS[3][1]},
            {"role": "user", "content": CONVERSATION_TURNS[4][1]}
        ]
        st.json(payload)
        st.markdown("**Model Knowledge:** Remembers databases, but **FORGOT** the user's name (Alex) because Turn 1 fell outside the sliding window!")

    else:
        st.success("✅ Optimal Memory: Compressed Summary + Recent Turns")
        payload = [
            {"role": "system", "content": "You are a helpful cloud architecture assistant.\n<conversation_summary>User is Alex, a fintech developer building on AWS. Discussed migrating transaction ledger from PostgreSQL to DynamoDB.</conversation_summary>"},
            {"role": "user", "content": CONVERSATION_TURNS[2][1]},
            {"role": "assistant", "content": CONVERSATION_TURNS[3][1]},
            {"role": "user", "content": CONVERSATION_TURNS[4][1]}
        ]
        st.json(payload)
        st.markdown("**Model Knowledge:** Perfectly recalls Alex and both databases using only 35% of the token count!")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: SummaryBuffer Memory Class", divider="rainbow")
st.markdown("A lightweight Python class implementing token-bounded sliding memory with background summarization:")

st.code(
    '''from typing import List, Dict, Optional

class SummaryBufferMemory:
    """
    Production conversation memory manager.
    Maintains a rolling summary of older turns once message count exceeds threshold.
    """
    def __init__(self, system_prompt: str, max_buffer_turns: int = 4):
        self.system_prompt = system_prompt
        self.max_buffer_turns = max_buffer_turns
        self.summary: Optional[str] = None
        self.recent_messages: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        if len(self.recent_messages) > self.max_buffer_turns:
            self._compact_oldest_turns()

    def _compact_oldest_turns(self):
        # Extract the oldest 2 turns to summarize
        oldest_turns = self.recent_messages[:2]
        self.recent_messages = self.recent_messages[2:]
        
        # In production: invoke a fast, lightweight model (e.g. gpt-4o-mini) to update summary
        extracted_facts = f"User discussed: '{oldest_turns[0]['content'][:40]}...'"
        if self.summary:
            self.summary += f" | {extracted_facts}"
        else:
            self.summary = extracted_facts

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        sys_content = self.system_prompt
        if self.summary:
            sys_content += f"\\n\\n<conversation_history_summary>\\n{self.summary}\\n</conversation_history_summary>"
        
        return [{"role": "system", "content": sys_content}] + self.recent_messages

# Demonstration
mem = SummaryBufferMemory(system_prompt="You are a personal finance assistant.", max_buffer_turns=2)
mem.add_turn("user", "I earn $120,000 annually and want to buy a house in Denver.")
mem.add_turn("assistant", "Denver median home price is $550k. What is your savings?")
mem.add_turn("user", "I have $60,000 saved.")
mem.add_turn("assistant", "That provides a solid 10% down payment.")

api_payload = mem.get_messages_for_api()
print(f"Messages sent to LLM: {len(api_payload)}")
print(f"System content includes summary: {'Denver' in api_payload[0]['content']}")
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: What is the 'Lost in the Middle' phenomenon and how does it affect multi-turn chat applications?", expanded=True):
        st.markdown(
            """
Published by Liu et al. (Stanford), the **Lost in the Middle** paper demonstrated that transformer models have U-shaped attention curves: they attend best to information at the very beginning (Primacy) and very end (Recency) of their input context. 
In long multi-turn conversations where relevant facts reside in the middle of a 20-message transcript, retrieval and reasoning accuracy degrades significantly. This is why extracting key facts into the System prompt (Primacy) or maintaining concise summaries yields superior performance over raw long contexts.
"""
        )
    with st.expander("Q2: When should an enterprise application choose Vector-Backed Episodic Memory over Summarization?"):
        st.markdown(
            """
- **Summarization** is best for **intra-session** continuity (e.g., an ongoing 30-minute support call), where the temporal sequence of actions matters.
- **Vector-Backed Episodic Memory** is essential for **cross-session / long-term** recall (e.g., remembering a user's allergy mentioned 3 months ago across separate logins). Searching an embedding index allows the assistant to pull specific episodic memories without bloating every single session's baseline prompt.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we just set max_tokens=128k and send the entire chat history forever?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Modern models have 128k or 1M context windows, so memory management is obsolete—just send all messages."*
- **Winning Response**: *"No. Just because an LLM accepts 128,000 tokens does not mean you should send them. Sending 100k tokens on every chat turn causes: (1) enormous quadratic API costs ($0.50-$2.00 per message), (2) high latency (5-10 second TTFT prefill delays), and (3) degraded reasoning accuracy due to attention dilution. Production systems always enforce memory compaction (sliding windows, entity profiles, summaries) to keep context windows lean and fast."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Stateless LLM Execution**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The architectural fact that LLM servers maintain zero memory between API calls, requiring clients to re-send history on every turn.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Summary-Buffer Memory**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A pattern that compresses older turns into an evolving summary while retaining the most recent K turns verbatim.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Lost in the Middle**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The degradation of model retrieval performance when critical information is placed in the middle of long context windows.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 13.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Stateless Nature**: LLMs have no persistent memory; client software must orchestrate conversational state.
2. **Avoid Full History**: Naive transcript accumulation creates quadratic costs and latency spikes.
3. **Summary-Buffer Efficiency**: Combine high-level rolling summaries with a sliding window of recent turns.
4. **Attention Hygiene**: Keep context windows lean to avoid the "Lost in the Middle" attention degradation.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why LLM APIs require re-sending previous conversation turns.", value=True, key="chk_13_3_1")
    st.checkbox("I know how to design a SummaryBufferMemory architecture in Python.", value=True, key="chk_13_3_2")
    st.checkbox("I understand the 'Lost in the Middle' effect on long context windows.", value=True, key="chk_13_3_3")
    st.checkbox("I know when to choose vector-backed episodic memory vs. sliding window memory.", value=True, key="chk_13_3_4")
