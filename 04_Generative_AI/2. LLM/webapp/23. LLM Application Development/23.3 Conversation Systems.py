import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="23.3 Conversation Systems - LLM Engineering",
    page_icon="💬",
    layout="wide"
)

# Custom Theme CSS
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
        background-color: #0369A1;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-assistant {
        background-color: #059669;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-system {
        background-color: #64748B;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE CLASSES FOR CONVERSATION MANAGEMENT & WORKBENCH
# ==============================================================================
class ChatMessage:
    """Represents an immutable conversational turn."""
    def __init__(self, role: str, content: str, turn_id: int, tokens: Optional[int] = None):
        self.role = role
        self.content = content
        self.turn_id = turn_id
        # Approximation: ~4 characters per token for English text
        self.tokens = tokens if tokens is not None else max(1, len(content) // 4 + 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens
        }

class ConversationManager:
    """
    Production-grade Conversation State and Context Management coordinator.
    Implements sliding window eviction, token budget truncation, and summary buffer compression.
    """
    def __init__(self, system_prompt: str, token_budget: int = 1000):
        self.system_prompt = system_prompt
        self.system_tokens = max(1, len(system_prompt) // 4 + 2)
        self.token_budget = token_budget
        self.messages: List[ChatMessage] = []
        self.running_summary: str = ""
        self.summary_tokens: int = 0
        self.turn_counter: int = 0

    def add_user_message(self, content: str) -> ChatMessage:
        self.turn_counter += 1
        msg = ChatMessage(role="user", content=content, turn_id=self.turn_counter)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> ChatMessage:
        self.turn_counter += 1
        msg = ChatMessage(role="assistant", content=content, turn_id=self.turn_counter)
        self.messages.append(msg)
        return msg

    def update_summary(self, new_summary: str):
        self.running_summary = new_summary
        self.summary_tokens = max(1, len(new_summary) // 4 + 2) if new_summary else 0

    def build_context(self, strategy: str = "Sliding Token Budget", max_window_turns: int = 4) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Assembles prompt context based on selected context engineering strategy.
        """
        payload: List[Dict[str, str]] = []
        metadata: Dict[str, Any] = {
            "total_stored_turns": len(self.messages),
            "evicted_turns": 0,
            "strategy": strategy,
            "system_tokens": self.system_tokens,
            "summary_tokens": self.summary_tokens,
            "history_tokens": 0,
            "total_tokens": 0
        }

        # 1. Base System Prompt
        sys_content = self.system_prompt
        if self.running_summary:
            sys_content += f"\n\n[PRIOR CONVERSATION SUMMARY]:\n{self.running_summary}"
        payload.append({"role": "system", "content": sys_content})

        if strategy == "Full Raw History":
            # Naive accumulation: append all messages
            for m in self.messages:
                payload.append({"role": m.role, "content": m.content})
                metadata["history_tokens"] += m.tokens

        elif strategy == "Sliding Turn Window":
            # Retain only last N turns
            retained = self.messages[-max_window_turns:] if len(self.messages) > max_window_turns else self.messages
            metadata["evicted_turns"] = len(self.messages) - len(retained)
            for m in retained:
                payload.append({"role": m.role, "content": m.content})
                metadata["history_tokens"] += m.tokens

        elif strategy == "Sliding Token Budget":
            # Greedily take messages from latest to oldest until budget is exhausted
            remaining_budget = self.token_budget - (self.system_tokens + self.summary_tokens)
            retained_rev: List[ChatMessage] = []
            curr_tokens = 0
            
            for m in reversed(self.messages):
                if curr_tokens + m.tokens <= remaining_budget:
                    retained_rev.append(m)
                    curr_tokens += m.tokens
                else:
                    break
            
            retained = list(reversed(retained_rev))
            metadata["evicted_turns"] = len(self.messages) - len(retained)
            metadata["history_tokens"] = curr_tokens
            for m in retained:
                payload.append({"role": m.role, "content": m.content})

        metadata["total_tokens"] = self.system_tokens + self.summary_tokens + metadata["history_tokens"]
        return payload, metadata


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("23.3 Conversation Systems: Chat History Persistence, Sliding Windows, & Memory Management")
st.caption("Chapter 23: LLM Application Development | Domain: 2. Large Language Models")

st.markdown("""
Modern Foundation Large Language Models are fundamentally **stateless mathematical operators**: they possess zero 
internal memory of previous prompts or interactions. Every illusion of a continuous, cohesive dialogue is an artifact 
of software orchestration—specifically, the application repeatedly injecting past interaction logs into the model's 
input context window for every successive generation turn.

However, as dialogues scale across dozens or hundreds of conversational turns, naively appending all historical messages 
triggers **Context Window Exhaustion**, balloons API inference costs quadratically (O(N^2) total token usage), introduces 
excessive prefill latency, and suffers from the **Lost in the Middle** attention degradation phenomenon.

Production conversational platforms implement industrial-grade context engineering: **Append-Only Immutable Chat Logs**, 
**Relational & Redis Multi-Tier Session Stores**, **Sliding Token Window Eviction**, **Background Summary Buffer Memory**, 
and **Long-Term Vector-Retrieved Episodic Memory**.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Tier State, Context, & Episodic Memory Lifecycle", divider="rainbow")

st.markdown("""
The diagram below illustrates the end-to-end conversation pipeline in an enterprise LLM platform. 
Incoming user queries flow through session validation, retrieve active working context from in-memory caches, 
pull long-term user facts from vector memory, enforce strict token budgets via compaction, and persist turns 
into immutable database stores.
""")

blueprint_dot = """
digraph ConversationArchitecture {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_client {
        label = "Tier 1: Client & Session Gateway";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        Client [label="User / Frontend Client\\n(Web Chat / Mobile / Slack)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Gateway [label="Session Gateway & Auth Guard\\n(JWT Verification & Tenant Isolation)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_persistence {
        label = "Tier 2: Dual-Store Conversation Persistence";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        RedisCache [label="Transient Working Cache (Redis)\\n(Active Thread TTL: 1 hr | Sub-5ms Read)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        PostgresDB [label="Append-Only Message Store (PostgreSQL)\\n(UUID Session, Parent-ID Tree, Audit Trail)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_context_engine {
        label = "Tier 3: Context & Memory Management Engine";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        MemoryRouter [label="Memory & Context Coordinator\\n(Token Budget Allocator: 4k/8k/32k)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        SlidingWindow [label="Sliding Token Window\\n(Verbatim Recent K Turns)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        SummaryBuffer [label="Summary Buffer Generator\\n(Background LLM Compression of Turns > K)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        VectorEpisodic [label="Episodic Vector Memory (HNSW)\\n(Cosine Retrieval of Relevant Historical Sessions)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_assembly {
        label = "Tier 4: Prompt Assembly & Model Inference";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        ContextAssembler [label="Unified Context Assembler\\n[System Prompt + User Facts + Summary + Recent Dialog + New Query]", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        LLMInference [label="Foundation LLM API\\n(GPT-4o / Claude 3.5 / Llama 3.3)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    Client -> Gateway [label="1. Send Message + session_id"];
    Gateway -> RedisCache [label="2. Fetch Active Buffer"];
    RedisCache -> PostgresDB [label="Fallback Cold Read", style="dashed"];
    Gateway -> MemoryRouter [label="3. Pass Raw History & Query"];
    
    MemoryRouter -> SlidingWindow [label="Extract Last K Turns"];
    MemoryRouter -> SummaryBuffer [label="Fetch Running Summary"];
    MemoryRouter -> VectorEpisodic [label="Query Past Facts / Preferences"];

    SlidingWindow -> ContextAssembler;
    SummaryBuffer -> ContextAssembler;
    VectorEpisodic -> ContextAssembler;

    ContextAssembler -> LLMInference [label="4. Standard ChatML Payload"];
    LLMInference -> RedisCache [label="5. Append Assistant Response"];
    LLMInference -> PostgresDB [label="6. Async Commit Turn & Token Telemetry"];
    LLMInference -> Client [label="7. Stream SSE Tokens"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - CHAT HISTORY
# ==============================================================================
st.header("2. Chat History: Append-Only Immutable Message Logs, Roles, and Schemas", divider="rainbow")

st.markdown("""
In enterprise conversational applications, **Chat History** is the chronological, append-only log of every interaction 
between human participants, software systems, and artificial intelligence models. 

Historically, legacy chatbots concatenated raw strings into unstructured text buffers. In contrast, modern foundation 
models rely strictly on **Structured Role Schemas** (formalized by the Chat Markup Language standard, ChatML).
""")

col_ch1, col_ch2 = st.columns([1, 1])

with col_ch1:
    st.markdown("""
    ### Canonical Conversational Roles
    * **`system`**: The foundational steering prompt defining the model's persona, constraints, safety boundaries, and tool descriptions.
    * **`user`**: Raw natural language queries or structured instructions issued by the end-user.
    * **`assistant`**: Model-generated natural language responses, internal reasoning traces (in reasoning models like o1/DeepSeek-R1), or tool invocation requests.
    * **`tool` / `function`**: Structured JSON execution results returned by the host runtime following an assistant's tool call.
    * **`developer`**: (Introduced in latest OpenAI models) Replaces or augments the system role with higher instruction hierarchy priority to combat jailbreaks.
    """)

with col_ch2:
    st.markdown("""
    ### Append-Only Invariant & Schema Design
    * **Immutability Principle**: A chat history message, once generated and committed, is never mutated in place. If a user edits an earlier message, the system creates a new branch pointer.
    * **Complete Auditability**: Each message records:
      * `message_id`: Global UUIDv7 (time-ordered).
      * `session_id`: Unique thread identifier.
      * `parent_id`: Pointer to previous turn, enabling tree-based branching.
      * `token_count`: Pre-calculated token length for rapid budget math.
      * `latency_ms`: Time-to-first-token and total generation duration.
    """)

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
Modern foundation models (such as Llama 3, GPT-4o, and Claude 3.5) enforce strict token-level special delimiters 
(e.g., `<|start_header_id|>assistant<|end_header_id|>`) under the hood. Any violation of alternating role turns 
or invalid message schemas causes immediate API rejection or severe hallucinations. Furthermore, prompt caching 
relies entirely on exact prefix matching of historical turns; mutating past history invalidates the entire KV cache!
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - CONVERSATION STATE
# ==============================================================================
st.header("3. Conversation State: Transient Buffers, Relational Persistence, and Branching Forks", divider="rainbow")

st.markdown("""
A conversation is not merely a flat list of text; it is a **stateful computational graph**. Conversation state encompasses 
the active topic, transient slots/variables (e.g., `user_authenticated=True`, `target_account=4821`), conversation status 
(active, paused, human_handoff, resolved), and message lineage.
""")

col_cs1, col_cs2 = st.columns([1, 1])

with col_cs1:
    st.markdown("""
    ### Relational Database Schema (PostgreSQL)
    Production conversation systems store conversation state across normalized tables:
    """)
    st.code("""
-- Production PostgreSQL Conversation State Schema
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    tenant_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) DEFAULT 'New Conversation',
    system_prompt_override TEXT,
    running_summary TEXT DEFAULT '',
    status VARCHAR(32) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES chat_messages(id),
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    tool_calls JSONB,
    prompt_tokens INT NOT NULL,
    completion_tokens INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_session_parent ON chat_messages(session_id, parent_id);
    """, language="sql")

with col_cs2:
    st.markdown("""
    ### Branching Forks & Message Trees
    When a user clicks **Regenerate Response** or edits a message sent 5 turns ago, naive linear arrays break down. 
    Production platforms structure conversation history as a **Directed Tree (DAG)**:
    * Every message points to its `parent_id`.
    * A conversational thread view is simply the single path traced backward from the **Active Leaf Node** to the Root:
    """)
    branch_dot = """
    digraph MessageTree {
        graph [rankdir=TB, bgcolor="#0B1120", pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [fontname="Helvetica", fontsize=8, color="#64748B"];

        M1 [label="Turn 1: User\\n'Explain Attention'", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        M2 [label="Turn 2: Assistant (v1)\\n'Attention is a matrix...'", fillcolor="#1E293B", fontcolor="#94A3B8", color="#64748B"];
        M2_alt [label="Turn 2: Assistant (Regenerated v2)\\n'Attention routes token focus...'", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        M3 [label="Turn 3: User\\n'Show me the math'", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];

        M1 -> M2 [label="Branch 1 (Archived)"];
        M1 -> M2_alt [label="Branch 2 (Active)", color="#10B981", penwidth=2.0];
        M2_alt -> M3 [label="Active Thread", color="#10B981", penwidth=2.0];
    }
    """
    st.graphviz_chart(branch_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CONTEXT MANAGEMENT
# ==============================================================================
st.header("4. Context Management: Token Budgets, Sliding Windows, and Dynamic Compaction", divider="rainbow")

st.markdown("""
Even with models sporting 128k to 2M token context windows, unmanaged context accumulation is an anti-pattern. 
Every additional token passed to the model:
1. **Increases Prefill Latency**: Matrix multiplications for initial KV-cache compute scale linearly with input tokens.
2. **Multiplies Inference Costs**: Every turn re-processes all prior tokens, leading to quadratic cumulative token expenditures.
3. **Triggers "Lost in the Middle"**: Key facts placed in the central 40-70% of a massive context window suffer dramatic recall degradation compared to facts placed at the extreme beginning or end.
""")

col_cm1, col_cm2 = st.columns([1, 1])

with col_cm1:
    st.markdown("""
    ### The Three Classical Context Strategies
    1. **Full History Buffer**:
       * Injects all past turns verbatim.
       * *Pros*: Zero information loss on short chats.
       * *Cons*: Quadratic token costs; crashes on long conversations.
    2. **Sliding Turn / Token Window**:
       * Retains only the most recent $K$ turns (or latest $N$ tokens); older turns are dropped.
       * *Pros*: Constant predictable latency and cost.
       * *Cons*: Complete amnesia regarding instructions or preferences stated in turn 1.
    3. **Conversation Summary Buffer (Modern Standard)**:
       * Maintains a compact running summary of older turns while keeping the most recent $K$ turns verbatim.
       * *Pros*: Retains long-term constraints while preserving natural immediate dialogue flow.
    """)

with col_cm2:
    st.markdown("""
    ### Token Budget Allocation Model
    Production systems enforce strict partitioning of the model's total context window:
    """)
    budget_data = {
        "Context Segment": ["System Prompt & Tools", "Retrieved RAG Documents", "User Persona & Memory", "Sliding Chat History", "Generation Headroom"],
        "Token Budget": ["1,500 tokens", "4,000 tokens", "500 tokens", "2,000 tokens", "4,000 tokens"],
        "Priority": ["Priority 1 (Mandatory)", "Priority 3 (Prunable)", "Priority 2 (High)", "Priority 4 (Sliding FIFO)", "Reserved Output"]
    }
    st.dataframe(pd.DataFrame(budget_data), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - MEMORY MANAGEMENT
# ==============================================================================
st.header("5. Memory Management: Short-Term Working Memory vs Long-Term Vector Episodic Memory", divider="rainbow")

st.markdown("""
A human conversing with a doctor or assistant expects two distinct types of memory: immediate conversational memory 
(remembering the sentence spoken 10 seconds ago) and longitudinal episodic memory (remembering a penicillin allergy 
disclosed three months ago). LLM application architectures mirror human cognitive science by implementing a dual-memory system.
""")

col_mm1, col_mm2 = st.columns([1, 1])

with col_mm1:
    st.markdown("""
    ### Cognitive Memory Types in LLMs
    * **Working / Short-Term Memory (Context Window)**:
      * Stored in RAM / Redis.
      * Lifespan: Current conversational session (minutes to hours).
      * High-fidelity, sequential, verbatim text.
    * **Episodic Long-Term Memory (Vector Database)**:
      * Stored in HNSW vector indices (Pinecone, Qdrant, Chroma).
      * Past sessions are chunked, embedded, and tagged with metadata.
      * Retrieved dynamically using semantic cosine similarity when relevant topics re-emerge.
    * **Semantic Entity Profile Memory (Document Store)**:
      * Stored in JSON/PostgreSQL (e.g., `user_profiles.extracted_facts`).
      * Extracted by background LLMs: `"Alice is allergic to peanuts"`, `"Prefers Python over TypeScript"`.
      * Automatically injected into every future system prompt.
    """)

with col_mm2:
    st.markdown("""
    ### Memory Synthesis Lifecycle
    """)
    mem_flow_dot = """
    digraph MemoryFlow {
        graph [rankdir=TB, bgcolor="#0B1120", pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [fontname="Helvetica", fontsize=8, color="#64748B"];

        UserTurn [label="User Input: 'As you know, I work remotely in Berlin'", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        WorkingMem [label="Working Memory (Active Window)\\nImmediate dialogue continuation", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Extractor [label="Async Background Entity Extractor\\n(LLM extracts structured user facts)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        ProfileStore [label="User Profile Key-Value Store\\n{location: 'Berlin', work: 'remote'}", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        VectorStore [label="Episodic Vector Embeddings\\n(Archive for cross-session semantic search)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];

        UserTurn -> WorkingMem;
        UserTurn -> Extractor [style="dashed", label="Async Event"];
        Extractor -> ProfileStore [label="Upsert Fact"];
        Extractor -> VectorStore [label="Index Chunks"];
    }
    """
    st.graphviz_chart(mem_flow_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - SESSION MANAGEMENT
# ==============================================================================
st.header("6. Session Management: Session Lifecycles, Multi-Tenancy, TTL Caching, and Invalidation", divider="rainbow")

st.markdown("""
Session management governs the boundaries, security, and lifecycle of user conversations across a distributed 
infrastructure. Without strict session management, systems suffer from memory leaks, cross-tenant data contamination, 
and concurrency race conditions.
""")

col_sm1, col_sm2 = st.columns([1, 1])

with col_sm1:
    st.markdown("""
    ### Session Lifecycle States
    1. **Initialization (`CREATED`)**: Authenticated handshake creates session UUID, binds `tenant_id`, and loads user profile preferences.
    2. **Active Interaction (`ACTIVE`)**: Messages stream back and forth; session TTL in Redis is refreshed upon each turn (e.g., 60-minute sliding expiry).
    3. **Idle Compaction (`IDLE`)**: After 15 minutes of inactivity, background workers run summary compaction and push raw logs to cold relational storage.
    4. **Termination (`CLOSED` / `EXPIRED`)**: Explicit user closure or TTL expiration flushes state and invalidates Redis cache keys.
    """)

with col_sm2:
    st.markdown("""
    ### Concurrency & Distributed Locks (Redlock)
    What happens if a user submits a second query while the model is still generating the response to the first?
    * **Race Condition**: Both requests read state $S_0$. Request A appends turn 1 and writes $S_1$. Request B overwrites with $S_2$, completely losing turn 1!
    * **Solution**: Distributed Redis Locks:
      * When a generation initiates, acquire a session-scoped mutex: `SET session:{id}:lock {token} NX EX 30`.
      * Subsequent requests either queue gracefully or reject with HTTP 429: *"Generation already in progress for this session."*
    """)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full-Stack Enterprise Conversation State & Context Manager", divider="rainbow")

st.markdown("""
Below is the production-grade, self-contained Python coordinator implementing the complete conversation lifecycle: 
session creation, message ingestion, token tracking, sliding token window budgeting, and summary buffer compression.
""")

prod_code = """
from typing import Dict, List, Any, Optional, Tuple

class ProductionConversationCoordinator:
    \"\"\"
    Enterprise conversation state and context manager.
    Coordinates token accounting, sliding windows, and summary compression.
    \"\"\"
    def __init__(self, session_id: str, system_prompt: str, max_token_budget: int = 2000):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.max_token_budget = max_token_budget
        self.messages: List[Dict[str, Any]] = []
        self.running_summary: str = ""
        self.turn_id: int = 0

    def _estimate_tokens(self, text: str) -> int:
        # Standard production heuristic: 1 token ≈ 4 characters
        return max(1, len(text) // 4 + 2)

    def append_turn(self, role: str, content: str) -> Dict[str, Any]:
        self.turn_id += 1
        turn = {
            "turn_id": self.turn_id,
            "role": role,
            "content": content,
            "tokens": self._estimate_tokens(content)
        }
        self.messages.append(turn)
        return turn

    def compress_history(self, summary_text: str):
        \"\"\"Updates running summary of conversations evicted from active working memory.\"\"\"
        self.running_summary = summary_text

    def assemble_llm_payload(self) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        \"\"\"
        Compiles the strict ChatML message array while strictly enforcing token budget.
        \"\"\"
        sys_tokens = self._estimate_tokens(self.system_prompt)
        sum_tokens = self._estimate_tokens(self.running_summary) if self.running_summary else 0
        fixed_tokens = sys_tokens + sum_tokens
        
        available_budget = max(200, self.max_token_budget - fixed_tokens)
        
        # Greedily take most recent messages that fit into budget
        retained_messages = []
        used_tokens = 0
        for msg in reversed(self.messages):
            if used_tokens + msg["tokens"] <= available_budget:
                retained_messages.append(msg)
                used_tokens += msg["tokens"]
            else:
                break
                
        retained_messages.reverse()
        
        # Build API payload
        payload = []
        full_system = self.system_prompt
        if self.running_summary:
            full_system += f"\\n\\n[Conversation Summary So Far]:\\n{self.running_summary}"
        payload.append({"role": "system", "content": full_system})
        
        for m in retained_messages:
            payload.append({"role": m["role"], "content": m["content"]})
            
        telemetry = {
            "total_tokens": fixed_tokens + used_tokens,
            "history_tokens": used_tokens,
            "evicted_turns": len(self.messages) - len(retained_messages)
        }
        return payload, telemetry
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Multi-Turn Conversation Simulator & Token Context Compactor", divider="rainbow")

st.markdown("""
Experiment interactively with conversation state management, sliding window eviction, and summary buffer compression. 
Observe in real time how different context engineering strategies assemble prompt payloads and stay within token budgets.
""")

# Initialize Session State Manager
if "conv_manager" not in st.session_state:
    mgr = ConversationManager(
        system_prompt="You are an enterprise AI architect. Provide concise, highly technical guidance.",
        token_budget=350
    )
    # Seed with 3 realistic turns
    mgr.add_user_message("Hello! I need to design a high-throughput RAG system for 10M documents.")
    mgr.add_assistant_message("Understood. For 10M documents, implement HNSW indexing in Qdrant with hybrid BM25 search and cross-encoder reranking.")
    mgr.add_user_message("What about chunking strategies for legal contracts?")
    mgr.add_assistant_message("Use hierarchical chunking: 1000-token parent chunks for LLM context, with 250-token child chunks for dense vector embedding search.")
    st.session_state.conv_manager = mgr

mgr: ConversationManager = st.session_state.conv_manager

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Add Dialogue Turn & Configure Engine")
    
    strategy = st.selectbox(
        "Context Management Strategy",
        ["Sliding Token Budget", "Sliding Turn Window", "Full Raw History"],
        index=0,
        help="Select how older messages are pruned or compressed."
    )
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        token_budget = st.slider("Total Token Budget Cap", min_value=150, max_value=800, value=350, step=25)
        mgr.token_budget = token_budget
    with col_cfg2:
        turn_window = st.slider("Window Turn Count (if Turn-based)", min_value=1, max_value=8, value=3)

    st.markdown("#### Input New Message")
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        new_user_msg = st.text_input("User Prompt", value="How do we handle session state persistence across multiple replicas?")
    with col_input2:
        st.write("")
        st.write("")
        send_btn = st.button("Send Turn ➡️", use_container_width=True)

    if send_btn and new_user_msg.strip():
        mgr.add_user_message(new_user_msg.strip())
        # Simulate realistic assistant reply
        mgr.add_assistant_message("Store active session state in a Redis cluster with Redis Sentinel for failover, while streaming durable append-only chat logs to PostgreSQL.")
        st.rerun()

    st.markdown("#### Background Summary Buffer")
    col_sum1, col_sum2 = st.columns([3, 1])
    with col_sum1:
        custom_summary = st.text_area(
            "Current Running Summary",
            value=mgr.running_summary,
            placeholder="No summary generated yet. Enter synthetic summary or click Auto-Compress.",
            height=70
        )
    with col_sum2:
        st.write("")
        st.write("")
        if st.button("Auto-Compress Older Turns 🗜️", use_container_width=True):
            mgr.update_summary("User is architecting an enterprise RAG system (10M docs, hierarchical chunking) and querying Redis session persistence.")
            st.rerun()
        if st.button("Clear Summary", use_container_width=True):
            mgr.update_summary("")
            st.rerun()

    if custom_summary != mgr.running_summary:
        mgr.update_summary(custom_summary)

with col_wb2:
    st.subheader("2. Compiled Context Payload & Token Telemetry")
    
    payload, meta = mgr.build_context(strategy=strategy, max_window_turns=turn_window)
    
    # Telemetry metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Tokens", meta["total_tokens"], f"Max: {token_budget}")
    mcol2.metric("History Tokens", meta["history_tokens"])
    mcol3.metric("Total Turns", meta["total_stored_turns"])
    mcol4.metric("Evicted Turns", meta["evicted_turns"])

    # Token progress bar
    budget_pct = min(1.0, meta["total_tokens"] / max(1, token_budget))
    st.progress(budget_pct, text=f"Context Budget Consumption: {meta['total_tokens']}/{token_budget} tokens ({int(budget_pct*100)}%)")

    st.markdown("#### Final Assembled Prompt (ChatML Format)")
    with st.container(height=360):
        for msg in payload:
            role = msg["role"]
            if role == "system":
                st.markdown(f"<span class='badge-system'>SYSTEM</span>", unsafe_allow_html=True)
                st.info(msg["content"])
            elif role == "user":
                st.markdown(f"<span class='badge-user'>USER</span>", unsafe_allow_html=True)
                st.markdown(f"**{msg['content']}**")
            else:
                st.markdown(f"<span class='badge-assistant'>ASSISTANT</span>", unsafe_allow_html=True)
                st.markdown(msg["content"])
            st.divider()

if st.button("Reset Conversation Simulator 🔄"):
    st.session_state.conv_manager = ConversationManager(
        system_prompt="You are an enterprise AI architect. Provide concise, highly technical guidance.",
        token_budget=350
    )
    st.rerun()

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering conversation systems requires answering tough architectural interview questions and understanding 
critical production pitfalls.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: What causes the 'Lost in the Middle' phenomenon, and how does context management mitigate it?"):
    st.markdown("""
    **Answer**: The 'Lost in the Middle' phenomenon (documented by Liu et al., 2023) demonstrates that transformer 
    causal language models retrieve and reason over facts with highest accuracy when they appear at the very beginning 
    (primacy bias) or very end (recency bias) of the input context. Facts positioned in the central 40% to 70% of a massive 
    context window experience recall degradation of 20% to 50%.
    
    Conversation systems mitigate this through:
    1. **Summary Buffers**: Compacting central historical dialogue into concise facts prepended directly adjacent to the system prompt.
    2. **Recency Windows**: Guaranteeing that the most recent 3-5 dialogue turns remain uncompressed at the immediate end of the prompt.
    3. **Semantic Fact Extraction**: Pulling critical user constraints out of dialogue and inserting them into dedicated system prompt headers.
    """)

with st.expander("Q2: How do you design conversation branching (e.g. Regenerate Response) in a relational database?"):
    st.markdown("""
    **Answer**: In relational databases (PostgreSQL), conversation branching is modeled as a **Directed Acyclic Graph (DAG)** 
    or tree rather than a flat array. Every message record stores an optional `parent_id` foreign key referencing the preceding `message_id`.
    
    * When a user generates an initial turn: `Message A (parent=None) -> Message B (parent=A)`.
    * When the user clicks 'Regenerate' on Message B: The system does *not* delete Message B. Instead, it generates `Message C (parent=A)`.
    * The conversation UI records the current `active_leaf_message_id`. 
    * To render the chat thread, a recursive SQL Common Table Expression (CTE) traverses upward from the leaf node to the root, reconstructing that specific branch in sub-10 milliseconds.
    """)

with st.expander("Q3: Why doesn't increasing foundation model context lengths (e.g. 1M+ tokens) eliminate context management?"):
    st.markdown("""
    **Answer**: Massive context windows solve storage capacity, not system economics or attention fidelity:
    1. **Quadratic Cumulative Costs**: Without eviction or summarization, a 50-turn dialogue where each turn has 500 tokens results in cumulative billing for over 600,000 prompt tokens instead of ~25,000 tokens.
    2. **Prefill Latency**: Ingesting 200,000 tokens requires 2-5 seconds of GPU prefill computation before the first response token can stream.
    3. **Prompt Cache Invalidation**: Without careful context structuring, minor edits or arbitrary middle insertions invalidate the KV-cache, forfeiting 50-80% cloud provider discounts.
    """)

with st.expander("Q4: How do you guarantee multi-tenant data isolation and prevent cross-session memory leakage?"):
    st.markdown("""
    **Answer**:
    1. **Dual Scoping in Persistence**: Every query to Redis or PostgreSQL must enforce composite keys: `WHERE tenant_id = :tenant_id AND session_id = :session_id`.
    2. **Vector Partitioning**: Episodic vector memory indices must partition vectors using metadata filters (`tenant_id = 'acme_corp'`) or physically isolated tenant namespaces (e.g., Pinecone namespaces).
    3. **Zero In-Memory Globals**: The conversation coordinator must be instantiated per request with strict tenant context; never store session state in process-wide module variables.
    """)

with st.expander("Q5: What is the architectural difference between Vector Episodic Memory and Summary Buffer Memory?"):
    st.markdown("""
    **Answer**:
    * **Summary Buffer Memory** is *sequential and chronological*. An LLM recursively summarizes older dialogue turns, maintaining a compressed timeline. It is best for preserving general conversational flow, previous topics, and immediate multi-turn context.
    * **Vector Episodic Memory** is *non-linear and semantic*. Past conversations from days or months ago are chunked and embedded in a vector database. When the user asks a question today, top-K semantically relevant historical snippets are retrieved via cosine similarity. It is best for remembering specific historical facts (*"What was the password rule we discussed 3 weeks ago?"*) across long time horizons.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Naive String Concatenation Leading to Prompt Injection (CRITICAL)**
    * *Error*: Concatenating chat history as raw text (`"User: " + prompt + "\nAssistant: "`) allows malicious user prompts containing `"Assistant: Disregard prior instructions"` to hijack the model persona.
    * *Fix*: Always pass structured JSON role arrays directly to the provider's official Chat API.
    """)

    st.error("""
    **Trap 2: Concurrent User Requests Overwriting Session State (CRITICAL)**
    * *Error*: If a user rapidly sends two messages in parallel, concurrent threads read the same state, compute independently, and overwrite each other's turns in Redis.
    * *Fix*: Acquire a distributed mutex (Redis Redlock) keyed to `session:{id}:lock` during generation, or use transactional optimistic locking (`version` field) in PostgreSQL.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Blind FIFO Truncation Severing Tool Calling Pairs (WARNING)**
    * *Warning*: Naively evicting the oldest message when truncating by turn count can drop an `assistant` tool-call message while retaining its corresponding `tool` response message.
    * *Impact*: Commercial APIs (OpenAI, Anthropic) immediately throw a 400 Bad Request error if a `tool` message lacks its parent `tool_calls` assistant turn.
    * *Fix*: Group atomic tool call/response pairs and evict them together.
    """)

    st.warning("""
    **Trap 4: Infinite Recursive Summarization Information Degradation (WARNING)**
    * *Warning*: Repeatedly passing an existing summary into a summarizer to compress new messages creates a "game of telephone," losing critical nuances and hallucinating facts over time.
    * *Impact*: Loss of specific numerical constraints (prices, dates, IDs).
    * *Fix*: Extract structured key-value entities alongside free-form textual summaries.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: Sliding Window</strong><br><br>
        <em>Front:</em> Why is sliding token window preferred over sliding turn window?<br><br>
        <em>Back:</em> Turn lengths vary wildly (5 tokens vs 3,000 tokens). Token budgeting guarantees strict physical limits.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Summary Buffer</strong><br><br>
        <em>Front:</em> What is the primary engineering cost of Summary Buffer Memory?<br><br>
        <em>Back:</em> Requires an asynchronous background LLM call to compress evicted turns into text.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Conversation DAG</strong><br><br>
        <em>Front:</em> How does a conversation DAG model message regeneration?<br><br>
        <em>Back:</em> By creating sibling nodes that share the identical <code>parent_id</code>.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Session TTL</strong><br><br>
        <em>Front:</em> What is the standard TTL strategy for active Redis chat caches?<br><br>
        <em>Back:</em> Sliding expiry (e.g. 60 min), refreshed on each user or assistant message.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical mastery of enterprise conversation systems:")

chk1 = st.checkbox("I understand why foundation LLMs are stateless and how conversational continuity is orchestrated.", value=True)
chk2 = st.checkbox("I can implement sliding token window algorithms to prevent context window overflow.", value=True)
chk3 = st.checkbox("I know how to design relational PostgreSQL schemas with parent_id pointers to support conversation branching.", value=True)
chk4 = st.checkbox("I can explain the difference between working memory, summary buffer memory, and long-term vector episodic memory.", value=True)
chk5 = st.checkbox("I understand how distributed locks (Redis Redlock) prevent concurrency race conditions in multi-turn dialogues.", value=True)
chk6 = st.checkbox("I know how to prevent tool-call / tool-response decoupling errors during conversation history pruning.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Conversation Systems! Proceed to Lesson 23.4: Production Features.")
