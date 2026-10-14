import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="38.2 Agent Memory - LangChain Systems",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agent memory telemetry
st.markdown("""
<style>
    .mem-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-short {
        background-color: #0284C7;
        color: #F0F9FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-long {
        background-color: #16A34A;
        color: #F0FDF4;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-episodic {
        background-color: #7C3AED;
        color: #F5F3FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-profile {
        background-color: #D97706;
        color: #FFFBEB;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class AgentMemorySimulator:
    """
    Simulates tiered multi-layer agent memory systems:
    Short-Term working context, Long-Term durable state,
    Semantic concept recall, Episodic trajectory retrieval,
    and Dynamic User Profile adaptation.
    """
    def __init__(self):
        self.preset_memory_contexts = {
            "Senior Cloud Architect (Enterprise Profile)": {
                "user_id": "usr_arch_9021",
                "short_term_buffer": [
                    {"role": "user", "content": "Let's review the Terraform configuration for the Kafka cluster."},
                    {"role": "assistant", "content": "Understood. Inspecting terraform/kafka-prod.tf for partition count and broker sizing."},
                    {"role": "user", "content": "Make sure we follow our company's disaster recovery standards."}
                ],
                "long_term_state": {
                    "active_project": "Global Event Streaming Architecture",
                    "cloud_provider": "AWS us-east-1 and eu-central-1",
                    "last_synced_commit": "git-sha-7f89c02"
                },
                "semantic_concepts": [
                    {"concept": "Multi-Region Active-Active DR", "relevance": 0.96, "axiom": "RPO must be under 5 seconds; RTO under 60 seconds using MSK mirror-maker."},
                    {"concept": "Zero-Trust VPC Peering", "relevance": 0.89, "axiom": "All cross-account broker connections must enforce mTLS with Vault PKI."}
                ],
                "episodic_trajectories": [
                    {"episode_id": "EP-3841", "scenario": "Kafka Partition Rebalance Outage (March 2026)", "learned_lesson": "Never increase partition count during business peak hours without throttling replica reassignment."}
                ],
                "user_profile": {
                    "technical_level": "Senior Principal Architect",
                    "preferred_language": "HCL / Terraform and Python",
                    "verbosity": "Terse, direct code snippets without high-level tutorials",
                    "strict_constraints": ["Zero AWS console click-ops", "All resources must include CostCenter and Owner tags"]
                }
            },
            "Junior Data Analyst (Growth Marketing)": {
                "user_id": "usr_mktg_4412",
                "short_term_buffer": [
                    {"role": "user", "content": "Can you help me write an SQL query to calculate customer churn?"},
                    {"role": "assistant", "content": "Certainly! Let's examine the subscriptions table schema."},
                    {"role": "user", "content": "What is the difference between voluntary and involuntary churn?"}
                ],
                "long_term_state": {
                    "active_project": "Q3 Churn Reduction Campaign",
                    "warehouse": "Snowflake Analytics DB",
                    "bi_tool": "Metabase"
                },
                "semantic_concepts": [
                    {"concept": "Cohort Retention Metrics", "relevance": 0.94, "axiom": "Churn rate = churned customers in period / active customers at start of period."},
                    {"concept": "Involuntary Churn Triggers", "relevance": 0.91, "axiom": "Payment gateway failure, expired credit cards, card processor outages."}
                ],
                "episodic_trajectories": [
                    {"episode_id": "EP-1102", "scenario": "Duplicate Metric Reporting (January 2026)", "learned_lesson": "Always deduplicate user IDs across multiple active trial subscriptions."}
                ],
                "user_profile": {
                    "technical_level": "Junior Analyst",
                    "preferred_language": "Standard SQL (Snowflake dialect)",
                    "verbosity": "Educational, structured explanations with inline SQL comments",
                    "strict_constraints": ["Provide sample data tables before showing final query"]
                }
            }
        }

    def simulate_memory_recall(self, context_key: str) -> Dict[str, Any]:
        return self.preset_memory_contexts.get(context_key, self.preset_memory_contexts["Senior Cloud Architect (Enterprise Profile)"])


# --- LESSON HEADER ---
st.title("38.2 Agent Memory: Short-Term, Long-Term, Semantic & Episodic Systems")
st.markdown("""
A foundation model has no intrinsic memory between execution turns: each API invocation begins with a clean slate.
For autonomous agents to maintain coherent multi-turn reasoning, retain cross-session user preferences, and learn
from past execution trajectories, engineers build **Cognitive Memory Architectures**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Cognitive Memory Architecture for Autonomous Agents", divider="rainbow")
st.markdown("""
An enterprise agent memory architecture stratifies information into four complementary storage tiers:
1. **Short-Term (Working Memory)**: Fast, in-context state containing the active conversation thread and tool scratchpads.
2. **Long-Term (Persistent State)**: Durable external checkpointers (Postgres, Redis) storing thread histories and session state across restarts.
3. **Semantic Memory (Factual & Conceptual)**: Vector databases indexing domain knowledge, guidelines, and corporate axioms.
4. **Episodic Memory (Experiential Trajectories)**: Historical record of past solved problems, failures, and reflection critiques.
5. **User Profile (Personalization)**: Continuously updated structured persona detailing user preferences, skill level, and constraints.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AgentMemory_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_short_term {
            label = "Short-Term Working Memory (In-Context)";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            thread_state [label="Thread State Buffer\\n[System, Human, AI, ToolMessages]", fillcolor="#0F172A"];
            context_trim [label="Context Window Manager\\nSliding Window / Summary Trimmer", fillcolor="#1E3A8A"];
        }

        subgraph cluster_long_term {
            label = "Long-Term Durable State (Cross-Session)";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            checkpointer [label="LangGraph Checkpointer\\nPostgresSaver / SqliteSaver / Redis", fillcolor="#047857"];
            durable_db   [label="Durable State DB\\nThread Snapshots & Time-Travel Diffs", fillcolor="#059669"];
        }

        subgraph cluster_semantic_episodic {
            label = "External Cognitive Stores (Semantic & Episodic)";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            vector_store [label="Semantic Memory Store\\nConcept & Entity Embeddings", fillcolor="#7C3AED"];
            episodic_buf [label="Episodic Trajectory Buffer\\nPast Tasks, Failure Reflections & Wins", fillcolor="#6D28D9"];
            user_profile [label="User Profile Persona Store\\nStructured Preferences & Constraints", fillcolor="#D97706"];
        }

        subgraph cluster_agent {
            label = "Agent Context Assembly & Reasoning Engine";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            context_assembler [label="Unified Prompt Synthesizer\\nSystem + Profile + Semantic + Working Memory", fillcolor="#0369A1"];
            agent_core        [label="Frontier LLM Reasoning Node\\nGPT-4o / Claude 3.5 Sonnet", fillcolor="#0284C7"];
        }

        thread_state -> context_trim;
        context_trim -> checkpointer [label="Persist State"];
        checkpointer -> durable_db;

        durable_db   -> context_assembler [label="Hydrate Prior Session"];
        vector_store -> context_assembler [label="Relevant Domain Axioms"];
        episodic_buf -> context_assembler [label="Similar Past Trajectories"];
        user_profile -> context_assembler [label="User Preferences & Style"];

        context_assembler -> agent_core;
        agent_core        -> thread_state [label="Emit New Tool/Message Event"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)


# --- SECTION 2: SHORT-TERM MEMORY ---
st.header("2. Short-Term Memory: Thread State, Checkpoints and Working Memory", divider="rainbow")
st.markdown("""
**Short-Term Memory** represents the working memory accessible within the model's active context window during an ongoing execution thread.
In LangChain and LangGraph, short-term memory is managed via typed state channels and message reducers (`add_messages`).
""")

col_st1, col_st2 = st.columns(2)
with col_st1:
    st.subheader("Working Memory Pruning Tactics")
    st.markdown("""
    - **Sliding Window Trimming**: Preserves only the most recent $N$ messages, discarding older messages while retaining the permanent system prompt.
    - **Conversation Summary Memory**: Uses a lightweight secondary LLM call to progressively summarize past conversation turns into a running abstract.
    - **Tool Scratchpad Pruning**: Strips large tool observation payloads (such as raw JSON dumps or HTML blobs) once the model has synthesized a summary.
    """)

with col_st2:
    st.subheader("Sliding Window Trimming Code")
    st.code("""
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Trim messages to fit within a strict 4,000 token working memory
selected_messages = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human"
)
    """, language="python")


# --- SECTION 3: LONG-TERM MEMORY ---
st.header("3. Long-Term Memory: Persistent Cross-Session State and Storage Backends", divider="rainbow")
st.markdown("""
**Long-Term Memory** persists agent state across server restarts, worker crashes, and multi-day user sessions.
In LangGraph, durable long-term persistence is implemented via **Checkpointers** that save snapshots of thread state at every graph super-step.
""")

col_lt1, col_lt2 = st.columns(2)
with col_lt1:
    st.markdown("### Production Storage Backends")
    lt_table = {
        "Storage Engine": ["MemorySaver", "SqliteSaver", "PostgresSaver", "RedisSaver"],
        "Durability": ["In-Memory (Ephemeral)", "Local File (Durable)", "Enterprise Distributed", "Ultra-Low Latency (RAM)"],
        "Multi-Tenancy": ["No", "Single Server", "Full Multi-Tenant", "Full Multi-Tenant"],
        "Production Readiness": ["Development / Unit Tests", "Edge / Local CLI", "Recommended Enterprise Default", "High-Throughput Caching"]
    }
    st.dataframe(pd.DataFrame(lt_table), use_container_width=True, hide_index=True)

with col_lt2:
    st.markdown("### Persistent Checkpointer Setup")
    st.code("""
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph

# Enterprise PostgreSQL checkpointer
DB_URI = "postgresql://agent_user:secret@pg-cluster.corp:5432/agent_state"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # Auto-creates checkpoint tables
    
    # Compile graph with durable persistence
    app = workflow.compile(checkpointer=checkpointer)
    
    # Resume any thread across sessions
    config = {"configurable": {"thread_id": "thread_user_102"}}
    state = app.get_state(config)
    """, language="python")


# --- SECTION 4: SEMANTIC MEMORY ---
st.header("4. Semantic Memory: Concept Vectors, Knowledge Graphs and VectorDB Retrieval", divider="rainbow")
st.markdown("""
**Semantic Memory** stores structured and unstructured factual knowledge about the world, corporate domain rules,
and technical terminology. Unlike short-term memory, semantic memory is shared across all agent threads.
""")

col_sem1, col_sem2 = st.columns(2)
with col_sem1:
    st.markdown("""
    ### Semantic Memory Ingestion & Querying
    - **Entity & Concept Embeddings**: Text passages are chunked and converted into dense vector embeddings (e.g. OpenAI `text-embedding-3-small`).
    - **Knowledge Graph Triples**: Representing relational facts as subject-predicate-object triples (`(PaymentGateway, enforces, 500msTimeout)`).
    - **Hybrid Dense + Sparse Search**: Combining BM25 keyword matching with dense cosine similarity to retrieve exact identifiers and concepts.
    """)

with col_sem2:
    st.markdown("### Semantic Memory Retrieval Pattern")
    st.code("""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_store = Chroma(collection_name="enterprise_axioms", embedding_function=embeddings)

def retrieve_semantic_context(user_query: str, top_k: int = 3) -> str:
    docs = semantic_store.similarity_search(user_query, k=top_k)
    return "\\n".join([f"- {d.page_content}" for d in docs])
    """, language="python")


# --- SECTION 5: EPISODIC MEMORY ---
st.header("5. Episodic Memory: Past Trajectory Experience and Few-Shot Reflection Recall", divider="rainbow")
st.markdown("""
Humans solve new problems by recalling how they resolved similar situations in the past.
**Episodic Memory** stores complete execution trajectories—including the initial goal, steps taken, tool errors encountered,
and final reflection critiques.
""")

col_ep1, col_ep2 = st.columns(2)
with col_ep1:
    st.markdown("""
    ### Dynamic Few-Shot Exemplar Injection
    1. When an agent receives a complex objective, it queries its episodic memory using the objective text.
    2. The store retrieves the top 2-3 most similar past trajectories that resulted in successful completions.
    3. These successful trajectories are formatted as few-shot in-context demonstrations, guiding the model toward proven action sequences.
    """)

with col_ep2:
    st.markdown("### Episodic Record Schema (Pydantic)")
    st.code("""
class AgentEpisode(BaseModel):
    episode_id: str
    goal: str
    successful: bool
    trajectory_summary: list[str]
    reflection_notes: str
    tools_used: list[str]

class EpisodicMemoryManager:
    def __init__(self, vectorstore):
        self.store = vectorstore

    def record_episode(self, episode: AgentEpisode):
        doc_text = f"Goal: {episode.goal}\\nOutcome: {episode.reflection_notes}"
        self.store.add_texts(texts=[doc_text], metadatas=[episode.model_dump()])

    def retrieve_similar_episodes(self, current_goal: str, k: int = 2):
        return self.store.similarity_search(current_goal, k=k)
    """, language="python")


# --- SECTION 6: USER PROFILES ---
st.header("6. User Profiles: Preference Tracking, Personalization and Dynamic Persona Stores", divider="rainbow")
st.markdown("""
Autonomous agents achieve higher user satisfaction when they adapt to the user's communication style, technical expertise,
and operational constraints over time. **User Profile Memory** maintains a structured profile that is dynamically updated
after every interaction.
""")

col_prof1, col_prof2 = st.columns(2)
with col_prof1:
    st.markdown("""
    ### Profile Evolution Mechanics
    - **Implicit Extraction**: After each conversation turn, an asynchronous background task analyzes the dialogue for revealed preferences (e.g. *"I only work in Python, don't show JavaScript"*).
    - **Pydantic Validation**: Extracted preferences are merged into a typed schema with confidence scores and last-updated timestamps.
    - **Prompt Conditioning**: The user's active profile is injected directly into the agent's top-level system prompt.
    """)

with col_prof2:
    st.markdown("### User Profile Schema & Updater")
    st.code("""
class UserProfile(BaseModel):
    user_id: str
    expertise_level: str  # 'Beginner', 'Intermediate', 'Expert'
    preferred_stack: list[str]
    formatting_preferences: str
    negative_constraints: list[str] = Field(description="Things the user explicitly dislikes")

def inject_profile_into_system_prompt(profile: UserProfile) -> str:
    return f\"\"\"
    User Profile:
    - Technical Level: {profile.expertise_level}
    - Preferred Technologies: {', '.join(profile.preferred_stack)}
    - Style: {profile.formatting_preferences}
    - Strict Rules: {'; '.join(profile.negative_constraints)}
    \"\"\"
    """, language="python")


# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Tiered Multi-Layer Memory Engine", divider="rainbow")
st.markdown("""
The following production class integrates Short-Term working memory, Long-Term state checkpoints,
Semantic retrieval, and User Profile conditioning into a unified agent memory manager.
""")

st.code("""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, trim_messages
from langchain_openai import ChatOpenAI

class EnterpriseMultiLayerMemory:
    def __init__(self, model_name: str = "gpt-4o", max_working_tokens: int = 4000):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.max_tokens = max_working_tokens
        
    def assemble_context(
        self,
        user_id: str,
        user_query: str,
        working_messages: List[BaseMessage],
        user_profile: Dict[str, Any],
        semantic_axioms: List[str],
        episodic_exemplars: List[str]
    ) -> List[BaseMessage]:
        \"\"\"
        Assembles all four memory tiers into a coherent, token-managed prompt.
        \"\"\"
        # 1. System Prompt with Profile & Constraints
        profile_block = (
            f"User Expertise: {user_profile.get('technical_level', 'General')}\\n"
            f"Preferred Style: {user_profile.get('verbosity', 'Concise')}\\n"
            f"Constraints: {'; '.join(user_profile.get('strict_constraints', []))}"
        )
        
        # 2. Semantic Memory Block
        semantic_block = "Enterprise Domain Knowledge:\\n" + "\\n".join([f"- {a}" for a in semantic_axioms])
        
        # 3. Episodic Exemplars Block
        episodic_block = "Past Successful Trajectories:\\n" + "\\n".join([f"- {e}" for e in episodic_exemplars])
        
        system_text = (
            f"You are an enterprise AI assistant.\\n\\n"
            f"{profile_block}\\n\\n"
            f"{semantic_block}\\n\\n"
            f"{episodic_block}"
        )
        
        assembled: List[BaseMessage] = [SystemMessage(content=system_text)]
        
        # 4. Short-Term Working Memory (trimmed to token budget)
        trimmed_history = trim_messages(
            working_messages,
            max_tokens=self.max_tokens,
            strategy="last",
            token_counter=self.llm,
            include_system=False,
            start_on="human"
        )
        assembled.extend(trimmed_history)
        assembled.append(HumanMessage(content=user_query))
        
        return assembled
""", language="python")


# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Tier Agent Memory Simulator", divider="rainbow")
st.markdown("""
Inspect how multi-tier agent memory systems synthesize context across short-term working buffers,
long-term durable checkpoints, semantic domain axioms, and dynamic user profiles.
""")

mem_sim = AgentMemorySimulator()

col_m_cfg, col_m_res = st.columns([1, 1])

with col_m_cfg:
    st.subheader("Select Persona & Memory Context")
    selected_context = st.selectbox(
        "Choose User Memory Persona:",
        list(mem_sim.preset_memory_contexts.keys()),
        index=0
    )
    
    mem_data = mem_sim.preset_memory_contexts[selected_context]
    st.markdown(f"**User ID**: `{mem_data['user_id']}`")
    st.markdown(f"**Technical Level**: `{mem_data['user_profile']['technical_level']}`")
    
    inspect_btn = st.button("🔍 Inspect Memory Tiers & Recall", use_container_width=True)

with col_m_res:
    st.subheader("Memory Tier Inspection Telemetry")
    if inspect_btn:
        res = mem_sim.simulate_memory_recall(selected_context)
        
        st.markdown(
            f"<div class='mem-card'>"
            f"<b>Active Persona:</b> <span class='badge-profile'>{res['user_profile']['technical_level']}</span><br><br>"
            f"<b>Active Project:</b> {res['long_term_state']['active_project']}<br>"
            f"<b>Style Constraints:</b> {'; '.join(res['user_profile']['strict_constraints'])}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("### 1. Short-Term Working Context (Recent Turns)")
        for msg in res["short_term_buffer"]:
            role_badge = "badge-short" if msg["role"] == "user" else "badge-long"
            st.markdown(f"<span class='{role_badge}'>{msg['role'].upper()}</span> {msg['content']}", unsafe_allow_html=True)
            
        st.markdown("### 2. Semantic Memory (Retrieved Axioms)")
        for ax in res["semantic_concepts"]:
            st.markdown(f"- 📌 **{ax['concept']}** (Relevance: `{ax['relevance']}`): {ax['axiom']}")
            
        st.markdown("### 3. Episodic Memory (Recalled Trajectories)")
        for ep in res["episodic_trajectories"]:
            st.markdown(f"- 🧠 **{ep['episode_id']} ({ep['scenario']})**: *{ep['learned_lesson']}*")
            
        st.success("All 4 memory layers successfully retrieved and formatted for model invocation.")
    else:
        st.info("Select a user memory persona and click Inspect to view the assembled memory tiers.")


# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does LangGraph checkpointer persistence differ from simple database message logging?"):
    st.markdown("""
    **Answer**: Message logging merely records input and output strings. A LangGraph Checkpointer (`PostgresSaver`) captures
    a complete, serialized snapshot of the graph's internal state—including intermediate channel variables, pending tool calls,
    active thread forks, and node execution indices. This enables deterministic crash recovery, human-in-the-loop interrupts,
    and time-travel state rollbacks.
    """)

with st.expander("Q2: What is the primary operational risk of storing unbounded conversation history in short-term memory?"):
    st.markdown("""
    **Answer**: Unbounded message accumulation causes context window exhaustion, latency inflation, and token cost spikes.
    Furthermore, as context length grows, models suffer from the 'lost in the middle' phenomenon, where the model's attention
    fails to prioritize instructions buried inside dozens of past conversational turns.
    """)

with st.expander("Q3: How do you prevent outdated or contradictory user preferences from polluting User Profile memory?"):
    st.markdown("""
    **Answer**: Implement a structured profile manager with:
    1. Timestamped preference entries.
    2. Explicit confidence scores.
    3. Conflict resolution logic: when a new preference directly contradicts an old one (e.g. 'Use Python' vs 'Use Go'),
       the newer preference supersedes the older entry.
    4. Periodic decay of low-confidence inferences.
    """)

with st.expander("Q4: How does Episodic Memory differ from Semantic Memory?"):
    st.markdown("""
    **Answer**: Semantic Memory stores general, timeless facts and conceptual knowledge (e.g. 'Kafka uses partitions for parallelism').
    Episodic Memory stores specific autobiographical experiences tied to time and context (e.g. 'When I increased partition count on March 14,
    the cluster rebalance timed out, so I had to throttle replica bandwidth').
    """)

with st.expander("Q5: What is the recommended strategy for pruning large tool observation outputs from working memory?"):
    st.markdown("""
    **Answer**: Once an agent has generated an answer based on a large tool observation (e.g. a 50KB JSON document from a database query),
    the raw tool message content can be replaced with a concise summary in the message history, or the raw payload can be offloaded
    to external blob storage with only its reference ID retained in the active context.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (In-Memory Checkpointer in Production)**: Using `MemorySaver` in multi-instance or serverless environments. When a container restarts, all active threads and human-approval states are permanently destroyed.")
    st.warning("**Trap 2 (Blind Message Truncation)**: Truncating message lists from the middle, separating a `ToolMessage` from its preceding `AIMessage` with tool calls. This violates API schemas and triggers hard 400 Bad Request errors.")
with col_trap2:
    st.error("**Trap 3 (Injecting Hallucinated User Preferences)**: Updating user profiles on single conversational flukes without confidence thresholds, permanently trapping the user in an unwanted persona.")
    st.warning("**Trap 4 (Oversized Semantic Injection)**: Injecting dozens of retrieved vector chunks into the prompt, diluting model focus and crowding out user instructions.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What does a LangGraph Checkpointer store?")
        st.caption("Flip for answer:")
        st.info("**Back**: Complete serialized snapshots of graph state at every super-step, enabling crash recovery and time-travel.")
    with st.container(border=True):
        st.markdown("**Front**: What is Episodic Memory in autonomous agents?")
        st.caption("Flip for answer:")
        st.info("**Back**: Experiential memory storing past problem-solving trajectories and lessons learned.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Which message trimming strategy preserves the permanent system prompt?")
        st.caption("Flip for answer:")
        st.info("**Back**: `trim_messages(..., include_system=True, strategy='last')`.")
    with st.container(border=True):
        st.markdown("**Front**: What type of memory stores timeless domain facts and axioms?")
        st.caption("Flip for answer:")
        st.info("**Back**: Semantic Memory (typically backed by Vector Databases or Knowledge Graphs).")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural differences between Short-Term, Long-Term, Semantic, and Episodic memory.", value=True)
c2 = st.checkbox("I know how to manage working context limits using LangChain's trim_messages utility.", value=True)
c3 = st.checkbox("I understand how to configure enterprise checkpointers (PostgresSaver) in LangGraph.", value=True)
c4 = st.checkbox("I can design episodic memory retrieval to provide few-shot demonstrations to agents.", value=True)
c5 = st.checkbox("I know how to extract and update structured User Profiles without introducing hallucinations.", value=True)
c6 = st.checkbox("I understand how to prune large tool observations from context without corrupting tool call schemas.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 38.3: Agent Reliability.")
