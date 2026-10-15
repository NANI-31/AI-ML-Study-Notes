import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="15.2 Long-Term Memory - LangGraph Curriculum",
    page_icon="🏛️",
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
    .badge-store {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-ns {
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
class LongTermMemoryStoreSimulator:
    """
    Simulates LangGraph Store (BaseStore):
    Hierarchical namespaces, user profiles, cross-session retrieval, and store.put().
    """
    def __init__(self):
        self.store: Dict[Tuple[str, ...], Dict[str, Dict[str, Any]]] = {
            ("users", "alice"): {
                "profile": {
                    "preferred_language": "Python",
                    "timezone": "PST",
                    "dietary_preference": "Vegetarian",
                    "updated_at": "2026-10-10"
                }
            },
            ("app", "company_policies"): {
                "travel_policy": {
                    "max_hotel_rate_usd": 250.0,
                    "per_diem_usd": 75.0
                }
            }
        }

    def put_memory(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]) -> str:
        if namespace not in self.store:
            self.store[namespace] = {}
        self.store[namespace][key] = value
        return f"Committed key '{key}' under namespace {namespace}"

    def get_memory(self, namespace: Tuple[str, ...], key: str) -> Optional[Dict[str, Any]]:
        return self.store.get(namespace, {}).get(key)


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The LangGraph Store Architecture and Cross-Session Memory", divider="rainbow")

st.markdown("""
While checkpointers persist state within a single thread, **Long-Term Memory** persists data **across all threads and sessions**.
LangGraph introduces the **`BaseStore` (LangGraph Store)** primitive.
The Store organizes information into **hierarchical namespaces** (such as `("users", user_id)` or `("org", org_id)`).
Nodes can query the store via `store.get()` and update memories via `store.put()`, enabling personalized,
continuous agent interactions across months of user engagement.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    ltm_blueprint_dot = """
    digraph LongTermMemoryArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CLIENT_CALL [label="Client Invocations (Distinct Threads)\\nThread A (Monday) | Thread B (Thursday)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_store {
            label = "LangGraph Long-Term Store (BaseStore)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            NS_USER [label="Namespace: ('users', 'alice')\\n- Key: 'profile' (Python, Vegetarian, PST)\\n- Key: 'past_topics' [FastAPI, LangGraph]", fillcolor="#064E3B", color="#34D399"];
            NS_APP [label="Namespace: ('app', 'global_config')\\n- Key: 'compliance_rules'\\n- Key: 'pricing_tiers'", fillcolor="#2E1065", color="#C084FC"];
        }

        subgraph cluster_nodes {
            label = "Agent Graph Nodes with Store Injected";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            NODE_LOAD [label="Node 1: Read Memory\\nprofile = store.get(('users', user_id), 'profile')\\nInjects profile into prompt context", fillcolor="#0F172A", color="#38BDF8"];
            NODE_REASON [label="Node 2: LLM Reasoning\\nGenerates personalized response grounded in memory", fillcolor="#1E1E38", color="#8B5CF6"];
            NODE_SAVE [label="Node 3: Write Memory\\nstore.put(('users', user_id), 'profile', updated_profile)", fillcolor="#431407", color="#FB923C"];

            NODE_LOAD -> NODE_REASON;
            NODE_REASON -> NODE_SAVE;
        }

        CLIENT_CALL -> NODE_LOAD;
        NS_USER -> NODE_LOAD [label="store.get()"];
        NODE_SAVE -> NS_USER [label="store.put()"];
    }
    """
    st.graphviz_chart(ltm_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Distinction:</b> Checkpointer = Short-term memory (scoped to <code>thread_id</code>).<br>
Store = Long-term memory (scoped to hierarchical namespaces like <code>("users", user_id)</code>, shared across threads).
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: CROSS-SESSION MEMORY
# ==============================================================================
st.header("2. Cross-Session Memory: Persisting Knowledge Beyond Single Conversation Threads", divider="rainbow")

st.markdown("""
Cross-session memory solves the amnesia problem across conversations:
- **Thread 1 (Day 1)**: User says: *"I only code in TypeScript, never write Python."*
- **Thread 2 (Day 15)**: User asks: *"Write a script to upload files to S3."*
Without long-term memory, the agent might generate Python. With cross-session memory, the agent
retrieves the user's preference and generates TypeScript automatically.
""")

# ==============================================================================
# SECTION 3: USER MEMORY
# ==============================================================================
st.header("3. User Memory: Personalization, User Profiles, and Behavioral Preferences", divider="rainbow")

st.markdown("""
User memory stores personalization profiles:
- **Explicit Facts**: Role, team, preferred frameworks, language, timezone.
- **Implicit Preferences**: Preferred answer verbosity, formatting style, domain familiarity.
""")

st.code("""
# Retrieving user profile inside a LangGraph node:
def personalized_agent_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    
    # Retrieve user profile from long-term store
    memory_item = store.get(("users", user_id), "profile")
    user_profile = memory_item.value if memory_item else {}
    
    # Prepend personalized instructions to system prompt
    pref = user_profile.get("preferred_language", "Python")
    system_msg = SystemMessage(content=f"User prefers coding examples in {pref}.")
    
    response = model.invoke([system_msg] + state["messages"])
    return {"messages": [response]}
""", language="python")

# ==============================================================================
# SECTION 4: APPLICATION MEMORY
# ==============================================================================
st.header("4. Application Memory: Shared Enterprise Knowledge and Global Organizational Cache", divider="rainbow")

st.markdown("""
Application memory stores knowledge shared across all users in the organization:
- Global compliance policies.
- Frequently asked solutions cached across customer service sessions.
- System feature flags and configuration settings.
""")

# ==============================================================================
# SECTION 5: NAMESPACE-BASED MEMORY
# ==============================================================================
st.header("5. Namespace-Based Memory: Hierarchical Namespaces and Isolation", divider="rainbow")

st.markdown("""
In LangGraph Store, keys are stored under a tuple of strings representing **namespaces**:
""")

ns_df = pd.DataFrame([
    {
        "Namespace Tuple": "('users', 'usr_8820')",
        "Key": "'profile'",
        "Scope": "Private to user 8820. Inaccessible to other users."
    },
    {
        "Namespace Tuple": "('org', 'acme_corp')",
        "Key": "'billing_rules'",
        "Scope": "Shared across all users belonging to Acme Corp tenant."
    },
    {
        "Namespace Tuple": "('app', 'global')",
        "Key": "'feature_toggles'",
        "Scope": "Global across entire LangGraph deployment."
    }
])
st.table(ns_df)

# ==============================================================================
# SECTION 6: MEMORY RETRIEVAL
# ==============================================================================
st.header("6. Memory Retrieval: Searching Stores with Keys and Semantic Vector Queries", divider="rainbow")

st.markdown("""
The LangGraph Store supports both **key-value retrieval** and **semantic vector search**:
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("#### Key-Value Retrieval (`get`)")
    st.code("""
# Fast deterministic lookup by key
item = store.get(("users", user_id), "profile")
print(item.value)
""", language="python")

with col_m2:
    st.markdown("#### Semantic Search (`search`)")
    st.code("""
# Semantic vector search across memories
matches = store.search(
    ("users", user_id),
    query="user travel preferences",
    limit=3
)
for match in matches:
    print(match.key, match.value, match.score)
""", language="python")

# ==============================================================================
# SECTION 7: MEMORY UPDATES
# ==============================================================================
st.header("7. Memory Updates: store.put(), In-Place Mutations, and Version Tracking", divider="rainbow")

st.markdown("""
Updating memory is accomplished with `store.put()`:
""")

st.code("""
# Save or update a memory item
store.put(
    namespace=("users", user_id),
    key="profile",
    value={
        "preferred_language": "TypeScript",
        "last_active": "2026-10-13"
    }
)

# Deleting a memory item
store.delete(namespace=("users", user_id), key="obsolete_key")
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Session Agent with LangGraph InMemoryStore", divider="rainbow")

st.markdown("""
Below is the complete implementation of an agent compiled with both a checkpointer (short-term)
and a store (long-term memory):
""")

st.code("""
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# 1. Instantiate Store and Checkpointer
checkpointer = MemorySaver()
store = InMemoryStore()

# Pre-populate store with user memory
store.put(("users", "usr_alice"), "profile", {"coding_language": "Rust", "role": "Senior Architect"})

# 2. Node with Store Access
def personalized_agent(state: MessagesState, *, store: BaseStore, config: dict) -> dict:
    user_id = config["configurable"]["user_id"]
    
    # Read long-term memory
    item = store.get(("users", user_id), "profile")
    user_lang = item.value.get("coding_language", "Python") if item else "Python"
    
    system_instruction = SystemMessage(
        content=f"You are an assistant. The user's preferred language is {user_lang}. Always provide code in this language."
    )
    
    # Formulate answer
    user_query = state["messages"][-1].content
    ai_reply = AIMessage(content=f"[{user_lang} Solution]: Here is how to implement '{user_query}' in {user_lang}...")
    return {"messages": [ai_reply]}

# 3. Compile Graph with Store
builder = StateGraph(MessagesState)
builder.add_node("agent", personalized_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# Injected both checkpointer (short-term) and store (long-term)
app = builder.compile(checkpointer=checkpointer, store=store)
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Cross-Thread User Profile & Long-Term Memory Store Sandbox", divider="rainbow")

st.markdown("""
Experiment with reading and writing long-term memories under hierarchical namespaces.
Verify that updates to user profiles persist across multiple simulated conversation threads.
""")

store_engine = LongTermMemoryStoreSimulator()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Store Memory Mutator")
    target_user = st.selectbox("Select Target User ID:", ["alice", "bob", "charlie"])
    pref_lang = st.selectbox("Preferred Framework / Language:", ["Python", "Rust", "Go", "TypeScript", "C++"])
    pref_tz = st.selectbox("User Timezone:", ["PST", "EST", "UTC", "JST", "CET"])
    
    commit_mem_btn = st.button("🚀 Commit to Long-Term Store", type="primary")

with wb_col2:
    st.subheader("🏛️ Live LangGraph Store Inspector")
    if commit_mem_btn:
        msg = store_engine.put_memory(
            ("users", target_user),
            "profile",
            {"preferred_language": pref_lang, "timezone": pref_tz, "updated_at": "2026-10-13"}
        )
        st.success(msg)

    st.markdown(f"#### Memory for User: `('{target_user}')`")
    profile_data = store_engine.get_memory(("users", target_user), "profile")
    if profile_data:
        st.json(profile_data)
    else:
        st.warning(f"No profile memory found for '{target_user}'. Click Commit to create one.")

    st.markdown("#### Global Application Memory: `('app', 'company_policies')`")
    st.json(store_engine.store.get(("app", "company_policies"), {}))

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the fundamental difference between LangGraph Checkpointers and LangGraph Store?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Checkpointer**: Manages short-term state scoped strictly to a single `thread_id`. Used for conversation multi-turn context and crash recovery.
    - **Store**: Manages long-term state across all threads and users, organized into hierarchical namespaces `(namespace, key)`. Used for user profiles, semantic memory, and global app settings.
    """)

with st.expander("Q2: How is the store accessed inside a graph node?", expanded=False):
    st.markdown("""
    **Answer**:
    By adding `*, store: BaseStore` as a keyword argument to the node function signature:
    `def my_node(state: MyState, config: RunnableConfig, *, store: BaseStore) -> dict: ...`.
    LangGraph automatically injects the compiled store instance at execution time.
    """)

with st.expander("Q3: How does namespace hierarchy prevent multi-tenant data leaks?", expanded=False):
    st.markdown("""
    **Answer**:
    By prepending the `tenant_id` to the namespace tuple: `("tenant_102", "users", user_id)`.
    Any query made on behalf of Tenant A specifies `("tenant_102", ...)` and cannot access records stored under `("tenant_103", ...)`.
    """)

with st.expander("Q4: Can the LangGraph Store perform semantic vector search?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! When compiled with an embedding provider (e.g. `index={"dims": 1536, "embed": embeddings}`),
    `store.search(namespace, query="...")` embeds the query and returns semantically relevant memory values with similarity scores.
    """)

with st.expander("Q5: What is the storage backend used for LangGraph Store in production?", expanded=False):
    st.markdown("""
    **Answer**:
    `PostgresStore` (backed by PostgreSQL with the `pgvector` extension for semantic search)
    is the standard production store implementation.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting to Compile the Graph with store=store")
    st.markdown("""
    *Mechanism*: Using `*, store: BaseStore` in a node while compiling with `builder.compile(checkpointer=checkpointer)`.  
    *Consequence*: Node invocation crashes with `ValueError: Node requested 'store', but no store was provided to compile()`.  
    *Fix*: Always pass `builder.compile(checkpointer=checkpointer, store=store)`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Flat Namespace Key Collisions")
    st.markdown("""
    *Mechanism*: Storing user memories under a single flat namespace `("profiles",)` with key `profile`.  
    *Consequence*: User B overwrites User A's profile on every write.  
    *Fix*: Always partition namespaces by user ID: `("users", user_id)`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Fetching Massive Memory Payloads")
    st.markdown("""
    *Mechanism*: Retrieving 50+ memory items and injecting them all into the system prompt.  
    *Consequence*: Inflates prompt token overhead and confuses model attention.  
    *Fix*: Use semantic search with `limit=3` to retrieve only relevant memories.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Storing PII Without Retention Policies")
    st.markdown("""
    *Mechanism*: Persisting un-redacted user financial or personal info permanently in long-term stores.  
    *Consequence*: Violates GDPR / CCPA right-to-be-forgotten regulations.  
    *Fix*: Provide explicit `store.delete()` endpoints for user data erasure.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**BaseStore**\\nThe LangGraph abstract interface for cross-thread, long-term memory storage.")
with col_fc2:
    st.info("**Hierarchical Namespace**\\nA tuple of strings (e.g. ('users', user_id)) partitioning memory items.")
with col_fc3:
    st.info("**store.put()**\\nThe method used to write or update a persistent memory item under a namespace and key.")
with col_fc4:
    st.info("**store.search()**\\nSemantic search method querying memories using vector embeddings.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural difference between Checkpointers and LangGraph Store.", value=True)
c2 = st.checkbox("I know how to define hierarchical namespaces to isolate user data.", value=True)
c3 = st.checkbox("I can access the store inside nodes using the *, store: BaseStore parameter.", value=True)
c4 = st.checkbox("I understand how to query memories using store.get() and store.search().", value=True)
c5 = st.checkbox("I know how to compile graphs with both a checkpointer and a store.", value=True)
c6 = st.checkbox("I understand how to enforce GDPR compliance using store.delete().", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 15.2 'Long-Term Memory' with full architectural and practical comprehension!")
