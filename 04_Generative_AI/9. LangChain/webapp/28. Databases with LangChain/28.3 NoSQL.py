import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="28.3 NoSQL Databases - LangChain Systems",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NoSQL telemetry
st.markdown("""
<style>
    .nosql-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-nosql {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class NoSQLDatabaseSimulator:
    """
    Simulates NoSQL document querying (MongoDB JSON documents)
    and high-throughput key-value lookups (Redis hashes and caching).
    """
    def __init__(self):
        self.mock_mongo_docs = [
            {"_id": "usr_9912", "profile": {"name": "Alice Chen", "role": "VP Operations"}, "settings": {"notifications": True, "theme": "dark"}, "tags": ["enterprise", "active"]},
            {"_id": "usr_8831", "profile": {"name": "Bob Smith", "role": "Lead Architect"}, "settings": {"notifications": False, "theme": "system"}, "tags": ["developer", "trial"]}
        ]
        self.mock_redis_kv = {
            "session:usr_9912": {"active_thread": "thr_alpha", "ttl_seconds": 3600, "token_count": 450},
            "cache:prompt:09a1": {"query": "What is our refund SLA?", "cached_response": "Standard refund SLA is 48 business hours.", "hits": 24}
        }

    def query_mongo(self, filter_tag: str) -> List[Dict[str, Any]]:
        return [doc for doc in self.mock_mongo_docs if filter_tag.lower() in [t.lower() for t in doc["tags"]]]

    def query_redis(self, key: str) -> Optional[Dict[str, Any]]:
        return self.mock_redis_kv.get(key, None)


# --- LESSON HEADER ---
st.title("28.3 NoSQL Databases with LangChain: MongoDB, Redis & Document Stores")
st.markdown("""
While relational SQL databases excel at rigid tabular schemas, modern AI architectures rely on NoSQL databases
for semi-structured documents, hierarchical JSON data, dynamic metadata, and microsecond key-value caching.
This lesson covers connecting LangChain to document databases (MongoDB) and in-memory key-value stores (Redis)
for message history, semantic caching, and dynamic document querying.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: NoSQL Multi-Model Integration", divider="rainbow")
st.markdown("""
NoSQL stores fulfill two complementary roles in LangChain ecosystems:
1. **Document Stores (MongoDB)**: Persist nested, polymorphic JSON documents, dynamic user profiles, and chat message histories.
2. **Key-Value Stores (Redis)**: Serve as high-throughput, low-latency caches for exact query matches, semantic vector indices, and ephemeral session checkpoints.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph NoSQL_Databases_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_app_layer {
            label = "LangChain / LangGraph Application Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            AgentCore [label="Agent / LCEL Chain Core\\n(Orchestration & Decision Logic)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_kv_layer {
            label = "Redis In-Memory Key-Value & Caching Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            RedisCache [label="Redis Cache (Exact & Semantic)\\n(Microsecond query response, TTL expiration)", fillcolor="#1E293B", color="#60A5FA"];
            RedisCheckpoint [label="RedisSaver / Session Checkpoints\\n(Thread session state & ephemeral memory)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_doc_layer {
            label = "MongoDB Document Storage Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            MongoChatMessageHistory [label="MongoDBChatMessageHistory\\n(Nested JSON conversational history)", fillcolor="#0369A1", color="#38BDF8"];
            MongoDocStore [label="MongoDB Document Store\\n(Polymorphic document catalogs & metadata)", fillcolor="#0369A1", color="#38BDF8"];
            MongoVectorSearch [label="MongoDB Atlas Vector Search\\n(Hybrid vector & document filter indexing)", fillcolor="#0369A1", color="#38BDF8"];
        }

        AgentCore -> RedisCache [label="1. Check cache (O(1))"];
        AgentCore -> RedisCheckpoint [label="2. Load active thread state"];
        AgentCore -> MongoChatMessageHistory [label="3. Append conversation turn"];
        AgentCore -> MongoDocStore [label="4. Query polymorphic documents"];
        AgentCore -> MongoVectorSearch [label="5. Hybrid vector search"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: MONGODB ---
st.header("2. MongoDB Integration: JSON Document Storage, Aggregation Pipelines & BSON", divider="rainbow")
st.markdown("""
MongoDB stores data as BSON (Binary JSON) documents. For LangChain applications, MongoDB is ideal for storing
unstructured document metadata, customer interaction histories, and conversation logs.
LangChain integrates natively via `MongoDBChatMessageHistory` and MongoDB Atlas Vector Search.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### Why MongoDB for GenAI?")
    st.markdown("""
    - **Schema Flexibility**: Documents can have varied structures (e.g. metadata with varying custom fields per tenant).
    - **Native JSON Support**: LLM tool calls and structured outputs serialize directly into MongoDB collections without ORM conversions.
    - **Atlas Vector Search**: MongoDB Atlas combines full-text search, relational-style aggregations, and vector embeddings in a single cluster.
    """)
with col_m2:
    st.code("""
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

# Persistent conversation history in MongoDB
history = MongoDBChatMessageHistory(
    connection_string="mongodb://localhost:27017/",
    session_id="session_usr_9912",
    database_name="ai_chat_db",
    collection_name="conversations"
)

# Appends message documents to MongoDB
history.add_user_message("Hello, tell me about my account.")
history.add_ai_message("Your enterprise account is active.")
    """, language="python")

# --- SECTION 3: REDIS ---
st.header("3. Redis Integration: In-Memory Key-Value Storage, Semantic Cache & Session State", divider="rainbow")
st.markdown("""
Redis provides sub-millisecond in-memory data structures (strings, hashes, sets, vectors).
In LangChain, Redis is primarily deployed for:
1. **LLM Response Caching**: Storing prompt-completion pairs to eliminate duplicate API calls.
2. **LangGraph Checkpointing**: Persisting session thread states with automatic TTL expiration.
3. **Real-time Semantic Caching**: Using RediSearch to match queries via cosine similarity.
""")

st.code("""
import redis
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache

# Connect Redis client
redis_client = redis.Redis.from_url("redis://localhost:6379/0")

# Enable global Redis cache in LangChain
set_llm_cache(RedisCache(redis_=redis_client))

# Subsequent identical prompts return from Redis in < 1ms
# with zero LLM provider API charges
""", language="python")

# --- SECTION 4: DOCUMENT DATABASES ---
st.header("4. Document Databases: Schema-Flexible Ingestion & Nested Object Traversals", divider="rainbow")
st.markdown("""
When building agents that query document databases, the agent synthesizes **MongoDB Query Language (MQL)**
rather than SQL. MQL queries use JSON filter objects (`{ "status": "active", "balance": { "$gt": 1000 } }`).
Specialized tools allow agents to execute find, aggregate, and projection operations safely.
""")

st.code("""
from langchain_core.tools import tool
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["enterprise_crm"]

@tool
def query_customer_documents(filter_json: str) -> str:
    \"\"\"Queries customer JSON documents using a valid MongoDB filter string.\"\"\"
    import json
    filter_dict = json.loads(filter_json)
    results = list(db["customers"].find(filter_dict, {"_id": 0}).limit(5))
    return json.dumps(results)
""", language="python")

# --- SECTION 5: KEY-VALUE STORES ---
st.header("5. Key-Value Stores: High-Throughput Lookup Patterns and Time-To-Live (TTL)", divider="rainbow")
st.markdown("""
Key-value stores excel at $O(1)$ key lookups. In production architectures, key-value stores manage:
- **Rate-Limiting Counters**: Tracking tokens consumed per tenant per minute.
- **Session State**: Keeping user authorization tokens and active agent thread IDs with Time-To-Live (TTL) expiration.
- **Short-Term Memory**: Storing conversation state buffers that automatically purge after 24 hours of inactivity.
""")

col_kv1, col_kv2 = st.columns(2)
with col_kv1:
    st.markdown("### Key-Value Design Patterns")
    st.markdown("""
    - **Namespaced Keys**: Structure keys hierarchically (e.g. `tenant:101:user:9912:session`).
    - **TTL Expiration**: Always attach TTL timers to ephemeral data to prevent unbounded RAM growth.
    - **Atomic Increments**: Use Redis `INCR` for concurrency-safe token rate limiting.
    """)
with col_kv2:
    st.code("""
# High-speed session store pattern
def save_user_session(redis_conn, user_id: str, thread_id: str):
    key = f"user:{user_id}:active_thread"
    # Store thread ID with 24-hour expiration
    redis_conn.setex(key, 86400, thread_id)

def get_user_session(redis_conn, user_id: str) -> str:
    key = f"user:{user_id}:active_thread"
    val = redis_conn.get(key)
    return val.decode("utf-8") if val else None
    """, language="python")

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Hybrid Document & Key-Value Persistence Layer", divider="rainbow")
st.markdown("""
Below is an enterprise persistence bridge uniting Redis for microsecond caching and session TTLs
with MongoDB for durable document message histories.
""")

st.code("""
# hybrid_nosql_bridge.py
from typing import Dict, Any, Optional
import redis
import pymongo
from langchain_core.messages import HumanMessage, AIMessage

class EnterpriseNoSQLBridge:
    def __init__(self, redis_url: str, mongo_url: str):
        self.redis = redis.Redis.from_url(redis_url)
        self.mongo_client = pymongo.MongoClient(mongo_url)
        self.chat_coll = self.mongo_client["ai_production"]["message_logs"]

    def cache_get(self, prompt_hash: str) -> Optional[str]:
        val = self.redis.get(f"cache:{prompt_hash}")
        return val.decode("utf-8") if val else None

    def cache_set(self, prompt_hash: str, completion: str, ttl_seconds: int = 3600):
        self.redis.setex(f"cache:{prompt_hash}", ttl_seconds, completion)

    def persist_turn(self, session_id: str, user_prompt: str, ai_response: str):
        document = {
            "session_id": session_id,
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            "timestamp": pymongo.helpers.datetime.utcnow()
        }
        self.chat_coll.insert_one(document)
""", language="python")

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: MongoDB Document & Redis Key-Value Simulator", divider="rainbow")
st.markdown("""
Simulate NoSQL interactions: query nested MongoDB JSON documents by tag filters,
and perform microsecond Redis key lookups with TTL tracking.
""")

nosql_sim = NoSQLDatabaseSimulator()

col_m, col_r = st.columns([1, 1])

with col_m:
    st.subheader("🍃 MongoDB Document Query")
    chosen_tag = st.selectbox("Filter Documents by Tag:", ["enterprise", "developer", "active", "trial"])
    run_mongo = st.button("🚀 Query MongoDB Collection", type="primary")
    
    if run_mongo:
        docs = nosql_sim.query_mongo(chosen_tag)
        st.success(f"Matched {len(docs)} BSON Document(s):")
        for d in docs:
            with st.container(border=True):
                st.json(d)

with col_r:
    st.subheader("🔴 Redis Key-Value Store")
    selected_key = st.selectbox("Select Redis Key:", list(nosql_sim.mock_redis_kv.keys()))
    run_redis = st.button("⚡ Execute O(1) Key Lookup", type="secondary")
    
    if run_redis:
        val = nosql_sim.query_redis(selected_key)
        st.markdown(f"<div class='nosql-card'><span class='badge-nosql'>REDIS HIT (0.4ms)</span></div>", unsafe_allow_html=True)
        st.json(val)

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: When should an enterprise use MongoDB over a relational SQL database for LangChain message history?"):
    st.markdown("""
    **Answer**: MongoDB is preferred for message histories when conversation turns feature polymorphic payloads
    (e.g. structured tool call outputs, varying metadata, image URLs, and dynamic schema evolutions) that do not fit neatly
    into rigid tabular rows. MongoDB stores each turn as a flexible JSON/BSON document, avoiding complex relational joins.
    """)

with st.expander("Q2: How does Redis accelerate LangChain response times?"):
    st.markdown("""
    **Answer**: Redis operates entirely in RAM with $O(1)$ lookup performance. By serving as an exact or semantic cache for LLM completions,
    Redis intercepts repeated queries and returns responses in sub-millisecond latency (vs 1-3 seconds for cloud LLMs),
    completely eliminating API token charges for cached queries.
    """)

with st.expander("Q3: What is the purpose of Time-To-Live (TTL) in Redis session management?"):
    st.markdown("""
    **Answer**: Because RAM is expensive and finite, storing session histories indefinitely leads to out-of-memory crashes.
    Attaching TTL expiration (e.g. `SETEX key 86400 val`) automatically cleans up stale or abandoned conversation sessions,
    freeing memory for active users.
    """)

with st.expander("Q4: How does MongoDB Atlas Vector Search integrate with standard document queries?"):
    st.markdown("""
    **Answer**: MongoDB Atlas Vector Search allows pre-filtering and post-filtering vector searches using native MongoDB MQL queries.
    This enables hybrid searches where the vector index searches embeddings while MongoDB MQL simultaneously restricts results
    to documents matching `{ "tenant_id": 101, "is_active": true }` in a single query execution.
    """)

with st.expander("Q5: What are the security risks of allowing agents to generate MongoDB queries?"):
    st.markdown("""
    **Answer**: Attackers can attempt NoSQL injection (e.g. passing `{ "$where": "sleep(5000)" }` or `$regex` DOS attacks).
    Tool inputs must sanitize JSON filters and restrict queries to safe operators (`$eq`, `$in`, `$gt`), blocking arbitrary JavaScript execution (`$where`).
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Unbounded Redis RAM Usage)**: Storing caching keys without setting TTL expiration timers. Redis RAM steadily grows until the instance exhausts memory and crashes.")
    st.warning("**Trap 2 (Allowing $where in MongoDB Queries)**: Permitting arbitrary JavaScript evaluation via `$where` in LLM-generated MongoDB filters, exposing the database to code execution.")
with col_trap2:
    st.error("**Trap 3 (Treating Redis as Durable Primary Storage)**: Relying solely on in-memory Redis without persistence backups (AOF/RDB) for critical business data; server restarts wipe all records.")
    st.warning("**Trap 4 (Storing Massive Embeddings in Chat History)**: Serializing high-dimensional embedding vectors directly into chat history documents, bloating document sizes and slowing queries.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What class in LangChain stores chat history in MongoDB?")
        st.caption("Flip for answer:")
        st.info("**Back**: `MongoDBChatMessageHistory` from `langchain_community.chat_message_histories`.")
    with st.container(border=True):
        st.markdown("**Front**: What data structure does MongoDB use to store documents?")
        st.caption("Flip for answer:")
        st.info("**Back**: BSON (Binary JSON), supporting nested objects, arrays, and typed fields.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the lookup time complexity of Redis exact caching?")
        st.caption("Flip for answer:")
        st.info("**Back**: $O(1)$ constant time lookup in RAM.")
    with st.container(border=True):
        st.markdown("**Front**: Why is TTL important in Redis session management?")
        st.caption("Flip for answer:")
        st.info("**Back**: It automatically expires and purges abandoned session keys to prevent memory leaks.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural roles of document stores vs key-value stores in GenAI.", value=True)
c2 = st.checkbox("I can persist conversation history in MongoDB using MongoDBChatMessageHistory.", value=True)
c3 = st.checkbox("I know how to configure RedisCache to intercept repeated queries with sub-millisecond latency.", value=True)
c4 = st.checkbox("I understand how to structure MongoDB filters (MQL) for dynamic document querying.", value=True)
c5 = st.checkbox("I can manage session state with TTL expiration in Redis to prevent memory leaks.", value=True)
c6 = st.checkbox("I know how to secure MongoDB tools against NoSQL injection and arbitrary JavaScript execution.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 28.4: Vector Databases.")
