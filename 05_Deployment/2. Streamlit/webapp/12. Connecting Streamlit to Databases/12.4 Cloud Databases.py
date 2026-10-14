import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Cloud Databases in Streamlit",
    page_icon="☁️",
    layout="wide",
)

st.title("☁️ Cloud & Modern Databases in Streamlit — Complete Guide")
st.markdown(
    """
Modern AI and SaaS architectures rarely rely on local files. They leverage cloud-hosted, 
serverless, and distributed databases: **Supabase** (Postgres + pgvector + Auth), 
**Firebase / Firestore** (NoSQL real-time documents), **MongoDB** (JSON document stores), 
**Snowflake** (Cloud enterprise data warehousing), and **Managed Cloud PostgreSQL** (Neon, AWS RDS). 
This guide explores client initialization, connection security, secrets management, and 12-factor configuration.
"""
)

st.divider()

# ==============================================================================
# 1. Supabase Integration (PostgreSQL + pgvector + Auth)
# ==============================================================================
st.header("1. Supabase Integration (Postgres, pgvector & Auth)", divider="rainbow")
st.markdown(
    r"""
**Supabase** is an open-source Firebase alternative built atop PostgreSQL. 
It provides native **Row-Level Security (RLS)**, real-time websockets, and **`pgvector`** for storing LLM embeddings.

```python
# Installation: pip install supabase
from supabase import create_client, Client
import streamlit as st

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()
response = supabase.table("llm_prompts").select("*").execute()
```
"""
)

col_sup1, col_sup2 = st.columns(2)

with col_sup1:
    st.markdown("**💻 Python Source Code (Supabase Client & pgvector)**")
    code_sup = '''from supabase import create_client
import streamlit as st

@st.cache_resource
def get_supabase_client():
    url = st.secrets["connections"]["supabase"]["url"]
    key = st.secrets["connections"]["supabase"]["key"]
    return create_client(url, key)

client = get_supabase_client()

# Query embeddings table using RPC function (pgvector):
def search_similar_embeddings(query_vector, threshold=0.8):
    return client.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": threshold,
        "match_count": 5
    }).execute()'''
    st.code(code_sup, language="python")

with col_sup2:
    st.markdown("**🖥️ Live Interactive Supabase Query Simulator**")
    class MockSupabaseTable:
        def __init__(self):
            self.data = [
                {"id": 1, "prompt": "Summarize quarterly AI earnings", "model": "Claude 3.7", "tokens": 420},
                {"id": 2, "prompt": "Generate unit tests for auth middleware", "model": "GPT-4.5", "tokens": 850},
                {"id": 3, "prompt": "Explain multi-head attention", "model": "DeepSeek R1", "tokens": 1200}
            ]

        def select(self, *args):
            return self

        def execute(self):
            return type("Response", (), {"data": self.data})()

    mock_client = MockSupabaseTable()
    res = mock_client.execute()

    with st.container(border=True):
        st.markdown("##### ⚡ Supabase Table Record Stream (`llm_prompts`)")
        st.dataframe(pd.DataFrame(res.data), width="stretch", hide_index=True)
        st.caption("Interacting with Supabase tables yields JSON payloads automatically formatted to DataFrames.")

st.divider()

# ==============================================================================
# 2. Firebase / Firestore Integration (NoSQL Documents)
# ==============================================================================
st.header("2. Firebase & Cloud Firestore Integration", divider="rainbow")
st.markdown(
    r"""
**Google Cloud Firestore** is a serverless, document-oriented NoSQL database. 
Data is organized in hierarchical collections and JSON documents (`collection("users").document("user_123")`).
"""
)

col_fb1, col_fb2 = st.columns(2)

with col_fb1:
    st.markdown("**💻 Python Source Code (Firebase Admin SDK)**")
    code_fb = '''import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

@st.cache_resource
def init_firestore():
    # Pass service account credentials dict from st.secrets:
    cred_dict = dict(st.secrets["firebase_service_account"])
    cred = credentials.Certificate(cred_dict)
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore()

# Write document:
db.collection("chats").document("session_01").set({
    "user": "alex@ai.org",
    "messages_count": 14,
    "created_at": firestore.SERVER_TIMESTAMP
})'''
    st.code(code_fb, language="python")

with col_fb2:
    st.markdown("**🖥️ Live Firestore Document Model Visualization**")
    with st.container(border=True):
        st.markdown("##### 📁 Collection: `agent_sessions/`")
        sample_doc = {
            "_document_id": "sess_89410_prod",
            "agent_type": "Autonomous Coder",
            "active_model": "Claude 3.7 Sonnet",
            "parameters": {"temperature": 0.2, "top_p": 0.95},
            "status": "RUNNING",
            "last_heartbeat": datetime.now().isoformat()
        }
        st.json(sample_doc)
        st.caption("Firestore documents store arbitrary nested JSON schemas with millisecond sync latency.")

st.divider()

# ==============================================================================
# 3. MongoDB Integration (Document Store with PyMongo)
# ==============================================================================
st.header("3. MongoDB Integration (`pymongo`)", divider="rainbow")
st.markdown(
    r"""
**MongoDB** stores BSON/JSON documents with dynamic schemas, making it popular for unstructured 
telemetry, LLM conversation logs, and agent tool execution traces.

```python
# Installation: pip install pymongo dnspython
from pymongo import MongoClient
import streamlit as st

@st.cache_resource
def get_mongo_client():
    uri = st.secrets["MONGO_URI"]  # mongodb+srv://user:pass@cluster.mongodb.net/
    return MongoClient(uri, serverSelectionTimeoutMS=5000)

client = get_mongo_client()
collection = client["ai_platform"]["interaction_logs"]
```
"""
)

col_mg1, col_mg2 = st.columns(2)

with col_mg1:
    st.markdown("**💻 Python Source Code (PyMongo Ingestion & Query)**")
    code_mongo = '''from pymongo import MongoClient
import pandas as pd

@st.cache_resource
def init_mongo():
    uri = st.secrets["mongo"]["connection_string"]
    return MongoClient(uri)

client = init_mongo()
coll = client["llm_db"]["audit_logs"]

# Query with MongoDB BSON filter:
cursor = coll.find(
    {"status": "SUCCESS", "latency_ms": {"$gt": 500}},
    {"_id": 0, "model": 1, "latency_ms": 1, "prompt_hash": 1}
).limit(50)

df_logs = pd.DataFrame(list(cursor))'''
    st.code(code_mongo, language="python")

with col_mg2:
    st.markdown("**🖥️ Live Interactive BSON Query Filter Simulator**")
    mock_mongo_docs = [
        {"model": "Claude 3.7", "latency_ms": 380, "tokens": 450, "status": "SUCCESS"},
        {"model": "GPT-4.5", "latency_ms": 720, "tokens": 1200, "status": "SUCCESS"},
        {"model": "DeepSeek R1", "latency_ms": 940, "tokens": 2100, "status": "SUCCESS"},
        {"model": "Llama 3.3", "latency_ms": 210, "tokens": 300, "status": "RATE_LIMIT"}
    ]

    min_lat_slider = st.slider("Filter: `latency_ms: {'$gt': ...}`", 100, 1000, 400, 50, key="sl_mongo_lat")
    filtered_docs = [d for d in mock_mongo_docs if d["latency_ms"] > min_lat_slider]

    with st.container(border=True):
        st.markdown(f"**Matches Found:** `{len(filtered_docs)}` documents")
        st.dataframe(pd.DataFrame(filtered_docs), width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 4. Snowflake Integration (`st.connection("snowflake")`)
# ==============================================================================
st.header("4. Snowflake Enterprise Data Warehouse Integration", divider="rainbow")
st.markdown(
    r"""
Streamlit features native, first-class support for **Snowflake**:
- Configure via `st.connection("snowflake")`.
- Execute SQL queries cached with `conn.query(sql, ttl="10m")`.
- Leverages the Snowflake Snowpark Python ecosystem.

```toml
# .streamlit/secrets.toml
[connections.snowflake]
account = "xy12345.us-east-1"
user = "ANALYST_USER"
password = "SecretPassword"
role = "DATA_SCIENTIST"
warehouse = "COMPUTE_WH"
database = "ANALYTICS_PROD"
schema = "PUBLIC"
```
"""
)

col_sn1, col_sn2 = st.columns(2)

with col_sn1:
    st.markdown("**💻 Python Source Code (Native Snowflake Connection)**")
    code_snow = '''import streamlit as st

# Native Snowflake connector:
conn = st.connection("snowflake")

# Query with automatic caching and Pandas conversion:
df_warehouse = conn.query("""
    SELECT 
        MODEL_NAME, 
        DATE_TRUNC('day', EVENT_TIMESTAMP) AS DAY,
        SUM(TOTAL_TOKENS) AS TOKENS_SUM
    FROM ANALYTICS_PROD.PUBLIC.LLM_BILLING_EVENTS
    GROUP BY 1, 2
    ORDER BY 2 DESC;
""", ttl="15m")

st.line_chart(df_warehouse, x="DAY", y="TOKENS_SUM")'''
    st.code(code_snow, language="python")

with col_sn2:
    st.markdown("**🖥️ Live Snowflake Warehouse Telemetry Simulation**")
    snow_data = pd.DataFrame({
        "WAREHOUSE": ["COMPUTE_WH_XS", "ANALYTICS_WH_L", "TRAINING_WH_2XL"],
        "STATE": ["SUSPENDED", "STARTED", "STARTED"],
        "AUTO_SUSPEND": ["300s", "600s", "180s"],
        "CREDITS_USED": [14.2, 85.6, 240.1]
    })
    with st.container(border=True):
        st.markdown("##### ❄️ Snowflake Warehouse Fleet Status")
        st.dataframe(snow_data, width="stretch", hide_index=True)
        st.caption("Snowflake queries return Apache Arrow/Pandas tables directly to Streamlit.")

st.divider()

# ==============================================================================
# 5. Cloud-Hosted PostgreSQL (Neon, AWS RDS, Supabase, Railway)
# ==============================================================================
st.header("5. Cloud-Hosted PostgreSQL (Neon Serverless & AWS RDS)", divider="rainbow")
st.markdown(
    r"""
When deploying to cloud platforms (Streamlit Community Cloud, AWS ECS, GCP Cloud Run), 
connecting to hosted PostgreSQL requires **connection poolers (e.g. PgBouncer)** 
and strict SSL transport encryption (`sslmode=require`).

| Hosting Provider | Serverless Features | Standard Port | Pooling Technology |
| :--- | :--- | :--- | :--- |
| **Neon** | Scales to zero, instant branching | 5432 (or 6543 pooled) | Built-in PgBouncer |
| **Supabase** | RLS, pgvector, Edge functions | 5432 (direct) / 6543 (pool) | Supavisor pooler |
| **AWS RDS / Aurora** | Multi-AZ replication, IAM auth | 5432 | RDS Proxy |
"""
)

# ==============================================================================
# 6. Database Credentials & 7. Environment Variables
# ==============================================================================
st.header("6. Credentials Management & 7. Environment Variables", divider="rainbow")
st.markdown(
    r"""
### The 12-Factor Hierarchy: Secrets vs. Environment Variables
Streamlit provides a clean resolution order when loading sensitive credentials:

```text
Credential Lookup Hierarchy:
1. Streamlit Secrets (`st.secrets["DB_PASSWORD"]`)   <-- Primary in Streamlit apps
2. Environment Variables (`os.environ["DB_PASSWORD"]`) <-- Standard Docker / Kubernetes
3. Local Default Fallback (`"localhost"`)             <-- Development only
```

```python
import os
import streamlit as st

def get_secret(key: str, default: str = "") -> str:
    # Robust credential resolver across Streamlit Cloud and Docker
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)
```
"""
)

col_sec1, col_sec2 = st.columns(2)

with col_sec1:
    st.markdown("**💻 Python Source Code (Universal Secret Resolver)**")
    code_resolver = '''import os
import streamlit as st

def load_database_uri() -> str:
    # 1. Try Streamlit Secrets (.streamlit/secrets.toml)
    if "DATABASE_URL" in st.secrets:
        return st.secrets["DATABASE_URL"]
    
    # 2. Try OS Environment Variable (Docker / Cloud Run)
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
        
    # 3. Fallback to local SQLite development database
    return "sqlite:///local_dev.db"'''
    st.code(code_resolver, language="python")

with col_sec2:
    st.markdown("**🖥️ Live Interactive Credential Audit & Masker**")
    # Simulate credential masking
    test_raw_key = st.text_input("Simulate Secret Key Value", "sk-proj-49281a9f01bc4987", type="password", key="txt_secret_mask")
    
    masked_key = test_raw_key[:4] + "••••••••••••" + test_raw_key[-4:] if len(test_raw_key) > 8 else "••••"
    
    with st.container(border=True):
        st.markdown("##### 🔒 Secure Audit Display")
        st.write(f"• **Raw String Length**: `{len(test_raw_key)}` characters")
        st.write(f"• **Safe UI Masked Representation**: `{masked_key}`")
        st.write(f"• **Environment Source**: `Streamlit Secrets Sandbox`")
        st.success("Never display raw credentials in plaintext on screen or in error logs!")

st.divider()

# ==============================================================================
# 8. Connection Security & Hardening Best Practices
# ==============================================================================
st.header("8. Connection Security & Hardening Best Practices", divider="rainbow")
st.markdown(
    r"""
### Enterprise Production Checklist

1. **Mandatory SSL / TLS Encryption**: Always append `?sslmode=require` to your connection strings to prevent man-in-the-middle eavesdropping on cloud networks.
2. **IP Allowlisting & VPC Peering**: Restrict database access to Streamlit server IP ranges or use VPC peering/tunnels.
3. **Principle of Least Privilege**: Never connect Streamlit using the database `postgres` or `root` superuser! Create dedicated read-only application users (`analyst_readonly`).
4. **Git Safety Guard**: Add `.streamlit/secrets.toml` to your `.gitignore` immediately:
   ```gitignore
   # .gitignore
   .streamlit/secrets.toml
   .env
   *.pem
   *.key
   ```
"""
)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Cloud Database Architecture Matrix", divider="rainbow")
st.markdown(
    r"""
| Database Cloud | Transport / Protocol | Primary Python Library | Recommended Streamlit Pattern |
| :--- | :--- | :--- | :--- |
| **Supabase** | REST API & PostgreSQL | `supabase-py` / `psycopg` | `init_supabase()` with `@st.cache_resource` |
| **Firestore** | gRPC / Google SDK | `firebase-admin` | Service account JSON dictionary in `st.secrets` |
| **MongoDB** | MongoDB Wire Protocol | `pymongo` | `MongoClient(uri)` with connection timeout |
| **Snowflake** | Snowflake SQL API | `snowflake-connector-python` | `st.connection("snowflake")` |
| **Cloud PostgreSQL**| TCP / SSL / PgBouncer | `psycopg2` / `SQLAlchemy` | `create_engine()` with connection pooling |
"""
)
