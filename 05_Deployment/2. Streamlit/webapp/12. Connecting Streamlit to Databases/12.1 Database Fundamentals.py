import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Database Fundamentals in Streamlit",
    page_icon="🗄️",
    layout="wide",
)

st.title("🗄️ Database Fundamentals & SQL in Streamlit — Complete Guide")
st.markdown(
    """
Modern data science, analytics, and Generative AI applications rely on persistent databases 
to record audit telemetry, maintain model metadata, store evaluation benchmarks, 
and manage user session credentials. Understanding relational databases (RDBMS), 
ACID transactions, parameterized queries, and thread-safe connection pooling 
is foundational to building enterprise-grade Streamlit solutions.
"""
)

st.divider()

# ==============================================================================
# 1. Database Connection Concepts
# ==============================================================================
st.header("1. Database Connection Concepts & Architecture", divider="rainbow")
st.markdown(
    r"""
### Embedded vs. Client-Server Databases
- **Embedded (e.g., SQLite, DuckDB)**: The database engine runs directly inside the Streamlit Python process. Zero network latency, single-file storage, ideal for local analytics and prototypes.
- **Client-Server (e.g., PostgreSQL, MySQL, Snowflake)**: Runs as an independent network service. Supports high concurrency, automated replication, connection pooling, and strict multi-user access control.

### The Streamlit Connection Rule
A database connection is a stateful TCP socket or file descriptor. 
**Never open and close connections on every widget rerun!** 
Always maintain a persistent connection pool using **`@st.cache_resource`** or Streamlit's native **`st.connection()`**.
"""
)

col_cn1, col_cn2 = st.columns(2)

with col_cn1:
    st.markdown("**💻 Python Source Code (Cached Connection Singleton)**")
    code_conn = '''import sqlite3
import streamlit as st

# ✅ Keep the database connection alive across reruns:
@st.cache_resource
def get_database_connection(db_path: str = "enterprise_ai.db"):
    # check_same_thread=False allows Streamlit's multi-threaded runners to access it
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Returns dict-like rows
    return conn

conn = get_database_connection(":memory:")'''
    st.code(code_conn, language="python")

with col_cn2:
    st.markdown("**🖥️ Live Interactive Connection Inspector**")
    @st.cache_resource
    def get_in_memory_db():
        c = sqlite3.connect(":memory:", check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    active_conn = get_in_memory_db()
    with st.container(border=True):
        st.markdown("##### 🔌 Active Database Connection State")
        st.write(f"• **Engine**: `SQLite (In-Memory :memory:)`")
        st.write(f"• **Connection Object Address**: `{hex(id(active_conn))}`")
        st.write(f"• **Isolation Level**: `{active_conn.isolation_level or 'Default'}`")
        st.success("Connection singleton verified: Persistent across page reruns.")

st.divider()

# ==============================================================================
# 2. SQL Basics (DDL, DML & Aggregations)
# ==============================================================================
st.header("2. SQL Basics (DDL, DML & Aggregations)", divider="rainbow")
st.markdown(
    r"""
Structured Query Language (SQL) is divided into core sub-languages:
- **DDL (Data Definition Language)**: `CREATE TABLE`, `ALTER TABLE`, `DROP TABLE`.
- **DML (Data Manipulation Language)**: `SELECT`, `INSERT`, `UPDATE`, `DELETE`.
- **DQL & Aggregations**: `COUNT()`, `AVG()`, `SUM()`, `MIN()`, `MAX()`, combined with `GROUP BY` and `HAVING`.
"""
)

col_sql1, col_sql2 = st.columns(2)

with col_sql1:
    st.markdown("**💻 Standard SQL Queries Reference**")
    code_sql = '''-- 1. Create Table (DDL)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    pass_rate REAL CHECK(pass_rate >= 0.0 AND pass_rate <= 1.0)
);

-- 2. Insert Record (DML)
INSERT INTO models (name, provider, pass_rate)
VALUES ('Claude 3.7 Sonnet', 'Anthropic', 0.924);

-- 3. Filter & Aggregate (DQL)
SELECT provider, COUNT(*) as count, AVG(pass_rate) as avg_score
FROM models
WHERE pass_rate >= 0.85
GROUP BY provider
ORDER BY avg_score DESC;'''
    st.code(code_sql, language="sql")

with col_sql2:
    st.markdown("**🖥️ Live Interactive Query Runner**")
    # Setup initial demonstration data in our memory DB
    cur = active_conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS demo_sql_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        provider TEXT NOT NULL,
        latency_ms INTEGER NOT NULL,
        accuracy REAL NOT NULL
    );
    """)
    cur.executemany("""
    INSERT OR IGNORE INTO demo_sql_models (name, provider, latency_ms, accuracy)
    VALUES (?, ?, ?, ?)
    """, [
        ("Claude 3.7 Sonnet", "Anthropic", 380, 0.924),
        ("GPT-4.5 Omni", "OpenAI", 420, 0.902),
        ("DeepSeek R1", "DeepSeek", 920, 0.936),
        ("Llama 3.3 70B", "Meta", 240, 0.821),
        ("Qwen 2.5 72B", "Alibaba", 260, 0.843)
    ])
    active_conn.commit()

    min_acc_slider = st.slider("Filter by Minimum Accuracy", 0.80, 0.95, 0.85, 0.01, key="sl_min_acc")
    sql_query = f"""
    SELECT provider, COUNT(*) as model_count, ROUND(AVG(accuracy), 3) as avg_accuracy, ROUND(AVG(latency_ms), 0) as avg_latency
    FROM demo_sql_models
    WHERE accuracy >= {min_acc_slider}
    GROUP BY provider
    ORDER BY avg_accuracy DESC
    """
    df_agg = pd.read_sql_query(sql_query, active_conn)
    with st.container(border=True):
        st.markdown(f"**Executing Query:** `WHERE accuracy >= {min_acc_slider}`")
        st.dataframe(df_agg, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 3. Relational Databases & ACID Guarantees
# ==============================================================================
st.header("3. Relational Model & ACID Transaction Guarantees", divider="rainbow")
st.markdown(
    r"""
### The ACID Guarantees in Production Applications
Relational databases enforce four mathematical guarantees that prevent corruption during multi-user operations:

| Principle | Meaning | Concrete Production AI Example |
| :--- | :--- | :--- |
| **Atomicity** | All or Nothing | Deducting 500 token credits and logging the prompt completion happen together; if the server crashes mid-way, all changes roll back. |
| **Consistency** | Invariant Integrity | Foreign key constraints ensure an evaluation score cannot point to a non-existent model ID. |
| **Isolation** | Concurrent Safety | Two concurrent users updating the same model rating cannot overwrite each other's updates simultaneously. |
| **Durability** | Non-Volatile Commits | Once a transaction issues `conn.commit()`, recorded prompts and billing logs survive server power outages. |
"""
)

# ==============================================================================
# 4. Tables and Schemas & 5. Primary and Foreign Keys
# ==============================================================================
st.header("4. Tables, Schemas & 5. Primary and Foreign Keys", divider="rainbow")
st.markdown(
    r"""
### Relational Schema Design: 1-to-Many Relationships
In normalized relational schemas, tables are split into entities connected by **Primary Keys (`PK`)** 
and **Foreign Keys (`FK`)**:

""")

st.graphviz_chart(
    r'''
    digraph DBConnection {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        app [label="Streamlit Application Run\nUser interaction triggers rerun", fillcolor="#DBEAFE", color="#1D4ED8"];
        conn [label="st.connection('postgresql')\n• Persistent Engine Instance\n• Connection Pool (reused across reruns)\n• Auto credential binding from secrets.toml", fillcolor="#FEF3C7", color="#D97706"];
        db [label="Enterprise Database\nPostgreSQL / MySQL / Snowflake", fillcolor="#D1FAE5", color="#059669"];

        app -> conn [label="conn.query('SELECT...')"];
        conn -> db [label="Pooled SQL Connection"];
    }
    '''
)


col_rel1, col_rel2 = st.columns(2)

with col_rel1:
    st.markdown("**💻 Python Source Code (Relational Schema with Foreign Keys)**")
    code_schema = '''# Create normalized relational tables:
cur.execute("""
CREATE TABLE IF NOT EXISTS ai_providers (
    provider_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    headquarters TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS ai_deployments (
    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    context_window INTEGER DEFAULT 128000,
    FOREIGN KEY (provider_id) REFERENCES ai_providers(provider_id)
        ON DELETE CASCADE
);
""")'''
    st.code(code_schema, language="python")

with col_rel2:
    st.markdown("**🖥️ Live Schema & Relational Join Query**")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_providers (
        provider_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        region TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_deployments (
        deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider_id INTEGER NOT NULL,
        model_name TEXT NOT NULL UNIQUE,
        context_window INTEGER DEFAULT 128000,
        FOREIGN KEY (provider_id) REFERENCES ai_providers(provider_id)
    );
    """)
    cur.executemany("INSERT OR IGNORE INTO ai_providers (name, region) VALUES (?, ?)", [
        ("Anthropic", "US-West"),
        ("DeepSeek", "AP-East")
    ])
    cur.executemany("INSERT OR IGNORE INTO ai_deployments (provider_id, model_name, context_window) VALUES (?, ?, ?)", [
        (1, "Claude 3.7 Sonnet", 200000),
        (1, "Claude 3.5 Haiku", 200000),
        (2, "DeepSeek R1", 128000)
    ])
    active_conn.commit()

    join_sql = """
    SELECT d.deployment_id, d.model_name, p.name AS provider, p.region, d.context_window
    FROM ai_deployments d
    JOIN ai_providers p ON d.provider_id = p.provider_id
    """
    df_join = pd.read_sql_query(join_sql, active_conn)
    with st.container(border=True):
        st.markdown("##### 🔗 Relational JOIN Output (`ai_deployments` ⨝ `ai_providers`)")
        st.dataframe(df_join, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 6. Full Interactive CRUD Operations (Create, Read, Update, Delete)
# ==============================================================================
st.header("6. Interactive CRUD Operations Manager", divider="rainbow")
st.markdown(
    """
A complete, real-world demonstration of the 4 fundamental database operations:
- **Create**: Insert new evaluation records via form submission.
- **Read**: Live virtualized tabular display.
- **Update**: Edit benchmark scores or descriptions in-place.
- **Delete**: Drop obsolete or faulty model configurations.
"""
)

# Setup CRUD table in memory
cur.execute("""
CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    template TEXT NOT NULL,
    created_at TEXT NOT NULL
);
""")
cur.execute("""
INSERT OR IGNORE INTO prompt_templates (title, category, template, created_at)
VALUES 
    ('Python Refactor', 'Coding', 'Refactor the following Python code for PEP8 compliance: {code}', '2026-10-10 10:00:00'),
    ('RAG Summarizer', 'Reasoning', 'Summarize the context chunks in 3 bullet points: {chunks}', '2026-10-11 14:30:00')
""")
active_conn.commit()

crud_c1, crud_c2 = st.columns([1, 1])

with crud_c1:
    with st.container(border=True):
        st.subheader("➕ Create: Add New Template")
        with st.form("form_create_template", clear_on_submit=True):
            in_title = st.text_input("Template Title")
            in_cat = st.selectbox("Category", ["Coding", "Reasoning", "Safety", "Evaluation"])
            in_body = st.text_area("Prompt Body", height=70)
            btn_add = st.form_submit_button("Save Template")

            if btn_add:
                if in_title.strip() and in_body.strip():
                    try:
                        cur.execute("""
                        INSERT INTO prompt_templates (title, category, template, created_at)
                        VALUES (?, ?, ?, ?)
                        """, (in_title.strip(), in_cat, in_body.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        active_conn.commit()
                        st.success(f"Added template `{in_title}`!")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("A template with this title already exists!")
                else:
                    st.warning("Title and body cannot be empty.")

with crud_c2:
    with st.container(border=True):
        st.subheader("🗑️ Delete: Remove Obsolete Template")
        df_existing = pd.read_sql_query("SELECT id, title FROM prompt_templates", active_conn)
        if not df_existing.empty:
            sel_to_del = st.selectbox("Select Template to Delete", df_existing["title"].tolist(), key="sel_del_tmpl")
            if st.button("Delete Selected Template", type="primary", key="btn_del_tmpl"):
                cur.execute("DELETE FROM prompt_templates WHERE title = ?", (sel_to_del,))
                active_conn.commit()
                st.warning(f"Deleted template `{sel_to_del}`.")
                st.rerun()
        else:
            st.info("No templates available to delete.")

st.markdown("##### 📖 Read: Current Registered Templates in Database")
df_all_templates = pd.read_sql_query("SELECT id, title, category, template, created_at FROM prompt_templates", active_conn)
st.dataframe(df_all_templates, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 7. Parameterized Queries (Preventing SQL Injection)
# ==============================================================================
st.header("7. Parameterized Queries vs. SQL Injection Vulnerabilities", divider="rainbow")
st.markdown(
    r"""
### The Number One Database Security Rule
**Never concatenate raw user strings into SQL queries!** 
Direct f-string formatting (`f"SELECT * FROM users WHERE name = '{user}'"`) enables 
**SQL Injection Attacks**, where a malicious user inputs `' OR '1'='1` to bypass security 
or `'; DROP TABLE users; --` to destroy your database!

### Parameterized Queries (Prepared Statements)
Always pass user input as separate query parameter tuples. 
The database engine treats the arguments strictly as data, never as executable code:

| Vulnerable (SQL Injection Risk ❌) | Secure Parameterized Query (✅) |
| :--- | :--- |
| `f"SELECT * FROM tbl WHERE cat = '{val}'"` | `"SELECT * FROM tbl WHERE cat = ?", (val,)` |
| `f"UPDATE tbl SET score = {score}"` | `"UPDATE tbl SET score = ? WHERE id = ?", (score, uid)` |
"""
)

col_sec1, col_sec2 = st.columns(2)

with col_sec1:
    st.markdown("**💻 Python Source Code (Parameterized Protection)**")
    code_sec = '''# ❌ VULNERABLE TO SQL INJECTION:
# If user inputs: admin' OR '1'='1
unsafe_sql = f"SELECT * FROM users WHERE username = '{user_input}'"
cur.execute(unsafe_sql)  # ⚠️ Leaks all user records!

# ✅ SECURE: PARAMETERIZED PLACEHOLDER (?)
safe_sql = "SELECT * FROM users WHERE username = ?"
cur.execute(safe_sql, (user_input,))  # Database treats input strictly as a literal string!'''
    st.code(code_sec, language="python")

with col_sec2:
    st.markdown("**🖥️ Live Interactive SQL Injection Defense Demo**")
    malicious_probe = st.text_input("Simulate User Input Probe", value="' OR '1'='1", key="txt_malicious_probe")
    
    # Safe execution using parameterized query
    safe_sql_demo = "SELECT id, title, category FROM prompt_templates WHERE category = ?"
    cur.execute(safe_sql_demo, (malicious_probe,))
    results_safe = cur.fetchall()

    with st.container(border=True):
        st.write(f"Executing: `SELECT ... WHERE category = ?` with parameter: `{malicious_probe}`")
        if len(results_safe) == 0:
            st.success("🛡️ **Defense Verified**: Query returned 0 records because no category literally equals `' OR '1'='1`. SQL injection was safely neutralized!")
        else:
            st.write(f"Returned {len(results_safe)} records.")

st.divider()

# ==============================================================================
# 8. Connection Management & Context Managers
# ==============================================================================
st.header("8. Connection Management & Python Context Managers", divider="rainbow")
st.markdown(
    r"""
### Automated Commit and Rollback
Using Python's **`with connection:`** context manager ensures that:
1. `conn.commit()` is called automatically if all operations succeed without error.
2. `conn.rollback()` is triggered automatically if an exception or database error is raised, leaving the database clean and uncorrupted!
"""
)

col_cm1, col_cm2 = st.columns(2)

with col_cm1:
    st.markdown("**💻 Python Source Code (Transaction Context Manager)**")
    code_tx = '''def transfer_credits(sender_id, receiver_id, amount):
    # Context manager automatically commits on success, rolls back on error:
    with conn:
        conn.execute(
            "UPDATE balances SET credits = credits - ? WHERE user_id = ?",
            (amount, sender_id)
        )
        conn.execute(
            "UPDATE balances SET credits = credits + ? WHERE user_id = ?",
            (amount, receiver_id)
        )
    # Both statements succeed or both fail together (Atomicity)!'''
    st.code(code_tx, language="python")

with col_cm2:
    st.markdown("**🖥️ Transaction Rollback Resilience Preview**")
    with st.container(border=True):
        st.markdown("##### 🛡️ ACID Transaction Workflow")
        st.markdown("1. `BEGIN TRANSACTION` (Auto-managed)")
        st.markdown("2. Execute Step 1: Deduct credits")
        st.markdown("3. Execute Step 2: Credit destination")
        st.markdown("4. **`COMMIT`** if zero exceptions | **`ROLLBACK`** if error raised")
        st.info("Context managers prevent dangling open transactions from locking database tables.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Database Fundamentals Quick Reference", divider="rainbow")
st.markdown(
    r"""
| Concept | Primary Purpose | Best Practice in Streamlit |
| :--- | :--- | :--- |
| **Connection Pooling** | Keep TCP sockets / file handles alive | Wrap in `@st.cache_resource` |
| **Data Ingestion** | Read database tables into memory | Use `pd.read_sql_query(sql, conn)` |
| **Parameterized Queries**| Defend against SQL Injection attacks | Pass tuples to placeholders: `cur.execute(sql, (val,))` |
| **Transactions** | Ensure ACID consistency | Wrap write operations inside `with conn:` |
| **Primary Keys (PK)** | Uniquely identify rows | Use `INTEGER PRIMARY KEY AUTOINCREMENT` |
| **Foreign Keys (FK)** | Enforce referential integrity | Enable `PRAGMA foreign_keys = ON;` in SQLite |
| **Row Factory** | Dict-like row access | Set `conn.row_factory = sqlite3.Row` |
"""
)
