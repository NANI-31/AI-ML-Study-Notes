import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="PostgreSQL and MySQL in Streamlit",
    page_icon="🐬",
    layout="wide",
)

st.title("🐬 PostgreSQL & MySQL Enterprise Integration — Complete Guide")
st.markdown(
    """
For scalable multi-user applications, enterprise dashboards, and production GenAI pipelines, 
relational client-server databases like **PostgreSQL** and **MySQL** provide industrial-grade 
concurrency, row-level locking, automated replication, and strict ACID guarantees. 
In Streamlit, connecting to these databases requires **SQLAlchemy 2.0**, robust **connection pooling**, 
secure **secrets management**, and defensive **transaction error handling**.
"""
)

st.divider()

# ==============================================================================
# 1. Connecting to PostgreSQL & 2. Connecting to MySQL
# ==============================================================================
st.header("1. PostgreSQL vs. 2. MySQL in the Streamlit Ecosystem", divider="rainbow")
st.markdown(
    r"""
Streamlit connects to PostgreSQL and MySQL using two complementary patterns:
1. **Native `st.connection("sql")`**: Built-in wrapper managing caching, secrets, and pool lifecycle automatically.
2. **Custom SQLAlchemy 2.0 Engine**: Provides granular control over custom pooling parameters, interceptors, and ORM sessions.

| Database | Recommended Python Driver | Standard Dialect Prefix | Key Enterprise Strengths |
| :--- | :--- | :--- | :--- |
| **PostgreSQL** | `psycopg` (v3) or `psycopg2-binary` | `postgresql+psycopg://` | Rich JSONB, pgvector for embeddings, window functions |
| **MySQL** | `pymysql` or `mysqlclient` | `mysql+pymysql://` | High-throughput OLTP, widespread cloud hosting support |
"""
)

# ==============================================================================
# 3. Using SQLAlchemy & 4. Connection Strings
# ==============================================================================
st.header("3. Using SQLAlchemy 2.0 & 4. Connection String Anatomy", divider="rainbow")
st.markdown(
    r"""
### Connection String (RFC-1738 URI) Specification
```text
dialect+driver://username:password@host:port/database?sslmode=require
```

### Storing Credentials in `.streamlit/secrets.toml`
Never hardcode passwords in your Python files! Store them in your local secrets file:

```toml
# .streamlit/secrets.toml
[connections.production_db]
dialect = "postgresql"
driver = "psycopg2"
host = "aws-us-east-1.postgres.database.azure.com"
port = 5432
database = "enterprise_ai"
username = "postgres_admin"
password = "SuperSecretPassword123!"
```
"""
)

col_cs1, col_cs2 = st.columns(2)

with col_cs1:
    st.markdown("**💻 Python Source Code (Engine Creation & secrets.toml)**")
    code_engine = '''import streamlit as st
from sqlalchemy import create_engine, text

# Pattern A: Native Streamlit Connector (Zero Boilerplate)
conn = st.connection("production_db", type="sql")
df = conn.query("SELECT * FROM model_evals LIMIT 10;", ttl="10m")

# Pattern B: Custom SQLAlchemy 2.0 Engine with Pool Tuning
@st.cache_resource
def get_custom_engine():
    db_uri = st.secrets["connections"]["production_db"]["url"]
    return create_engine(
        db_uri,
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,
        pool_pre_ping=True
    )'''
    st.code(code_engine, language="python")

with col_cs2:
    st.markdown("**🖥️ Live Interactive Connection URI Builder**")
    with st.container(border=True):
        st.markdown("##### 🔧 Interactive URI Generator")
        b_dialect = st.selectbox("Database Type", ["PostgreSQL (psycopg2)", "MySQL (pymysql)", "SQLite (Local)"], key="uri_gen_db")
        b_host = st.text_input("Host Address", "ai-cluster.internal.net", key="uri_gen_host")
        b_user = st.text_input("Username", "readonly_analyst", key="uri_gen_user")
        b_db = st.text_input("Database Name", "llm_telemetry_prod", key="uri_gen_dbname")

        if "Postgres" in b_dialect:
            sample_uri = f"postgresql+psycopg2://{b_user}:••••••••@{b_host}:5432/{b_db}?sslmode=require"
        elif "MySQL" in b_dialect:
            sample_uri = f"mysql+pymysql://{b_user}:••••••••@{b_host}:3306/{b_db}?charset=utf8mb4"
        else:
            sample_uri = "sqlite:///./enterprise_local.db"

        st.code(sample_uri, language="bash")
        st.caption("RFC-1738 compliant database URI generated for production secrets.")

st.divider()

# ==============================================================================
# 5. Connection Pooling Architecture
# ==============================================================================
st.header("5. Connection Pooling Architecture (`QueuePool`)", divider="rainbow")
st.markdown(
    r"""
### Why Connection Pooling is Critical
Creating a new TCP socket connection to a remote PostgreSQL/MySQL cluster takes **100–300 ms** 
due to network latency, TLS handshakes, and authentication negotiation. 
A connection pool maintains a warm set of open sockets in memory, recycling them across user requests.

| Parameter | Recommended Value | Purpose |
| :--- | :--- | :--- |
| **`pool_size`** | `5` – `15` | Steady-state persistent open connections maintained in memory. |
| **`max_overflow`** | `10` – `20` | Temporary burst connections allowed during heavy traffic spikes. |
| **`pool_timeout`** | `30` (seconds) | Seconds to wait before raising `TimeoutError` if pool is full. |
| **`pool_recycle`** | `1800` (30 mins) | Reconnects sockets older than 30 mins to avoid cloud firewall dropouts. |
| **`pool_pre_ping`**| **`True`** | Tests socket liveness before checkout (`SELECT 1`), auto-reconnecting dead sockets! |
"""
)

col_pool1, col_pool2 = st.columns(2)

with col_pool1:
    st.markdown("**💻 Python Source Code (Engine Pool Configuration)**")
    code_pool = '''from sqlalchemy import create_engine

# Production Engine with Enterprise QueuePool settings:
engine = create_engine(
    "postgresql+psycopg2://user:pass@host:5432/db",
    pool_size=10,            # 10 persistent connections
    max_overflow=20,         # 20 burst connections
    pool_timeout=30,         # 30s timeout
    pool_recycle=1800,       # Recycle sockets every 30 mins
    pool_pre_ping=True,      # Proactively test socket before query
    echo=False               # Set True only for debug logging
)'''
    st.code(code_pool, language="python")

with col_pool2:
    st.markdown("**🖥️ Live Interactive Pool Telemetry Simulator**")
    @st.cache_resource
    def get_simulated_engine():
        # In-memory SQLite simulating a connection pool with modern SQLAlchemy 2.0
        from sqlalchemy.pool import StaticPool
        eng = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        with eng.begin() as c:
            c.execute(text("""
            CREATE TABLE enterprise_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_billed INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """))
            c.execute(text("""
            INSERT INTO enterprise_tokens (tenant, model, tokens_billed, cost_usd)
            VALUES 
                ('Acme Corp', 'Claude 3.7 Sonnet', 840000, 2.52),
                ('Stark Tech', 'GPT-4.5 Omni', 1200000, 3.60),
                ('Wayne AI', 'DeepSeek R1', 3400000, 1.87),
                ('Acme Corp', 'Claude 3.5 Haiku', 420000, 0.42);
            """))
        return eng

    live_engine = get_simulated_engine()

    with st.container(border=True):
        st.markdown("##### 📊 SQLAlchemy Engine Pool State")
        st.write(f"• **Dialect**: `{live_engine.dialect.name.upper()}`")
        st.write(f"• **Pool Implementation**: `{live_engine.pool.__class__.__name__}`")
        st.write(f"• **Engine Object Pointer**: `{hex(id(live_engine))}`")
        st.success("SQLAlchemy 2.0 engine initialized with automated pool recycling.")

st.divider()

# ==============================================================================
# 6. Querying Database Records (text() & Pandas Integration)
# ==============================================================================
st.header("6. Querying Database Records with SQLAlchemy `text()`", divider="rainbow")
st.markdown(
    r"""
In SQLAlchemy 2.0, raw SQL strings must always be explicitly wrapped in **`sqlalchemy.text()`**:
- **Named Parameterization**: Use `:param_name` syntax for clear readability.
- **`pd.read_sql_query`**: Passes the bound engine or connection and returns a clean DataFrame.
"""
)

col_q1, col_q2 = st.columns(2)

with col_q1:
    st.markdown("**💻 Python Source Code (Named Parameter Queries)**")
    code_query = '''from sqlalchemy import text
import pandas as pd

# Named parameters syntax (:param_name):
sql_stmt = text("""
    SELECT tenant, model, SUM(tokens_billed) as total_tokens, SUM(cost_usd) as total_cost
    FROM enterprise_tokens
    WHERE tenant = :target_tenant
    GROUP BY tenant, model
    ORDER BY total_tokens DESC;
""")

# Execute cleanly through connection context:
with engine.connect() as conn:
    df = pd.read_sql_query(sql_stmt, conn, params={"target_tenant": "Acme Corp"})
    st.dataframe(df)'''
    st.code(code_query, language="python")

with col_q2:
    st.markdown("**🖥️ Live Interactive Named Query Execution**")
    tenants = ["All", "Acme Corp", "Stark Tech", "Wayne AI"]
    chosen_tenant = st.selectbox("Select Tenant Parameter (`:target_tenant`)", tenants, key="sel_tenant_param")

    if chosen_tenant == "All":
        stmt = text("SELECT id, tenant, model, tokens_billed, cost_usd, created_at FROM enterprise_tokens ORDER BY id DESC;")
        df_tokens = pd.read_sql_query(stmt, live_engine.connect())
    else:
        stmt = text("SELECT id, tenant, model, tokens_billed, cost_usd, created_at FROM enterprise_tokens WHERE tenant = :t ORDER BY id DESC;")
        df_tokens = pd.read_sql_query(stmt, live_engine.connect(), params={"t": chosen_tenant})

    st.dataframe(
        df_tokens,
        column_config={
            "tokens_billed": st.column_config.NumberColumn("Billed Tokens", format="%d tok"),
            "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.2f")
        },
        width="stretch",
        hide_index=True
    )

st.divider()

# ==============================================================================
# 7. Transaction Management (`with engine.begin()`)
# ==============================================================================
st.header("7. Transaction Management: The `engine.begin()` Pattern", divider="rainbow")
st.markdown(
    r"""
### Atomic Transactions with Automatic Rollback
SQLAlchemy 2.0 provides **`with engine.begin() as conn:`**:
1. Automatically starts a database transaction (`BEGIN`).
2. Issues an automatic **`COMMIT`** upon exiting the code block cleanly.
3. Automatically triggers an immediate **`ROLLBACK`** if any exception occurs inside the block!
"""
)

col_tx1, col_tx2 = st.columns(2)

with col_tx1:
    st.markdown("**💻 Python Source Code (Atomic Multi-Table Mutation)**")
    code_atomic = '''from sqlalchemy import text

# Guaranteed all-or-nothing atomicity:
with engine.begin() as conn:
    # 1. Deduct customer credit:
    conn.execute(
        text("UPDATE accounts SET credits = credits - :amount WHERE id = :user_id"),
        {"amount": 10.0, "user_id": 42}
    )
    # 2. Record billing audit log:
    conn.execute(
        text("INSERT INTO billing_logs (user_id, amount) VALUES (:user_id, :amount)"),
        {"amount": 10.0, "user_id": 42}
    )
# Both operations commit together; if step 2 fails, step 1 is undone automatically!'''
    st.code(code_atomic, language="python")

with col_tx1:
    st.markdown("**Commit vs. Rollback Lifecycle:**")
    st.markdown(
        r"""
- Never manage `conn.commit()` manually if you can use `engine.begin()`.
- Unhandled Python errors will never leave uncommitted orphaned records.
"""
    )

with col_tx2:
    st.markdown("**🖥️ Live Interactive Atomic Ingestion Form**")
    with st.form("form_atomic_insert", clear_on_submit=True):
        st.markdown("##### ➕ Ingest Tenant Token Usage Log")
        f_tenant = st.selectbox("Tenant", ["Acme Corp", "Stark Tech", "Wayne AI", "Cyberdyne Systems"])
        f_model = st.selectbox("Model", ["Claude 3.7 Sonnet", "GPT-4.5 Omni", "DeepSeek R1"])
        f_tokens = st.number_input("Tokens Ingested", min_value=1000, max_value=10000000, value=250000, step=50000)
        f_cost = round(f_tokens * 0.000003, 3)
        st.caption(f"Estimated Cost: **${f_cost:.3f}**")

        btn_commit_tx = st.form_submit_button("Commit Transaction to Database")

        if btn_commit_tx:
            with live_engine.begin() as conn_tx:
                conn_tx.execute(
                    text("""
                    INSERT INTO enterprise_tokens (tenant, model, tokens_billed, cost_usd)
                    VALUES (:tenant, :model, :tokens, :cost)
                    """),
                    {"tenant": f_tenant, "model": f_model, "tokens": f_tokens, "cost": f_cost}
                )
            st.success(f"Transaction Committed: Billed {f_tokens:,} tokens to `{f_tenant}`.")
            st.rerun()

st.divider()

# ==============================================================================
# 8. Handling Database Errors Graciously
# ==============================================================================
st.header("8. Handling Database Errors Graciously", divider="rainbow")
st.markdown(
    r"""
In production web apps, database failures (network partition, unique key violation, bad SQL syntax) 
must be caught gracefully. Never let raw SQL stack traces expose internal database topology to users!

| SQLAlchemy Exception | Common Real-World Cause | User-Friendly Response Pattern |
| :--- | :--- | :--- |
| **`OperationalError`** | Database unreachable / TLS drop / timeout | `st.error("Database unavailable. Please retry in 30s.")` |
| **`IntegrityError`** | Unique constraint / Foreign key violation | `st.warning("A record with this identifier already exists.")` |
| **`ProgrammingError`** | Syntax error / Missing table or column | Log to Sentry/cloud; show generic error to client |
"""
)

col_err1, col_err2 = st.columns(2)

with col_err1:
    st.markdown("**💻 Python Source Code (Defensive Exception Wrapping)**")
    code_err = '''from sqlalchemy.exc import IntegrityError, OperationalError

def safe_database_operation():
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO unique_table VALUES (:id)"), {"id": 1})
    except IntegrityError as e:
        st.warning("⚠️ Duplicate key violation: This ID is already registered.")
    except OperationalError as e:
        st.error("🔌 Network connectivity lost: Unable to reach database cluster.")
    except Exception as e:
        st.error("❌ An unexpected database error occurred.")'''
    st.code(code_err, language="python")

with col_err2:
    st.markdown("**🖥️ Live Interactive Exception Handling Simulator**")
    test_scenario = st.selectbox(
        "Simulate Database Error Type",
        [
            "Normal Successful Query",
            "Simulate IntegrityError (Duplicate Constraint)",
            "Simulate ProgrammingError (Bad SQL Column)"
        ],
        key="sel_sim_err"
    )

    if st.button("Trigger Test Execution", key="btn_exec_sim_err"):
        try:
            with live_engine.begin() as test_conn:
                if "Duplicate" in test_scenario:
                    # Violate constraint
                    test_conn.execute(text("INSERT INTO enterprise_tokens (id, tenant, model, tokens_billed, cost_usd) VALUES (1, 'Fail', 'Fail', 0, 0)"))
                elif "Bad SQL" in test_scenario:
                    # Invalid column name
                    test_conn.execute(text("SELECT non_existent_column_xyz FROM enterprise_tokens;"))
                else:
                    test_conn.execute(text("SELECT COUNT(*) FROM enterprise_tokens;"))
            st.success("✅ Query executed successfully with zero errors!")
        except IntegrityError:
            st.warning("⚠️ **Graceful Error Caught (`IntegrityError`)**: Primary key conflict intercepted. Database was automatically rolled back.")
        except ProgrammingError:
            st.error("❌ **Graceful Error Caught (`ProgrammingError`)**: Bad SQL statement intercepted. Database was automatically rolled back.")
        except Exception as e:
            st.error(f"Captured generic exception: {type(e).__name__}")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 PostgreSQL & MySQL Enterprise Checklist", divider="rainbow")
st.markdown(
    r"""
| Capability | Recommended Modern Pattern | Purpose |
| :--- | :--- | :--- |
| **SQL Syntax** | Wrap queries in `sqlalchemy.text("...")` | Strict SQLAlchemy 2.0 compliance |
| **Driver (Postgres)** | `psycopg` (v3) or `psycopg2-binary` | High-performance PostgreSQL transport |
| **Driver (MySQL)** | `pymysql` | Pure-Python lightweight MySQL driver |
| **Pool Sizing** | `pool_size=10, max_overflow=20` | High-concurrency worker connection safety |
| **Stale Connections**| `pool_recycle=1800, pool_pre_ping=True` | Auto-detects disconnected firewall sockets |
| **Transactions** | `with engine.begin() as conn:` | Guaranteed atomic commit and rollback |
| **Credentials** | `.streamlit/secrets.toml` | Prevents hardcoded secret credentials in Git |
| **Error Handling** | Catch `IntegrityError` & `OperationalError` | User-friendly alerts without leaking server logs |
"""
)
