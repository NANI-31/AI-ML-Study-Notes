import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="SQLite Integration in Streamlit",
    page_icon="🗃️",
    layout="wide",
)

st.title("🗃️ SQLite Database Integration — Complete Guide")
st.markdown(
    """
SQLite is Python's built-in, serverless, zero-configuration relational database engine. 
Because it requires no external database server, it is the ideal persistence backend for 
local prototypes, internal enterprise tools, edge deployments, and lightweight analytics dashboards. 
In Streamlit, mastering SQLite involves understanding **thread-safety (`check_same_thread=False`)**, 
**Write-Ahead Logging (WAL mode)**, **parameterized CRUD pipelines**, and **interactive table synchronization**.
"""
)

st.divider()

# ==============================================================================
# 1. Connecting with sqlite3 (Thread-Safety & WAL Mode)
# ==============================================================================
st.header("1. Connecting with `sqlite3` in Streamlit", divider="rainbow")
st.markdown(
    r"""
### Two Essential Flags for Streamlit & SQLite

1. **`check_same_thread=False`**: 
   By default, SQLite restricts connection usage to the exact thread that initialized it. Because Streamlit handles user requests across dynamic thread pools, omitting this flag causes `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`.
2. **`PRAGMA journal_mode=WAL;` (Write-Ahead Logging)**: 
   Default SQLite uses rollback journaling, which locks the entire database file during writes. WAL mode allows **multiple concurrent readers** to read while a writer writes!

```python
@st.cache_resource
def get_connection(db_file: str = "app.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_file, check_same_thread=False, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL;")  # Massive concurrency boost
    conn.row_factory = sqlite3.Row            # Dict-like row indexing
    return conn
```
"""
)

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("**💻 Python Source Code (Cached SQLite Singleton)**")
    code_conn = '''import sqlite3
import streamlit as st

@st.cache_resource
def init_sqlite_db(db_path: str = "enterprise_models.db"):
    # check_same_thread=False is mandatory for Streamlit:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Enable WAL mode for high read/write concurrency:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

conn = init_sqlite_db(":memory:")'''
    st.code(code_conn, language="python")

with col_c2:
    st.markdown("**🖥️ Live Interactive Connection Inspector**")
    @st.cache_resource
    def get_sqlite_master():
        c = sqlite3.connect(":memory:", check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL;")
        c.row_factory = sqlite3.Row
        return c

    master_conn = get_sqlite_master()
    cur_journal = master_conn.execute("PRAGMA journal_mode;").fetchone()[0]

    with st.container(border=True):
        st.markdown("##### 🔌 SQLite Connection Telemetry")
        st.write(f"• **Database Engine**: `SQLite {sqlite3.sqlite_version}`")
        st.write(f"• **Active Journal Mode**: `{cur_journal.upper()}`")
        st.write(f"• **Connection Instance ID**: `{hex(id(master_conn))}`")
        st.success("Persistent SQLite connection singleton ready for operations.")

st.divider()

# ==============================================================================
# 2. Creating Database Tables
# ==============================================================================
st.header("2. Creating Database Tables (`CREATE TABLE IF NOT EXISTS`)", divider="rainbow")
st.markdown(
    r"""
Always use `IF NOT EXISTS` so the DDL executes safely across app reruns without crashing:
- **`INTEGER PRIMARY KEY AUTOINCREMENT`**: Generates unique row identifiers.
- **`CHECK` Constraints**: Enforces domain integrity (e.g., scores must lie between 0.0 and 100.0).
- **`DEFAULT CURRENT_TIMESTAMP`**: Automatically timestamps record creation.
"""
)

col_ddl1, col_ddl2 = st.columns(2)

with col_ddl1:
    st.markdown("**💻 Python Source Code (DDL Initialization)**")
    code_ddl = '''def create_schema(conn: sqlite3.Connection):
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            accuracy_score REAL CHECK(accuracy_score >= 0.0 AND accuracy_score <= 1.0),
            status TEXT DEFAULT 'Staging' CHECK(status IN ('Production', 'Staging', 'Deprecated')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

create_schema(conn)'''
    st.code(code_ddl, language="python")

with col_ddl2:
    st.markdown("**🖥️ Schema Execution & Validation**")
    with master_conn:
        master_conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            accuracy_score REAL CHECK(accuracy_score >= 0.0 AND accuracy_score <= 1.0),
            status TEXT DEFAULT 'Staging' CHECK(status IN ('Production', 'Staging', 'Deprecated')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

    # Query sqlite_master to verify table exists
    schema_info = pd.read_sql_query(
        "PRAGMA table_info(model_registry);",
        master_conn
    )
    with st.container(border=True):
        st.markdown("##### 📐 Discovered Table Columns (`PRAGMA table_info`)")
        st.dataframe(schema_info[["cid", "name", "type", "notnull", "pk"]], width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 3. Inserting Records (Single & Bulk executemany)
# ==============================================================================
st.header("3. Inserting Records (`cur.execute` vs `cur.executemany`)", divider="rainbow")
st.markdown(
    r"""
- **Single Record**: Use `cur.execute(sql, tuple)` with `cur.lastrowid` to capture the auto-generated ID.
- **Bulk Insert**: Use `cur.executemany(sql, list_of_tuples)` for high-throughput batch operations (up to 50× faster than a Python loop).
"""
)

col_ins1, col_ins2 = st.columns(2)

with col_ins1:
    st.markdown("**💻 Python Source Code (Batch Insert)**")
    code_ins = '''# Batch insert initial seed records:
models_seed = [
    ("Claude 3.7 Sonnet", "Anthropic", 200000, 0.924, "Production"),
    ("GPT-4.5 Omni", "OpenAI", 128000, 0.902, "Production"),
    ("DeepSeek R1", "DeepSeek", 128000, 0.936, "Production"),
    ("Llama 3.3 70B", "Meta", 128000, 0.821, "Staging"),
    ("Qwen 2.5 72B", "Alibaba", 128000, 0.843, "Staging")
]

with conn:
    conn.executemany("""
    INSERT OR IGNORE INTO model_registry 
    (model_name, provider, context_window, accuracy_score, status)
    VALUES (?, ?, ?, ?, ?)
    """, models_seed)'''
    st.code(code_ins, language="python")

with col_ins2:
    st.markdown("**🖥️ Live Ingestion Execution**")
    models_seed = [
        ("Claude 3.7 Sonnet", "Anthropic", 200000, 0.924, "Production"),
        ("GPT-4.5 Omni", "OpenAI", 128000, 0.902, "Production"),
        ("DeepSeek R1", "DeepSeek", 128000, 0.936, "Production"),
        ("Llama 3.3 70B", "Meta", 128000, 0.821, "Staging"),
        ("Qwen 2.5 72B", "Alibaba", 128000, 0.843, "Staging")
    ]
    with master_conn:
        master_conn.executemany("""
        INSERT OR IGNORE INTO model_registry 
        (model_name, provider, context_window, accuracy_score, status)
        VALUES (?, ?, ?, ?, ?)
        """, models_seed)

    count_res = master_conn.execute("SELECT COUNT(*) FROM model_registry;").fetchone()[0]
    with st.container(border=True):
        st.success(f"✅ Ingestion Verified: Database contains **{count_res}** active records.")
        st.caption("Duplicates are safely ignored via `INSERT OR IGNORE`.")

st.divider()

# ==============================================================================
# 4. Reading Records & 7. Displaying Database Results
# ==============================================================================
st.header("4. Reading Records & 7. Displaying Formatted Results", divider="rainbow")
st.markdown(
    r"""
Streamlit and Pandas integrate seamlessly with SQLite via **`pd.read_sql_query(sql, conn)`**:
- Direct parameterized query execution.
- Display with rich formatting via **`st.column_config`** (progress bars, number formatting).
"""
)

col_rd1, col_rd2 = st.columns(2)

with col_rd1:
    st.markdown("**💻 Python Source Code (Filtered Read & Styled Display)**")
    code_read = '''# Read filtered records:
query = """
SELECT id, model_name, provider, context_window, accuracy_score, status, created_at
FROM model_registry
WHERE status = ?
ORDER BY accuracy_score DESC
"""
df = pd.read_sql_query(query, conn, params=("Production",))

# Rich column configuration display:
st.dataframe(
    df,
    column_config={
        "accuracy_score": st.column_config.ProgressColumn(
            "Pass@1 Accuracy",
            format="%.1f%%",
            min_value=0.0,
            max_value=1.0
        ),
        "context_window": st.column_config.NumberColumn(
            "Context Tokens",
            format="%d tok"
        )
    },
    width="stretch",
    hide_index=True
)'''
    st.code(code_read, language="python")

with col_rd2:
    st.markdown("**🖥️ Live Filtered Query & Column Config Display**")
    status_filter = st.radio("Filter by Deployment Status", ["All", "Production", "Staging"], horizontal=True, key="rad_status_filter")
    
    if status_filter == "All":
        read_sql = "SELECT id, model_name, provider, context_window, accuracy_score, status FROM model_registry ORDER BY accuracy_score DESC;"
        df_display = pd.read_sql_query(read_sql, master_conn)
    else:
        read_sql = "SELECT id, model_name, provider, context_window, accuracy_score, status FROM model_registry WHERE status = ? ORDER BY accuracy_score DESC;"
        df_display = pd.read_sql_query(read_sql, master_conn, params=(status_filter,))

    st.dataframe(
        df_display,
        column_config={
            "accuracy_score": st.column_config.ProgressColumn(
                "Accuracy Score",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0
            ),
            "context_window": st.column_config.NumberColumn(
                "Context Window",
                format="%d tok"
            )
        },
        width="stretch",
        hide_index=True
    )

st.divider()

# ==============================================================================
# 5. Updating Records & 6. Deleting Records
# ==============================================================================
st.header("5. Updating & 6. Deleting Records Safely", divider="rainbow")
st.markdown(
    r"""
Always use parameterized `UPDATE` and `DELETE` queries matching the primary key `id`:
- Check `cur.rowcount` to verify how many records were affected.
- Wrap modifications inside `with conn:` to ensure atomic commits!
"""
)

col_upd1, col_upd2 = st.columns(2)

with col_upd1:
    st.markdown("**💻 Python Source Code (Atomic Update & Delete)**")
    code_mod = '''# 1. Update Record
def update_model_status(model_id: int, new_status: str):
    with conn:
        cur = conn.execute(
            "UPDATE model_registry SET status = ? WHERE id = ?",
            (new_status, model_id)
        )
        return cur.rowcount > 0

# 2. Delete Record
def delete_model(model_id: int):
    with conn:
        cur = conn.execute(
            "DELETE FROM model_registry WHERE id = ?",
            (model_id,)
        )
        return cur.rowcount > 0'''
    st.code(code_mod, language="python")

with col_upd2:
    st.markdown("**🖥️ Live Quick Actions Demonstration**")
    all_recs = pd.read_sql_query("SELECT id, model_name, status FROM model_registry", master_conn)
    
    if not all_recs.empty:
        target_name = st.selectbox("Select Model for Quick Actions", all_recs["model_name"].tolist(), key="sel_mod_quick")
        target_id = int(all_recs[all_recs["model_name"] == target_name]["id"].values[0])
        current_st = all_recs[all_recs["model_name"] == target_name]["status"].values[0]

        btn_promote, btn_deprecate = st.columns(2)
        with btn_promote:
            if st.button("🚀 Promote to Production", key="btn_prom_demo"):
                with master_conn:
                    master_conn.execute("UPDATE model_registry SET status = 'Production' WHERE id = ?", (target_id,))
                st.success(f"Promoted `{target_name}` to Production!")
                st.rerun()

        with btn_deprecate:
            if st.button("⚠️ Set to Deprecated", key="btn_depr_demo"):
                with master_conn:
                    master_conn.execute("UPDATE model_registry SET status = 'Deprecated' WHERE id = ?", (target_id,))
                st.warning(f"Marked `{target_name}` as Deprecated!")
                st.rerun()
    else:
        st.info("No records available.")

st.divider()

# ==============================================================================
# 8. Full Production CRUD Application Capstone
# ==============================================================================
st.header("8. Building a Full Production CRUD Application", divider="rainbow")
st.markdown(
    """
The capstone below integrates all 8 concepts into an enterprise-grade 
**AI Model & Benchmark Governance Registry** featuring multi-tab workflows:
"""
)

crud_tab1, crud_tab2, crud_tab3, crud_tab4 = st.tabs([
    "📋 Model Catalog (Read)",
    "➕ Register Model (Create)",
    "✏️ Edit Benchmark (Update)",
    "🗑️ Retire Model (Delete)"
])

# ----------------- Tab 1: Read & Search -----------------
with crud_tab1:
    st.markdown("##### 🔍 Search & Inspect Model Registry")
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_query = st.text_input("Search by Model Name or Provider", "", key="txt_search_reg")
    with search_col2:
        prov_filter = st.selectbox("Filter Provider", ["All"] + list(pd.read_sql_query("SELECT DISTINCT provider FROM model_registry", master_conn)["provider"]), key="sel_prov_reg")

    base_query = "SELECT id, model_name, provider, context_window, accuracy_score, status, created_at FROM model_registry WHERE 1=1"
    params = []

    if search_query.strip():
        base_query += " AND (model_name LIKE ? OR provider LIKE ?)"
        params.extend([f"%{search_query.strip()}%", f"%{search_query.strip()}%"])
    if prov_filter != "All":
        base_query += " AND provider = ?"
        params.append(prov_filter)

    base_query += " ORDER BY accuracy_score DESC"
    df_catalog = pd.read_sql_query(base_query, master_conn, params=params)

    st.dataframe(
        df_catalog,
        column_config={
            "accuracy_score": st.column_config.ProgressColumn(
                "Pass@1 Accuracy",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0
            ),
            "context_window": st.column_config.NumberColumn(
                "Context Window",
                format="%d tok"
            )
        },
        width="stretch",
        hide_index=True
    )

# ----------------- Tab 2: Create -----------------
with crud_tab2:
    st.markdown("##### ➕ Register New Model Candidate")
    with st.form("form_register_model", clear_on_submit=True):
        reg_c1, reg_c2 = st.columns(2)
        with reg_c1:
            new_name = st.text_input("Model Identifier", placeholder="e.g. Mistral Large 2")
            new_provider = st.selectbox("Provider", ["Anthropic", "OpenAI", "DeepSeek", "Meta", "Mistral", "Google"])
        with reg_c2:
            new_ctx = st.selectbox("Context Window (Tokens)", [32000, 64000, 128000, 200000, 1000000])
            new_acc = st.slider("Benchmark Accuracy (%)", 50.0, 100.0, 85.0, 0.5) / 100.0
            new_status = st.selectbox("Initial Deployment Status", ["Staging", "Production", "Deprecated"])

        btn_submit_reg = st.form_submit_button("Deploy & Register Model")

        if btn_submit_reg:
            if new_name.strip():
                try:
                    with master_conn:
                        master_conn.execute("""
                        INSERT INTO model_registry (model_name, provider, context_window, accuracy_score, status)
                        VALUES (?, ?, ?, ?, ?)
                        """, (new_name.strip(), new_provider, new_ctx, new_acc, new_status))
                    st.success(f"Successfully registered `{new_name}`!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error(f"Error: Model `{new_name}` is already registered in database!")
            else:
                st.warning("Model name cannot be empty.")

# ----------------- Tab 3: Update -----------------
with crud_tab3:
    st.markdown("##### ✏️ Update Existing Model Benchmark & Status")
    df_for_update = pd.read_sql_query("SELECT id, model_name, accuracy_score, status FROM model_registry", master_conn)
    if not df_for_update.empty:
        upd_choice = st.selectbox("Select Model to Edit", df_for_update["model_name"].tolist(), key="sel_upd_model")
        row_match = df_for_update[df_for_update["model_name"] == upd_choice].iloc[0]

        with st.form("form_update_model"):
            upd_c1, upd_c2 = st.columns(2)
            with upd_c1:
                updated_acc = st.slider("Updated Accuracy (%)", 50.0, 100.0, float(row_match["accuracy_score"] * 100), 0.5) / 100.0
            with upd_c2:
                updated_status = st.selectbox("Updated Status", ["Production", "Staging", "Deprecated"], index=["Production", "Staging", "Deprecated"].index(row_match["status"]))

            btn_do_update = st.form_submit_button("Save Benchmark Updates")

            if btn_do_update:
                with master_conn:
                    master_conn.execute("""
                    UPDATE model_registry
                    SET accuracy_score = ?, status = ?
                    WHERE id = ?
                    """, (updated_acc, updated_status, int(row_match["id"])))
                st.success(f"Updated `{upd_choice}`: Accuracy = {updated_acc*100:.1f}%, Status = {updated_status}!")
                st.rerun()
    else:
        st.info("No records available to edit.")

# ----------------- Tab 4: Delete -----------------
with crud_tab4:
    st.markdown("##### 🗑️ Retire & Delete Model Record")
    df_for_del = pd.read_sql_query("SELECT id, model_name FROM model_registry", master_conn)
    if not df_for_del.empty:
        del_choice = st.selectbox("Select Model to Permanently Remove", df_for_del["model_name"].tolist(), key="sel_del_perm")
        row_del_id = int(df_for_del[df_for_del["model_name"] == del_choice]["id"].values[0])

        if st.button(f"Permanently Delete '{del_choice}'", type="primary", key="btn_do_del"):
            with master_conn:
                master_conn.execute("DELETE FROM model_registry WHERE id = ?", (row_del_id,))
            st.warning(f"Model `{del_choice}` (ID: {row_del_id}) was removed from the database.")
            st.rerun()
    else:
        st.info("No records available to delete.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 SQLite Integration Best Practices", divider="rainbow")
st.markdown(
    r"""
| Task | Best Practice | Code Example |
| :--- | :--- | :--- |
| **Multi-Threading** | Allow cross-thread access | `sqlite3.connect(db, check_same_thread=False)` |
| **Concurrency** | Enable WAL journaling mode | `conn.execute("PRAGMA journal_mode=WAL;")` |
| **Connection Scope** | Cache singleton across reruns | `@st.cache_resource` |
| **Row Formatting** | Access columns by name | `conn.row_factory = sqlite3.Row` |
| **Transaction Safety**| Automatic commit/rollback | `with conn: conn.execute(...)` |
| **Batch Ingestion** | High-performance bulk insert | `conn.executemany(sql, list_of_tuples)` |
| **Pandas Integration**| Direct DataFrame ingestion | `pd.read_sql_query(sql, conn, params=(...))` |
| **Injection Defense** | Always parameterize | `cur.execute("... WHERE id = ?", (val,))` |
"""
)
