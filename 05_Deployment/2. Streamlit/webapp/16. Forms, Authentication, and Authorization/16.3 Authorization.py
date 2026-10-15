import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import time

st.set_page_config(
    page_title="16.3 Authorization",
    page_icon="🛡️",
    layout="wide"
)

st.title("16.3 Authorization & Access Control in Streamlit")
st.caption("Implementing Role-Based Access Control (RBAC), Row-Level Security (RLS), admin gates, parameterized database isolation, and immutable audit logging.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Authorization Security Architecture

Authorization (AuthZ) governs what an authenticated identity is permitted to execute.
Production Streamlit apps enforce access control across three defensive boundaries:
1. **View & Navigation Boundary**: Selectively rendering pages, tabs, and widgets based on user roles.
2. **Server-Side Mutation Boundary**: Verifying permissions in Python before executing operations (e.g. model deployment, deletion).
3. **Data Access Boundary**: Enforcing Row-Level Security (RLS) and tenant isolation directly at the database query level.

""")

st.graphviz_chart(
    r'''
    digraph AuthorizationRBAC {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        session [label="Authenticated User Session\nst.session_state['user'] = {'role': user_role}", fillcolor="#DBEAFE", color="#1D4ED8"];
        guard [label="Server-Side Guard (RBAC)\nrequire_role(role, min_level)\n(Stops unauthorized execution)", fillcolor="#FEF3C7", color="#D97706"];
        admin_v [label="Admin View\nSystem config, DB reset, audit logs", fillcolor="#FEE2E2", color="#DC2626"];
        analyst_v [label="Analyst View\nModel scoring, batch runs", fillcolor="#FEF3C7", color="#D97706"];
        viewer_v [label="Viewer View\nRead-only dashboard metrics", fillcolor="#D1FAE5", color="#059669"];

        session -> guard [label="Evaluates permissions"];
        guard -> admin_v [label="Role = Admin"];
        guard -> analyst_v [label="Role = Analyst"];
        guard -> viewer_v [label="Role = Viewer"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. RBAC Matrix",
    "2. Admin-Only Views",
    "3. User Data & RLS",
    "4. Restricting SQL Queries",
    "5. Protecting Mutations",
    "6. Permission Scopes",
    "7. Server-Side Checks",
    "8. Audit Logging Ledger"
])

# =============================================================================
# 1. RBAC MATRIX
# =============================================================================
with tabs[0]:
    st.header("1. Role-Based Access Control (RBAC) Architecture")
    st.markdown(r"""
    RBAC decouples individual users from permissions by assigning permissions to **Roles** 
    and roles to **Users**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Granular Enterprise Role-to-Scope Matrix
ROLE_PERMISSIONS = {
    "Viewer": {
        "models:read", "leaderboard:read"
    },
    "Data Scientist": {
        "models:read", "leaderboard:read", 
        "experiments:create", "experiments:run", "weights:download"
    },
    "Platform Admin": {
        "models:read", "leaderboard:read", 
        "experiments:create", "experiments:run", "weights:download",
        "models:deploy", "models:delete", "tenants:manage", "audit:read"
    }
}

def user_has_permission(user_role: str, required_scope: str) -> bool:
    scopes = ROLE_PERMISSIONS.get(user_role, set())
    return required_scope in scopes
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Active RBAC Matrix

            | Role | `models:read` | `experiments:run` | `models:deploy` | `tenants:manage` |
            | :--- | :---: | :---: | :---: | :---: |
            | **Viewer** | ✅ | ❌ | ❌ | ❌ |
            | **Data Scientist** | ✅ | ✅ | ❌ | ❌ |
            | **Platform Admin** | ✅ | ✅ | ✅ | ✅ |
            """)
            st.caption("Users receive only the minimal set of privileges required for their role.")

# =============================================================================
# 2. ADMIN-ONLY VIEWS
# =============================================================================
with tabs[1]:
    st.header("2. Admin-Only Views & Navigation Gating")
    st.markdown(r"""
    Restricting administrative dashboards, billing portals, and tenant management views 
    strictly to verified admin identities.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

role = st.session_state.get("active_role", "Viewer")

if role != "Platform Admin":
    st.error("🚨 403 Forbidden: You do not have permission to access the Administrative Panel.")
    st.info("Contact security@enterprise.ai to request elevated permissions.")
else:
    st.success("👑 Administrator Access Granted")
    st.subheader("Global Tenant Management")
    st.dataframe(tenant_directory)
    if st.button("Purge Stale Evaluation Caches"):
        st.success("Cache cleared.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sel_role = st.selectbox("Simulate Session Role", ["Viewer", "Data Scientist", "Platform Admin"], key="admin_view_role_sel")

            if sel_role != "Platform Admin":
                st.error("🚨 403 Forbidden: Administrative Panel restricted.")
                st.caption("Contact security@enterprise.ai to request elevated permissions.")
            else:
                st.success("👑 Administrator Access Verified")
                st.caption("Global Tenant Directory (Admin Eyes Only):")
                st.dataframe(pd.DataFrame({
                    "Tenant ID": ["org_01", "org_02", "org_03"],
                    "Cluster": ["us-east-1", "eu-central-1", "ap-southeast-1"],
                    "Monthly Quota": ["$15,000", "$42,000", "$8,500"]
                }), width="stretch", hide_index=True)

# =============================================================================
# 3. USER DATA & RLS
# =============================================================================
with tabs[2]:
    st.header("3. User-Specific Data & Row-Level Security (RLS)")
    st.markdown(r"""
    In multi-tenant or multi-user applications, users must only see records they authored or belong to their tenant.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Multi-tenant experiments database
ALL_RUNS = pd.DataFrame([
    {"run_id": "EXP-101", "owner_id": "usr_sarah", "model": "Llama-3-8B", "val_loss": 1.42},
    {"run_id": "EXP-102", "owner_id": "usr_alex", "model": "Mistral-7B", "val_loss": 1.38},
    {"run_id": "EXP-103", "owner_id": "usr_sarah", "model": "Qwen-2.5", "val_loss": 1.29},
    {"run_id": "EXP-104", "owner_id": "usr_liam", "model": "Phi-3.5", "val_loss": 1.55}
])

current_user_id = "usr_sarah"
current_user_role = "Data Scientist"

# Admin sees all records; standard users see only their own
if current_user_role == "Platform Admin":
    visible_runs = ALL_RUNS
else:
    visible_runs = ALL_RUNS[ALL_RUNS["owner_id"] == current_user_id]

st.dataframe(visible_runs, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            ALL_RUNS = pd.DataFrame([
                {"run_id": "EXP-101", "owner_id": "usr_sarah", "model": "Llama-3-8B", "val_loss": 1.42},
                {"run_id": "EXP-102", "owner_id": "usr_alex", "model": "Mistral-7B", "val_loss": 1.38},
                {"run_id": "EXP-103", "owner_id": "usr_sarah", "model": "Qwen-2.5", "val_loss": 1.29},
                {"run_id": "EXP-104", "owner_id": "usr_liam", "model": "Phi-3.5", "val_loss": 1.55}
            ])

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                curr_user = st.selectbox("Active User Context", ["usr_sarah", "usr_alex", "usr_liam"], key="rls_user_sel")
            with r_col2:
                as_admin = st.checkbox("View with Admin Override", value=False, key="rls_admin_chk")

            if as_admin:
                filtered_runs = ALL_RUNS
            else:
                filtered_runs = ALL_RUNS[ALL_RUNS["owner_id"] == curr_user]

            st.caption(f"Showing {len(filtered_runs)} authorized experiment records:")
            st.dataframe(filtered_runs, width="stretch", hide_index=True)

# =============================================================================
# 4. RESTRICTING SQL QUERIES
# =============================================================================
with tabs[3]:
    st.header("4. Restricting Database Queries at the SQL Level")
    st.markdown(r"""
    **Security Anti-Pattern**: Querying `SELECT * FROM table` into a DataFrame and filtering in Python allows unauthorized rows into server memory.
    **Security Best Practice**: Bind the user's verified identity directly inside parameterized SQL query predicates.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# ❌ VULNERABLE ANTI-PATTERN:
# Ingests all tenants' private data into RAM, then filters client-side
df_all = pd.read_sql("SELECT * FROM fine_tune_jobs", connection)
df_user = df_all[df_all["tenant_id"] == session_tenant_id]

# ✅ SECURE DEFENSIVE PATTERN:
# Server-side enforcement with parameterized SQL:
safe_query = """
    SELECT job_id, model_name, training_loss, status
    FROM fine_tune_jobs
    WHERE tenant_id = :tenant_id
      AND (owner_id = :user_id OR :is_admin = 1)
"""
params = {
    "tenant_id": current_user["tenant_id"],
    "user_id": current_user["user_id"],
    "is_admin": 1 if "admin" in current_user["roles"] else 0
}
df_safe = pd.read_sql(safe_query, connection, params=params)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Database Isolation Safeguards

            1. **Zero Client Leakage**: Unauthorized tenant records never cross the network socket from the database to Streamlit.
            2. **Parameterized Binding**: Eliminates SQL injection vulnerabilities by separating SQL grammar from parameters.
            3. **Database-Native RLS**: When using PostgreSQL, consider leveraging native `CREATE POLICY` Row-Level Security rules.
            """)

# =============================================================================
# 5. PROTECTING MUTATIONS
# =============================================================================
with tabs[4]:
    st.header("5. Protecting Administrative Mutations with `@st.dialog`")
    st.markdown(r"""
    Destructive operations (e.g., deleting checkpoints, altering access roles) require explicit confirmation 
    and server-side re-verification inside `@st.dialog`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

@st.dialog("⚠️ Confirm Destructive Action")
def confirm_delete_dialog(checkpoint_id: str):
    st.warning(f"Are you sure you want to permanently delete checkpoint `{checkpoint_id}`?")
    st.caption("This action is irreversible and purges weights from S3.")
    
    col_c, col_d = st.columns(2)
    if col_c.button("Cancel"):
        st.rerun()
    if col_d.button("Confirm Delete", type="primary"):
        # Server-side authorization re-check
        if st.session_state.get("role") == "Platform Admin":
            st.success(f"Checkpoint {checkpoint_id} deleted.")
            # Audit log event...
        else:
            st.error("Unauthorized action blocked.")
        st.rerun()

if st.button("Delete Model Checkpoint"):
    confirm_delete_dialog("ckpt_llama3_70b_v4")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Interactive Mutation Protection Trigger:")
            
            @st.dialog("⚠️ Confirm Destructive Action")
            def demo_confirm_dialog(checkpoint_id: str):
                st.warning(f"Permanently delete `{checkpoint_id}`?")
                st.caption("This action purges serialized weights from S3 object storage.")
                c1, c2 = st.columns(2)
                if c1.button("Cancel", key="dlg_cancel"):
                    st.rerun()
                if c2.button("Confirm Delete", type="primary", key="dlg_confirm"):
                    st.success(f"Purged {checkpoint_id}.")
                    time.sleep(0.5)
                    st.rerun()

            if st.button("Delete Model Checkpoint (Protected Action)", key="trigger_del_btn"):
                demo_confirm_dialog("ckpt_llama3_70b_v4")

# =============================================================================
# 6. PERMISSION SCOPES
# =============================================================================
with tabs[5]:
    st.header("6. Granular Permission Scopes Model")
    st.markdown(r"""
    Fine-grained permissions follow the `resource:action` naming convention (e.g., `models:deploy`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
class SecurityContext:
    def __init__(self, user_id: str, scopes: set[str]):
        self.user_id = user_id
        self.scopes = scopes
        
    def allows(self, required_scope: str) -> bool:
        return required_scope in self.scopes

# Example usage on action triggers
ctx = SecurityContext("usr_sarah", {"models:read", "experiments:run"})

def trigger_deployment(model_name: str, ctx: SecurityContext):
    if not ctx.allows("models:deploy"):
        raise PermissionError(f"Missing required scope 'models:deploy'")
    # Execute deployment logic...
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            active_scopes = st.multiselect(
                "User Assigned Scopes",
                ["models:read", "experiments:run", "weights:download", "models:deploy"],
                default=["models:read", "experiments:run"],
                key="scopes_demo_ms"
            )

            st.caption("Action Permission Checkers:")
            for scope, desc in [
                ("models:read", "View Model Telemetry"),
                ("experiments:run", "Launch Training Pipeline"),
                ("models:deploy", "Deploy to Production Gateway")
            ]:
                allowed = scope in active_scopes
                badge = "🟢 Allowed" if allowed else "🔴 Blocked"
                st.write(f"- **{desc}** (`{scope}`): {badge}")

# =============================================================================
# 7. SERVER-SIDE CHECKS
# =============================================================================
with tabs[6]:
    st.header("7. Server-Side vs Client-Side Authorization Checks")
    st.markdown(r"""
    Never rely on hiding DOM elements or disabling buttons for security.
    An attacker can manipulate browser DOM nodes or trigger WebSocket messages directly. 
    **Always re-verify permissions server-side in Python before executing mutations.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Server-Side Decorator Pattern**")
        st.code(r'''
import functools
import streamlit as st

def require_scope(required_scope: str):
    """Decorator enforcing server-side scope authorization."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_scopes = st.session_state.get("user_scopes", set())
            if required_scope not in user_scopes:
                st.error(f"Security Alert: Blocked unauthorized execution of '{func.__name__}'")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_scope("models:deploy")
def deploy_model_to_production(model_id: str):
    st.success(f"Model '{model_id}' successfully deployed to inference cluster.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Defense-in-Depth Principle

            1. **UI Layer (Usability)**: Conditionally disable or hide buttons so users understand their permissions.
            2. **Python Layer (Security)**: The backend function independently verifies the active session role before writing to disk or calling APIs.
            3. **Database Layer (Storage)**: Relational foreign keys and Row-Level Security policies reject unauthorized writes.
            """)

# =============================================================================
# 8. AUDIT LOGGING LEDGER
# =============================================================================
with tabs[7]:
    st.header("8. Immutable Audit Logging for Sensitive Actions")
    st.markdown(r"""
    Compliance standards (SOC-2, HIPAA, ISO 27001) require auditing every sensitive action:
    $$\text{Audit Event} = \langle\text{Timestamp, Actor, Action, Target, Outcome, IP}\rangle$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import datetime
import pandas as pd

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

def record_audit_event(actor: str, action: str, target: str, outcome: str):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "actor": actor,
        "action": action,
        "target": target,
        "outcome": outcome
    }
    st.session_state.audit_log.insert(0, entry)

# Trigger a logged administrative event
if st.button("Promote Checkpoint to Staging"):
    record_audit_event(
        actor="alex.m@enterprise.ai",
        action="PROMOTE_CHECKPOINT",
        target="ckpt_v4_holdout",
        outcome="SUCCESS"
    )

st.dataframe(pd.DataFrame(st.session_state.audit_log), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if "demo_audit_ledger" not in st.session_state:
                st.session_state.demo_audit_ledger = [
                    {"timestamp": "2024-10-12 14:15:22 UTC", "actor": "sarah.c@ai.com", "action": "LOGIN_OIDC", "target": "AUTH_GATEWAY", "outcome": "SUCCESS"},
                    {"timestamp": "2024-10-12 14:20:05 UTC", "actor": "sarah.c@ai.com", "action": "LAUNCH_TRAINING", "target": "JOB-9102", "outcome": "SUCCESS"},
                    {"timestamp": "2024-10-12 14:35:10 UTC", "actor": "guest_user", "action": "ACCESS_ADMIN_PANEL", "target": "ADMIN_VIEW", "outcome": "DENIED (403)"}
                ]

            if st.button("Simulate Administrative Promotion", key="audit_trigger_btn"):
                st.session_state.demo_audit_ledger.insert(0, {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "actor": "alex.m@ai.com",
                    "action": "PROMOTE_CHECKPOINT",
                    "target": "ckpt_llama3_prod",
                    "outcome": "SUCCESS"
                })

            st.caption("Immutable Audit Ledger (Latest Events First):")
            st.dataframe(pd.DataFrame(st.session_state.demo_audit_ledger), width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Authorization Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Client-Side Hiding** | Hiding admin buttons with CSS or HTML | Gate execution directly in Python (`if user.has_role("admin"):`). |
| **SQL Data Leakage** | `SELECT * FROM tbl` followed by Python filtering | Filter by `tenant_id` and `user_id` inside parameterized SQL WHERE clauses. |
| **Destructive Actions** | Immediate deletion on single click | Protect with `@st.dialog` confirmation and explicit authorization re-checks. |
| **Compliance Gaps** | No record of who deployed or purged models | Log every state-mutating operation to an immutable audit table. |
""")
