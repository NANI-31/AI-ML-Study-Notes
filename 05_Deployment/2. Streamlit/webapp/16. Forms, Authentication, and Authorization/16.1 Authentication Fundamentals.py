import streamlit as st
import hashlib
import secrets
import datetime
import time

st.set_page_config(
    page_title="16.1 Authentication Fundamentals",
    page_icon="🔒",
    layout="wide"
)

st.title("16.1 Authentication Fundamentals in Streamlit")
st.caption("Core security engineering: AuthN vs AuthZ, user identity, cryptographic password hashing, session expiration, RBAC access control, and threat modeling.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Authentication & Authorization Security Model

Securing Streamlit enterprise applications requires cleanly decoupling **Authentication (AuthN)** from **Authorization (AuthZ)**:

""")

st.graphviz_chart(
    r'''
    digraph AuthFundamentals {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        client [label="Client Browser Session\nUser inputs username + password", fillcolor="#EFF6FF", color="#3B82F6"];
        engine [label="Validation Engine\n• Lookup user in database\n• bcrypt.checkpw(password, hash)\n• Enforce lockout & salt verification", fillcolor="#FEF3C7", color="#D97706"];
        state [label="Session State Initialization\nst.session_state['authenticated'] = True\nst.session_state['user'] = {...}", fillcolor="#D1FAE5", color="#059669"];

        client -> engine [label="Submits credentials"];
        engine -> state [label="Hash match -> Success"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. AuthN vs AuthZ",
    "2. User Identity",
    "3. Login & Logout Flows",
    "4. Password Hashing",
    "5. Session Expiration",
    "6. User Roles (RBAC)",
    "7. Access Control Gates",
    "8. Secure Design Principles"
])

# =============================================================================
# 1. AUTHN VS AUTHZ
# =============================================================================
with tabs[0]:
    st.header("1. Authentication (AuthN) versus Authorization (AuthZ)")
    st.markdown(r"""
    A fundamental security flaw in amateur applications is confusing **identity verification** with **permission checking**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Architectural Comparison**")
        st.markdown(r"""
        | Dimension | Authentication (AuthN) | Authorization (AuthZ) |
        | :--- | :--- | :--- |
        | **Core Question** | *"Who are you?"* | *"What are you permitted to do?"* |
        | **Mechanism** | Passwords, MFA OTPs, SSO OAuth2, Passkeys. | Roles (RBAC), Access Control Lists (ACLs), Scopes. |
        | **Occurrence** | Once at session startup or login. | Continuous on every widget action and page transition. |
        | **Failure Result** | HTTP 401 Unauthorized (Identity unproven). | HTTP 403 Forbidden (Identity known, access denied). |
        | **Streamlit State** | `st.session_state["authenticated"] = True` | `if "admin" in st.session_state["roles"]:` |
        """)

    with col2:
        st.markdown("**🖥️ Live Security Check Demonstration**")
        with st.container(border=True):
            user_logged_in = st.toggle("Simulate User Authenticated (AuthN)", value=True, key="demo_authn_tog")
            user_role = st.selectbox("Assigned Role (AuthZ)", ["Viewer", "Data Scientist", "Platform Admin"], key="demo_authz_role")

            st.write("---")
            if not user_logged_in:
                st.error("🚨 401 Unauthorized: Please log in to verify your identity.")
            else:
                st.success(f"✅ 200 OK: User authenticated as `{user_role}`.")
                if user_role == "Platform Admin":
                    st.button("🔥 Delete Production Checkpoint (Admin Only)", type="primary")
                elif user_role == "Data Scientist":
                    st.info("ℹ️ Permitted to trigger fine-tuning jobs and view validation loss.")
                else:
                    st.warning("🔒 Read-Only Viewer: Administrative deployment controls are hidden.")

# =============================================================================
# 2. USER IDENTITY
# =============================================================================
with tabs[1]:
    st.header("2. User Identity Modeling & Claims")
    st.markdown(r"""
    In Streamlit, authenticated user identity is stored as an immutable dictionary inside `st.session_state`.
    Identity records should contain unique identifiers (UUIDs), user handles, verified emails, and authorization claims.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Initializing structured user identity context
user_profile = {
    "user_id": "usr_94a821e0bc",
    "username": "sarah_chen",
    "email": "sarah.chen@enterprise.ai",
    "tenant_id": "org_nlp_research",
    "roles": ["researcher", "model_evaluator"],
    "permissions": ["models:read", "experiments:write"],
    "authenticated_at": "2024-10-12T14:30:00Z"
}

st.session_state["current_user"] = user_profile

# Display user badge in sidebar or header
st.write(f"Logged in as **{user_profile['username']}** ({user_profile['email']})")
st.caption(f"Tenant: `{user_profile['tenant_id']}` | Roles: `{', '.join(user_profile['roles'])}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            mock_user = {
                "user_id": "usr_94a821e0bc",
                "username": "sarah_chen",
                "email": "sarah.chen@enterprise.ai",
                "tenant_id": "org_nlp_research",
                "roles": ["researcher", "model_evaluator"],
                "permissions": ["models:read", "experiments:write"]
            }
            c_badge1, c_badge2 = st.columns([3, 1])
            c_badge1.markdown(f"👤 **{mock_user['username']}**  \n`{mock_user['email']}`")
            c_badge2.caption(f"Tenant: `{mock_user['tenant_id']}`")
            st.json(mock_user)

# =============================================================================
# 3. LOGIN & LOGOUT FLOWS
# =============================================================================
with tabs[2]:
    st.header("3. State-Managed Login & Logout Workflows")
    st.markdown(r"""
    A robust authentication flow isolates credentials within `st.form` to debounce submissions, 
    verifies passwords against hashes, and purges `st.session_state` upon logout.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

if "is_auth" not in st.session_state:
    st.session_state.is_auth = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.is_auth:
    with st.form("login_form"):
        st.subheader("Platform Sign In")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
        
        if submitted:
            # Simulated verification (Demo credentials: admin / pass123)
            if u == "admin" and p == "pass123":
                st.session_state.is_auth = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    st.success(f"Welcome back, **{st.session_state.username}**!")
    if st.button("Sign Out"):
        st.session_state.is_auth = False
        st.session_state.username = ""
        st.rerun()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if "demo_flow_auth" not in st.session_state:
                st.session_state.demo_flow_auth = False
            if "demo_flow_user" not in st.session_state:
                st.session_state.demo_flow_user = ""

            if not st.session_state.demo_flow_auth:
                with st.form("demo_login_form"):
                    st.caption("Sign In (Demo credentials: `admin` / `pass123`)")
                    u_in = st.text_input("Username", key="flow_u_in")
                    p_in = st.text_input("Password", type="password", key="flow_p_in")
                    sub_btn = st.form_submit_button("Sign In")
                    if sub_btn:
                        if u_in == "admin" and p_in == "pass123":
                            st.session_state.demo_flow_auth = True
                            st.session_state.demo_flow_user = u_in
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
            else:
                st.success(f"Authenticated as **{st.session_state.demo_flow_user}**")
                if st.button("Sign Out", key="flow_signout_btn"):
                    st.session_state.demo_flow_auth = False
                    st.session_state.demo_flow_user = ""
                    st.rerun()

# =============================================================================
# 4. PASSWORD HASHING
# =============================================================================
with tabs[3]:
    st.header("4. Cryptographic Password Security Concepts")
    st.markdown(r"""
    **Never store plain-text passwords or plain MD5/SHA-256 hashes.**
    Modern password storage requires:
    1. **Cryptographic Salt**: A unique 16+ byte random string per user, defeating rainbow table precomputations.
    2. **Adaptive Key Derivation**: High-iteration algorithms (Argon2id, PBKDF2-HMAC-SHA256, bcrypt) that force brute-force attackers to spend significant CPU/GPU compute time per guess.
    3. **Constant-Time Verification**: Using `secrets.compare_digest` to prevent timing attacks.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import hashlib
import secrets

def hash_password(password: str) -> tuple[str, str]:
    """Generates a secure 16-byte salt and PBKDF2 hash."""
    salt = secrets.token_hex(16)
    # 100,000 rounds of PBKDF2 with SHA-256
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", 
        password.encode("utf-8"), 
        salt.encode("utf-8"), 
        iterations=100000
    ).hex()
    return salt, pwd_hash

def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time verification against timing attacks."""
    calculated = hashlib.pbkdf2_hmac(
        "sha256", 
        password.encode("utf-8"), 
        salt.encode("utf-8"), 
        iterations=100000
    ).hex()
    return secrets.compare_digest(calculated, expected_hash)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            test_pwd = st.text_input("Enter Demo Password to Hash", "CorrectHorseBatteryStaple", type="password", key="pwd_hash_in")
            
            # Compute live salt and PBKDF2 hash
            demo_salt = "e4f891b2c40a7192d3f6a8e10b457891"
            computed_hash = hashlib.pbkdf2_hmac(
                "sha256", 
                test_pwd.encode("utf-8"), 
                demo_salt.encode("utf-8"), 
                iterations=50000
            ).hex()

            st.caption("Cryptographic Derivation Output:")
            st.write(f"**Random Salt (16 bytes):** `{demo_salt}`")
            st.write(f"**Derived Hash (PBKDF2 50k rounds):**  \n`{computed_hash[:32]}...{computed_hash[-16:]}`")
            st.info("🔒 Salt prevents rainbow tables; iteration rounds mitigate GPU cracking.")

# =============================================================================
# 5. SESSION EXPIRATION
# =============================================================================
with tabs[4]:
    st.header("5. Session Expiration & Inactivity Timers")
    st.markdown(r"""
    Enterprise security policies mandate terminating inactive sessions after a defined idle threshold (e.g., 15 minutes).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import time

MAX_IDLE_SECONDS = 300  # 5 minutes

def enforce_session_expiration():
    now = time.time()
    last_activity = st.session_state.get("last_active_time", now)
    
    # Check if idle threshold was exceeded
    if now - last_activity > MAX_IDLE_SECONDS:
        st.session_state.is_auth = False
        st.session_state.last_active_time = now
        st.error("Session expired due to inactivity. Please sign in again.")
        st.stop()
    else:
        # Refresh activity timestamp on valid interaction
        st.session_state.last_active_time = now
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            idle_timeout_sec = 60
            curr_sim_idle = st.slider("Simulated Idle Seconds Since Last Action", 0, 90, 25, key="idle_sim_slider")
            
            if curr_sim_idle > idle_timeout_sec:
                st.error(f"🚨 Session Expired! Idle time ({curr_sim_idle}s) exceeded policy limit ({idle_timeout_sec}s). User must re-authenticate.")
            else:
                remaining_sec = idle_timeout_sec - curr_sim_idle
                st.success(f"✅ Session Active. Auto-logout in {remaining_sec} seconds if inactive.")
                st.progress(curr_sim_idle / idle_timeout_sec)

# =============================================================================
# 6. USER ROLES (RBAC)
# =============================================================================
with tabs[5]:
    st.header("6. Role-Based Access Control (RBAC) Hierarchies")
    st.markdown(r"""
    RBAC maps business personas to discrete capabilities following the **Principle of Least Privilege**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Standard Enterprise AI RBAC Role Matrix
RBAC_PERMISSIONS = {
    "Viewer": ["dashboards:read", "leaderboard:read"],
    "Data Scientist": ["dashboards:read", "leaderboard:read", "experiments:run", "notebooks:edit"],
    "Platform Admin": ["dashboards:read", "experiments:run", "deployments:delete", "secrets:manage"]
}

def has_permission(user_role: str, required_scope: str) -> bool:
    scopes = RBAC_PERMISSIONS.get(user_role, [])
    return required_scope in scopes
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Role Capability Matrix

            | Role | View Leaderboard | Run Evaluation | Deploy Model | Manage Secrets |
            | :--- | :---: | :---: | :---: | :---: |
            | **Viewer** | ✅ | ❌ | ❌ | ❌ |
            | **Data Scientist** | ✅ | ✅ | ❌ | ❌ |
            | **MLOps Engineer** | ✅ | ✅ | ✅ | ❌ |
            | **Platform Admin** | ✅ | ✅ | ✅ | ✅ |
            """)

# =============================================================================
# 7. ACCESS CONTROL GATES
# =============================================================================
with tabs[6]:
    st.header("7. Implementing Dynamic Access Control Gates")
    st.markdown(r"""
    Gatekeeping protects sensitive widgets, actions, and entire pages based on verified user roles.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

role = st.session_state.get("role", "Viewer")

# Gate 1: Read-only analytics accessible to all
st.subheader("Model Evaluation Metrics")
st.line_chart([0.72, 0.79, 0.85, 0.89], height=180)

# Gate 2: Mutation controls restricted to Data Scientists and Admins
if role in ["Data Scientist", "Platform Admin"]:
    st.button("🚀 Trigger Model Fine-Tuning Job")
else:
    st.info("🔒 Fine-tuning requires Data Scientist or Admin permissions.")

# Gate 3: Destructive controls restricted strictly to Platform Admin
if role == "Platform Admin":
    st.button("🗑️ Purge Checkpoint Storage", type="primary")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            active_role = st.selectbox("Switch Active Role Simulation", ["Viewer", "Data Scientist", "Platform Admin"], key="gate_role_sel")

            st.caption("Shared Telemetry (All Roles):")
            st.line_chart([0.72, 0.79, 0.85, 0.89], height=150)

            st.caption("Permission-Gated Controls:")
            if active_role in ["Data Scientist", "Platform Admin"]:
                st.button("🚀 Trigger Fine-Tuning Job", key="gate_ft_btn")
            else:
                st.info("🔒 Fine-tuning controls locked for Viewers.")

            if active_role == "Platform Admin":
                st.button("🗑️ Purge Checkpoint Storage (Admin Only)", type="primary", key="gate_purge_btn")

# =============================================================================
# 8. SECURE DESIGN PRINCIPLES
# =============================================================================
with tabs[7]:
    st.header("8. Secure Authentication Threat Modeling")
    st.markdown(r"""
    Threat vectors specific to web applications and Streamlit architectures.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Threat Vectors & Defenses**")
        st.markdown(r"""
        - **Session Fixation**: Always generate fresh session identifiers upon authentication.
        - **Credential Leaks in URLs**: Never pass passwords or session tokens as URL query parameters (`?pwd=...`).
        - **Brute Force Attacks**: Implement exponential login delays or lockout counters in `st.session_state`.
        - **CSRF & Form Tampering**: Streamlit's native WebSocket connection enforces origin checks automatically.
        """)

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Security Engineering Checklist

            | Security Control | Threat Mitigated | Streamlit Implementation |
            | :--- | :--- | :--- |
            | **PBKDF2 / Argon2** | Offline password cracking | Standard `hashlib` with random 16-byte salt. |
            | **`type="password"`** | Shoulder surfing & DOM leak | `st.text_input(..., type="password")`. |
            | **`st.rerun()` upon Logout**| Session persistence / stale DOM | Clear `st.session_state` and invoke `st.rerun()`. |
            | **Constant-Time Compare** | Timing side-channel attacks | `secrets.compare_digest(hash_a, hash_b)`. |
            | **Secrets TOML Isolation**| Accidental Git commits | Store credentials in `.streamlit/secrets.toml`. |
            """)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Authentication Fundamentals Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Passwords in Code** | `if password == "admin123":` | Hash with cryptographic salt via PBKDF2/Argon2; check via `secrets.compare_digest`. |
| **Session Cleanup** | Merely setting `authenticated = False` | Purge entire `st.session_state` dictionary on sign out. |
| **Input Masking** | Plain text fields for API keys/passwords | Always use `st.text_input(type="password")`. |
| **Permission Check** | Hiding widgets client-side with CSS | Enforce gatekeeping server-side in Python (`if role in allowed:`). |
""")
