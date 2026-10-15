import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="16.2 Streamlit Authentication",
    page_icon="🛡️",
    layout="wide"
)

st.title("16.2 Native Streamlit Authentication (OIDC & SSO)")
st.caption("Enterprise Single Sign-On (SSO): Streamlit's native OpenID Connect (OIDC) engine, st.login(), st.logout(), st.user claims, page protection guards, and provider configs.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Native OpenID Connect (OIDC) Architecture in Streamlit

In Streamlit 1.39+, native OpenID Connect (OIDC) authentication allows connecting directly to enterprise Identity Providers (Google, Microsoft Entra ID, Okta, Auth0, Keycloak) 
without relying on third-party wrapper packages.

""")

st.graphviz_chart(
    r'''
    digraph AuthWorkflow {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        visitor [label="Unauthenticated Visitor\nArrives at Streamlit URL", fillcolor="#F1F5F9", color="#64748B"];
        gatekeeper [label="Authentication Gatekeeper\nif not st.session_state.get('authenticated'):\n    render_login_dialog(); st.stop()", fillcolor="#FEF3C7", color="#D97706"];
        verify [label="Credential Verification\nbcrypt.checkpw(entered_pwd, stored_hash)", fillcolor="#DBEAFE", color="#1D4ED8"];
        protected [label="Authenticated Enterprise App\nFull access to protected models & data", fillcolor="#D1FAE5", color="#059669"];

        visitor -> gatekeeper;
        gatekeeper -> verify [label="Form submitted"];
        verify -> protected [label="Valid password"];
        verify -> gatekeeper [label="Invalid password (retry)"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Native OIDC Overview",
    "2. Provider Configuration",
    "3. Login & Logout Flows",
    "4. Secrets.toml Auth Spec",
    "5. Using st.user Claims",
    "6. Protecting Pages",
    "7. Unauthenticated Handling",
    "8. Auth Error Management"
])

# =============================================================================
# 1. NATIVE OIDC OVERVIEW
# =============================================================================
with tabs[0]:
    st.header("1. Streamlit's Built-in OIDC Authentication Engine")
    st.markdown(r"""
    Native authentication introduces three core primitives:
    - `st.login(provider=None)`: Redirects the user's browser to the configured identity provider's consent screen.
    - `st.logout()`: Clears the authenticated session cookie and resets the user identity state.
    - `st.user`: Global session object providing identity attributes (`is_logged_in`, `email`, `name`, `sub`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Check native authentication state
if not st.user.is_logged_in:
    st.info("You are currently browsing as a guest.")
    if st.button("Log in with Enterprise SSO"):
        st.login()  # Dispatches OIDC authorization handshake
else:
    st.success(f"Authenticated as **{st.user.name}** ({st.user.email})")
    if st.button("Log out"):
        st.logout()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Native Auth Primitives

            | Primitive | Type | Description |
            | :--- | :--- | :--- |
            | `st.login(provider=...)` | Function | Dispatches user to IdP SSO login screen. |
            | `st.logout()` | Function | Terminates session cookie and signs out user. |
            | `st.user.is_logged_in` | Boolean | True if user holds a verified JWT session. |
            | `st.user.email` | String | Verified email from the IdP ID token. |
            | `st.user.name` | String | Display name from the IdP ID token. |
            | `st.user` (Mapping) | Dict-like | Read-only claims dictionary containing custom JWT fields. |
            """)

# =============================================================================
# 2. PROVIDER CONFIGURATION
# =============================================================================
with tabs[1]:
    st.header("2. Identity Provider Configuration (Google, Entra ID, Okta)")
    st.markdown(r"""
    Streamlit supports any standards-compliant OpenID Connect provider. 
    The provider is configured by specifying its `server_metadata_url` (`/.well-known/openid-configuration`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Provider Metadata URLs**")
        st.code(r'''
# Standard Enterprise OIDC Discovery Endpoints:

# 1. Google Workspace:
# https://accounts.google.com/.well-known/openid-configuration

# 2. Microsoft Entra ID (Azure AD):
# https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration

# 3. Okta:
# https://{YOUR_OKTA_DOMAIN}.okta.com/.well-known/openid-configuration

# 4. Auth0:
# https://{YOUR_DOMAIN}.auth0.com/.well-known/openid-configuration

# 5. Keycloak (Self-Hosted):
# https://{KEYCLOAK_HOST}/realms/{REALM}/.well-known/openid-configuration
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### IdP Setup Requirements

            1. **Authorized Redirect URI**: In your IdP console (Google Cloud Console / Azure Portal), register the callback:
               - Localhost: `http://localhost:8501/oauth2callback`
               - Production: `https://app.enterprise.ai/oauth2callback`
            2. **OAuth2 Scopes**: Ensure `openid`, `email`, and `profile` scopes are granted.
            3. **Client Secret Protection**: Never check client secrets into Git.
            """)

# =============================================================================
# 3. LOGIN & LOGOUT FLOWS
# =============================================================================
with tabs[2]:
    st.header("3. Complete Login & Logout Flow Lifecycle")
    st.markdown(r"""
    Managing the authentication lifecycle within Streamlit's reactive rerun model.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def render_auth_header():
    # Simulated auth state container
    is_authenticated = st.session_state.get("sim_auth", False)
    
    col_logo, col_action = st.columns([4, 1])
    with col_logo:
        st.markdown("### 🏢 Enterprise AI Portal")
    with col_action:
        if not is_authenticated:
            if st.button("🔐 Sign In", type="primary"):
                st.session_state.sim_auth = True
                st.session_state.sim_user = {"name": "Alex Mercer", "email": "alex.m@enterprise.ai"}
                st.rerun()
        else:
            if st.button("🚪 Sign Out"):
                st.session_state.sim_auth = False
                st.session_state.sim_user = None
                st.rerun()

render_auth_header()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if "sim_auth" not in st.session_state:
                st.session_state.sim_auth = False
            if "sim_user" not in st.session_state:
                st.session_state.sim_user = None

            c_head_l, c_head_r = st.columns([3, 1])
            with c_head_l:
                if st.session_state.sim_auth:
                    st.write(f"Logged in as **{st.session_state.sim_user['name']}**")
                    st.caption(f"`{st.session_state.sim_user['email']}`")
                else:
                    st.write("Guest Session (Unauthenticated)")
            with c_head_r:
                if not st.session_state.sim_auth:
                    if st.button("🔐 Sign In", type="primary", key="oidc_login_demo_btn"):
                        st.session_state.sim_auth = True
                        st.session_state.sim_user = {"name": "Alex Mercer", "email": "alex.m@enterprise.ai"}
                        st.rerun()
                else:
                    if st.button("🚪 Sign Out", key="oidc_logout_demo_btn"):
                        st.session_state.sim_auth = False
                        st.session_state.sim_user = None
                        st.rerun()

# =============================================================================
# 4. SECRETS.TOML AUTH SPEC
# =============================================================================
with tabs[3]:
    st.header("4. Authentication Configuration (`.streamlit/secrets.toml`)")
    st.markdown(r"""
    Streamlit reads authentication configuration from the `[auth]` section in `.streamlit/secrets.toml`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# .streamlit/secrets.toml (Keep out of Git!)
[auth]
# OAuth2 Callback URI (Must match IdP registration)
redirect_uri = "http://localhost:8501/oauth2callback"

# Cryptographically strong 32-byte secret for session cookie encryption
cookie_secret = "7d4b9f1a8e2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e"

# Client credentials issued by your Identity Provider
client_id = "49102840192-example-client-id.apps.googleusercontent.com"
client_secret = "GOCSPX-your-secret-token-key"

# OIDC Discovery endpoint
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### `[auth]` Configuration Key Reference

            | Parameter | Required? | Purpose |
            | :--- | :---: | :--- |
            | `redirect_uri` | **Yes** | Authorized redirect endpoint (`/oauth2callback`). |
            | `cookie_secret` | **Yes** | 32+ byte string used to encrypt session JWTs in the browser. |
            | `client_id` | **Yes** | Public application ID registered with the IdP. |
            | `client_secret` | **Yes** | Private secret shared with the IdP. |
            | `server_metadata_url` | **Yes** | Discovery URL resolving token endpoints & public JWKS keys. |
            """)

# =============================================================================
# 5. USING ST.USER CLAIMS
# =============================================================================
with tabs[4]:
    st.header("5. Extracting and Utilizing `st.user` Claims")
    st.markdown(r"""
    The `st.user` object is a dictionary-like container representing verified OIDC ID token claims.
    Custom claims (e.g., department, security clearance, groups) can be unpacked for business logic.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Inspecting verified claims
if st.user.is_logged_in:
    email = st.user.email
    full_name = st.user.name
    sub_id = st.user.sub
    
    # Access custom enterprise claims
    user_groups = st.user.get("groups", ["default_users"])
    tenant_org = st.user.get("hd", "enterprise.ai")  # Hosted Domain
    
    st.write(f"Welcome, {full_name}!")
    st.json(dict(st.user))
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            simulated_claims = {
                "sub": "google-oauth2|109284019284",
                "name": "Sarah Connor",
                "given_name": "Sarah",
                "family_name": "Connor",
                "email": "sarah.connor@cyberdyne.ai",
                "email_verified": True,
                "hd": "cyberdyne.ai",
                "roles": ["ai_researcher", "model_reviewer"]
            }
            st.caption("Decoded OIDC ID Token Claims:")
            st.json(simulated_claims)
            st.success("Claims are cryptographically verified against the provider's public JWKS keys.")

# =============================================================================
# 6. PROTECTING SENSITIVE PAGES
# =============================================================================
with tabs[5]:
    st.header("6. Protecting Sensitive Pages and Views")
    st.markdown(r"""
    In multi-page architectures, sensitive pages must execute an authentication guard 
    before rendering any proprietary models, data, or operational controls.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def require_authentication():
    """Gatekeeper function halting execution for unauthenticated users."""
    if not st.user.is_logged_in:
        st.warning("🔒 Access Restricted: Please sign in to view this page.")
        if st.button("Log In with SSO", type="primary"):
            st.login()
        st.stop()  # Prevents execution of remaining page code

# At the top of protected pages (e.g., pages/admin_console.py):
require_authentication()

# Code below only executes for verified users:
st.header("Confidential Financial Modeling")
st.dataframe(sensitive_data)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            guard_sim = st.toggle("Simulate Gated Page Access", value=False, key="guard_demo_toggle")

            if not guard_sim:
                st.warning("🔒 Access Restricted: You must authenticate before accessing the Model Deployment Engine.")
                if st.button("Log In with SSO", type="primary", key="guard_demo_login"):
                    st.info("Triggering `st.login()` redirect...")
            else:
                st.success("✅ Access Granted: Verified identity matches permissions.")
                st.caption("Rendering protected model registry table:")
                st.dataframe(pd.DataFrame({
                    "Model": ["Mistral-Large-v2", "Llama-3-70B-Instruct"],
                    "Status": ["Production Active", "Staging"],
                    "VRAM": ["48 GB", "140 GB"]
                }), width="stretch", hide_index=True)

# =============================================================================
# 7. UNAUTHENTICATED USER HANDLING
# =============================================================================
with tabs[6]:
    st.header("7. Handling Unauthenticated Users (Guest Mode vs Paywall)")
    st.markdown(r"""
    Depending on application requirements, unauthenticated users can experience:
    - **Hard Gate / Paywall**: Immediate halt (`st.stop()`) until authentication completes.
    - **Freemium / Guest Preview**: Public read-only teaser with interactive features locked behind a login prompt.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

is_logged_in = st.user.is_logged_in

st.header("Public AI Benchmark Leaderboard")
st.dataframe(public_benchmarks)

if not is_logged_in:
    with st.container(border=True):
        st.info("💡 Want to evaluate your own model checkpoints?")
        st.markdown("Sign in with your enterprise credentials to unlock private benchmarking.")
        if st.button("Sign In to Submit Models"):
            st.login()
else:
    with st.form("submission_form"):
        st.subheader("Submit Custom Model Checkpoint")
        st.text_input("Hugging Face Repo ID")
        st.form_submit_button("Start Evaluation")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            guest_preview_mode = st.toggle("User Signed In?", value=False, key="guest_preview_tog")

            st.caption("Public Leaderboard (Always Visible):")
            st.dataframe(pd.DataFrame({
                "Rank": [1, 2],
                "Model": ["Claude-3.5-Sonnet", "GPT-4o"],
                "MMLU": [0.887, 0.885]
            }), width="stretch", hide_index=True)

            if not guest_preview_mode:
                with st.container(border=True):
                    st.info("💡 Want to submit custom evaluation checkpoints?")
                    st.caption("Sign in with enterprise SSO to submit models.")
                    st.button("Sign In to Submit Models", key="guest_cta_btn")
            else:
                st.success("✅ Submission Form Unlocked (Logged in as Engineer)")

# =============================================================================
# 8. AUTH ERROR MANAGEMENT
# =============================================================================
with tabs[7]:
    st.header("8. Managing OAuth2 & OIDC Authentication Errors")
    st.markdown(r"""
    Common OIDC error conditions and how to recover gracefully:
    - `access_denied`: User cancelled the consent screen.
    - `invalid_grant`: Expired authorization code or clock skew between servers.
    - `redirect_uri_mismatch`: Mismatch between `.streamlit/secrets.toml` and IdP console.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Error Handling Best Practices**")
        st.markdown(r"""
        ```python
        # Checking for OAuth error parameters in query string
        query_params = st.query_params

        if "error" in query_params:
            err = query_params["error"]
            if err == "access_denied":
                st.warning("Authentication was cancelled.")
            elif err == "invalid_request":
                st.error("Configuration error. Contact admin.")
            
            # Clear error param and allow user to retry
            if st.button("Dismiss and Retry"):
                st.query_params.clear()
                st.rerun()
        ```
        """)

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Common OIDC Errors & Remediation

            | Error Code | Root Cause | Remediation |
            | :--- | :--- | :--- |
            | `redirect_uri_mismatch` | Callback URL differs from IdP settings | Verify exact protocol (`http` vs `https`) and port in `secrets.toml`. |
            | `access_denied` | User clicked 'Cancel' on Google/Microsoft prompt | Prompt user with a retry button. |
            | `clock_skew` | Local server time drifted $>5$ minutes | Synchronize system clock with NTP. |
            | `invalid_client` | Bad `client_id` or `client_secret` | Regenerate client secret in IdP developer console. |
            """)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Native Authentication Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Custom Auth Packages** | Installing unmaintained community auth libraries | Use native `st.login()`, `st.logout()`, and `st.user` (Streamlit 1.39+). |
| **Secrets Exposure** | Committing `client_secret` to Git | Store in `.streamlit/secrets.toml` and add to `.gitignore`. |
| **Page Gatekeeping** | Merely hiding widgets with CSS | Call `require_authentication()` with `st.stop()` to halt Python execution. |
| **Cookie Encryption** | Using weak or default `cookie_secret` | Generate a 32-byte hex secret via `secrets.token_hex(32)`. |
""")
