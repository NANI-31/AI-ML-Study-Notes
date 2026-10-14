import streamlit as st
import re
import hashlib
import secrets
import pandas as pd

st.set_page_config(
    page_title="16.4 Security Best Practices",
    page_icon="🛡️",
    layout="wide"
)

st.title("16.4 Security Best Practices in Streamlit")
st.caption("Application security engineering: OWASP Top 10 mitigation, credential protection, SQL injection prevention, safe file I/O, PII masking, and server-side authorization.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Streamlit Threat Model & Defense-in-Depth

Web applications built on Streamlit require defense-in-depth across the entire software stack.
Because Streamlit runs as a continuous Python process communicating over WebSockets, vulnerabilities in Python code directly impact the host server.

""")

st.graphviz_chart(
    r'''
    digraph SecurityArch {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        client [label="Untrusted Client Input\nUser Prompts, Forms, File Uploads", fillcolor="#FEE2E2", color="#DC2626"];
        defense [label="Streamlit Server Defense Layer\n• Pydantic / Regex Validation\n• Parameterized SQL (:param bindings)\n• Secrets Management (st.secrets)\n• Server-Side Authorization Enforcer", fillcolor="#FEF3C7", color="#D97706"];
        backend [label="Protected Enterprise Backend\nRelational DB / AI Models / Storage Vault", fillcolor="#D1FAE5", color="#059669"];

        client -> defense [label="1. Sanitize & Authenticate"];
        defense -> backend [label="2. Authorized Execution"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Never Plaintext Passwords",
    "2. Credential Exposure",
    "3. Secrets Management",
    "4. Input Validation",
    "5. Prevent SQL Injection",
    "6. Safe File Processing",
    "7. Protect PII Data",
    "8. Server-Side Gates"
])

# =============================================================================
# 1. NEVER PLAINTEXT PASSWORDS
# =============================================================================
with tabs[0]:
    st.header("1. Never Store Plaintext Passwords")
    st.markdown(r"""
    Storing passwords in plaintext or using outdated hash algorithms (MD5, SHA-1, plain SHA-256) is a severe vulnerability.
    Always mandate **Salted PBKDF2-HMAC-SHA256**, **bcrypt**, or **Argon2id** with a minimum of 100,000 iterations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Vulnerable vs Secure Password Handling**")
        st.code(r'''
# ❌ CATASTROPHIC VULNERABILITY (Plaintext / Plain MD5):
# db.execute("INSERT INTO users VALUES (?, ?)", (user, password))
# bad_hash = hashlib.md5(password.encode()).hexdigest()  # Rainbow table cracked instantly!

# ✅ ENTERPRISE DEFENSIVE PATTERN (Salted PBKDF2):
import hashlib
import secrets

def secure_hash_password(raw_password: str) -> tuple[str, str]:
    # 16-byte cryptographically secure random salt
    salt = secrets.token_hex(16)
    derived_hash = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100000
    ).hex()
    return salt, derived_hash
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            test_pw = st.text_input("Enter Sample Password", "UltraSecretKey2026!", type="password", key="sec_pw_input")
            
            # Demonstration of salted output
            sample_salt = secrets.token_hex(16)
            sample_hash = hashlib.pbkdf2_hmac(
                "sha256", test_pw.encode("utf-8"), sample_salt.encode("utf-8"), iterations=100000
            ).hex()
            
            st.caption("Derived Cryptographic Values:")
            st.code(f"Unique Salt: {sample_salt}\nPBKDF2 Hash: {sample_hash[:32]}...{sample_hash[-16:]}")
            st.success("✅ Salt ensures identical passwords produce distinct hashes across users.")

# =============================================================================
# 2. AVOID EXPOSING CREDENTIALS
# =============================================================================
with tabs[1]:
    st.header("2. Avoid Exposing Credentials in UI & Source Code")
    st.markdown(r"""
    API keys, database passwords, and private certificates must never appear in raw text in UI elements, 
    error tracebacks, console logs, or Git commits.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def mask_api_key(key: str) -> str:
    """Masks secret tokens for safe UI display."""
    if len(key) <= 8:
        return "********"
    return f"{key[:4]}...{key[-4:]}"

# ❌ BAD: Displaying raw keys
# st.write(f"Active Key: {raw_api_key}")

# ✅ GOOD: Always mask before rendering
api_key = "sk-proj-91a82f0c8831e9a012bc78"
st.metric("Connected API Key", mask_api_key(api_key))
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            raw_key_demo = "sk-live-0918a2bc44391e88029147"
            masked_key = f"{raw_key_demo[:4]}...{raw_key_demo[-4:]}"
            
            st.metric("Connected OpenAI Key", masked_key)
            st.caption("🔒 Masking prevents shoulder-surfing, screen recording leaks, and accidental screenshots.")

# =============================================================================
# 3. USE SECRETS MANAGEMENT
# =============================================================================
with tabs[2]:
    st.header("3. Native Secrets Management (`.streamlit/secrets.toml`)")
    st.markdown(r"""
    Store sensitive tokens in `.streamlit/secrets.toml` and ensure the file is added to `.gitignore`.
    In cloud environments (Streamlit Community Cloud, AWS, GCP, Kubernetes), inject secrets via environment variables.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import os

# Resilient secret resolution pattern
def get_service_credential(key_name: str) -> str:
    # 1. Check Streamlit native secrets
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    # 2. Check OS environment variable
    return os.environ.get(key_name, "")

# Example .gitignore verification
# Add the following line to your .gitignore:
# .streamlit/secrets.toml
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Secrets Hierarchy Checklist

            1. **Local Development**: `.streamlit/secrets.toml` (Never commit to Git).
            2. **Docker / Containers**: Pass secrets via `docker run -e OPENAI_API_KEY=...` or Docker Secrets.
            3. **Kubernetes**: Inject via `Secret` volume mounts or environment variables.
            4. **Automated Scanning**: Use tools like `git-secrets` or `trufflehog` in CI/CD pipelines to block commits with credential patterns.
            """)

# =============================================================================
# 4. VALIDATE USER INPUT
# =============================================================================
with tabs[3]:
    st.header("4. Validate and Sanitize All User Input")
    st.markdown(r"""
    Treat all incoming user inputs as potentially hostile. 
    Enforce strict boundary checks, regex formatting, and sanitization before processing.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import re

def validate_user_email(email_str: str) -> bool:
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return bool(re.match(email_regex, email_str.strip()))

def sanitize_filepath(user_path: str) -> str:
    # Prevent directory traversal attacks (e.g. "../../etc/passwd")
    clean = re.sub(r"\.\.+[/\\/]", "", user_path)
    return clean

email_input = st.text_input("User Email")
if email_input:
    if validate_user_email(email_input):
        st.success("Valid email format.")
    else:
        st.error("Invalid email address syntax.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            email_val = st.text_input("Test Email Input", "sarah.chen@enterprise.ai", key="sec_email_in")
            email_regex = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
            if re.match(email_regex, email_val.strip()):
                st.success("✅ Valid enterprise email syntax verified.")
            else:
                st.error("🚨 Invalid email address format.")

# =============================================================================
# 5. PREVENT SQL INJECTION
# =============================================================================
with tabs[4]:
    st.header("5. Prevent SQL Injection with Parameterized Queries")
    st.markdown(r"""
    **SQL Injection** occurs when untrusted input is concatenated directly into SQL queries (`f"WHERE name = '{input}'"`).
    Always use **Parameterized Queries** where SQL grammar is separated from runtime data.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Vulnerable vs Parameterized Queries**")
        st.code(r'''
# ❌ VULNERABLE SQL INJECTION (String Interpolation):
# If user inputs: admin' OR '1'='1
# cursor.execute(f"SELECT * FROM users WHERE username = '{user_input}'")
# The query becomes: WHERE username = 'admin' OR '1'='1' (Bypasses login!)

# ✅ IMMUNE PARAMETERIZED QUERY:
import sqlite3

# Query grammar and parameter values are sent through separate channels
cursor.execute(
    "SELECT user_id, email FROM users WHERE username = :username",
    {"username": user_input}
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Parameterized SQL Protection Matrix

            | Database Engine | Vulnerable Syntax | Parameterized Syntax |
            | :--- | :--- | :--- |
            | **SQLite / sqlite3** | `f"SELECT * FROM tbl WHERE id = '{x}'"` | `cursor.execute("... WHERE id = ?", (x,))` |
            | **SQLAlchemy 2.0** | `text(f"... WHERE id = '{x}'")` | `text("... WHERE id = :id"), {"id": x}` |
            | **PostgreSQL (psycopg)**| `f"... WHERE id = '{x}'"` | `cursor.execute("... WHERE id = %s", (x,))` |
            | **Pandas read_sql** | `pd.read_sql(f"... WHERE id = '{x}'", ...)` | `pd.read_sql("... WHERE id = :id", params={"id": x})` |
            """)

# =============================================================================
# 6. AVOID UNSAFE FILE PROCESSING
# =============================================================================
with tabs[5]:
    st.header("6. Avoid Unsafe File Processing & Deserialization")
    st.markdown(r"""
    Allowing arbitrary file uploads creates multiple attack vectors:
    - **Zip Slip**: Extracting archives containing relative paths (`../../etc/shadow`).
    - **Pickle Deserialization**: `pickle.load()` executes arbitrary Python code embedded in model weights. Use **SafeTensors** instead.
    - **Extension Spoofing**: Validate file headers, not just file extensions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# ❌ DANGEROUS: Unrestricted pickle deserialization
# import pickle
# model = pickle.load(uploaded_file)  # Can execute arbitrary remote code!

# ✅ SECURE: Restrict MIME types and use safe serializers
uploaded = st.file_uploader(
    "Upload Safe Dataset or Model", 
    type=["csv", "safetensors", "json"]
)

if uploaded:
    # Validate file size cap (e.g. 50 MB)
    MAX_BYTES = 50 * 1024 * 1024
    if uploaded.size > MAX_BYTES:
        st.error("File exceeds 50MB maximum allowable payload size.")
    else:
        st.success(f"File '{uploaded.name}' validated ({uploaded.size / 1024:.1f} KB).")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### File Safety Rules

            1. **No Untrusted Pickle**: Never deserialize `.pkl`, `.joblib`, or `.pt` files from unverified external users. Use Hugging Face `safetensors` or ONNX.
            2. **MIME Type Whitelist**: Pass `type=["csv", "parquet", "json"]` to `st.file_uploader`.
            3. **Memory Limits**: Always check `uploaded_file.size` before reading entire streams into server memory.
            """)

# =============================================================================
# 7. PROTECT PERSONAL INFORMATION
# =============================================================================
with tabs[6]:
    st.header("7. Protect Personally Identifiable Information (PII)")
    st.markdown(r"""
    Regulatory compliance (GDPR, HIPAA, CCPA) mandates redacting sensitive personal attributes 
    (SSNs, credit cards, emails, phone numbers) before displaying them in UI tables or saving them to logs.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import re

def redact_pii_fields(text_content: str) -> str:
    # Redact US Social Security Numbers (SSN: XXX-XX-XXXX)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    text_content = re.sub(ssn_pattern, "[REDACTED-SSN]", text_content)
    
    # Redact Credit Card Numbers (16 digits)
    card_pattern = r"\b(?:\d{4}[ -]?){3}\d{4}\b"
    text_content = re.sub(card_pattern, "[REDACTED-CARD]", text_content)
    
    return text_content
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sample_raw = "User Customer Record: SSN 123-45-6789, Card 4111-2222-3333-4444."
            
            ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
            card_pattern = r"\b(?:\d{4}[ -]?){3}\d{4}\b"
            clean_text = re.sub(ssn_pattern, "[REDACTED-SSN]", sample_raw)
            clean_text = re.sub(card_pattern, "[REDACTED-CARD]", clean_text)

            st.caption("Raw Source String:")
            st.code(sample_raw)
            st.caption("PII Redacted Sanitized Output:")
            st.code(clean_text)
            st.success("✅ Sensitive personal identifiers stripped prior to presentation or logging.")

# =============================================================================
# 8. SERVER-SIDE GATES
# =============================================================================
with tabs[7]:
    st.header("8. Avoid Treating Hidden UI Elements as Access Control")
    st.markdown(r"""
    **The Golden Rule of Web Security**: Client-side visibility is not authorization.
    Hiding a widget with CSS (`display: none`) or `disabled=True` does not prevent a motivated actor 
    from triggering actions. **Always verify permissions in Python before executing operations.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Vulnerable vs Enforced Gating**")
        st.code(r'''
# ❌ INSECURE: Cosmetic UI hiding only
if not is_admin:
    st.html("<style> .admin-btn { display: none; } </style>")

# An attacker can un-hide the button in DevTools and click it!
if st.button("Delete Database", key="admin-btn"):
    delete_database()  # Executes because Python didn't check!

# ✅ SECURE: Server-side validation in Python
if st.button("Delete Database"):
    # Re-verify identity & role in the backend
    if st.session_state.get("user_role") == "Platform Admin":
        delete_database()
        st.success("Database purged.")
    else:
        st.error("Security Alert: Blocked unauthorized action attempt.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sim_role = st.selectbox("Current Simulated Role", ["Viewer", "Platform Admin"], key="sec_gate_role")
            
            if st.button("Purge All Model Artifacts", key="sec_purge_demo_btn"):
                if sim_role == "Platform Admin":
                    st.success("Operation authorized: Artifacts purged by Platform Admin.")
                else:
                    st.error("🚨 403 Forbidden: Server-side check rejected mutation. Action blocked.")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Streamlit Security Best Practices Cheatsheet

| Threat Vector | Amateur Vulnerability | Production Enterprise Defense |
| :--- | :--- | :--- |
| **Passwords** | Storing in plaintext or MD5 | Salted PBKDF2-HMAC-SHA256 (100k+ iterations) or Argon2id. |
| **API Keys** | Hardcoding `api_key = "sk-..."` | Store in `.streamlit/secrets.toml`; add to `.gitignore`. |
| **SQL Injection** | String formatting `f"SELECT ... WHERE id = '{x}'"` | Parameterized binding (`:id` or `?`). |
| **Model Deserialization**| `pickle.load(uploaded_file)` | Use SafeTensors or ONNX runtimes. |
| **PII Data** | Storing raw SSNs/Cards in DataFrames | Regex scrubbing and tokenization before display. |
| **Authorization** | Hiding UI buttons with CSS | Python server-side gatekeeping (`if role == "Admin":`). |
""")
