import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="20.1 Security Basics - Vector DB Curriculum",
    page_icon="🔒",
    layout="wide"
)

# Title & Overview
st.title("20.1 Vector Database Security Basics: Authentication, Authorization, API Keys & RBAC")
st.caption("Chapter 20: Vector Database Security | Domain: 4. Vector Database")
st.markdown("""
In the rapid rush to deploy enterprise AI applications, the vector database is often mistakenly treated as an 
unrestricted internal cache. Developers frequently expose vector database ports directly or configure permissive 
master credentials across all services.

However, a compromised vector database is a **catastrophic enterprise data leak vector**:
* Vector databases store raw, unmasked proprietary knowledge: internal source code, customer support tickets, financial statements, and HR records.
* An attacker who gains unauthorized read access can query the database using semantic search, exfiltrating the company's most sensitive secrets in seconds.

To safeguard corporate IP, production vector databases must enforce **Zero-Trust Security**:
1. **Strong Authentication**: Verifying caller identity via Mutual TLS (mTLS), scoped API keys, or JWT tokens.
2. **Granular Authorization**: Enforcing Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC).
3. **API Key Lifecycle Management**: Automated key rotation, least-privilege scoping, and vault injection.
4. **Administrative vs. Search Segregation**: Restricting mutation APIs (create, delete, upsert) to secure backend ETL pipelines.

In this lesson, we explore the vector security boundary, configure authenticated client connections, 
and test an interactive Role-Based Access Control (RBAC) Simulator.
""")

st.divider()

# Section 1: The Zero-Trust Vector Security Boundary
st.header("1. The Zero-Trust Vector Security Boundary", divider="rainbow")

col_sec1, col_sec2 = col_sec1, col_sec2 = st.columns([1.1, 0.9])

with col_sec1:
    st.markdown("""
    ### Four Core Pillars of Vector Access Security
    
    * **1. Authentication (Who are you?)**:
      * Verifies that the client connecting to the vector database is a legitimate service.
      * *Protocols*: Mutual TLS (mTLS) with SPIFFE X.509 certs, or Bearer Authorization headers via OAuth2 / JWT.
    
    * **2. Authorization (What are you allowed to do?)**:
      * Determines permissions once authenticated:
        * *Cluster Admin*: Create/drop collections, manage cluster nodes.
        * *Ingestion Worker*: Upsert and delete documents in assigned collections.
        * *Search Consumer*: Read-only search execution.
    
    * **3. Scoped API Keys**:
      * Never use global admin master tokens across web clients.
      * Issue time-limited, collection-scoped keys with read-only privileges.
    
    * **4. Network Perimeter Isolation**:
      * Vector database ports (e.g. Chroma 8000, Qdrant 6333, Milvus 19530) must **never be exposed to the public Internet**.
      * Restrict access strictly to private VPC subnets and Kubernetes service meshes.
    """)

with col_sec2:
    st.markdown("### The Zero-Trust Boundary Topology")
    sec_dot = """
    digraph ZeroTrustBoundary {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_external {
            label="Untrusted Zone"; style="dashed"; color="#DC2626";
            PublicWeb [label="Public Web Client / Mobile App\\n(Zero direct access to DB)", fillcolor="#FEE2E2"];
        }

        APIGateway [label="API Gateway & Auth Proxy\\nValidates JWT & Extracts User Identity", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        subgraph cluster_private {
            label="Private VPC / Service Mesh (mTLS)"; style="dashed"; color="#059669";
            AppService [label="RAG Query Service\\n(Scoped Read-Only Key)", fillcolor="#D1FAE5"];
            ETLWorker [label="Ingestion ETL Worker\\n(Scoped Write Key)", fillcolor="#D1FAE5"];
            VectorStore [label="ChromaDB Cluster (Private Port)\\nEnforces RBAC & TLS Termination", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        PublicWeb -> APIGateway [label="HTTPS / WAF"];
        APIGateway -> AppService [label="Identity Propagated"];
        AppService -> VectorStore [label="mTLS + Scoped Read Key"];
        ETLWorker -> VectorStore [label="mTLS + Write Key"];
    }
    """
    st.graphviz_chart(sec_dot, use_container_width=False)

st.divider()

# Section 2: Production Authentication Configurations
st.header("2. Production Authentication Configurations", divider="rainbow")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### 1. ChromaDB Server Authentication (Docker)")
    st.code("""# docker-compose.yml: Running ChromaDB with Token Auth
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "127.0.0.1:8000:8000"  # Bind strictly to localhost / private VPC
    environment:
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS_FILE=/chroma/server_tokens.json
      - CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER=X-Chroma-Token
    volumes:
      - ./server_tokens.json:/chroma/server_tokens.json:ro
      - ./chroma_data:/chroma/chroma
""", language="yaml")

with col_code2:
    st.markdown("### 2. Secure Authenticated Client in Python")
    st.code("""import chromadb
from chromadb.config import Settings

# Connect to secured remote Chroma server using Token Auth
client = chromadb.HttpClient(
    host="internal-chroma.vpc.local",
    port=8000,
    ssl=True,  # Enforce TLS encryption in transit
    headers={
        "X-Chroma-Token": "sk-prod-search-read-only-908124"
    },
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials="sk-prod-search-read-only-908124"
    )
)

# Access allowed collection
collection = client.get_collection("public_knowledge_base")
""", language="python")

st.divider()

# Section 3: Role-Based Access Control (RBAC) Matrix
st.header("3. Role-Based Access Control (RBAC) Matrix", divider="rainbow")

st.markdown("""
Enforce least privilege by granting microservices strictly the permissions necessary for their operational role:
""")

RBAC_MATRIX = [
    {
        "Role Name": "Platform Admin (SRE / DevOps)",
        "Authentication Mechanism": "Hardware MFA + Short-Lived Vault Token",
        "Allowed Operations": "Create Collection, Drop Collection, Modify Index HNSW, View Health Metrics, Backup/Restore.",
        "Disallowed Operations": "Unrestricted raw data exports without audit logging."
    },
    {
        "Role Name": "Ingestion Pipeline (Airflow / Temporal)",
        "Authentication Mechanism": "Scoped Service Account API Key (mTLS)",
        "Allowed Operations": "Add Documents, Update Embeddings, Delete by ID in assigned collection.",
        "Disallowed Operations": "Drop Collection, Query across other tenant collections."
    },
    {
        "Role Name": "RAG Query Service (Chatbot Backend)",
        "Authentication Mechanism": "Scoped Read-Only API Key",
        "Allowed Operations": "Vector Query (similarity search), Get Documents by ID.",
        "Disallowed Operations": "Add, Update, Delete, or Modify Collection Settings."
    },
    {
        "Role Name": "Public Web / Mobile User",
        "Authentication Mechanism": "User JWT via OAuth2 (Okta / Auth0)",
        "Allowed Operations": "ZERO DIRECT ACCESS. Must query via backend API Gateway.",
        "Disallowed Operations": "Direct TCP/HTTP access to vector database port is completely blocked."
    }
]

st.dataframe(pd.DataFrame(RBAC_MATRIX), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Role-Based Access Control (RBAC) Simulator
st.header("4. Interactive Workbench: RBAC & Permission Enforcement Simulator", divider="rainbow")

st.markdown("""
Simulate an API request against a secured ChromaDB cluster. Select a client role, an API operation, 
and observe how authentication and authorization policies permit or reject the request.
""")

col_rbac_role, col_rbac_op = col_rbac_role, col_rbac_op = st.columns(2)

with col_rbac_role:
    st.subheader("1. Client Identity & Credentials")
    selected_role = st.selectbox(
        "Active Client Persona:",
        [
            "RAG Query Service (Role: Read-Only Consumer)",
            "Data Pipeline Worker (Role: Ingestion ETL)",
            "Anonymous Web Client (Role: Unauthenticated)",
            "DevOps Admin (Role: Superuser)"
        ]
    )

with col_rbac_op:
    st.subheader("2. Target Vector DB API Operation")
    selected_op = st.selectbox(
        "Requested Operation:",
        [
            "Execute Vector Similarity Search (collection.query)",
            "Batch Insert 500 Documents (collection.add)",
            "Delete Document by ID (collection.delete)",
            "Drop Entire Collection (client.delete_collection)"
        ]
    )

# Evaluation Logic
is_authenticated = "Unauthenticated" not in selected_role

if not is_authenticated:
    is_allowed = False
    verdict_reason = "HTTP 401 Unauthorized: Missing or invalid API Key."
    http_code = 401
else:
    if "Superuser" in selected_role:
        is_allowed = True
        verdict_reason = "HTTP 200 OK: Superuser administrative permissions verified."
        http_code = 200
    elif "Read-Only" in selected_role:
        if "query" in selected_op:
            is_allowed = True
            verdict_reason = "HTTP 200 OK: Read-only search permitted."
            http_code = 200
        else:
            is_allowed = False
            verdict_reason = "HTTP 403 Forbidden: Read-only role cannot execute mutating operations."
            http_code = 403
    elif "Ingestion ETL" in selected_role:
        if "query" in selected_op or "add" in selected_op or "delete Document" in selected_op:
            is_allowed = True
            verdict_reason = "HTTP 200 OK: Data ingestion and cleanup permitted."
            http_code = 200
        else:
            is_allowed = False
            verdict_reason = "HTTP 403 Forbidden: Ingestion role lacks cluster-level administrative permissions (cannot drop collections)."
            http_code = 403

st.divider()

col_perm_status, col_perm_audit = col_perm_status, col_perm_audit = st.columns([1, 1])

with col_perm_status:
    st.subheader("Authorization Decision")
    
    if is_allowed:
        st.success(f"🟢 **REQUEST PERMITTED** ({verdict_reason})")
        st.metric(label="HTTP Response Status", value=f"{http_code} OK")
    else:
        st.error(f"🔴 **REQUEST BLOCKED** ({verdict_reason})")
        st.metric(label="HTTP Response Status", value=f"{http_code} Denied")

with col_perm_audit:
    st.subheader("Security Audit Telemetry")
    st.markdown(f"""
    * **Caller Identity**: `{selected_role.split('(')[0].strip()}`
    * **Action**: `{selected_op.split('(')[0].strip()}`
    * **TLS Certificate**: `Enforced (TLS 1.3 / mTLS SPIFFE)`
    * **Audit Log Entry**: `[AUDIT_LOG] Timestamp=2024-10-13T05:30Z Principal={selected_role.split(':')[1].replace(')', '').strip()} Action={selected_op.split('(')[1].replace(')', '')} Status={http_code}`
    """)
    st.info("💡 Production Best Practice: Stream all 403 Forbidden events directly into Datadog or AWS CloudWatch to detect automated credential probing.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why must vector database ports NEVER be exposed directly to client-side applications (React, iOS)?"):
    st.markdown("""
    **Answer**:
    Exposing a vector database port (e.g. Chroma 8000) directly to frontend web or mobile apps is an **architectural catastrophe**:
    1. **Credential Exposure**: Any API key shipped inside client-side JavaScript or mobile binaries can be decompiled and stolen in seconds.
    2. **Unrestricted Search Access**: An attacker can bypass UI constraints and query the entire collection, using semantic search to exfiltrate proprietary manuals, HR data, or customer PII.
    3. **Denial of Service (DoS)**: High-dimensional vector searches are CPU-intensive. Attackers can spam complex queries with `efSearch=500`, exhausting cluster CPU and crashing the database.
    * **Requirement**: All client traffic must pass through a **Backend API Gateway** that authenticates users, enforces rate limits, and injects mandatory tenant filters.
    """)

with st.expander("Q2: What is Mutual TLS (mTLS) and why is it standard for vector database communication?"):
    st.markdown("""
    **Answer**:
    In standard TLS, only the server proves its identity with a certificate. 
    In **Mutual TLS (mTLS)**:
    * The vector database validates the client's cryptographic X.509 certificate, AND the client validates the database's certificate.
    * Communication is fully encrypted in transit.
    * Even if an attacker enters the internal network or discovers an API key, they cannot connect without possessing a valid, non-revoked client certificate issued by the internal Certificate Authority (e.g. HashiCorp Vault or SPIRE).
    """)

with st.expander("Q3: What is the difference between Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC)?"):
    st.markdown("""
    **Answer**:
    * **RBAC**: Grants permissions based on static role definitions (e.g., `role: Searcher` can search, `role: Admin` can write). RBAC is typically coarse-grained (applied at the collection or database level).
    * **ABAC**: Evaluates dynamic metadata attributes attached to both the user and the document at query time (e.g. `User.clearance_level >= Document.security_classification AND User.department == Document.department`). ABAC enables fine-grained document-level security.
    """)

with st.expander("Q4: How should API keys be rotated in production without incurring application downtime?"):
    st.markdown("""
    **Answer**:
    Use the **Dual-Key Overlap Rotation Pattern**:
    1. The vector database is configured to accept two active keys simultaneously: `Key_Active` and `Key_Next`.
    2. Deploy `Key_Next` to the secret manager (e.g. AWS Secrets Manager / Vault).
    3. Perform a rolling restart of application consumer pods so they adopt `Key_Next`.
    4. Once all application traffic is using `Key_Next`, revoke and delete `Key_Active` from the database configuration.
    5. This guarantees zero failed requests during rotation.
    """)

with st.expander("Q5: What is the principle of Least Privilege applied to vector data ingestion?"):
    st.markdown("""
    **Answer**:
    Ingestion workers (e.g. Airflow or cron jobs that crawl documents and update embeddings) should **never be granted cluster-admin tokens**:
    * If an ingestion worker token is compromised, an attacker should only be able to upsert data to a specific, scoped collection.
    * The ingestion token should NOT have permissions to drop collections, read other tenants' data, or modify cluster HNSW configuration parameters.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Hardcoding Master Admin Tokens in Code Repositories**  
    Hardcoding Chroma admin tokens in Python scripts causes immediate credential leakage when pushed to GitHub. Inject tokens strictly via environment variables or secret vaults.
    """)
    st.warning("""
    **Trap 2: Disabling TLS for Internal VPC Traffic**  
    Assuming internal VPC traffic is safe and running plain HTTP exposes vector queries and retrieved raw documents to packet sniffing. Enforce TLS 1.3 everywhere.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using a Single Shared API Key Across All Services**  
    Using one key for ingestion, search, and admin prevents auditing and makes key rotation impossible without coordinating multiple teams. Issue scoped keys per service.
    """)
    st.warning("""
    **Trap 4: Missing Request Rate Limiting**  
    Omitting query rate limits allows rogue internal scripts to spam expensive vector searches, starving production traffic of CPU cores. Rate limit by client identity.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Why is a Vector DB a major leak vector?"):
        st.markdown("""
        **Unmasked IP**:  
        Stores raw proprietary text searchable via semantic intent, allowing rapid automated data exfiltration.
        """)
    with st.expander("🗂️ Flashcard: What is mTLS?"):
        st.markdown("""
        **Mutual TLS**:  
        Both client and server authenticate each other using cryptographic certificates; fully encrypted in transit.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is RBAC in vector DBs?"):
        st.markdown("""
        **Role-Based Access**:  
        Restricts mutating operations (create, drop, add) to admins/ETL while granting read-only query access to apps.
        """)
    with st.expander("🗂️ Flashcard: How to rotate keys safely?"):
        st.markdown("""
        **Dual-Key Overlap**:  
        Allow both old and new keys during rolling deployment before revoking the retired key.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why vector databases require Zero-Trust security and private VPC perimeter isolation.", value=True)
chk2 = st.checkbox("I know how to configure Token Authentication and TLS encryption in ChromaDB.", value=True)
chk3 = st.checkbox("I understand the Role-Based Access Control (RBAC) matrix for Admin, Ingestion, and Search roles.", value=True)
chk4 = st.checkbox("I know how to implement safe dual-key API rotation without application downtime.", value=True)
chk5 = st.checkbox("I understand why public web clients must never connect directly to vector database ports.", value=True)
chk6 = st.checkbox("I can architect an end-to-end authenticated and audited vector search backend in production.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Vector Security Basics and are ready for Chapter 20.2: Data Security & PII Protection!")
