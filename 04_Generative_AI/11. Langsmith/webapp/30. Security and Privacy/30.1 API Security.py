import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="30.1 API Security - LangSmith Master Curriculum",
    page_icon="🔐",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class APISecurityEngine:
    """
    Production Enterprise API Security & Credential Lifecycle Manager.
    Emulates secure key management for LangSmith in enterprise environments:
    1. Scoped API Key Management (Role-based scopes: Admin, Contributor, Viewer, Service-Account)
    2. Environment Variable Configuration (LANGSMITH_API_KEY, LANGSMITH_TRACING)
    3. Secret Management Integration (HashiCorp Vault, AWS Secrets Manager)
    4. Zero-Downtime Key Rotation (Dual-key grace period, active/retiring states)
    5. Role-Based Access Control (RBAC) & Organization Workspace Partitioning
    """
    def __init__(self, project_name: str = "enterprise-api-security"):
        self.project_name = project_name
        self.key_store = [
            {"key_id": "key_prod_svc_01", "name": "CI/CD Deployment Pipeline", "scope": "Contributor", "status": "active", "created_at": "2024-01-10", "expires_in_days": 45, "masked_secret": "ls__svc_...a89f"},
            {"key_id": "key_analytics_02", "name": "Looker Analytics Read-Only", "scope": "Viewer", "status": "active", "created_at": "2024-02-15", "expires_in_days": 120, "masked_secret": "ls__view_...b12c"},
            {"key_id": "key_legacy_03", "name": "Legacy Ingestion Worker", "scope": "Contributor", "status": "retiring", "created_at": "2023-11-01", "expires_in_days": 3, "masked_secret": "ls__leg_...e44d"}
        ]
        self._key_counter = 100

    def rotate_api_key(self, target_key_id: str) -> Dict[str, Any]:
        """
        Executes a zero-downtime key rotation:
        Marks current key as 'retiring' (grace period) and issues a new active key.
        """
        self._key_counter += 1
        new_key_id = f"key_rot_{self._key_counter}"

        target = None
        for k in self.key_store:
            if k["key_id"] == target_key_id:
                target = k
                break

        if not target:
            return {"status": "error", "message": f"Key {target_key_id} not found."}

        # Demote existing key to retiring with 7-day grace window
        target["status"] = "retiring"
        target["expires_in_days"] = 7

        # Generate new active key
        new_key = {
            "key_id": new_key_id,
            "name": f"{target['name']} (Rotated)",
            "scope": target["scope"],
            "status": "active",
            "created_at": "Today",
            "expires_in_days": 90,
            "masked_secret": f"ls__new_{self._key_counter}_...f9a1"
        }
        self.key_store.append(new_key)

        return {
            "status": "success",
            "message": f"Rotated key {target_key_id}. Old key demoted to 7-day grace period; new key {new_key_id} active.",
            "old_key_id": target_key_id,
            "new_key_id": new_key_id,
            "new_key_masked": new_key["masked_secret"]
        }


# Instantiate engine
api_sec_engine = APISecurityEngine()

# --- HEADER SECTION ---
st.title("🔐 Chapter 30.1: API Security & Secret Management")
st.markdown("""
Welcome to the enterprise security handbook for **API Security and Credential Management** in **LangSmith**.
In production GenAI systems, observability platforms ingest sensitive operational data and execute high-privilege operations.
A compromised LangSmith API key can expose internal dataset benchmarks, proprietary system prompts,
and confidential user conversation traces.
This module establishes strict security controls: scoped service account credentials, secret management integration,
automated zero-downtime key rotation, and role-based access control.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise API Security Architecture & Credential Lifecycle", divider="rainbow")

st.markdown("""
The architecture below outlines the credential lifecycle for LangSmith in a production cloud environment.
Developers and CI/CD pipelines never hardcode API keys. Instead, secrets are provisioned from HashiCorp Vault
or AWS Secrets Manager into container runtime environments. Dual-key rotation guarantees continuous ingestion
without service interruptions during regular 90-day credential refreshes.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph APISecurity {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_vault {
            label="1. Enterprise Secret Store (Vault / AWS KMS)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            vault [label="HashiCorp Vault / AWS Secrets Manager\nEncrypted at rest (AES-256)\nAutomated 90-day rotation policy", fillcolor="#1E293B", color="#38BDF8"];
            iam_auth [label="IAM Role / Kubernetes ServiceAccount\nShort-lived STS Token Auth", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_runtime {
            label="2. Application Container Runtime (Kubernetes)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            k8s_pod [label="AI Worker Pod\nInjected Environment Variables:\nLANGSMITH_API_KEY, LANGSMITH_TRACING=true", fillcolor="#1E293B", color="#38BDF8"];
            zero_code [label="Zero-Hardcoding Guardrail\nPre-commit trufflehog / git-secrets scan\nRejects commits containing raw keys", fillcolor="#0F172A", color="#EF4444"];
        }

        subgraph cluster_langsmith_cloud {
            label="3. LangSmith Cloud / Self-Hosted Gateway";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            auth_gateway [label="LangSmith Authentication Gateway\nValidates Bearer token & signature\nChecks key expiration & status", fillcolor="#0F172A", color="#10B981"];
            rbac_engine [label="RBAC Policy Engine\nScopes: Admin | Contributor | Viewer\nWorkspace Partitioning: 'Prod-Fin' vs 'Staging'", fillcolor="#0F172A", color="#38BDF8"];
            audit_log [label="Security Audit Log\nLogs key_id, IP, timestamp, action", fillcolor="#064E3B", color="#10B981"];
        }

        vault -> iam_auth;
        iam_auth -> k8s_pod [label="Inject at Pod Startup"];
        zero_code -> k8s_pod [label="Audit Codebase"];

        k8s_pod -> auth_gateway [label="HTTPS TLS 1.3 Request\nx-api-key header"];
        auth_gateway -> rbac_engine [label="Token Validated"];
        rbac_engine -> audit_log [label="Record Access"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of API Security
The following matrix details the operational specifications for each syllabus topic:
""")

api_sec_topics_df = pd.DataFrame([
    {
        "Pillar": "1. API Key Management",
        "Operational Standard": "Scoped, named service account credentials",
        "Enforcement Mechanism": "Assign principle of least privilege (e.g., CI runners receive Contributor scope, not Admin)",
        "Failure Risk": "Compromised admin key allows unauthorized deletion of production datasets"
    },
    {
        "Pillar": "2. Environment Variables",
        "Operational Standard": "Standardized LANGSMITH_* environment variable namespace",
        "Enforcement Mechanism": "Injected via Kubernetes Secrets or Docker container configs at runtime",
        "Failure Risk": "Hardcoded keys accidentally leaked to public GitHub repositories"
    },
    {
        "Pillar": "3. Secret Management",
        "Operational Standard": "HashiCorp Vault, AWS Secrets Manager, or Google Secret Manager",
        "Enforcement Mechanism": "Dynamic secret fetching or mutating webhook secret injection",
        "Failure Risk": "Plaintext configuration files stored unencrypted on developer laptops"
    },
    {
        "Pillar": "4. Key Rotation",
        "Operational Standard": "Zero-downtime dual-key rotation with 7-day grace period",
        "Enforcement Mechanism": "Issue new key -> deploy to workers -> verify ingestion -> revoke old key",
        "Failure Risk": "Sudden key invalidation causes production application crash and dropped traces"
    },
    {
        "Pillar": "5. Access Control",
        "Operational Standard": "Role-Based Access Control (RBAC) & Multi-Workspace Isolation",
        "Enforcement Mechanism": "Partition production, staging, and development into distinct LangSmith workspaces",
        "Failure Risk": "Junior developer accidentally edits production evaluation benchmarks"
    }
])
st.dataframe(api_sec_topics_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: API KEY MANAGEMENT
# ==============================================================================
st.header("2. API Key Management (Scoped Permissions, Organization Keys & Token Scopes)", divider="rainbow")

st.markdown("""
In enterprise deployments, personal developer API keys must never be used for automated applications.
Personal accounts change when employees transition teams or leave the organization.
Production systems require **Service Accounts** configured with specific, bounded scopes.
""")

st.markdown("### Permission Scopes in LangSmith")
st.markdown("""
1. **Organization Admin**: Complete administrative access. Can create workspaces, manage billing, invite users, and delete projects. Never used in application code.
2. **Workspace Admin**: Can manage project configurations, annotation queues, and datasets within a specific workspace.
3. **Contributor**: Can write new traces, log feedback, and read datasets. Standard scope for production runtime applications and CI/CD runners.
4. **Viewer (Read-Only)**: Can read traces, dashboards, and evaluation reports. Ideal for external reporting tools (Tableau, Looker) and auditor accounts.
""")

st.code('''
# LangSmith Python SDK client initialization using scoped service account
from langsmith import Client

# Client automatically reads LANGSMITH_API_KEY from environment
client = Client()

# Verifies connection and active organization without exposing secret in code
print(f"Connected to LangSmith Organization: {client.current_tenant.name}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production architectures:
* Each microservice receives its own dedicated Service Account API key (e.g., `svc-customer-support-agent`, `svc-code-reviewer`).
* If a specific service is compromised, security teams can revoke that single key instantly without impacting any other running system.
""")

st.divider()

# ==============================================================================
# SECTION 3: ENVIRONMENT VARIABLES
# ==============================================================================
st.header("3. Environment Variables (LANGSMITH_API_KEY, Container Ingestion & Runtime Isolation)", divider="rainbow")

st.markdown("""
LangChain and LangGraph natively integrate with LangSmith using standard environment variables.
When these variables are present in the process environment, tracing is enabled automatically
without requiring manual client instantiation or code refactoring.
""")

st.markdown("### Core LangSmith Environment Variable Namespace")
st.markdown("""
* **`LANGSMITH_API_KEY`**: The secret API key string (e.g., `lsv2_pt_...`).
* **`LANGSMITH_TRACING`**: Boolean string (`"true"` / `"false"`). Enables or disables trace emission globally.
* **`LANGSMITH_PROJECT`**: The destination project name (e.g., `"prod-financial-advisor"`). Defaults to `"default"` if omitted.
* **`LANGSMITH_ENDPOINT`**: Destination URL (default `"https://api.smith.langchain.com"`). Set to private domain for self-hosted enterprise VPC installations.
""")

st.code('''
# Secure environment injection in Docker / Kubernetes deployment
# Avoid passing keys via command-line arguments (visible in 'ps aux')
# Use Kubernetes Secrets:
"""
apiVersion: v1
kind: Pod
metadata:
  name: genai-agent-worker
spec:
  containers:
  - name: worker
    image: company/agent:v2.4
    env:
    - name: LANGSMITH_TRACING
      value: "true"
    - name: LANGSMITH_PROJECT
      value: "production-customer-bot"
    - name: LANGSMITH_API_KEY
      valueFrom:
        secretKeyRef:
          name: langsmith-credentials
          key: api-key
"""
''', language="yaml")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In secure cloud setups:
* Secrets are mounted as in-memory files (e.g., `/var/run/secrets/langsmith/token`) or dynamic environment variables.
* Containers run under non-root user accounts with read-only root filesystems to prevent memory dumps.
""")

st.divider()

# ==============================================================================
# SECTION 4: SECRET MANAGEMENT
# ==============================================================================
st.header("4. Secret Management (Vault, AWS Secrets Manager & Zero-Hardcoding Enforcement)", divider="rainbow")

st.markdown("""
Hardcoded secrets in source code represent the single most common cause of enterprise security breaches.
Even in private repositories, hardcoded secrets persist in Git commit histories permanently.
Enterprise policy enforces **Zero-Hardcoding Guardrails** integrated with dedicated Secret Managers.
""")

st.markdown("### Secret Management Ingestion Pattern")
st.markdown("""
1. **Centralized Vault**: Secrets are stored in HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault, encrypted using customer-managed KMS keys.
2. **Dynamic Ingestion**: When an application pod starts up, a sidecar container or Kubernetes mutating webhook fetches the latest API key and populates the pod's environment.
3. **Pre-Commit Linting**: CI/CD pipelines run automated scanning tools (`trufflehog`, `gitleaks`) that scan every Git commit diff for regex patterns matching `lsv2_pt_...`. Any commit containing a key is blocked immediately.
""")

st.code('''
# Example of dynamic secret retrieval in production startup script
import boto3
import json

def get_langsmith_secret_from_aws(secret_name="prod/langsmith/api_key", region="us-east-1") -> str:
    """Fetches API key from AWS Secrets Manager using IAM role credentials."""
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    secret_dict = json.loads(response["SecretString"])
    return secret_dict["LANGSMITH_API_KEY"]
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In Fortune 500 deployments:
* API keys are never viewed by human engineers in plaintext after creation.
* Secret creation scripts generate keys via Terraform and store them directly into Vault without human eyes ever seeing the token.
""")

st.divider()

# ==============================================================================
# SECTION 5: KEY ROTATION
# ==============================================================================
st.header("5. Key Rotation (Zero-Downtime Dual-Key Grace Periods & Automated Renewal)", divider="rainbow")

st.markdown("""
Enterprise security standards (SOC-2, ISO 27001) mandate that API keys must be rotated every 90 days.
However, abrupt key revocation causes production outages: if workers across 50 Kubernetes nodes
are still using Key A when Key A is invalidated, hundreds of in-flight user requests drop traces.
""")

st.markdown("### Zero-Downtime Rotation Lifecycle")
st.markdown("""
1. **Step 1 (Generate New Key)**: Provision Key B while Key A remains fully active.
2. **Step 2 (Update Secret Store)**: Update Vault / AWS Secrets Manager with Key B.
3. **Step 3 (Rolling Pod Restart)**: Kubernetes performs a rolling deployment; new pods spin up with Key B while existing pods complete work with Key A.
4. **Step 4 (Grace Period)**: Key A is placed in a 7-day 'retiring' state. LangSmith accepts requests from both keys.
5. **Step 5 (Revocation)**: Once audit logs confirm zero requests have used Key A for 48 hours, Key A is permanently deleted.
""")

st.code('''
# Automated key rotation script using LangSmith Admin API
from langsmith import Client

client = Client() # Admin credentials

def execute_zero_downtime_rotation(old_key_id: str):
    # 1. Create new service key
    new_key = client.create_api_key(description="Production Ingestion Key (Rotated)")
    print(f"Generated new key: {new_key.id}")
    
    # 2. Update AWS Secrets Manager
    update_aws_secret(new_key.secret)
    
    # 3. Schedule old key revocation in 7 days
    # (Existing workers can drain without dropped traces)
    schedule_key_deletion(old_key_id, delay_days=7)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In automated infrastructure:
* Key rotation is executed by an automated cron job or AWS Lambda function every 90 days.
* Zero human intervention is required, and zero production requests are dropped.
""")

st.divider()

# ==============================================================================
# SECTION 6: ACCESS CONTROL
# ==============================================================================
st.header("6. Access Control (Role-Based Access Control, Workspace Partitioning & Least Privilege)", divider="rainbow")

st.markdown("""
Role-Based Access Control (RBAC) ensures that team members and service accounts possess only
the minimum permissions necessary to perform their roles.
In LangSmith, RBAC is enforced through **Workspaces** and **Permission Roles**.
""")

st.markdown("### Enterprise Workspace Partitioning")
st.markdown("""
* **Environment Partitioning**: Separate workspaces for `Production`, `Staging`, and `Development`. A bug or rogue script in `Development` cannot overwrite `Production` datasets.
* **Business Unit Partitioning**: Partitioning sensitive customer support traces (`Support-Workspace`) from internal financial analytics (`Finance-Workspace`).
* **Audit Trail Compliance**: Every API request to LangSmith logs the caller's key identity, IP address, and timestamp, providing an immutable audit trail for compliance officers.
""")

st.code('''
# Verifying least-privilege scope in code
from langsmith import Client

# A 'Viewer' client can query runs but will throw PermissionError on create_dataset
viewer_client = Client(api_key="ls__view_only_key")

try:
    viewer_client.create_dataset("unauthorized_dataset")
except Exception as e:
    print(f"Access correctly denied by RBAC: {e}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise organizations:
* Single Sign-On (SSO) via Okta or Azure AD is integrated with LangSmith.
* Team membership automatically syncs via SCIM: developers assigned to the Finance team in Okta automatically gain Viewer access to the Finance LangSmith workspace, and lose it upon offboarding.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise API Security & Key Lifecycle Manager (APISecurityEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `APISecurityEngine` models enterprise credential management, role-based key scoping,
and zero-downtime key rotation with grace periods.
""")

st.code('''
class APISecurityEngine:
    """
    Production Enterprise API Security & Credential Lifecycle Manager.
    Emulates secure key management for LangSmith in enterprise environments:
    1. Scoped API Key Management (Role-based scopes: Admin, Contributor, Viewer)
    2. Environment Variable Configuration (LANGSMITH_API_KEY, LANGSMITH_TRACING)
    3. Secret Management Integration (HashiCorp Vault, AWS Secrets Manager)
    4. Zero-Downtime Key Rotation (Dual-key grace period, active/retiring states)
    5. Role-Based Access Control (RBAC) & Organization Workspace Partitioning
    """
    def __init__(self, project_name: str = "enterprise-api-security"):
        self.project_name = project_name
        self.key_store = [...]
        self._key_counter = 100

    def rotate_api_key(self, target_key_id: str) -> Dict[str, Any]:
        """
        Executes a zero-downtime key rotation:
        Marks current key as 'retiring' (grace period) and issues a new active key.
        """
        # Demotes current key, generates active replacement, preserves uptime
        ...
''', language="python")

st.info("The security engine is instantiated in module scope and powers the interactive credential rotation workbench below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Secret Manager Sandbox & Zero-Downtime Key Rotator", divider="rainbow")

st.markdown("""
Manage enterprise LangSmith API credentials in a simulated HashiCorp Vault / Secret Store.
Select an active production key, execute an automated zero-downtime rotation,
and observe how dual-key grace periods preserve continuous application uptime.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Credential Management Controls")

    target_key = st.selectbox(
        "Select Key to Rotate:",
        options=[k["key_id"] for k in api_sec_engine.key_store if k["status"] == "active"],
        format_func=lambda kid: f"{kid} — {next(k['name'] for k in api_sec_engine.key_store if k['key_id'] == kid)}"
    )

    rotation_reason = st.selectbox(
        "Rotation Trigger:",
        options=["Scheduled 90-Day Policy", "Suspected Credential Leak", "Team Offboarding"]
    )

    confirm_rotate = st.button("🔄 Execute Zero-Downtime Rotation", use_container_width=True)

with wb_col2:
    st.subheader("Enterprise Key Store & Vault Registry")

    if confirm_rotate:
        rot_res = api_sec_engine.rotate_api_key(target_key_id=target_key)
        if rot_res["status"] == "success":
            st.success(f"✅ **Rotation Complete**: {rot_res['message']}")
        else:
            st.error(rot_res["message"])

    # Display active key inventory
    st.markdown("#### Active & Retiring API Key Inventory")
    keys_df = pd.DataFrame(api_sec_engine.key_store)
    st.dataframe(keys_df[["key_id", "name", "scope", "status", "expires_in_days", "masked_secret"]], use_container_width=True)

    with st.container(border=True):
        st.markdown("**Enterprise Security Policy Checklist:**")
        st.markdown("- Zero hardcoded keys detected in repository (`gitleaks: clean`).")
        st.markdown("- Secrets encrypted at rest via AWS KMS CMK (`aes-256-gcm`).")
        st.markdown("- Dual-key grace period configured for 7 days.")
        st.markdown("- Automated audit logging enabled on all authentication endpoints.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What are the security risks of hardcoding a LangSmith API key in application source code?"):
    st.markdown("""
    **Answer:**
    1. **Permanent Git History Exposure**: Even if deleted in a later commit, the secret persists permanently in Git commit trees unless purged with BFG Repo-Cleaner.
    2. **Supply Chain Leakage**: Public repositories, CI build logs, or compromised dependency scanners can extract the key.
    3. **Unauthorized Data Access**: An attacker with a LangSmith API key can view confidential customer conversation traces, extract proprietary benchmark datasets, or corrupt evaluation baselines.
    Always inject keys via environment variables populated by HashiCorp Vault or AWS Secrets Manager.
    """)

with st.expander("Q2: How does a Zero-Downtime Key Rotation workflow operate in practice?"):
    st.markdown("""
    **Answer:**
    1. **Dual-Key Support**: The auth system accepts both the existing key (Key A) and a newly provisioned key (Key B).
    2. **Generate Key B**: Admin provisions Key B while Key A remains active.
    3. **Deploy Key B**: Update the secret manager; worker pods pick up Key B via rolling restarts.
    4. **Grace Period**: Key A is marked as 'retiring' for 7 days to allow in-flight batch jobs to complete.
    5. **Revocation**: Key A is deleted once traffic logs show zero requests for 48 hours.
    This prevents dropped requests during credential rotation.
    """)

with st.expander("Q3: What is the Principle of Least Privilege in LangSmith RBAC?"):
    st.markdown("""
    **Answer:**
    The Principle of Least Privilege mandates granting only the minimum permissions required for a role:
    * **Automated Ingestion Workers**: Assigned **Contributor** scope (can write runs, cannot delete projects or view billing).
    * **Analytics Dashboards (Looker/Tableau)**: Assigned **Viewer** scope (read-only, cannot modify datasets or delete traces).
    * **Human Developers**: Restricted to `Development` or `Staging` workspaces; only authorized Release Engineers have Contributor access to the `Production` workspace.
    """)

with st.expander("Q4: Why should personal user API keys never be used in production CI/CD pipelines?"):
    st.markdown("""
    **Answer:**
    Personal API keys are tied to individual human identities. If that employee leaves the company or changes teams:
    1. Their corporate account is de-provisioned, immediately invalidating the key and breaking production CI/CD builds.
    2. Audit logs attribute automated pipeline actions to the individual human rather than a service identity, corrupting compliance records.
    Always use non-human **Service Accounts** for automated workloads.
    """)

with st.expander("Q5: How does LangSmith handle self-hosted enterprise deployments with strict VPC egress controls?"):
    st.markdown("""
    **Answer:**
    For enterprises with air-gapped or strict VPC egress boundaries (defense, banking):
    * LangSmith offers a self-hosted installation (Kubernetes Helm chart) running entirely within the customer's private AWS/GCP VPC.
    * Applications point to internal DNS: `LANGSMITH_ENDPOINT="https://langsmith.internal.company.com"`.
    * No telemetry data ever leaves the customer's secure network boundary.
    """)

st.markdown("### Production Traps & Failure Modes")

as_tr_col1, as_tr_col2 = st.columns(2)

with as_tr_col1:
    st.error("""
    **Trap 1: Hardcoding Keys in Dockerfile ENV Directives**
    * **Failure**: Writing `ENV LANGSMITH_API_KEY="lsv2_pt_..."` inside a Dockerfile.
    * **Consequence**: The API key is baked into the public image layers; anyone pulling the container image can extract the key with `docker history`.
    * **Remedy**: Never put secrets in Dockerfiles. Inject environment variables dynamically at pod runtime via Kubernetes secret references.
    """)

    st.error("""
    **Trap 2: Abrupt Single-Key Invalidation (Outage Trigger)**
    * **Failure**: Revoking an old API key immediately upon generating a new one without a grace period.
    * **Consequence**: In-flight requests across distributed Kubernetes nodes fail authentication, resulting in hundreds of dropped customer traces and 500 error alerts.
    * **Remedy**: Always maintain a dual-key grace period (5-7 days) during rotation.
    """)

with as_tr_col2:
    st.warning("""
    **Trap 3: Reusing Single Global Key Across All Environments**
    * **Failure**: Using the same API key in Local Development, Staging, and Production.
    * **Consequence**: Test traces from local developer laptops pollute production project dashboards, corrupting production SLA metrics.
    * **Remedy**: Partition environments into distinct LangSmith workspaces with dedicated scoped keys.
    """)

    st.warning("""
    **Trap 4: Missing Secret Scanning in Git Pre-Commit Hooks**
    * **Failure**: Relying solely on developer discipline to avoid committing raw API keys.
    * **Consequence**: A junior developer commits an experimental notebook containing an exposed key to GitHub.
    * **Remedy**: Install pre-commit hooks (`trufflehog` or `gitleaks`) that automatically block Git commits containing secret patterns.
    """)

st.markdown("### Core Technical Flashcards")

fcas_c1, fcas_c2 = st.columns(2)

with fcas_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the four core LangSmith environment variables?")
        st.markdown("**Back**: `LANGSMITH_API_KEY` (secret token), `LANGSMITH_TRACING` ('true'), `LANGSMITH_PROJECT` (project name), and `LANGSMITH_ENDPOINT` (gateway URL).")

    with fcas_c2:
        st.markdown("**Front**: What is the purpose of a 7-day grace period in key rotation?")
        st.markdown("**Back**: It allows existing distributed worker pods and long-running batch jobs to complete using the retiring key while new pods adopt the active key, preventing dropped requests.")

with fcas_c1:
    with st.container(border=True):
        st.markdown("**Front**: Why should production applications use Service Account credentials instead of user keys?")
        st.markdown("**Back**: Service accounts decouple system uptime from individual employee employment status, provide bounded scopes, and preserve accurate compliance audit trails.")

    with fcas_c2:
        st.markdown("**Front**: What tool prevents developers from accidentally committing LangSmith keys to Git?")
        st.markdown("**Back**: Automated pre-commit secret scanners such as `trufflehog` or `gitleaks` configured to detect `lsv2_` API key token patterns.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of enterprise API security and secret management:")

as_chk1 = st.checkbox("I know how to provision scoped service account keys with least-privilege permissions.", value=True)
as_chk2 = st.checkbox("I understand the LANGSMITH_* environment variable namespace and container injection patterns.", value=True)
as_chk3 = st.checkbox("I can integrate LangSmith credentials with HashiCorp Vault and AWS Secrets Manager.", value=True)
as_chk4 = st.checkbox("I know how to execute zero-downtime key rotation with dual-key grace periods.", value=True)
as_chk5 = st.checkbox("I can configure pre-commit secret scanners (trufflehog/gitleaks) to enforce zero hardcoding.", value=True)
as_chk6 = st.checkbox("I understand how to partition Production, Staging, and Dev into isolated LangSmith workspaces.", value=True)

if as_chk1 and as_chk2 and as_chk3 and as_chk4 and as_chk5 and as_chk6:
    st.success("Mastery Certified: You have mastered enterprise API security, secret management, and zero-downtime key rotation in LangSmith.")
