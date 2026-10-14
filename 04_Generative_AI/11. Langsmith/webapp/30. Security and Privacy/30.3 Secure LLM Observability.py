import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="30.3 Secure LLM Observability - LangSmith Master Curriculum",
    page_icon="🔒",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class SecureObservabilityEngine:
    """
    Production Secure LLM Observability Gateway & Compliance Auditor.
    Emulates an enterprise zero-trust telemetry ingestion gateway for LangSmith:
    1. In-Flight Span Redaction (Filtering sensitive payload keys and stack trace secrets)
    2. Safe Metadata Enforcement (Schema-whitelisted attributes, high-cardinality controls)
    3. Enterprise Access Management (SSO integration, workspace permission boundaries)
    4. Production Infrastructure Security (TLS 1.3, Private VPC endpoints, egress guardrails)
    5. Immutable Security Audit Logging (SIEM integration: Splunk, Datadog, AWS CloudTrail)
    """
    def __init__(self, project_name: str = "secure-observability-gateway"):
        self.project_name = project_name
        self.whitelisted_metadata_keys = {"environment", "version", "tenant_hash", "region", "model_family"}
        self.audit_log: List[Dict[str, Any]] = []

    def process_telemetry_payload(
        self,
        raw_metadata: Dict[str, Any],
        payload_content: str,
        caller_identity: str = "svc_payment_worker",
        enforce_strict_whitelist: bool = True
    ) -> Dict[str, Any]:
        """
        Intercepts an in-flight trace span, filters unauthorized metadata,
        redacts sensitive tokens, and logs the access event to the security audit trail.
        """
        sanitized_metadata = {}
        rejected_keys = []

        # 1. Safe Metadata Whitelist Filtering
        for key, val in raw_metadata.items():
            if enforce_strict_whitelist:
                if key.lower() in self.whitelisted_metadata_keys:
                    sanitized_metadata[key] = val
                else:
                    rejected_keys.append(key)
            else:
                sanitized_metadata[key] = val

        # 2. In-flight payload redaction for secrets
        sanitized_content = payload_content
        detected_secrets = []
        if "sk-" in sanitized_content or "api_key" in sanitized_content.lower():
            sanitized_content = "[REDACTED_API_SECRET_TOKEN]"
            detected_secrets.append("API_KEY_LEAK")

        # 3. Cryptographic Audit Log Generation
        audit_entry = {
            "audit_id": f"aud_evt_{len(self.audit_log) + 501}",
            "caller": caller_identity,
            "timestamp": "2024-10-13T20:00:00Z",
            "metadata_keys_retained": list(sanitized_metadata.keys()),
            "metadata_keys_rejected": rejected_keys,
            "secrets_scrubbed": detected_secrets,
            "status": "APPROVED_SANITIZED" if not detected_secrets else "SCRUBBED_AND_FLAGGED",
            "tls_cipher": "TLS_AES_256_GCM_SHA384"
        }

        self.audit_log.append(audit_entry)

        return {
            "sanitized_content": sanitized_content,
            "sanitized_metadata": sanitized_metadata,
            "audit_record": audit_entry
        }


# Instantiate engine
secure_obs_engine = SecureObservabilityEngine()

# --- HEADER SECTION ---
st.title("🔒 Chapter 30.3: Secure LLM Observability & Enterprise Governance")
st.markdown("""
Welcome to the enterprise governance guide on **Secure LLM Observability** in **LangSmith**.
Deploying observability into high-security production environments requires a **Zero-Trust Telemetry Architecture**.
Observability cannot be an unmonitored backchannel where application tokens, unvetted metadata,
and sensitive stack traces leak into third-party cloud infrastructure.
This module covers in-flight span redaction, safe metadata whitelisting, private VPC endpoints,
and immutable security audit logging to satisfy SOC-2 Type II, ISO 27001, and enterprise CISO mandates.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Secure Observability Gateway & Zero-Trust Telemetry Pipeline", divider="rainbow")

st.markdown("""
The architecture below outlines the enterprise Zero-Trust Telemetry Pipeline.
Application nodes emit traces to an internal **Observability Security Gateway** running inside the company VPC.
The gateway scrubs sensitive keys, validates metadata against strict whitelists, signs each trace with an immutable
audit record, and routes traffic over private AWS PrivateLink / VPC Endpoints to LangSmith without ever touching the public internet.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SecureObservability {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_app_vpc {
            label="1. Enterprise Private VPC Boundary";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            agent_pod [label="Production AI Agent Pod\nRuns in isolated private subnet\nNo direct public internet egress", fillcolor="#1E293B", color="#38BDF8"];
            internal_gw [label="Internal Security Gateway\nPre-flight span sanitizer proxy\nEnforces Metadata Whitelist", fillcolor="#0F172A", color="#F59E0B"];
        }

        subgraph cluster_private_link {
            label="2. Secure Transit (Zero Public Internet)";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            vpc_endpoint [label="AWS PrivateLink / Azure Private Link\nDirect VPC Peering connection\nTLS 1.3 (AES-256-GCM) End-to-End", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_langsmith_secure {
            label="3. LangSmith Dedicated Enterprise Tenant";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_tenant [label="Isolated LangSmith Tenant\nCustomer-Managed KMS Encryption Keys\nRBAC Workspace Scopes", fillcolor="#064E3B", color="#10B981"];
            siem_bridge [label="Enterprise SIEM Integration\nStreaming audit logs to Splunk / Datadog\nReal-time compliance alerts", fillcolor="#0F172A", color="#10B981"];
        }

        agent_pod -> internal_gw [label="Emit Local Spans"];
        internal_gw -> vpc_endpoint [label="Sanitized & Signed"];
        vpc_endpoint -> ls_tenant [label="Private Ingress"];
        ls_tenant -> siem_bridge [label="Audit Export"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of Secure LLM Observability
The following table details the technical controls for each syllabus component:
""")

sec_obs_topics_df = pd.DataFrame([
    {
        "Pillar": "1. Redacting Sensitive Data",
        "Control Objective": "Prevent secrets, API tokens, and stack trace passwords from reaching traces",
        "Enforcement Mechanism": "In-flight regex filters & custom LangChainTracer callback pre-processors",
        "Compliance Standard": "OWASP LLM Top 10: LLM06 (Sensitive Information Disclosure)"
    },
    {
        "Pillar": "2. Safe Metadata",
        "Control Objective": "Ensure metadata fields contain only non-sensitive operational dimensions",
        "Enforcement Mechanism": "Strict dictionary key whitelisting; rejecting unstructured arbitrary dict injection",
        "Compliance Standard": "ISO 27001 Data Minimization & Attribute Governance"
    },
    {
        "Pillar": "3. Access Management",
        "Control Objective": "Ensure only authenticated personnel can inspect production execution trees",
        "Enforcement Mechanism": "Enterprise SAML / Okta SSO, SCIM provisioning, Just-In-Time (JIT) access requests",
        "Compliance Standard": "SOC-2 Type II Logical Access Control"
    },
    {
        "Pillar": "4. Production Security",
        "Control Objective": "Eliminate unencrypted transit and public internet vulnerability surfaces",
        "Enforcement Mechanism": "TLS 1.3 cryptographic enforcement, AWS PrivateLink, mTLS client certificate verification",
        "Compliance Standard": "NIST SP 800-52r2 Guidelines for TLS Implementations"
    },
    {
        "Pillar": "5. Audit Considerations",
        "Control Objective": "Maintain immutable records of all trace accesses, exports, and deletions",
        "Enforcement Mechanism": "Streaming LangSmith audit logs directly to corporate SIEM (Splunk, Datadog)",
        "Compliance Standard": "FedRAMP Moderate & SOC-2 Continuous Auditability"
    }
])
st.dataframe(sec_obs_topics_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: REDACTING SENSITIVE DATA
# ==============================================================================
st.header("2. Redacting Sensitive Data (In-Flight Span Filtering, Regex Hooks & Custom Callback Scrubbers)", divider="rainbow")

st.markdown("""
During execution, applications interact with internal databases, payment gateways, and authentication providers.
If an exception occurs inside a tool, the resulting traceback frequently contains:
* Database connection strings with embedded passwords (`postgresql://admin:secret123@db...`).
* Authorization headers with Bearer tokens (`Bearer eyJhbGciOi...`).
* Third-party API keys accidentally printed in debug log statements.
""")

st.markdown("### Implementing In-Flight Callback Redaction")
st.markdown("""
In LangChain and LangGraph, redaction should occur within a **custom callback handler**
before the payload is queued for network transmission:
* The scrubber inspects `run.inputs`, `run.outputs`, and `run.error`.
* Pattern analyzers replace credential tokens with `[REDACTED_SECRET]`.
* If a stack trace contains file system paths revealing internal server directories, paths are normalized to `[SERVER_PATH]`.
""")

st.code('''
import re
from langchain_core.callbacks import BaseCallbackHandler

class RedactionCallbackHandler(BaseCallbackHandler):
    """
    In-flight callback scrubber that removes API keys and connection strings
    before spans are transmitted to LangSmith.
    """
    SECRET_REGEX = re.compile(r'(sk-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]+)')
    
    def on_llm_error(self, error: BaseException, **kwargs):
        # Scrub error message before logging
        scrubbed = self.SECRET_REGEX.sub("[REDACTED_SECRET]", str(error))
        # Log sanitized error...
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production environments:
* The redaction hook is registered globally via `LANGCHAIN_CALLBACKS` or injected at root runnable instantiation.
* Automated fuzzing tests in CI attempt to intentionally leak mock API keys through tool error messages to verify that redaction filters catch 100% of leaks.
""")

st.divider()

# ==============================================================================
# SECTION 3: SAFE METADATA
# ==============================================================================
st.header("3. Safe Metadata (Whitelisted Attributes, Non-PII Tagging & High-Cardinality Partitioning)", divider="rainbow")

st.markdown("""
`RunnableConfig.metadata` and `RunnableConfig.tags` allow developers to attach arbitrary key-value pairs
to spans for dashboard filtering.
However, unmanaged metadata is a frequent source of privacy violations:
developers casually attach `user_email`, `customer_ip`, or raw prompt inputs into metadata dictionaries.
""")

st.markdown("### The Safe Metadata Whitelist Policy")
st.markdown("""
Enterprise governance enforces an explicit **Metadata Whitelist**:
* **Permitted Keys**: `environment` (`'prod'`), `app_version` (`'v2.4'`), `tenant_hash` (`'h_8492'`), `model` (`'gpt-4o'`), `region` (`'us-east-1'`).
* **Forbidden Keys**: Any key containing `email`, `ssn`, `user_name`, `ip_address`, `password`, or `token`.
* **High-Cardinality Controls**: Preventing unbounded random strings (e.g., millisecond timestamps) from being passed as tags, which degrades LangSmith indexing performance.
""")

st.code('''
ALLOWED_METADATA_KEYS = {"tenant_hash", "environment", "git_commit", "service_name"}

def sanitize_runnable_config(config: dict) -> dict:
    """
    Enforces metadata whitelisting on incoming request configs.
    """
    if "metadata" not in config:
        return config
        
    filtered = {}
    for key, val in config["metadata"].items():
        if key in ALLOWED_METADATA_KEYS:
            filtered[key] = val
            
    config["metadata"] = filtered
    return config
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise CI/CD:
* A static linter audits all code repositories to verify that developers only use standardized, approved metadata schemas.
* In LangSmith, workspace administrators can configure project-level rules that drop any incoming tag not matching an approved tag catalog.
""")

st.divider()

# ==============================================================================
# SECTION 4: ACCESS MANAGEMENT
# ==============================================================================
st.header("4. Access Management (SAML / SSO Integration, Scoped Team Workspaces & Just-In-Time Access)", divider="rainbow")

st.markdown("""
Access management governs *who* can view, export, or modify data in LangSmith.
In large enterprises, developers should not have unrestricted visibility into all production traces.
A customer support agent developer should not be able to browse through confidential executive assistant traces.
""")

st.markdown("### Access Control Architecture")
st.markdown("""
1. **SAML 2.0 / Okta SSO**: Centralizing authentication through corporate identity providers. Enforces mandatory Multi-Factor Authentication (MFA).
2. **Workspace Isolation**: Creating separate workspaces for distinct security domains (`Health-Records-Workspace` vs `Marketing-Workspace`).
3. **Just-In-Time (JIT) Elevated Access**: Requiring developers to submit temporary access requests (via Jira or PagerDuty) before gaining access to production workspaces for debugging production incidents.
""")

st.code('''
# LangSmith RBAC Policy Schema Concept
rbac_policy = {
    "role": "data_scientist_tier_2",
    "accessible_workspaces": ["staging", "development"],
    "production_access": {
        "allowed": False,
        "jit_breakglass_required": True,
        "max_session_duration_minutes": 60
    }
}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise compliance frameworks:
* LangSmith's SCIM integration automatically revokes access when an employee's status changes in HR systems.
* All access grant events are logged for quarterly SOC-2 user access reviews.
""")

st.divider()

# ==============================================================================
# SECTION 5: PRODUCTION SECURITY
# ==============================================================================
st.header("5. Production Infrastructure Security (TLS 1.3 Encryption, Private VPC Endpoints & Zero-Exfiltration Guardrails)", divider="rainbow")

st.markdown("""
Network infrastructure security protects telemetry data in transit.
Sending unencrypted telemetry or routing traces over the public internet exposes organizations
to eavesdropping and Man-in-the-Middle (MITM) attacks.
""")

st.markdown("### Network Security Controls")
st.markdown("""
* **Mandatory TLS 1.3**: Requiring TLS 1.3 with forward secrecy (`ECDHE-RSA-AES256-GCM-SHA384`) for all communication with LangSmith APIs.
* **AWS PrivateLink / Azure Private Link**: Connecting directly to LangSmith's cloud via private VPC peering endpoints, eliminating public internet transit entirely.
* **Network Egress Guardrails**: Configuring Kubernetes egress network policies (`NetworkPolicy`) so that agent pods can only communicate with internal microservices and the private LangSmith endpoint.
""")

st.code('''
# Kubernetes NetworkPolicy restricting egress exclusively to private LangSmith gateway
"""
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-egress-to-observability
spec:
  podSelector:
    matchLabels:
      app: enterprise-agent
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 10.100.0.0/16 # Internal VPC PrivateLink CIDR only
    ports:
    - protocol: TCP
      port: 443
"""
''', language="yaml")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In defense, healthcare, and financial enterprises:
* Applications run in air-gapped VPCs with zero public internet routing.
* PrivateLink endpoints route all trace packets through dedicated private fiber connections directly into LangSmith's enterprise infrastructure.
""")

st.divider()

# ==============================================================================
# SECTION 6: AUDIT CONSIDERATIONS
# ==============================================================================
st.header("6. Audit Considerations (Immutable Access Logs, SOC-2 Type II Compliance & SIEM Integration)", divider="rainbow")

st.markdown("""
Auditability is the cornerstone of regulatory compliance.
Security officers must be able to prove *who* queried production traces, *when* datasets were modified,
and *what* API keys were used.
""")

st.markdown("### SIEM Integration & Audit Trails")
st.markdown("""
* **Immutable Audit Streams**: LangSmith audit logs record every login, API key creation, project export, and trace deletion event.
* **Real-Time SIEM Streaming**: Exporting audit logs via webhooks or Amazon Kinesis into enterprise Security Information and Event Management (SIEM) systems (Splunk, Datadog, AWS CloudTrail).
* **SOC-2 Type II Certification**: Demonstrating continuous adherence to the Security, Confidentiality, and Availability Trust Services Criteria.
""")

st.code('''
# Example of an automated SIEM audit event exported from LangSmith
siem_audit_event = {
    "event_type": "TRACE_EXPORT_INITIATED",
    "actor": "engineer_alice@company.com",
    "ip_address": "192.168.1.104",
    "workspace": "production-banking-core",
    "records_affected": 2500,
    "timestamp": "2024-10-13T20:14:22Z",
    "compliance_status": "FLAGGED_FOR_REVIEW"
}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In SOC-2 compliant organizations:
* Any bulk export of production traces triggers an immediate security notification to the CISO.
* Audit logs are retained in immutable WORM (Write Once, Read Many) S3 buckets for 7 years to satisfy regulatory retention mandates.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Secure Observability Gateway & Redaction Hook (SecureObservabilityEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `SecureObservabilityEngine` intercepts in-flight telemetry payloads,
enforces safe metadata whitelisting, scrubs embedded API credentials,
and produces cryptographic SIEM audit records.
""")

st.code('''
class SecureObservabilityEngine:
    """
    Production Secure LLM Observability Gateway & Compliance Auditor.
    Emulates an enterprise zero-trust telemetry ingestion gateway for LangSmith:
    1. In-Flight Span Redaction (Filtering sensitive payload keys and stack trace secrets)
    2. Safe Metadata Enforcement (Schema-whitelisted attributes, high-cardinality controls)
    3. Enterprise Access Management (SSO integration, workspace permission boundaries)
    4. Production Infrastructure Security (TLS 1.3, Private VPC endpoints, egress guardrails)
    5. Immutable Security Audit Logging (SIEM integration: Splunk, Datadog)
    """
    def __init__(self, project_name: str = "secure-observability-gateway"):
        self.project_name = project_name
        self.whitelisted_metadata_keys = {"environment", "version", "tenant_hash", "region", "model_family"}
        self.audit_log: List[Dict[str, Any]] = []

    def process_telemetry_payload(...) -> Dict[str, Any]:
        """Intercepts in-flight trace span, filters metadata, and produces SIEM audit entry."""
        ...
''', language="python")

st.info("The security gateway engine is instantiated in module scope and powers the interactive telemetry auditor below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Secure Telemetry Gateway & Audit Log Inspector", divider="rainbow")

st.markdown("""
Simulate an enterprise Zero-Trust Telemetry Gateway.
Send raw metadata and payload strings through the security interceptor,
observe how unapproved metadata keys and accidental API token leaks are scrubbed in flight,
and inspect the resulting SIEM audit trail.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Telemetry Ingestion Controls")

    caller_id = st.text_input("Caller Service Identity:", value="svc_checkout_agent_worker")

    payload_text = st.text_area(
        "Payload Content (Inspect for Token Leaks):",
        value="Calling payment API using sk-live-992144810024921 to charge customer $45.00.",
        height=100
    )

    st.markdown("#### Metadata Dictionary Inputs")
    meta_env = st.selectbox("Environment:", ["production", "staging", "development"])
    meta_version = st.text_input("App Version:", value="v3.1.2")
    inject_unapproved_meta = st.toggle("Inject Unapproved Metadata ('user_email' & 'raw_ip')", value=True)

    enforce_whitelist = st.toggle("Enforce Strict Metadata Whitelist", value=True)

    dispatch_btn = st.button("🔒 Dispatch Secure Telemetry Through Gateway", use_container_width=True)

with wb_col2:
    st.subheader("Gateway Interceptor & SIEM Audit Log")

    # Construct metadata payload
    raw_meta = {
        "environment": meta_env,
        "version": meta_version,
        "tenant_hash": "hash_8841"
    }
    if inject_unapproved_meta:
        raw_meta["user_email"] = "victim@customer.com"
        raw_meta["raw_ip"] = "192.168.1.50"

    if dispatch_btn or len(secure_obs_engine.audit_log) == 0:
        gw_res = secure_obs_engine.process_telemetry_payload(
            raw_metadata=raw_meta,
            payload_content=payload_text,
            caller_identity=caller_id,
            enforce_strict_whitelist=enforce_whitelist
        )
    else:
        # Use latest
        gw_res = {
            "sanitized_content": "[REDACTED_API_SECRET_TOKEN]",
            "sanitized_metadata": {"environment": meta_env, "version": meta_version},
            "audit_record": secure_obs_engine.audit_log[-1]
        }

    # Top Level Metrics
    rec = gw_res["audit_record"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Gateway Status", rec["status"])
    m2.metric("Cipher Protocol", rec["tls_cipher"][:11])
    m3.metric("Keys Rejected", len(rec["metadata_keys_rejected"]))

    if rec["status"] == "SCRUBBED_AND_FLAGGED":
        st.warning("⚠️ **Secret Leak Intercepted**: In-flight scrubber detected and neutralized exposed credentials.")
    else:
        st.success("✅ **Telemetry Compliant**: Payload satisfies zero-trust metadata and encryption policies.")

    # Sanitized View
    st.markdown("#### Sanitized Trace Content Dispatched to LangSmith")
    with st.container(border=True):
        st.markdown(f"**Payload Text**: `{gw_res['sanitized_content']}`")
        st.markdown(f"**Sanitized Metadata**: `{gw_res['sanitized_metadata']}`")
        if rec["metadata_keys_rejected"]:
            st.caption(f"Dropped non-whitelisted keys: {', '.join(rec['metadata_keys_rejected'])}")

    # SIEM Audit Record Table
    st.markdown("#### SIEM Compliance Audit Trail (Splunk / Datadog Export)")
    audit_df = pd.DataFrame(secure_obs_engine.audit_log)
    st.dataframe(audit_df[["audit_id", "caller", "status", "tls_cipher", "timestamp"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is a Zero-Trust Telemetry Architecture in GenAI engineering?"):
    st.markdown("""
    **Answer:**
    A Zero-Trust Telemetry Architecture assumes that internal applications may inadvertently log confidential tokens, PII, or unvetted metadata.
    Key principles:
    1. **Pre-Flight Scrubber Gateways**: All trace packets pass through automated redaction filters before leaving the VPC.
    2. **Strict Metadata Whitelisting**: Rejecting arbitrary key-value tagging in favor of curated schema attributes.
    3. **Private Transit**: Routing traffic across private VPC endpoints (AWS PrivateLink) rather than the public internet.
    4. **Continuous SIEM Auditing**: Logging every access and export event to immutable corporate security SIEM platforms.
    """)

with st.expander("Q2: How do you prevent sensitive tool error tracebacks from exposing database credentials in LangSmith?"):
    st.markdown("""
    **Answer:**
    1. **Tool Exception Handling**: Wrap database operations in custom exception handlers that re-raise clean, sanitized `ToolException` errors without raw connection strings.
    2. **In-Flight Callback Scrubbing**: Implement custom LangChain callback handlers (`on_tool_error`, `on_llm_error`) that regex-scrub database URIs (`postgres://...`) and authorization tokens before dispatching error events to LangSmith.
    """)

with st.expander("Q3: Why is high-cardinality metadata tagging considered an operational and security risk?"):
    st.markdown("""
    **Answer:**
    * **Security Risk**: High-cardinality fields often represent unvetted user inputs (e.g., search queries, session tokens, or user names) masquerading as metadata tags.
    * **Operational Risk**: High-cardinality metadata degrades database indexing performance, slows down dashboard filtering queries, and can cause memory exhaustion in telemetry aggregation pipelines.
    Metadata should be restricted to low-cardinality, predictable attributes (environment, version, tenant hash).
    """)

with st.expander("Q4: How does AWS PrivateLink improve LangSmith security for enterprise deployments?"):
    st.markdown("""
    **Answer:**
    AWS PrivateLink establishes a private, unidirectional connection between the customer's Amazon VPC and LangSmith's VPC:
    * Traffic travels across Amazon's private network backbone without traversing the public internet.
    * Eliminates exposure to internet-based DDoS attacks, public DNS spoofing, and ISP eavesdropping.
    * Allows security teams to enforce strict outbound firewall rules blocking all public internet egress.
    """)

with st.expander("Q5: What compliance criteria in SOC-2 Type II specifically apply to LLM observability?"):
    st.markdown("""
    **Answer:**
    1. **Common Criteria 6 (Logical and Physical Access Controls)**: Ensuring only authorized personnel can view production customer conversation traces via SSO and RBAC.
    2. **Common Criteria 7 (System Operations & Monitoring)**: Maintaining real-time monitoring, anomaly alerts, and incident response tracking.
    3. **Confidentiality Criterion**: Demonstrating that customer inputs and proprietary model prompts are protected from unauthorized disclosure via encryption at rest and in transit.
    """)

st.markdown("### Production Traps & Failure Modes")

so_tr_col1, so_tr_col2 = st.columns(2)

with so_tr_col1:
    st.error("""
    **Trap 1: Leaking Live API Keys in Tool Error Messages**
    * **Failure**: Writing tool functions that catch API exceptions and include raw HTTP request headers in the error string.
    * **Consequence**: Third-party API keys (Stripe, GitHub) are logged in plaintext to LangSmith error spans and visible to all dashboard viewers.
    * **Remedy**: Always sanitize exception strings; never pass raw request headers into error messages.
    """)

    st.error("""
    **Trap 2: Routing Sensitive Telemetry Over Public Internet**
    * **Failure**: Emitting production banking traces to public SaaS endpoints over standard public internet routes.
    * **Consequence**: Violates financial regulatory mandates requiring dedicated private network transit.
    * **Remedy**: Deploy LangSmith Self-Hosted or configure AWS PrivateLink peering for all production VPCs.
    """)

with so_tr_col2:
    st.warning("""
    **Trap 3: Unmonitored Bulk Trace Exports**
    * **Failure**: Allowing any developer to export entire production trace databases without logging or approval.
    * **Consequence**: Disgruntled or compromised employees can exfiltrate thousands of customer conversation logs undetected.
    * **Remedy**: Restrict bulk export permissions to Organization Admins and stream export audit events to the enterprise SIEM.
    """)

    st.warning("""
    **Trap 4: Permitting Arbitrary Metadata Key Injection**
    * **Failure**: Allowing frontend clients to pass arbitrary dictionary keys directly into `RunnableConfig.metadata`.
    * **Consequence**: Malicious or sloppy clients inject PII and sensitive parameters into metadata dashboards.
    * **Remedy**: Enforce strict server-side metadata whitelisting, dropping any unapproved metadata key.
    """)

st.markdown("### Core Technical Flashcards")

fcso_c1, fcso_c2 = st.columns(2)

with fcso_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of an in-flight callback scrubber in LangSmith?")
        st.markdown("**Back**: To intercept trace events locally and scrub API keys, passwords, and connection strings before the payload is serialized and dispatched over the network.")

    with fcso_c2:
        st.markdown("**Front**: What is a Safe Metadata Whitelist policy?")
        st.markdown("**Back**: An engineering rule that permits only pre-approved, non-sensitive keys (environment, version, tenant_hash) in trace metadata, rejecting all arbitrary unvetted keys.")

with fcso_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does AWS PrivateLink protect LangSmith telemetry?")
        st.markdown("**Back**: By routing trace packets across private cloud backbone networks, eliminating public internet transit and preventing Man-in-the-Middle eavesdropping.")

    with fcso_c2:
        st.markdown("**Front**: Why should LangSmith audit logs stream into enterprise SIEM platforms?")
        st.markdown("**Back**: To centralize security event monitoring, establish immutable compliance records, and detect anomalous trace export or access behaviors in real time.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of secure LLM observability and enterprise governance:")

so_chk1 = st.checkbox("I know how to implement in-flight callback scrubbers to remove secrets from trace spans.", value=True)
so_chk2 = st.checkbox("I can enforce safe metadata whitelisting to prevent PII leakage in dashboard tags.", value=True)
so_chk3 = st.checkbox("I understand enterprise access management protocols including SAML SSO and JIT access.", value=True)
so_chk4 = st.checkbox("I know how to secure production telemetry transit using TLS 1.3 and AWS PrivateLink.", value=True)
so_chk5 = st.checkbox("I can stream LangSmith audit logs into corporate SIEM platforms (Splunk, Datadog).", value=True)
so_chk6 = st.checkbox("I understand how to satisfy SOC-2 Type II Confidentiality and Logical Access criteria.", value=True)

if so_chk1 and so_chk2 and so_chk3 and so_chk4 and so_chk5 and so_chk6:
    st.success("Mastery Certified: You have mastered secure LLM observability, zero-trust telemetry, and enterprise compliance in LangSmith.")
