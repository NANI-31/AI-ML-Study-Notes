import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.3 Enterprise Security - RAG Master Curriculum",
    page_icon="🏢",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class EnterpriseComplianceEngine:
    """
    Enterprise Multi-Tenant Security, Granular ACL Authorization,
    and Immutable Audit Compliance Engine.
    Operates using strictly allowed standard data structures.
    """
    def __init__(self):
        # Multi-tenant document database with granular user/group ACLs
        self.doc_store: Dict[str, Dict[str, Any]] = {
            "DOC_T1_001": {
                "tenant_id": "TENANT_ACME",
                "title": "ACME Cloud Architecture Specification",
                "owner_user": "alice_acme",
                "allowed_groups": ["acme_engineers", "acme_admins"],
                "allowed_users": ["alice_acme", "bob_acme"],
                "content": "ACME Kubernetes clusters utilize Cilium eBPF for networking."
            },
            "DOC_T1_002": {
                "tenant_id": "TENANT_ACME",
                "title": "ACME Executive Payroll Ledger",
                "owner_user": "cfo_acme",
                "allowed_groups": ["acme_executives"],
                "allowed_users": ["cfo_acme"],
                "content": "Executive annual bonuses approved for fiscal year 2026."
            },
            "DOC_T2_001": {
                "tenant_id": "TENANT_STARK",
                "title": "Stark Industries Arc Reactor Blueprints",
                "owner_user": "tony_stark",
                "allowed_groups": ["stark_rd"],
                "allowed_users": ["tony_stark", "pepper_stark"],
                "content": "Clean energy palladium core replacement design specifications."
            }
        }
        # Immutable compliance audit log ledger
        self.audit_trail: List[Dict[str, Any]] = []
        self._event_counter = 1000

    def evaluate_document_access(
        self,
        requester_user: str,
        requester_tenant: str,
        requester_groups: List[str],
        doc_id: str
    ) -> Tuple[bool, str]:
        """
        Evaluates strict multi-tenant isolation and granular document-level ACLs.
        Rule 1: Hard Tenant Isolation - Requester tenant must match Document tenant.
        Rule 2: User/Group ACL - Requester must be explicit allowed user or member of allowed group.
        """
        if doc_id not in self.doc_store:
            return False, "DENY: Document not found in tenant index."

        doc = self.doc_store[doc_id]

        # 1. Tenant Isolation Verification
        if requester_tenant != doc["tenant_id"]:
            return False, f"DENY: Tenant Boundary Violation! Requester '{requester_tenant}' cannot access '{doc['tenant_id']}' data."

        # 2. Document ACL Check
        if requester_user in doc["allowed_users"]:
            return True, f"ALLOW: Direct user ACL match for '{requester_user}'."

        for grp in requester_groups:
            if grp in doc["allowed_groups"]:
                return True, f"ALLOW: Group ACL membership match for group '{grp}'."

        return False, f"DENY: User '{requester_user}' lacks required group or individual permissions for document."

    def execute_rag_access(
        self,
        requester_user: str,
        requester_tenant: str,
        requester_groups: List[str],
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Executes access check and generates an immutable compliance audit record.
        """
        self._event_counter += 1
        allowed, reason = self.evaluate_document_access(
            requester_user, requester_tenant, requester_groups, doc_id
        )

        audit_entry = {
            "event_id": f"AUDIT-EVT-{self._event_counter}",
            "user_id": requester_user,
            "tenant_id": requester_tenant,
            "doc_id": doc_id,
            "decision": "GRANTED" if allowed else "BLOCKED",
            "policy_reason": reason
        }
        self.audit_trail.append(audit_entry)

        return {
            "doc_id": doc_id,
            "allowed": allowed,
            "reason": reason,
            "audit_id": audit_entry["event_id"],
            "content": self.doc_store[doc_id]["content"] if allowed and doc_id in self.doc_store else None
        }

    def execute_gdpr_deletion(self, doc_id: str, requester_tenant: str) -> Dict[str, Any]:
        """
        Simulates cryptographic erasure / GDPR Right to be Forgotten.
        Purges document from active index and appends compliance deletion tombstone.
        """
        if doc_id not in self.doc_store:
            return {"success": False, "message": "Document not found."}

        if self.doc_store[doc_id]["tenant_id"] != requester_tenant:
            return {"success": False, "message": "Unauthorized: Cross-tenant deletion prohibited."}

        # Remove from active store
        del self.doc_store[doc_id]
        self._event_counter += 1

        tombstone_entry = {
            "event_id": f"AUDIT-EVT-{self._event_counter}",
            "user_id": "COMPLIANCE_OFFICER",
            "tenant_id": requester_tenant,
            "doc_id": doc_id,
            "decision": "CRYPTO_PURGED",
            "policy_reason": "GDPR Article 17 Right to Erasure Executed. Vector and raw payload purged."
        }
        self.audit_trail.append(tombstone_entry)

        return {
            "success": True,
            "message": f"Document '{doc_id}' successfully purged from vector store and primary index.",
            "tombstone_id": tombstone_entry["event_id"]
        }


# --- HERO SECTION ---
st.title("24.3 Enterprise Security")
st.caption("User-Level Permissions, Granular Document ACLs, Multi-Tenant Isolation, Compliance Audit Logging & GDPR Erasure")
st.markdown("""
In enterprise B2B SaaS, an AI system rarely serves a single team or organization.
It must securely support **thousands of distinct corporate tenants, millions of users, and granular file-level permissions synchronized from Google Drive, SharePoint, and Salesforce**.
A security breach where **Tenant A retrieves confidential contracts belonging to Tenant B is an existential business failure**.
**Enterprise RAG Security** enforces complete tenant isolation (siloed vs partitioned vector namespaces),
granular Access Control Lists (ACLs) mapped to enterprise Active Directory groups,
tamper-evident immutable audit trails for regulatory compliance (SOC 2, ISO 27001), and cryptographic data erasure pipelines.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Multi-Tenant Isolation & Immutable Audit Architecture", divider="rainbow")

st.markdown("""
The architecture below illustrates the multi-tenant isolation and compliance auditing workflow for enterprise RAG.
Requests are routed to **Tenant-Isolated Vector Namespaces**, evaluated against **Granular Document-Level ACLs**,
and logged to an **Immutable SIEM Audit Log Ledger** for SOC 2 compliance.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    enterprise_blueprint_dot = """
    digraph EnterpriseSecurity {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_ingress_id {
            label="1. Ingress & Identity Propagation Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            UserTenantA [label="User Alice (Tenant ACME)\nJWT: {tenant: ACME, groups: [eng]}", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            UserTenantB [label="User Tony (Tenant STARK)\nJWT: {tenant: STARK, groups: [rd]}", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            TenantRouter [label="Tenant Isolation Router\nEnforces Hard Namespace Partitioning", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_vector_namespaces {
            label="2. Multi-Tenant Vector Database Tier (Cryptographic Namespace Partitioning)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            NamespaceACME [label="Namespace: TENANT_ACME\nEncrypted with KMS Key A", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            NamespaceSTARK [label="Namespace: TENANT_STARK\nEncrypted with KMS Key B", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_acl_engine {
            label="3. Document-Level ACL Resolution Tier (Active Directory / LDAP Sync)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            ACLEvaluator [label="Granular ACL Policy Engine\nChecks doc.allowed_groups & doc.allowed_users", shape=ellipse, fillcolor="#451A03", fontcolor="#FED7AA", color="#F59E0B"];
            AuthorizedContext [label="Authorized Context Chunks\nDelivered to Generator", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
        }

        subgraph cluster_audit_compliance {
            label="4. Compliance Audit & Data Lifecycle Tier (SOC 2 / GDPR)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#EC4899";
            fontcolor="#F472B6";
            fontsize=11;

            AuditLedger [label="Immutable Audit Ledger (Kafka / SIEM)\nRecords: user, tenant, doc_id, decision", shape=component, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            GDPRPurge [label="Cryptographic Erasure Worker\nRight-to-be-Forgotten Vector Purge", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#EC4899"];
        }

        UserTenantA -> TenantRouter;
        UserTenantB -> TenantRouter;

        TenantRouter -> NamespaceACME [label="Tenant ACME Traffic Only"];
        TenantRouter -> NamespaceSTARK [label="Tenant STARK Traffic Only"];

        NamespaceACME -> ACLEvaluator [label="Candidate Chunks"];
        NamespaceSTARK -> ACLEvaluator [label="Candidate Chunks"];

        ACLEvaluator -> AuthorizedContext [label="Passed ACL Verification"];
        ACLEvaluator -> AuditLedger [label="Log ALLOW / DENY Event", style=dotted, color="#EC4899"];

        GDPRPurge -> NamespaceACME [label="Purge Vector on Request", style=dashed, color="#EC4899"];
    }
    """
    st.graphviz_chart(enterprise_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: USER-LEVEL PERMISSIONS
# ==============================================================================
st.header("2. User-Level Permissions (Identity Propagation, Personal Knowledge Stores & Per-User ACLs)", divider="rainbow")

st.markdown("""
Enterprise knowledge bases contain data private to individual users—such as personal draft emails, private Notion workspaces,
and 1-on-1 performance review notes.
When an employee interacts with a RAG application, their identity must be propagated through every stage of the pipeline via **Identity Propagation**.
""")

col_up1, col_up2 = st.columns([1, 1])

with col_up1:
    st.markdown("### Identity Propagation Principles:")
    st.markdown("""
    * **Downstream Context Injection**:
      The web client transmits an OIDC JWT bearer token. The API Gateway verifies the token and injects the authenticated `user_id` into the downstream gRPC retrieval metadata headers.
    * **Personal Knowledge Siloing**:
      Personal documents are indexed with a mandatory ownership tag:
      `{"owner_user_id": "alice_acme", "visibility": "Private"}`.
    * **Pre-Retrieval Predicate**:
      Vector searches for private personal queries enforce:
      `filter = {"$or": [{"visibility": "Public"}, {"owner_user_id": user_id}]}`.
    """)

with col_up2:
    st.markdown("### User-Level Access Matrix")
    user_matrix_df = pd.DataFrame({
        "Document Type": ["Team Engineering Wiki", "Personal Sprint Scratchpad", "Manager 1-on-1 Meeting Notes"],
        "Visibility Scope": ["Engineering Department", "Private to Alice", "Alice + Manager Bob"],
        "Vector Metadata Filter": ["dept == 'Engineering'", "owner == 'alice'", "users IN ['alice', 'bob']"],
        "Can Colleague Charlie View?": ["Yes", "Denied (403)", "Denied (403)"]
    })
    st.dataframe(user_matrix_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: DOCUMENT-LEVEL PERMISSIONS
# ==============================================================================
st.header("3. Document-Level Permissions (Granular ACL Synchronization, LDAP/Active Directory Mapping)", divider="rainbow")

st.markdown("""
Enterprise file storage systems (Google Drive, SharePoint, Box, Confluence) maintain intricate permission structures:
a single document may be readable by *"All Full-Time Employees in Legal"*, plus *"Contractor Dan"*, but strictly excluding *"Interns"*.
Synchronizing these granular Access Control Lists (ACLs) with high-dimensional vector databases is a primary enterprise engineering challenge.
""")

col_dp1, col_dp2 = st.columns([1, 1])

with col_dp1:
    st.markdown("### Ingestion ACL Synchronization Patterns:")
    st.markdown("""
    1. **Metadata List Tagging**:
       During document chunk ingestion, extract the source system's Access Control List and attach it as vector metadata:
       ```python
       chunk_metadata = {
           "doc_id": "SP-DOC-8891",
           "allowed_groups": ["group:legal_counsel", "group:compliance_officers"],
           "allowed_users": ["user:dan_contractor"],
           "denied_users": []
       }
       ```
    2. **Active Directory Webhook Sync**:
       When an employee joins or leaves an AD security group, a background worker updates the user's cached permission manifest in Redis without requiring re-indexing of documents.
    """)

with col_dp2:
    st.markdown("### Ingestion ACL Synchronization Flow")
    dot_acl_sync = """
    digraph ACLSync {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        SharePoint [label="SharePoint / Confluence\nFile Permissions Updated", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        WebhookWorker [label="Permission Sync Worker\nExtracts Group SIDs & User ACLs", shape=ellipse, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        VectorMetadata [label="Update Vector Chunks in Milvus\nUpsert allowed_groups Metadata", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        SharePoint -> WebhookWorker -> VectorMetadata;
    }
    """
    st.graphviz_chart(dot_acl_sync, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: TENANT ISOLATION
# ==============================================================================
st.header("4. Tenant Isolation (Multi-Tenancy Architectures: Shared vs Partitioned vs Siloed Vectors)", divider="rainbow")

st.markdown("""
In multi-tenant B2B architectures, customer data must be isolated to prevent cross-tenant data leakage.
There are three fundamental multi-tenancy deployment models for vector databases:
""")

col_ti1, col_ti2 = st.columns([1, 1])

with col_ti1:
    st.markdown("### Multi-Tenancy Architecture Comparison:")
    tenant_comp_df = pd.DataFrame({
        "Model": ["Siloed (Instance-per-Tenant)", "Partitioned (Namespace/Collection)", "Shared (Metadata Filtered)"],
        "Isolation Level": ["Cryptographic / Physical Hardware", "Logical Collection Boundary", "Software Filter (Row-Level)"],
        "Cost Efficiency": ["Low (High infrastructure overhead)", "High (Optimal resource sharing)", "Highest (Single cluster)"],
        "Leakage Risk": ["Zero (Hardware separated)", "Near Zero (Engine partition)", "Moderate (Filter bug risk)"],
        "Target Customers": ["Government, Banks, Defense", "Enterprise B2B SaaS", "Free Tier, Low-Cost Consumers"]
    })
    st.dataframe(tenant_comp_df, use_container_width=True)

with col_ti2:
    st.markdown("### Partitioned Multi-Tenancy Topology")
    dot_tenant = """
    digraph TenantTopology {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Gateway [label="Multi-Tenant API Gateway\nInspects tenant_id Claim in JWT", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        TenantA_Col [label="Tenant ACME Collection\nNamespace: tenant_acme", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        TenantB_Col [label="Tenant STARK Collection\nNamespace: tenant_stark", shape=cylinder, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];

        Gateway -> TenantA_Col [label="Route Tenant ACME"];
        Gateway -> TenantB_Col [label="Route Tenant STARK"];
    }
    """
    st.graphviz_chart(dot_tenant, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: AUDIT LOGGING
# ==============================================================================
st.header("5. Audit Logging (Immutable Compliance Trails, Query Lineage, Tamper-Evident Logs & SIEM)", divider="rainbow")

st.markdown("""
Regulatory frameworks (**SOC 2 Type II**, **ISO 27001**, **HIPAA Security Rule**) mandate that enterprise RAG systems maintain
**tamper-evident, immutable audit trails** of every access attempt.
In the event of an internal security investigation, the audit log must reconstruct the exact query lineage:
* *Who* asked the question (`user_id`, `ip_address`).
* *What* exact document chunks were retrieved and included in the prompt context.
* *Which* clearance policies authorized or denied the access attempt.
* *What* response was synthesized by the language model.
""")

col_al1, col_al2 = st.columns([1, 1])

with col_al1:
    st.markdown("### Anatomy of an Immutable Audit Event:")
    st.code('''
{
  "event_id": "AUDIT-EVT-99812",
  "timestamp": "2026-10-13T10:14:22Z",
  "requester": {
    "user_id": "usr_alice",
    "tenant_id": "TENANT_ACME",
    "ip": "10.240.12.8"
  },
  "action": "RETRIEVE_AND_SYNTHESIZE",
  "evaluated_docs": ["DOC-101", "DOC-102"],
  "decision": "GRANTED",
  "policy": "ABAC_POLICY_PROD_V4",
  "checksum": "e3b0c44298fc1c149afbf4c8996fb924..."
}
''', language="json")

with col_al2:
    st.markdown("### SIEM Streaming Architecture")
    dot_audit = """
    digraph AuditStream {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        RAGApp [label="RAG Query Execution Core", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        Kafka [label="Apache Kafka / AWS Kinesis\nAppend-Only Audit Stream", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        Splunk [label="Enterprise SIEM (Splunk / Datadog)\nImmutable WORM Storage Tier", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        RAGApp -> Kafka [label="Non-Blocking Async Event"];
        Kafka -> Splunk [label="Durable Ingestion"];
    }
    """
    st.graphviz_chart(dot_audit, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: SECURE DATA DELETION
# ==============================================================================
st.header("6. Secure Data Deletion (Cryptographic Erasure, Vector Purging & Right-to-be-Forgotten GDPR)", divider="rainbow")

st.markdown("""
Under **GDPR Article 17 (Right to be Forgotten)**, individuals have the legal right to demand the permanent erasure of their personal data.
In traditional relational databases, executing `DELETE FROM users WHERE id = X` immediately removes the record.
In high-dimensional vector databases, deleting records is far more complex:
* Deleting a vector node from an HNSW graph leaves 'tombstone' fragments and degrades graph connectivity.
* Complete physical erasure requires segment rebuilding and compaction.
""")

col_sdd1, col_sdd2 = st.columns([1, 1])

with col_sdd1:
    st.markdown("### Enterprise Deletion Paradigms:")
    st.markdown("""
    1. **Vector Tombstoning & Compaction**:
       The vector ID is marked with a tombstone bitmask, immediately excluding it from runtime queries.
       A background compaction worker rebuilds the HNSW segment to reclaim physical RAM.
    2. **Cryptographic Erasure (Crypto-Shredding)**:
       Each document or tenant is encrypted with a dedicated symmetric encryption key stored in a KMS.
       To execute irreversible deletion, simply destroy the encryption key in the KMS. The vector payload becomes permanently unrecoverable gibberish without needing complex database restructuring.
    """)

with col_sdd2:
    st.markdown("### Crypto-Shredding Lifecycle")
    dot_crypto = """
    digraph CryptoShred {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Doc [label="Encrypted Document & Vector Payload", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        KMS [label="AWS KMS / Vault Key\nKey ID: doc-key-8891", shape=component, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        DeleteReq [label="GDPR Erasure Request", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
        DestroyKey [label="Destroy KMS Key doc-key-8891\n(Immediate Crypto-Shredding)", shape=ellipse, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];

        Doc -> KMS [label="Decrypted at Runtime"];
        DeleteReq -> DestroyKey -> KMS;
    }
    """
    st.graphviz_chart(dot_crypto, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Tenant Enterprise Security & Audit Compliance Engine (EnterpriseComplianceEngine)", divider="rainbow")

st.markdown("""
Below is the verified production engine `EnterpriseComplianceEngine`.
It implements multi-tenant isolation routing, granular user/group ACL evaluations,
immutable compliance audit logging, and GDPR cryptographic erasure simulations.
""")

st.code('''
class EnterpriseComplianceEngine:
    """
    Enterprise Multi-Tenant Security, Granular ACL Authorization,
    and Immutable Audit Compliance Engine.
    """
    def __init__(self):
        self.doc_store: Dict[str, Dict[str, Any]] = {
            "DOC_T1_001": {
                "tenant_id": "TENANT_ACME",
                "title": "ACME Cloud Architecture Specification",
                "owner_user": "alice_acme",
                "allowed_groups": ["acme_engineers", "acme_admins"],
                "allowed_users": ["alice_acme", "bob_acme"],
                "content": "ACME Kubernetes clusters utilize Cilium eBPF for networking."
            },
            "DOC_T1_002": {
                "tenant_id": "TENANT_ACME",
                "title": "ACME Executive Payroll Ledger",
                "owner_user": "cfo_acme",
                "allowed_groups": ["acme_executives"],
                "allowed_users": ["cfo_acme"],
                "content": "Executive annual bonuses approved for fiscal year 2026."
            },
            "DOC_T2_001": {
                "tenant_id": "TENANT_STARK",
                "title": "Stark Industries Arc Reactor Blueprints",
                "owner_user": "tony_stark",
                "allowed_groups": ["stark_rd"],
                "allowed_users": ["tony_stark", "pepper_stark"],
                "content": "Clean energy palladium core replacement design specifications."
            }
        }
        self.audit_trail: List[Dict[str, Any]] = []
        self._event_counter = 1000

    def evaluate_document_access(
        self,
        requester_user: str,
        requester_tenant: str,
        requester_groups: List[str],
        doc_id: str
    ) -> Tuple[bool, str]:
        if doc_id not in self.doc_store:
            return False, "DENY: Document not found."

        doc = self.doc_store[doc_id]

        # 1. Hard Tenant Isolation
        if requester_tenant != doc["tenant_id"]:
            return False, f"DENY: Tenant Boundary Violation! '{requester_tenant}' cannot access '{doc['tenant_id']}'."

        # 2. Document ACL Check
        if requester_user in doc["allowed_users"]:
            return True, f"ALLOW: Direct user ACL match for '{requester_user}'."

        for grp in requester_groups:
            if grp in doc["allowed_groups"]:
                return True, f"ALLOW: Group ACL membership match for '{grp}'."

        return False, f"DENY: User '{requester_user}' lacks required permissions."

    def execute_rag_access(
        self,
        requester_user: str,
        requester_tenant: str,
        requester_groups: List[str],
        doc_id: str
    ) -> Dict[str, Any]:
        self._event_counter += 1
        allowed, reason = self.evaluate_document_access(
            requester_user, requester_tenant, requester_groups, doc_id
        )

        audit_entry = {
            "event_id": f"AUDIT-EVT-{self._event_counter}",
            "user_id": requester_user,
            "tenant_id": requester_tenant,
            "doc_id": doc_id,
            "decision": "GRANTED" if allowed else "BLOCKED",
            "policy_reason": reason
        }
        self.audit_trail.append(audit_entry)
        return {
            "doc_id": doc_id,
            "allowed": allowed,
            "reason": reason,
            "content": self.doc_store[doc_id]["content"] if allowed and doc_id in self.doc_store else None
        }
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
comp_engine = EnterpriseComplianceEngine()

# Test 1: Authorized intra-tenant access
auth_res = comp_engine.execute_rag_access("alice_acme", "TENANT_ACME", ["acme_engineers"], "DOC_T1_001")
assert auth_res["allowed"] is True, "Alice must be granted access to DOC_T1_001"
assert auth_res["content"] is not None, "Content must be returned upon authorization"

# Test 2: Cross-tenant isolation enforcement
leak_res = comp_engine.execute_rag_access("alice_acme", "TENANT_ACME", ["acme_engineers"], "DOC_T2_001")
assert leak_res["allowed"] is False, "Cross-tenant access must be strictly blocked"
assert "Tenant Boundary Violation" in leak_res["reason"], "Must report tenant violation"

# Test 3: GDPR Cryptographic Erasure
purge_res = comp_engine.execute_gdpr_deletion("DOC_T1_002", "TENANT_ACME")
assert purge_res["success"] is True, "Document purge must succeed"
assert "DOC_T1_002" not in comp_engine.doc_store, "Document must be purged from active store"

st.success("Automated Verification Passed: EnterpriseComplianceEngine successfully verified across tenant isolation, group ACLs, and GDPR erasure.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Multi-Tenant RBAC Access Matrix & Immutable Audit Trail Simulator", divider="rainbow")

st.markdown("""
Test multi-tenant access control and inspect real-time audit ledger generation.
Configure user credentials and target documents to evaluate policy decisions and GDPR erasure workflows.
""")

wb_ep1, wb_ep2 = st.columns([1, 1])

with wb_ep1:
    st.markdown("#### User Persona & Security Context")
    sim_tenant = st.selectbox("Requester Tenant", ["TENANT_ACME", "TENANT_STARK"])
    sim_user = st.selectbox("Requester Username", ["alice_acme", "bob_acme", "cfo_acme", "tony_stark", "external_intruder"])
    sim_groups_selected = st.multiselect(
        "Assigned Security Groups",
        ["acme_engineers", "acme_admins", "acme_executives", "stark_rd"],
        default=["acme_engineers"] if "acme" in sim_user else ["stark_rd"]
    )

with wb_ep2:
    st.markdown("#### Target Document Selection")
    sim_doc_target = st.selectbox(
        "Select Target Document to Query:",
        list(comp_engine.doc_store.keys()),
        format_func=lambda d: f"{d}: {comp_engine.doc_store[d]['title']} ({comp_engine.doc_store[d]['tenant_id']})" if d in comp_engine.doc_store else f"{d} (Purged)"
    )

if st.button("Execute Access & Log to Compliance Trail", key="btn_exec_access"):
    eval_out = comp_engine.execute_rag_access(
        requester_user=sim_user,
        requester_tenant=sim_tenant,
        requester_groups=sim_groups_selected,
        doc_id=sim_doc_target
    )
    if eval_out["allowed"]:
        st.success(f"Access GRANTED: {eval_out['reason']}")
        st.markdown(f"**Retrieved Content**: *\"{eval_out['content']}\"*")
    else:
        st.error(f"Access BLOCKED: {eval_out['reason']}")

st.markdown("### Immutable Compliance Audit Ledger (Real-Time)")
if comp_engine.audit_trail:
    st.dataframe(pd.DataFrame(comp_engine.audit_trail), use_container_width=True)
else:
    st.info("No audit events recorded yet.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What are the three primary multi-tenancy models for vector databases, and what are their trade-offs?**"):
    st.markdown("""
    **Answer**:
    1. **Siloed (Instance-per-Tenant)**: Each tenant has a completely separate vector database cluster or virtual machine.
       * *Pros*: Maximum cryptographic and physical hardware isolation. Zero risk of cross-tenant leakage.
       * *Cons*: Extremely expensive and operationally burdensome to manage thousands of clusters.
    2. **Partitioned (Namespace/Collection-per-Tenant)**: A shared vector cluster where each tenant has a dedicated collection or isolated namespace.
       * *Pros*: Excellent cost efficiency while preserving engine-level boundary separation.
       * *Cons*: Requires cluster-level quota management to prevent one tenant from consuming all cluster memory.
    3. **Shared (Metadata-Filtered)**: All tenants share the same collection, partitioned only by a `tenant_id` filter predicate.
       * *Pros*: Maximum resource pooling and simplest schema.
       * *Cons*: Highest risk of data leakage if an application bug omits the tenant filter predicate.
    """)

with st.expander("**Q2: How do you synchronize fast-changing Active Directory user permissions with static document vectors?**"):
    st.markdown("""
    **Answer**:
    Re-indexing millions of document vectors every time an employee changes security groups is computationally unfeasible.
    * **The Enterprise Pattern (Dual-Tier Decoupling)**:
      1. Documents are tagged only with **Security Group IDs (SIDs)** (e.g., `allowed_groups: ["S-1-5-21-397..."]`), which rarely change.
      2. At query runtime, the API Gateway queries the Active Directory cache (in Redis) to fetch the user's current group memberships.
      3. The Gateway constructs a vector filter querying for any document whose `allowed_groups` overlaps with the user's current group list.
      This allows instant permission revocation without modifying a single document in the vector database.
    """)

with st.expander("**Q3: What is Crypto-Shredding, and why is it preferred over physical vector deletion for GDPR compliance?**"):
    st.markdown("""
    **Answer**:
    In HNSW vector graphs, physically deleting a vector requires unlinking graph edges, marking tombstones, and executing expensive segment compactions that degrade query performance.
    * **Crypto-Shredding**:
      Every tenant (or individual document) is encrypted with a unique Data Encryption Key (DEK) managed in AWS KMS or HashiCorp Vault.
      When a GDPR Right-to-be-Forgotten request arrives, the application deletes the DEK from the KMS.
      Without the key, the vector and its document payload are mathematically impossible to decrypt, satisfying legal standards for permanent erasure in milliseconds with zero database compaction overhead.
    """)

with st.expander("**Q4: What data must an immutable RAG audit log contain to satisfy SOC 2 Type II compliance?**"):
    st.markdown("""
    **Answer**:
    To satisfy SOC 2 and ISO 27001 requirements, audit logs must be append-only and stream to Write-Once-Read-Many (WORM) storage.
    Each event record must include:
    1. **Cryptographic Timestamps** and unique Event IDs.
    2. **User Identity Context**: `user_id`, `tenant_id`, IP address, and authenticated role.
    3. **Exact Query Lineage**: The user's prompt text and the unique IDs of all retrieved chunks evaluated.
    4. **Policy Decision & Rationale**: Whether access was ALLOWED or BLOCKED, along with the policy rule ID.
    5. **Digital Signature / Checksum**: To guarantee tamper-evidence.
    """)

with st.expander("**Q5: How do you prevent a 'Confused Deputy' attack in multi-tenant RAG systems?**"):
    st.markdown("""
    **Answer**:
    A Confused Deputy attack occurs when an authenticated user tricks a privileged backend service into retrieving data on their behalf that they lack permission to view directly.
    * **Mitigation**:
      Never allow the backend RAG service to query the vector database using global superuser credentials.
      Enforce **Identity Propagation**: pass the user's scoped JWT or IAM context all the way to the vector database and storage layers.
      The vector database must evaluate permissions using the user's identity token, rejecting queries if the token lacks authorization.
    """)

st.markdown("### Production Traps & Failure Modes")

etr_col1, etr_col2 = st.columns(2)

with etr_col1:
    st.error("""
    **Trap 1: Software-Only Tenant Isolation in Multi-Tenant Collections**
    * **Failure**: Storing multiple corporate tenants in a single shared vector collection and relying purely on application code to append `where tenant_id = X`.
    * **Consequence**: A single developer omission or query parser bug omits the filter, exposing competitor confidential records in search results.
    * **Remedy**: Enforce hard partition separation (dedicated collections or cryptographically isolated namespaces) at the infrastructure layer.
    """)

    st.error("""
    **Trap 2: Storing Plain-Text Personally Identifiable Information in Audit Logs**
    * **Failure**: Logging full unredacted query prompts and retrieved context chunks directly into cloud monitoring logs (Datadog / CloudWatch).
    * **Consequence**: The audit logging system itself becomes a massive compliance violation under GDPR and HIPAA by centralizing unencrypted PII.
    * **Remedy**: Always pass logged text through automated PII redaction and DLP tokenizers before emitting events to the audit stream.
    """)

with etr_col2:
    st.warning("""
    **Trap 3: Re-Indexing Documents on Every User Permission Change**
    * **Failure**: Indexing individual user usernames directly into document vector metadata instead of Active Directory group IDs.
    * **Consequence**: When an employee changes teams, thousands of documents must be re-embedded or updated, overwhelming the vector database.
    * **Remedy**: Tag documents exclusively with static security group IDs, resolving user-to-group mappings dynamically at query time.
    """)

    st.warning("""
    **Trap 4: Neglecting Vector Index Tombstone Degradation**
    * **Failure**: Deleting thousands of vectors daily using standard soft-deletion without scheduling background index compaction.
    * **Consequence**: The HNSW navigation graph develops dead links, causing search recall to deteriorate from 95% down to 70%.
    * **Remedy**: Implement scheduled off-peak index compaction and segment rebuilding to purge deleted tombstones permanently.
    """)

st.markdown("### Core Technical Flashcards")

fce_c1, fce_c2 = st.columns(2)

with fce_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is Identity Propagation in an enterprise RAG pipeline?")
        st.markdown("**Back**: The architectural practice of passing the user's authenticated identity context (JWT claims, tenant ID, security groups) through all layers from API Gateway to Vector DB, ensuring downstream lookups execute with user-level privileges.")

    with st.container(border=True):
        st.markdown("**Front**: Why is tagging documents with Group SIDs superior to User IDs for ACLs?")
        st.markdown("**Back**: Group memberships change frequently in Active Directory, whereas document-to-group mappings remain static. Resolving user groups at query time avoids expensive vector re-indexing when employees change roles.")

with fce_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is Crypto-Shredding and how does it satisfy GDPR Right to Erasure?")
        st.markdown("**Back**: Deleting the dedicated cryptographic encryption key corresponding to a document or tenant in a KMS, rendering the vector and payload permanently unrecoverable without requiring physical index restructuring.")

    with st.container(border=True):
        st.markdown("**Front**: What is the Confused Deputy problem in RAG systems?")
        st.markdown("**Back**: When a backend service with global database privileges executes queries on behalf of an unprivileged user, accidentally returning restricted records because the user's permissions were not forwarded to the database.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Enterprise RAG Security & Multi-Tenancy principles:")

e_chk1 = st.checkbox("I can differentiate between Siloed, Partitioned, and Shared multi-tenancy models for vector databases.", value=True)
e_chk2 = st.checkbox("I understand Identity Propagation and how to forward JWT claims to downstream retrieval microservices.", value=True)
e_chk3 = st.checkbox("I know how to decouple Active Directory group permission changes from static document vector indices.", value=True)
e_chk4 = st.checkbox("I understand the mechanics of Crypto-Shredding for instantaneous GDPR Right-to-be-Forgotten compliance.", value=True)
e_chk5 = st.checkbox("I can design an immutable, tamper-evident audit logging stream satisfying SOC 2 Type II compliance.", value=True)
e_chk6 = st.checkbox("I know how to prevent the Confused Deputy vulnerability by enforcing user-scoped retrieval permissions.", value=True)

if e_chk1 and e_chk2 and e_chk3 and e_chk4 and e_chk5 and e_chk6:
    st.success("Mastery Certified: You have mastered enterprise RAG multi-tenancy, granular ACL synchronization, compliance audit logging, and secure data deletion.")
