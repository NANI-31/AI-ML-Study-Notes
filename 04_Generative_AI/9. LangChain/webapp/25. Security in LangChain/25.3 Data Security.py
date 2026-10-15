import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="25.3 Data Security - LangChain Security Architecture",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for data privacy telemetry
st.markdown("""
<style>
    .pii-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-redacted {
        background-color: #78350F;
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-clean {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class PIISanitizerSimulator:
    """
    Lightweight PII detection and reversible pseudonymization engine
    demonstrating enterprise compliance before dispatching to external LLMs.
    """
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
        "PHONE": r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
    }

    def __init__(self):
        self.pseudonym_map = {}
        self.reverse_map = {}

    def mask_pii(self, text: str) -> Dict[str, Any]:
        masked_text = text
        detected_entities = []

        for entity_type, pat in self.PATTERNS.items():
            matches = list(re.finditer(pat, masked_text))
            for idx, m in enumerate(matches):
                original_val = m.group(0)
                pseudo_token = f"[{entity_type}_{idx+1}]"
                self.pseudonym_map[pseudo_token] = original_val
                self.reverse_map[original_val] = pseudo_token
                detected_entities.append({"type": entity_type, "original": original_val, "token": pseudo_token})
                masked_text = masked_text.replace(original_val, pseudo_token)

        return {
            "masked_text": masked_text,
            "detected_count": len(detected_entities),
            "entities": detected_entities
        }

    def unmask_pii(self, generated_text: str) -> str:
        restored = generated_text
        for token, original_val in self.pseudonym_map.items():
            restored = restored.replace(token, original_val)
        return restored


# --- LESSON HEADER ---
st.title("25.3 Data Security, PII Redaction & Multi-Tenant Isolation")
st.caption("Chapter 25: Security in LangChain | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Deploying enterprise GenAI applications requires rigorous data governance. Sensitive customer data,
personally identifiable information (PII), and proprietary corporate secrets must never leak into training datasets,
unauthorized vector indexes, or external third-party API logs.

In this lesson, you will master:
1. Identifying PII (Personally Identifiable Information) using heuristic patterns and NER models.
2. Implementing reversible PII Redaction and surrogate token anonymization.
3. Integrating enterprise Secrets Management to prevent credential leakages.
4. Securing API Keys with ephemeral tokens, rotation, and server-side isolation.
5. Partitioning Multi-Tenant Vector Stores with strict tenant ID metadata filters.
6. Enforcing Document-Level Access Control (ABAC/RBAC) across RAG retrievers.
7. Establishing Zero-Data Retention (ZDR) and ephemeral cache lifecycle policies.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Enterprise Data Protection Gateway", divider="rainbow")
st.markdown("""
A production-grade LangChain privacy gateway intercepts all ingress prompts, replaces sensitive entities with
surrogate tokens, enforces multi-tenant retriever filters, and unmasks surrogate tokens only on sanitized client egress.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    data_sec_dot = """
    digraph DataSecurityBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_ingress_data {
            label="1. Ingress & PII Anonymization";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            raw_input [label="User Input\\n'Transfer $500 to alice@work.com, SSN: 123-45-6789'", fillcolor="#7F1D1D", fontcolor="#FCA5A5", color="#EF4444"];
            presidio_engine [label="PII Scrubber (Presidio / Regex)\\nDetects SSN, Email, CC", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            token_vault [label="In-Memory Ephemeral Vault\\nMaps [EMAIL_1] -> alice@work.com", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_retrieval_sec {
            label="2. Multi-Tenant Vector Retrieval";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            tenant_filter [label="Tenant Metadata Filter\\nfilter={'tenant_id': 'acme_corp'}", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            doc_rbac [label="Document ABAC Filter\\nfilter={'clearance_level': 'confidential'}", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_external_llm {
            label="3. External Cloud LLM (Zero-Data Retention)";
            bgcolor="#1E293B";
            fontcolor="#A855F7";
            color="#334155";
            sanitized_prompt [label="Anonymized Prompt\\n'Transfer $500 to [EMAIL_1], SSN: [SSN_1]'", fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            llm_inference [label="Model Inference (ZDR Agreement)\\nNo customer training, No persistence", fillcolor="#4C1D95", fontcolor="#EDE9FE", color="#A855F7"];
        }

        subgraph cluster_egress_restore {
            label="4. Egress De-Anonymization";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            de_anonymizer [label="Surrogate Restorer\\nReplaces [EMAIL_1] with alice@work.com", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            clean_client [label="Client Response (Encrypted TLS)\\nDelivered directly to user session", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        raw_input -> presidio_engine [label="Ingress Text"];
        presidio_engine -> token_vault [label="Stores Mappings"];
        presidio_engine -> sanitized_prompt [label="Masked String"];
        tenant_filter -> doc_rbac [label="Applies Scope"];
        doc_rbac -> sanitized_prompt [label="Filtered Chunks"];
        sanitized_prompt -> llm_inference [label="Zero PII Exposed"];
        llm_inference -> de_anonymizer [label="Model Generation"];
        token_vault -> de_anonymizer [label="Reverses Tokens"];
        de_anonymizer -> clean_client [label="Final Output"];
    }
    """
    st.graphviz_chart(data_sec_dot, use_container_width=True)

# --- SECTION 2: PII DETECTION ---
st.header("2. PII Detection", divider="rainbow")
st.markdown("""
Personally Identifiable Information (PII) includes direct identifiers (Social Security Numbers, Passport numbers, full names)
and quasi-identifiers (IP addresses, zip codes, birth dates) that can uniquely identify an individual.
""")

col_pii1, col_pii2 = st.columns(2)
with col_pii1:
    st.markdown("""
    #### Detection Methodologies
    - **Pattern-Based Regex Matching**: Fast, deterministic detection for structured entities with standard checksum algorithms (Credit Cards via Luhn algorithm, US SSNs, emails, phone numbers).
    - **Named Entity Recognition (NER)**: Machine learning models (e.g. spaCy, RoBERTa) trained to identify unstructured entities like personal names, organizations, and medical conditions in free-form prose.
    - **Contextual Heuristics**: Multi-word scanning that increases entity confidence based on neighboring tokens (e.g. "routing number" appearing within 3 tokens of a 9-digit string).
    """)

with col_pii2:
    with st.container(border=True):
        st.markdown("🔍 **Enterprise PII Scanning with Presidio**")
        st.code("""
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
results = analyzer.analyze(
    text="Contact Dr. Sarah Chen at sarah@health.org",
    entities=["PERSON", "EMAIL_ADDRESS"],
    language="en"
)
# Returns entity bounding boxes and confidence scores
        """, language="python")

# --- SECTION 3: PII REDACTION ---
st.header("3. PII Redaction", divider="rainbow")
st.markdown("""
Once PII is detected, the application must neutralize it before dispatching prompts to external LLMs.
Two primary techniques exist: **Permanent Anonymization** and **Reversible Pseudonymization**.
""")

st.subheader("Comparison: Permanent Anonymization vs Reversible Pseudonymization")
redaction_df = pd.DataFrame({
    "Approach": ["Permanent Anonymization (Redaction)", "Reversible Pseudonymization (Surrogate Tokens)"],
    "Mechanism": ["Replaces PII with static placeholders: `<REDACTED>`", "Replaces PII with keyed surrogate tokens: `[PERSON_1]`, `[EMAIL_1]`"],
    "Downstream Usability": ["Loss of contextual coherence; LLM cannot refer to distinct people", "Preserves relationship structure; LLM can reason about `[PERSON_1]` vs `[PERSON_2]`"],
    "Egress Recovery": ["Irreversible; original entity lost forever", "Reversible; gateway replaces surrogate with original data for end-user"],
    "Recommended For": ["Long-term log storage, training fine-tuning sets", "Interactive enterprise agents and customer support workflows"]
})
st.dataframe(redaction_df, use_container_width=True)

# --- SECTION 4: SECRETS MANAGEMENT ---
st.header("4. Secrets Management", divider="rainbow")
st.markdown("""
LangChain pipelines interface with numerous external services: OpenAI, Anthropic, Pinecone, databases, and enterprise APIs.
Hardcoding API keys or database connection strings inside source code, Jupyter notebooks, or Git repositories is an extreme security violation.
""")

col_sec1, col_sec2 = st.columns(2)
with col_sec1:
    st.markdown("""
    #### Modern Secrets Architecture
    1. **Centralized Vaults**: Retrieve credentials dynamically at runtime from AWS Secrets Manager, HashiCorp Vault, or Azure Key Vault.
    2. **Short-Lived Ephemeral Credentials**: Use cloud IAM roles (AWS IRSA, GCP Workload Identity) rather than long-lived static API secrets.
    3. **Zero Context Leakage**: Ensure API keys are never interpolated into prompt strings or tool parameter descriptions where they can be reflected into telemetry traces.
    """)

with col_sec2:
    with st.container(border=True):
        st.markdown("🛡️ **Runtime Secret Fetch Pattern**")
        st.code("""
import boto3
import json

def get_secret(secret_name: str) -> str:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])["api_key"]

# Instantiated at runtime via IAM role
api_key = get_secret("prod/langchain/openai_key")
        """, language="python")

# --- SECTION 5: API KEY SECURITY ---
st.header("5. API Key Security", divider="rainbow")
st.markdown("""
API keys represent the financial identity and access privileges of your organization.
Leaked keys lead to devastating API bills (denial-of-wallet) and unauthorized access to proprietary fine-tuned model checkpoints.
""")

st.markdown("""
#### Five Mandatory API Key Security Rules:
- **Rule 1 (Reverse Proxy Gateway)**: Client frontends must never hold model API keys. All calls must route through your backend FastAPI/LangChain proxy.
- **Rule 2 (Automated Secret Rotation)**: Rotate production API keys automatically every 30 to 90 days.
- **Rule 3 (Budget Ceilings & Alerts)**: Configure hard monthly spend limits on provider dashboards (OpenAI/Anthropic) to halt billing automatically upon threshold breach.
- **Rule 4 (Scoped Keys)**: If a provider supports granular key scoping, generate keys restricted strictly to `model:read` and `inference:execute` without administrative account permissions.
- **Rule 5 (Git Pre-Commit Scanners)**: Enforce tools like `trufflehog` and `gitleaks` in CI/CD pipelines to block commits containing key signatures.
""")

# --- SECTION 6: USER ISOLATION ---
st.header("6. User Isolation", divider="rainbow")
st.markdown("""
In SaaS and multi-tenant systems, User A must never retrieve data belonging to User B.
In vector search systems, multi-tenancy cannot rely on semantic similarity alone; cosine distance does not understand corporate ownership boundaries.
""")

isolation_dot = """
digraph MultiTenantIsolation {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    query [label="User Query\\n'Show quarterly revenue'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    auth_ctx [label="Authenticated Session\\nTenant: Tenant_A (Acme)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    v_db [label="Shared Vector Database\\nContains Docs from Tenant A and B", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    partition_a [label="Partition A Results\\nTenant_A records ONLY", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    partition_b [label="Partition B Records\\nBLOCKED BY METADATA FILTER", fillcolor="#7F1D1D", fontcolor="#FCA5A5", color="#EF4444"];

    query -> v_db [label="Vector Search"];
    auth_ctx -> v_db [label="Enforces filter={'tenant_id': 'Tenant_A'}"];
    v_db -> partition_a [label="Returned"];
    v_db -> partition_b [label="Unauthorized / Filtered Out"];
}
"""
st.graphviz_chart(isolation_dot, use_container_width=True)

st.markdown("""
#### Vector Database Partitioning Architectures:
1. **Namespace Isolation**: Supported natively by Pinecone and Milvus. Each tenant receives a completely segregated vector namespace.
2. **Metadata Filtering**: In Chroma and Qdrant, every inserted chunk carries metadata `{"tenant_id": "org_123"}`. The LangChain retriever MUST hardcode this filter on every query.
3. **Database-Per-Tenant**: Separate vector database instances for high-security enterprise tiers (banking, defense).
""")

# --- SECTION 7: ACCESS CONTROL ---
st.header("7. Access Control", divider="rainbow")
st.markdown("""
Beyond organization-level multi-tenancy, Document-Level Access Control (Attribute-Based Access Control, or ABAC)
restricts which documents within an organization an individual user can view.
""")

col_ac1, col_ac2 = st.columns(2)
with col_ac1:
    st.markdown("""
    #### Document-Level Metadata Enforcement
    When indexing documents into your RAG pipeline, store the document's access control list (ACL) directly in vector metadata:
    ```json
    {
      "text": "Executive compensation memo...",
      "metadata": {
        "tenant_id": "acme_corp",
        "department": "HR",
        "clearance": "executive",
        "allowed_user_ids": ["usr_001", "usr_009"]
      }
    }
    ```
    """)

with col_ac2:
    st.markdown("""
    #### Retriever-Level Enforcement in LCEL
    When a user queries the retriever, their session permissions are automatically injected as a filter constraint:
    ```python
    def get_secure_retriever(user_session: dict):
        return vectorstore.as_retriever(
            search_kwargs={
                "filter": {
                    "$and": [
                        {"tenant_id": user_session["tenant_id"]},
                        {"department": user_session["dept"]}
                    ]
                }
            }
        )
    ```
    """)

# --- SECTION 8: DATA RETENTION ---
st.header("8. Data Retention", divider="rainbow")
st.markdown("""
Under regulations like GDPR, CCPA, and HIPAA, organizations must enforce strict data lifecycle and retention policies.
Data ingested by GenAI pipelines must not persist indefinitely in third-party model caches.
""")

st.subheader("Key Pillars of LLM Data Retention Governance")
r_col1, r_col2, r_col3 = st.columns(3)
with r_col1:
    with st.container(border=True):
        st.markdown("📜 **Zero-Data Retention (ZDR)**")
        st.markdown("""
        Enterprise agreements with OpenAI, Microsoft Azure, and Anthropic ensuring prompts and completions are
        **never persisted to disk** or used to train public foundation models.
        """)

with r_col2:
    with st.container(border=True):
        st.markdown("⏳ **Cache TTL & Purge**")
        st.markdown("""
        LangChain semantic caches (e.g. Redis SemanticCache) must carry explicit **Time-To-Live (TTL)** limits
        (e.g. 24 hours) to prevent stale PII accumulation.
        """)

with r_col3:
    with st.container(border=True):
        st.markdown("🧹 **Ephemeral Scratchpads**")
        st.markdown("""
        Agent intermediate scratchpads, reasoning traces, and temporary document files must be purged immediately
        upon task termination.
        """)

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise Privacy Gateway with Pseudonymization", divider="rainbow")
st.markdown("""
Below is a production-grade LangChain pipeline demonstrating pre-execution PII pseudonymization,
tenant-scoped vector retrieval, and post-execution entity restoration.
""")

prod_data_code = r'''from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class EnterprisePrivacyGateway:
    def __init__(self):
        self.patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "SSN": r"\\b\\d{3}-\\d{2}-\\d{4}\\b"
        }

    def anonymize_ingress(self, input_dict: dict) -> dict:
        text = input_dict["query"]
        vault = {}
        anonymized = text
        
        for p_name, pat in self.patterns.items():
            matches = list(re.finditer(pat, anonymized))
            for i, m in enumerate(matches):
                orig = m.group(0)
                token = f"[{p_name}_{i+1}]"
                vault[token] = orig
                anonymized = anonymized.replace(orig, token)

        return {
            "anonymized_query": anonymized,
            "vault": vault,
            "tenant_id": input_dict["tenant_id"]
        }

    def restore_egress(self, input_payload: dict) -> str:
        text = input_payload["llm_output"]
        vault = input_payload["vault"]
        restored = text
        for token, orig in vault.items():
            restored = restored.replace(token, orig)
        return restored

gateway = EnterprisePrivacyGateway()

# Step 1: Pre-execution Anonymization
# Step 2: Prompt execution with Zero PII
# Step 3: De-anonymize response for authenticated client
privacy_chain = (
    RunnableLambda(gateway.anonymize_ingress)
    | {
        "llm_output": (
            ChatPromptTemplate.from_template("Answer query securely: {anonymized_query}")
            # | chat_model (Runs without exposing customer PII)
            | StrOutputParser()
        ),
        "vault": lambda x: x["vault"]
    }
    | RunnableLambda(gateway.restore_egress)
)
'''
st.code(prod_data_code, language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Live PII Redaction & Multi-Tenant Isolation Simulator", divider="rainbow")
st.markdown("""
Test real-time PII detection, reversible pseudonymization, and tenant metadata gating.
Enter sensitive data to inspect the exact anonymized prompt transmitted to the external model.
""")

sanitizer = PIISanitizerSimulator()

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("📝 Input Data & Tenant Configuration")
    sample_texts = [
        "Customer John Doe (SSN: 345-12-9876) requested an update regarding order 9921.",
        "Please send password recovery instructions to alice.smith@acme-corp.com or call 415-555-2671.",
        "Credit card payment failed for card 4532-8821-9943-1209. Please notify finance@example.com."
    ]
    selected_sample = st.selectbox("Choose a sample input containing PII:", sample_texts)
    input_text = st.text_area("Or type custom text with PII:", value=selected_sample, height=100)
    tenant_choice = st.selectbox("Active Tenant Session:", ["tenant_acme_corp", "tenant_globex_inc", "tenant_stark_industries"])
    run_scrub_btn = st.button("🔒 Run Privacy Gateway Scrub & Isolation", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("🛡️ Privacy Gateway Telemetry")
    if run_scrub_btn:
        res = sanitizer.mask_pii(input_text)
        
        st.markdown(
            f"<div class='pii-card'><span class='badge-redacted'>PII DETECTED & MASKED</span>"
            f"<br><br><b>Entities Identified:</b> {res['detected_count']}"
            f"<br><b>Tenant Metadata Lock:</b> <code>tenant_id == '{tenant_choice}'</code></div>",
            unsafe_allow_html=True
        )

        st.markdown("##### 📤 Anonymized Prompt (Dispatched to Model)")
        st.code(res["masked_text"], language="text")

        st.markdown("##### 🔐 In-Memory Pseudonym Vault (Never leaves server)")
        if res["entities"]:
            vault_df = pd.DataFrame(res["entities"])
            st.dataframe(vault_df[["type", "token", "original"]], use_container_width=True)

        st.markdown("##### 📥 Reconstructed Client Egress (Restored via Vault)")
        simulated_llm_reply = f"Acknowledged request for {res['masked_text']}. Status updated."
        restored_output = sanitizer.unmask_pii(simulated_llm_reply)
        st.success(f"**Client Response:** {restored_output}")
    else:
        st.info("Click 'Run Privacy Gateway Scrub & Isolation' to simulate PII anonymization.")

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is vector similarity search alone insufficient for multi-tenant enterprise RAG?"):
    st.markdown("""
    **Answer**: Vector similarity search identifies documents based solely on cosine similarity in high-dimensional embedding space.
    If Tenant A and Tenant B both possess documents discussing 'quarterly earnings', a semantic search query will retrieve chunks
    from both tenants if they are stored in the same vector index. Multi-tenant security mandates hard partition isolation: either
    physically segregated vector collections/namespaces or mandatory, non-bypassable metadata filters (`tenant_id == user_tenant_id`)
    applied at retrieval time.
    """)

with st.expander("Q2: What is the difference between static PII redaction and reversible pseudonymization?"):
    st.markdown("""
    **Answer**: Static redaction replaces sensitive entities with generic markers like `<REDACTED>`. While secure, this destroys
    the semantic relationships between entities (an LLM cannot determine if `<REDACTED>` in sentence 1 is the same person as `<REDACTED>`
    in sentence 2). Reversible pseudonymization replaces entities with keyed surrogate tokens (`[PERSON_1]`, `[EMAIL_1]`) stored in a secure
    server-side lookup vault. The LLM can reason about distinct entities, and the gateway restores original names upon response delivery.
    """)

with st.expander("Q3: How do enterprise Zero-Data Retention (ZDR) agreements protect corporate IP?"):
    st.markdown("""
    **Answer**: Standard public consumer APIs often retain conversation logs for up to 30 days for abuse monitoring and potential model training.
    Enterprise ZDR agreements with providers (such as OpenAI, Anthropic, AWS Bedrock, and Azure OpenAI) legally and technically guarantee that
    prompts and completions are processed entirely in ephemeral RAM, never persisted to disk logs, and never used to train or improve foundation models.
    """)

with st.expander("Q4: How should API keys and secrets be handled in production LangChain applications?"):
    st.markdown("""
    **Answer**: API keys must never be stored in code repositories, client-side applications, or hardcoded in configuration files.
    Production systems retrieve keys dynamically at runtime from centralized secret stores (AWS Secrets Manager, HashiCorp Vault)
    using cloud IAM role authentication (e.g. AWS IRSA or GCP Workload Identity). Keys should be scoped to minimal permissions and rotated automatically.
    """)

with st.expander("Q5: What is Document-Level Access Control (ABAC) in RAG pipelines?"):
    st.markdown("""
    **Answer**: ABAC restricts document access based on attributes assigned to both the user and the document. When indexing documents,
    metadata tags such as `department: 'legal'`, `security_clearance: 'confidential'`, and `region: 'EMEA'` are embedded. When a user queries
    the system, their authenticated claims are translated into boolean metadata filter expressions in the vector retriever, ensuring the
    model only retrieves documents the user has explicit permission to view.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Client-Side Vector Filtering)**: Performing vector search across all tenants and filtering by tenant_id in Python code after retrieval allows sensitive competitor chunks into server memory and wastes retrieval top-k slots.")
    st.warning("**Trap 2 (Logging Unmasked Prompts in Telemetry)**: Anonymizing prompts before sending them to the LLM, but logging the raw, unmasked user query in debug application logs (Datadog, LangSmith) violates PII compliance.")
with col_trap2:
    st.error("**Trap 3 (Leaking PII in Embeddings)**: Storing unredacted sensitive names in vector database metadata fields exposes PII to anyone with vector index read access, even if the main database is secured.")
    st.warning("**Trap 4 (Indefinite Semantic Caching)**: Caching LLM responses without TTL expiration creates a persistent, unmonitored store of customer queries that violates GDPR 'Right to be Forgotten' mandates.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is Reversible Pseudonymization?")
        st.caption("Flip for answer:")
        st.info("**Back**: Replacing PII with surrogate tokens (`[PERSON_1]`) before calling an external model, then swapping original values back into the response before returning it to the user.")
    with st.container(border=True):
        st.markdown("**Front**: Why is metadata filtering mandatory in multi-tenant RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: Because vector similarity is purely semantic and will retrieve chunks across different tenant boundaries unless hard metadata constraints are enforced.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is Zero-Data Retention (ZDR)?")
        st.caption("Flip for answer:")
        st.info("**Back**: An enterprise SLA guaranteeing that an LLM provider will not log, persist to disk, or train on submitted prompts and completions.")
    with st.container(border=True):
        st.markdown("**Front**: What is ABAC in the context of RAG retrievers?")
        st.caption("Flip for answer:")
        st.info("**Back**: Attribute-Based Access Control: filtering retrieved chunks by matching document metadata attributes (e.g. clearance, department) against the user's session claims.")

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the distinction between static PII redaction and reversible pseudonymization.", value=True)
c2 = st.checkbox("I know how to implement multi-tenant vector isolation using metadata filters.", value=True)
c3 = st.checkbox("I understand how secrets management vaults protect LLM API keys.", value=True)
c4 = st.checkbox("I know how to enforce Document-Level Access Control (ABAC) in retrievers.", value=True)
c5 = st.checkbox("I understand the compliance significance of Zero-Data Retention agreements.", value=True)
c6 = st.checkbox("I know how to prevent PII leakage into telemetry and caching layers.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 25: Security in LangChain.")
