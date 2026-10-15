import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="30.2 Data Privacy - LangSmith Master Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class DataPrivacyEngine:
    """
    Production Data Privacy Governance & Telemetry Anonymization Engine.
    Emulates enterprise privacy compliance before sending traces to LangSmith:
    1. Automated PII Entity Masking (Presidio / Regex: SSN, Credit Cards, Emails, Phone Numbers)
    2. Sensitive Corporate Prompt Redaction (Protecting confidential instructions and IP)
    3. User Data Isolation & Cryptographic Pseudonymization (Hashing tenant/user IDs)
    4. Production Trace Privacy (Selective tracing: masking inputs/outputs, preserving telemetry)
    5. Data Retention Policies (GDPR / CCPA Time-to-Live [TTL] automatic trace purging)
    """
    def __init__(self, project_name: str = "enterprise-privacy-governance"):
        self.project_name = project_name
        self.sanitized_records: List[Dict[str, Any]] = []

    def sanitize_trace_payload(
        self,
        raw_prompt: str,
        mask_pii: bool = True,
        mask_inputs_completely: bool = False,
        tenant_id: str = "enterprise_cust_44"
    ) -> Dict[str, Any]:
        """
        Processes a raw input string, applies privacy masks, and returns an anonymized trace payload.
        """
        sanitized_text = raw_prompt
        entities_detected = []

        if mask_inputs_completely:
            sanitized_text = "[INPUT_REDACTED_BY_ENTERPRISE_POLICY]"
            entities_detected.append("CompleteInputMask")
        elif mask_pii:
            # Emulated Presidio entity detection and replacement
            patterns = [
                ("SSN", "000-12-3456", "[REDACTED_SSN]"),
                ("CreditCard", "4111-2222-3333-4444", "[REDACTED_CREDIT_CARD]"),
                ("Email", "john.doe@company.com", "[REDACTED_EMAIL]"),
                ("Phone", "555-839-2019", "[REDACTED_PHONE]")
            ]

            for label, sample_token, mask_str in patterns:
                if sample_token in sanitized_text:
                    sanitized_text = sanitized_text.replace(sample_token, mask_str)
                    entities_detected.append(label)

        # Cryptographically pseudonymize tenant and user IDs
        hashed_tenant = f"pseudonym_{hash(tenant_id) % 100000}"

        record = {
            "record_id": f"priv_rec_{len(self.sanitized_records) + 1}",
            "original_length": len(raw_prompt),
            "sanitized_payload": sanitized_text,
            "pseudonymized_tenant": hashed_tenant,
            "detected_pii_entities": entities_detected,
            "is_compliant": len(entities_detected) > 0 or not mask_pii,
            "retention_policy_days": 30
        }

        self.sanitized_records.append(record)
        return record


# Instantiate engine
privacy_engine = DataPrivacyEngine()

# --- HEADER SECTION ---
st.title("🛡️ Chapter 30.2: Data Privacy & Trace Anonymization")
st.markdown("""
Welcome to the production guide on **Data Privacy and Regulatory Compliance** in **LangSmith**.
While observability is critical for monitoring LLM reliability, unmanaged telemetry can easily violate global privacy laws
including **GDPR**, **CCPA**, **HIPAA**, and corporate confidentiality mandates.
Production traces frequently capture customer conversations containing Personally Identifiable Information (PII),
financial account numbers, medical disclosures, and proprietary business secrets.
This module establishes strict engineering protocols to sanitize, mask, and pseudonymize telemetry
before it is persisted in LangSmith.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Data Privacy Governance & Telemetry Anonymization Architecture", divider="rainbow")

st.markdown("""
The architecture below outlines the privacy boundary between user ingress and the LangSmith cloud.
Before any trace is dispatched over the network, a privacy interceptor inspects the payload:
running automated Presidio PII entity recognizers, pseudonymizing user IDs, applying Time-To-Live (TTL) retention tags,
and optionally redacting input/output text bodies entirely while preserving operational latency telemetry.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DataPrivacy {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. User Ingress & Raw Interaction";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            raw_req [label="Raw User Request\n'My SSN is 000-12-3456 and email is john@corp.com'\nUser ID: 'usr_jane_doe'", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_privacy_gate {
            label="2. Pre-Tracing Privacy & Anonymization Interceptor";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ner_scrubber [label="Microsoft Presidio / NER Scrubber\nDetects SSN, Credit Cards, Medical IDs\nReplaces with [REDACTED_TYPE] tokens", fillcolor="#0F172A", color="#F59E0B"];
            hasher [label="Cryptographic Pseudonymizer\nhash(user_id) -> 'pseudo_84920'\nTenant Scoped Salt", fillcolor="#0F172A", color="#38BDF8"];
            selective_mask [label="Selective Masking Policy\nOption: Mask raw inputs/outputs completely\nPreserve latency, status, token count only", fillcolor="#0F172A", color="#A855F7"];
        }

        subgraph cluster_langsmith {
            label="3. Sanitized LangSmith Observability Platform";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            clean_trace [label="Compliant Trace Record\nInputs: 'My SSN is [REDACTED_SSN]...'\nMetadata: {tenant_pseudo: 'pseudo_84920'}", fillcolor="#064E3B", color="#10B981"];
            ttl_policy [label="Automated Data Retention Engine\nTTL Policy: 30-Day Auto-Purge\nGDPR Right-to-Erasure API", fillcolor="#0F172A", color="#10B981"];
        }

        raw_req -> ner_scrubber [label="Intercept Execution"];
        raw_req -> hasher [label="Extract User Identity"];
        ner_scrubber -> selective_mask;
        hasher -> selective_mask;

        selective_mask -> clean_trace [label="Send via TLS 1.3"];
        clean_trace -> ttl_policy [label="Enforce Expiration"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of Data Privacy in LLM Observability
The following table details the compliance standards and technical implementations for each syllabus topic:
""")

privacy_matrix_df = pd.DataFrame([
    {
        "Pillar": "1. Sensitive Prompts",
        "Regulatory Concern": "Confidential trade secrets, corporate financial disclosures, proprietary system instructions",
        "Technical Remediation": "Selective trace masking, hiding system prompt instructions while logging tool operations",
        "Compliance Impact": "SOC-2 Type II Confidentiality Criterion"
    },
    {
        "Pillar": "2. Personally Identifiable Information",
        "Regulatory Concern": "Names, emails, SSNs, phone numbers, IP addresses, credit card numbers",
        "Technical Remediation": "Automated NER entity recognition (Microsoft Presidio) and regex token replacement",
        "Compliance Impact": "GDPR Article 5, CCPA / CPRA Protection"
    },
    {
        "Pillar": "3. User Data",
        "Regulatory Concern": "Associating raw corporate employee or customer identities with chat logs",
        "Technical Remediation": "Cryptographic one-way hashing with per-tenant secret salts (Pseudonymization)",
        "Compliance Impact": "GDPR Pseudonymization Exemption"
    },
    {
        "Pillar": "4. Production Trace Privacy",
        "Regulatory Concern": "Storing sensitive user query text in cloud-hosted third-party observability databases",
        "Technical Remediation": "Custom trace filter hooks that scrub inputs and outputs, sending only operational spans",
        "Compliance Impact": "HIPAA Privacy Rule / Zero Data Retention Mandates"
    },
    {
        "Pillar": "5. Data Retention",
        "Regulatory Concern": "Indefinite storage of historical conversation traces",
        "Technical Remediation": "Automated 30/90-day Time-To-Live (TTL) trace deletion policies & programmatic erasure APIs",
        "Compliance Impact": "GDPR Article 17 (Right to be Forgotten)"
    }
])
st.dataframe(privacy_matrix_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: SENSITIVE PROMPTS
# ==============================================================================
st.header("2. Sensitive Prompts (Corporate IP Protection, Confidential System Instructions & Redaction)", divider="rainbow")

st.markdown("""
Enterprise system prompts frequently contain proprietary IP:
confidential negotiation guidelines, proprietary business heuristics, internal API route documentation,
and trade-secret algorithms.
Logging these system prompts in plaintext across thousands of traces creates significant IP leakage risks.
""")

st.markdown("### Protecting Sensitive Prompts in LangSmith")
st.markdown("""
* **Selective Span Redaction**: Configuring tracing to log the model call and latency metrics while replacing the `system_prompt` input payload with a static reference: `[SYSTEM_PROMPT_VERSION_3.2_PROTECTED]`.
* **Zero-Knowledge Architecture**: If prompt text contains sensitive compliance data, teams configure custom callbacks that strip the system message before emitting the run payload to LangSmith.
""")

st.code('''
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import SystemMessage, HumanMessage

class PrivacyPreservingTracer(LangChainTracer):
    """
    Custom LangChain tracer that masks proprietary system prompts before serialization.
    """
    def _persist_run(self, run):
        # Redact system messages from inputs dictionary
        if "messages" in run.inputs:
            sanitized = []
            for msg in run.inputs["messages"]:
                if isinstance(msg, SystemMessage) or msg.get("type") == "system":
                    sanitized.append({"type": "system", "content": "[CONFIDENTIAL_SYSTEM_PROMPT_V4]"})
                else:
                    sanitized.append(msg)
            run.inputs["messages"] = sanitized
            
        super()._persist_run(run)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise environments:
* Prompts are managed via the LangSmith Prompt Hub with strict RBAC access.
* Traces reference the prompt by Hub ID (`hub://org/sales-prompt:v2`) rather than embedding the entire prompt string in every trace.
""")

st.divider()

# ==============================================================================
# SECTION 3: PERSONALLY IDENTIFIABLE INFORMATION
# ==============================================================================
st.header("3. Personally Identifiable Information (Automated PII Scrubbing, Presidio NER & Regex Masking)", divider="rainbow")

st.markdown("""
Personally Identifiable Information (PII) must never be transmitted or stored in unencrypted third-party logs.
If a customer provides their Social Security Number or credit card number during a support chat,
persisting that number in LangSmith traces constitutes a serious regulatory violation.
""")

st.markdown("### Presidio Integration Pattern")
st.markdown("""
1. **Entity Detection**: Using Microsoft Presidio (or regex engines) to scan text for named entities: `US_SSN`, `CREDIT_CARD`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `IBAN_CODE`.
2. **Deterministic Token Replacement**: Replacing the sensitive text with typed placeholder tokens (`[REDACTED_SSN]`, `[REDACTED_EMAIL]`).
3. **Pre-Inference Execution**: Scrubbing happens *before* the prompt reaches the foundation model or the LangSmith callback handler.
""")

st.code('''
from langchain_core.runnables import RunnableLambda

def presidio_pii_anonymizer(data: dict) -> dict:
    """
    Middleware runnable that scrubs PII before model inference and trace emission.
    """
    raw_query = data["query"]
    # Simulated Presidio entity masking
    clean_query = scrub_pii_entities(raw_query) # e.g. regex/NER pipeline
    return {"query": clean_query}

# Injected at the ingress boundary of every LCEL chain
secure_chain = RunnableLambda(presidio_pii_anonymizer) | prompt | model | parser
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In regulated industries (healthcare, banking):
* PII scrubbing runs as an API gateway middleware (e.g., Envoy or Kong filter).
* Neither the LLM provider nor LangSmith ever sees the raw PII, ensuring HIPAA and PCI-DSS compliance.
""")

st.divider()

# ==============================================================================
# SECTION 4: USER DATA
# ==============================================================================
st.header("4. User Data (Customer Consent, Tenancy Isolation & Cryptographic Pseudonymization)", divider="rainbow")

st.markdown("""
GDPR Article 4 defines **Pseudonymization** as processing personal data so that it can no longer
be attributed to a specific data subject without the use of additional separate information.
Attaching raw user emails (`user_id: 'john.smith@gmail.com'`) in `RunnableConfig.metadata` is a major privacy violation.
""")

st.markdown("### Cryptographic Pseudonymization Protocol")
st.markdown("""
* **One-Way HMAC Hashing**: Instead of logging real user IDs, pass a salted SHA-256 HMAC hash:
  `pseudo_id = hmac.new(TENANT_SALT, raw_user_id.encode(), hashlib.sha256).hexdigest()[:16]`.
* **Tenant Isolation**: Separate customer organizations using distinct tenant metadata (`tenant_id: 'tenant_hash_819'`), enabling multi-tenant trace filtering in LangSmith without exposing customer corporate names.
""")

st.code('''
import hmac
import hashlib

SECRET_SALT = b"enterprise_tenant_salt_v1"

def pseudonymize_user_identity(real_user_id: str) -> str:
    """Generates an irreversible GDPR-compliant pseudonymized user ID."""
    return hmac.new(SECRET_SALT, real_user_id.encode(), hashlib.sha256).hexdigest()[:16]

# Injected safely into LangSmith RunnableConfig
config = {
    "metadata": {
        "user_pseudonym": pseudonymize_user_identity("jane.doe@enterprise.com"),
        "tenant_id": "cust_hash_9482"
    }
}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith dashboards:
* Engineers query and filter traces by `metadata.user_pseudonym`.
* If a specific user reports a bug, customer support looks up the user's pseudonym in an internal secure database to find their specific traces in LangSmith.
""")

st.divider()

# ==============================================================================
# SECTION 5: PRODUCTION TRACE PRIVACY
# ==============================================================================
st.header("5. Production Trace Privacy (Selective Tracing, Masked Inputs/Outputs & Metadata Auditing)", divider="rainbow")

st.markdown("""
Certain enterprise compliance agreements forbid transmitting *any* text prompts or responses to cloud services.
However, engineering teams still require observability to track latency, token costs, tool failures, and system errors.
LangSmith supports **Selective Tracing and Payload Masking**.
""")

st.markdown("### Masked Tracing (Metadata-Only Mode)")
st.markdown("""
* **Payload Masking**: LangChain callbacks can be configured to strip `inputs` and `outputs` completely from run records before dispatch.
* **Telemetry Retention**: The resulting LangSmith run tree preserves the entire graph hierarchy, node execution times, model token counts, tool execution statuses, and stack traces.
* **Zero Text Ingestion**: No conversational text or retrieved documents ever leave the local VPC network.
""")

st.code('''
import os

# Client-side configuration to disable text payload persistence
# LangSmith logs span execution latency, token counts, and errors,
# while replacing prompt and generation text with empty strings or hashes.
os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In defense and financial applications:
* `LANGSMITH_HIDE_INPUTS="true"` and `LANGSMITH_HIDE_OUTPUTS="true"` are enforced in production environments.
* Staging environments retain full text for debugging, while production records only operational telemetry.
""")

st.divider()

# ==============================================================================
# SECTION 6: DATA RETENTION CONSIDERATIONS
# ==============================================================================
st.header("6. Data Retention Considerations (GDPR Right to Be Forgotten, TTL Policies & Automated Purging)", divider="rainbow")

st.markdown("""
Under GDPR Article 17, users have the **Right to be Forgotten (Right to Erasure)**.
If a customer exercises this right, an enterprise is legally obligated to delete all personal data
associated with that user—including records stored in observability trace databases.
Additionally, enterprise policies require automated **Time-To-Live (TTL)** expiration rules.
""")

st.markdown("### Data Retention Lifecycle")
st.markdown("""
1. **Automated TTL Purging**: LangSmith projects are configured with automatic data retention windows (e.g., 14, 30, or 90 days). Traces older than the retention threshold are permanently purged.
2. **Programmatic Erasure API**: If a user submits a GDPR deletion request, an automated script queries LangSmith via the SDK for all traces matching `metadata.user_pseudonym` and deletes them immediately.
""")

st.code('''
from langsmith import Client

client = Client()

def execute_gdpr_right_to_erasure(user_pseudonym: str):
    """
    Locates and permanently purges all traces associated with a pseudonymized user.
    """
    # 1. Query matching runs
    runs = client.list_runs(
        project_name="prod-agent",
        filter=f'eq(metadata.user_pseudonym, "{user_pseudonym}")'
    )
    
    # 2. Delete runs permanently
    run_ids = [run.id for run in runs]
    if run_ids:
        client.delete_runs(run_ids)
        print(f"Purged {len(run_ids)} traces under GDPR erasure request.")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise compliance workflows:
* Data retention policies are enforced at the organization level in LangSmith.
* Audit reports certify to external regulators that all traces older than 30 days are automatically scrubbed from cold storage.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Data Privacy Filter & PII Masking Engine (DataPrivacyEngine)", divider="rainbow")

st.markdown("""
Below is the reference production implementation.
The `DataPrivacyEngine` models PII detection, entity token replacement,
cryptographic pseudonymization, and selective payload masking before trace emission.
""")

st.code('''
class DataPrivacyEngine:
    """
    Production Data Privacy Governance & Telemetry Anonymization Engine.
    Emulates enterprise privacy compliance before sending traces to LangSmith:
    1. Automated PII Entity Masking (SSN, Credit Cards, Emails, Phone Numbers)
    2. Sensitive Corporate Prompt Redaction
    3. User Data Isolation & Cryptographic Pseudonymization
    4. Production Trace Privacy (Selective tracing: masking inputs/outputs)
    5. Data Retention Policies (GDPR / CCPA TTL automatic trace purging)
    """
    def __init__(self, project_name: str = "enterprise-privacy-governance"):
        self.project_name = project_name
        self.sanitized_records: List[Dict[str, Any]] = []

    def sanitize_trace_payload(...) -> Dict[str, Any]:
        """Processes raw prompt, applies privacy masks, and returns anonymized payload."""
        ...
''', language="python")

st.info("The privacy engine is instantiated in module scope and powers the interactive sanitization sandbox below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Real-Time PII Sanitizer & Privacy Governance Sandbox", divider="rainbow")

st.markdown("""
Test enterprise PII sanitization and payload masking in real time.
Input a sensitive customer prompt containing emails, SSNs, and credit cards,
and observe how privacy filters transform the payload before trace emission to LangSmith.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Privacy Interceptor Controls")

    default_sample = "Please verify my account. My SSN is 000-12-3456, email is john.doe@company.com, and card is 4111-2222-3333-4444."
    raw_input_text = st.text_area("Raw Ingress Prompt with PII:", value=default_sample, height=120)

    user_tenant = st.text_input("Raw Tenant ID:", value="enterprise_client_finance_corp")

    enable_pii_mask = st.toggle("Enable Automated PII Entity Masking", value=True)
    mask_all = st.toggle("Enable Complete Input/Output Masking (Zero-Text Mode)", value=False)

    sanitize_btn = st.button("🛡️ Sanitize & Dispatch Trace", use_container_width=True)

with wb_col2:
    st.subheader("Sanitized Trace & Compliance Audit")

    if sanitize_btn or len(privacy_engine.sanitized_records) == 0:
        san_res = privacy_engine.sanitize_trace_payload(
            raw_prompt=raw_input_text,
            mask_pii=enable_pii_mask,
            mask_inputs_completely=mask_all,
            tenant_id=user_tenant
        )
    else:
        san_res = privacy_engine.sanitized_records[-1]

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Compliance Status", "COMPLIANT (PASSED)" if san_res["is_compliant"] else "RAW DATA (RISK)")
    m2.metric("Entities Redacted", len(san_res["detected_pii_entities"]))
    m3.metric("Retention TTL", f"{san_res['retention_policy_days']} Days")

    st.markdown("#### Sanitized Payload (Persisted in LangSmith)")
    with st.container(border=True):
        st.code(san_res["sanitized_payload"], language="text")

    st.markdown("#### Privacy Governance Audit Metadata")
    st.markdown(f"- **Pseudonymized Tenant**: `{san_res['pseudonymized_tenant']}` (One-way hashed)")
    st.markdown(f"- **Detected PII Categories**: {', '.join(san_res['detected_pii_entities']) if san_res['detected_pii_entities'] else 'None'}")
    st.markdown(f"- **GDPR Retention Window**: Purging automatically scheduled after {san_res['retention_policy_days']} days.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do you comply with GDPR Right to Erasure in LangSmith?"):
    st.markdown("""
    **Answer:**
    1. **Pseudonymize at Ingress**: Attach a deterministic one-way hash of the user ID (e.g., `user_pseudonym: hash(user_id)`) to the trace metadata rather than raw names or emails.
    2. **Query Matching Runs**: When an erasure request is received, query LangSmith via `client.list_runs(filter='eq(metadata.user_pseudonym, "...")')`.
    3. **Delete Run Records**: Call `client.delete_runs(run_ids)` to permanently remove all associated traces from LangSmith databases.
    """)

with st.expander("Q2: What is the purpose of LANGSMITH_HIDE_INPUTS and LANGSMITH_HIDE_OUTPUTS?"):
    st.markdown("""
    **Answer:**
    `LANGSMITH_HIDE_INPUTS="true"` and `LANGSMITH_HIDE_OUTPUTS="true"` enable **Metadata-Only Tracing**:
    * They instruct the LangSmith client library to strip text prompts and generated completions before sending telemetry.
    * The platform still captures the full execution graph, latency per node, model token counts, tool execution statuses, and stack traces.
    * This allows organizations with strict zero-data-retention mandates (defense, banking) to achieve full observability without transmitting confidential text.
    """)

with st.expander("Q3: Why must PII scrubbing occur before model inference rather than after?"):
    st.markdown("""
    **Answer:**
    1. **LLM Provider Privacy**: If PII is not scrubbed before inference, the raw PII is sent to external cloud LLM providers (OpenAI, Anthropic), violating data minimization principles.
    2. **Parametric memorization**: LLMs can accidentally memorize and reproduce sensitive numbers in subsequent completions.
    3. **Pre-Trace Filtering**: Scrubbing at the ingress boundary ensures that both the model and the observability tracer receive already-sanitized payloads.
    """)

with st.expander("Q4: How does Presidio NER compare to regular expressions for PII detection?"):
    st.markdown("""
    **Answer:**
    * **Regex**: Fast and deterministic for structured tokens (Credit Cards, SSNs, Emails), but brittle on unstructured PII (person names, home addresses, job titles).
    * **Presidio (Named Entity Recognition)**: Uses transformer-based NLP models to detect contextual entities (e.g., distinguishing "I live in Paris" from "Paris is a celebrity") and combines them with regex pattern analyzers and checksum algorithms (Luhn algorithm for credit cards).
    Production architectures combine regex for structured entities with Presidio NER for unstructured text.
    """)

with st.expander("Q5: What is the difference between Anonymization and Pseudonymization under GDPR?"):
    st.markdown("""
    **Answer:**
    * **Anonymization**: Irreversibly scrubs all identifying data so that the individual can *never* be re-identified by anyone using any reasonable means. Truly anonymized data falls outside GDPR scope.
    * **Pseudonymization**: Replaces direct identifiers with artificial pseudonyms (e.g., hashes). Re-identification is possible only if combined with a separately stored secret key. Pseudonymized data remains within GDPR scope, but benefits from relaxed compliance obligations.
    """)

st.markdown("### Production Traps & Failure Modes")

dp_tr_col1, dp_tr_col2 = st.columns(2)

with dp_tr_col1:
    st.error("""
    **Trap 1: Logging Raw User Emails in Trace Metadata**
    * **Failure**: Passing `config={"metadata": {"email": user.email}}` to track user sessions in LangSmith.
    * **Consequence**: PII is persisted in unencrypted metadata fields, violating GDPR Article 5 data minimization rules.
    * **Remedy**: Always pass a salted cryptographic hash (`pseudonym_id = hash(user.email + salt)`) instead of raw emails.
    """)

    st.error("""
    **Trap 2: Scrubbing Prompts but Leaving Retrieved Documents Unscrubbed**
    * **Failure**: Scrubbing the user query with Presidio, but passing unscrubbed customer PDF chunks from the vector database.
    * **Consequence**: Confidential customer information contained in retrieved chunks is logged to LangSmith child spans.
    * **Remedy**: Ensure privacy scrubbers inspect both the user query and the retrieved document chunks before logging.
    """)

with dp_tr_col2:
    st.warning("""
    **Trap 3: Indefinite Data Retention Without TTL Policies**
    * **Failure**: Leaving LangSmith projects on default unlimited retention for years.
    * **Consequence**: Traces accumulate indefinitely, creating massive legal exposure in the event of a security audit or subpoena.
    * **Remedy**: Configure automated 30-day or 90-day Time-To-Live (TTL) trace deletion policies across all production projects.
    """)

    st.warning("""
    **Trap 4: Masking Tokens That Break Model Reasoning**
    * **Failure**: Replacing critical domain numbers with generic `[MASK]` tokens that prevent the LLM from solving mathematical queries.
    * **Consequence**: The LLM fails to calculate taxes or billing amounts because numbers were scrubbed prematurely.
    * **Remedy**: Use format-preserving tokenization or surrogate fake data for numerical tasks requiring calculation.
    """)

st.markdown("### Core Technical Flashcards")

fcdp_c1, fcdp_c2 = st.columns(2)

with fcdp_c1:
    with st.container(border=True):
        st.markdown("**Front**: What environment variables enable Metadata-Only tracing in LangSmith?")
        st.markdown("**Back**: `LANGSMITH_HIDE_INPUTS='true'` and `LANGSMITH_HIDE_OUTPUTS='true'`. They strip prompt and generation text while preserving node latencies, token counts, and error statuses.")

    with fcdp_c2:
        st.markdown("**Front**: How does Microsoft Presidio enhance PII detection over simple regex?")
        st.markdown("**Back**: Presidio combines regex pattern analyzers with NLP Named Entity Recognition (NER) models and checksum algorithms to detect contextual entities (names, locations) with high accuracy.")

with fcdp_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the GDPR Right to be Forgotten protocol in LangSmith?")
        st.markdown("**Back**: Locating all traces matching a user's pseudonymized ID via `client.list_runs(filter='...')` and permanently purging them using `client.delete_runs(run_ids)`.")

    with fcdp_c2:
        st.markdown("**Front**: Why should tenant salts be used when hashing user IDs?")
        st.markdown("**Back**: Salted HMAC hashes prevent rainbow table reverse-lookup attacks, ensuring that identical user IDs across different organizations yield completely distinct pseudonymized hashes.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of data privacy and telemetry anonymization:")

dp_chk1 = st.checkbox("I know how to redact confidential corporate system prompts using selective span masking.", value=True)
dp_chk2 = st.checkbox("I can implement automated PII scrubbing (Presidio / regex) for SSNs, credit cards, and emails.", value=True)
dp_chk3 = st.checkbox("I understand how to cryptographically pseudonymize user and tenant IDs using salted hashes.", value=True)
dp_chk4 = st.checkbox("I know how to configure `LANGSMITH_HIDE_INPUTS` for zero-text metadata-only observability.", value=True)
dp_chk5 = st.checkbox("I can implement automated 30-day Time-To-Live (TTL) trace deletion policies.", value=True)
dp_chk6 = st.checkbox("I understand how to execute programmatic Right-to-Erasure workflows using `client.delete_runs()`.", value=True)

if dp_chk1 and dp_chk2 and dp_chk3 and dp_chk4 and dp_chk5 and dp_chk6:
    st.success("Mastery Certified: You have mastered data privacy governance, PII anonymization, and regulatory compliance in LangSmith.")
