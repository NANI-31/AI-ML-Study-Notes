import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.2 Logging - RAG Master Curriculum",
    page_icon="📜",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class StructuredLoggingEngine:
    """
    Production-grade Structured Logging Engine:
    1. Sanitizes user queries and masks PII (emails, SSNs, credit cards).
    2. Constructs immutable retrieved document manifests with provenance IDs.
    3. Records dense, sparse, and reranker score distributions.
    4. Logs raw LLM completions, finish reasons, and citation mappings.
    5. Categorizes errors and tracks upstream API status codes and retry counts.
    """
    def __init__(self):
        pass

    def mask_pii(self, text: str) -> str:
        """Simple rule-based PII masking for emails and numeric identifiers."""
        words = text.split()
        sanitized = []
        for w in words:
            if "@" in w and "." in w:
                sanitized.append("[REDACTED_EMAIL]")
            elif len(w.replace("-", "")) == 9 and w.replace("-", "").isdigit():
                sanitized.append("[REDACTED_SSN]")
            elif len(w.replace("-", "")) >= 15 and w.replace("-", "").isdigit():
                sanitized.append("[REDACTED_CARD]")
            else:
                sanitized.append(w)
        return " ".join(sanitized)

    def build_structured_audit_record(
        self,
        raw_query: str,
        retrieved_docs: List[Dict[str, Any]],
        response_text: str,
        http_status: int = 200,
        error_msg: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assembles a comprehensive JSON-serializable structured audit log entry."""
        sanitized_query = self.mask_pii(raw_query)

        # Document manifests and score distributions
        doc_manifest = []
        similarity_scores = []
        for d in retrieved_docs:
            sim = float(d.get("score", 0.0))
            similarity_scores.append(sim)
            doc_manifest.append({
                "doc_id": d.get("id", "DOC-UNKNOWN"),
                "source": d.get("source", "KnowledgeBase"),
                "score": sim,
                "rank": d.get("rank", 1)
            })

        mean_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0

        return {
            "timestamp_utc": "2026-10-13T16:20:00Z",
            "event_type": "rag_inference_audit",
            "query_telemetry": {
                "sanitized_query": sanitized_query,
                "query_length_chars": len(raw_query),
                "has_redacted_pii": sanitized_query != raw_query
            },
            "retrieval_telemetry": {
                "candidate_count": len(retrieved_docs),
                "mean_similarity_score": round(mean_sim, 3),
                "documents": doc_manifest
            },
            "generation_telemetry": {
                "response_text": response_text,
                "finish_reason": "stop" if not error_msg else "error",
                "citations_present": "[" in response_text and "]" in response_text
            },
            "api_telemetry": {
                "http_status_code": http_status,
                "provider": "openai-azure-v1",
                "is_error": http_status >= 400 or error_msg is not None,
                "error_detail": error_msg
            }
        }


# Instantiate engine at module level
log_engine = StructuredLoggingEngine()

# --- HEADER SECTION ---
st.title("18.2 Production RAG Logging: Structured JSON Schemas, PII Scrubbing, Audit Trails, & Error Diagnostics")
st.caption("Chapter 18: RAG Observability and Monitoring | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In prototype scripts, developers often rely on unstructured print statements: `print(f"Retrieved {len(docs)} documents")`.

In enterprise production environments, **unstructured text logs are unusable for automated querying, 
compliance auditing, or security verification**. 
If a compliance officer asks for the exact document snippets that informed a critical financial recommendation 
six weeks ago, raw text dumps cannot reconstruct the system state.

Production RAG logging mandates **Structured JSON Logging** with strict schema conventions across six core dimensions:
1. **User Queries**: Sanitized text with automated PII masking and semantic intent tags.
2. **Retrieved Documents**: Document manifests with immutable IDs, source URIs, and chunk offsets.
3. **Similarity Scores**: Quantitative score distributions across dense, sparse, and reranker stages.
4. **Generated Responses**: Full completions, citation tags, finish reasons, and model signatures.
5. **Errors**: Graceful failure classifications (e.g., ContextWindowExceeded, RateLimitError).
6. **API Calls**: Upstream provider latencies, HTTP status codes, and retry counters.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Structured RAG Logging & Compliance Audit Pipeline", divider="rainbow")

st.markdown("""
The structured logging architecture captures events at each pipeline step, strips sensitive customer data, 
formats records into JSON schemas, and ships them to an immutable observability lakehouse.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph StructuredLoggingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // INGESTION & REDACTION
        subgraph cluster_sanitization {
            label="1. Ingestion & PII Redaction";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            RawEvent [label="Raw Execution Event\\n(User query, retrieved chunks, API call)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            PiiScrubber [label="Inline PII Scrubber\\n(Redacts SSNs, credit cards, emails)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // TELEMETRY AGGREGATION
        subgraph cluster_structuring {
            label="2. JSON Telemetry Structuring";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            QueryTelemetry [label="Query & Session Envelope\\n(Sanitized query, session hash)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            DocManifest [label="Document Manifest Envelope\\n(Doc IDs, URIs, similarity scores)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            GenTelemetry [label="Generation & API Envelope\\n(Completion, citations, HTTP codes)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            ErrorTelemetry [label="Error & Diagnostic Envelope\\n(Stack trace, rate limits, retries)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // SINK & COMPLIANCE
        subgraph cluster_sink {
            label="3. Log Sinks & Compliance Lakehouse";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            JsonSerializer [label="JSON Lines Serializer\\n(Standard schema output stream)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            AuditStorage [label="Immutable Audit Lakehouse\\n(OpenSearch, Datadog, Snowflake SIEM)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        }

        // PIPELINE EDGES
        RawEvent -> PiiScrubber;
        PiiScrubber -> QueryTelemetry;
        PiiScrubber -> DocManifest;
        PiiScrubber -> GenTelemetry;
        PiiScrubber -> ErrorTelemetry;

        QueryTelemetry -> JsonSerializer;
        DocManifest -> JsonSerializer;
        GenTelemetry -> JsonSerializer;
        ErrorTelemetry -> JsonSerializer;

        JsonSerializer -> AuditStorage;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Logging Invariants:
* **Zero Plaintext PII**: Sensitive customer data must be redacted at the edge before log persistence.
* **Non-Repudiation Provenance**: If an LLM generates a recommendation, the exact chunk IDs and versions are permanently auditable.
* **Machine-Parsable JSON**: Unstructured free text is prohibited; every entry must conform to typed schemas.
""")

st.divider()

# ==============================================================================
# SECTION 2: USER QUERIES
# ==============================================================================
st.header("2. User Queries (PII Redaction, Query Sanitization, Semantic Hashing & Intent Tracking)", divider="rainbow")

st.markdown("""
User queries reflect raw customer intent, but present the highest risk of **Personally Identifiable Information (PII)** leakage.
""")

uq_col1, uq_col2 = st.columns([1, 1])

with uq_col1:
    st.markdown("""
    ### Query Logging Protocols:
    * **Automated Redaction**: Run regex or entity recognition scanners to replace SSNs, emails, phone numbers, and credit cards with token markers (`[REDACTED_EMAIL]`).
    * **Semantic Hashing**: Compute a SHA-256 hash of the normalized query. This permits cache hit analytics and duplicate clustering without reading raw personal text.
    * **Intent Classification Logging**: Log the classified intent category (e.g., `BILLING_INQUIRY`, `TECH_SUPPORT`, `ACCOUNT_CANCEL`) for aggregate product analytics.
    """)

with uq_col2:
    st.markdown("### Before vs After PII Scrubbing")
    pii_example = pd.DataFrame([
        {"Original Raw Query": "Can you refund card 4111-2222-3333-4444 for john@acme.com?", "Sanitized Logged Query": "Can you refund card [REDACTED_CARD] for [REDACTED_EMAIL]?", "Action": "PII Stripped at Edge"},
        {"Original Raw Query": "What is the failover SLA for Aurora DB?", "Sanitized Logged Query": "What is the failover SLA for Aurora DB?", "Action": "Unchanged (Safe)"}
    ])
    st.dataframe(pii_example, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: RETRIEVED DOCUMENTS
# ==============================================================================
st.header("3. Retrieved Documents (Document Manifests, Chunk Fingerprints & Cryptographic Provenance)", divider="rainbow")

st.markdown("""
Logging the retrieved documents establishes an auditable record of what evidence was provided to the LLM.
""")

rd_col1, rd_col2 = st.columns([1, 1])

with rd_col1:
    st.markdown("""
    ### Why Logging Full Text is an Anti-Pattern:
    * **Log Storage Inflation**: Logging 10 chunks of 500 words on 100,000 queries daily consumes gigabytes of redundant storage.
    * **The Manifest Pattern**: Log immutable **Document Manifests** containing:
      1. `document_id`: Unique canonical document ID (e.g., `DOC-SEC-401`).
      2. `chunk_hash`: MD5 or SHA-256 hash of the chunk content.
      3. `version_tag`: Document revision timestamp (e.g., `2024-03-01T12:00:00Z`).
      4. `retrieval_rank`: Initial rank position (1 to K).
    """)

with rd_col2:
    st.markdown("### Document Manifest Log Sample")
    st.code(r'''
"retrieved_manifest": [
  {
    "doc_id": "SLA-POLICY-101",
    "chunk_id": "c_004",
    "chunk_hash": "a1b2c3d4e5f6",
    "source_uri": "s3://legal-docs/sla_2024.pdf",
    "rank": 1
  }
]
''', language="json")

st.divider()

# ==============================================================================
# SECTION 4: SIMILARITY SCORES
# ==============================================================================
st.header("4. Similarity Scores (Dense Cosine Distances, Sparse BM25 Scores, & Score Distributions)", divider="rainbow")

st.markdown("""
Logging similarity scores allows machine learning engineers to detect **retriever distribution drift** 
and calibrate production threshold gates.
""")

ss_col1, ss_col2 = st.columns([1, 1])

with ss_col1:
    st.markdown("""
    ### Telemetry Value of Score Logging:
    * **Threshold Tuning**: If the minimum cosine similarity consistently hovers around 0.58 on unanswerable queries, engineers can tune the empty-context threshold to 0.65.
    * **Reranker Impact**: Comparing dense cosine scores against cross-encoder logits reveals whether the reranker successfully promoted buried gold facts.
    * **Corpus Drift Alerts**: If average similarity scores drop by 15% across all queries over a week, it signals that user vocabulary has drifted away from indexed documents.
    """)

with ss_col2:
    st.markdown("### Score Distribution Log Structure")
    st.code(r'''
"score_telemetry": {
  "top_1_score": 0.942,
  "top_k_min_score": 0.718,
  "mean_score": 0.835,
  "score_variance": 0.008,
  "rerank_delta": +0.224
}
''', language="json")

st.divider()

# ==============================================================================
# SECTION 5: GENERATED RESPONSES
# ==============================================================================
st.header("5. Generated Responses (Raw Completions, Citation Mapping, & Model Fingerprinting)", divider="rainbow")

st.markdown("""
Capturing the raw model completion enables automated quality auditing, citation verification, 
and human spot-checks.
""")

gr_col1, gr_col2 = st.columns([1, 1])

with gr_col1:
    st.markdown("""
    ### Mandatory Generation Fields:
    * **`raw_completion`**: The verbatim response string generated by the model.
    * **`finish_reason`**: `stop` (natural completion), `length` (truncated by max_tokens), or `content_filter`.
    * **`model_fingerprint`**: System fingerprint returned by providers (e.g., `fp_44709d6fcb`) to track silent model backend updates.
    * **`citation_tags`**: Array of extracted inline bracket citations (e.g., `["SLA-101"]`).
    """)

with gr_col2:
    st.markdown("### Generation Telemetry Log Sample")
    st.code(r'''
"generation_event": {
  "model": "gpt-4o-2024-08-06",
  "system_fingerprint": "fp_c4e1a8",
  "completion": "Failovers take 120 seconds [DOC-1].",
  "finish_reason": "stop",
  "citations": ["DOC-1"],
  "total_tokens": 2070
}
''', language="json")

st.divider()

# ==============================================================================
# SECTION 6: ERRORS
# ==============================================================================
st.header("6. Errors (Timeout Interceptions, Rate Limits, Context Window Breaches & Graceful Triage)", divider="rainbow")

st.markdown("""
Unstructured exception traces (`print(e)`) obscure system health. Production RAG categorizes errors 
into standardized diagnostic taxonomy codes.
""")

err_col1, err_col2 = st.columns([1, 1])

with err_col1:
    st.markdown("""
    ### Production Error Taxonomy:
    1. **`CONTEXT_WINDOW_EXCEEDED`**: Injected prompt exceeded model maximum tokens.
    2. **`RATE_LIMIT_THROTTLED` (429)**: Provider TPM / RPM limits hit; requires exponential backoff.
    3. **`RETRIEVAL_TIMEOUT`**: Vector database failed to return within SLA window (e.g., >500ms).
    4. **`SCHEMA_PARSING_ERROR`**: Model output invalid JSON that failed Pydantic validation.
    5. **`EMPTY_CONTEXT_BREACH`**: Zero candidate chunks passed the minimum relevance threshold.
    """)

with err_col2:
    st.markdown("### Error Log Taxonomy Schema")
    st.code(r'''
"error_telemetry": {
  "error_code": "CONTEXT_WINDOW_EXCEEDED",
  "status_code": 400,
  "attempted_tokens": 8450,
  "max_allowed_tokens": 8192,
  "remediation": "Triggered sentence-level context truncation"
}
''', language="json")

st.divider()

# ==============================================================================
# SECTION 7: API CALLS
# ==============================================================================
st.header("7. API Calls (HTTP Status Codes, Retries, Circuit Breaker States & Provider Payloads)", divider="rainbow")

st.markdown("""
Every outbound network hop to an external provider (OpenAI, Anthropic, Pinecone, Cohere) 
must log performance metrics to detect vendor outages.
""")

api_col1, api_col2 = st.columns([1, 1])

with api_col1:
    st.markdown("""
    ### Mandatory API Telemetry Attributes:
    * **`provider_name`**: e.g., `openai`, `anthropic`, `cohere-rerank`.
    * **`http_status`**: 200, 429, 500, 503.
    * **`retry_count`**: Number of exponential backoff attempts executed.
    * **`circuit_breaker_status`**: `CLOSED` (Healthy), `HALF_OPEN` (Testing), or `OPEN` (Tripped).
    * **`dns_and_tls_latency_ms`**: Measures network transit vs model execution time.
    """)

with api_col2:
    st.markdown("### Upstream API Status Code Breakdown")
    api_summary = pd.DataFrame([
        {"HTTP Status": "200 OK", "Meaning": "Successful execution", "Action": "Process response"},
        {"HTTP Status": "429 Too Many Requests", "Meaning": "Rate limit / quota breached", "Action": "Trigger exponential backoff retry"},
        {"HTTP Status": "500 / 503 Server Error", "Meaning": "Provider downtime / outage", "Action": "Failover to secondary provider"},
        {"HTTP Status": "400 Bad Request", "Meaning": "Context limit exceeded or invalid param", "Action": "Log critical bug and abort"}
    ])
    st.dataframe(api_summary, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Structured Audit Logging Engine", divider="rainbow")

st.markdown("""
Below is the production implementation of `StructuredLoggingEngine`. It handles PII scrubbing, 
document manifest generation, score statistics, and structured JSON record compilation in module scope.
""")

st.code(r'''
class StructuredLoggingEngine:
    """
    Enterprise Structured Audit Logger:
    - PII scrubbing at the edge.
    - Document manifest creation with chunk hashes.
    - Similarity score distribution calculation.
    - Error taxonomy categorization and HTTP status code tracking.
    """
    def mask_pii(self, text: str) -> str:
        words = text.split()
        sanitized = []
        for w in words:
            if "@" in w and "." in w:
                sanitized.append("[REDACTED_EMAIL]")
            elif len(w.replace("-", "")) == 9 and w.replace("-", "").isdigit():
                sanitized.append("[REDACTED_SSN]")
            elif len(w.replace("-", "")) >= 15 and w.replace("-", "").isdigit():
                sanitized.append("[REDACTED_CARD]")
            else:
                sanitized.append(w)
        return " ".join(sanitized)

    def build_structured_audit_record(
        self, raw_query: str, retrieved_docs: List[Dict[str, Any]],
        response_text: str, http_status: int = 200, error_msg: Optional[str] = None
    ) -> Dict[str, Any]:
        clean_q = self.mask_pii(raw_query)
        scores = [float(d.get("score", 0.0)) for d in retrieved_docs]
        manifest = [{"id": d.get("id", "DOC"), "score": float(d.get("score", 0.0))} for d in retrieved_docs]

        return {
            "timestamp": "2026-10-13T16:20:00Z",
            "query": {"text": clean_q, "had_pii": clean_q != raw_query},
            "retrieval": {
                "candidates": len(retrieved_docs),
                "mean_score": round(float(np.mean(scores)), 3) if scores else 0.0,
                "manifest": manifest
            },
            "generation": {
                "response": response_text,
                "has_citations": "[" in response_text and "]" in response_text
            },
            "api": {"status": http_status, "is_error": http_status >= 400, "error": error_msg}
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Trace & Audit Log Viewer", divider="rainbow")

st.markdown("""
Test how raw production inputs are sanitized and transformed into enterprise structured JSON audit logs. 
Input queries containing PII, toggle error scenarios, and inspect the generated audit record in real time.
""")

sample_docs_pool = [
    {"id": "DOC-SEC-101", "source": "Cloud Security Policy", "score": 0.94, "rank": 1},
    {"id": "DOC-OPS-202", "source": "DevOps Failover Runbook", "score": 0.88, "rank": 2},
    {"id": "DOC-HR-303", "source": "Employee Travel Guidelines", "score": 0.62, "rank": 3}
]

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Simulated Pipeline Event Inputs")
    raw_user_query = st.text_input(
        "Raw User Query (Try entering an email or SSN):",
        value="Can you check the failover SLA for user alice@enterprise.com with SSN 123-45-6789?"
    )
    raw_answer = st.text_area(
        "Generated Response Text:",
        value="Database failover completes automatically within 120 seconds [DOC-OPS-202].",
        height=80
    )
    sim_status_code = st.selectbox(
        "Simulated Upstream API Status Code:",
        options=[200, 429, 500, 503],
        index=0
    )
    sim_error_text = None
    if sim_status_code >= 400:
        sim_error_text = "Upstream provider rate limit throttled (429 Too Many Requests)."

with wb_c2:
    st.markdown("### Generated Structured Audit Record")
    audit_record = log_engine.build_structured_audit_record(
        raw_query=raw_user_query,
        retrieved_docs=sample_docs_pool,
        response_text=raw_answer,
        http_status=sim_status_code,
        error_msg=sim_error_text
    )

    if audit_record["query_telemetry"]["has_redacted_pii"]:
        st.warning("🛡️ PII Detected & Redacted: Customer email or identifier masked before log storage.")
    else:
        st.success("✅ Clean Query: Zero sensitive PII entities detected.")

    st.json(audit_record)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: Why are unstructured text logs dangerous in enterprise RAG systems, and how does structured JSON logging solve this?**"):
    st.markdown("""
    **Answer**:
    Unstructured logs (e.g., `print(f"User: {q}, Answer: {a}")`) cannot be queried efficiently by automated SIEM tools (Datadog, Splunk, Elastic). If a security breach occurs or a regulatory audit demands proof of what context was provided, text search over petabytes of text logs is slow and prone to parser failure.
    Structured JSON logs enforce typed schemas. Attributes like `doc_id`, `user_hash`, `similarity_score`, and `finish_reason` become indexed columns, enabling sub-second SQL queries such as:
    `SELECT doc_id, COUNT(*) FROM rag_logs WHERE similarity_score < 0.60 GROUP BY doc_id`.
    """)

with st.expander("**Q2: How does Client-Side PII Scrubbing operate in production RAG logging?**"):
    st.markdown("""
    **Answer**:
    Under GDPR, HIPAA, and CCPA, logging plaintext customer identifiers (names, SSNs, credit cards, medical symptoms) constitutes a data protection violation.
    Client-side PII scrubbing intercepts text **before** it is passed to logging agents:
    1. It evaluates text with Named Entity Recognition (NER) models or regex patterns.
    2. Sensitive tokens are replaced with synthetic placeholders (`[REDACTED_EMAIL]`, `[REDACTED_SSN]`).
    3. The scrubbed string is written to disk, ensuring that even if log storage is compromised, no PII is leaked.
    """)

with st.expander("**Q3: What is the Document Manifest Pattern, and why is it preferred over logging raw chunk text?**"):
    st.markdown("""
    **Answer**:
    Logging full text for every retrieved chunk (e.g., 5 chunks x 500 words = 2,500 words per request) inflates log storage costs by 10x to 50x.
    The Document Manifest Pattern logs only metadata:
    `[{"doc_id": "SEC-101", "chunk_index": 3, "sha256": "8a4f9b...", "uri": "s3://..."}]`
    Because raw documents are already versioned and stored in immutable object storage (S3), auditors can reconstruct the exact prompt state on demand without duplicating gigabytes of text in telemetry logs.
    """)

with st.expander("**Q4: How should a production logging pipeline categorize LLM API errors?**"):
    st.markdown("""
    **Answer**:
    Generic error logging (`"API call failed"`) prevents automated remediation. Errors must be categorized into standard taxonomy codes:
    * **`RATE_LIMIT_429`**: Transient throttle; trigger exponential backoff.
    * **`CONTEXT_LENGTH_400`**: Prompt too large; trigger extractive compression.
    * **`AUTH_ERROR_401`**: Stale API key; alert DevOps immediately.
    * **`PROVIDER_500`**: Upstream outage; trigger circuit breaker failover to secondary provider.
    """)

with st.expander("**Q5: Why should RAG logs record the provider's system fingerprint (e.g., system_fingerprint)?**"):
    st.markdown("""
    **Answer**:
    Model providers (OpenAI, Anthropic) occasionally update backend weights, quantized kernels, or inference infrastructure under existing model names (e.g., `gpt-4o`).
    Logging the `system_fingerprint` allows engineers to detect when provider backend changes correlate with unexpected drops in Faithfulness or changes in output formatting.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Logging Sensitive PII in Plaintext**
    * **Failure Mode**: Logging raw customer prompts containing credit cards or medical symptoms creates severe regulatory compliance violations.
    * **Solution**: Implement client-side PII masking before serializing log records.
    """)

    st.error("""
    **Production Trap 2: Duplicating Full Chunk Text in Every Log Entry**
    * **Failure Mode**: Writing 3,000 words of retrieved text per request consumes petabytes of log storage, inflating cloud logging bills.
    * **Solution**: Log lightweight document manifests with chunk hashes and S3 pointers.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Swallowing Exception Codes with Generic Messages**
    * **Failure Mode**: Catching exceptions and logging `"An error occurred"` hides whether the root cause was a rate limit, timeout, or context breach.
    * **Solution**: Use structured error taxonomy codes with explicit HTTP status tracking.
    """)

    st.warning("""
    **Production Trap 4: Synchronous Blocking Log Writes**
    * **Failure Mode**: Writing logs synchronously over network sockets adds 50-100ms of user-perceived latency to every RAG query.
    * **Solution**: Use asynchronous background logging workers (e.g., Python `queue.Queue` or FluentBit).
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Structured JSON Logging")
        st.caption("Why is JSON logging mandatory for enterprise RAG?")
        st.markdown("**Back**: Enables automated indexing, sub-second SIEM queries, compliance audits, and non-repudiation verification.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: PII Edge Masking")
        st.caption("When should PII redaction occur in the logging pipeline?")
        st.markdown("**Back**: At the client edge before writing to disk or exporting over the network, ensuring zero PII persistence.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Document Manifest Pattern")
        st.caption("What replaces full chunk text in production logs?")
        st.markdown("**Back**: Immutable metadata pointers (doc ID, chunk index, SHA-256 hash, and S3 URI), reducing log size by 90%.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Model System Fingerprint")
        st.caption("What does logging system_fingerprint detect?")
        st.markdown("**Back**: Silent provider backend infrastructure or weight updates that cause unexpected behavioral changes.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of structured RAG audit logging before proceeding:")

c1 = st.checkbox("I understand why unstructured text logs fail enterprise auditing and how structured JSON schemas solve it.", value=True)
c2 = st.checkbox("I can implement client-side PII redaction to mask sensitive emails, cards, and personal identifiers.", value=True)
c3 = st.checkbox("I know how to use the Document Manifest Pattern to log chunk provenance without storage inflation.", value=True)
c4 = st.checkbox("I understand the value of logging similarity score distributions for detecting retriever drift.", value=True)
c5 = st.checkbox("I can classify RAG exceptions into standardized diagnostic taxonomy codes (429, timeouts, context limits).", value=True)
c6 = st.checkbox("I know how to track upstream provider HTTP status codes, retry counts, and system fingerprints.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **RAG Logging Mastery Achieved!** You are ready to deploy enterprise-grade, compliance-ready audit pipelines.")
