import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="20.2 Data Security - Vector DB Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("20.2 Vector Database Data Security: PII Scrubbing, Encryption at Rest & Vector Inversion Defense")
st.caption("Chapter 20: Vector Database Security | Domain: 4. Vector Database")
st.markdown("""
A dangerous myth in machine learning engineering is that *“vectors are mathematical hashes that inherently protect privacy.”* 
This belief is provably false.

Recent cybersecurity research has demonstrated **Vector Inversion Attacks** (Morris et al., 2023): using specialized decoder models, 
adversaries can reconstruct up to 80% of the original natural language text directly from a high-dimensional embedding vector! 

**Vectors are sensitive data, not one-way cryptographic hashes.**

Furthermore, vector databases store the unencrypted source text alongside the vectors to support retrieval hydration. 
If an attacker compromises your storage volume, all ingested customer documents are exposed in plaintext.

To satisfy compliance standards (GDPR, HIPAA, SOC 2, PCI-DSS), enterprise vector architectures must implement 
a rigorous **Data Security & Privacy Pipeline**:
1. **Pre-Ingestion PII Redaction**: Scrubbing Social Security Numbers, credit cards, emails, and phone numbers *before* embedding.
2. **Encryption In Transit**: Enforcing TLS 1.3 across all client-server and inter-node replication traffic.
3. **Encryption At Rest**: AES-256 encryption across NVMe block storage and SQLite database files via SQLCipher/LUKS.
4. **Data Retention & Right-to-be-Forgotten**: Cryptographic document purging to comply with GDPR Article 17.

In this lesson, we explore PII sanitization pipelines, analyze vector inversion risks, inspect enterprise encryption blueprints, 
and test an interactive Live PII Detection & Redaction Lab.
""")

st.divider()

# Section 1: The Vector Inversion Threat & Data Sensitivity
st.header("1. The Vector Inversion Threat & Data Sensitivity", divider="rainbow")

col_inv1, col_inv2 = col_inv1, col_inv2 = st.columns([1.1, 0.9])

with col_inv1:
    st.markdown("""
    ### Why Vectors Are NOT Privacy-Preserving
    
    * **The Misconception**:
      * *"We don't store raw text; we only store 1536-dimensional floating point vectors, so customer privacy is guaranteed."*
    
    * **The Mathematical Reality (Vector Inversion Attacks)**:
      * Embeddings are continuous semantic representations trained to retain maximum information about the original sentence.
      * Attackers train an autoregressive decoder model (`vec -> tokens`) using contrastive alignment.
      * Research proves that given a single vector embedding, decoders can recover exact names, medical conditions, and passwords with startling accuracy!
    
    * **The Security Mandate**:
      * Treat embedding vectors with the **exact same legal classification and encryption rigor** as unencrypted plaintext documents!
      * Any sensitive Personally Identifiable Information (PII) must be sanitized **before** the text ever reaches an embedding model.
    """)

with col_inv2:
    st.markdown("### Vector Inversion Attack Mechanics")
    inv_dot = """
    digraph VectorInversion {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        SecretText [label="Raw Text Containing PII\\n'Patient Alice has Stage 3 Lymphoma'", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        EmbModel [label="Embedding Model (Bi-Encoder)\\nEncodes to 1536-dim Float Array", fillcolor="#FEF3C7", color="#D97706"];

        VecStore [label="Exposed Vector in Database\\n[0.082, -0.412, 0.991, ...]", fillcolor="#DBEAFE", color="#2563EB", style="filled,bold"];

        AttackerDec [label="Adversarial Inversion Decoder\\n(Trained to reverse embeddings)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Reconstructed [label="Reconstructed Text (Exfiltrated!)\\n'Patient Alice diagnosed with Lymphoma'", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        SecretText -> EmbModel;
        EmbModel -> VecStore;
        VecStore -> AttackerDec [label="Stolen Vectors", color="#DC2626", style="dashed"];
        AttackerDec -> Reconstructed [label="80%+ Semantic Fidelity", color="#DC2626", style="bold"];
    }
    """
    st.graphviz_chart(inv_dot, use_container_width=False)

st.divider()

# Section 2: Automated PII Scrubbing & Redaction Architecture
st.header("2. Automated PII Scrubbing & Redaction Architecture", divider="rainbow")

col_pii1, col_pii2 = col_pii1, col_pii2 = st.columns([1, 1])

with col_pii1:
    st.markdown("""
    ### Three Anonymization Strategies
    
    1. **Direct Redaction (Entity Replacement)**:
       * Replaces PII with generic categorical placeholders:  
         `"John Doe"` -> `"<PERSON>"`, `"415-555-0199"` -> `"<PHONE_NUMBER>"`.
       * *Pros*: 100% irreversible privacy guarantee.
       * *Cons*: May slightly reduce semantic richness in downstream LLM generation.
    
    2. **Synthetic Entity Masking (Pseudonymization)**:
       * Replaces real names with realistic fake synthetic names:  
         `"Alice Smith"` -> `"Customer_8291"`.
       * Maintains consistent mappings in a secure vault so authorized users can re-identify entities if needed.
    
    3. **Salted Cryptographic Hashing**:
       * Hashes identifiers: `SHA256(SSN + Secret_Salt)`.
       * Allows exact-match lookups across databases without exposing raw numbers.
    """)

with col_pii2:
    st.markdown("### Pre-Ingestion Sanitization Code")
    st.code(r"""import re
from typing import Dict, Any

class PIIScrubber:
    \"\"\"Scrubs sensitive entities before embedding generation.\"\"\"
    
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+",
        "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    }

    @classmethod
    def sanitize(cls, text: str) -> str:
        sanitized = text
        for entity_type, pattern in cls.PATTERNS.items():
            sanitized = re.sub(pattern, f"<{entity_type}_REDACTED>", sanitized)
        return sanitized

# Ingestion Flow:
raw_text = "Contact Alice at alice@hospital.org or 415-555-0199 regarding SSN 000-12-3456."
clean_text = PIIScrubber.sanitize(raw_text)

# Result: "Contact Alice at <EMAIL_REDACTED> or <PHONE_REDACTED> regarding SSN <SSN_REDACTED>."
# Send clean_text to Embedding Model & ChromaDB!
""", language="python")

st.divider()

# Section 3: Dual-Layer Encryption Architecture (At Rest & In Transit)
st.header("3. Dual-Layer Encryption Architecture (At Rest & In Transit)", divider="rainbow")

col_enc1, col_enc2 = col_enc1, col_enc2 = st.columns([1, 1])

with col_enc1:
    st.markdown("""
    ### 1. Encryption In Transit (Network Wire)
    * **Protocol**: Enforce **TLS 1.3** across all HTTP REST, gRPC, and inter-node cluster replication traffic.
    * **Cipher Suites**: Restrict to forward-secret AEAD ciphers (`TLS_AES_256_GCM_SHA384` or `TLS_CHACHA20_POLY1305_SHA256`).
    * **Certificate Management**: Automated certificate rotation via cert-manager or HashiCorp Vault. Plain unencrypted HTTP connections must be immediately dropped at the ingress controller.
    """)

with col_enc2:
    st.markdown("""
    ### 2. Encryption At Rest (Disk Storage)
    * **Block-Level NVMe Encryption**: AWS EBS / GCP Persistent Disk encryption with Customer Managed Keys (AWS KMS / Cloud KMS).
    * **Application-Level SQLite Encryption (SQLCipher)**:
      * ChromaDB uses SQLite for metadata persistence.
      * In regulated environments, compile SQLite with **SQLCipher** to enforce 256-bit AES encryption across all database pages on disk.
    * **Vector Index Files (`.bin` / HNSW graphs)**: Stored on encrypted filesystems with POSIX permissions restricted to the unprivileged `chroma` system user (`chmod 600`).
    """)

st.divider()

# Section 4: Interactive Workbench: Live PII Detection & Sanitization Lab
st.header("4. Interactive Workbench: Live PII Detection & Sanitization Lab", divider="rainbow")

st.markdown("""
Paste or modify unstructured text containing sensitive customer records. 
Observe how the pre-ingestion security pipeline detects PII, sanitizes entities, 
and generates safe, compliance-ready embeddings.
""")

SAMPLE_SENSITIVE_TEXT = """Customer Account Verification Log:
Account Holder: Jane Doe
Contact Email: jane.doe@fintechcorp.io
Phone Number: (555) 234-5678
Payment Card: 4111 2222 3333 4444
Social Security Number: 123-45-6789
Notes: User requested loan approval for $45,000 to purchase real estate property in California."""

col_lab_in, col_lab_cfg = col_lab_in, col_lab_cfg = st.columns([1.1, 0.9])

with col_lab_in:
    st.subheader("1. Raw Unstructured Input Document")
    user_input_text = st.text_area("Source Document Text:", value=SAMPLE_SENSITIVE_TEXT, height=180)

with col_lab_cfg:
    st.subheader("2. Redaction Policy Configuration")
    mask_mode = st.radio(
        "Sanitization Method:",
        [
            "Categorical Redaction (<ENTITY_TYPE>)",
            "Pseudonym Masking (Synthetic IDs)",
            "Cryptographic Hash Masking (SHA-256)"
        ]
    )
    
    scrub_emails = st.checkbox("Scrub Email Addresses", value=True)
    scrub_phones = st.checkbox("Scrub Phone Numbers", value=True)
    scrub_cc = st.checkbox("Scrub Credit Card Numbers", value=True)
    scrub_ssn = st.checkbox("Scrub Social Security Numbers", value=True)

# Sanitization Engine
def apply_sanitization(text: str, mode: str) -> Tuple[str, List[Dict[str, str]]]:
    detected = []
    sanitized = text
    
    patterns = []
    if scrub_emails: patterns.append(("EMAIL", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+"))
    if scrub_phones: patterns.append(("PHONE", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"))
    if scrub_cc: patterns.append(("CREDIT_CARD", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"))
    if scrub_ssn: patterns.append(("SSN", r"\b\d{3}-\d{2}-\d{4}\b"))
    
    for entity_type, pattern in patterns:
        for match in re.finditer(pattern, sanitized):
            val = match.group(0)
            detected.append({"Type": entity_type, "Raw Value": val})
            
            if "Categorical" in mode:
                replacement = f"<{entity_type}_REDACTED>"
            elif "Pseudonym" in mode:
                replacement = f"SYNTH_{entity_type}_{abs(hash(val)) % 10000}"
            else:
                replacement = f"HASH_{abs(hash(val)) % 99999999}"
                
            sanitized = sanitized.replace(val, replacement)
            
    return sanitized, detected

sanitized_result, entities_found = apply_sanitization(user_input_text, mask_mode)

st.divider()

col_lab_out, col_lab_audit = col_lab_out, col_lab_audit = st.columns([1.1, 0.9])

with col_lab_out:
    st.subheader("3. Sanitized Document (Safe for ChromaDB)")
    st.text_area("Processed Text:", value=sanitized_result, height=180)

with col_lab_audit:
    st.subheader("Detected PII Entities & Audit Trail")
    if entities_found:
        st.dataframe(pd.DataFrame(entities_found), use_container_width=True, hide_index=True)
        st.success(f"✅ Successfully sanitized **{len(entities_found)} PII entities**. Zero plaintext secrets will leak to vector embeddings.")
    else:
        st.info("No PII entities detected based on active filters.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is a Vector Inversion Attack, and why does it invalidate the idea that embeddings are privacy-preserving?"):
    st.markdown("""
    **Answer**:
    A Vector Inversion Attack is a technique where an attacker trains an autoregressive decoder model to **reverse a high-dimensional vector back into natural language**:
    * Neural embeddings are trained specifically to preserve semantic information.
    * Research (e.g. Morris et al., 2023 - *Text Embeddings Reveal (Almost) As Much As Text*) proves that attackers can reconstruct up to 80% of original words and recover exact sensitive entities (names, conditions, locations) directly from raw vectors.
    * **Conclusion**: Vectors must be classified as sensitive personal data under GDPR and HIPAA, requiring encryption at rest and strict access controls.
    """)

with st.expander("Q2: Why must PII scrubbing occur BEFORE generating embeddings rather than after retrieval?"):
    st.markdown("""
    **Answer**:
    If PII scrubbing occurs after retrieval (right before passing text to the LLM):
    1. **Permanent Vector Leakage**: The embedding model has already processed the raw PII, baking the sensitive entity into the high-dimensional vector permanently stored in your database.
    2. **Searchability of Secrets**: An attacker can search for sensitive terms (e.g. searching for a phone number or SSN) and successfully retrieve matching candidate vectors.
    3. **Regulatory Non-Compliance**: Storing unmasked PII in the vector index violates GDPR Article 5 (Data Minimization) and HIPAA regulations.
    * **Requirement**: Always sanitize text **prior to calling the embedding model**.
    """)

with st.expander("Q3: How do you handle GDPR 'Right to be Forgotten' (Article 17) in a vector database?"):
    st.markdown("""
    **Answer**:
    When a customer requests the erasure of their personal data:
    1. **Delete by ID / Metadata**: Call `collection.delete(where={"customer_id": "cust_123"})`.
    2. **HNSW Soft vs Hard Deletion**: Most HNSW implementations merely mark nodes as "deleted" (tombstones) in memory.
    3. **Compaction & Re-indexing**: To guarantee cryptographic erasure, the database must periodically run a compaction job that purges tombstoned nodes from the graph and flushes SQLite WAL archives.
    4. **Audit Logging**: Record the deletion timestamp and document IDs in a compliance audit log without recording the underlying sensitive text.
    """)

with st.expander("Q4: What is SQLCipher and how is it used to secure ChromaDB metadata?"):
    st.markdown("""
    **Answer**:
    By default, ChromaDB stores collection metadata, document mappings, and inverted index state in a standard SQLite file (`chroma.sqlite3`).
    * Anyone who gains access to the filesystem can open the file using `sqlite3 chroma.sqlite3` and view all document text in plaintext.
    * **SQLCipher** is an open-source extension that provides transparent, 256-bit AES encryption for SQLite database files.
    * Every page in the database is encrypted before writing to disk and decrypted on-the-fly in RAM using a secret passphrase supplied during database initialization.
    """)

with st.expander("Q5: What are the latency and accuracy trade-offs of PII redaction pipelines?"):
    st.markdown("""
    **Answer**:
    * **Latency**: Rule-based Regex scrubbers add <0.5ms per document. Transformer-based Named Entity Recognition (NER) models (like Microsoft Presidio / Spacy) add 10ms–30ms per document during offline ETL ingestion.
    * **Accuracy**: Replacing specific names with `<PERSON_REDACTED>` may slightly degrade the embedding model's ability to differentiate between two people in the same paragraph. 
    * **Solution**: Use **Pseudonym Masking** (`Customer_1` vs `Customer_2`), which preserves structural distinction while hiding real identity.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Believing Embeddings Cannot Be Inverted**  
    Assuming floating point vectors are secure hashes allows unauthorized users to exfiltrate private text via inversion decoders. Encrypt all vector volumes.
    """)
    st.warning("""
    **Trap 2: Redacting PII at Prompt Construction Instead of Ingestion**  
    Filtering PII only before sending to the LLM leaves raw unencrypted PII permanently indexed in the vector database. Filter before embedding.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Forgetting Backup Volume Encryption**  
    Encrypting the live NVMe disk but taking unencrypted backups or S3 snapshots leaves corporate IP exposed in backup buckets. Enforce KMS encryption on all snapshots.
    """)
    st.warning("""
    **Trap 4: Incomplete PII Regex Patterns**  
    Using a naive regex that misses international phone numbers or formatted account numbers leaves partial secrets in the database. Use production NER suites like Presidio.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Vector Inversion Attack?"):
        st.markdown("""
        **Reversing Vectors**:  
        Using trained decoder models to reconstruct original natural language text directly from dense vector embeddings.
        """)
    with st.expander("🗂️ Flashcard: When should PII scrubbing happen?"):
        st.markdown("""
        **Pre-Ingestion**:  
        Must execute *before* calling the embedding model to prevent baking secrets into vector space.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is SQLCipher?"):
        st.markdown("""
        **Encrypted SQLite**:  
        Transparent 256-bit AES full-database encryption extension for SQLite files at rest.
        """)
    with st.expander("🗂️ Flashcard: What is Pseudonym Masking?"):
        st.markdown("""
        **Synthetic Mapping**:  
        Replaces real names with synthetic IDs (e.g. Customer_42) to preserve syntax while ensuring privacy.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why vector embeddings are sensitive data vulnerable to Vector Inversion Attacks.", value=True)
chk2 = st.checkbox("I know why automated PII scrubbing must occur *before* vector embedding generation.", value=True)
chk3 = st.checkbox("I can implement pre-ingestion PII sanitization pipelines using Regex and Named Entity Recognition.", value=True)
chk4 = st.checkbox("I understand dual-layer encryption: TLS 1.3 in transit and AES-256 / SQLCipher at rest.", value=True)
chk5 = st.checkbox("I know how to execute compliant data deletion under GDPR Right-to-be-Forgotten.", value=True)
chk6 = st.checkbox("I can architect a privacy-first, compliance-certified vector database storage system.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Vector Data Security and are ready for Chapter 20.3: Retrieval Security & Multi-Tenant Filtering!")
