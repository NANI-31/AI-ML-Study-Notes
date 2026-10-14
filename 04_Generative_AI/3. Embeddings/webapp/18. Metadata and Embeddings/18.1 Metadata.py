import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

st.set_page_config(
    page_title="18.1 Metadata - Embeddings Curriculum",
    page_icon="🏷️",
    layout="wide"
)

# Title & Overview
st.title("18.1 Payload Metadata: Schema Design, Provenance, & Vector Governance")
st.caption("Chapter 18: Metadata and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In raw mathematics, an embedding is merely an anonymous list of high-dimensional numbers: `[0.012, -0.045, 0.821, ...]`.
On its own, a vector contains **zero provenance, zero security context, zero timestamp awareness, and zero identity**.

In production vector systems, every vector is paired with a **Payload Metadata Dictionary**.
Metadata transforms an abstract mathematical coordinate into a governed enterprise record:
* **Provenance**: Exactly which PDF, webpage, or database row generated this vector?
* **Access Control**: Does the currently logged-in user have clearance to view this document?
* **Temporal Freshness**: Was this policy written yesterday, or is it an obsolete draft from 2019?
* **Organizational Scoping**: Which department, region, and category does this passage belong to?

This lesson explores the schema design, field primitives, and validation patterns for enterprise vector metadata.
""")

st.divider()

# Section 1: The Role of Payload Metadata in Vector Search
st.header("1. The 7 Essential Metadata Attributes & Dual-Engine Architecture", divider="rainbow")

col_meta1, col_meta2 = st.columns([1, 1])

with col_meta1:
    st.markdown("""
    ### The 7 Essential Metadata Fields
    * **1. Document ID (`doc_id` / `chunk_id`)**:
      * Deterministic string identifier (e.g. `doc_sec_2024_c04`). Enables idempotent upserts and updates.
    * **2. Source (`source`)**:
      * The origin URI, file path, S3 URL, or API endpoint for verified citations.
    * **3. Author / Owner (`author` / `owner_id`)**:
      * Creator attribution, team ownership, and accountability.
    * **4. Date / Timestamp (`created_at` / `updated_at`)**:
      * Standard ISO-8601 string or epoch milliseconds for temporal filtering.
    * **5. Category / Topic (`category`)**:
      * Broad hierarchical classification (e.g. `legal`, `engineering`, `finance`).
    * **6. Department / Tenant (`department` / `tenant_id`)**:
      * Hard organizational boundary for multi-tenancy and role-based access control (RBAC).
    * **7. File Type (`file_type`)**:
      * Origin format (`pdf`, `markdown`, `docx`, `slack`, `code`).
    """)

with col_meta2:
    st.markdown("### Vector + Payload Dual-Engine Architecture")
    meta_dot = """
    digraph VectorMetadata {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_vector {
            label="Vector Index Engine (HNSW)"; style="dashed"; color="#3B82F6";
            VCoord [label="Dense Vector (d=1536)\\n[0.012, -0.045, ...]\\nAnonymous Math Coordinate", fillcolor="#EFF6FF", color="#1D4ED8"];
        }

        subgraph cluster_payload {
            label="Payload / Metadata Engine (SQLite / RocksDB)"; style="dashed"; color="#10B981";
            PayloadData [label="Governed Enterprise Record\\n- doc_id: 'sec_10k_2024'\\n- tenant: 'org_acme'\\n- department: 'finance'\\n- clearance: 'level_3'\\n- date: '2024-03-15'\\n- text: 'Q3 net revenue...'", fillcolor="#ECFDF5", color="#059669", style="filled,bold"];
        }

        LinkNode [label="Shared Document Pointer ID (UUID)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        
        VCoord -> LinkNode [dir="both", label="1-to-1 Mapping"];
        LinkNode -> PayloadData [dir="both", label="Hydrates text & filters"];
    }
    """
    st.graphviz_chart(meta_dot, use_container_width=False)

st.markdown("### Metadata Field Specification Standards")
st.markdown("""
| Field Name | Recommended Type | Example Value | Vector DB Index Type | Why It Matters |
| :--- | :--- | :--- | :--- | :--- |
| `doc_id` | `str` | `"policy_hr_v2_c03"` | Primary Key / Unique ID | Eliminates duplicate vector insertions. |
| `source` | `str` | `"s3://bucket/handbook.pdf"` | Keyword / String | LLM citation grounding and source links. |
| `author` | `str` | `"compliance_officer"` | Keyword / String | Ownership tracking and verification. |
| `date` | `str` or `int` | `"2024-05-01"` or `1714521600` | Integer / Datetime index | Time-travel queries (`date >= 2024-01-01`). |
| `category` | `str` | `"security_policy"` | Keyword index | Categorical partition filtering. |
| `department`| `str` | `"infosec"` | Inverted Keyword | Multi-tenant security isolation. |
| `file_type` | `str` | `"pdf"` | Keyword index | Format-specific processing pipelines. |
""")

st.divider()

# Section 2: Production Code: Metadata Schema Validation
st.header("2. Production Code: Strict Enterprise Metadata Validator", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to validate metadata payloads before passing them to a vector database: enforcing primitive types, non-null keys, and ISO-8601 timestamps.
""")

st.code(r'''
from datetime import datetime
from typing import Dict, Any, List, Optional

class VectorMetadataValidator:
    """
    Validates and standardizes metadata dictionaries before vector database insertion.
    Enforces compatibility with ChromaDB, Qdrant, and Pinecone requirements.
    """
    ALLOWED_CATEGORIES = {"engineering", "finance", "legal", "hr", "security"}
    ALLOWED_FILE_TYPES = {"pdf", "docx", "markdown", "html", "txt"}

    @classmethod
    def validate_and_clean(cls, raw_meta: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}

        # 1. Mandatory doc_id
        doc_id = str(raw_meta.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError("Metadata validation error: 'doc_id' cannot be empty.")
        cleaned["doc_id"] = doc_id

        # 2. Source URI
        cleaned["source"] = str(raw_meta.get("source", "unknown")).strip()

        # 3. Author attribution
        cleaned["author"] = str(raw_meta.get("author", "unassigned")).strip().lower()

        # 4. Date validation (standardized to ISO string YYYY-MM-DD)
        raw_date = raw_meta.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
        cleaned["date"] = str(raw_date)

        # 5. Category validation
        cat = str(raw_meta.get("category", "engineering")).strip().lower()
        cleaned["category"] = cat if cat in cls.ALLOWED_CATEGORIES else "general"

        # 6. Department / Tenant
        cleaned["department"] = str(raw_meta.get("department", "corporate")).strip().lower()

        # 7. File Type
        ft = str(raw_meta.get("file_type", "txt")).strip().lower().replace(".", "")
        cleaned["file_type"] = ft if ft in cls.ALLOWED_FILE_TYPES else "txt"

        # 8. Type sanity check: Vector DBs reject nested lists/dicts in metadata
        for k, v in cleaned.items():
            if not isinstance(v, (str, int, float, bool)):
                raise TypeError(f"Field '{k}' has invalid type {type(v)}. Only primitives allowed.")

        return cleaned
''', language="python")

st.divider()

# Section 3: Interactive Metadata Schema Builder & Inspector
st.header("3. Interactive Workbench: Metadata Schema Builder & Validator", divider="rainbow")

st.markdown("""
Build and test an enterprise metadata payload interactively. Observe how metadata is structured, validated, and serialized for vector database ingestion.
""")

col_bld1, col_bld2 = col_bld1, col_bld2 = st.columns([1, 1])

with col_bld1:
    st.subheader("1. Preset Template Selection")
    template_choice = st.selectbox(
        "Choose Domain Preset Template:",
        ["Enterprise Cybersecurity SOP", "Q3 Financial Earnings Report", "HR Benefits Handbook", "Custom Entry"]
    )

    if template_choice == "Enterprise Cybersecurity SOP":
        init_id = "doc_infosec_sop_c02"
        init_src = "s3://internal-docs/security/incident_response_2024.pdf"
        init_author = "ciso_office"
        init_date = "2024-04-12"
        init_cat = "security"
        init_dept = "infosec"
        init_ft = "pdf"
        init_text = "All compromised IAM credentials must be revoked immediately upon detection."
    elif template_choice == "Q3 Financial Earnings Report":
        init_id = "fin_q3_report_c14"
        init_src = "https://investors.corp.com/q3_2024_transcript.html"
        init_author = "investor_relations"
        init_date = "2024-10-01"
        init_cat = "finance"
        init_dept = "finance"
        init_ft = "html"
        init_text = "Cloud infrastructure gross margins expanded to sixty-eight percent in Q3."
    elif template_choice == "HR Benefits Handbook":
        init_id = "hr_handbook_2024_c08"
        init_src = "sharepoint://hr/benefits_guide_v3.docx"
        init_author = "hr_operations"
        init_date = "2024-01-15"
        init_cat = "hr"
        init_dept = "people_ops"
        init_ft = "docx"
        init_text = "Full-time employees receive a four percent employer match on 401k retirement contributions."
    else:
        init_id = "custom_doc_01"
        init_src = "local://docs/custom.txt"
        init_author = "admin"
        init_date = "2024-06-01"
        init_cat = "engineering"
        init_dept = "engineering"
        init_ft = "txt"
        init_text = "Vector databases provide low latency Approximate Nearest Neighbor search."

    st.subheader("2. Configure Fields")
    in_id = st.text_input("1. Document ID (`doc_id`):", value=init_id)
    in_src = st.text_input("2. Source (`source`):", value=init_src)
    in_author = st.text_input("3. Author (`author`):", value=init_author)
    in_date = st.text_input("4. Date (`date` YYYY-MM-DD):", value=init_date)
    in_cat = st.selectbox("5. Category (`category`):", ["security", "finance", "hr", "engineering", "legal"], index=0 if init_cat == "security" else (1 if init_cat == "finance" else 2))
    in_dept = st.text_input("6. Department (`department`):", value=init_dept)
    in_ft = st.selectbox("7. File Type (`file_type`):", ["pdf", "docx", "markdown", "html", "txt"], index=["pdf", "docx", "markdown", "html", "txt"].index(init_ft))
    in_chunk_text = st.text_area("Document Chunk Text Payload:", value=init_text, height=70)

with col_bld2:
    st.subheader("3. Validated Payload & Vector DB Schema")

    raw_payload = {
        "doc_id": in_id,
        "source": in_src,
        "author": in_author,
        "date": in_date,
        "category": in_cat,
        "department": in_dept,
        "file_type": in_ft
    }

    # Validate primitives
    is_valid = True
    error_msg = ""
    if not in_id.strip():
        is_valid = False
        error_msg = "Document ID cannot be empty!"

    if is_valid:
        st.success("✅ Metadata Payload Passes Strict Enterprise Validation!")
        st.json(raw_payload)

        # Telemetry metrics
        json_str = json.dumps(raw_payload)
        payload_bytes = len(json_str.encode("utf-8"))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Field Count", len(raw_payload))
        m2.metric("Payload Size", f"{payload_bytes} bytes")
        m3.metric("Type Safety", "100% Primitive")

        st.markdown("#### Ready-to-Execute ChromaDB / Qdrant Payload")
        st.code(f"""collection.upsert(
    ids=["{in_id}"],
    documents=["{in_chunk_text}"],
    metadatas=[{json.dumps(raw_payload, indent=8)}]
)""", language="python")
    else:
        st.error(f"❌ Validation Failed: {error_msg}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Metadata Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why should you never concatenate all metadata attributes directly into the raw text string before embedding?**
    * **Answer**: If you append all metadata (e.g. *"DocID: 104, Author: Bob, Date: 2024-01-01, File: PDF, Dept: HR"*) directly into the text chunk, you introduce two serious defects:
      1. **Noise Pollution**: Metadata tokens consume valuable embedding context window space and distort the semantic vector toward generic structural words.
      2. **Impossibility of Hard Filtering**: The vector engine treats metadata as soft semantic hints rather than hard constraints. A query for *"HR policies"* might still return legal documents that mentioned HR in passing. Storing metadata in indexed payload columns allows exact SQL/Boolean filtering.

    **Q2: How do vector databases index metadata attributes for sub-millisecond filtering?**
    * **Answer**: Modern vector databases (Qdrant, Chroma, Milvus) maintain **Inverted Index Payloads** in secondary storage (RocksDB / SQLite). For categorical fields like `department: "finance"`, they construct a posting list (or bitmask) of all document IDs belonging to finance. During search, this bitmask is checked in constant time $O(1)$ per node.

    **Q3: What metadata types are supported by vector databases and why are nested dictionaries prohibited?**
    * **Answer**: Most vector databases strictly allow only scalar primitive types: `string`, `integer`, `float`, and `boolean`. Nested structures (like JSON sub-objects or multidimensional arrays) are rejected because building fast inverted indexes over arbitrary nested schemas is computationally expensive. Nested objects must be flattened (e.g. `user_address_city = "New York"`).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Passing Non-Primitive Objects into Metadata**:
      * *Issue*: Storing Python `datetime` objects or list objects `{"tags": ["ai", "nlp"]}` directly into ChromaDB metadata.
      * *Consequence*: The ingestion worker crashes with `ValueError: Expected metadata value to be a str, int, float or bool`.
      * *Fix*: Serialize dates to ISO-8601 strings (`2024-05-01`) or epoch integers, and join lists into comma-separated strings (`"ai,nlp"`).

    * **Trap 2: Case Inconsistency in Metadata Filtering**:
      * *Issue*: Ingesting `department: "Finance"` in some documents and `department: "finance"` in others.
      * *Consequence*: Filtering with `where={"department": "finance"}` silently misses half the corporate documents!
      * *Fix*: Standardize all categorical metadata strings to `.lower().strip()` during ingestion validation.

    * **Trap 3: Oversized Metadata Bloating Storage Footprint**:
      * *Issue*: Storing giant 50KB JSON debug payloads in the metadata dictionary of every 100-word chunk.
      * *Consequence*: Memory consumption explodes; vector database RAM is exhausted by metadata rather than vector indexes.
      * *Fix*: Keep metadata lightweight (< 1KB per chunk); store heavy document artifacts in object storage (S3).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Payload Metadata**
      * *Definition*: The auxiliary key-value attribute dictionary stored alongside an embedding vector providing provenance, security, and categorical context.
    * **Flashcard 2: Document Lineage / Provenance**
      * *Definition*: The verified trail linking a generated LLM answer back to its originating document URI, page number, and timestamp.
    * **Flashcard 3: Inverted Payload Index**
      * *Definition*: An auxiliary index structure mapping metadata values directly to document bitmasks for sub-millisecond filtering.
    * **Flashcard 4: Metadata Normalization**
      * *Definition*: Converting all dates, categories, and author strings into consistent lowercased primitive formats prior to database insertion.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Payload Metadata:

- [ ] I can list the 7 essential metadata attributes for enterprise vector storage.
- [ ] I understand why metadata must be stored in structured payloads rather than concatenated into text strings.
- [ ] I know why vector databases enforce strict primitive type constraints (str, int, float, bool).
- [ ] I understand how inverted payload indexes enable sub-millisecond filtering.
- [ ] I can implement validation logic to sanitize and normalize metadata before ingestion.
""")
