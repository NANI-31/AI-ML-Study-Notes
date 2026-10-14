import streamlit as st
import pandas as pd
import json
import hashlib
import time

st.set_page_config(
    page_title="4.3 Document Metadata - RAG Master Curriculum",
    page_icon="🏷️",
    layout="wide"
)

# Title & Overview
st.title("4.3 Document Metadata: Filtering, Provenance, Temporal Tracking, & Permission Scopes")
st.caption("Chapter 4: Documents and Data Sources | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
In naive RAG implementations, developers focus entirely on the text content (`page_content`) and treat 
**Document Metadata** as an afterthought. 
In enterprise production, however, **metadata is what makes retrieval accurate, compliant, and secure**. 
Without metadata, a system cannot filter by publication date (retrieving obsolete 2018 policies instead of current 2024 rules), 
cannot enforce security permissions (allowing junior employees to retrieve board-level compensation memos), 
and cannot provide auditable source citations (page numbers, section headings, and verified URLs).
This lesson explores the enterprise metadata taxonomy, payload indexing, temporal filtering, and permission scoping.
""")

st.divider()

# Section 1: The Enterprise Metadata Taxonomy
st.header("1. The Enterprise Metadata Taxonomy", divider="rainbow")
st.markdown("""
Production Document objects attach structured metadata across five critical dimensions:
""")

metadata_table = pd.DataFrame([
    {
        "Metadata Dimension": "1. Provenance & Attribution",
        "Key Fields": "`source_url`, `file_name`, `page_number`, `chunk_id`, `doc_hash`",
        "Primary Operational Role": "Generates exact user-facing citations; enables document deduplication via SHA-256 hashes."
    },
    {
        "Metadata Dimension": "2. Temporal & Recency Tracking",
        "Key Fields": "`created_at`, `updated_at`, `effective_year`, `expires_at`",
        "Primary Operational Role": "Allows time-travel queries ('Show 2023 10-K') and filters out expired documents."
    },
    {
        "Metadata Dimension": "3. Security & Multitenancy (RBAC)",
        "Key Fields": "`tenant_id`, `department`, `allowed_roles`, `confidentiality_level`",
        "Primary Operational Role": "Enforces Pre-Filtering so users can only retrieve chunks matching their security clearance."
    },
    {
        "Metadata Dimension": "4. Structural & Hierarchical Context",
        "Key Fields": "`section_title`, `header_path`, `parent_doc_id`, `token_count`",
        "Primary Operational Role": "Preserves document hierarchy ('Handbook > Benefits > Healthcare') for contextual retrieval."
    },
    {
        "Metadata Dimension": "5. Business & Domain Tags",
        "Key Fields": "`product_sku`, `customer_segment`, `region`, `language`",
        "Primary Operational Role": "Enables faceted search ('Only search region: EMEA and language: German')."
    }
])

st.dataframe(metadata_table, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Why Vector Search Needs Metadata Filters")
    st.markdown("""
    * **The Temporal Blindspot**: Dense embedding vectors represent *concepts*, not exact calendar timestamps. A question like *"What is our travel meal budget for 2024?"* is semantically identical to the 2019 policy! Without metadata filtering on `effective_year == 2024`, the vector search will often retrieve the older policy.
    * **Multi-Tenant Security**: In a B2B SaaS application, Customer A's vectors and Customer B's vectors often reside in the same physical database collection. A mandatory metadata filter (`tenant_id == active_user.tenant_id`) is the ONLY barrier preventing catastrophic cross-tenant data leakage!
    """)

with col2:
    st.subheader("Metadata-Filtered Query Topology")
    meta_chart = """
    digraph MetadataFilter {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin=0.15];
        edge [fontname="Helvetica", fontsize=9];

        User [label="User Query + User Context\\n(Role: 'HR', Year: 2024, Tenant: 'Acme')", fillcolor="#E0E7FF", color="#4338CA"];
        Filter [label="Boolean Metadata Pre-Filter\\n(tenant = 'Acme' AND allowed_role = 'HR' AND year = 2024)", fillcolor="#EDE9FE", color="#6D28D9"];
        VectorDB [label="Enterprise Vector Store\\n(Evaluates Filter BEFORE ANN Search)", fillcolor="#DCFCE7", color="#16A34A"];
        FilteredChunks [label="Permitted Recent Gold Chunks", fillcolor="#E0F2FE", color="#0284C7"];
        Synthesis [label="LLM Answer with Exact Page Citations", fillcolor="#D1FAE5", color="#059669"];

        User -> Filter;
        Filter -> VectorDB;
        VectorDB -> FilteredChunks [label="HNSW Cosine Match"];
        FilteredChunks -> Synthesis;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(meta_chart, use_container_width=False)

st.divider()

# Section 2: Production Code: Metadata Enrichment & Filtered Retrieval
st.header("2. Production Code: Metadata Enrichment & Filtered Search", divider="rainbow")
st.markdown("""
Here is a complete, runnable Python engine that enriches raw documents with cryptographic provenance and executes **boolean metadata-filtered semantic search**:
""")

st.code("""
import hashlib
import time
from typing import List, Dict, Any, Optional

class EnrichedDocument:
    def __init__(self, content: str, source_file: str, page: int, year: int, department: str, roles: List[str]):
        self.page_content = content
        # Cryptographic provenance hash
        self.doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        self.metadata = {
            "source": source_file,
            "page": page,
            "year": year,
            "department": department,
            "allowed_roles": roles,
            "doc_hash": self.doc_hash,
            "created_at": time.time()
        }

    def __repr__(self):
        return f"EnrichedDoc(hash={self.doc_hash}, page={self.metadata['page']}, dept={self.metadata['department']})"

class FilteredMetadataVectorEngine:
    def __init__(self):
        self.documents: List[EnrichedDocument] = []

    def add_document(self, doc: EnrichedDocument):
        self.documents.append(doc)

    def search_with_metadata_filter(self, query: str, user_role: str, target_year: int) -> List[Dict[str, Any]]:
        \"\"\"
        Applies strict boolean metadata pre-filtering before semantic scoring.
        \"\"\"
        results = []
        for doc in self.documents:
            meta = doc.metadata
            
            # Filter 1: Temporal filter
            if meta["year"] != target_year:
                continue
                
            # Filter 2: Role-based security filter
            if user_role not in meta["allowed_roles"] and "Public" not in meta["allowed_roles"]:
                continue
                
            # Simulated keyword score
            score = 1.0 if any(term in doc.page_content.lower() for term in query.lower().split()) else 0.5
            results.append({
                "content": doc.page_content,
                "citation": f"{meta['source']} (Page {meta['page']}, Year {meta['year']})",
                "score": score
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# Demonstration
engine = FilteredMetadataVectorEngine()

# Ingest historical & confidential docs
engine.add_document(EnrichedDocument(
    content="2023 Parental Leave Policy: 8 weeks unpaid leave.",
    source_file="benefits_2023.pdf", page=4, year=2023, department="HR", roles=["All"]
))
engine.add_document(EnrichedDocument(
    content="2024 Parental Leave Policy: 16 weeks 100% paid leave for all full-time staff.",
    source_file="benefits_2024.pdf", page=6, year=2024, department="HR", roles=["All"]
))
engine.add_document(EnrichedDocument(
    content="2024 Executive Bonus Formula: 40% target based on EBITDA margins.",
    source_file="comp_2024.pdf", page=1, year=2024, department="Executive", roles=["Executive"]
))

# Test Query: Employee asking for 2024 policy
query = "What is the parental leave duration?"
matches = engine.search_with_metadata_filter(query, user_role="Employee", target_year=2024)

print("Filtered Search Results for Employee (Year=2024):")
for m in matches:
    print(f"- {m['content']}")
    print(f"  Citation: {m['citation']}")
""", language="python")

st.divider()

# Section 3: Interactive Metadata Filter & Citation Playground
st.header("3. Interactive Metadata Filter & Citation Playground", divider="rainbow")
st.markdown("""
Test how metadata filters isolate the candidate pool and guarantee current, permitted citations:
""")

col_m1, col_m2 = st.columns(2)

with col_m1:
    user_clearance = st.selectbox("Current User Role:", ["Standard Employee", "HR Manager", "Executive"])
    filter_year = st.selectbox("Document Year Filter:", [2024, 2023])

with col_m2:
    dept_filter = st.selectbox("Department Category:", ["Human Resources", "Engineering", "Executive Board"])

test_documents = [
    {"title": "2024 HR Benefits Guide", "year": 2024, "dept": "Human Resources", "role": "Standard Employee", "citation": "HR Guide 2024, p. 12", "content": "16 weeks paid parental leave."},
    {"title": "2023 HR Benefits Guide", "year": 2023, "dept": "Human Resources", "role": "Standard Employee", "citation": "HR Guide 2023, p. 9", "content": "8 weeks unpaid parental leave."},
    {"title": "2024 Executive Bonus Targets", "year": 2024, "dept": "Executive Board", "role": "Executive", "citation": "Board Memo 2024, p. 2", "content": "Bonus multiplier targets based on net operating profit."},
    {"title": "2024 Eng On-Call Handbook", "year": 2024, "dept": "Engineering", "role": "Standard Employee", "citation": "Eng Wiki 2024, p. 1", "content": "Weekly on-call rotation with $500 weekly stipend."}
]

# Apply filter
allowed = []
for d in test_documents:
    if d["year"] == filter_year and d["dept"] == dept_filter:
        if d["role"] == "Standard Employee" or user_clearance == "Executive" or (d["role"] == "HR Manager" and user_clearance in ["HR Manager", "Executive"]):
            allowed.append(d)

st.subheader("Filtered Ingestion Results")
if allowed:
    for item in allowed:
        st.success(f"✅ **Retrieved Match**: *{item['content']}*")
        st.caption(f"Verifiable Citation: `{item['citation']}`")
else:
    st.error("🔒 **Zero Chunks Permitted**: No documents matched your specific Year, Department, and Role clearance filters.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Metadata Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is metadata filtering essential for temporal queries in RAG?**
    * **Answer**: Dense vector representations encode linguistic semantic meaning, but cannot reliably distinguish dates or temporal boundaries. A 2020 document stating *"The company 401(k) match is 3%"* and a 2024 document stating *"The company 401(k) match is 5%"* have near-identical vector embeddings. Without a metadata filter restricting the query to `year == 2024` or `is_active == True`, the vector search will often retrieve the older policy due to minor keyword matching variances.
    
    **Q2: How does Payload Indexing work in vector databases like Qdrant and Pinecone?**
    * **Answer**: When executing metadata-filtered vector searches across millions of documents, scanning every vector's metadata linearly creates massive latency bottlenecks. **Payload Indexing** builds secondary inverted indexes (B-trees, hash maps) over specific metadata keys (like `tenant_id` or `timestamp`). The vector database uses the payload index to instantly narrow the HNSW search space to permitted vectors *before* computing distance algorithms, maintaining sub-10ms latency.
    
    **Q3: What is the risk of excessive metadata payload bloat?**
    * **Answer**: Storing massive payloads (e.g. embedding 500-token chunks but attaching 10KB of raw uncompressed JSON metadata per chunk) balloons vector database RAM requirements. Because vector databases keep active indexes and payloads in memory or fast SSD caches for real-time latency, payload bloat can increase infrastructure hosting costs 3x to 5x. Best practice is to store only queryable filter keys and provenance IDs in metadata, fetching full raw text from primary object storage (S3) when needed.
    """)

with traps_tab:
    st.error("""
    **Trap 1: Filtering metadata in application code after vector search (Post-Filtering)**  
    If an unauthorized user asks for executive compensation, the top 10 vector matches will all be executive documents. If you post-filter them out in Python, the user receives 0 results! Always apply metadata **Pre-Filtering** directly in the vector database query.
    """)
    st.error("""
    **Trap 2: Forgetting to normalize metadata string types**  
    Storing `year: "2024"` as a string in some chunks and `year: 2024` as an integer in others causes equality filters (`year == 2024`) to fail silently on half your database. Enforce strict Pydantic schemas on all metadata payloads during ingestion!
    """)

with flashcards_tab:
    st.info("**Payload Pre-Filtering**: The process of applying boolean metadata constraints (tenant, role, date) directly inside the vector engine prior to distance calculation.")
    st.info("**Provenance Hash**: A cryptographic hash (SHA-256) of document content used to identify identical chunks, track revisions, and prevent redundant re-indexing.")
    st.info("**Payload Indexing**: Creating secondary search indexes over metadata attributes in vector databases to ensure sub-millisecond filtering at massive scale.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
checklist = [
    "I understand the five core dimensions of enterprise document metadata.",
    "I know why temporal queries require metadata filtering to prevent obsolete retrieval.",
    "I can explain how metadata Pre-Filtering enforces multi-tenant security and RBAC.",
    "I understand the role of Payload Indexing in vector database performance.",
    "I can implement a cryptographic provenance and metadata enrichment pipeline in Python."
]

for item in checklist:
    st.checkbox(item, value=False)
