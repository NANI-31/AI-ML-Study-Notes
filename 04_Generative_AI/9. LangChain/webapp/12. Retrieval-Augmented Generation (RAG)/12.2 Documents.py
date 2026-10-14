import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class DocumentEntity:
    """Represents a standardized LangChain Document schema."""
    def __init__(self, page_content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = doc_id

class MockDocumentIngestionStudio:
    """Simulates enterprise document creation, metadata filtering, deduplication, and transformations."""
    def __init__(self):
        self.raw_corpus = [
            {
                "title": "Q1 2026 Earnings Report",
                "raw_text": "<p>Total revenue reached <strong>14.2B USD</strong>, beating expectations by 8.4%.</p><!-- footer -->",
                "department": "Finance",
                "classification": "Confidential",
                "author": "CFO Office",
                "date": "2026-04-02"
            },
            {
                "title": "Employee Healthcare Benefits Guide",
                "raw_text": "<div>Comprehensive medical coverage includes <em>dental and optical</em> with zero copay for in-network clinics.</div>",
                "department": "HR",
                "classification": "Internal",
                "author": "Benefits Team",
                "date": "2026-01-10"
            },
            {
                "title": "Production Database Backup Protocol",
                "raw_text": "<pre>Automated snapshots run daily at 02:00 UTC with 30-day retention across S3 multi-region buckets.</pre>",
                "department": "Engineering",
                "classification": "Confidential",
                "author": "DevOps",
                "date": "2025-12-15"
            },
            {
                "title": "Public Press Release: AI Innovation Hub",
                "raw_text": "<h1>We are expanding our open research facility</h1> into Tokyo and London to foster academic collaborations.",
                "department": "Marketing",
                "classification": "Public",
                "author": "PR Team",
                "date": "2026-02-20"
            }
        ]

    def _generate_doc_id(self, title: str, dept: str) -> str:
        # Deterministic synthetic UUID generator
        clean_key = f"{dept.lower()}-{title.lower().replace(' ', '-')}"
        return f"doc-uuid-{abs(hash(clean_key)) % 1000000:06d}"

    def _strip_html(self, text: str) -> str:
        for tag in ["<p>", "</p>", "<strong>", "</strong>", "<!-- footer -->", "<div>", "</div>", "<em>", "</em>", "<pre>", "</pre>", "<h1>", "</h1>"]:
            text = text.replace(tag, "")
        return text.strip()

    def process(self, selected_dept: str, max_classification: str, clean_html: bool, enrich_stats: bool) -> Dict[str, Any]:
        docs: List[DocumentEntity] = []
        rejected_count = 0
        total_tokens = 0

        classification_hierarchy = {"Public": 1, "Internal": 2, "Confidential": 3}
        max_level = classification_hierarchy.get(max_classification, 3)

        for item in self.raw_corpus:
            # 1. Metadata Authorization Filtering
            item_level = classification_hierarchy.get(item["classification"], 3)
            if item_level > max_level:
                rejected_count += 1
                continue

            if selected_dept != "All" and item["department"] != selected_dept:
                rejected_count += 1
                continue

            # 2. Content Transformation
            content = self._strip_html(item["raw_text"]) if clean_html else item["raw_text"]

            # 3. Metadata Enrichment
            meta = {
                "source": item["title"],
                "department": item["department"],
                "classification": item["classification"],
                "author": item["author"],
                "date": item["date"]
            }

            word_count = len(content.split())
            token_estimate = word_count * 4
            total_tokens += token_estimate

            if enrich_stats:
                meta["word_count"] = word_count
                meta["token_estimate"] = token_estimate
                meta["char_length"] = len(content)

            # 4. Deterministic ID Assignment
            doc_id = self._generate_doc_id(item["title"], item["department"])
            docs.append(DocumentEntity(page_content=content, metadata=meta, doc_id=doc_id))

        return {
            "documents": docs,
            "accepted_count": len(docs),
            "rejected_count": rejected_count,
            "total_tokens": total_tokens
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="12.2 Documents",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">12.2 Documents</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">The core unit of LangChain knowledge: Document schema, metadata filtering, deterministic UUIDs, loading patterns, and transformation pipelines.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The LangChain Document Ingestion Architecture", divider="rainbow")

st.markdown("""
In LangChain, all text ingested for indexing or retrieved for prompt augmentation is standardized as a **`Document`** object.
The Document object is the universal lingua franca across loaders, splitters, vector stores, and retrievers, 
binding unstructured prose with structured organizational metadata.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DocumentArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_raw_sources {
            label="1. Ingest Sources";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            RawFile [label="Raw Text Source\\n('Q1 Revenue was 14.2B USD...')", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            RawMeta [label="Source Context\\n(FilePath, Author, Timestamp, ACLs)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_doc_entity {
            label="2. LangChain Document Primitive";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            DocEntity [label="Document Schema\\n• page_content: str\\n• metadata: dict (flat key-values)\\n• id: Optional[str] (UUID)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
        }

        subgraph cluster_transformation {
            label="3. Document Transformation Pipeline";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            Cleaner [label="HTML & Whitespace Stripper\\n(Sanitize raw markup tags)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            Enricher [label="Metadata Enricher\\n(Token counts, summary tags, tenant_id)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            IdAssigner [label="Deterministic UUID Hash\\n(RecordManager deduplication)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_downstream {
            label="4. Downstream Consumers";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            VectorDB [label="Vector Store Indexing\\n(Embedding text + metadata filter)", shape=cylinder, fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            PromptContext [label="Augmented Prompt Context\\n(Injected into LLM with citations)", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
        }

        RawFile -> DocEntity [label="1. Wrap Text"];
        RawMeta -> DocEntity [label="2. Attach Metadata"];
        DocEntity -> Cleaner [label="3. Pipeline Ingress"];
        Cleaner -> Enricher [label="4. Enriched Attributes"];
        Enricher -> IdAssigner [label="5. Unique Hash Key"];
        IdAssigner -> VectorDB [label="6. Index in DB"];
        IdAssigner -> PromptContext [label="7. Ground LLM"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: DOCUMENT OBJECTS ---
st.header("2. Document Objects", divider="rainbow")

st.markdown("""
In `langchain_core.documents`, the fundamental knowledge primitive is the `Document` class.
Unlike simple strings, a Document binds text content directly to structured context:
""")

st.code("""from langchain_core.documents import Document

doc = Document(
    page_content="The enterprise annual vacation accrual rate is 20 business days.",
    metadata={
        "source": "handbook_2026.pdf",
        "page": 14,
        "department": "People Ops",
        "classification": "Internal"
    },
    id="doc_handbook_page_14"
)""", language="python")

doc_cols = st.columns(3)
with doc_cols[0]:
    st.subheader("📝 `page_content`")
    st.markdown("""
    The string payload containing the raw or processed natural language prose. 
    This is what is converted into a vector embedding and injected into LLM prompts.
    """)
with doc_cols[1]:
    st.subheader("🏷️ `metadata`")
    st.markdown("""
    A standard Python dictionary holding structured metadata key-value pairs. 
    Used by vector databases for hard pre-filtering and citation attribution.
    """)
with doc_cols[2]:
    st.subheader("🔑 `id`")
    st.markdown("""
    An optional unique string identifier for the document. 
    Crucial for idempotent batch updates and deduplication in vector databases.
    """)

st.divider()

# --- SECTION 3: DOCUMENT METADATA ---
st.header("3. Document Metadata", divider="rainbow")

st.markdown("""
Metadata is the backbone of production retrieval. Without rich metadata, RAG systems suffer from:
1. **Context Blindness**: The LLM receives an answer but cannot identify who wrote it or which document revision it comes from.
2. **Security Vulnerabilities**: Users can retrieve confidential documents across organizational boundaries.
3. **Noisy Search**: The vector search scans millions of irrelevant documents instead of scoping directly to the user's department.
""")

st.markdown("#### Enterprise Metadata Best Practices:")
meta_df = pd.DataFrame([
    {"Metadata Key": "source", "Data Type": "string", "Example Value": "s3://corporate-docs/handbook.pdf", "Purpose": "Verifiable source citations and URL links."},
    {"Metadata Key": "tenant_id", "Data Type": "string", "Example Value": "tenant-enterprise-alpha", "Purpose": "Hard multi-tenant isolation; prevents cross-company leakage."},
    {"Metadata Key": "department", "Data Type": "string", "Example Value": "Legal / Compliance", "Purpose": "Pre-filtering queries to domain-specific knowledge subsets."},
    {"Metadata Key": "classification", "Data Type": "string", "Example Value": "Confidential / Public", "Purpose": "Role-based access control (RBAC) enforcement at query time."},
    {"Metadata Key": "updated_at", "Data Type": "integer (Unix)", "Example Value": "1775088000", "Purpose": "Recency filtering and chronological sorting of legal policies."}
])
st.dataframe(meta_df, use_container_width=True)

st.divider()

# --- SECTION 4: DOCUMENT IDS ---
st.header("4. Document IDs", divider="rainbow")

st.markdown("""
When re-indexing a documentation site or document folder, what happens to existing embeddings in your vector store?
Without unique Document IDs, every re-ingestion **duplicates** existing records, diluting vector search quality and multiplying storage costs.

### Deterministic ID Generation:
Production systems generate deterministic IDs by hashing the document source and content:
""")

st.code("""import hashlib

def generate_deterministic_doc_id(source_path: str, chunk_index: int, content: str) -> str:
    \"\"\"Generates a reproducible SHA-256 hash ID for idempotent vector storage.\"\"\"
    hash_payload = f"{source_path}::{chunk_index}::{content}"
    return hashlib.sha256(hash_payload.encode('utf-8')).hexdigest()[:16]

doc_id = generate_deterministic_doc_id("handbook.pdf", 0, "The vacation accrual...")
# doc_id: 'a89c3b7f201e548d'""", language="python")

st.info("""
**LangChain RecordManager**: LangChain provides an indexing API (`langchain.indexes.SQLRecordManager`) 
that tracks document hashes in a relational database, automatically deleting obsolete vector chunks and updating modified files without re-embedding identical content.
""")

st.divider()

# --- SECTION 5: LOADING DOCUMENTS ---
st.header("5. Loading Documents", divider="rainbow")

st.markdown("""
LangChain provides two standardized methods across all document loaders:
- **`loader.load() -> List[Document]`**: Synchronously loads and reads the entire dataset into memory at once. Fast for small directories, but risks `OutOfMemoryError` on 50GB file archives.
- **`loader.lazy_load() -> Iterator[Document]`**: Yields documents one at a time as a Python generator. Allows streaming multi-gigabyte corpora directly into embedding models with minimal RAM footprint.
""")

st.code("""# Comparing Eager vs Lazy Loading
from langchain_community.document_loaders import TextLoader

loader = TextLoader("massive_server_logs.txt")

# Approach 1: Eager (High Memory Consumption)
# all_docs = loader.load()

# Approach 2: Lazy Generator (Constant Memory Consumption)
for document in loader.lazy_load():
    # Process and embed document chunk-by-chunk
    print(f"Loaded {len(document.page_content)} characters with 0MB RAM spikes.")""", language="python")

st.divider()

# --- SECTION 6: DOCUMENT TRANSFORMATIONS ---
st.header("6. Document Transformations", divider="rainbow")

st.markdown("""
Raw documents scraped from web pages, Confluence, or PDFs contain formatting artifacts: raw HTML markup, 
boilerplate navigation menus, excessive whitespace, and confidential customer data.

The `BaseDocumentTransformer` interface standardizes document preprocessing pipelines:
""")

st.code("""from typing import Sequence
from langchain_core.documents import Document, BaseDocumentTransformer

class HTMLStrippingTransformer(BaseDocumentTransformer):
    \"\"\"Transforms a sequence of documents by removing raw HTML tags.\"\"\"
    def transform_documents(self, documents: Sequence[Document], **kwargs) -> Sequence[Document]:
        transformed = []
        for doc in documents:
            clean_text = doc.page_content.replace("<p>", "").replace("</p>", "").strip()
            # Preserve original metadata while recording transformation history
            new_metadata = dict(doc.metadata)
            new_metadata["cleaned"] = True
            transformed.append(Document(page_content=clean_text, metadata=new_metadata, id=doc.id))
        return transformed""", language="python")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Document Pipeline & Record Manager", divider="rainbow")

st.markdown("""
The following complete Python program implements an end-to-end Document ingestion and transformation engine 
with deterministic UUID generation, metadata filtering, and content sanitization.
""")

st.code("""import hashlib
from typing import List, Dict, Any, Optional

class EnterpriseDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = doc_id or self._compute_hash()

    def _compute_hash(self) -> str:
        raw = f"{self.metadata.get('source', '')}::{self.page_content}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

class EnterpriseDocumentPipeline:
    \"\"\"Ingests, transforms, and dedupes documents for vector stores.\"\"\"
    def __init__(self):
        self.registry: Dict[str, EnterpriseDocument] = {}

    def ingest(self, raw_items: List[Dict[str, Any]], clean_markup: bool = True) -> Dict[str, Any]:
        added = 0
        skipped = 0

        for item in raw_items:
            content = item.get("text", "")
            if clean_markup:
                for tag in ["<div>", "</div>", "<p>", "</p>", "<b>", "</b>"]:
                    content = content.replace(tag, "")
                content = " ".join(content.split())

            meta = {
                "source": item.get("source", "unknown"),
                "department": item.get("department", "general"),
                "access_level": item.get("access_level", "internal"),
                "char_length": len(content)
            }

            doc = EnterpriseDocument(page_content=content, metadata=meta)

            # Deduplication Check
            if doc.id in self.registry:
                skipped += 1
            else:
                self.registry[doc.id] = doc
                added += 1

        return {
            "total_processed": len(raw_items),
            "added_to_index": added,
            "duplicate_skipped": skipped,
            "registry_size": len(self.registry)
        }

    def filter_by_permissions(self, user_clearance: str) -> List[EnterpriseDocument]:
        clearance_hierarchy = {"public": 1, "internal": 2, "confidential": 3}
        user_level = clearance_hierarchy.get(user_clearance.lower(), 1)

        permitted = []
        for doc in self.registry.values():
            doc_level = clearance_hierarchy.get(doc.metadata.get("access_level", "internal").lower(), 2)
            if doc_level <= user_level:
                permitted.append(doc)
        return permitted""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Document Ingestion, Metadata Filtering & Transformation Studio", divider="rainbow")

st.markdown("""
Test document parsing, metadata access control, and text transformations on a live corporate corpus. 
Adjust user clearance levels and department scopes to see how metadata filters control document visibility.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Pipeline & Permissions Configuration")
    dept_filter = st.selectbox("Department Scope Filter", options=["All", "Finance", "HR", "Engineering", "Marketing"])
    user_clearance = st.selectbox("Simulated User Security Clearance", options=["Public", "Internal", "Confidential"], index=1)

    st.markdown("#### Document Transformers:")
    t_clean_html = st.toggle("Enable HTML & Markup Stripping", value=True)
    t_stats = st.toggle("Enrich Metadata with Token & Word Statistics", value=True)

    btn_process = st.button("🚀 Process & Ingest Documents", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Ingested Document Registry")

    if btn_process:
        studio = MockDocumentIngestionStudio()
        res = studio.process(dept_filter, user_clearance, t_clean_html, t_stats)

        m1, m2, m3 = st.columns(3)
        m1.metric("Indexed Documents", res["accepted_count"])
        m2.metric("Filtered Out (ACL)", res["rejected_count"])
        m3.metric("Est. Total Tokens", res["total_tokens"])

        st.markdown("#### 📄 Ingested Document Entities:")
        if res["documents"]:
            for d in res["documents"]:
                with st.expander(f"📌 {d.metadata['source']} ({d.id})"):
                    st.write(f"**Content**: `{d.page_content}`")
                    st.json(d.metadata)
        else:
            st.warning("No documents match the current department filter and security clearance.")
    else:
        st.info("Configure pipeline options and click **Process & Ingest Documents** to inspect the Document objects.")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What are the two core attributes of a LangChain Document object, and why is metadata mandatory in enterprise RAG?",
     "A Document object contains page_content (the raw text payload) and metadata (a dictionary of associated attributes). Metadata is critical in enterprise systems because it powers access control filtering (e.g. scoping to allowed tenant_ids or departments), temporal sorting, and verifiable citations with page numbers and source URLs."),
    ("Why should document IDs be deterministic hashes rather than random UUID4s?",
     "Random UUID4s assign different IDs on every ingestion run. If a document is re-ingested during a scheduled cron sync, random IDs cause duplicate vector insertions in the database. Deterministic hashing (e.g. SHA-256 on source URL and content) guarantees that re-indexing identical text produces the exact same ID, enabling idempotent deduplication."),
    ("What is the difference between loader.load() and loader.lazy_load(), and when should you use each?",
     "loader.load() reads all documents into memory eagerly as a List[Document]. It is simple but causes OutOfMemory crashes on massive corpora. loader.lazy_load() returns an iterator/generator that yields Document objects one by one, allowing streaming chunking and embedding pipelines with constant memory usage (O(1) RAM)."),
    ("How do vector databases handle metadata pre-filtering versus post-filtering?",
     "Pre-filtering applies metadata constraints (e.g. WHERE department = 'Finance') before performing the approximate nearest neighbor vector search, guaranteeing all top-K results satisfy the filter. Post-filtering retrieves top-K vectors first and then discards non-matching metadata; if the nearest neighbors lack the required metadata, post-filtering returns fewer than K results or zero matches."),
    ("How does LangChain's RecordManager prevent orphaned chunks in vector databases?",
     "When an original PDF is updated from 10 pages to 8 pages, two old chunks become orphaned. RecordManager tracks all indexed document IDs in a relational database. When a document is re-indexed, it diffs current IDs against historical IDs and automatically sends hard DELETE commands to the vector store for obsolete chunk keys.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Storing Nested Dictionaries in Document Metadata. Many vector stores (such as Chroma or Pinecone) only support flat scalar metadata values (strings, integers, floats, booleans). Nested dicts or lists cause silent serialization errors or crashes during indexing. Always flatten metadata dictionaries.")
    st.error("**Trap 2 (Breaking Error)**: Omitting Source Metadata. Creating Document objects without recording the source file path or document title makes it impossible to cite sources in final generative responses, violating enterprise audit requirements.")
with trap_c2:
    st.warning("**Trap 3 (Memory Warning)**: Calling loader.load() on Massive Multi-Gigabyte Folders. Loading 50,000 PDFs using eager loading exhausts process RAM. Always utilize loader.lazy_load() for batch ingestion.")
    st.warning("**Trap 4 (Quality Warning)**: Embedding Uncleaned Raw HTML/Markdown. Ingesting raw HTML table tags, navigation bars, and footer disclaimers severely distorts embedding vectors. Always run document cleaners prior to vectorization.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Document Primitive"):
        st.markdown("**Core LangChain knowledge entity binding page_content string with metadata dictionary.**")
    with st.expander("🎴 Flashcard 2: Deterministic ID"):
        st.markdown("**Content and source hash enabling idempotent updates and zero vector duplications.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Lazy Loading"):
        st.markdown("**Generator-based loading yielding Document objects one-by-one to prevent OOM memory crashes.**")
    with st.expander("🎴 Flashcard 4: Metadata Pre-Filtering"):
        st.markdown("**Filters vector database by metadata attributes before executing nearest neighbor distance calculations.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the schema and role of the LangChain Document object in RAG pipelines.", value=True)
chk2 = st.checkbox("I know how to design flat, queryable metadata schemas for access control and citations.", value=True)
chk3 = st.checkbox("I can implement deterministic hash IDs to guarantee idempotent vector store indexing.", value=True)
chk4 = st.checkbox("I understand the architectural differences between eager load() and streaming lazy_load().", value=True)
chk5 = st.checkbox("I can build custom DocumentTransformer pipelines to clean and enrich text before embedding.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 Document architecture interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 12.2 Documents!")
