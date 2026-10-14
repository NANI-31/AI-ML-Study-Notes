import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, date

st.set_page_config(
    page_title="9.1 Why Metadata Filtering - Vector DB Curriculum",
    page_icon="🏷️",
    layout="wide"
)

# Title & Overview
st.title("9.1 Why Metadata Filtering: The Bridge Between Semantic & Structured Search")
st.caption("Chapter 9: Metadata Filtering | Domain: 4. Vector Database")
st.markdown("""
Vector embeddings are extraordinarily capable at finding **thematically similar concepts**, but they are fundamentally 
**unsuited for strict logical constraints**.

For example, asking a vector database *"Show me Q3 financial statements for European operations"* using pure vector similarity 
might retrieve an article about *"2021 European market expansion strategy"* simply because the vocabulary is semantically aligned, 
even though it completely violates your strict business requirements (it is from 2021 instead of Q3 current year, and is a blog post instead of a financial statement).

**Metadata Filtering** introduces deterministic boolean logic alongside high-dimensional vector search:
* **Category Filtering**: Restricting searches to verified taxonomies (`category == 'financial_reports'`).
* **Temporal Filtering**: Enforcing recency windows (`created_timestamp >= 1704067200`).
* **Authorship & Provenance**: Isolating verified corporate policies from unofficial forum posts (`author == 'Legal Team'`).
* **Organizational Boundaries**: Enforcing departmental security and tenancy (`department == 'Engineering'`).
* **Document Type Filtering**: Distinguishing source code, PDFs, spreadsheets, and markdown notes (`doc_type == 'policy'`).

This lesson explores why metadata filtering is mandatory for enterprise search, examines how filtering interacts with 
vector index traversal (Pre-filtering vs. Post-filtering vs. Single-Stage Filtering), and demonstrates practical schema design.
""")

st.divider()

# Section 1: The Failure Modes of Pure Vector Search
st.header("1. The Failure Modes of Pure Vector Search", divider="rainbow")

col_fail1, col_fail2 = col_fail1, col_fail2 = st.columns([1, 1])

with col_fail1:
    st.markdown("""
    ### Why Semantic Similarity Alone Is Not Enough
    Pure vector search relies on geometric distance in embedding space. In high-dimensional geometry:
    1. **Vectors Cannot Enforce Negations**: A prompt *"Show me Python tutorials, NOT for beginners"* embeds all keywords including "beginners". The nearest neighbors will frequently be beginner Python tutorials because the semantic cluster is dominated by Python tutorial tokens.
    2. **Vectors Cannot Enforce Hard Boundaries**: A user querying for *"Documents modified in the last 7 days"* cannot be satisfied with vector math. Embeddings represent semantic meaning, not chronological timestamps.
    3. **Vectors Cannot Guarantee Security (ACLs)**: If an unauthorized user queries *"Executive compensation packages"*, vector similarity will retrieve confidential HR files unless an immutable deterministic filter (`where={"access_level": {"$lte": user_clearance}}`) stops the traversal.
    4. **Semantic Drift Across Doc Types**: An API reference manual, a sales deck, and a customer complaint about the same product will map to overlapping vector regions despite serving fundamentally different operational purposes.
    """)

with col_fail2:
    st.markdown("### The Dual-Engine Search Reality")
    arch_dot = """
    digraph DualEngine {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQuery [label="User Query\\n'Q3 engineering roadmap by Alice'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_split {
            label="Query Decomposition"; style="dashed"; color="#6B7280";
            SemanticPart [label="Semantic Component\\n'engineering roadmap'", fillcolor="#E0E7FF"];
            StructuredPart [label="Metadata Filters\\nquarter == 'Q3'\\nauthor == 'Alice'", fillcolor="#FEF3C7"];
        }

        VectorIndex [label="Vector Index (HNSW)\\nCosine Similarity Traversal", fillcolor="#D1FAE5", color="#059669"];
        MetadataIndex [label="Metadata Inverted Index / B-Tree\\nDeterministic Boolean Evaluation", fillcolor="#FCE7F3", color="#DB2777"];

        UnifiedEngine [label="Filtered Vector Search Engine\\nPre-filtered or Single-stage Graph Search", fillcolor="#C7D2FE", color="#4338CA", style="filled,bold"];

        Results [label="Precision Result Set\\nRelevant + Correct Quarter + Correct Author", fillcolor="#A7F3D0", style="filled,bold"];

        UserQuery -> SemanticPart;
        UserQuery -> StructuredPart;
        SemanticPart -> VectorIndex;
        StructuredPart -> MetadataIndex;
        VectorIndex -> UnifiedEngine;
        MetadataIndex -> UnifiedEngine;
        UnifiedEngine -> Results;
    }
    """
    st.graphviz_chart(arch_dot, use_container_width=False)

st.divider()

# Section 2: Filtering Architectures: Pre, Post, and Single-Stage
st.header("2. Filtering Execution Strategies: Pre, Post, and Single-Stage", divider="rainbow")

st.markdown("""
How a vector database executes metadata filters fundamentally dictates **retrieval latency** and **recall accuracy**. 
There are three primary architectural approaches in modern vector databases:
""")

col_strat1, col_strat2, col_strat3 = st.columns(3)

with col_strat1:
    st.markdown("""
    #### 1. Post-Filtering (Filter-After-Search)
    * **Mechanism**: Execute full vector ANN search to retrieve top-K (e.g., K=10), then iterate through the retrieved set and discard records that fail the metadata predicate.
    * **Severe Flaw: Recall Collapse**: If only 5% of documents match the metadata filter, retrieving top-10 vectors might yield **zero** passing documents!
    * **When Used**: Only in naive prototypes or when the filter is expected to match >90% of the entire database.
    """)
    st.error("Antipattern for production: Causes massive false-negative rates when filters are selective.")

with col_strat2:
    st.markdown("""
    #### 2. Pre-Filtering (Filter-Before-Search)
    * **Mechanism**: Evaluate the metadata query first using an inverted index / SQLite index to build a candidate ID set. Then execute vector search strictly over the filtered candidate set.
    * **Advantage**: 100% recall accuracy; guarantees every returned vector satisfies the metadata conditions.
    * **Trade-off**: If the filtered candidate set contains 500,000 vectors, traversing an existing global HNSW index becomes disconnected (graph hops may land on excluded nodes). Often requires brute-force exact scan over candidate embeddings.
    """)
    st.info("High precision, but can suffer latency spikes if candidate set is large.")

with col_strat3:
    st.markdown("""
    #### 3. Single-Stage / Iterative Graph Filtering
    * **Mechanism**: The vector index graph traversal actively checks metadata predicates **during the HNSW navigation step**.
    * If a neighbor node fails the metadata filter, the search algorithm keeps it in the exploration queue to find downstream connected nodes, but excludes it from the final candidate top-K list.
    * **Advantage**: Combines low latency graph traversal with strict 100% metadata compliance without recall collapse.
    """)
    st.success("Modern standard utilized in enterprise engines like ChromaDB, Qdrant, and Milvus.")

st.divider()

# Section 3: The 5 Core Metadata Filtering Dimensions
st.header("3. The Five Core Metadata Filtering Dimensions", divider="rainbow")

st.markdown("""
In enterprise knowledge systems, vector metadata should be structured across five foundational dimensions:
""")

tab_d1, tab_d2, tab_d3, tab_d4, tab_d5 = st.tabs([
    "1. Category / Taxonomy",
    "2. Temporal / Date",
    "3. Author / Provenance",
    "4. Department / Tenancy",
    "5. Document Type / Format"
])

with tab_d1:
    st.markdown("""
    ### Dimension 1: Category & Domain Taxonomy
    * **Purpose**: Partitions knowledge into high-level subject domains to prevent cross-domain contamination.
    * **Examples**: `{"category": "legal"}`, `{"category": "marketing"}`, `{"category": "infrastructure"}`.
    * **ChromaDB Usage**:
    ```python
    results = collection.query(
        query_texts=["termination clause and severance packages"],
        n_results=5,
        where={"category": "legal"}
    )
    ```
    * **Best Practice**: Use standardized lowercase enum strings (`"legal"` rather than `"Legal"`, `"LEGAL"`, `"legal_affairs"`).
    """)

with tab_d2:
    st.markdown("""
    ### Dimension 2: Temporal & Date Filtering
    * **Purpose**: Prevents stale information from polluting current business answers (e.g., retrieving 2019 remote work rules instead of 2024 policy).
    * **ChromaDB Restriction**: Chroma metadata values must be **primitive numbers or strings**. Chroma does not have a native `DATETIME` type!
    * **Best Practice**: Store timestamps as **Unix Epoch Integers** (seconds or milliseconds since 1970-01-01) or **ISO 8601 Strings** (`YYYY-MM-DD`):
    ```python
    # Find documents published since January 1, 2024 (epoch: 1704067200)
    results = collection.query(
        query_texts=["quarterly revenue guidance"],
        n_results=5,
        where={"created_epoch": {"$gte": 1704067200}}
    )
    ```
    """)

with tab_d3:
    st.markdown("""
    ### Dimension 3: Author & Provenance
    * **Purpose**: Source verification, hallucination mitigation, and copyright tracking.
    * **Examples**: `{"author": "Security Audit Team"}`, `{"verified_by": "Compliance Officer"}`.
    * **Code Pattern**:
    ```python
    results = collection.query(
        query_texts=["firewall exception approval workflow"],
        n_results=3,
        where={"author": "SecOps Team"}
    )
    ```
    """)

with tab_d4:
    st.markdown("""
    ### Dimension 4: Department & Multi-Tenancy (Security Isolation)
    * **Purpose**: Hard security boundaries preventing data leakage across organizational departments or client organizations.
    * **Examples**: `{"department": "Finance"}`, `{"tenant_id": "acme_corp_881"}`.
    * **Production Rule**: In multi-tenant SaaS, tenant filtering must **never be optional** or user-controlled. It must be hardcoded in the server-side API middleware:
    ```python
    # Enforced at API layer before query execution
    tenant_filter = {"tenant_id": current_authenticated_tenant}
    results = collection.query(
        query_texts=[sanitized_user_query],
        where=tenant_filter
    )
    ```
    """)

with tab_d5:
    st.markdown("""
    ### Dimension 5: Document Type & Format
    * **Purpose**: Tailors retrieval to the desired output format (e.g., extracting executable code vs reading narrative architecture overviews).
    * **Examples**: `{"doc_type": "api_reference"}`, `{"doc_type": "meeting_notes"}`, `{"doc_type": "code_snippet"}`.
    * **Code Pattern**:
    ```python
    # Retrieve technical specifications only, excluding meeting chatter
    results = collection.query(
        query_texts=["Redis cache eviction policy setup"],
        n_results=4,
        where={"doc_type": "api_reference"}
    )
    ```
    """)

st.divider()

# Section 4: Interactive Workbench: Pure Vector vs Metadata-Filtered Retrieval
st.header("4. Interactive Workbench: Pure Vector vs Filtered Search", divider="rainbow")

st.markdown("""
Test how metadata filters eliminate false-positive semantic matches and enforce business rules on an enterprise knowledge base.
""")

# Mock enterprise knowledge base
ENTERPRISE_KB = [
    {
        "id": "KB-ENG-01",
        "title": "Production Deployment Safety Protocol 2024",
        "category": "Engineering",
        "department": "DevOps",
        "author": "Alice Lin",
        "doc_type": "Policy",
        "year": 2024,
        "text": "All production releases require two senior approvals, green canary tests, and verified database rollback migrations.",
        "sim_score": 0.88
    },
    {
        "id": "KB-ENG-02",
        "title": "Legacy Deploy Script Guide (Deprecated)",
        "category": "Engineering",
        "department": "DevOps",
        "author": "Bob Ross",
        "doc_type": "Guide",
        "year": 2019,
        "text": "Run deploy.sh on the primary bastion host with root permissions to update web server binaries.",
        "sim_score": 0.94  # High semantic similarity to "how do I deploy", but dangerous!
    },
    {
        "id": "KB-HR-03",
        "title": "Engineering Onboarding & Hardware Policy",
        "category": "Human Resources",
        "department": "HR",
        "author": "Clara Evans",
        "doc_type": "Policy",
        "year": 2024,
        "text": "New software engineers are provisioned 64GB laptops and required to complete security awareness training within 7 days.",
        "sim_score": 0.65
    },
    {
        "id": "KB-SEC-04",
        "title": "SOC2 Incident Response Handbook",
        "category": "Security",
        "department": "SecOps",
        "author": "Alice Lin",
        "doc_type": "Handbook",
        "year": 2023,
        "text": "In the event of anomalous data egress, notify the security team within 15 minutes and isolate affected VPC subnets.",
        "sim_score": 0.72
    },
    {
        "id": "KB-LEGAL-05",
        "title": "Vendor SLA Agreement Standards",
        "category": "Legal",
        "department": "Legal",
        "author": "David Vance",
        "doc_type": "Contract",
        "year": 2022,
        "text": "All third-party cloud infrastructure vendors must guarantee 99.95% uptime with financial penalties for unscheduled downtime.",
        "sim_score": 0.58
    },
    {
        "id": "KB-ENG-06",
        "title": "Microservices Observability Best Practices",
        "category": "Engineering",
        "department": "DevOps",
        "author": "Alice Lin",
        "doc_type": "Guide",
        "year": 2024,
        "text": "Instrument distributed tracing using OpenTelemetry with standard span tags for service name, environment, and latency.",
        "sim_score": 0.81
    }
]

col_ui_query, col_ui_filters = st.columns([1, 1.2])

with col_ui_query:
    st.subheader("1. Vector Query Simulation")
    user_q = st.selectbox(
        "Select User Prompt Scenario:",
        [
            "How do I deploy changes to production safely?",
            "What is our security incident notification protocol?",
            "What are engineering onboarding requirements?"
        ]
    )
    
    top_k_select = st.slider("Top-K Results to Fetch:", min_value=1, max_value=6, value=3)

with col_ui_filters:
    st.subheader("2. Metadata Filter Predicates (`where` clause)")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_dept = st.multiselect("Department:", ["DevOps", "HR", "SecOps", "Legal"], default=["DevOps"])
        f_type = st.multiselect("Document Type:", ["Policy", "Guide", "Handbook", "Contract"], default=["Policy", "Guide"])
    with col_f2:
        f_min_year = st.slider("Published Year >= :", min_value=2018, max_value=2024, value=2023)
        f_author = st.multiselect("Author:", ["Alice Lin", "Bob Ross", "Clara Evans", "David Vance"], default=["Alice Lin", "Bob Ross"])

# Calculate pure vector vs filtered
st.divider()

col_out_pure, col_out_filt = st.columns([1, 1])

with col_out_pure:
    st.markdown("### 🔴 Pure Vector Search (No Filters)")
    st.caption("Sorts strictly by semantic similarity. Notice how outdated or deprecated docs can rank #1!")
    
    # Sort purely by sim_score
    pure_docs = sorted(ENTERPRISE_KB, key=lambda x: x["sim_score"], reverse=True)[:top_k_select]
    df_pure = pd.DataFrame([
        {
            "ID": d["id"],
            "Title": d["title"],
            "Similarity": f"{d['sim_score']:.2f}",
            "Year": d["year"],
            "Dept": d["department"],
            "Type": d["doc_type"]
        }
        for d in pure_docs
    ])
    st.dataframe(df_pure, use_container_width=True, hide_index=True)
    
    if any(d["id"] == "KB-ENG-02" for d in pure_docs):
        st.error("⚠️ **Risk Alert**: Pure vector search recommended the 2019 deprecated deploy script because its vocabulary matched 'deploy' strongly!")

with col_out_filt:
    st.markdown("### 🟢 Filtered Vector Search (`where={...}`)")
    st.caption("Evaluates metadata predicates first, then ranks qualifying candidates by semantic similarity.")
    
    # Filter candidates
    passing_candidates = [
        d for d in ENTERPRISE_KB
        if (not f_dept or d["department"] in f_dept)
        and (not f_type or d["doc_type"] in f_type)
        and (d["year"] >= f_min_year)
        and (not f_author or d["author"] in f_author)
    ]
    
    filtered_ranked = sorted(passing_candidates, key=lambda x: x["sim_score"], reverse=True)[:top_k_select]
    
    if filtered_ranked:
        df_filt = pd.DataFrame([
            {
                "ID": d["id"],
                "Title": d["title"],
                "Similarity": f"{d['sim_score']:.2f}",
                "Year": d["year"],
                "Dept": d["department"],
                "Type": d["doc_type"]
            }
            for d in filtered_ranked
        ])
        st.dataframe(df_filt, use_container_width=True, hide_index=True)
        st.success(f"✅ Filter matched **{len(filtered_ranked)}** verified, compliant documents. Deprecated records strictly excluded!")
    else:
        st.warning("⚠️ No documents matched all specified metadata filter conditions. Zero candidates passed.")

st.markdown("""
### Generated ChromaDB Query Payload
```python
results = collection.query(
    query_texts=["{0}"],
    n_results={1},
    where={{
        "$and": [
            {{"department": {{"$in": {2}}}}},
            {{"doc_type": {{"$in": {3}}}}},
            {{"year": {{"$gte": {4}}}}}
        ]
    }}
)
```
""".format(user_q, top_k_select, f_dept, f_type, f_min_year))

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is 'Recall Collapse' in vector database post-filtering, and how does it happen?"):
    st.markdown("""
    **Answer**:
    Recall collapse occurs when a vector database executes vector ANN search first to find the top-K nearest neighbors, and only then applies metadata filters to discard non-matching candidates.
    
    If the metadata condition is highly selective (for instance, matching only 2% of documents in the collection, such as `tenant_id == 'acme'`), the global top-K nearest neighbors will almost certainly belong to other tenants. When post-filtering discards non-matching records, the final result set shrinks to 0 or 1 document, even though the target tenant had dozens of highly relevant documents in the database.
    """)

with st.expander("Q2: Why does ChromaDB restrict metadata values to primitive types (str, int, float, bool)?"):
    st.markdown("""
    **Answer**:
    ChromaDB stores metadata inside an embedded SQLite relational table using standard typed columns or structured JSON extraction. 
    1. **Relational Indexing**: Primitive scalars map directly to SQLite B-Tree indices (`CREATE INDEX idx_meta ON ...`), providing sub-millisecond lookups.
    2. **Serialization Predictability**: Arbitrary nested dictionaries, lists, or custom Python classes introduce ambiguous comparison semantics (e.g., how to compare a list against an integer in `$gt`), slow down serialization, and prevent deterministic B-tree range queries.
    """)

with st.expander("Q3: How do you handle timestamp and date range queries in a vector database that lacks a native DATETIME type?"):
    st.markdown("""
    **Answer**:
    There are two standard production solutions:
    1. **Unix Epoch Integers (Recommended)**: Store timestamps as integers representing seconds since 1970-01-01 (`int(time.time())`). This allows fast integer comparisons using `$gt`, `$gte`, `$lt`, and `$lte`.
    2. **ISO 8601 Strings (`YYYY-MM-DDTHH:MM:SSZ`)**: Because lexicographical string sorting of ISO 8601 timestamps mirrors chronological order (`"2024-03-01" > "2024-02-15"`), string range filters can be applied directly. However, integer epoch comparison is faster and consumes less RAM.
    """)

with st.expander("Q4: Explain the difference between Pre-Filtering and Single-Stage Iterative Graph Filtering."):
    st.markdown("""
    **Answer**:
    * **Pre-Filtering**: Identifies all document IDs that pass the metadata filter first (via inverted index / B-tree). If the candidate set is small, the database must brute-force compare query vectors against candidates, or construct a temporary sub-index. If the candidate set is large, the disconnected global HNSW graph cannot be traversed cleanly.
    * **Single-Stage Iterative Filtering**: The search algorithm navigates the existing global HNSW graph. During traversal, it evaluates metadata predicates dynamically: non-matching nodes are traversed to reach deeper connected nodes, but only matching nodes are admitted into the final nearest neighbor candidate priority queue.
    """)

with st.expander("Q5: How should Multi-Tenancy be implemented when designing a vector database architecture?"):
    st.markdown("""
    **Answer**:
    There are three architectural patterns:
    1. **Collection-Per-Tenant**: Each tenant gets a separate Chroma collection or table. Provides total isolation, but incurs heavy overhead if you have 10,000+ tenants (resource starvation from thousands of open index files).
    2. **Metadata-Filtered Shared Collection**: All tenants share a single collection, and every record has `tenant_id` in metadata. The API gateway forcefully injects `where={"tenant_id": current_tenant}` on every query. High resource efficiency, but requires strict code auditing to prevent accidental data leaks.
    3. **Namespace / Partition-Based**: Supported in distributed engines (e.g., Pinecone namespaces, Milvus partitions), providing isolated sub-indexes within a shared cluster.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Capitalization Inconsistency in String Filters**  
    ChromaDB metadata equality is case-sensitive: `where={"category": "Legal"}` will **not** match a record saved with `category: "legal"`.  
    **Fix**: Always normalize all string metadata keys and values to `.strip().lower()` during the ingestion pipeline.
    """)
    st.warning("""
    **Trap 2: Storing Nested Dictionaries as Metadata**  
    Passing `metadata={"author": {"name": "Alice", "role": "admin"}}` causes ChromaDB to raise `ValueError: Expected metadata value to be a str, int, float or bool`.  
    **Fix**: Flatten nested fields into dot-notation strings: `metadata={"author_name": "Alice", "author_role": "admin"}`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Client-Controlled Tenancy Filters**  
    Trusting the frontend client to pass `tenant_id` in query parameters allows malicious users to tamper with request payloads and inspect another organization's confidential vectors.  
    **Fix**: Extract tenant ID strictly from authenticated JWT tokens on the backend API server.
    """)
    st.warning("""
    **Trap 4: Over-Filtering Resulting in Empty Top-K**  
    Applying overly aggressive `where` filters with 5+ AND conditions can reduce matching records below `n_results`. The client application must gracefully handle empty result arrays (`results['ids'][0] == []`).
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Recall Collapse?"):
        st.markdown("""
        **Recall Collapse**:  
        When post-filtering discards non-matching top-K vectors, leaving zero or near-zero results because the filter was too selective.
        """)
    with st.expander("🗂️ Flashcard: How to store dates in ChromaDB?"):
        st.markdown("""
        **Date Storage**:  
        As **Unix Epoch Integers** (`int(time.time())`) or **ISO 8601 strings** (`YYYY-MM-DD`). No native DATETIME type exists.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What metadata types are supported in Chroma?"):
        st.markdown("""
        **Supported Types**:  
        `str`, `int`, `float`, and `bool`. (Lists, dicts, and None values are rejected).
        """)
    with st.expander("🗂️ Flashcard: Why can't vectors enforce negations?"):
        st.markdown("""
        **Semantic Negation**:  
        Embeddings embed all tokens including negative qualifiers. The nearest neighbor will still share the semantic cluster of the negated topic.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why semantic vector similarity alone cannot enforce hard boolean constraints.", value=True)
chk2 = st.checkbox("I can explain how Post-Filtering causes recall collapse on selective queries.", value=True)
chk3 = st.checkbox("I know how Single-Stage iterative graph filtering solves both latency and recall issues.", value=True)
chk4 = st.checkbox("I know how to store dates as Unix epoch integers to execute range queries in ChromaDB.", value=True)
chk5 = st.checkbox("I understand the 5 core metadata dimensions: Category, Date, Author, Department, and Doc Type.", value=True)
chk6 = st.checkbox("I can design a flattened, validated metadata schema for enterprise multi-tenant search.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Why Metadata Filtering and are ready for Chapter 9.2: Filter Operators!")
