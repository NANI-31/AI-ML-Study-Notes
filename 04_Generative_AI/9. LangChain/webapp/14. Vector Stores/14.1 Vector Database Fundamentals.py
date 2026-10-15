import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class StoredVectorRecord:
    """Represents a persisted vector record with payload metadata."""
    def __init__(self, doc_id: str, text: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.text = text
        self.vector = vector
        self.metadata = metadata

class MockVectorDBEngine:
    """Simulates an indexed vector database supporting HNSW clustering and boolean metadata filtering."""
    def __init__(self):
        # 3D normalized synthetic vector space representing three enterprise clusters
        self.records: List[StoredVectorRecord] = [
            StoredVectorRecord(
                "rec-001", "Production Aurora PostgreSQL automated snapshot SLA is 35 days.",
                np.array([0.91, 0.12, 0.25]),
                {"department": "DevOps", "classification": "Confidential", "year": 2026}
            ),
            StoredVectorRecord(
                "rec-002", "Kubernetes cluster ingress uses Cilium eBPF service mesh routing.",
                np.array([0.88, 0.20, 0.30]),
                {"department": "DevOps", "classification": "Internal", "year": 2026}
            ),
            StoredVectorRecord(
                "rec-003", "Employee dental coverage allowance is capped at 2,000 USD annually.",
                np.array([0.15, 0.92, 0.18]),
                {"department": "HR", "classification": "Internal", "year": 2025}
            ),
            StoredVectorRecord(
                "rec-004", "Parental leave policy provides 16 weeks of 100% compensated leave.",
                np.array([0.12, 0.89, 0.22]),
                {"department": "HR", "classification": "Internal", "year": 2026}
            ),
            StoredVectorRecord(
                "rec-005", "Zero Trust access proxy enforces continuous mTLS identity challenge.",
                np.array([0.35, 0.18, 0.92]),
                {"department": "Security", "classification": "Confidential", "year": 2026}
            ),
            StoredVectorRecord(
                "rec-006", "Public disclosure program guidelines for external white-hat researchers.",
                np.array([0.30, 0.25, 0.88]),
                {"department": "Security", "classification": "Public", "year": 2025}
            )
        ]
        # Ensure L2 normalization
        for r in self.records:
            norm = np.linalg.norm(r.vector)
            if norm > 0:
                r.vector = r.vector / norm

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.lower()
        v = np.array([0.2, 0.2, 0.2])
        if any(w in q for w in ["database", "postgres", "k8s", "devops", "cloud", "cluster"]):
            v = np.array([0.92, 0.14, 0.22])
        elif any(w in q for w in ["dental", "health", "leave", "hr", "benefits", "employee"]):
            v = np.array([0.14, 0.94, 0.18])
        elif any(w in q for w in ["security", "zero trust", "vulnerability", "disclosure", "mtls"]):
            v = np.array([0.32, 0.16, 0.93])
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def query(self, query_text: str, dept_filter: str, class_filter: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_vec = self._embed_query(query_text)
        candidates = []

        for r in self.records:
            # Metadata pre-filtering
            if dept_filter != "All" and r.metadata.get("department") != dept_filter:
                continue
            if class_filter != "All" and r.metadata.get("classification") != class_filter:
                continue

            # Cosine similarity on unit vectors
            sim = float(np.dot(q_vec, r.vector))
            candidates.append({
                "doc_id": r.doc_id,
                "text": r.text,
                "metadata": r.metadata,
                "similarity": round(sim, 4),
                "distance": round(1.0 - sim, 4)
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="14.1 Vector Database Fundamentals",
    page_icon="🗄️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">14.1 Vector Database Fundamentals</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Vector indexing structures, approximate nearest neighbors (ANN), metadata pre-filtering, and similarity query execution.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Vector Database Storage Engine & Index Topology", divider="rainbow")

st.markdown("""
Relational databases (RDBMS) are built around B-Trees, optimized for exact matches, ranges, and tabular joins.
Document databases (NoSQL) index JSON keys for structural lookup.
**Vector Databases** are specialized storage engines engineered around multi-dimensional geometric indices. 
They store dense floating-point arrays alongside metadata payloads, executing **Approximate Nearest Neighbor (ANN)** searches 
over millions of vectors in single-digit milliseconds.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph VectorDBArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_ingestion_layer {
            label="1. Inbound Vector & Payload Storage";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            VectorPayload [label="Vector Record Inbound\\n• Vector: [0.042, -0.198, ..., 0.812] (d=1536)\\n• Payload: {dept: 'HR', tenant: 'org-42'}\\n• Primary Key: 'doc-uuid-84920'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_indexing_engine {
            label="2. Specialized Vector Index Structures";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            HNSWIndex [label="HNSW Graph Index (Hierarchical Navigable Small World)\\nMulti-layer skip-list graph for sub-millisecond traversal", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            IVFIndex [label="IVF-PQ Index (Inverted File + Product Quantization)\\nVoronoi cell clustering for massive billion-scale compression", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            InvertedPayload [label="Inverted Payload Index\\nB-Tree / Bitmap index on structured metadata fields", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_search_execution {
            label="3. Hybrid Query & Filter Engine";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            QueryRouter [label="Query Ingress with Filter\\n'Find vacation policy WHERE dept = HR'", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            PreFilter [label="Metadata Pre-Filter\\nCandidate set reduced via Bitmap index", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            ANNSearch [label="ANN Vector Search Kernel\\nSIMD Dot-Product distance across candidate graph", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_result_delivery {
            label="4. Top-K Ranked Delivery";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            TopKResults [label="Top-K Ranked Documents\\nRanked by similarity score with verified payloads", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399", penwidth=2];
        }

        VectorPayload -> HNSWIndex [label="Insert vector"];
        VectorPayload -> InvertedPayload [label="Insert metadata"];
        HNSWIndex -> IVFIndex [label="Optional quantization"];

        QueryRouter -> PreFilter [label="1. Evaluate filter"];
        InvertedPayload -> PreFilter [label="Matching row IDs"];
        PreFilter -> ANNSearch [label="2. Scoped search space"];
        HNSWIndex -> ANNSearch [label="Graph traversal"];
        ANNSearch -> TopKResults [label="3. Return top-K matches"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT IS A VECTOR DATABASE? ---
st.header("2. What is a Vector Database?", divider="rainbow")

st.markdown("""
A **Vector Database** is a dedicated database system designed to store, manage, and rapidly query multi-dimensional 
vector embeddings alongside unstructured text and structured metadata.

Unlike standard databases that perform deterministic point lookups (`WHERE id = 42`) or range scans (`WHERE age > 30`), 
a vector database solves the **Nearest Neighbor Search problem**: finding the data records whose vectors have the smallest 
geometric distance to a query vector.
""")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("#### 🏛️ Traditional RDBMS (PostgreSQL / MySQL)")
    st.markdown("""
    - Indexed with B-Trees and Hash Indexes.
    - Optimized for scalar equality, joins, and ACID transactions.
    - Unsuited for high-dimensional nearest-neighbor distance (brute force table scan O(N)).
    """)
with col_v2:
    st.markdown("#### ⚡ Dedicated Vector Database (Chroma / Pinecone / Milvus)")
    st.markdown("""
    - Indexed with HNSW graphs and IVF cluster trees.
    - Optimized for geometric vector proximity in high dimensions (e.g. 1536D).
    - Executes sub-5ms similarity searches across millions of documents.
    """)

st.divider()

# --- SECTION 3: VECTOR INDEXING ---
st.header("3. Vector Indexing", divider="rainbow")

st.markdown("""
Calculating the distance between a query vector and every vector in a database of 10 million items takes hundreds of seconds (Brute Force O(N * d)).
To achieve real-time search, vector databases construct **Approximate Nearest Neighbor (ANN)** index graphs:
""")

idx_c1, idx_c2, idx_c3 = st.columns(3)
with idx_c1:
    st.subheader("🕸️ HNSW (Hierarchical Navigable Small World)")
    st.markdown("""
    - Constructs a multi-layer graph of vectors where higher layers have long-range skip edges and lower layers have dense local connections.
    - Search navigates down layers in logarithmic time: O(log N * d).
    - Offers the highest recall (98%+) with ultra-low latency; higher RAM usage.
    """)

with idx_c2:
    st.subheader("🎯 IVF (Inverted File Index)")
    st.markdown("""
    - Clusters vectors into Voronoi cells using k-means.
    - Query vector is compared only to the nearest cluster centroids, searching only the vectors inside those specific cells.
    - Drastically reduces the search space at the cost of slight recall loss at cluster boundaries.
    """)

with idx_c3:
    st.subheader("🗜️ Product Quantization (PQ)")
    st.markdown("""
    - Compresses 1536 float32 numbers (6,144 bytes) into compact 8-bit byte codes (e.g. 64 bytes).
    - Reduces vector memory footprint by 95%, allowing hundreds of millions of vectors to fit into server RAM.
    """)

st.divider()

# --- SECTION 4: VECTOR SEARCH ---
st.header("4. Vector Search", divider="rainbow")

st.markdown("""
Vector search navigates the ANN index to discover the nearest candidates to the query embedding.
LangChain supports multiple search modalities:
- **Standard Similarity Search (`similarity_search`)**: Returns the k vectors with highest cosine similarity.
- **Similarity Search with Relevance Scores (`similarity_search_with_score`)**: Returns both the Document objects and their normalized confidence scores.
- **Maximal Marginal Relevance (MMR) (`max_marginal_relevance_search`)**: Balances semantic similarity with **result diversity**, penalizing redundant candidate documents.
""")

st.code("""# Standard Similarity Search vs MMR in LangChain
from langchain_community.vectorstores import Chroma

# 1. Pure Similarity Search (risk of returning 3 near-duplicate chunks)
docs = vectorstore.similarity_search("PostgreSQL failover", k=3)

# 2. Maximal Marginal Relevance (guarantees diverse perspectives)
diverse_docs = vectorstore.max_marginal_relevance_search(
    "PostgreSQL failover",
    k=3,
    fetch_k=10,  # Evaluate top 10 candidates
    lambda_mult=0.7  # 1.0 = pure similarity, 0.0 = pure diversity
)""", language="python")

st.divider()

# --- SECTION 5: METADATA FILTERING ---
st.header("5. Metadata Filtering", divider="rainbow")

st.markdown("""
In enterprise architectures, vector search must be constrained by business rules, tenant boundaries, and access permissions.
Vector databases implement **Metadata Pre-Filtering**: applying scalar conditions (e.g. `WHERE tenant_id = 'org_42' AND status = 'active'`) 
before or during graph traversal.
""")

st.code("""# Metadata Filtering Query in LangChain
filtered_docs = vectorstore.similarity_search(
    "vacation policy",
    k=2,
    filter={
        "department": "HR",
        "year": 2026
    }
)""", language="python")

st.info("""
**Pre-Filtering vs Post-Filtering**: Always verify that your vector database supports **Pre-Filtering**. 
Post-filtering (filtering results after top-k vector search) can return zero documents if all nearest neighbors happen to belong to other departments!
""")

st.divider()

# --- SECTION 6: SIMILARITY SEARCH ---
st.header("6. Similarity Search", divider="rainbow")

st.markdown("""
Executing similarity search involves converting the query string into a vector using the exact same embedding model 
used during ingestion, querying the index, and sorting matches by distance.
""")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: In-Memory Indexed Vector Database Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements an in-memory vector database with L2-normalized vector storage, 
exact dot-product search, and boolean metadata pre-filtering.
""")

st.code("""import numpy as np
from typing import List, Dict, Any, Optional

class VectorRecord:
    def __init__(self, doc_id: str, content: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        norm = np.linalg.norm(vector)
        self.vector = vector / norm if norm > 0 else vector
        self.metadata = metadata

class SimpleProductionVectorStore:
    \"\"\"Production vector store demonstrating normalized dot-product and pre-filtering.\"\"\"
    def __init__(self):
        self.records: List[VectorRecord] = []

    def add_texts(self, texts: List[str], vectors: List[np.ndarray], metadatas: List[Dict[str, Any]]) -> List[str]:
        ids = []
        for i, (text, vec, meta) in enumerate(zip(texts, vectors, metadatas)):
            doc_id = f"vec-{len(self.records) + 1:05d}"
            self.records.append(VectorRecord(doc_id, text, vec, meta))
            ids.append(doc_id)
        return ids

    def similarity_search_with_filter(self,
                                      query_vec: np.ndarray,
                                      k: int = 3,
                                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # 1. Normalize query
        norm = np.linalg.norm(query_vec)
        q_norm = query_vec / norm if norm > 0 else query_vec

        candidates = []
        for rec in self.records:
            # 2. Metadata Pre-Filtering
            if filter_dict:
                match = True
                for fk, fv in filter_dict.items():
                    if rec.metadata.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            # 3. Dot Product (Cosine Similarity)
            score = float(np.dot(q_norm, rec.vector))
            candidates.append({
                "id": rec.doc_id,
                "content": rec.content,
                "metadata": rec.metadata,
                "score": round(score, 4)
            })

        # 4. Sort and return Top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Vector Indexing & Hybrid Metadata Filtering Studio", divider="rainbow")

st.markdown("""
Search our live indexed enterprise vector store. Toggle department and classification metadata filters to see how 
pre-filtering constrains the search space while preserving high semantic precision.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Query & Pre-Filter Configuration")
    user_query = st.text_input(
        "Natural Language Search Query",
        value="What is our database automated backup and failover policy?"
    )

    st.markdown("#### Metadata Pre-Filters:")
    filter_dept = st.selectbox("Department Filter", options=["All", "DevOps", "HR", "Security"])
    filter_class = st.selectbox("Classification Filter", options=["All", "Internal", "Confidential", "Public"])
    k_val = st.slider("Max Results (Top-K)", min_value=1, max_value=5, value=3)

    btn_exec = st.button("🚀 Execute Filtered Vector Search", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Query Results & Confidence Scores")

    if btn_exec:
        vdb = MockVectorDBEngine()
        results = vdb.query(user_query, filter_dept, filter_class, top_k=k_val)

        st.metric("Matches Found", len(results))

        if results:
            for idx, r in enumerate(results, 1):
                with st.expander(f"Result #{idx} (Score: {r['similarity']}) — {r['doc_id']}", expanded=True):
                    st.write(f"**Content**: `{r['text']}`")
                    c1, c2 = st.columns(2)
                    c1.metric("Cosine Similarity", r["similarity"])
                    c2.metric("Cosine Distance", r["distance"])
                    st.json(r["metadata"])
        else:
            st.warning("No records matched the combined semantic query and metadata filter criteria.")
    else:
        st.info("Configure your filters and click **Execute Filtered Vector Search** to run queries.")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does an HNSW index achieve logarithmic O(log N) search latency?",
     "HNSW organizes vectors into a multi-layer geometric graph analogous to a skip-list. The top layers contain sparse long-range highway edges connecting distant clusters. Search begins at the top layer, rapidly jumping across large distances until a local minimum is reached, then drops down to denser layers to fine-tune nearest neighbors, achieving logarithmic complexity instead of scanning every vector."),
    ("What is the difference between Metadata Pre-filtering and Metadata Post-filtering?",
     "Pre-filtering applies metadata constraints (e.g. WHERE tenant_id = 'org-42') before or during graph traversal, guaranteeing that all returned top-K vectors satisfy the metadata filter. Post-filtering retrieves top-K nearest neighbors from the global index first and then discards non-matching vectors. If the nearest neighbors happen to belong to other tenants, post-filtering returns fewer than K items or zero results."),
    ("What is Maximal Marginal Relevance (MMR), and why is it used in RAG?",
     "Standard similarity search often returns multiple chunks that say the exact same thing (redundant context). MMR uses a trade-off parameter (lambda) to score candidates based on both their similarity to the query and their dissimilarity to previously selected chunks. This ensures the prompt receives diverse context covering different aspects of the topic."),
    ("When should an engineering team choose pgvector over a dedicated vector database like Pinecone or Milvus?",
     "Choose pgvector if your team already uses PostgreSQL, your vector dataset is under a few million records, and you require strict ACID relational joins between business tables and vector embeddings. Choose dedicated engines (Pinecone, Qdrant, Milvus) when your dataset scales to tens or hundreds of millions of vectors, requires distributed horizontal scaling, or demands sub-5ms latency under heavy concurrent query loads."),
    ("How does Product Quantization (PQ) compress high-dimensional vectors?",
     "PQ breaks a high-dimensional vector (e.g. 1536D) into smaller sub-vectors (e.g. 64 sub-vectors of 24 dimensions each). It runs k-means clustering on each sub-space to find 256 cluster centroids. Each sub-vector is then replaced with the 1-byte index of its nearest centroid. This compresses a 6,144-byte vector down to 64 bytes (a 96x reduction), allowing massive datasets to fit in RAM.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Relying on Post-Filtering for Multi-Tenant Security. Using post-filtering in multi-tenant environments is a catastrophic security vulnerability: if the top-K nearest neighbors belong to another tenant, the filter will strip them, but in leaky implementations, raw data can cross tenant boundaries. Always enforce hard pre-filtering or isolated collections.")
    st.error("**Trap 2 (Breaking Error)**: Indexing Without Re-building ANN Graphs. In many vector databases, inserting millions of vectors into an active HNSW index causes graph fragmentation. Always trigger an index compaction or build the index after batch bulk ingestion.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Keeping All Vectors in High-Cost RAM Without Disk Offloading. Storing billions of 1536D vectors in unquantized fp32 RAM is prohibitively expensive. Enable Product Quantization (PQ) or disk-backed vector storage (e.g. DiskANN / pgvector on NVMe).")
    st.warning("**Trap 4 (Quality Warning)**: Setting MMR Lambda Too Low. Setting lambda = 0 in MMR prioritizes diversity so heavily that completely irrelevant documents are selected. Keep lambda between 0.6 and 0.8.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Vector Database"):
        st.markdown("**Specialized database optimized for geometric distance search over multi-dimensional embeddings.**")
    with st.expander("🎴 Flashcard 2: HNSW Index"):
        st.markdown("**Hierarchical graph index enabling sub-millisecond logarithmic approximate search.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Metadata Pre-Filtering"):
        st.markdown("**Applies metadata filters before graph traversal to guarantee compliance and prevent empty results.**")
    with st.expander("🎴 Flashcard 4: Maximal Marginal Relevance"):
        st.markdown("**Balances query relevance with result diversity to eliminate redundant prompt context.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between relational B-Trees and vector ANN index graphs.", value=True)
chk2 = st.checkbox("I know how HNSW and IVF algorithms achieve sub-millisecond search latencies.", value=True)
chk3 = st.checkbox("I understand the difference between similarity search and Maximal Marginal Relevance (MMR).", value=True)
chk4 = st.checkbox("I can articulate why metadata pre-filtering is mandatory for multi-tenant enterprise RAG.", value=True)
chk5 = st.checkbox("I understand how Product Quantization compresses vectors by over 90% in RAM.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 vector database fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 14.1 Vector Database Fundamentals!")
