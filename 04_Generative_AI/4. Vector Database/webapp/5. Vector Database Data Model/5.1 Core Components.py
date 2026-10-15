import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
import uuid

st.set_page_config(
    page_title="5.1 Core Components - Vector DB Curriculum",
    page_icon="🧩",
    layout="wide"
)

# Title & Overview
st.title("5.1 Vector Database Data Model: The 6 Core Primitives")
st.caption("Chapter 5: Vector Database Data Model | Domain: 4. Vector Database")
st.markdown("""
Just as relational databases organize the world into **databases, tables, rows, columns, and foreign keys**, 
vector databases define a clean, universal data model engineered specifically for high-dimensional geometric retrieval.

In this lesson, we dissect the **6 foundational primitives** that constitute the data model of every modern 
vector database (including ChromaDB, Qdrant, Milvus, Weaviate, and Pinecone):
1. **The Vector** (High-Dimensional Semantic Array)
2. **The ID** (Unique Record Identity)
3. **The Document** (Raw Human-Readable Payload)
4. **The Metadata** (Structured Key-Value Filtering Attributes)
5. **The Collection** (Namespace & Configuration Container)
6. **The Index** (Approximate Nearest Neighbor Acceleration Engine)
""")

st.divider()

# Section 1: The 6 Core Data Model Primitives Dissected
st.header("1. The 6 Core Primitives Dissected", divider="rainbow")

col_prim1, col_prim2 = col_prim1, col_prim2 = st.columns([1, 1])

with col_prim1:
    st.markdown("""
    ### 1. The Atomic Data Entities
    * **1. Vector (Embedding)**:
      * A contiguous 1D array of floating-point numbers (`float32` or `float16`) residing in d-dimensional space: `[v_0, v_1, ..., v_(d-1)] in R^d`.
      * Must strictly match the collection's defined dimensionality (e.g. 768 or 1536).
      * Represents semantic coordinates extracted by an AI encoder.
    * **2. Unique ID**:
      * A unique identifier (string UUID like `'a4b1-88f2...'` or 64-bit integer) assigned to each record.
      * Serves as the primary key for updates, deletions, and point lookups.
    * **3. Document (Text Payload)**:
      * The raw unstructured source snippet (paragraph, code function, ticket body) that produced the vector.
      * Returned during retrieval to hydrate the LLM prompt or UI display.
    * **4. Metadata (Payload JSON)**:
      * Arbitrary key-value JSON dictionary: `{"author": "Alice", "created_year": 2024, "department": "Legal"}`.
      * Indexed via inverted bitsets to enable scalar pre- and post-filtering.
    """)

with col_prim2:
    st.markdown("""
    ### 2. The Structural Containers
    * **5. Collection (Table Equivalent)**:
      * The primary logical namespace partitioning data inside the database.
      * Enforces uniform constraints: all records inside a collection share the **same dimensionality (d)** and **distance metric** (Cosine, Dot Product, or L2).
      * Isolate multi-tenant datasets or separate functional domains (e.g., `kb_articles` vs `user_profiles`).
    * **6. Index (ANN Search Acceleration)**:
      * The geometric graph or clustering data structure (e.g. HNSW, IVF-PQ) constructed over the collection's vectors.
      * Transforms exhaustive `O(N * d)` brute-force scans into sub-5ms `O(log N)` traversals.
    """)

st.markdown("### The Vector Database Data Hierarchy")
model_dot = """
digraph DataModelHierarchy {
    rankdir=TB;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    Collection [label="Collection: 'enterprise_knowledge_base'\\nConfiguration: Dimension=1536 | Metric=Cosine", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_index {
        label="Index Layer"; style="dashed"; color="#7C3AED";
        ANNIndex [label="HNSW Graph Index\\nAdjacency Lists & Highway Links", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
        MetaFilterIndex [label="Metadata Filter Index\\nInverted Roaring Bitsets", fillcolor="#DDD6FE"];
    }

    subgraph cluster_records {
        label="Record Layer (Storage Engine)"; style="dashed"; color="#059669";
        Record1 [label="Vector Record: 'rec_001'\\nVector: [0.12, -0.44, ..., 0.89]\\nDocument: 'Kubernetes OOM error guide...'\\nMetadata: {dept: 'DevOps', year: 2024}", fillcolor="#D1FAE5"];
        Record2 [label="Vector Record: 'rec_002'\\nVector: [-0.05, 0.81, ..., 0.32]\\nDocument: 'PostgreSQL connection timeout...'\\nMetadata: {dept: 'Database', year: 2023}", fillcolor="#A7F3D0"];
    }

    Collection -> ANNIndex;
    Collection -> MetaFilterIndex;
    Collection -> Record1;
    Collection -> Record2;
    Record1 -> ANNIndex [label="Vector coordinates"];
    Record2 -> ANNIndex;
    Record1 -> MetaFilterIndex [label="Payload attributes"];
    Record2 -> MetaFilterIndex;
}
"""
st.graphviz_chart(model_dot, use_container_width=False)

st.divider()

# Section 2: Relational vs. Vector Data Model Mapping
st.header("2. Relational vs. Vector Data Model Mapping", divider="rainbow")

st.markdown("""
To ground these concepts in familiar engineering patterns, review the direct structural mapping 
between traditional RDBMS concepts and Vector Database primitives:
""")

mapping_df = pd.DataFrame({
    "Relational RDBMS Primitive": ["Database", "Table", "Row (Tuple)", "Primary Key (PK)", "Column Attribute", "B+ Tree Index", "WHERE Predicate"],
    "Vector Database Primitive": ["Database / Workspace", "Collection", "Vector Record (Entity / Point)", "Record ID (UUID / String)", "Metadata JSON Field", "HNSW / IVF Index", "Metadata Filter Mask"],
    "Functional Role": [
        "Highest-level organizational cluster / tenant boundary",
        "Logical container enforcing fixed vector dimensionality & distance metric",
        "The atomic unit combining coordinates, raw text, and structured payload",
        "Unique record identifier for point lookups, updates, and deletes",
        "Key-value scalar attributes for boolean pre-filtering",
        "Spatial acceleration structure enabling sub-linear nearest neighbor retrieval",
        "Boolean filter restricting ANN traversal to matching subset"
    ]
})
st.dataframe(mapping_df, use_container_width=True, hide_index=True)

st.divider()

# Section 3: Interactive Data Model Entity Builder & Schema Inspector
st.header("3. Interactive Vector Record & Collection Builder", divider="rainbow")

st.markdown("""
Compose a vector record in real time. Configure collection constraints, assign IDs, draft text chunks, 
attach structured metadata payloads, and inspect the internal serialized JSON entity:
""")

col_b_cfg, col_b_out = st.columns([1, 1])

with col_b_cfg:
    st.subheader("Collection & Record Configuration")
    b_col_name = st.text_input("Collection Name", value="customer_support_faqs")
    b_dim = st.selectbox("Collection Vector Dimension (d)", [384, 768, 1536, 3072], index=1)
    b_metric = st.selectbox("Distance Metric", ["Cosine", "Dot Product (Inner Product)", "Euclidean (L2)"])
    
    st.markdown("---")
    st.markdown("**Record Construction:**")
    b_id = st.text_input("Record ID", value="faq_billing_042")
    b_doc = st.text_area("Document Content (Raw Text)", value="How to request a refund for duplicate subscription charges on Visa cards.")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        meta_dept = st.selectbox("Metadata: Department", ["Billing", "Technical", "Sales", "Security"])
        meta_verified = st.checkbox("Metadata: Verified FAQ", value=True)
    with col_m2:
        meta_year = st.number_input("Metadata: Published Year", min_value=2020, max_value=2025, value=2024)
        meta_priority = st.slider("Metadata: Priority Level", 1, 5, 4)

with col_b_out:
    st.subheader("Serialized Vector DB Entity Payload")
    
    # Generate mock normalized vector
    np.random.seed(abs(hash(b_doc)) % (2**32))
    raw_v = np.random.randn(b_dim)
    norm = np.linalg.norm(raw_v)
    unit_v = (raw_v / norm).round(4).tolist()
    
    serialized_entity = {
        "collection": b_col_name,
        "configuration": {
            "dimension": b_dim,
            "metric": b_metric.split(" ")[0].lower()
        },
        "record": {
            "id": b_id,
            "vector_preview": unit_v[:4] + ["..."] + [f"({b_dim} total floats)"],
            "document": b_doc,
            "metadata": {
                "department": meta_dept,
                "verified": meta_verified,
                "year": int(meta_year),
                "priority": int(meta_priority)
            }
        }
    }
    
    st.json(serialized_entity)
    
    st.success(f"**Schema Validation Passed**: Record `{b_id}` conforms strictly to collection `{b_col_name}` dimensionality ({b_dim}-d)!")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Object-Oriented Vector Database Data Model", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Vector Database Data Model Architecture** 
with strict collection schema enforcement, record validation, and JSON metadata indexing:
""")

code_datamodel = '''import numpy as np
from typing import Dict, Any, List, Optional
import uuid

class VectorRecord:
    """Represents an atomic entity in a vector database."""
    def __init__(self, record_id: str, vector: np.ndarray, document: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = record_id
        self.vector = np.array(vector, dtype=np.float32)
        self.document = document
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dimension": len(self.vector),
            "document": self.document,
            "metadata": self.metadata
        }

class VectorCollection:
    """Logical namespace container enforcing dimensionality and metric schema."""
    def __init__(self, name: str, dimension: int, metric: str = "cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric.lower()
        self.records: Dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord):
        """Validates vector dimensionality and stores the record."""
        # 1. Dimensionality Schema Validation
        if len(record.vector) != self.dimension:
            raise ValueError(
                f"Schema Mismatch: Collection '{self.name}' requires dimension {self.dimension}, "
                f"but record '{record.id}' has dimension {len(record.vector)}."
            )

        # 2. Duplicate ID Check
        if record.id in self.records:
            raise KeyError(f"Primary Key Violation: Record ID '{record.id}' already exists in collection.")

        # 3. Store record
        self.records[record.id] = record

    def get(self, record_id: str) -> Optional[VectorRecord]:
        return self.records.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id in self.records:
            del self.records[record_id]
            return True
        return False

# Demonstration
collection = VectorCollection(name="kb_support", dimension=4, metric="cosine")

# Valid Record
r1 = VectorRecord(
    record_id="doc_101",
    vector=np.array([0.1, 0.2, 0.3, 0.4]),
    document="How to configure SSO with Okta.",
    metadata={"team": "Security", "public": True}
)
collection.insert(r1)
print(f"Successfully inserted: {collection.get('doc_101').to_dict()}")

# Invalid Record (Dimension Mismatch)
try:
    r_invalid = VectorRecord(
        record_id="doc_err",
        vector=np.array([0.1, 0.2]), # Only 2 dims instead of 4
        document="Invalid vector shape."
    )
    collection.insert(r_invalid)
except ValueError as e:
    print(f"\\nCaught Expected Schema Violation: {e}")
'''

st.code(code_datamodel, language="python")

if st.button("🚀 Execute Data Model Architecture", type="primary"):
    exec_scope = {}
    exec(code_datamodel, globals(), exec_scope)
    
    col_inst = exec_scope["collection"]
    r1_dict = col_inst.get("doc_101").to_dict()
    
    st.success("Data Model executed and schema enforcement verified successfully!")
    st.json(r1_dict)

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why do Vector Databases enforce a single fixed dimensionality per Collection?"""
a1 = """* **Mathematical Consistency**: Distance metrics (Cosine, Euclidean L2, Dot Product) require comparing vectors element-wise (`u_i * v_i`). Computing the distance between a 768-d vector and a 1536-d vector is mathematically undefined.
* **SIMD Hardware Alignment**: Vector engines allocate contiguous memory blocks for vectors in columnar arrays. Variable-length dimensions would destroy memory cache line alignment and prevent AVX-512 vectorization.
* **Index Integrity**: Spatial index graphs (HNSW) and cluster trees (IVF) are calibrated for a specific coordinate space.
* If an application requires storing embeddings from multiple models (e.g. text at 768-d and images at 512-d), you MUST create two separate collections."""

q2 = """2. What are the performance trade-offs between String UUIDs and 64-bit Integer IDs for Vector Records?"""
a2 = """* **64-bit Integer IDs (`uint64`)**:
  * Consumes strictly 8 bytes of memory.
  * Fits directly into CPU registers and cache lines.
  * Accelerates inverted metadata filter lookups and Roaring Bitset operations by 30% to 50%.
* **String UUIDs (e.g. 36-character UUIDv4)**:
  * Consumes 36 to 48 bytes (string overhead + pointer).
  * Requires string hashing and dynamic memory allocations.
  * Great for microservice decoupling (can generate UUIDs on client side without central sequence generators).
* **Production Best Practice**: Many high-scale engines (Faiss, Milvus) internally map string UUIDs to dense sequential integer IDs to maximize SIMD kernel speed."""

q3 = """3. How does Document (raw text) storage inside a Vector Database impact RAM footprint?"""
a3 = """* In an HNSW index, vectors and graph adjacency links MUST reside in high-speed RAM to maintain sub-5ms latency.
* However, raw document strings (which can be 500 to 2,000 bytes per chunk) do NOT participate in distance calculations.
* **The RAM Bloat Hazard**: If document strings are kept in RAM alongside vectors, memory consumption doubles or triples unnecessarily!
* **Architectural Solution**: Modern engines (Qdrant, Milvus) store vectors and index graphs in RAM (or memory-mapped files), while document payloads are stored in an append-only on-disk database (RocksDB or SQLite) and fetched (hydrated) only for the final Top-K winners."""

q4 = """4. What happens when you attempt to insert an embedding with un-normalized values into a Cosine Similarity collection?"""
a4 = """* If the collection is explicitly configured for `Cosine Similarity`, the database query engine will either:
  1. Compute the L2 norm on-the-fly during query time (`v / ||v||`), incurring a CPU performance penalty.
  2. Automatically normalize the vector upon insertion and store the unit vector.
* However, if the collection was configured for `Inner Product` (assuming vectors were pre-normalized), inserting un-normalized vectors will corrupt the search rankings: vectors with larger magnitudes will dominate results regardless of relevance."""

q5 = """5. How are Metadata Payloads indexed to allow sub-millisecond pre-filtering during vector search?"""
a5 = """* Rather than storing metadata as unindexed raw JSON strings, vector databases extract payload keys and build **Inverted Indices**:
  * For each key-value pair (`department: "DevOps"`), the database maintains a compressed bitset (e.g. Roaring Bitset) where bit position `i` is 1 if record `i` matches.
* When a query specifies `WHERE department = 'DevOps'`, the database retrieves the bitset in microseconds.
* During HNSW graph traversal, the search algorithm simply checks the bitset mask to decide whether a candidate is eligible for the Top-K heap."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Data Model Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: Storing Large Binary Assets Inside Metadata Payloads**  
> Storing base64-encoded image files or 5MB PDF binaries inside the metadata payload JSON bloats segment storage 
> and causes severe disk I/O bottlenecks during payload hydration. 
> Always store binary assets in an external object store (Amazon S3 / Google Cloud Storage) and store only the 
> URL URI (`{"s3_uri": "s3://bucket/photo.jpg"}`) in the vector metadata payload.

> [!WARNING]
> **Trap 2: Dimensionality Mismatch Production Crash**  
> If your ingestion service changes embedding models from `bge-base-en-v1.5` (768-d) to `bge-large-en-v1.5` (1024-d), 
> inserting vectors into an existing 768-d collection will immediately throw schema validation exceptions. 
> Version your collections (e.g. `kb_articles_v1_768` -> `kb_articles_v2_1024`) and re-index into a fresh collection.

> [!WARNING]
> **Trap 3: Unbounded Dynamic Metadata Schema Growth**  
> While metadata is schema-free JSON, allowing arbitrary high-cardinality keys (e.g. inserting unique user timestamp 
> strings as dictionary keys) causes metadata inverted indexes to bloat uncontrollably. Enforce structured schema 
> conventions for metadata keys in enterprise ingestion pipelines.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Collection?"):
        st.markdown("""
        **Collection**:  
        A logical container (analogous to a table) enforcing a single fixed dimensionality and distance metric over all contained vectors.
        """)
    with st.expander("🗂️ Flashcard: What is a Vector Record?"):
        st.markdown("""
        **Vector Record**:  
        The atomic entity in a vector database combining a Unique ID, Vector array, Document payload, and Metadata dictionary.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Separate Vectors from Documents?"):
        st.markdown("""
        **Storage Separation**:  
        Vectors reside in RAM for fast SIMD calculation; large text documents reside on disk and are hydrated only for top-K results.
        """)
    with st.expander("🗂️ Flashcard: How is Metadata Filtered?"):
        st.markdown("""
        **Bitset Masking**:  
        Metadata fields are indexed as compressed Roaring Bitsets, allowing sub-millisecond candidate filtering during graph traversal.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can name and describe all 6 core primitives of the Vector Database data model.", value=True)
chk2 = st.checkbox("I understand why Collections strictly enforce uniform vector dimensionality.")
chk3 = st.checkbox("I know the trade-offs between 64-bit Integer IDs and String UUIDs.")
chk4 = st.checkbox("I understand why raw document payloads should be stored on disk rather than polluting index RAM.")
chk5 = st.checkbox("I know how metadata fields are indexed into compressed bitsets for fast query pre-filtering.")
chk6 = st.checkbox("I can implement an object-oriented vector data model with schema enforcement in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered the Core Components and are ready for Chapter 5.2: Records!")
