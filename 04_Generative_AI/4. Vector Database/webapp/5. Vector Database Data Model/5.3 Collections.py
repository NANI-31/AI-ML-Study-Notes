import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="5.3 Collections - Vector DB Curriculum",
    page_icon="📚",
    layout="wide"
)

# Title & Overview
st.title("5.3 Collections: Namespaces, Configurations, Ingestion Modes & Multi-Tenancy")
st.caption("Chapter 5: Vector Database Data Model | Domain: 4. Vector Database")
st.markdown("""
In a vector database, the **Collection** (analogous to a table in an RDBMS) is the top-level organizational namespace 
that defines the **geometric rules of engagement** for a group of vectors.

Every collection enforces a strict contract: all contained records must share the **identical vector dimensionality (d)** 
and be evaluated using the **same distance metric** (Cosine, Dot Product, or Euclidean L2).

In this lesson, we master the complete architectural engineering of collections:
1. Creating & Naming Collections with Enterprise Conventions
2. Ingestion Modes: Automated Client-Side Embedding vs. Pre-Computed Vector Insertion
3. Deep Index Configuration (Tuning HNSW `M` and `efConstruction`)
4. Multi-Tenancy Architecture: **Collection-per-Tenant** vs. **Shared Collection with Metadata Partitioning**
5. Lifecycle Management: Monitoring Collection Stats, Compaction & Safe Deletion
""")

st.divider()

# Section 1: Anatomy of a Collection & Configuration
st.header("1. Anatomy of a Collection & Configuration Parameters", divider="rainbow")

col_col1, col_col2 = col_col1, col_col2 = st.columns([1, 1])

with col_col1:
    st.markdown("""
    ### 1. The Collection Contract
    * **Dimensionality Schema (Immutable)**:
      * Once created with `dimension = 1536`, a collection permanently rejects any vector with `d != 1536`.
      * Upgrading an embedding model requires creating a new collection and executing a background re-index.
    * **Distance Metric (Immutable)**:
      * Configured as `Cosine`, `Dot Product (Inner Product)`, or `Euclidean (L2)`.
      * Dictates which hardware SIMD distance kernel is dispatched during search.
    * **Enterprise Naming Best Practices**:
      * Use lowercase alphanumeric characters, underscores, and semantic versioning tags:
        * `prod_support_faq_v2_768`
        * `tenant_acme_legal_docs_1536`
        * `staging_product_catalog_vit_512`
    """)

with col_col2:
    st.markdown("""
    ### 2. Low-Level HNSW Index Tuning Parameters
    * **`M` (Max Bi-directional Links per Node)**:
      * Controls the degree (number of edges) of each node in the graph.
      * *Default*: `16` to `32`.
      * *Trade-off*: Higher `M` (e.g. `64`) improves search recall on complex, clustered datasets at the cost of higher RAM usage and longer index build times.
    * **`efConstruction` (Exploration Depth during Indexing)**:
      * Size of the candidate queue evaluated when inserting a new vector into the graph.
      * *Default*: `100` to `200`.
      * *Trade-off*: Higher `efConstruction` builds a higher-quality graph index, increasing write latency during ingestion without affecting read search latency.
    """)

st.markdown("### Collection Configuration Parameters Breakdown")
cfg_df = pd.DataFrame({
    "Configuration Field": ["dimension", "metric", "hnsw.M", "hnsw.efConstruction", "quantization", "wal_flush_interval"],
    "Type": ["Integer (Fixed)", "Enum (Cosine/IP/L2)", "Integer (8 - 64)", "Integer (64 - 512)", "Enum (None/SQ8/PQ)", "Milliseconds"],
    "Default Value": ["768 or 1536", "Cosine", "16", "128", "None (float32)", "1000 ms"],
    "Impact on System": [
        "Enforces strict input vector validation",
        "Selects SIMD distance instruction set",
        "RAM footprint per vector; link connectivity",
        "Index build throughput vs graph quality",
        "Cuts vector RAM footprint by 75% to 90%",
        "Write durability SLA vs write latency"
    ]
})
st.dataframe(cfg_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Ingestion Modes: Auto-Embedding vs Pre-Computed
st.header("2. Ingestion Modes: Auto-Embedding vs. Pre-Computed Embeddings", divider="rainbow")

st.markdown("""
Vector databases support two distinct ingestion modes: **Automated In-Database Embedding** and **Pre-Computed Vector Ingestion**.
Understanding the trade-offs is critical for production scalability:
""")

col_ing1, col_ing2 = col_ing1, col_ing2 = st.columns([1, 1])

with col_ing1:
    st.markdown("""
    ### Mode A: Auto-Embedding (Client SDK Helper)
    * **How it works (e.g. ChromaDB / Weaviate)**:
      * You pass raw text: `collection.add(documents=["Text..."], ids=["id1"])`.
      * The database client binds a registered embedding model (e.g. OpenAI or FastEmbed) and computes vectors automatically under the hood.
    * **Pros**:
      * Extreme developer convenience; requires minimal boilerplate for quick prototypes and notebooks.
    * **Cons (The Production Bottleneck)**:
      * Tightly couples your database client to external AI APIs.
      * Harder to parallelize batch embedding jobs across GPU clusters independently of database ingestion.
    """)

with col_ing2:
    st.markdown("""
    ### Mode B: Pre-Computed Embeddings (Production Enterprise)
    * **How it works (e.g. Qdrant / Milvus)**:
      * Your dedicated worker pipeline generates vectors upstream: `vec = model.encode(text)`.
      * You pass the pre-computed array: `collection.add(ids=[...], vectors=[vec], documents=[...])`.
    * **Pros**:
      * **Complete Pipeline Decoupling**: Vector generation can run across distributed GPU Ray / Celery workers.
      * Allows local caching of embeddings, reducing costly external API calls.
      * The vector database focuses 100% of its resources on storage and indexing without embedding compute overhead.
    """)

st.divider()

# Section 3: Multi-Tenancy Architecture Patterns
st.header("3. Multi-Tenancy Architecture: Collection-per-Tenant vs. Shared Collection", divider="rainbow")

st.markdown("""
In SaaS applications serving hundreds or thousands of corporate customers, how do you isolate data 
between different organizations?
""")

col_mt1, col_mt2 = col_mt1, col_mt2 = st.columns([1, 1])

with col_mt1:
    st.markdown("""
    #### Pattern A: Collection-per-Tenant (Hard Isolation)
    * **Design**: Every customer gets their own isolated collection: `tenant_101_docs`, `tenant_102_docs`.
    * **Strengths**:
      * Absolute physical isolation: zero risk of cross-tenant data leakage.
      * Dropping a tenant's data is as simple as `drop_collection()`.
    * **The Fatal Flaw (Collection Sprawl)**:
      * Every collection maintains its own in-memory HNSW index structures, memory buffers, and file descriptors.
      * Having 5,000 collections—even if each has only 100 vectors—consumes tens of gigabytes of idle RAM and degrades database startup times!
      * **Rule of Thumb**: Only use if tenant count is small (< 100) and tenants have massive datasets.
    """)

with col_mt2:
    st.markdown("""
    #### Pattern B: Shared Collection with Metadata Partitioning (Soft Isolation)
    * **Design**: All tenants share a single large collection: `enterprise_documents`.
    * Every record includes a tenant identifier in its metadata: `{"tenant_id": 42}`.
    * Every query injects a mandatory filter: `filter={"tenant_id": 42}`.
    * **Strengths**:
      * Highly resource-efficient: single shared HNSW graph.
      * Easily scales to millions of small and medium tenants without RAM bloat.
    * **Requirement**:
      * Requires a vector database with **Single-Stage Bitset Pre-Filtering** to guarantee that the search visits only nodes belonging to the target tenant without recall loss.
    """)

st.markdown("### Multi-Tenancy Architecture Comparison")
mt_dot = """
digraph MultiTenancy {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    subgraph cluster_cpt {
        label="Pattern A: Collection-per-Tenant (Isolated)"; style="dashed"; color="#DC2626";
        ColA [label="Collection: tenant_01\\nHNSW Graph in RAM", fillcolor="#FEE2E2"];
        ColB [label="Collection: tenant_02\\nHNSW Graph in RAM", fillcolor="#FEE2E2"];
        ColC [label="Collection: tenant_03\\nHNSW Graph in RAM", fillcolor="#FEE2E2"];
        NoticeSprawl [label="Risk: RAM Exhaustion at 1k+ tenants!", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];
        ColA -> NoticeSprawl; ColB -> NoticeSprawl; ColC -> NoticeSprawl;
    }

    subgraph cluster_shared {
        label="Pattern B: Shared Collection with Metadata Partitioning"; style="dashed"; color="#059669";
        SharedCol [label="Shared Collection: 'all_tenant_docs'\\nSingle Unified HNSW Graph", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        FilterMask [label="Single-Stage Filter Mask\\nWHERE tenant_id = 42\\nSub-millisecond Bitset Check", fillcolor="#A7F3D0"];
        SharedSuccess [label="Scales to 100k+ tenants efficiently!\\n(Production Industry Standard)", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
        SharedCol -> FilterMask -> SharedSuccess;
    }
}
"""
st.graphviz_chart(mt_dot, use_container_width=False)

st.divider()

# Section 4: Interactive Multi-Collection Workspace Manager
st.header("4. Interactive Multi-Collection Workspace Manager", divider="rainbow")

st.markdown("""
Simulate managing multiple collections in an enterprise vector database cluster. 
Create collections with customized dimensionality and distance metrics, inspect memory usage, 
and test multi-tenant query isolation:
""")

# Session State for Multi-Collection DB
if "collections_db" not in st.session_state:
    st.session_state.collections_db = {
        "kb_support_v1": {"dimension": 768, "metric": "cosine", "hnsw_M": 16, "hnsw_ef": 128, "count": 14200, "tenant_strategy": "shared"},
        "legal_contracts_v2": {"dimension": 1536, "metric": "cosine", "hnsw_M": 32, "hnsw_ef": 200, "count": 8900, "tenant_strategy": "shared"},
        "product_catalog_vit": {"dimension": 512, "metric": "dot", "hnsw_M": 16, "hnsw_ef": 128, "count": 65000, "tenant_strategy": "isolated"}
    }

col_mgr_in, col_mgr_out = st.columns([1, 1])

with col_mgr_in:
    st.subheader("Collection Management Console")
    col_action = st.radio("Action", ["Inspect Existing Collection", "Create New Collection", "Drop / Delete Collection"])
    
    if col_action == "Inspect Existing Collection":
        inspect_name = st.selectbox("Select Collection", list(st.session_state.collections_db.keys()))
        info = st.session_state.collections_db[inspect_name]
        st.info(f"""
        **Collection:** `{inspect_name}`
        * **Vector Dimensionality:** `{info['dimension']}`
        * **Distance Metric:** `{info['metric'].upper()}`
        * **HNSW Configuration:** `M = {info['hnsw_M']}`, `efConstruction = {info['hnsw_ef']}`
        * **Current Record Count:** `{info['count']:,} vectors`
        * **Multi-Tenancy Mode:** `{info['tenant_strategy'].capitalize()}`
        """)
        
    elif col_action == "Create New Collection":
        new_col_name = st.text_input("New Collection Name", value="employee_handbook_v1")
        new_col_dim = st.selectbox("Dimensionality (d)", [384, 768, 1024, 1536, 3072], index=3)
        new_col_metric = st.selectbox("Distance Metric", ["cosine", "dot", "l2"])
        new_col_M = st.slider("HNSW M (Links per Node)", 8, 64, 16)
        new_col_ef = st.slider("HNSW efConstruction", 64, 512, 128)
        new_col_mt = st.selectbox("Multi-Tenancy Pattern", ["shared (Metadata Filtered)", "isolated (Single Tenant)"])
        
        if st.button("🚀 Initialize Collection", type="primary"):
            if new_col_name in st.session_state.collections_db:
                st.error(f"Collection `{new_col_name}` already exists!")
            else:
                st.session_state.collections_db[new_col_name] = {
                    "dimension": new_col_dim,
                    "metric": new_col_metric,
                    "hnsw_M": new_col_M,
                    "hnsw_ef": new_col_ef,
                    "count": 0,
                    "tenant_strategy": new_col_mt.split(" ")[0]
                }
                st.success(f"Collection `{new_col_name}` initialized successfully with dimension {new_col_dim}!")
                
    else: # Drop
        drop_name = st.selectbox("Select Collection to Drop", list(st.session_state.collections_db.keys()))
        if st.button("⚠️ Permanently Drop Collection", type="primary"):
            del st.session_state.collections_db[drop_name]
            st.warning(f"Collection `{drop_name}` dropped and its index memory purged!")

with col_mgr_out:
    st.subheader("Cluster Collection Inventory")
    
    summary_rows = []
    total_vectors = 0
    total_est_ram_mb = 0.0
    
    for cname, cinfo in st.session_state.collections_db.items():
        vcount = cinfo["count"]
        vdim = cinfo["dimension"]
        # Estimate RAM: (N * d * 4 bytes) + HNSW graph overhead (~M * 8 bytes * N)
        v_ram = (vcount * vdim * 4) / (1024 * 1024)
        g_ram = (vcount * cinfo["hnsw_M"] * 8) / (1024 * 1024)
        col_ram_mb = v_ram + g_ram
        
        total_vectors += vcount
        total_est_ram_mb += col_ram_mb
        
        summary_rows.append({
            "Collection Name": cname,
            "Dimension": vdim,
            "Metric": cinfo["metric"].upper(),
            "Record Count": f"{vcount:,}",
            "Est. RAM Footprint": f"{col_ram_mb:.1f} MB",
            "Multi-Tenancy": cinfo["tenant_strategy"]
        })
        
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric("Total Indexed Vectors Across Cluster", f"{total_vectors:,}")
    with c_m2:
        st.metric("Total Estimated Index RAM", f"{total_est_ram_mb:.1f} MB")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Multi-Collection Management Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Multi-Collection Vector Database Engine** featuring 
**Collection-level schema enforcement**, **HNSW configuration tuning**, and **Multi-Tenant query routing**:
""")

code_multi_col = '''import numpy as np
from typing import Dict, Any, List, Optional

class EnterpriseCollection:
    """Represents an isolated collection with dedicated index and schema."""
    def __init__(self, name: str, dimension: int, metric: str = "cosine", hnsw_M: int = 16):
        self.name = name
        self.dimension = dimension
        self.metric = metric.lower()
        self.hnsw_M = hnsw_M
        self.records: Dict[str, Dict[str, Any]] = {}

    def insert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any], doc: str = ""):
        assert len(vector) == self.dimension, f"Vector dimension must be {self.dimension}"
        norm = np.linalg.norm(vector)
        unit_vec = vector / (norm if norm > 0 else 1.0)
        self.records[record_id] = {
            "id": record_id,
            "vector": unit_vec,
            "metadata": metadata,
            "document": doc
        }

    def search(self, query_vec: np.ndarray, top_k: int = 2, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        q_norm = query_vec / np.linalg.norm(query_vec)
        
        # Apply metadata filter
        candidates = []
        for rid, r in self.records.items():
            if filter_dict:
                match = all(r["metadata"].get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            candidates.append(r)

        if not candidates:
            return []

        matrix = np.array([c["vector"] for c in candidates])
        scores = np.dot(matrix, q_norm)
        ranked = np.argsort(scores)[::-1][:top_k]

        return [{
            "id": candidates[i]["id"],
            "score": float(scores[i]),
            "metadata": candidates[i]["metadata"],
            "document": candidates[i]["document"]
        } for i in ranked]

class EnterpriseVectorDBCluster:
    """Cluster-level manager orchestrating multiple collections."""
    def __init__(self):
        self.collections: Dict[str, EnterpriseCollection] = {}

    def create_collection(self, name: str, dimension: int, metric: str = "cosine", hnsw_M: int = 16) -> EnterpriseCollection:
        if name in self.collections:
            raise KeyError(f"Collection '{name}' already exists.")
        col = EnterpriseCollection(name, dimension, metric, hnsw_M)
        self.collections[name] = col
        return col

    def get_collection(self, name: str) -> EnterpriseCollection:
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' not found.")
        return self.collections[name]

    def drop_collection(self, name: str) -> bool:
        if name in self.collections:
            del self.collections[name]
            return True
        return False

# Demonstration
cluster = EnterpriseVectorDBCluster()

# 1. Create Collection A (768-d Support FAQs)
col_support = cluster.create_collection("support_faqs", dimension=4, metric="cosine")
col_support.insert("faq_1", np.array([0.9, 0.1, 0.1, 0.0]), {"tenant_id": 101}, "How to reset password?")
col_support.insert("faq_2", np.array([0.9, 0.1, 0.1, 0.0]), {"tenant_id": 102}, "How to reset password (Tenant 102)?")

# 2. Multi-Tenant Query with Filter
query = np.array([0.9, 0.1, 0.1, 0.0])
hits = col_support.search(query, top_k=1, filter_dict={"tenant_id": 101})

print(f"Cluster Active Collections: {list(cluster.collections.keys())}")
print(f"Isolated Tenant 101 Search Hit: ID={hits[0]['id']} | Tenant={hits[0]['metadata']['tenant_id']}")
'''

st.code(code_multi_col, language="python")

if st.button("🚀 Execute Multi-Collection Cluster Demo", type="primary"):
    exec_scope = {}
    exec(code_multi_col, globals(), exec_scope)
    
    col_list = list(exec_scope["cluster"].collections.keys())
    hits_out = exec_scope["hits"]
    
    st.success("Cluster Manager executed successfully!")
    st.markdown(f"**Active Collections:** `{col_list}`")
    st.markdown(f"**Tenant-Filtered Query Result:** `ID: {hits_out[0]['id']}` | Metadata: `{hits_out[0]['metadata']}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. In SaaS multi-tenancy, when should you choose Collection-per-Tenant versus a Shared Collection with Metadata Partitioning?"""
a1 = """* **Collection-per-Tenant (Isolated)**:
  * Choose when you have a small number of enterprise tenants (< 100) who require strict regulatory or physical data isolation (e.g. defense, private banking).
  * Makes GDPR "Right to be Forgotten" instantaneous: deleting a customer's data is simply `drop_collection()`.
* **Shared Collection with Metadata Partitioning (Soft)**:
  * Choose for B2B or B2C SaaS serving hundreds to millions of tenants.
  * Every collection consumes memory overhead for HNSW graph buffers, segment metadata, and OS file descriptors.
  * Maintaining 10,000 idle collections causes severe RAM exhaustion and slow startup times. A single shared collection with single-stage metadata filtering scales to millions of tenants with minimal overhead."""

q2 = """2. What is the impact of tuning HNSW `M` from 16 to 64 when creating a Collection?"""
a2 = """* **`M`** controls the maximum number of bi-directional edges connected to each node in the HNSW graph.
* **Positive Impact**:
  * Higher `M` (e.g. 64) creates denser graph connectivity with more alternative routing paths.
  * Drastically improves search recall on highly clustered or high-dimensional datasets where sparse graphs get trapped in local minima.
* **Negative Impact**:
  * Graph memory consumption scales linearly: each edge consumes 4 to 8 bytes. At M = 64, graph memory overhead quadruples!
  * Index building time increases significantly because every inserted vector must establish and prune 64 bidirectional connections."""

q3 = """3. Why is it dangerous to change an embedding model without creating a fresh Collection?"""
a3 = """* A vector database does not store the original semantic meaning of words; it only stores numerical coordinates generated by a specific neural network.
* Vectors from `text-embedding-ada-002` occupy a completely different mathematical universe than vectors from `text-embedding-3-small`—even if both output 1536 dimensions!
* If you insert new vectors into an existing collection without re-indexing the existing data, the HNSW graph will connect incompatible points.
* Searches will return nonsensical results. Upgrading an embedding model requires initializing a new collection and running a backfill re-indexing pipeline."""

q4 = """4. What is the difference between HNSW `efConstruction` and `efSearch`?"""
a4 = """* **`efConstruction` (Write Path / Indexing)**:
  * Configured during collection creation.
  * Controls the size of the dynamic candidate list evaluated when inserting a vector into the graph.
  * Dictates index quality: higher `efConstruction` produces better graph routing links at the cost of slower ingestion throughput.
* **`efSearch` (Read Path / Querying)**:
  * Configured at query time.
  * Controls the size of the candidate list evaluated during search traversal.
  * Tunes runtime query latency vs search recall on-the-fly without rebuilding the index."""

q5 = """5. How does a production Vector Database safely handle `drop_collection()` under active query load?"""
a5 = """* Immediate deletion of underlying memory-mapped files while worker threads are executing SIMD queries causes segmentation faults (`SIGSEGV`).
* **The Safe Drop Protocol**:
  1. **Mark Dropping**: The collection is marked in the catalog as `DROPPING`, rejecting all incoming search and write requests with `CollectionNotFoundException`.
  2. **Drain In-Flight Queries**: Wait for active query worker threads to complete (typically a 5-second timeout).
  3. **Unmap Memory**: Unmap memory-mapped HNSW graph segments (`munmap`).
  4. **Delete Physical Files**: Remove segment and WAL files from disk."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Collection Management Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The "One Collection Per User" Memory Exhaustion Disaster**  
> A startup building a personal AI note-taking app created a new Chroma/Milvus collection for every registered user. 
> At 25,000 users, the database pod crashed with an Out-of-Memory (OOM) error despite storing less than 50 MB 
> of actual text, because 25,000 HNSW graph headers exhausted all available RAM! 
> Always use a shared collection with `{"user_id": 123}` metadata filtering for user-level multi-tenancy.

> [!WARNING]
> **Trap 2: Distance Metric Mismatch Between Model and Collection**  
> If an embedding model was trained for Cosine Similarity (e.g. OpenAI embeddings), but you create the collection 
> with `metric="l2"` without pre-normalizing vectors, search rankings will be corrupted by document length variances. 
> Always check the model documentation and match the collection distance metric accordingly.

> [!WARNING]
> **Trap 3: Inadequate Index Sizing for Batch Ingestion**  
> Setting `efConstruction = 512` during bulk ingestion of 10 million documents causes index construction to take days! 
> When performing massive initial backfills, set `efConstruction` to a moderate value (e.g. 128), batch your inserts, 
> or ingest into a flat collection and build the HNSW index in a single batch pass after data loading completes.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is a Collection Schema?"):
        st.markdown("""
        **Collection Schema**:  
        The immutable contract specifying fixed vector dimensionality (d) and distance metric (Cosine, L2, Dot Product) for all member vectors.
        """)
    with st.expander("🗂️ Flashcard: What Does HNSW `M` Control?"):
        st.markdown("""
        **HNSW `M`**:  
        The maximum number of bi-directional connections per graph node. Higher `M` increases recall and RAM usage.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How Scale Multi-Tenancy?"):
        st.markdown("""
        **Multi-Tenancy Scaling**:  
        Use a shared collection with metadata partitioning (`tenant_id`) and bitset pre-filtering to avoid the RAM overhead of thousands of idle collections.
        """)
    with st.expander("🗂️ Flashcard: What is `efConstruction`?"):
        st.markdown("""
        **`efConstruction`**:  
        The exploration depth evaluated during vector insertion, determining index quality and ingestion throughput.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why Collections enforce strict, immutable dimensionality and distance metrics.", value=True)
chk2 = st.checkbox("I know how to tune HNSW `M` and `efConstruction` for RAM and index quality trade-offs.")
chk3 = st.checkbox("I can articulate when to use Auto-Embedding helpers versus Pre-Computed vector ingestion.")
chk4 = st.checkbox("I understand the architectural differences between Collection-per-Tenant and Shared Collections.")
chk5 = st.checkbox("I know why changing embedding models requires creating a brand new collection.")
chk6 = st.checkbox("I can implement a multi-collection cluster management engine in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding achievement! You have mastered Collections and completed Chapter 5 of the Vector Database curriculum!")
