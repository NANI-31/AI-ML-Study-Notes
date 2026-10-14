import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class VectorOperationsManager:
    """
    Simulates enterprise Vector Database CRUD operations, index management,
    soft deletion / tombstoning, and metadata filtering.
    """
    def __init__(self, collection_name: str = "rag_knowledge_base", dimension: int = 4):
        self.collection_name = collection_name
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.tombstones: set = set()
        self.index_state: str = "OPTIMIZED"
        self.operations_log: List[str] = []

    def create_collection(self, name: str, dim: int) -> None:
        self.collection_name = name
        self.dimension = dim
        self.vectors.clear()
        self.payloads.clear()
        self.tombstones.clear()
        self.index_state = "EMPTY"
        self.operations_log.append(f"COLLECTION_CREATED: '{name}' (dim={dim})")

    def insert(self, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        v_norm = v / norm if norm > 0 else v
        self.vectors[doc_id] = v_norm
        self.payloads[doc_id] = payload
        if doc_id in self.tombstones:
            self.tombstones.remove(doc_id)
        self.index_state = "MODIFIED_NEEDS_COMPACT"
        self.operations_log.append(f"INSERT: doc_id='{doc_id}'")
        return True

    def update(self, doc_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> bool:
        if doc_id not in self.vectors or doc_id in self.tombstones:
            return False
        if vector is not None:
            v = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(v)
            self.vectors[doc_id] = v / norm if norm > 0 else v
        if payload is not None:
            self.payloads[doc_id].update(payload)
        self.index_state = "MODIFIED_NEEDS_COMPACT"
        self.operations_log.append(f"UPDATE: doc_id='{doc_id}'")
        return True

    def delete(self, doc_id: str, hard_delete: bool = False) -> bool:
        if doc_id not in self.vectors:
            return False
        if hard_delete:
            del self.vectors[doc_id]
            del self.payloads[doc_id]
            if doc_id in self.tombstones:
                self.tombstones.remove(doc_id)
            self.operations_log.append(f"HARD_DELETE: doc_id='{doc_id}'")
        else:
            self.tombstones.add(doc_id)
            self.operations_log.append(f"SOFT_DELETE_TOMBSTONE: doc_id='{doc_id}'")
        self.index_state = "MODIFIED_NEEDS_COMPACT"
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qv)
        qv = qv / norm if norm > 0 else qv

        active_docs = [d_id for d_id in self.vectors.keys() if d_id not in self.tombstones]
        results = []
        for d_id in active_docs:
            if category_filter and self.payloads[d_id].get("category") != category_filter:
                continue
            sim = float(np.dot(qv, self.vectors[d_id]))
            results.append({
                "id": d_id,
                "score": round(sim, 4),
                "title": self.payloads[d_id].get("title", ""),
                "category": self.payloads[d_id].get("category", "")
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def compact_index(self) -> int:
        removed_count = len(self.tombstones)
        for d_id in list(self.tombstones):
            if d_id in self.vectors:
                del self.vectors[d_id]
            if d_id in self.payloads:
                del self.payloads[d_id]
        self.tombstones.clear()
        self.index_state = "OPTIMIZED"
        self.operations_log.append(f"INDEX_COMPACTED: pruned {removed_count} tombstones")
        return removed_count


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="8.3 Vector Database Operations - RAG Master Curriculum",
    page_icon="⚡",
    layout="wide"
)

st.title("8.3 Vector Database Operations: The Complete CRUD Lifecycle, Query Routing, and Index Management")
st.caption("Chapter 8: Vector Databases | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
A production vector database is not a static data warehouse; it is a dynamic operational store. 
Enterprise RAG architectures require robust **CRUD (Create, Read, Update, Delete)** operations, 
idempotent batch ingestion, atomic in-place updates, tombstoned soft-deletions, metadata filtering, 
and background index compaction.

Mastering vector operations ensures your data layer remains performant, synchronized, and resilient 
under continuous production ingest and query workloads.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Vector Database CRUD Operations Lifecycle", divider="rainbow")
st.markdown("""
The architecture below outlines how CRUD requests flow through the API gateway, write-ahead log (WAL), 
active memory segments, and background index compaction workers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    crud_bp_dot = """
    digraph VectorCRUDLifecycle {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_clients {
            label = "1. Client Operations Gateway";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            upsert_req [label="Upsert / Insert / Update Request\\n(ID + Vector + Metadata Payload)", fillcolor="#0F172A", color="#10B981"];
            delete_req [label="Delete Request\\n(Point IDs or Filter Predicate)", fillcolor="#0F172A", color="#EF4444"];
            search_req [label="Similarity Search Request\\n(Query Vector + Top-K + Filter)", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_engine_core {
            label = "2. Storage & Memory Subsystem";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            wal_log [label="Write-Ahead Log (WAL)\\nSynchronous Append for Durability", fillcolor="#334155", color="#F59E0B"];
            active_mem [label="Mutable Memory Segment\\nIn-Memory Buffer & Tombstone Bitset", fillcolor="#334155", color="#10B981"];
            immutable_seg [label="Immutable Disk Segments\\nFlushed HNSW Graphs & Inverted Index", fillcolor="#334155", color="#8B5CF6"];

            wal_log -> active_mem [label="Apply Mutations"];
            active_mem -> immutable_seg [label="Flush Segment"];
        }

        subgraph cluster_background {
            label = "3. Background Compaction & Search Routing";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            search_router [label="Query Coordinator & Merger\\nMulti-segment Graph Traversal", fillcolor="#0F172A", color="#38BDF8"];
            compactor [label="Vacuum & Compaction Worker\\nPurge Tombstones & Merge HNSW Segments", fillcolor="#0F172A", color="#F59E0B"];

            immutable_seg -> compactor [label="Periodic Merge"];
            compactor -> immutable_seg [label="Optimized Segment"];
            immutable_seg -> search_router [label="Segment Results"];
            active_mem -> search_router [label="Fresh Data"];
        }

        upsert_req -> wal_log [label="Append"];
        delete_req -> wal_log [label="Tombstone"];
        search_req -> search_router [label="Execute k-NN"];
    }
    """
    st.graphviz_chart(crud_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production engines (like **Qdrant**, **Milvus**, or **Pinecone**), deletions and updates are **never executed directly on the live HNSW graph in real time**. 
Because breaking graph edges dynamically is computationally prohibitive, engines mark deleted IDs in a **tombstone bitset** in RAM. 
When searches run, candidate nodes matching the tombstone bitset are instantly filtered out. 
A background **compactor (vacuum worker)** runs asynchronously to physically purge tombstones and rebuild dense graph segments.
""")

st.divider()

# ==============================================================================
# SECTION 2: CREATING COLLECTIONS
# ==============================================================================

st.header("2. Creating Collections and Defining Schemas", divider="rainbow")
st.markdown("""
A **collection** (or index/namespace) is an isolated organizational namespace inside a vector database, 
equivalent to a table in relational databases.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Collection Configuration Parameters")
    st.markdown("""
    When creating a collection, several architectural parameters must be permanently defined:
    1. **Vector Dimension**:
       Specifies the exact length of coordinate arrays (e.g., 1536 for OpenAI `text-embedding-3-small`, 1024 for Cohere v3). Vectors of differing dimensions will be rejected.
    2. **Distance Metric**:
       Defines the geometric proximity formula:
       * **Cosine Similarity**: Directional angle match (standard for text embeddings).
       * **Dot Product**: Scaled projection (requires unit-normalized vectors).
       * **Euclidean (L2)**: Physical distance in coordinate space.
    3. **Index Type**:
       Determines the indexing algorithm (e.g., HNSW, IVF, Flat).
    4. **Payload Schema Enforcement**:
       Whether scalar metadata fields require strict typing (string, integer, geo) or accept flexible JSON payloads.
    """)

with col2:
    st.subheader("Code Example: Creating Collections Across Engines")
    st.code('''# Qdrant Collection Creation
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="enterprise_knowledge",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# Chroma Collection Creation
collection = chroma_client.get_or_create_collection(
    name="enterprise_knowledge",
    metadata={"hnsw:space": "cosine"}
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: INSERTING VECTORS
# ==============================================================================

st.header("3. Inserting Vectors & Batch Upsert Operations", divider="rainbow")
st.markdown("""
Vector insertion requires moving high-dimensional coordinate arrays and associated document text 
into the database storage tier.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Batching vs. Single Ingestion")
    st.markdown("""
    * **Single Point Inserts**:
      Inserting vectors one-by-one incurs massive network round-trip overhead and repeatedly triggers lock contention in write-ahead logs.
    * **Batch Upsert (Idempotent)**:
      Ingesting vectors in batches of 100 to 500 records maximizes network payload efficiency, saturates vector normalization SIMD units, and reduces disk sync calls.
    * **Upsert Semantics**:
      `Upsert` (Insert or Update) is preferred over plain `Insert`. If a record with ID `chunk_102` already exists, upsert overwrites its vector and metadata, guaranteeing idempotency during re-ingestion pipelines.
    """)

with col2:
    st.subheader("Payload Structure Example")
    st.code('''# Batch Upsert Structure
points = [
    {
        "id": "doc_001_chunk_00",
        "vector": [0.021, -0.045, 0.112, 0.089, ...], # 1536 floats
        "payload": {
            "title": "Quarterly Financial Report Q3",
            "department": "Finance",
            "year": 2024,
            "confidential": True,
            "text": "Revenue increased by 14% year-over-year..."
        }
    },
    # ... more chunks
]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 4: UPDATING VECTORS
# ==============================================================================

st.header("4. Updating Vectors & In-Place Mutation", divider="rainbow")
st.markdown("""
In real-world RAG systems, document permissions, tags, or text contents change frequently. 
Vector databases handle two distinct update pathways:
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Payload-Only Updates vs. Vector Re-indexing")
    st.markdown("""
    * **Payload-Only Mutation (Cheap)**:
      When only scalar metadata changes (e.g., updating `access_level = 'public'` or adding a `view_count`), the database modifies the metadata document in-place. The high-dimensional HNSW vector index remains untouched.
    * **Coordinate Mutation (Expensive)**:
      If document text changes, the new text must be re-embedded. Updating vector coordinates requires removing the previous vector from the HNSW graph (via tombstoning) and inserting the new vector into the graph frontier.
    """)

with col2:
    st.subheader("Mutation Performance Impact")
    mut_df = pd.DataFrame([
        {"Update Type", "Engine Action", "Latency Impact", "Graph Rebuild Required"},
        {"Change Metadata Tag", "Update JSON payload", "< 2 ms (Instant)", "No"},
        {"Change Document Text", "Re-embed + Re-insert", "15 - 50 ms (Embedding + HNSW hop)", "Yes (Soft delete + insert)"},
        {"Bulk Permission Update", "Batch inverted index update", "< 10 ms", "No"}
    ])
    st.dataframe(mut_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 5: DELETING VECTORS
# ==============================================================================

st.header("5. Deleting Vectors & Tombstoning", divider="rainbow")
st.markdown("""
How vector databases delete records without degrading graph index integrity is one of the most critical 
operational concepts in systems engineering.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Challenge of Graph Deletions")
    st.markdown("""
    In an HNSW index, every node is connected to M neighbors (e.g., 16 or 32 links). 
    If a node is instantly ripped out of the graph:
    * Neighbor paths are severed, creating orphaned nodes and disconnected subgraphs.
    * Query traversals hit dead ends, causing massive recall degradation.
    * Re-wiring neighbor edges dynamically requires expensive geometric recalculations.
    """)

with col2:
    st.subheader("The Tombstone Solution")
    st.markdown("""
    * **Soft Deletion via Tombstone Bitset**:
      When `delete(id)` is called, the database marks the vector's ID in a fast memory bitset.
    * **Query-Time Masking**:
      During graph traversal, the search algorithm uses deleted nodes as intermediate transit bridges to reach other points, but filters them out of final result candidate heaps.
    * **Compaction / Vacuuming**:
      When tombstones exceed a threshold (e.g., 20% of the collection), a background process merges segments and rebuilds clean HNSW graphs.
    """)

st.divider()

# ==============================================================================
# SECTION 6: SEARCHING VECTORS
# ==============================================================================

st.header("6. Searching Vectors & k-NN Query Routing", divider="rainbow")
st.markdown("""
Querying a vector database involves routing query vectors to active memory segments and immutable disk segments, 
evaluating distance metrics, and merging candidate priority heaps.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query Pipeline Steps")
    st.markdown("""
    1. **Client Dispatches Vector**:
       The client sends the embedded query vector with query parameters: `top_k`, `score_threshold`, and optional `filters`.
    2. **Segment Fan-Out**:
       In a segmented engine (e.g., Lucene or Qdrant), the query executes concurrently across all active and sealed segments.
    3. **Local Top-K Extraction**:
       Each segment extracts its local top-K nearest neighbors using its local HNSW index.
    4. **Global Candidate Merge**:
       The query coordinator merges segment candidate heaps, discards duplicates and tombstones, and applies global ranking.
    5. **Payload Hydration**:
       Extracts document text and metadata only for the final top-K items to minimize I/O bandwidth.
    """)

with col2:
    st.subheader("Query Flow Visualization")
    search_flow_dot = """
    digraph SearchRouting {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        q [label="Query Vector (Top-K = 5)", fillcolor="#0F172A"];
        coord [label="Query Coordinator"];
        seg1 [label="Segment A (HNSW)"];
        seg2 [label="Segment B (HNSW)"];
        seg_mem [label="Mutable Memory Buffer"];
        merger [label="Global Heap Merger\\nPrune Tombstones"];
        out [label="Final Top-5 Hydrated Documents", fillcolor="#0F172A", color="#10B981"];

        q -> coord;
        coord -> seg1;
        coord -> seg2;
        coord -> seg_mem;
        seg1 -> merger;
        seg2 -> merger;
        seg_mem -> merger;
        merger -> out;
    }
    """
    st.graphviz_chart(search_flow_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: FILTERING METADATA
# ==============================================================================

st.header("7. Filtering Metadata with Boolean Query ASTs", divider="rainbow")
st.markdown("""
Vector databases allow filtering candidate vectors using rich boolean expressions over scalar fields.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Supported Filter Conditions")
    st.markdown("""
    * **Exact Match**: `department = 'Engineering'`
    * **Set Inclusion**: `tag IN ['python', 'rag', 'llm']`
    * **Numeric Range**: `price >= 10.0 AND price <= 50.0`
    * **Nested Boolean Logic**: `(category = 'Legal' OR category = 'Compliance') AND archived = False`
    * **Full-Text Keyword Sub-queries**: `title CONTAINS 'architecture'`
    """)

with col2:
    st.subheader("Filtered Search Code Example")
    st.code('''# Qdrant Boolean Filter Query
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

query_filter = Filter(
    must=[
        FieldCondition(
            key="department",
            match=MatchValue(value="Engineering")
        ),
        FieldCondition(
            key="year",
            range=Range(gte=2023)
        )
    ]
)

results = client.search(
    collection_name="enterprise_knowledge",
    query_vector=query_emb,
    query_filter=query_filter,
    limit=5
)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: MANAGING INDEXES
# ==============================================================================

st.header("8. Managing Indexes: Rebuilding, Compaction, and Segment Merging", divider="rainbow")
st.markdown("""
As documents are inserted, updated, and deleted, the vector database requires ongoing maintenance 
to prevent index fragmentation and memory bloat.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Key Maintenance Operations")
    st.markdown("""
    * **Segment Compaction (Vacuuming)**:
      Merges multiple small immutable segments into larger optimized segments, reclaiming disk space by physically stripping tombstones.
    * **Index Rebuilding (IVF Centroid Retraining)**:
      For IVF indexes, newly ingested vectors are assigned to existing cluster centroids. Over time, centroid drift degrades recall, requiring periodic index retraining.
    * **HNSW Parameter Tuning**:
      Adjusting `efSearch` at runtime to trade query latency against recall without rebuilding the index.
    * **Quantization Optimization**:
      Running scalar or product quantization on cold segments to compress RAM footprint by 75%.
    """)

with col2:
    st.subheader("Segment Merging Architecture")
    seg_dot = """
    digraph Compaction {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        subgraph cluster_uncompacted {
            label = "Fragmented Segments (With Tombstones)";
            fillcolor = "#1E293B";
            style = "rounded";
            seg_a [label="Seg 1 (100k vecs, 25k deleted)", color="#EF4444"];
            seg_b [label="Seg 2 (50k vecs, 10k deleted)", color="#EF4444"];
        }

        vacuum [label="Background Vacuum Worker\\n(Merge & Purge Tombstones)", fillcolor="#0F172A", color="#F59E0B"];
        seg_clean [label="Unified Sealed Segment\\n(115k active vecs, 0 deleted)", fillcolor="#0F172A", color="#10B981"];

        seg_a -> vacuum;
        seg_b -> vacuum;
        vacuum -> seg_clean;
    }
    """
    st.graphviz_chart(seg_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("9. Production Implementation: Enterprise Vector CRUD Pipeline with Idempotency & Batching", divider="rainbow")
st.markdown("""
Below is an enterprise CRUD manager implementing idempotent batch upserts, soft deletions with tombstones, 
payload filtering, and background compaction.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseVectorEngine:
    """
    Demonstrates production CRUD operations, soft-deletion tombstones,
    and segmented index compaction.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.tombstones: set = set()

    def batch_upsert(self, records: List[Dict[str, Any]]) -> int:
        """Idempotent batch upsert of vectors and payloads."""
        count = 0
        for rec in records:
            doc_id = rec["id"]
            vec = np.array(rec["vector"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.vectors[doc_id] = vec
            self.payloads[doc_id] = rec.get("payload", {})
            if doc_id in self.tombstones:
                self.tombstones.remove(doc_id)
            count += 1
        return count

    def update_payload(self, doc_id: str, new_payload: Dict[str, Any]) -> bool:
        """In-place update of metadata without touching vector coordinates."""
        if doc_id not in self.payloads or doc_id in self.tombstones:
            return False
        self.payloads[doc_id].update(new_payload)
        return True

    def delete(self, doc_id: str, hard_delete: bool = False) -> bool:
        """Soft-deletes via tombstone by default; hard-delete physically removes."""
        if doc_id not in self.vectors:
            return False
        if hard_delete:
            del self.vectors[doc_id]
            del self.payloads[doc_id]
            if doc_id in self.tombstones:
                self.tombstones.remove(doc_id)
        else:
            self.tombstones.add(doc_id)
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm

        candidates = []
        for doc_id, vec in self.vectors.items():
            if doc_id in self.tombstones:
                continue
            if filters:
                match = all(self.payloads[doc_id].get(k) == v for k, v in filters.items())
                if not match:
                    continue
            sim = float(np.dot(qv, vec))
            candidates.append((doc_id, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": d_id, "score": round(s, 4), "payload": self.payloads[d_id]}
            for d_id, s in candidates[:top_k]
        ]

    def compact(self) -> int:
        """Physically removes tombstoned vectors to reclaim memory."""
        count = len(self.tombstones)
        for d_id in list(self.tombstones):
            if d_id in self.vectors:
                del self.vectors[d_id]
            if d_id in self.payloads:
                del self.payloads[d_id]
        self.tombstones.clear()
        return count
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("10. Interactive Workbench: Vector CRUD and Real-Time Query Console", divider="rainbow")
st.markdown("""
Experiment with vector operations in real time. Create collections, insert or update document vectors, 
test soft deletions and compaction, and execute similarity searches with metadata filters.
""")

if "crud_sim" not in st.session_state:
    mgr = VectorOperationsManager("knowledge_base", dimension=4)
    mgr.insert("doc_1", [0.8, 0.2, 0.1, 0.0], {"title": "RAG Overview", "category": "Architecture"})
    mgr.insert("doc_2", [0.9, 0.3, 0.0, 0.1], {"title": "HNSW Graph Indexing", "category": "Algorithms"})
    mgr.insert("doc_3", [0.7, 0.1, 0.5, 0.2], {"title": "Product Quantization", "category": "Algorithms"})
    mgr.insert("doc_4", [0.4, 0.8, 0.1, 0.2], {"title": "PostgreSQL pgvector", "category": "Databases"})
    st.session_state.crud_sim = mgr

sim = st.session_state.crud_sim

tab_op1, tab_op2 = st.columns([1, 1])

with tab_op1:
    st.subheader("Manage Records (CRUD Actions)")
    
    op_type = st.selectbox("Select Action", ["Insert New Vector", "Update Existing Payload", "Delete Vector", "Compact Index"])

    if op_type == "Insert New Vector":
        new_id = st.text_input("Document ID", value="doc_5")
        new_title = st.text_input("Document Title", value="Milvus Distributed Architecture")
        new_cat = st.selectbox("Category", ["Architecture", "Algorithms", "Databases"])
        if st.button("Execute Insert"):
            sim.insert(new_id, [0.5, 0.5, 0.4, 0.1], {"title": new_title, "category": new_cat})
            st.success(f"Inserted '{new_id}' successfully!")

    elif op_type == "Update Existing Payload":
        target_id = st.selectbox("Select Target Doc ID", list(sim.vectors.keys()))
        new_title = st.text_input("Updated Title", value="Updated RAG Architecture 2026")
        if st.button("Execute Update"):
            if sim.update(target_id, payload={"title": new_title}):
                st.success(f"Updated '{target_id}' successfully!")
            else:
                st.error("Failed to update target doc.")

    elif op_type == "Delete Vector":
        del_id = st.selectbox("Select Doc ID to Delete", list(sim.vectors.keys()))
        hard_del = st.toggle("Hard Delete (Physically remove)", value=False)
        if st.button("Execute Delete"):
            if sim.delete(del_id, hard_delete=hard_del):
                st.warning(f"Deleted '{del_id}' (Hard={hard_del})!")
            else:
                st.error("Document not found.")

    elif op_type == "Compact Index":
        st.write(f"Active Tombstones in Index: **{len(sim.tombstones)}**")
        if st.button("Run Index Compactor"):
            pruned = sim.compact_index()
            st.success(f"Compaction complete! Pruned {pruned} tombstoned records.")

with tab_op2:
    st.subheader("Execute Proximity Search")
    filter_category = st.selectbox("Filter Category", ["All", "Architecture", "Algorithms", "Databases"])
    cat_arg = None if filter_category == "All" else filter_category
    top_k = st.slider("Top-K Candidates", 1, 5, 3)

    if st.button("Run Search Query"):
        q_vec = [0.85, 0.25, 0.1, 0.05]
        results = sim.search(q_vec, top_k=top_k, category_filter=cat_arg)
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("No records matched search and filter criteria.")

    st.subheader("Current Index State")
    state_df = pd.DataFrame([
        {"Metric": "Collection Name", "Value": sim.collection_name},
        {"Metric": "Total Vectors", "Value": len(sim.vectors)},
        {"Metric": "Tombstoned Deleted", "Value": len(sim.tombstones)},
        {"Metric": "Index Health", "Value": sim.index_state}
    ])
    st.dataframe(state_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: Why are deletions in graph-based vector databases (HNSW) handled via soft tombstones rather than immediate hard deletes?"):
    st.markdown("""
    **Answer**:
    HNSW graphs depend on navigable small-world pathways where nodes act as routing bridges across layers. 
    Hard-deleting a node immediately requires re-wiring neighbor connections for every linked node, which is computationally expensive 
    and risks disconnecting graph components. Instead, engines mark deleted nodes in a memory bitset (tombstone). 
    During queries, the search algorithm can traverse through tombstoned nodes to reach valid points, but omits them from final result heaps. 
    A background vacuum process rebuilds clean graph segments asynchronously.
    """)

with st.expander("Q2: What is the operational difference between inserting a vector one-by-one vs. batch upserting?"):
    st.markdown("""
    **Answer**:
    Single-vector inserts incur massive HTTP/gRPC round-trip network latency, trigger continuous lock contention on the write-ahead log (WAL), 
    and cause uncoalesced memory allocations. Batch upserting (typically 100 to 500 vectors per batch) amortizes network overhead, 
    allows SIMD vector normalization across contiguous arrays, and writes batched sequential entries to disk, resulting in a 10x to 50x ingestion speedup.
    """)

with st.expander("Q3: When updating an existing document's metadata tags, should you re-embed and re-index the vector?"):
    st.markdown("""
    **Answer**:
    No. In modern vector databases (Qdrant, Pinecone, Weaviate), scalar metadata is decoupled from the vector index. 
    Modifying metadata tags requires only an in-place update to the document payload store or scalar inverted index. 
    Re-embedding the text wastes LLM API costs, and re-inserting the vector triggers unnecessary HNSW graph modifications.
    """)

with st.expander("Q4: How does a vector database ensure data durability across server crashes?"):
    st.markdown("""
    **Answer**:
    Vector databases use a **Write-Ahead Log (WAL)**. Before any upsert or deletion is applied to active in-memory segments, 
    the transaction is written sequentially to disk in the WAL. If the database crashes or suffers power failure, 
    it replays the WAL on reboot to reconstruct in-memory buffers and rebuild any incomplete segment indices.
    """)

with st.expander("Q5: What is the purpose of background index compaction and segment merging?"):
    st.markdown("""
    **Answer**:
    Continuous ingestion creates numerous small immutable segment files and accumulates tombstones from deleted records. 
    Searching across dozens of uncompacted segments degrades query latency and wastes RAM. 
    Background compaction merges small segments into unified, dense segments and physically purges tombstones, restoring optimal search speeds.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Forgetting Background Index Compaction and Leaking Memory**  
    Continuously deleting documents without triggering compaction leaves thousands of dead vectors in RAM. 
    Tombstoned vectors continue occupying memory and slowing down graph traversals until a vacuum cycle is executed.
    """)
    st.error("""
    **Trap 2: Non-Idempotent Ingestion Pipelines Causing Duplicate Records**  
    Using random UUIDs for chunks instead of deterministic hash IDs (e.g., `hash(doc_url + chunk_index)`) 
    causes duplicate records when ingestion pipelines retry failed jobs, bloating the database and corrupting top-K results.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Re-Embedding Unchanged Text on Metadata Updates**  
    Calling embedding APIs when only scalar metadata (e.g., `access_tier` or `department`) changed wastes thousands of dollars 
    in model API fees. Always use payload-only update APIs.
    """)
    st.warning("""
    **Trap 4: Submitting Oversized Ingestion Batches Leading to Timeout Failures**  
    Sending batches larger than 1,000 vectors at 1536-dim (each payload > 6 MB) can exceed HTTP body size limits 
    or trigger gateway timeouts. Optimal batch sizes range between 100 and 500 records.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Tombstone Soft Delete")
        st.caption("Why is it used in HNSW?")
        st.markdown("**Back**: Avoids breaking graph connectivity dynamically. Marked in bitset and traversed during search, but filtered from results until compaction.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Idempotent Upsert ID")
        st.caption("How to prevent duplicates?")
        st.markdown("**Back**: Use deterministic IDs derived from `hash(doc_id + chunk_index)`. Upserting with the same ID overwrites cleanly.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: WAL in Vector Databases")
        st.caption("What role does it play?")
        st.markdown("**Back**: Write-Ahead Log guarantees durability. Replayed on crash recovery to reconstruct in-memory segments.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Segment Compaction")
        st.caption("What does it accomplish?")
        st.markdown("**Back**: Merges fragmented segment files, purges tombstones, and rebuilds unified graph indices for optimal search latency.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Database Operations:")

c1 = st.checkbox("I understand the collection creation lifecycle and schema configuration parameters.", value=True)
c2 = st.checkbox("I know how to design idempotent batch upsert pipelines using deterministic chunk hashing.", value=True)
c3 = st.checkbox("I understand why HNSW graphs use tombstones for deletions rather than immediate hard deletes.", value=True)
c4 = st.checkbox("I know when to perform payload-only updates to avoid expensive re-embedding API costs.", value=True)
c5 = st.checkbox("I understand the role of the Write-Ahead Log (WAL) and segment compaction in vector stores.", value=True)
c6 = st.checkbox("I can implement an enterprise vector CRUD pipeline with background compaction in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Database Operations and understand how to maintain high-performance, durable RAG data pipelines.")
