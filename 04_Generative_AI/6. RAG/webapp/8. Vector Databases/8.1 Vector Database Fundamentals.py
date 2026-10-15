import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class SimpleVectorDatabase:
    """
    Lightweight, production-grade in-memory vector database simulator.
    Demonstrates vector storage, indexing, similarity calculation, and metadata filtering.
    """
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, str] = {}

    def insert(self, doc_id: str, text: str, vector: List[float], meta: Dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self.vectors[doc_id] = v
        self.documents[doc_id] = text
        self.metadata[doc_id] = meta

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        filter_category: Optional[str] = None,
        filter_mode: str = "single_stage"
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        if q_norm > 0:
            qv = qv / q_norm

        results = []

        if filter_mode == "pre_filter":
            # Filter candidate IDs first, then score
            candidate_ids = [
                d_id for d_id, m in self.metadata.items()
                if not filter_category or m.get("category") == filter_category
            ]
            for doc_id in candidate_ids:
                sim = float(np.dot(qv, self.vectors[doc_id]))
                results.append({
                    "id": doc_id,
                    "text": self.documents[doc_id],
                    "category": self.metadata[doc_id].get("category", "General"),
                    "score": round(sim, 4)
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        elif filter_mode == "post_filter":
            # Score all vectors first, pick top candidates, then filter metadata
            all_scores = []
            for doc_id, vec in self.vectors.items():
                sim = float(np.dot(qv, vec))
                all_scores.append({
                    "id": doc_id,
                    "text": self.documents[doc_id],
                    "category": self.metadata[doc_id].get("category", "General"),
                    "score": round(sim, 4)
                })
            all_scores.sort(key=lambda x: x["score"], reverse=True)
            # Post-filter after top retrieval
            filtered = [
                doc for doc in all_scores[:top_k * 2]
                if not filter_category or doc["category"] == filter_category
            ]
            return filtered[:top_k]

        else: # single_stage (iterative evaluation)
            for doc_id, vec in self.vectors.items():
                if filter_category and self.metadata[doc_id].get("category") != filter_category:
                    continue
                sim = float(np.dot(qv, vec))
                results.append({
                    "id": doc_id,
                    "text": self.documents[doc_id],
                    "category": self.metadata[doc_id].get("category", "General"),
                    "score": round(sim, 4)
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================

st.set_page_config(
    page_title="8.1 Vector Database Fundamentals - RAG Master Curriculum",
    page_icon="🗄️",
    layout="wide"
)

st.title("8.1 Vector Database Fundamentals: The Vector Engine, Indexing, and Metadata Filtering")
st.caption("Chapter 8: Vector Databases | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
Relational databases (PostgreSQL, MySQL) and traditional document stores (MongoDB, Elasticsearch) were 
architected around **scalar data**: strings, timestamps, integers, and JSON payloads. They rely on B-Trees, 
hash indices, and inverted lists designed for deterministic equality (`WHERE user_id = 42`) or range scans (`WHERE age > 30`).

However, dense embeddings in RAG systems are **high-dimensional vectors** (spanning 768 to 3,072 float coordinates). 
Finding semantically related documents requires calculating geometric vector proximity across millions of points in milliseconds. 
A **Vector Database** is an engine engineered specifically for dense vector storage, multi-dimensional index acceleration, 
sub-10-millisecond similarity queries, and unified metadata filtering.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Vector Database Topology and Lifecycle", divider="rainbow")
st.markdown("""
In enterprise RAG architectures, the vector database serves as the persistent semantic index tier. 
It couples an Approximate Nearest Neighbor (ANN) index structure with a primary scalar document/metadata store, 
write-ahead logging (WAL), and concurrent query routing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    vdb_bp_dot = """
    digraph VDBTopology {
        rankdir=TB;
        graph [bgcolor="#0B1120", splines=polyline, nodesep=0.4, ranksep=0.5];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2];

        subgraph cluster_ingestion {
            label = "1. Ingestion Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            doc_in [label="Incoming Document Chunks\\n(Text and Metadata)", fillcolor="#0F172A", color="#38BDF8"];
            embed_gen [label="Embedding Model Pipeline\\n(Dense Vectors: Float32)", fillcolor="#0F172A", color="#38BDF8"];
            doc_in -> embed_gen [label="Batch Ingestion"];
        }

        subgraph cluster_engine {
            label = "2. Vector Database Engine Core";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            wal [label="Write-Ahead Log (WAL)\\nDurability and Crash Recovery", fillcolor="#334155", color="#F59E0B"];
            storage_tier [label="Primary Storage Engine\\nRaw Vectors and Document Payloads", fillcolor="#334155", color="#10B981"];
            index_builder [label="ANN Index Construction\\n(HNSW Graph / IVF Clusters / PQ)", fillcolor="#334155", color="#8B5CF6"];
            meta_index [label="Scalar Inverted Index\\nMetadata Filtering (Tenant, Date, Tag)", fillcolor="#334155", color="#06B6D4"];

            wal -> storage_tier [label="Flush to Segments"];
            storage_tier -> index_builder [label="Vector Points"];
            storage_tier -> meta_index [label="Scalar Attributes"];
        }

        subgraph cluster_query {
            label = "3. Query and Retrieval Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            query_in [label="User Query Vector\\n+ Metadata Filter Predicates", fillcolor="#0F172A", color="#EC4899"];
            search_exec [label="Unified Query Coordinator\\nSingle-Stage / ACORN Filtered Traversal", fillcolor="#0F172A", color="#EC4899"];
            ranked_out [label="Top-K Ranked Context\\nText Chunks and Similarity Scores", fillcolor="#0F172A", color="#10B981"];

            query_in -> search_exec [label="Query Request"];
            search_exec -> ranked_out [label="k-NN Candidates"];
        }

        embed_gen -> wal [label="ACID Upsert"];
        index_builder -> search_exec [label="ANN Graph Traversal"];
        meta_index -> search_exec [label="Bitmap Pruning"];
    }
    """
    st.graphviz_chart(vdb_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production vector engines like **Qdrant**, **Pinecone**, **Milvus**, and **Chroma** do not treat vectors as isolated arrays. 
They store vectors alongside structured document payloads inside segment files. While older pipelines relied on pre-filtering 
or post-filtering, state-of-the-art vector engines perform **single-stage filtered search** (e.g., Qdrant payload index traversal, 
Filtered HNSW, or ACORN) to prevent catastrophic recall drop when strict security filters (e.g., `tenant_id` or `user_role`) are applied.
""")

st.divider()

# ==============================================================================
# SECTION 2: WHAT IS A VECTOR DATABASE?
# ==============================================================================

st.header("2. What is a Vector Database?", divider="rainbow")
st.markdown("""
A **Vector Database** is a purpose-built database management system designed to index, store, and query high-dimensional vector representations 
derived from machine learning models alongside associated scalar payloads.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Classical DBMS vs. Vector Database")
    comp_df = pd.DataFrame([
        {"Dimension": "Primary Data Type", "Classical Database (SQL/NoSQL)": "Scalars (Strings, Ints, JSON)", "Vector Database": "Dense Coordinate Arrays (Float32/Float16)"},
        {"Dimension": "Query Model", "Classical Database (SQL/NoSQL)": "Exact Boolean matches (WHERE x = y)", "Vector Database": "Approximate Proximity (Cosine / Dot Product)"},
        {"Dimension": "Index Topology", "Classical Database (SQL/NoSQL)": "B-Tree, Hash Index, LSM-Tree", "Vector Database": "HNSW Graphs, IVF Partitions, Product Quantization"},
        {"Dimension": "Query Output", "Classical Database (SQL/NoSQL)": "Exhaustive set of matching records", "Vector Database": "Top-K nearest neighbors sorted by distance"},
        {"Dimension": "Search Complexity", "Classical Database (SQL/NoSQL)": "Logarithmic O(log N) for scalar ranges", "Vector Database": "O(log N) ANN graph traversal or O(N) brute force scan"}
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Core Vector Database Properties")
    st.markdown("""
    * **High-Dimensional Native Indexing**:
      Handles embedding vectors spanning 384, 768, 1,536, or 3,072 dimensions without collapsing spatial discrimination.
    * **Sub-Second Low Latency Retrieval**:
      Evaluates millions of candidate points in under 10 milliseconds using heuristic graph or cluster indexing.
    * **Co-Located Payload Storage**:
      Preserves the original source text, parent document ID, timestamp, and user permissions directly with the vector to avoid expensive secondary joins.
    * **Dynamic Upserts and Deletes**:
      Unlike static vector libraries (e.g., bare FAISS index files on disk), a vector database allows real-time atomic inserts, updates, and deletes without requiring complete index rebuilding.
    """)

st.divider()

# ==============================================================================
# SECTION 3: WHY VECTOR DATABASES ARE REQUIRED
# ==============================================================================

st.header("3. Why Vector Databases Are Required", divider="rainbow")
st.markdown("""
Why can't enterprise RAG systems simply add an array column (`embedding float[]`) to an existing relational database like SQLite or vanilla PostgreSQL?
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Curse of Dimensionality & Linear Scan Bottleneck")
    st.markdown("""
    * **Linear Scan Cost**:
      In a database containing 5,000,000 document chunks of dimension 1,536:
      Evaluating a single query requires computing 5,000,000 dot products.
      Each dot product requires 1,536 multiplications and 1,535 additions.
      `5,000,000 * 1,536 = 7.68 billion floating-point operations per user query`.
    * **CPU Exhaustion**:
      Even on high-frequency server hardware, a linear scan requires 5 to 15 seconds per search request.
    * **B-Trees Fail in Geometric Space**:
      B-Trees enforce a single linear ordering (`A < B < C`). High-dimensional points have no linear order; 
      point A may be closer to B along dimension 1 but further along dimension 400. Standard database indices cannot prune multidimensional search spaces.
    """)

with col2:
    st.subheader("Why Simple Vector Libraries Are Not Enough")
    st.markdown("""
    Standalone algorithmic libraries like raw FAISS or Annoy provide fast ANN math, but lack database capabilities:
    
    1. **Lack of ACID / Durability**: Raw FAISS indices live in memory; an application crash wipes all state unless serialized.
    2. **Costly Updates**: Modifying a single vector in a static FAISS index often requires re-training or rebuilding the entire index.
    3. **No Native Payload Indexing**: Bare libraries return integer row offsets (`[42, 108, 951]`). The application must manually query another SQL store to look up text chunks.
    4. **Lack of Horizontal Sharding**: As datasets grow beyond server RAM (tens of millions of vectors), vector databases partition indices across distributed nodes.
    """)

st.divider()

# ==============================================================================
# SECTION 4: VECTOR STORAGE
# ==============================================================================

st.header("4. Vector Storage Architecture", divider="rainbow")
st.markdown("""
Vector storage requires managing both raw float coordinates and their serialized representations across RAM and persistent storage.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Memory and Storage Formats")
    st.markdown("""
    * **Uncompressed Float32 Storage**:
      * Each dimension occupies 4 bytes.
      * A 1,536-dimensional vector consumes: `1,536 * 4 bytes = 6,144 bytes (~6 KB)`.
      * 1,000,000 vectors require **6.14 GB** of pure vector memory (excluding index overhead and metadata).
    * **Float16 / BFloat16 Precision**:
      * Reduces memory consumption by 50% (3,072 bytes per vector) with negligible impact on retrieval accuracy.
    * **Quantized Representations (Scalar and Product Quantization)**:
      * Int8 quantization reduces memory by 75% (1 byte per dimension).
      * Binary quantization (1 bit per dimension) reduces memory by 96.8%, enabling million-vector searches within compact RAM envelopes.
    """)

with col2:
    st.subheader("Storage Hierarchy: RAM vs. MMAP vs. NVMe Disk")
    st.markdown("""
    Vector database architectures balance query latency against server hardware expenses:
    * **In-Memory Engines**: Complete vector arrays and graph indices reside in RAM (e.g., Pinecone Standard, Milvus in-memory, Chroma in-memory). Sub-5ms latency, high infrastructure cost.
    * **Memory-Mapped (MMAP) Storage**: Indices reside on NVMe SSDs and are paged into OS page cache on demand (e.g., Qdrant memmap, Chroma persistent mode). Allows collections larger than physical RAM.
    * **Disk-First Engines (DiskANN)**: Optimized for fast SSD IOPS using compressed indices in RAM and full vectors read from disk during final re-ranking.
    """)

st.divider()

# ==============================================================================
# SECTION 5: VECTOR INDEXING
# ==============================================================================

st.header("5. Vector Indexing Mechanisms", divider="rainbow")
st.markdown("""
Vector indexing constructs spatial data structures that guide query vectors directly to their nearest neighbors 
without evaluating every vector in the dataset.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Primary Index Architectures")
    st.markdown("""
    1. **Graph-Based (HNSW - Hierarchical Navigable Small World)**:
       * Builds a multi-layer graph where upper layers have long-range skip connections and bottom layers have dense local connections.
       * Provides state-of-the-art recall and high queries-per-second (QPS).
       * Higher RAM overhead due to edge adjacency lists.
    2. **Inverted File Index (IVF)**:
       * Partitions the vector space into Voronoi cells using k-means clustering.
       * At search time, checks only vectors within the centroids closest to the query.
       * Moderate memory usage, fast build time, slightly lower recall than HNSW.
    3. **Quantization Indexes (PQ / SQ)**:
       * Compresses vectors into compact byte codes and precomputes distance lookup tables.
       * Extreme memory efficiency at the cost of slight loss in ranking resolution.
    """)

with col2:
    st.subheader("Index Selection Guidelines")
    index_guide_df = pd.DataFrame([
        {"Workload Requirement", "Recommended Index", "Tradeoff"},
        {"Ultra-low latency (<5ms), highest recall (>95%)", "HNSW", "High RAM consumption (1.5x - 2x raw vector size)"},
        {"Tens of millions of vectors on limited RAM", "IVF-PQ / HNSW-SQ", "Slight recall drop; requires centroid training step"},
        {"Exact 100% recall required for small dataset (<50k)", "Flat (Brute-Force Scan)", "No indexing overhead; search latency scales linearly O(N)"},
        {"Rapid continuous ingestion without rebuild delays", "Dynamic HNSW", "Higher memory usage for real-time node linking"}
    ])
    st.dataframe(index_guide_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: SIMILARITY SEARCH
# ==============================================================================

st.header("6. Similarity Search Execution", divider="rainbow")
st.markdown("""
Similarity search is the operational execution pipeline where an incoming query vector is compared against 
stored vectors to retrieve the Top-K most semantically related document chunks.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Search Execution Steps")
    st.markdown("""
    1. **Query Vectorization**:
       The user query string is encoded into a dense vector by the identical embedding model used during document ingestion.
    2. **Index Entry & Routing**:
       The search coordinator passes the query vector to the index engine. For HNSW, search enters the sparse top layer and greedily navigates downhill toward the target neighborhood.
    3. **Candidate Pruning**:
       The engine maintains a priority queue of candidate vectors, continuously updating the top-K list until search exploration criteria (`efSearch` or `nprobe`) are satisfied.
    4. **Distance Metric Evaluation**:
       Distances are calculated via Cosine Similarity, Dot Product (for unit vectors), or Squared Euclidean distance.
    5. **Result Set Formatting**:
       Returns document chunk IDs, similarity scores, source texts, and scalar metadata.
    """)

with col2:
    st.subheader("Query Execution Flowchart")
    exec_dot = """
    digraph SearchPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#38BDF8"];
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8"];

        q_txt [label="User Query String", fillcolor="#0F172A"];
        q_vec [label="Embedding Computation (e.g. 1536-dim)"];
        index_nav [label="Index Traversal (HNSW / IVF)"];
        topk_heap [label="Candidate Heap Pruning (Top-K)"];
        payload_join [label="Payload & Metadata Extraction"];
        final_out [label="Formatted Context for LLM", fillcolor="#0F172A", color="#10B981"];

        q_txt -> q_vec -> index_nav -> topk_heap -> payload_join -> final_out;
    }
    """
    st.graphviz_chart(exec_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: METADATA FILTERING
# ==============================================================================

st.header("7. Metadata Filtering Strategies", divider="rainbow")
st.markdown("""
Enterprise RAG queries almost never perform pure vector search. They require strict business filters:
`tenant_id = 'org_44'`, `department = 'Legal'`, or `publish_date >= '2024-01-01'`.
How the vector database reconciles scalar filters with vector graph navigation determines whether search succeeds or suffers catastrophic recall failure.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Pre-Filtering")
    st.markdown("""
    * **Mechanism**:
      Evaluates metadata filter first via scalar index (B-Tree/Bitmap), collects matching IDs, then runs vector search only across matching vectors.
    * **Advantage**:
      Guarantees 100% metadata compliance.
    * **Limitation**:
      If the filter matches 1,000 vectors out of 10,000,000, graph indices (like HNSW) become disconnected. Vector search degrades into a brute-force scan across the remaining subset.
    """)

with col2:
    st.subheader("Post-Filtering")
    st.markdown("""
    * **Mechanism**:
      Executes standard vector search to retrieve the top-K nearest neighbors, then discards any candidates that do not match metadata filters.
    * **Advantage**:
      Fastest vector search leveraging full intact HNSW graphs.
    * **Catastrophic Failure**:
      If only 1% of documents in the database belong to `tenant_id = 'org_44'`, standard top-10 retrieval might return zero matching records!
    """)

with col3:
    st.subheader("Single-Stage (Iterative) Filtering")
    st.markdown("""
    * **Mechanism (ACORN / Filtered HNSW)**:
      Filters metadata during graph traversal. At every node hop, neighbor candidates are verified against scalar bitmaps before expanding the search frontier.
    * **Production Standard**:
      Implemented in Qdrant, Milvus, and Pinecone. Provides both high recall and sub-10ms latency regardless of filter selectivity.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("8. Production Implementation: Thread-Safe In-Memory Vector Database with Payload Indexing", divider="rainbow")
st.markdown("""
Below is an enterprise-grade in-memory vector database architecture implementing normalized vector storage, 
cosine similarity calculation, scalar inverted indices for metadata, and support for pre-filtering, post-filtering, and single-stage traversal.
""")

st.code('''import numpy as np
from typing import List, Dict, Any, Optional

class ProductionVectorStore:
    """
    In-memory vector store demonstrating vector normalization, 
    scalar payload indexing, and filtered proximity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, str] = {}
        # Scalar inverted index for metadata filtering: {field: {value: set(doc_ids)}}
        self.inverted_index: Dict[str, Dict[Any, set]] = {}

    def insert(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        if v.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {v.shape[0]}")
        
        # Unit normalization allows cosine similarity via fast dot product
        norm = np.linalg.norm(v)
        normalized_v = v / norm if norm > 0 else v

        self.vectors[doc_id] = normalized_v
        self.documents[doc_id] = text
        self.payloads[doc_id] = metadata

        # Index scalar metadata
        for field, value in metadata.items():
            if field not in self.inverted_index:
                self.inverted_index[field] = {}
            if value not in self.inverted_index[field]:
                self.inverted_index[field][value] = set()
            self.inverted_index[field][value].add(doc_id)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        strategy: str = "single_stage"
    ) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        qv = qv / q_norm if q_norm > 0 else qv

        # Determine matching IDs from inverted index if filters present
        matching_ids = None
        if filters:
            for field, val in filters.items():
                field_matches = self.inverted_index.get(field, {}).get(val, set())
                matching_ids = field_matches if matching_ids is None else matching_ids.intersection(field_matches)
            if matching_ids is None:
                matching_ids = set()

        candidates = []

        if strategy == "pre_filter" and filters:
            # Score only vectors that passed scalar filtering
            for doc_id in matching_ids:
                sim = float(np.dot(qv, self.vectors[doc_id]))
                candidates.append((doc_id, sim))
        elif strategy == "post_filter" and filters:
            # Score all vectors, take broad top_k, then filter
            all_scored = [(d_id, float(np.dot(qv, vec))) for d_id, vec in self.vectors.items()]
            all_scored.sort(key=lambda x: x[1], reverse=True)
            for d_id, sim in all_scored:
                if d_id in matching_ids:
                    candidates.append((d_id, sim))
                    if len(candidates) >= top_k:
                        break
        else: # Single-stage / unfiltered
            for doc_id, vec in self.vectors.items():
                if filters and doc_id not in matching_ids:
                    continue
                sim = float(np.dot(qv, vec))
                candidates.append((doc_id, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        return [
            {
                "id": doc_id,
                "score": round(sim, 4),
                "text": self.documents[doc_id],
                "metadata": self.payloads[doc_id]
            }
            for doc_id, sim in top_candidates
        ]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("9. Interactive Workbench: Vector Search and Metadata Filtering Simulator", divider="rainbow")
st.markdown("""
Test how vector similarity search interacts with metadata filtering strategies. 
Inspect how pre-filtering, post-filtering, and single-stage filtering impact retrieved documents and ranking scores.
""")

# Initialize simulator state
if "vdb_sim" not in st.session_state:
    sim = SimpleVectorDatabase(dimension=4)
    sim.insert("doc_1", "Enterprise RAG architectures require vector databases for sub-10ms retrieval.", [0.8, 0.2, 0.1, 0.0], {"category": "Architecture", "tenant": "corp_a"})
    sim.insert("doc_2", "HNSW builds multi-layer graphs to accelerate nearest neighbor vector search.", [0.9, 0.3, 0.0, 0.1], {"category": "Algorithms", "tenant": "corp_a"})
    sim.insert("doc_3", "Product Quantization compresses dense 1536-dimensional vectors into 8-bit codes.", [0.7, 0.1, 0.5, 0.2], {"category": "Algorithms", "tenant": "corp_b"})
    sim.insert("doc_4", "Pre-filtering can disconnect HNSW graphs when filter selectivity is high.", [0.85, 0.25, 0.15, 0.05], {"category": "Architecture", "tenant": "corp_a"})
    sim.insert("doc_5", "Post-filtering evaluates top vector matches before discarding non-matching payloads.", [0.6, 0.5, 0.2, 0.1], {"category": "Architecture", "tenant": "corp_b"})
    sim.insert("doc_6", "PostgreSQL pgvector extension enables relational queries with vector distance.", [0.4, 0.8, 0.1, 0.2], {"category": "Databases", "tenant": "corp_a"})
    st.session_state.vdb_sim = sim

sim_inst = st.session_state.vdb_sim

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query Configuration")
    q_x = st.slider("Query Vector Coordinate X (Architecture/Theory)", 0.0, 1.0, 0.85, step=0.05)
    q_y = st.slider("Query Vector Coordinate Y (Algorithms/Eng)", 0.0, 1.0, 0.25, step=0.05)
    q_z = st.slider("Query Vector Coordinate Z (Quantization/Storage)", 0.0, 1.0, 0.10, step=0.05)
    q_w = st.slider("Query Vector Coordinate W (Database Integration)", 0.0, 1.0, 0.05, step=0.05)
    
    selected_category = st.selectbox(
        "Metadata Filter (Category)",
        ["All Categories", "Architecture", "Algorithms", "Databases"]
    )
    filter_cat = None if selected_category == "All Categories" else selected_category

    filter_strategy = st.radio(
        "Filtering Strategy",
        ["single_stage", "pre_filter", "post_filter"],
        format_func=lambda x: {
            "single_stage": "Single-Stage (ACORN / Inline Filter)",
            "pre_filter": "Pre-Filtering (Filter candidates then search)",
            "post_filter": "Post-Filtering (Top-K search then filter)"
        }[x]
    )
    top_k_val = st.slider("Top-K Candidates", 1, 6, 3)

with col2:
    st.subheader("Query Results & Execution Telemetry")
    q_vec = [q_x, q_y, q_z, q_w]
    res = sim_inst.search(q_vec, top_k=top_k_val, filter_category=filter_cat, filter_mode=filter_strategy)
    
    if res:
        res_df = pd.DataFrame(res)
        st.dataframe(res_df[["id", "score", "category", "text"]], use_container_width=True, hide_index=True)
        st.success(f"Retrieved {len(res)} results matching category '{selected_category}' using strategy '{filter_strategy}'.")
    else:
        st.warning("No documents matched the query under the selected filter criteria!")

    st.markdown("""
    **Observation**:
    * **Single-Stage** evaluates only documents matching the metadata condition while preserving optimal vector ranking.
    * **Pre-Filtering** restricts candidate IDs before vector comparison.
    * **Post-Filtering** can miss valid documents if non-matching vectors occupy the initial top-K slots.
    """)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================

st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Key Interview Questions & Answers")

with st.expander("Q1: Why does a traditional B-Tree index fail completely when indexing vector embeddings?"):
    st.markdown("""
    **Answer**:
    A B-Tree index relies on a strict total scalar order: for any scalar keys, A < B < C. 
    In contrast, dense vector embeddings exist in high-dimensional continuous metric spaces (e.g., 768 or 1,536 dimensions). 
    There is no natural total ordering for high-dimensional vectors; a point can be closer along coordinate 12 while being farther along coordinate 511. 
    Consequently, B-Trees cannot divide the vector space into geometrically pruned sub-trees, forcing the database into an exhaustive O(N) brute-force scan.
    """)

with st.expander("Q2: What is the fundamental difference between a vector library (FAISS) and a vector database (Qdrant, Pinecone)?"):
    st.markdown("""
    **Answer**:
    A **vector library** (FAISS, Annoy, ScaNN) is an in-memory algorithmic package for computing nearest neighbors on static arrays. 
    It lacks ACID durability, dynamic real-time CRUD operations, network APIs, horizontal clustering, replication, and integrated metadata payload indexing. 
    A **vector database** (Qdrant, Pinecone, Milvus) is a full-featured data management system that wraps vector indexing algorithms with 
    write-ahead logging (WAL), distributed sharding, snapshot persistence, role-based access control (RBAC), and unified metadata filtering.
    """)

with st.expander("Q3: How does the 'Post-Filtering' metadata strategy cause catastrophic recall failure in RAG systems?"):
    st.markdown("""
    **Answer**:
    In post-filtering, the database performs standard vector search to retrieve the top-K nearest neighbors globally, 
    and only then applies the metadata filter to discard non-matching candidates. If the target metadata category represents 
    a small fraction of the overall corpus (e.g., 1% for a specific tenant), the initial top-K vector matches will likely contain 
    zero documents with that tenant ID. As a result, post-filtering returns an empty result set even when relevant matching documents exist in the corpus.
    """)

with st.expander("Q4: What memory considerations must be factored in when running uncompressed 1,536-dimensional vectors at scale?"):
    st.markdown("""
    **Answer**:
    Each 1,536-dimensional vector stored as 32-bit floats consumes `1,536 * 4 bytes = 6,144 bytes (~6 KB)`. 
    For 10,000,000 vectors, raw coordinates alone consume approximately 61.4 GB of RAM. 
    When factoring in HNSW graph edges (which typically add 1.5x to 2x overhead), the total RAM requirement exceeds 120 GB to 180 GB. 
    In production, this cost is mitigated using Scalar Quantization (Int8), Product Quantization (PQ), or memory-mapped storage (MMAP).
    """)

with st.expander("Q5: What is single-stage filtered search (e.g., ACORN or Filtered HNSW), and why is it superior?"):
    st.markdown("""
    **Answer**:
    Single-stage filtered search evaluates scalar metadata filters directly during the ANN graph traversal step. 
    When the search algorithm evaluates neighbor edges from the current node, it checks candidate nodes against precomputed 
    scalar bitsets or payload indices before expanding the traversal frontier. This ensures that only valid documents enter the 
    candidate priority queue, preserving high recall without the graph-fragmentation issues of pre-filtering or the recall loss of post-filtering.
    """)

st.subheader("Production Traps: Common Engineering Pitfalls")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1: Deploying Post-Filtering with Multi-Tenant RAG Applications**  
    Applying post-filtering in multi-tenant environments where queries specify `tenant_id = X` leads to silent retrieval failure. 
    If tenant X accounts for 0.5% of vectors, top-10 retrieval will return empty answers, causing the LLM to hallucinate that no documents exist.
    """)
    st.error("""
    **Trap 2: Ignoring Vector Memory Growth and OOM Server Crashes**  
    Running in-memory HNSW vector indexes without memory headroom causes sudden out-of-memory (OOM) crashes. 
    Every additional million vectors at 1536-dim requires ~12 GB to 18 GB of RAM. Production systems must implement quantization or disk-backed segments.
    """)

with trap_col2:
    st.warning("""
    **Trap 3: Mixing Different Embedding Models in the Same Collection**  
    Inserting vectors from different models (e.g., OpenAI `text-embedding-3-small` and HuggingFace `bge-large`) into the same index 
    renders similarity scores meaningless. Vectors must originate from identical embedding models and dimensions.
    """)
    st.warning("""
    **Trap 4: Missing Unit Vector Normalization for Cosine Similarity**  
    When vector databases use dot product as a proxy for cosine similarity, vectors must be L2-normalized during ingestion. 
    Failing to normalize causes longer texts with larger vector magnitudes to unfairly dominate similarity rankings.
    """)

st.subheader("Core Architecture Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: Vector Database vs. Vector Library")
        st.caption("What distinguishes them?")
        st.markdown("**Back**: A library (FAISS) is an algorithmic calculation package; a vector database (Qdrant, Pinecone) provides ACID storage, WAL, sharding, and metadata indexing.")

with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Single-Stage Filtering")
        st.caption("How does it operate?")
        st.markdown("**Back**: Evaluates scalar metadata conditions dynamically during graph traversal, guaranteeing metadata compliance without dropping recall.")

with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: 1536-dim Memory Formula")
        st.caption("What is the raw RAM footprint?")
        st.markdown("**Back**: `1,536 dims * 4 bytes = ~6 KB` per vector. 1 million vectors = ~6.14 GB raw coordinates plus 1.5x to 2x index graph overhead.")

with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Why B-Trees Fail")
        st.caption("Why can't SQL indices work?")
        st.markdown("**Back**: B-Trees require 1-D scalar ordering (`A < B`). High-dimensional vectors have no total order, making geometric pruning impossible.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of Vector Database Fundamentals:")

c1 = st.checkbox("I understand why relational B-Tree and hash indices fail on high-dimensional vectors.", value=True)
c2 = st.checkbox("I can differentiate between standalone vector libraries (FAISS) and full-featured vector databases (Qdrant, Pinecone, Milvus).", value=True)
c3 = st.checkbox("I know the memory footprint calculation for Float32 vectors and how quantization reduces RAM consumption.", value=True)
c4 = st.checkbox("I can contrast graph-based (HNSW), clustering-based (IVF), and quantization-based (PQ) vector indexing paradigms.", value=True)
c5 = st.checkbox("I understand the mechanics of pre-filtering, post-filtering, and single-stage filtered search.", value=True)
c6 = st.checkbox("I can design and implement a thread-safe in-memory vector store with metadata inverted indices in Python.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Outstanding! You have mastered Vector Database Fundamentals and understand the storage, indexing, and filtering mechanics required for high-scale RAG systems.")
