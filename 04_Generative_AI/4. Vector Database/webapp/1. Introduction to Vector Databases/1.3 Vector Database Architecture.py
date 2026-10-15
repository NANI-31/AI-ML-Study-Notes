import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.3 Vector Database Architecture - Vector DB Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulation
class ArchitectureTracer:
    """Simulates the 9 internal subsystems during vector search execution."""
    def __init__(self):
        self.subsystems = [
            {"step": 1, "subsystem": "Documents", "action": "Raw unstructured text payload loaded into memory buffer."},
            {"step": 2, "subsystem": "Chunks", "action": "Text split into 512-token chunk with 10% overlap boundaries."},
            {"step": 3, "subsystem": "Embeddings", "action": "Dense embedding model generates 1536-dimensional float32 vector."},
            {"step": 4, "subsystem": "Metadata", "action": "JSON payload attached with department, author, and timestamp attributes."},
            {"step": 5, "subsystem": "Vector Storage", "action": "Vector coordinates persisted to immutable segment file on SSD via WAL."},
            {"step": 6, "subsystem": "Index", "action": "Vector inserted into HNSW proximity graph across multi-layer skip list."},
            {"step": 7, "subsystem": "Query Engine", "action": "Incoming user query vector parsed and validated against schema constraints."},
            {"step": 8, "subsystem": "Similarity Calculation", "action": "Hardware AVX-512 SIMD instructions execute normalized dot product."},
            {"step": 9, "subsystem": "Retrieval Layer", "action": "Top-K nearest neighbor heap extracts top candidates with source citations."}
        ]

    def trace(self) -> List[Dict[str, Any]]:
        return self.subsystems

# Title & Overview
st.title("1.3 Vector Database Architecture: The 9 Core Subsystems")
st.caption("Chapter 1: Introduction to Vector Databases | Domain: 4. Vector Database")
st.markdown("""
A production Vector Database is not merely an algorithm script like FAISS or hnswlib—it is a **complete database management system** 
engineered for distributed persistence, fault tolerance, transaction isolation, metadata filtering, and high-throughput concurrent search.

To build, optimize, and debug enterprise vector infrastructure, engineers must understand how data traverses the 
**nine fundamental architectural subsystems**: from raw document ingestion and semantic chunking to dense embedding generation, 
immutable segment storage, graph indexing, SIMD distance evaluation, and final candidate retrieval.

In this lesson, you will master:
1. **The 9 Core Architectural Subsystems**: Dissecting the end-to-end lifecycle of vector database ingestion and query execution.
2. **Storage and Indexing Separation**: How modern vector databases isolate memory-mapped graph indices from raw payload stores.
3. **Hardware Acceleration**: The role of SIMD (AVX-512, NEON) and GPU acceleration in scaling distance calculations.
4. **Single-Stage Filter Integration**: How the retrieval layer coordinates with metadata indices during graph walks.
5. **Production Failure Modes**: Preventing memory fragmentation, segment compaction stalls, and index corruption.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: The 9 Core Subsystems of a Production Vector Database", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph NineSubsystemsBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingest {
            label="Phase 1: Ingestion and Representation Subsystems";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            sub_doc [label="Subsystem 1: Documents\\nRaw Unstructured Corpus Ingestion", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sub_chunk [label="Subsystem 2: Chunks\\nSemantic Splitting & Boundary Normalization", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sub_embed [label="Subsystem 3: Embeddings\\nDense Neural Vector Transformation", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sub_meta [label="Subsystem 4: Metadata\\nAttribute Extraction & Inverted Payload Index", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            sub_doc -> sub_chunk -> sub_embed;
            sub_chunk -> sub_meta;
        }

        subgraph cluster_store {
            label="Phase 2: Storage and Indexing Engine Subsystems";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            sub_store [label="Subsystem 5: Vector Storage\\nSegment Files, SSD mmap & Write-Ahead Logs", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            sub_index [label="Subsystem 6: Index\\nHNSW Proximity Graphs & IVF Quantization Partitions", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            sub_embed -> sub_store;
            sub_embed -> sub_index;
            sub_meta -> sub_store [label="Link Payloads"];
        }

        subgraph cluster_retrieval {
            label="Phase 3: Query Execution and Retrieval Subsystems";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            sub_query [label="Subsystem 7: Query Engine\\nQuery Validation, Filter AST & Plan Optimization", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            sub_calc [label="Subsystem 8: Similarity Calculation\\nSIMD AVX-512 / GPU Distance Dot Product", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            sub_layer [label="Subsystem 9: Retrieval Layer\\nPriority Queue Top-K Extraction & Payload Merge", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            sub_query -> sub_calc;
            sub_index -> sub_calc [label="Graph Traversal"];
            sub_calc -> sub_layer;
            sub_store -> sub_layer [label="Fetch Payloads"];
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Documents
st.header("2. Documents (Subsystem 1): Unstructured Raw Ingestion Payloads", divider="rainbow")
st.markdown("""
The **Documents Subsystem** serves as the initial ingestion ingress in a vector database:
* **Format Diversity**: Ingests heterogeneous data including Markdown, PDF text streams, HTML webpages, JSON API outputs, and raw source code.
* **Document Hygiene**: Strips redundant markup, deduplicates exact matches, normalizes unicode characters, and extracts structural hierarchy.
* **Preserving Provenance**: Generates a canonical document URI and tracks original author, access permissions, and version hashes.
""")
st.divider()

# Section 3: Chunks
st.header("3. Chunks (Subsystem 2): Semantic Splitting and Token Bounding", divider="rainbow")
st.markdown("""
Foundation embedding models have strict token context limits (e.g., 512, 1024, or 8192 tokens). 
The **Chunks Subsystem** divides long documents into digestible, semantically coherent segments:

* **Fixed-Window Chunking**: Splits text into fixed character or word spans with sliding overlaps (e.g., 500 characters with 50 character overlap). Simple, but can bisect sentences midway.
* **Recursive Character Splitting**: Attempts to split along paragraph breaks (`\\n\\n`), then sentence boundaries (`.`), and finally word boundaries (` `) to preserve grammatical coherence.
* **Document Hierarchy Splitting**: Splits along Markdown headers (`#`, `##`, `###`) to ensure chunks correspond to self-contained conceptual sections.
""")
st.divider()

# Section 4: Embeddings
st.header("4. Embeddings (Subsystem 3): High-Dimensional Geometric Projections", divider="rainbow")
st.markdown("""
The **Embeddings Subsystem** invokes deep neural networks to convert tokenized chunk strings into dense float32 coordinate arrays:

* **Dimensionality Selection**: Balances representational depth against memory consumption. Common standards include 384 dimensions (`MiniLM`), 1024 dimensions (`BGE`), and 1536 dimensions (`OpenAI`).
* **L2 Unit Normalization**: Automatically normalizes vectors to unit length (`magnitude = 1.0`), converting heavy Cosine calculations into ultra-fast SIMD Dot Products.
* **Batching and Throughput**: Batches embedding requests to saturate GPU inference pipelines during bulk ingestion runs.
""")
st.divider()

# Section 5: Metadata
st.header("5. Metadata (Subsystem 4): Payload Schemas and Filter Inverted Indices", divider="rainbow")
st.markdown("""
The **Metadata Subsystem** stores and indexes structured key-value attributes associated with each vector:

* **Payload Structure**: Stores JSON objects containing fields such as `user_id`, `tenant_id`, `created_at`, `category`, and `is_public`.
* **Secondary Inverted Indices**: To ensure that queries like `WHERE category = 'finance'` execute in microseconds, the database builds inverted indices mapping metadata values directly to internal vector record IDs.
* **Tenancy Enforcement**: Guarantees cryptographically isolated queries in multi-tenant SaaS environments.
""")
st.divider()

# Section 6: Vector Storage
st.header("6. Vector Storage (Subsystem 5): Segment Files, Memory Mappings & WAL", divider="rainbow")
st.markdown("""
The **Vector Storage Subsystem** governs persistence, durability, and memory management:

* **Write-Ahead Log (WAL)**: All incoming upsert operations are appended to an on-disk WAL before memory acknowledgement, guaranteeing zero data loss during sudden crashes.
* **Immutable Segment Architecture**: Vectors are batched into immutable segment blocks on disk (similar to LSM-Trees in RocksDB). Background compactor threads periodically merge small segments into optimized large segments.
* **Memory-Mapped I/O (`mmap`)**: Maps disk segment files directly into virtual memory space, allowing the operating system kernel to cache hot vectors in RAM while paging cold vectors to SSD automatically.
""")
st.divider()

# Section 7: Index
st.header("7. Index (Subsystem 6): HNSW Proximity Graphs & IVF Partitions", divider="rainbow")
st.markdown("""
The **Index Subsystem** creates high-dimensional routing topologies to bypass brute-force linear scans:

* **HNSW (Hierarchical Navigable Small World)**:
  * Multi-layer skip list graph. Layer 2 contains sparse expressway links for rapid global navigation; Layer 0 contains the dense full-corpus nearest neighbor graph.
  * Search begins at the top layer, performs greedy routing to find the closest node, and descends to lower layers, converging on the nearest neighbors in `O(log N)` time.
* **IVF (Inverted File Index)**:
  * Clusters vectors into `K` Voronoi cells using k-means. At query time, only the centroids nearest the query vector are inspected (cell probing).
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
State-of-the-art vector engines like **Qdrant** and **Milvus** utilize **Single-Stage Filtered HNSW Traversal**. 
Rather than searching vectors first and discarding mismatches later, the HNSW graph walker checks metadata index predicates 
at every step of graph navigation, ensuring that zero irrelevant vectors are ever visited.
""")
st.divider()

# Section 8: Query Engine
st.header("8. Query Engine (Subsystem 7): Parsing, Filter AST & Execution Planning", divider="rainbow")
st.markdown("""
The **Query Engine Subsystem** orchestrates the search request lifecycle:
* **Validation**: Checks that the query vector dimensionality exactly matches the collection configuration.
* **Filter Abstract Syntax Tree (AST)**: Parses complex metadata boolean expressions (`$and`, `$or`, `$in`, `$gte`).
* **Cost-Based Query Planning**: Determines whether to execute an index-driven graph walk, pre-filtered ID scan, or brute-force search based on metadata selectivity.
""")
st.divider()

# Section 9: Similarity Calculation
st.header("9. Similarity Calculation (Subsystem 8): Hardware SIMD & AVX-512 Distance Computation", divider="rainbow")
st.markdown("""
The **Similarity Calculation Subsystem** is the performance-critical computational engine:

* **Hardware SIMD Instructions**: Modern CPUs process 8 to 16 floating-point multiplications simultaneously using AVX-512 (Intel/AMD) or ARM NEON (Apple Silicon/AWS Graviton) instructions.
* **Distance Functions**:
  * **Normalized Dot Product**: Computes `Sum(u_i * v_i)` across 1536 dimensions in sub-microsecond time.
  * **Squared Euclidean Distance**: Bypasses the expensive square root operation during candidate ranking, as relative order is preserved.
* **Quantized Distance Table Lookups**: With Product Quantization (PQ), distances are computed via pre-calculated lookup tables rather than floating-point math.
""")
st.divider()

# Section 10: Retrieval Layer
st.header("10. Retrieval Layer (Subsystem 9): Candidate Ranking and Top-K Extraction", divider="rainbow")
st.markdown("""
The **Retrieval Layer Subsystem** synthesizes the final response delivered to the client application:

* **Bounded Min-Heap**: Maintains a priority queue of size `K`. As candidates are evaluated during graph traversal, only vectors with scores higher than the current heap minimum are retained.
* **Payload Hydration**: Once the top-K record IDs are finalized, the layer retrieves raw text chunks and metadata dictionaries from the Document Storage Subsystem.
* **Citation & Score Bounding**: Packages final scores, distance metrics, and source URLs into structured API responses.
""")
st.divider()

# Section 11: Production Implementation
st.header("11. Production Implementation: Modular Multi-Subsystem Vector Engine", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating how the 9 subsystems interact in a cohesive modular architecture.
""")

st.code('''
import numpy as np
from typing import List, Dict, Any, Optional

class ModularVectorEngine:
    """Demonstrates modular subsystem interaction: Ingestion -> Storage -> Index -> Retrieval."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.metadata_index: Dict[str, List[str]] = {}

    def ingest(self, doc_id: str, text_chunk: str, raw_vector: np.ndarray, metadata: Dict[str, Any]):
        """Executes Subsystems 1 to 5: Normalization, Storage, and Inverted Indexing."""
        # Embeddings & Normalization
        norm = np.linalg.norm(raw_vector)
        norm_v = raw_vector / norm if norm > 0 else raw_vector
        
        # Storage Subsystem
        self.storage[doc_id] = {
            "chunk": text_chunk,
            "vector": norm_v.astype(np.float32),
            "metadata": metadata
        }
        
        # Metadata Subsystem: Inverted Category Index
        cat = metadata.get("category", "default")
        if cat not in self.metadata_index:
            self.metadata_index[cat] = []
        self.metadata_index[cat].append(doc_id)

    def query(self, query_vector: np.ndarray, top_k: int = 2, filter_cat: Optional[str] = None) -> List[Dict[str, Any]]:
        """Executes Subsystems 7 to 9: Filter Resolution, SIMD Calculation, and Top-K Extraction."""
        norm = np.linalg.norm(query_vector)
        q_norm = query_vector / norm if norm > 0 else query_vector

        # Candidate selection via Metadata Filter
        candidate_ids = self.metadata_index.get(filter_cat, list(self.storage.keys())) if filter_cat else list(self.storage.keys())

        # SIMD Distance & Min-Heap Top-K Extraction
        ranked = []
        for cid in candidate_ids:
            record = self.storage[cid]
            sim = float(np.dot(q_norm, record["vector"]))
            ranked.append({
                "id": cid,
                "score": round(sim, 4),
                "chunk": record["chunk"],
                "metadata": record["metadata"]
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
''', language="python")
st.divider()

# Section 12: Interactive Workbench
st.header("12. Interactive Workbench: Live Architecture Subsystem Explorer & Query Tracer", divider="rainbow")
st.markdown("""
Trace an incoming query across the 9 internal architectural subsystems of a production vector database.
""")

tracer = ArchitectureTracer()
trace_data = tracer.trace()

st.subheader("Subsystem Execution Flow")
df_trace = pd.DataFrame(trace_data)
st.dataframe(df_trace, use_container_width=True)

selected_step = st.slider("Step Through Architecture Pipeline:", min_value=1, max_value=9, value=1)
active_subsystem = trace_data[selected_step - 1]

st.info(f"**Step {active_subsystem['step']} — {active_subsystem['subsystem']} Subsystem**\n\n{active_subsystem['action']}")
st.divider()

# Section 13: Technical Interview Preparation & Production Traps
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: Why do vector databases decouple Vector Storage from Payload / Metadata Storage?", expanded=True):
    st.markdown("""
    **Answer:**
    Vector embeddings are dense numerical arrays (e.g., 1536 float32 values = 6 KB per record). Graph traversal algorithms 
    like HNSW require evaluating tens of thousands of vector distances per second, necessitating maximum memory throughput 
    and SIMD cache alignment.
    
    In contrast, document payloads (raw text chunks, PDF pages, JSON objects) can reach 50–200 KB per record and are only needed 
    *after* the top-K record IDs are identified. Storing vectors and payloads together would cause massive memory bloat, thrashing 
    CPU caches and reducing search throughput by 80%. Decoupling allows vectors to reside in high-speed RAM/mmap while payloads 
    remain on inexpensive SSD storage.
    """)

with st.expander("Q2: What is the purpose of the Write-Ahead Log (WAL) in a vector database?"):
    st.markdown("""
    **Answer:**
    Building HNSW graph indices and updating quantization codebooks are compute-intensive operations that occur asynchronously in memory. 
    If a database server crashes or loses power before memory segments are flushed to disk, unindexed vectors would be permanently lost. 
    The WAL ensures durability by sequentially appending every insertion, update, and deletion to disk before acknowledging the client. 
    Upon restart, the database replays the WAL to reconstruct in-memory state.
    """)

with st.expander("Q3: How does memory-mapped I/O (`mmap`) enable vector databases to scale beyond physical RAM?"):
    st.markdown("""
    **Answer:**
    `mmap` maps disk-backed segment files directly into the operating system's virtual memory address space. 
    The database accesses vector coordinates as if they were in RAM, and the OS kernel automatically handles page faults—loading 
    frequently queried 'hot' vectors into RAM and evicting 'cold' vectors to SSD. This allows a server with 64 GB of RAM to host 
    hundreds of gigabytes of vector data without crashing with Out-Of-Memory errors.
    """)

with st.expander("Q4: What is the difference between HNSW and IVF indexing architectures?"):
    st.markdown("""
    **Answer:**
    * **HNSW (Hierarchical Navigable Small World)**: A multi-layer graph index. Highest recall precision (95-99%), lowest latency (1-5ms), fast incremental insertions, but consumes high RAM (often 1.5x-2x vector size) due to graph edge pointers.
    * **IVF (Inverted File)**: A clustering index using Voronoi cells. Partitions space into clusters; queries probe only the nearest centroids. Smaller memory footprint, faster build times, but lower recall on boundary vectors and requires periodic retraining.
    """)

with st.expander("Q5: Why do vector search engines compute Squared Euclidean Distance rather than standard Euclidean Distance?"):
    st.markdown("""
    **Answer:**
    Standard Euclidean Distance requires calculating `sqrt(Sum (u_i - v_i)^2)`. The square root function is computationally expensive 
    and adds significant CPU instruction cycles across millions of candidate comparisons. Because the square root is a strictly 
    monotonic function (if `a^2 > b^2`, then `a > b`), relative distance ranking is identical without the square root. Omitting it 
    boosts distance computation throughput by 20–30%.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Memory Exhaustion During Background Segment Compaction (Fatal Crash)**
    Running vector ingestion at high throughput while background compactor threads merge immutable disk segments. Compaction creates duplicate temporary indices in memory, causing total RAM usage to spike past system limits and triggering Linux OOM killer process termination.
    """)
    st.error("""
    **Trap 2: Graph Disconnection from Unpruned Node Deletions (Silent Recall Loss)**
    Deleting thousands of vector nodes from an HNSW graph without re-linking adjacent neighbor edges. Over time, the proximity graph fragments into isolated sub-graphs, causing queries entering from upper layers to miss relevant documents completely.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Thrashing Virtual Memory with Random mmap Access (Severe Latency Collapse)**
    Hosting huge HNSW indices on SSD via `mmap` when RAM is severely undersized (<10% of dataset). Graph traversal hops randomly across the address space, triggering continuous disk page faults and increasing search latency from 5ms to 800ms.
    """)
    st.warning("""
    **Trap 4: Missing Inverted Secondary Indices on Filter Payloads (High CPU Load)**
    Executing queries with complex metadata filters against payload fields that lack inverted secondary indexes. The database is forced to evaluate filters iteratively across millions of graph nodes, driving CPU usage to 100%.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: Why separate Vector Storage from Document Payload Storage?")
        st.markdown("**Back**: To prevent large text payloads from polluting CPU L1/L2 caches and RAM during high-speed vector distance computations.")
    with st.container(border=True):
        st.markdown("**Front**: What does Single-Stage Filtered HNSW Traversal accomplish?")
        st.markdown("**Back**: It evaluates metadata predicates on the fly during graph navigation, preventing traversal into invalid or unauthorized nodes.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why do vector databases omit the square root in L2 distance?")
        st.markdown("**Back**: Square root is strictly monotonic and does not change relative rankings; omitting it saves substantial CPU instruction cycles.")
    with st.container(border=True):
        st.markdown("**Front**: What is the function of the Write-Ahead Log (WAL)?")
        st.markdown("**Back**: To provide durability and crash recovery by sequentially recording all writes to disk before in-memory index updates.")

st.divider()

# Section 14: Self-Assessment Checklist
st.header("14. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Vector Database Internal Subsystems before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the 9 core subsystems and how data flows from ingestion to retrieval.", value=True, key="chk_vdb_arch_1")
    chk2 = st.checkbox("I can explain why vector coordinates and raw text payloads are stored separately.", value=True, key="chk_vdb_arch_2")
    chk3 = st.checkbox("I understand the role of the Write-Ahead Log (WAL) and immutable segment files.", value=True, key="chk_vdb_arch_3")
with c2:
    chk4 = st.checkbox("I know how hardware SIMD (AVX-512) accelerates vector similarity calculations.", value=True, key="chk_vdb_arch_4")
    chk5 = st.checkbox("I understand the difference between HNSW graph indices and IVF Voronoi partitions.", value=True, key="chk_vdb_arch_5")
    chk6 = st.checkbox("I can identify production traps such as compaction OOM crashes, graph fragmentation, and mmap thrashing.", value=True, key="chk_vdb_arch_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Vector Database Architecture Mastery Demonstrated!** You have completed Chapter 1 of the Vector Database curriculum.")
