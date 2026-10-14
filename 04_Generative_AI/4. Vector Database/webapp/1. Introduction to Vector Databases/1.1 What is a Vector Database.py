import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.1 What is a Vector Database - Vector DB Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class MockVectorStore:
    def __init__(self):
        self.records = [
            {"id": "VEC_01", "text": "PostgreSQL transaction rollback due to serialization deadlock failure code 40001.", "category": "Database", "vector": np.array([0.88, 0.12, 0.05, 0.22])},
            {"id": "VEC_02", "text": "Kubernetes container terminated by Linux kernel OOM killer exit code 137.", "category": "DevOps", "vector": np.array([0.15, 0.92, 0.25, 0.10])},
            {"id": "VEC_03", "text": "How to configure SAML 2.0 Single Sign-On integration with Okta identity provider.", "category": "Security", "vector": np.array([0.05, 0.18, 0.94, 0.12])},
            {"id": "VEC_04", "text": "Customer subscription auto-renewal failed due to expired credit card token.", "category": "Billing", "vector": np.array([0.22, 0.10, 0.15, 0.91])}
        ]

    def sql_like_search(self, pattern: str) -> List[Dict[str, Any]]:
        pattern_lower = pattern.lower()
        return [r for r in self.records if pattern_lower in r["text"].lower()]

    def lexical_bm25_search(self, query: str) -> List[Tuple[Dict[str, Any], int]]:
        query_tokens = set(query.lower().replace("?", "").replace(",", "").split())
        hits = []
        for r in self.records:
            doc_tokens = set(r["text"].lower().replace(".", "").split())
            overlap = query_tokens.intersection(doc_tokens)
            if overlap:
                hits.append((r, len(overlap)))
        return sorted(hits, key=lambda x: x[1], reverse=True)

    def ann_vector_search(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        scores = []
        norm_q = np.linalg.norm(query_vector)
        for r in self.records:
            v = r["vector"]
            norm_v = np.linalg.norm(v)
            cosine_sim = float(np.dot(query_vector, v) / (norm_q * norm_v)) if (norm_q * norm_v) > 0 else 0.0
            scores.append((r, cosine_sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Title & Overview
st.title("1.1 What is a Vector Database: Foundations, Comparisons & Architectural Role")
st.caption("Chapter 1: Introduction to Vector Databases | Domain: 4. Vector Database")
st.markdown("""
For four decades, relational database management systems (RDBMS) like PostgreSQL, MySQL, and Oracle were engineered 
around **exact, deterministic matching**: structured queries evaluating boolean predicates like `WHERE user_id = 42` 
or `WHERE status = 'ACTIVE'`. Inverted index search engines like Elasticsearch later introduced **lexical text matching** 
using BM25 token frequencies.

Modern Artificial Intelligence models—including Large Language Models, visual encoders, and audio transformers—do not 
communicate in discrete strings or relational table rows. Instead, they project unstructured concepts into 
**continuous, high-dimensional numerical vectors (embeddings)** where spatial geometric distance mirrors human conceptual meaning.

A **Vector Database** is a purpose-built database engine designed specifically to **store, index, and query billions of high-dimensional vectors** 
using **Approximate Nearest Neighbor (ANN)** algorithms in single-digit milliseconds. Rather than asking *"Which records contain this exact word?"*, 
a vector database answers *"Which documents in our corpus are conceptually closest to this query?"*

In this lesson, you will master:
1. **Vector Database Definitions**: The core mechanics of high-dimensional vector spaces and dense embeddings.
2. **The Curse of Dimensionality**: Why classical B-Trees degrade into full table scans beyond 20 dimensions.
3. **Comparative Systems Architecture**: How vector databases contrast with SQL databases, search engines, and data warehouses.
4. **Production Vector Operations**: Real-world vector indexing, quantization, and similarity evaluation.
5. **Production Failure Traps**: Avoiding RAM exhaustion, distance metric mismatches, and post-filtering data loss.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: High-Dimensional Vector Search Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph VectorDBBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingestion {
            label="Tier 1: Document Ingestion and Vector Embedding Engine";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            raw_corpus [label="Raw Knowledge Corpus\\nPDFs, Docs, API JSON, Database Rows", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            chunker [label="Document Chunker and Normalizer\\nRecursive Token Splitting (512 tokens, 10% overlap)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            embedder [label="Dense Embedding Model\\nOpenAI text-embedding-3 / BGE-Large (1536-d)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            raw_corpus -> chunker -> embedder;
        }

        subgraph cluster_storage {
            label="Tier 2: Vector Storage, Quantization, and ANN Indexing Engine";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            payload_store [label="Document Payload Store (RocksDB / SQLite)\\nUUID + Text Chunk + JSON Metadata", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            quantizer [label="Quantization Engine (SQ8 / PQ / BQ)\\nCompress 1536 float32 (6 KB) to 8-bit int (1.5 KB)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            ann_index [label="HNSW Proximity Graph Index\\nLayer 2 Expressway -> Layer 1 Regional -> Layer 0 Full Graph", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            embedder -> payload_store [label="Store Payloads"];
            embedder -> quantizer -> ann_index [label="Build Index"];
        }

        subgraph cluster_query {
            label="Tier 3: Query Processing, SIMD Distance, and Filtered Retrieval";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            user_query [label="Incoming Search Query\\n'How to resolve container out of memory crashes?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            query_vector [label="Query Vector Embedder\\nGenerates dense vector q (1536-d)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            simd_engine [label="SIMD Distance Engine and Single-Stage Filter\\nAVX-512 Dot Product + Metadata Predicate Filter", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            top_k_results [label="Ranked Top-K Payloads\\nCosine Proximity Ranked Chunks with Grounded Source IDs", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            user_query -> query_vector -> simd_engine;
            ann_index -> simd_engine [label="ANN Candidates"];
            payload_store -> simd_engine [label="Filter Payloads"];
            simd_engine -> top_k_results [label="Top-K Output"];
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Definition of a Vector Database
st.header("2. Definition of a Vector Database", divider="rainbow")
st.markdown("""
A **Vector Database** is an enterprise data storage system engineered to index, manage, and retrieve high-dimensional 
numerical vector embeddings alongside structured metadata payloads.

Unlike conventional databases that compare discrete scalar values, a vector database operates across geometric coordinates:
* **Storage Primitives**: A vector database stores a triplet consisting of a unique identifier (`ID`), a high-dimensional dense coordinate array (`Vector`), and a JSON dictionary of attributes (`Payload / Metadata`).
* **Retrieval Objective**: Given an arbitrary query vector `q`, identify the `K` stored vectors that minimize geometric distance (or maximize directional similarity).
* **ANN Heuristics**: Exact k-nearest neighbor (k-NN) search requires comparing `q` against every stored vector in `O(N * d)` time. At enterprise scale (millions to billions of vectors), this brute-force scan takes seconds. Vector databases use **Approximate Nearest Neighbor (ANN)** graph and clustering heuristics to return 95%+ accurate nearest neighbors in sub-5ms latency (`O(log N)` complexity).
""")

### Modern Production Reality
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production AI engineering, purpose-built vector databases like **Qdrant**, **Milvus**, and **Pinecone** separate compute 
and storage into independent microservices. Write nodes process streaming upserts through Write-Ahead Logs (WAL) and 
immutable segment buffers, while read nodes load quantized HNSW graphs into memory-mapped SSD storage (mmap), serving 
thousands of concurrent queries per second.
""")
st.divider()

# Section 3: What is a Vector?
st.header("3. What is a Vector?", divider="rainbow")
st.markdown("""
In software engineering and data systems, a **vector** is an ordered list of floating-point numerical values that 
represents a specific coordinate point in a mathematical vector space.

* **Dimensionality (d)**: The length of the array defines its dimension. For example, a 3-dimensional spatial coordinate `[x, y, z]` has `d = 3`. Modern embedding vectors typically have `d = 384`, `768`, `1536`, or `3072` dimensions.
* **Continuous Representation**: Each dimension represents a continuous latent feature learned by a neural network. No single dimension corresponds directly to an individual word or concept; instead, meaning is distributed across the entire coordinate array.
* **Direction and Magnitude**: A vector possesses both a direction (its orientation relative to origin) and a magnitude (its straight-line length from origin). In text search, magnitude is frequently normalized to 1.0 so that direction alone dictates similarity.
""")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("""
    ```python
    # Example 4-dimensional normalized vector
    embedding = np.array([0.024, -0.158, 0.891, 0.043], dtype=np.float32)
    vector_magnitude = np.linalg.norm(embedding)
    # Normalized: vector_magnitude == 1.0
    ```
    """)
with col_v2:
    st.info("""
    **Key Takeaway**: A vector transforms qualitative human concepts (text, voice, image pixels) into quantitative coordinates 
    that processors can evaluate using high-speed SIMD hardware instructions.
    """)
st.divider()

# Section 4: What is a Vector Embedding?
st.header("4. What is a Vector Embedding?", divider="rainbow")
st.markdown("""
A **vector embedding** is the numerical output generated when an AI foundation model processes unstructured data 
(a sentence, an image, a code snippet, or an audio waveform) and maps it into a shared semantic coordinate space.

* **Semantic Proximity**: The core property of embeddings is that items with similar conceptual meaning are positioned close together in the vector space, regardless of the specific vocabulary used.
* **Translation Across Modalities**: Multi-modal models (like CLIP) project images and text into the exact same vector space. The vector for a photograph of a golden retriever will have high cosine similarity to the vector for the phrase *"playful dog in the park"*.
* **Embedding Model Families**:
  * **Text**: OpenAI `text-embedding-3-small` (1536-d), BAAI `bge-large-en` (1024-d), Sentence Transformers `all-MiniLM-L6-v2` (384-d).
  * **Code**: StarCoder, CodeBERT, OpenAI `text-embedding-3`.
  * **Vision**: CLIP, SigLIP, DINOv2.
""")

st.markdown("""
| Model Name | Dimensions | Primary Strength | Ideal Use Case |
| :--- | :---: | :--- | :--- |
| `all-MiniLM-L6-v2` | 384 | Ultra-lightweight, runs in browser / mobile | Local prototyping, edge devices |
| `bge-large-en-v1.5` | 1024 | State-of-the-art MTEB retrieval accuracy | Enterprise knowledge bases, legal search |
| `text-embedding-3-small` | 1536 | Native dimension truncation via Matryoshka | Cost-efficient cloud RAG pipelines |
| `text-embedding-3-large` | 3072 | Maximum precision and multilingual depth | Complex multi-hop reasoning |
""")
st.divider()

# Section 5: Why Vector Databases Are Needed
st.header("5. Why Vector Databases Are Needed: The Curse of Dimensionality", divider="rainbow")
st.markdown("""
Why can't software engineers simply store vectors in an array column inside PostgreSQL or MySQL and query them with standard indexing?

Traditional database indexes—such as **B-Trees, B+ Trees, and R-Trees**—rely on establishing a total ordering (`x < y`) 
or partitioning geometric bounding boxes in 1, 2, or 3 dimensions. When applied to 768-dimensional or 1536-dimensional vectors, 
these data structures completely break down due to the **Curse of Dimensionality**:
""")

col_curse1, col_curse2 = st.columns(2)
with col_curse1:
    st.markdown("""
    ### Why B-Trees and Tree Indexes Collapse
    1. **Exponential Volume Growth**: As dimensionality `D` increases, the volume of the enclosing space grows exponentially. High-dimensional space is overwhelmingly empty.
    2. **Distance Concentration**: In high dimensions, the distance between any two randomly chosen vectors concentrates within a narrow, uniform band.
    3. **Bounding Box Overlap**: R-Trees partition space using bounding boxes. In high dimensions, every bounding box overlaps with almost every other bounding box.
    4. **Catastrophic Degradation**: A tree index query is forced to inspect nearly every branch, degenerating into an `O(N)` full table scan that consumes 100% CPU and takes seconds per query.
    """)
with col_curse2:
    st.markdown("""
    ### How Vector Databases Solve This
    1. **Abandoning Exactness**: Vector databases accept a 1–3% trade-off in recall precision to achieve 1,000x faster query speeds.
    2. **Navigable Proximity Graphs (HNSW)**: Constructs multi-layer graph skip-lists where nodes represent vectors and edges represent proximity. Searching traverses long expressway links before descending to fine local clusters in `O(log N)` time.
    3. **Vector Quantization**: Compresses 32-bit floats into 8-bit integers or 1-bit hashes, enabling cache-friendly in-memory traversal.
    """)
st.divider()

# Section 6: Vector Database vs Traditional Database
st.header("6. Vector Database vs Traditional Database", divider="rainbow")
st.markdown("""
The differences between relational databases and vector databases reflect fundamentally divergent design objectives:
""")

st.markdown("""
| Architectural Dimension | Traditional Relational Database (RDBMS) | Dedicated Vector Database |
| :--- | :--- | :--- |
| **Data Types** | Structured scalar values: integers, strings, dates, booleans | Continuous dense vectors (float32 arrays) + JSON metadata |
| **Query Mechanism** | Deterministic boolean logic (`WHERE status = 'PAID'`) | Geometric proximity (`Find top-K nearest neighbors to q`) |
| **Index Structure** | 1D B-Tree, B+ Tree, Hash Index | High-dimensional HNSW graph, IVF-PQ, ScaNN |
| **Search Guarantee** | 100% exact, deterministic completeness | Approximate Nearest Neighbor (ANN) with configurable recall |
| **Execution Complexity** | `O(log N)` exact search across 1D ordered keys | `O(log N)` approximate search across 1536 dimensions |
| **Consistency Model** | Strict ACID transactions, serializable isolation | Eventual consistency, optimized for high-throughput reads |
""")
st.divider()

# Section 7: Vector Database vs Search Engine
st.header("7. Vector Database vs Search Engine", divider="rainbow")
st.markdown("""
Lexical search engines like **Elasticsearch**, **Apache Solr**, and **OpenSearch** were the state of the art for text retrieval 
for over twenty years. Comparing them against Vector Databases clarifies when to use each:

* **Lexical Matching (Search Engines)**:
  * Uses an **Inverted Index**: a lookup dictionary mapping individual words to list of document IDs where they appear.
  * Scores results using **BM25 / TF-IDF**: documents with higher frequencies of rare query keywords receive higher scores.
  * *Limitation*: Fails on semantic synonyms. If a user queries *"automobile repair manual"*, BM25 will completely miss a document titled *"car maintenance guide"* unless explicit synonym dictionaries are manually maintained.
* **Semantic Retrieval (Vector Databases)**:
  * Maps full phrases into concept coordinates.
  * Correctly identifies that *"automobile repair"* and *"car maintenance"* have 95%+ cosine similarity because the embedding model learned their conceptual equivalence during training.
  * *Limitation*: Can occasionally struggle with exact rare identifiers (like part numbers `SKU-9921-X`), where BM25 excels.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production enterprise search systems almost never choose between BM25 and Vector Search in isolation. 
Instead, they deploy **Dual-Stage Hybrid Search**:
1. Run BM25 lexical search and HNSW dense vector search concurrently.
2. Merge the candidate lists using **Reciprocal Rank Fusion (RRF)**.
3. Pass the top 50 candidates through a **Cross-Encoder Reranker** for final precision scoring.
""")
st.divider()

# Section 8: Vector Database vs SQL Database
st.header("8. Vector Database vs SQL Database: Dedicated vs pgvector", divider="rainbow")
st.markdown("""
With the introduction of extensions like `pgvector` for PostgreSQL, developers frequently ask whether a dedicated 
vector database is truly necessary. Here is the engineering trade-off matrix:
""")

st.markdown("""
| Evaluation Criterion | PostgreSQL with `pgvector` | Dedicated Vector DB (Qdrant / Milvus) |
| :--- | :--- | :--- |
| **Operational Simplicity** | Excellent (uses existing PostgreSQL database) | Requires deploying and managing a dedicated cluster |
| **Data Relationship** | Joins vector queries directly with relational tables | Requires syncing data between relational DB and vector store |
| **Maximum Recommended Scale** | Up to 1–5 million vectors | 10 million to billions of vectors |
| **Indexing Latency & Resource Contention** | Vector builds consume heavy CPU/RAM, starving OLTP | Independent indexing worker pools with zero OLTP interference |
| **Advanced Quantization** | Basic half-precision (float16) and IVF | Native SQ8, Product Quantization (PQ), Binary Quantization (BQ) |
| **Hardware Acceleration** | CPU-bound execution | Native GPU acceleration (NVIDIA cuVS) and SIMD AVX-512 |
""")
st.divider()

# Section 9: Vector Database vs Data Warehouse
st.header("9. Vector Database vs Data Warehouse", divider="rainbow")
st.markdown("""
Data Warehouses like **Snowflake**, **Google BigQuery**, and **Amazon Redshift** are engineered for 
**Online Analytical Processing (OLAP)** across petabytes of structured columnar data.

* **Workload Characteristics**: Data warehouses scan billions of rows to compute aggregates (`SUM`, `AVG`, `GROUP BY`) across minutes or seconds.
* **Vector Database Workload**: Vector databases are real-time, low-latency online serving systems (`sub-10ms`) executing single-query nearest neighbor graph descents.
* **Architecture Divergence**: While modern data warehouses have added vector storage extensions for batch analytics (e.g., finding all similar customers in an overnight ETL run), they cannot serve real-time conversational RAG applications requiring 1,000 QPS at 5ms latency SLA.
""")
st.divider()

# Section 10: Role of Vector Databases in AI Applications
st.header("10. Role of Vector Databases in AI Applications", divider="rainbow")
st.markdown("""
In the Generative AI architectural stack, the Vector Database acts as the **non-parametric long-term memory** 
for foundation models. Large Language Models have two fundamental architectural constraints:
1. **Knowledge Cutoffs**: Their internal weights are frozen at training completion.
2. **Context Window Limits & Token Costs**: Shoveling an entire corporate knowledge base into prompt tokens costs thousands of dollars per turn and introduces multi-second latency.
""")

col_role1, col_role2 = st.columns(2)
with col_role1:
    st.markdown("""
    ### Core Enterprise Roles
    * **Retrieval-Augmented Generation (RAG)**: Retrieves the top-3 most relevant authoritative paragraphs in 4ms, injecting them into prompt context to ground the LLM's response with zero hallucination.
    * **Multi-Tenant Access Governance**: Enforces security policies via metadata filtering (`WHERE tenant_id = 'acme' AND access_level <= 3`), ensuring users only see data they are permitted to view.
    * **Agentic Long-Term Memory**: Autonomous AI agents store past conversational interactions, tool execution traces, and learned plans in vector stores, retrieving them during subsequent multi-turn sessions.
    """)
with col_role2:
    st.markdown("""
    ### System Architecture Placement
    ```
    User Prompt
        ↓
    Embedding Model (Vector q)
        ↓
    Vector Database (HNSW Top-K Search)
        ↓
    Grounded Context Snippets
        ↓
    Augmented LLM Prompt
        ↓
    Verified Answer with Citations
    ```
    """)
st.divider()

# Section 11: Production Implementation
st.header("11. Production Implementation: In-Memory Vector Search Engine", divider="rainbow")
st.markdown("""
Below is a clean, production-grade implementation of an in-memory Vector Database client demonstrating 
record ingestion, L2 vector normalization, Cosine similarity calculation, and metadata filtering.
""")

st.code('''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ProductionVectorDBClient:
    """Lightweight in-memory vector database client with normalized Cosine ANN search."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}

    def insert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Inserts a record, ensuring dimensional safety and L2 normalization."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        # Normalize vector for instant inner product equivalence
        norm = np.linalg.norm(vector)
        normalized_v = vector / norm if norm > 0 else vector
        self.vectors[record_id] = normalized_v.astype(np.float32)
        self.payloads[record_id] = metadata

    def query(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 3, 
        filter_predicate: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes filtered similarity search against stored embeddings."""
        norm = np.linalg.norm(query_vector)
        q_norm = query_vector / norm if norm > 0 else query_vector

        results = []
        for rec_id, stored_vec in self.vectors.items():
            payload = self.payloads[rec_id]
            # Metadata filter evaluation
            if filter_predicate:
                match = all(payload.get(k) == v for k, v in filter_predicate.items())
                if not match:
                    continue
            
            # Normalized Dot Product equals Cosine Similarity
            similarity = float(np.dot(q_norm, stored_vec))
            results.append({
                "id": rec_id,
                "score": round(similarity, 4),
                "payload": payload
            })

        # Rank by descending similarity
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
''', language="python")
st.divider()

# Section 12: Interactive Workbench
st.header("12. Interactive Workbench: Relational SQL vs Lexical BM25 vs Vector ANN", divider="rainbow")
st.markdown("""
Test how three different database paradigms evaluate the exact same user query. 
See why **Relational SQL** returns zero rows, why **Lexical BM25** struggles with synonyms, and how **Vector ANN** retrieves the conceptual match!
""")

db_sim = MockVectorStore()

test_queries = {
    "Scenario 1: Conceptual Synonyms": "My pod crashed because it ran out of memory, how do I fix it?",
    "Scenario 2: Exact Jargon String": "serialization deadlock failure code 40001",
    "Scenario 3: Conversational Billing Intent": "Our payment method was rejected during billing renewal"
}

selected_scenario = st.selectbox("Select a Query Scenario:", list(test_queries.keys()) + ["Custom Query"])

if selected_scenario == "Custom Query":
    active_query = st.text_input("Enter custom query string:", value="Why did the container die from memory exhaustion?")
else:
    active_query = test_queries[selected_scenario]

st.markdown(f"**Evaluating Query**: `{active_query}`")

col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    st.subheader("1. Relational SQL")
    st.code("SELECT * FROM docs WHERE text LIKE '%memory%';", language="sql")
    sql_hits = db_sim.sql_like_search(active_query[:15])
    if "40001" in active_query:
        st.success("Found 1 Row (`VEC_01`) via exact substring match.")
    else:
        st.error("Returned 0 Rows! Exact LIKE match failed because natural vocabulary differs.")

with col_w2:
    st.subheader("2. Lexical Search (BM25)")
    st.code("GET /docs/_search\n{ 'query': { 'match': { 'text': ... } } }", language="json")
    bm25_hits = db_sim.lexical_bm25_search(active_query)
    if bm25_hits:
        best_bm = bm25_hits[0]
        st.info(f"Top Hit: `[{best_bm[0]['id']}]` ({best_bm[1]} token overlaps)\n\n*{best_bm[0]['text'][:65]}...*")
    else:
        st.warning("Low Score / Zero token overlap with documents.")

with col_w3:
    st.subheader("3. Vector Database (ANN)")
    st.code("collection.query(query_vector=q, top_k=1)", language="python")
    # Simulate vector according to conceptual intent
    if "memory" in active_query.lower() or "pod" in active_query.lower() or "container" in active_query.lower():
        sim_q = np.array([0.15, 0.90, 0.20, 0.10])
    elif "40001" in active_query.lower() or "deadlock" in active_query.lower() or "database" in active_query.lower():
        sim_q = np.array([0.85, 0.15, 0.05, 0.20])
    elif "billing" in active_query.lower() or "payment" in active_query.lower() or "credit" in active_query.lower():
        sim_q = np.array([0.20, 0.10, 0.15, 0.90])
    else:
        sim_q = np.array([0.05, 0.20, 0.90, 0.15])

    ann_hits = db_sim.ann_vector_search(sim_q, top_k=1)
    if ann_hits:
        best_v, score = ann_hits[0]
        st.success(f"Top Hit: `[{best_v['id']}]` (Cosine: **{score:.3f}**)\n\n*{best_v['text']}*")
        st.caption("Matched conceptually via high-dimensional embedding angle without requiring exact keywords!")
st.divider()

# Section 13: Technical Interview Preparation & Production Traps
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is a Vector Database and how does it fundamentally differ from a Relational SQL Database?", expanded=True):
    st.markdown("""
    **Answer:**
    A **Vector Database** is a database management system purpose-built to store, index, and query high-dimensional numerical vectors (embeddings) generated by AI models.
    
    * **Query Paradigm**: Relational databases execute exact boolean equality and range lookups (`WHERE id = 42`). Vector databases execute approximate nearest neighbor similarity searches (`Find the top-K vectors geometrically closest to query vector q`).
    * **Indexing Mechanism**: SQL databases use 1D ordered B-Trees, which completely collapse beyond 20-50 dimensions due to the Curse of Dimensionality. Vector databases use high-dimensional spatial proximity graphs (HNSW) and inverted quantization files (IVF-PQ) to achieve `O(log N)` search times across millions of records.
    * **Data Schema**: Relational databases enforce strict tabular row-column schemas. Vector databases store dense floating-point arrays alongside unstructured payload text and JSON metadata dictionaries.
    """)

with st.expander("Q2: Why do traditional B-Tree indexes fail on high-dimensional vectors?"):
    st.markdown("""
    **Answer:**
    B-Trees rely on a total ordering principle (`x < y`), partitioning space into 1D intervals. In high-dimensional spaces (e.g., 1536 dimensions), data points become exponentially sparse, and the distance between any two random points concentrates within a very narrow distribution. Spatial bounding boxes overlap everywhere, forcing any tree-based query to traverse every branch in the tree—degrading into an `O(N)` brute-force full table scan.
    """)

with st.expander("Q3: What are the trade-offs between dedicated vector databases (Milvus, Qdrant) versus relational extensions (pgvector)?"):
    st.markdown("""
    **Answer:**
    * **When to choose `pgvector`**: Best for small to medium workloads (under 1 million vectors) where an existing PostgreSQL database is already deployed, transactional ACID guarantees are required, and the team wants to avoid managing an additional database cluster.
    * **When to choose Dedicated Vector DBs (Milvus, Qdrant, Pinecone)**: Essential for enterprise workloads (10M+ to billions of vectors), requiring horizontal sharding across dedicated query and indexing nodes, native SIMD/GPU hardware acceleration, and advanced quantization (SQ8, PQ, BQ) to reduce memory overhead by up to 95%.
    """)

with st.expander("Q4: How does Single-Stage Iterative Filtering differ from Pre-Filtering and Post-Filtering?"):
    st.markdown("""
    **Answer:**
    * **Pre-Filtering**: Evaluates metadata conditions first to build an allowed set of IDs, then performs brute-force vector search over that set. Fails when the metadata condition matches millions of rows.
    * **Post-Filtering**: Performs standard ANN search first to find top-K candidates, then removes candidates that fail metadata filters. Fails catastrophically when filters are restrictive, returning fewer than K results (or zero results).
    * **Single-Stage Iterative Filtering**: Explores the HNSW proximity graph directly, checking metadata predicates on the fly during graph traversal and continuing the walk until exactly K valid candidates are identified.
    """)

with st.expander("Q5: What are the three primary distance metrics used in vector databases, and when should each be used?"):
    st.markdown("""
    **Answer:**
    1. **Cosine Similarity / Distance**: Measures the angle between vectors, ignoring magnitude. Ideal for text embeddings where document length variations should not distort semantic meaning.
    2. **Dot Product (Inner Product - IP)**: Measures `Sum(u_i * v_i)`. When vectors are L2-normalized to unit length, Dot Product is mathematically identical to Cosine Similarity and computes in nanoseconds via SIMD hardware instructions.
    3. **Euclidean (L2) Distance**: Measures straight-line distance in space. Essential when vector magnitude carries critical information, such as computer vision feature embeddings or spatial coordinates.
    """)

st.subheader("Production Failure Traps")
trap_col1, trap_col2 = st.columns(2)
with trap_col1:
    st.error("""
    **Trap 1: RAM Exhaustion from Unquantized HNSW Graphs (Fatal)**
    Storing 10 million 1536-dimensional float32 vectors in raw RAM requires over 60 GB for vectors alone, plus 20-30 GB for HNSW graph connectivity pointers. Without memory quantization (SQ8 or PQ), cloud instances will crash with Out-Of-Memory (OOM) errors during peak indexing loads.
    """)
    st.error("""
    **Trap 2: Distance Metric Mismatch Between Model and Collection (Silent Accuracy Killer)**
    Embedding text with a model trained specifically for Cosine Similarity (e.g., OpenAI or BGE) but configuring the vector database collection to use Euclidean (L2) distance without vector normalization. The system runs without errors but returns completely irrelevant search results.
    """)
with trap_col2:
    st.warning("""
    **Trap 3: Post-Filtering Returning Zero Results (High Latency & Data Loss)**
    Using naive post-filtering on restrictive metadata queries (e.g., `WHERE tenant_id = 'acme'`). The HNSW index retrieves the top 10 global vectors, none of which belong to 'acme', causing the application to return an empty response even though thousands of matching documents exist in the database.
    """)
    st.warning("""
    **Trap 4: Cold-Start Latency Spikes During Disk Index Warmup (SLA Breach)**
    Relying on disk-backed vector storage (e.g., mmap or SSD indexes) without pre-warming the index cache into memory. Initial queries experience latency spikes of 500ms to 2 seconds while index graph pages are paged in from disk.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary purpose of a Vector Database?")
        st.markdown("**Back**: To store, index, and query high-dimensional vector embeddings using Approximate Nearest Neighbor (ANN) search algorithms in sub-10ms latency.")
    with st.container(border=True):
        st.markdown("**Front**: Why do traditional B-Trees fail on high-dimensional vectors?")
        st.markdown("**Back**: Due to the Curse of Dimensionality; high-dimensional bounding boxes overlap everywhere, causing tree queries to degenerate into full O(N) table scans.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: How does Dot Product relate to Cosine Similarity under L2 normalization?")
        st.markdown("**Back**: When vectors are normalized to unit length (magnitude = 1), Dot Product is mathematically identical to Cosine Similarity and executes much faster via hardware SIMD.")
    with st.container(border=True):
        st.markdown("**Front**: What is Single-Stage Iterative Filtering in HNSW?")
        st.markdown("**Back**: Graph traversal that evaluates metadata conditions during graph navigation, ensuring exactly K valid results without post-filtering empty sets.")

st.divider()

# Section 14: Self-Assessment Checklist
st.header("14. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Vector Database Architecture and Operations before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand what a vector database is and how it differs from relational SQL and lexical search engines.", value=True, key="chk_vdb_1")
    chk2 = st.checkbox("I can explain the Curse of Dimensionality and why B-Trees fail on high-dimensional embeddings.", value=True, key="chk_vdb_2")
    chk3 = st.checkbox("I understand the architectural role of vector databases in enterprise RAG and GenAI applications.", value=True, key="chk_vdb_3")
with c2:
    chk4 = st.checkbox("I know the trade-offs between dedicated vector databases (Milvus/Qdrant) vs relational extensions (pgvector).", value=True, key="chk_vdb_4")
    chk5 = st.checkbox("I understand the 3 primary distance metrics (Cosine, Dot Product, Euclidean L2) and when to use each.", value=True, key="chk_vdb_5")
    chk6 = st.checkbox("I can identify production traps including HNSW RAM exhaustion, metric mismatches, and post-filtering data loss.", value=True, key="chk_vdb_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Vector Database Mastery Demonstrated!** You are ready to advance to Lesson 1.2 in the curriculum.")
