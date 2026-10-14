import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="10.3 Popular Index Algorithms - Vector DB Curriculum",
    page_icon="🧬",
    layout="wide"
)

# Title & Overview
st.title("10.3 Popular Index Algorithms: HNSW, IVF, PQ, ScaNN, Annoy & FAISS")
st.caption("Chapter 10: Vector Indexing | Domain: 4. Vector Database")
st.markdown("""
Over the past decade, computer science researchers and industry leaders (Meta, Google, Spotify) have invented distinct 
mathematical approaches to solve the Approximate Nearest Neighbor (ANN) problem at web scale.

No single algorithm dominates all dimensions:
* **HNSW (Hierarchical Navigable Small World)** offers the highest recall and lowest query latency, but consumes substantial RAM.
* **IVF (Inverted File Index)** is fast to build and partitions space into Voronoi cells.
* **PQ (Product Quantization)** compresses vectors by 8x to 32x, fitting billions of vectors into server memory.
* **ScaNN (Scalable Nearest Neighbors)** by Google uses anisotropic vector quantization to maximize inner product recall.
* **Annoy (Approximate Nearest Neighbors Oh Yeah)** by Spotify builds a forest of random projection trees.
* **FAISS (Facebook AI Similarity Search)** by Meta provides the foundational C++ GPU/CPU engine unifying these algorithms.

In this lesson, we break down how each algorithm works internally, analyze their memory and speed trade-offs, 
and use an interactive algorithm selector to pick the ideal index for any production workload.
""")

st.divider()

# Section 1: The Vector Index Algorithm Taxonomy
st.header("1. The Vector Index Algorithm Taxonomy", divider="rainbow")

col_tax1, col_tax2 = col_tax1, col_tax2 = st.columns([1.1, 0.9])

with col_tax1:
    st.markdown("""
    ### The Four Algorithmic Families
    Every vector indexing algorithm falls into one of four mathematical families:
    
    1. **Graph-Based (HNSW, NSG, DiskANN)**:
       * Connects nearest vectors with directional edges.
       * Multi-layer skip graphs route queries from coarse long-range hops down to dense local neighborhoods in `O(log N)`.
       * **Strength**: State-of-the-art recall and sub-5ms latency. Supports streaming real-time inserts.
    2. **Cluster / Inverted Index (IVF, IVFFlat)**:
       * Clusters vectors into `nlist` Voronoi centroids using k-means.
       * At search time, only the vectors assigned to the closest `nprobe` centroids are scanned.
       * **Strength**: Fast build time, low index overhead.
    3. **Compression & Quantization (PQ, OPQ, SQ)**:
       * Lossy compression that quantizes continuous float32 vectors into 8-bit cluster indices (codebooks).
       * Distance calculation is replaced by fast table lookups in CPU L1/L2 cache.
       * **Strength**: Fits 100M+ vectors in RAM; up to 95% memory savings.
    4. **Tree-Based / Random Projection (Annoy, KD-Tree)**:
       * Recursively splits high-dimensional space with random hyperplanes into a forest of binary trees.
       * **Strength**: Read-only memory-mapped static indices; ideal for desktop or offline recommendations.
    """)

with col_tax2:
    st.markdown("### Algorithmic Lineage Map")
    tax_dot = """
    digraph AlgorithmTaxonomy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Root [label="Vector Indexing\\nParadigms", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Graph [label="Graph-Based\\nHNSW / DiskANN", fillcolor="#D1FAE5", color="#059669"];
        Cluster [label="Clustering\\nIVFFlat", fillcolor="#FEF3C7", color="#D97706"];
        Quant [label="Quantization\\nPQ / OPQ / SQ", fillcolor="#E0E7FF", color="#4338CA"];
        Tree [label="Tree Partition\\nAnnoy (Spotify)", fillcolor="#FCE7F3", color="#DB2777"];

        Hybrid [label="Anisotropic & Hybrids\\nScaNN (Google) / IVF-PQ (FAISS)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        Root -> Graph;
        Root -> Cluster;
        Root -> Quant;
        Root -> Tree;

        Cluster -> Hybrid;
        Quant -> Hybrid;
    }
    """
    st.graphviz_chart(tax_dot, use_container_width=False)

st.divider()

# Section 2: Deep Dive into the 6 Major Algorithms
st.header("2. Deep Dive into the Six Major Algorithms", divider="rainbow")

tab_alg1, tab_alg2, tab_alg3, tab_alg4, tab_alg5, tab_alg6 = st.tabs([
    "1. HNSW",
    "2. IVF & IVFFlat",
    "3. Product Quantization (PQ)",
    "4. ScaNN (Google)",
    "5. Annoy (Spotify)",
    "6. Meta FAISS"
])

with tab_alg1:
    st.markdown("""
    ### 1. Hierarchical Navigable Small World (HNSW)
    * **Creator**: Yury Malkov & Dmitry Yashunin (2018).
    * **Core Concept**: Multi-layer skip-graph. Top layers have few sparse nodes with long-distance links; bottom layers have all nodes with dense local links.
    * **Search Traversal**: Greedy routing starts at the top entry node, hops to the closest neighbor, drops down a layer when no closer neighbor exists, and finishes at Layer 0.
    * **Key Parameters**: `M` (max edges per node, typically 16-64), `efConstruction` (build depth, 100-200), `efSearch` (query depth, 32-128).
    * **Default Engine**: Powers **ChromaDB**, Pinecone, Weaviate, Qdrant, and Milvus.
    """)

with tab_alg2:
    st.markdown("""
    ### 2. Inverted File Index (IVF & IVFFlat)
    * **Creator**: Sivic & Zisserman (2003); popularized by FAISS.
    * **Core Concept**: Run k-means clustering on the dataset to create `nlist` Voronoi centroids (e.g. 4,096 centroids). Every vector is assigned to its nearest centroid.
    * **Search Traversal**:
      1. Compute distance from query vector to all `nlist` centroids (fast: only 4,096 comparisons).
      2. Pick the top `nprobe` closest centroids (e.g. `nprobe = 16`).
      3. Scan all vectors contained inside those 16 Voronoi cells.
    * **IVFFlat**: Stores raw uncompressed float32 vectors inside each inverted list.
    * **Limitation**: Must be retrained if data distribution shifts significantly.
    """)

with tab_alg3:
    st.markdown("""
    ### 3. Product Quantization (PQ)
    * **Creator**: Herve Jegou, Matthieu Douze, Cordelia Schmid (2011).
    * **Core Concept**: Lossy vector compression.
      1. Split a `D`-dimensional vector (e.g. 1536-dim) into `M` sub-vectors of dimension `D / M` (e.g. 96 sub-vectors of 16 dims).
      2. Run k-means on each sub-space to generate 256 cluster centroids (represented by a single 8-bit byte: 0-255).
      3. Replace the 1536 float32 values (6,144 bytes) with **96 bytes** (one byte per centroid index).
    * **Compression Ratio**: Cuts RAM consumption by **8x to 64x**.
    * **Asymmetric Distance Computation (ADC)**: The query vector is kept as uncompressed float32, while stored vectors are quantized. Distances are computed via precomputed lookup tables.
    """)

with tab_alg4:
    st.markdown("""
    ### 4. ScaNN (Scalable Nearest Neighbors - Google)
    * **Creator**: Google Research (Guo et al., 2020).
    * **Core Concept**: Anisotropic Vector Quantization.
    * **Why Standard PQ Fails on Maximum Inner Product Search (MIPS)**: Standard PQ minimizes mean squared error in all directions equally. However, for dot product similarity, error **orthogonal** to the vector does not affect the dot product, while error **parallel** to the vector drastically distorts the dot product.
    * **ScaNN Innovation**: Formulates an anisotropic loss function that penalizes parallel quantization error much more heavily than orthogonal error.
    * **Result**: Matches or beats HNSW QPS on MIPS with significantly less memory. Powers Google Cloud Vertex AI Vector Search.
    """)

with tab_alg5:
    st.markdown("""
    ### 5. Annoy (Approximate Nearest Neighbors Oh Yeah)
    * **Creator**: Erik Bernhardsson (Spotify, 2013).
    * **Core Concept**: Forest of Random Projection Trees.
      1. Pick two random points and create a dividing hyperplane equidistant between them.
      2. Split points into left and right child nodes. Repeat recursively until each leaf has fewer than `K` items.
      3. Build a forest of `T` trees (e.g. 50-100 trees) with different random hyperplanes.
    * **Search Traversal**: Traverse down all trees in parallel, collect candidate leaves, and evaluate exact distances.
    * **Strength**: Read-only mmap files; static Spotify music recommendation playlists.
    * **Limitation**: Cannot insert new vectors dynamically; index must be rebuilt from scratch.
    """)

with tab_alg6:
    st.markdown("""
    ### 6. Meta FAISS (Facebook AI Similarity Search)
    * **Creator**: Meta AI Research (Johnson, Douze, Jegou, 2017).
    * **Core Concept**: The world's most widely adopted low-level C++/CUDA vector indexing library.
    * **Composability**: Allows composing indices like building blocks:
      * `IndexFlatL2`: Brute force exact Euclidean search.
      * `IndexIVFFlat`: Inverted file with raw vectors.
      * `IndexIVFPQ`: Inverted file combined with Product Quantization (extreme scale).
      * `IndexHNSWFlat`: HNSW graph over raw vectors.
    * **GPU Acceleration**: Highly optimized CUDA kernels capable of evaluating billions of distances per second on Nvidia A100/H100 GPUs.
    """)

st.divider()

# Section 3: The Comprehensive Architectural Benchmark Matrix
st.header("3. The Comprehensive Architectural Benchmark Matrix", divider="rainbow")

st.markdown("""
Compare the six index algorithms across the seven core production engineering criteria:
""")

BENCHMARK_MATRIX = [
    {
        "Algorithm": "Flat (Brute Force)",
        "Recall@10": "100.0%",
        "Query Latency": "Very High (>200ms at scale)",
        "RAM Footprint": "Low (Raw vectors only)",
        "Build Time": "Zero (No index)",
        "Streaming Inserts": "✅ Instant",
        "Primary Use Case": "Ground truth evaluation & <10k vectors"
    },
    {
        "Algorithm": "HNSW",
        "Recall@10": "97% - 99.5%",
        "Query Latency": "Ultra-Low (<3ms)",
        "RAM Footprint": "High (Raw + Graph Pointers)",
        "Build Time": "Moderate (Graph hops)",
        "Streaming Inserts": "✅ Native dynamic graph insertion",
        "Primary Use Case": "Enterprise RAG, ChromaDB, Real-time Search"
    },
    {
        "Algorithm": "IVFFlat",
        "Recall@10": "90% - 96%",
        "Query Latency": "Low (5 - 15ms)",
        "RAM Footprint": "Low-Moderate (Centroids + Raw)",
        "Build Time": "Fast (k-means training)",
        "Streaming Inserts": "⚠️ Requires centroid assignment",
        "Primary Use Case": "Medium datasets (100k - 5M vectors)"
    },
    {
        "Algorithm": "IVF-PQ",
        "Recall@10": "85% - 94%",
        "Query Latency": "Low-Moderate (8 - 20ms)",
        "RAM Footprint": "Extremely Low (8x - 32x compressed)",
        "Build Time": "Slow (k-means + codebook training)",
        "Streaming Inserts": "⚠️ Degrades if distribution drifts",
        "Primary Use Case": "Billion-scale web search (Pinterest, Meta)"
    },
    {
        "Algorithm": "ScaNN (Google)",
        "Recall@10": "94% - 98%",
        "Query Latency": "Ultra-Low (<3ms on MIPS)",
        "RAM Footprint": "Low (Anisotropic Quantization)",
        "Build Time": "Moderate-Slow (Optimization)",
        "Streaming Inserts": "❌ Batch re-indexing required",
        "Primary Use Case": "Google Vertex AI, e-commerce dot product"
    },
    {
        "Algorithm": "Annoy (Spotify)",
        "Recall@10": "88% - 94%",
        "Query Latency": "Moderate (10 - 25ms)",
        "RAM Footprint": "Moderate (mmap file)",
        "Build Time": "Fast (Random tree splits)",
        "Streaming Inserts": "❌ Read-only; cannot update",
        "Primary Use Case": "Batch recommendations & offline embedded apps"
    }
]

st.dataframe(pd.DataFrame(BENCHMARK_MATRIX), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Production Algorithm Recommender
st.header("4. Interactive Workbench: Production Algorithm Recommender", divider="rainbow")

st.markdown("""
Input your application scale, latency constraints, and infrastructure budget to receive a recommended 
indexing algorithm with estimated resource sizing.
""")

col_rec_in1, col_rec_in2 = col_rec_in1, col_rec_in2 = st.columns([1, 1])

with col_rec_in1:
    st.subheader("1. System Requirements")
    target_vectors = st.selectbox(
        "Total Target Vector Scale:",
        [
            "Small: 5,000 - 50,000 vectors",
            "Medium: 100,000 - 1,000,000 vectors",
            "Large: 5,000,000 - 20,000,000 vectors",
            "Web-Scale: 50,000,000 - 500,000,000+ vectors"
        ],
        index=1
    )
    
    update_pattern = st.radio(
        "Update & Ingestion Pattern:",
        [
            "Real-Time Streaming (Continuous adds/updates 24/7)",
            "Batch Periodic (Nightly full re-indexes or offline jobs)"
        ]
    )

with col_rec_in2:
    st.subheader("2. Infrastructure & SLA Budget")
    target_sla = st.radio(
        "Latency & Recall Priority:",
        [
            "Maximum Speed & Recall (RAG < 5ms, Recall > 97%)",
            "Balanced Performance (Recall ~ 92-95%, latency < 15ms)",
            "Maximum Memory Economy (Fit massive scale in minimal RAM)"
        ]
    )
    
    dims_choice = st.selectbox("Embedding Model Dimension:", [384, 768, 1536], index=2)

# Recommendation Logic
st.divider()

rec_alg = ""
rec_rationale = ""
rec_ram_est = ""
rec_engine = ""

if "Small" in target_vectors:
    rec_alg = "Flat (Exact Scan) or Lightweight HNSW"
    rec_rationale = "At < 50k vectors, raw computation is tiny (~15-20ms max for flat, <1ms for HNSW). HNSW in ChromaDB gives instant sub-millisecond responses without complex tuning."
    rec_ram_est = "Under 350 MB RAM total."
    rec_engine = "ChromaDB (In-Memory or Persistent) / FAISS IndexFlatL2"
elif "Web-Scale" in target_vectors:
    if "Memory" in target_sla:
        rec_alg = "IVF-PQ (Inverted File + Product Quantization)"
        rec_rationale = "At 50M+ vectors, storing raw float32 vectors requires >300 GB of RAM. IVF-PQ compresses vectors by 16x-32x, allowing billion-scale retrieval on affordable commodity instances."
        rec_ram_est = "~15 GB - 25 GB RAM with 16x PQ compression."
        rec_engine = "FAISS (IndexIVFPQ) / Milvus / Qdrant with Scalar/Product Quantization"
    else:
        rec_alg = "DiskANN / Graph on NVMe SSD + Compressed Cache"
        rec_rationale = "Maintains high 96%+ recall at 50M scale by storing the full graph on fast NVMe SSDs and caching compressed vectors in RAM."
        rec_ram_est = "~30 GB RAM + 250 GB NVMe SSD."
        rec_engine = "DiskANN / Milvus Disk Index"
elif "Real-Time Streaming" in update_pattern:
    rec_alg = "HNSW (Hierarchical Navigable Small World)"
    rec_rationale = "HNSW natively supports live streaming inserts without retraining centroids or rebuilding binary trees. It delivers sub-3ms query latency with 98%+ recall."
    rec_ram_est = "~6 GB - 12 GB RAM for 1M 1536-dim vectors (including graph pointers)."
    rec_engine = "ChromaDB (Default) / Pinecone / Weaviate / Qdrant"
else:
    if "Maximum Memory Economy" in target_sla:
        rec_alg = "IVF-PQ or ScaNN"
        rec_rationale = "Batch-oriented ingestion allows optimal k-means centroid training. ScaNN or IVF-PQ maximizes throughput while slashing memory usage."
        rec_ram_est = "~1.5 GB - 3 GB RAM for 1M vectors."
        rec_engine = "Google ScaNN / FAISS IndexIVFPQ"
    else:
        rec_alg = "HNSW or IVFFlat"
        rec_rationale = "Provides top-tier recall (98%) and sub-5ms latency for standard enterprise generative AI applications."
        rec_ram_est = "~8 GB RAM."
        rec_engine = "ChromaDB / FAISS IndexHNSWFlat"

col_rec_out1, col_rec_out2 = st.columns([1.2, 0.8])

with col_rec_out1:
    st.subheader(f"🏆 Recommended Architecture: {rec_alg}")
    st.success(f"**Engineering Rationale**: {rec_rationale}")
    st.info(f"**Recommended Engine / Framework**: `{rec_engine}`")

with col_rec_out2:
    st.subheader("Resource Estimation")
    st.metric(label="Estimated RAM Sizing", value=rec_ram_est)
    st.markdown("""
    **Production Checklist**:
    * Validate embedding dimension consistency.
    * Establish an automated Recall@10 regression monitor.
    * Benchmark p99 query latency under concurrent thread load.
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why does Google ScaNN outperform standard Product Quantization on dot-product (MIPS) search?"):
    st.markdown("""
    **Answer**:
    Standard Product Quantization (PQ) trains codebooks to minimize Mean Squared Error (Euclidean L2 reconstruction error) uniformly across all spatial dimensions. 
    However, for Maximum Inner Product Search (`<Q, V>`), the geometric error can be decomposed into:
    1. **Parallel Error**: Error parallel to the vector, which directly changes the vector's magnitude and distorts the dot product.
    2. **Orthogonal Error**: Error perpendicular to the vector, which has minimal impact on the inner product.
    
    Google ScaNN introduces **Anisotropic Vector Quantization**, which heavily penalizes parallel quantization error while tolerating orthogonal error. This preserves accurate dot-product ranking with high compression ratios.
    """)

with st.expander("Q2: What is Asymmetric Distance Computation (ADC) in Product Quantization?"):
    st.markdown("""
    **Answer**:
    In Product Quantization, you have two choices for computing distance:
    * **Symmetric Distance Computation (SDC)**: Both the query vector and the stored database vectors are quantized. Very fast, but suffers from double quantization error.
    * **Asymmetric Distance Computation (ADC)**: The incoming query vector is kept as **exact uncompressed float32**, while the stored database vectors are quantized codes.
    
    In ADC, the system precomputes a small lookup table of distances from the query's sub-vectors to all 256 codebook centroids. Calculating distance to any stored vector then requires only `M` simple table lookups and additions. ADC yields significantly higher recall than SDC with negligible additional compute.
    """)

with st.expander("Q3: Why can't Spotify's Annoy index support live streaming document insertions?"):
    st.markdown("""
    **Answer**:
    Annoy's index structure consists of a forest of binary trees built via random projection hyperplanes. 
    To maintain balanced tree depth and uniform spatial partitioning, the hyperplanes are fitted globally across the entire dataset. 
    Inserting a new vector after build time would require either:
    1. Traversing down an existing tree and placing it in a leaf, which quickly unbalances the tree and destroys recall guarantees, or
    2. Modifying hyperplanes, which invalidates all downstream branches.
    Therefore, Annoy is strictly designed as a **read-only, write-once static index** optimized for memory-mapped sharing across worker processes.
    """)

with st.expander("Q4: What is the primary operational trade-off of IVF compared to HNSW?"):
    st.markdown("""
    **Answer**:
    * **Build Time & Memory**: IVF builds 3x-5x faster than HNSW and requires zero graph edge pointer memory overhead (saving 25-50% RAM).
    * **Query Latency & Recall**: IVF exhibits lower recall for the same latency because vectors near Voronoi cell boundaries are easily missed unless `nprobe` is set very high.
    * **Dynamic Updates**: If new vectors deviate from the initial k-means training distribution, IVF centroids become skewed, requiring an expensive full index retraining and rebuild. HNSW handles dynamic streaming inserts natively.
    """)

with st.expander("Q5: What role does Meta FAISS play in the vector database ecosystem?"):
    st.markdown("""
    **Answer**:
    FAISS is not a standalone distributed database with HTTP APIs or metadata engines; it is the **foundational low-level C++ algorithmic kernel**. 
    Full vector databases like Milvus, ChromaDB, and search engines incorporate FAISS (or hnswlib) as their underlying mathematical computation engine for vector distance operations, while the database provides ACID transactions, metadata filtering, storage persistence, distributed replication, and REST/gRPC interfaces.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Applying IVF Without Representative Training Data**  
    Training IVF centroids on 5,000 documents when your target corpus will reach 5,000,000 documents leads to severe centroid drift and poor recall. Train centroids on at least 50k - 200k representative embeddings.
    """)
    st.warning("""
    **Trap 2: Using PQ on Low-Dimensional Vectors (< 64 dims)**  
    Product Quantization relies on breaking high-dimensional vectors into multiple sub-vectors. For vectors with 32 or 64 dimensions, PQ degrades recall drastically with minimal RAM benefit.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Choosing Annoy for Real-Time Streaming RAG**  
    Choosing Annoy for an interactive application where users upload documents in real time fails because Annoy cannot dynamically add records without rebuilding the entire forest. Use HNSW.
    """)
    st.warning("""
    **Trap 4: Forgetting to Warm Up the RAM Page Cache with mmap**  
    When deploying memory-mapped indices (FAISS/Annoy), the first query after a cold restart triggers disk I/O page faults, resulting in a 500ms latency spike. Execute warm-up queries on startup.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What makes HNSW the industry favorite?"):
        st.markdown("""
        **HNSW Strengths**:  
        Sub-3ms query latency, 98%+ recall, and native dynamic streaming inserts without index retraining.
        """)
    with st.expander("🗂️ Flashcard: What is Product Quantization (PQ)?"):
        st.markdown("""
        **PQ Compression**:  
        Lossy compression that splits vectors into sub-vectors and encodes them with 8-bit centroid indices, cutting RAM by 8x-32x.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What makes ScaNN unique?"):
        st.markdown("""
        **Anisotropic Loss**:  
        Penalizes parallel error heavily while ignoring orthogonal error to optimize Maximum Inner Product Search (dot product).
        """)
    with st.expander("🗂️ Flashcard: How does IVF search?"):
        st.markdown("""
        **Voronoi Partitioning**:  
        Compares query to `nlist` centroids, then scans only the vectors inside the closest `nprobe` cells.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know the four algorithmic families: Graph, Cluster, Quantization, and Random Projection.", value=True)
chk2 = st.checkbox("I can explain how HNSW, IVF, PQ, ScaNN, and Annoy work internally.", value=True)
chk3 = st.checkbox("I understand Asymmetric Distance Computation (ADC) in Product Quantization.", value=True)
chk4 = st.checkbox("I understand why Google ScaNN uses anisotropic quantization for dot product search.", value=True)
chk5 = st.checkbox("I can evaluate trade-offs between RAM footprint, build time, and query latency.", value=True)
chk6 = st.checkbox("I can select the optimal index algorithm for any scale, SLA, and update pattern.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Popular Index Algorithms and are ready for Chapter 10.4: HNSW Deep Dive (M, efConstruction, efSearch)!")
