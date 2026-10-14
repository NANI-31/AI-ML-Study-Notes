import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.2 Why Vector Databases are Important - Vector DB Curriculum",
    page_icon="🗄️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulation
class ApplicationVectorEngine:
    """Demonstrates multi-workload vector routing across semantic and recommendation use cases."""
    def __init__(self):
        self.catalog = {
            "Semantic Search": [
                {"id": "DOC_101", "name": "Incident Postmortem: API Latency", "category": "Engineering", "score": 0.941},
                {"id": "DOC_102", "name": "Service Mesh Circuit Breaker Guide", "category": "DevOps", "score": 0.887}
            ],
            "Recommendation Systems": [
                {"id": "PROD_201", "name": "Ergonomic Mechanical Keyboard", "category": "Hardware", "score": 0.963},
                {"id": "PROD_202", "name": "Dual Monitor Desk Arm Mount", "category": "Hardware", "score": 0.912}
            ],
            "Retrieval-Augmented Generation": [
                {"id": "RAG_301", "name": "Corporate Travel Reimbursement Policy v3", "category": "HR", "score": 0.982},
                {"id": "RAG_302", "name": "Per Diem Rate Schedule 2026", "category": "Finance", "score": 0.895}
            ],
            "Image Search": [
                {"id": "IMG_401", "name": "sunset_over_pacific_coast.jpg", "category": "Visuals", "score": 0.925},
                {"id": "IMG_402", "name": "golden_hour_ocean_waves.png", "category": "Visuals", "score": 0.908}
            ]
        }

    def dispatch(self, application_type: str) -> List[Dict[str, Any]]:
        return self.catalog.get(application_type, [])

# Title & Overview
st.title("1.2 Why Vector Databases are Important: Core Enterprise Applications")
st.caption("Chapter 1: Introduction to Vector Databases | Domain: 4. Vector Database")
st.markdown("""
Vector databases have evolved from niche academic research components into critical, tier-0 data infrastructure 
for modern software engineering. While relational databases model strict structural relationships and key-value 
caches deliver millisecond lookups, vector databases unlock the ability to compute **conceptual affinity across unstructured data**.

Over 80% of enterprise information—including customer service tickets, emails, code repositories, architectural drawings, 
and audio recordings—is completely unstructured. Vector databases bridge the gap between foundation AI models and 
operational software systems by indexing these assets into searchable semantic coordinate spaces.

In this lesson, you will master:
1. **The 8 Foundational Application Domains**: From semantic search and real-time recommendations to multi-modal discovery.
2. **Retrieval-Augmented Generation (RAG)**: How vector storage prevents hallucinations and cuts LLM token costs.
3. **Similarity Search Dynamics**: Evaluating distance metrics and feature embeddings across text and imagery.
4. **Production Routing Architecture**: Dispatching multi-modal queries with sub-10ms latency SLAs.
5. **Production Engineering Traps**: Navigating cold cache penalties, dimension drift, and uncalibrated similarity scores.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: High-Performance Vector Application Topologies", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ApplicationTopologies {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_apps {
            label="Tier 1: Enterprise AI Consumer Applications";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            chat_app [label="Enterprise Chatbot & Copilot\\nNatural language question answering", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            rec_app [label="E-Commerce Recommendation Engine\\nBehavioral and visual affinity matching", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            search_app [label="Cross-Modal Enterprise Search\\nPDFs, Slack channels, code, diagrams", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_dispatch {
            label="Tier 2: Unified Vector Dispatcher & Query Transformation";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            query_encoder [label="Multi-Modal Embedder Pipeline\\nText, Vision, and Clickstream Encoders", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            filter_resolver [label="Security & Multi-Tenant Resolver\\nAttaches Role-Based Access Control filters", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            chat_app -> query_encoder;
            rec_app -> query_encoder;
            search_app -> query_encoder;
            query_encoder -> filter_resolver;
        }

        subgraph cluster_vectordb {
            label="Tier 3: Distributed Vector Database Cluster (HNSW / IVF-PQ)";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            text_index [label="Text & Document Index\\n1536-d HNSW Graph with mmap SSD", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            image_index [label="Vision & Multimodal Index\\n512-d CLIP Graph with GPU Acceleration", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            user_index [label="User Profile & Preference Index\\n256-d Two-Tower Collaborative Vectors", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];

            filter_resolver -> text_index [label="Route Document Query"];
            filter_resolver -> image_index [label="Route Vision Query"];
            filter_resolver -> user_index [label="Route RecSys Query"];
        }

        subgraph cluster_output {
            label="Tier 4: Grounded Results & Synthesis";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            grounded_out [label="Top-K Ranked Candidates with Confidence Metrics\\nSub-10ms end-to-end SLA delivery to clients", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            text_index -> grounded_out;
            image_index -> grounded_out;
            user_index -> grounded_out;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Semantic Search
st.header("2. Semantic Search: Understanding Intent Beyond Keywords", divider="rainbow")
st.markdown("""
Traditional search engines rely on lexical keyword matching: they split text into tokens, stem words, and compute BM25 scores. 
If a user searches for *"low latency network driver"*, BM25 prioritizes documents containing those exact keywords.

**Semantic Search** fundamentally transforms this paradigm by converting queries into conceptual coordinates:
* **Synonym Independence**: It understands that *"speedy HTTP transport"*, *"high throughput socket interface"*, and *"low latency network driver"* occupy virtually the identical coordinate region in 1536-dimensional space.
* **Typo and Paraphrase Resilience**: Minor misspellings and sentence rearrangements do not distort the overall vector angle.
* **Polysemy Resolution**: Dense models distinguish between the river *"bank"* and the financial *"bank"* based on surrounding contextual coordinates.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production search stacks combine dense semantic vectors with sparse BM25 token indices in a **Hybrid Search Pipeline**. 
Dense vectors capture broad conceptual meaning, while sparse indices guarantee exact precision on alphanumeric part numbers 
and system codes (e.g., `RFC-7231` or `CVE-2024-3094`).
""")
st.divider()

# Section 3: Similarity Search
st.header("3. Similarity Search: High-Dimensional Distance Mechanics", divider="rainbow")
st.markdown("""
At its algorithmic core, every vector database operation is a **Similarity Search**: measuring the geometric proximity 
between an unknown query vector `q` and a pre-indexed corpus of vectors `{v_1, v_2, ..., v_N}`.

Vector databases support three foundational distance calculations:
1. **Cosine Similarity**: Measures the directional cosine angle between two vectors:
   `Cosine Similarity = (u . v) / (||u|| * ||v||)`.
   * Completely scale-invariant: focuses strictly on conceptual angle, ignoring text length variations.
2. **Dot Product (Inner Product - IP)**: Measures `Sum(u_i * v_i)`.
   * When vectors are L2-normalized to unit magnitude (`||u|| = 1.0`), Dot Product equals Cosine Similarity and computes in nanoseconds via CPU SIMD instructions.
3. **Euclidean (L2) Distance**: Measures straight-line geometric distance:
   `L2 Distance = Sqrt(Sum (u_i - v_i)^2)`.
   * Sensitive to vector magnitude; widely used in computer vision embeddings and facial recognition.
""")

st.markdown("""
| Metric Name | Value Range | Optimal Interpretation | Production Use Case |
| :--- | :--- | :--- | :--- |
| **Cosine Similarity** | -1.0 to +1.0 | Higher is closer (+1.0 = identical angle) | Natural language text, RAG retrieval |
| **Cosine Distance** | 0.0 to 2.0 | Lower is closer (0.0 = identical) | Distance metric in ChromaDB, Qdrant |
| **Dot Product (Normalized)** | -1.0 to +1.0 | Higher is closer (+1.0 = identical) | Hardware-accelerated search (AVX-512) |
| **Euclidean L2 Distance** | 0.0 to Infinity | Lower is closer (0.0 = identical) | Computer vision, spatial cluster analysis |
""")
st.divider()

# Section 4: Recommendation Systems
st.header("4. Recommendation Systems: Two-Tower Collaborative Architectures", divider="rainbow")
st.markdown("""
Modern recommendation platforms (such as YouTube, Spotify, and Netflix) power candidate generation using 
**Two-Tower Neural Networks** connected directly to vector databases:

* **User Tower**: An ongoing neural network maps user history (watch history, click sequences, time of day, device) into a 256-dimensional **User Vector `u`**.
* **Item Tower**: Processes product catalogs (metadata, genre, transcript, cover art) into an **Item Vector `v`**.
* **Real-Time Candidate Generation**: When a user opens the mobile app, their user vector `u` is sent to the vector database. The database retrieves the top 100 nearest item vectors in 3ms, filtering out items the user has already consumed.
""")

st.info("""
**Scalability Impact**: Instead of evaluating complex neural ranking networks across 50 million catalog items, 
the vector database narrows the candidate pool from 50,000,000 items to the top 100 relevant candidates in 3 milliseconds.
""")
st.divider()

# Section 5: Retrieval-Augmented Generation (RAG)
st.header("5. Retrieval-Augmented Generation (RAG)", divider="rainbow")
st.markdown("""
Retrieval-Augmented Generation (RAG) is the primary driver behind enterprise vector database adoption:

* **Solving Knowledge Cutoffs**: LLM model weights cannot incorporate private company wikis or events that occurred after pre-training. Vector databases inject up-to-the-minute documentation directly into prompt context.
* **Eliminating Hallucinations**: By instructing the LLM to *"Answer only using the verified context passages provided below"*, hallucinations are dramatically reduced.
* **Token Budget Management**: Passing an entire 300-page operational manual into an LLM context window costs dollars per query and adds 6+ seconds of latency. Vector search isolates the 3 most relevant paragraphs, reducing token consumption by 98%.
""")
st.divider()

# Section 6: Image Search and Multi-Modal Discovery
st.header("6. Image Search and Multi-Modal Discovery", divider="rainbow")
st.markdown("""
Vector databases natively bridge disparate modalities using **Contrastive Language-Image Pre-Training (CLIP)** models:

* **Cross-Modal Coordinate Space**: CLIP trains an image encoder (Vision Transformer) and a text encoder simultaneously, ensuring that an image of a red sports car and the text phrase *"crimson convertible on highway"* project to near-identical coordinate vectors.
* **Text-to-Image Querying**: Users search visual catalogs using descriptive natural language without requiring manual tagging or metadata annotations.
* **Image-to-Image Querying**: Uploading a reference photograph retrieves visually and stylistically analogous products across millions of SKU images.
""")
st.divider()

# Section 7: Document Search and Enterprise Discovery
st.header("7. Document Search and Enterprise Discovery", divider="rainbow")
st.markdown("""
Enterprise intranets struggle with information fragmentation across Google Drive, Confluence, Jira, and Slack. 
Vector databases power centralized enterprise discovery engines:

* **Chunk-Level Indexing**: Long-form PDFs and wikis are divided into semantic chunks (e.g., 512 tokens with 10% overlap).
* **Cross-Silo Unification**: A single vector search query simultaneously scans engineering documentation, sales decks, and legal contracts.
* **Access Control Filtering**: Inverted metadata filters evaluate permissions at query time (`WHERE department IN ('Finance', 'Legal')`), ensuring compliance with corporate governance.
""")
st.divider()

# Section 8: Question Answering Systems
st.header("8. Question Answering Systems", divider="rainbow")
st.markdown("""
Unlike traditional search engines that return a list of ten blue links requiring human reading, 
Vector-powered Question Answering systems synthesize direct, factual solutions:

1. **Query Processing**: User submits: *"What is the standard SLA for Sev-1 database outages?"*
2. **Dense Retrieval**: Vector database retrieves the specific SLA clauses from the cloud vendor agreement in 4ms.
3. **Synthesis Engine**: An LLM extracts the exact quantitative target (*"15-minute response time, 99.99% uptime guarantee"*) and presents it as a concise, direct answer accompanied by source citations.
""")
st.divider()

# Section 9: Personalization Systems
st.header("9. Personalization Systems", divider="rainbow")
st.markdown("""
Vector databases enable dynamic, real-time personalization without requiring expensive model retraining:

* **Dynamic User State**: As a user browses products or clicks articles, their session vector shifts continuously in real time.
* **Vector Averaging & Decay**: Exponentially decaying moving averages of recent click embeddings allow the vector database to adapt instantly to a user's evolving intent within a single session.
* **Zero Cold-Start Retraining**: New users can be personalized after just 2 clicks by querying the vector space around their immediate interaction trajectory.
""")
st.divider()

# Section 10: Production Implementation
st.header("10. Production Implementation: Unified Multi-Application Vector Search Dispatcher", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating an enterprise Vector Dispatcher that routes diverse 
application workloads (RAG, Recommendations, Semantic Search) through a unified vector execution pipeline.
""")

st.code('''
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseVectorDispatcher:
    """Production vector service routing queries across multiple specialized collection indices."""
    def __init__(self):
        self.indices: Dict[str, Dict[str, Any]] = {
            "rag_docs": {},
            "product_recs": {},
            "image_catalog": {}
        }

    def register_record(self, collection_name: str, record_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """Registers a normalized vector into a named collection."""
        norm = np.linalg.norm(vector)
        norm_v = vector / norm if norm > 0 else vector
        self.indices[collection_name][record_id] = {
            "vector": norm_v.astype(np.float32),
            "payload": payload
        }

    def search(
        self, 
        collection_name: str, 
        query_vector: np.ndarray, 
        top_k: int = 2,
        required_tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Executes filtered similarity search against the designated collection."""
        if collection_name not in self.indices:
            raise KeyError(f"Collection '{collection_name}' not registered in dispatcher.")

        norm = np.linalg.norm(query_vector)
        q_norm = query_vector / norm if norm > 0 else query_vector

        candidates = []
        for rec_id, item in self.indices[collection_name].items():
            payload = item["payload"]
            if required_tag and payload.get("tag") != required_tag:
                continue

            similarity = float(np.dot(q_norm, item["vector"]))
            candidates.append({
                "id": rec_id,
                "score": round(similarity, 4),
                "payload": payload
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
''', language="python")
st.divider()

# Section 11: Interactive Workbench
st.header("11. Interactive Workbench: 8-in-1 Vector Application Simulator", divider="rainbow")
st.markdown("""
Explore how a unified vector database dynamically adapts to different enterprise application workloads. 
Select an application domain to inspect the query transformation, retrieved candidates, and confidence metrics.
""")

app_engine = ApplicationVectorEngine()

app_options = list(app_engine.catalog.keys())
selected_app = st.selectbox("Select Enterprise Application Domain:", app_options)

candidates = app_engine.dispatch(selected_app)

col_work1, col_work2 = st.columns([1.2, 0.8])

with col_work1:
    st.subheader(f"Retrieved Candidates for '{selected_app}'")
    df_results = pd.DataFrame(candidates)
    st.dataframe(df_results, use_container_width=True)

with col_work2:
    st.subheader("Workload Characteristics")
    if selected_app == "Semantic Search":
        st.info("**SLA**: Sub-8ms | **Index**: HNSW float32 | **Primary Metric**: Cosine Distance")
        st.markdown("Handles synonym expansion and semantic phrase understanding across corporate wikis.")
    elif selected_app == "Recommendation Systems":
        st.info("**SLA**: Sub-4ms | **Index**: IVF-PQ (Quantized) | **Primary Metric**: Normalized Dot Product")
        st.markdown("Narrows millions of candidate products to top-k personalized recommendations per session.")
    elif selected_app == "Retrieval-Augmented Generation":
        st.info("**SLA**: Sub-10ms | **Index**: HNSW with Metadata Filter | **Primary Metric**: Cosine Similarity")
        st.markdown("Injects verified enterprise policy chunks into LLM prompt contexts to prevent hallucinations.")
    else:
        st.info("**SLA**: Sub-15ms | **Index**: Multi-Modal CLIP HNSW | **Primary Metric**: Cosine Distance")
        st.markdown("Evaluates text queries directly against image vector coordinates.")
st.divider()

# Section 12: Technical Interview Preparation & Production Traps
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: How do vector databases enable sub-5ms candidate generation in modern recommendation systems?", expanded=True):
    st.markdown("""
    **Answer:**
    Recommendation systems operate in two stages: **Candidate Generation** (retrieving hundreds of relevant items from millions of options) 
    and **Ranking** (evaluating complex, latency-heavy neural ranking models on those candidates).
    
    Vector databases handle Candidate Generation by representing users and items in a shared latent space (Two-Tower models). 
    Instead of running complex inference against 100 million items, the user's vector is queried against an HNSW or IVF index in 
    the vector database, retrieving the top 100 closest items in 3ms with sub-linear `O(log N)` complexity.
    """)

with st.expander("Q2: Why is Cosine Similarity preferred over Euclidean Distance for text embeddings?"):
    st.markdown("""
    **Answer:**
    In natural language processing, vector magnitude frequently correlates with document length or token frequency rather than 
    semantic meaning. Cosine Similarity normalizes vector length to unit magnitude, measuring strictly the **cosine of the angle** 
    between vectors. Consequently, two articles discussing the identical topic—one a short summary and the other an exhaustive whitepaper—will 
    have high Cosine Similarity, whereas Euclidean Distance would penalize their length difference with a large distance gap.
    """)

with st.expander("Q3: What role does metadata filtering play in enterprise RAG systems?"):
    st.markdown("""
    **Answer:**
    Without metadata filtering, vector search queries retrieve documents globally across the entire corpus. In enterprise environments, 
    this introduces severe security and compliance violations (e.g., non-admin users seeing confidential payroll files). 
    Metadata filtering allows queries to enforce hard security boundaries (`WHERE tenant_id = 'acme' AND confidentiality <= 'internal'`) 
    while preserving sub-10ms semantic proximity retrieval.
    """)

with st.expander("Q4: How do multi-modal models like CLIP allow text queries to search image databases?"):
    st.markdown("""
    **Answer:**
    CLIP uses dual neural network encoders: a Vision Transformer for images and a Text Transformer for language. During training on 
    hundreds of millions of image-text pairs, contrastive learning pulls the vector representations of matching images and captions 
    together into a unified multi-modal coordinate space. At search time, the text query is embedded into a vector that directly 
    overlaps with the visual feature vectors of relevant images.
    """)

with st.expander("Q5: What is the primary difference between Candidate Generation and Final Reranking in RAG?"):
    st.markdown("""
    **Answer:**
    * **Candidate Generation (First Stage)**: High-throughput, low-latency retrieval using Bi-Encoder embeddings and Vector DB HNSW index. Retrieves top 50–100 candidate chunks from millions of documents in sub-10ms.
    * **Final Reranking (Second Stage)**: High-precision, compute-intensive scoring using a Cross-Encoder model. Evaluates query and candidate text jointly through full self-attention layers to produce precise relevance scores for the top 3–5 chunks.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Dimension Drift Across Embedding Model Upgrades (Fatal Outage)**
    Upgrading an embedding model (e.g., from `text-embedding-ada-002` to `text-embedding-3-small`) without fully re-indexing the existing vector database. Queries fail immediately with dimensional mismatch exceptions, or return completely scrambled results if dimensions happen to match.
    """)
    st.error("""
    **Trap 2: Cold Cache Index Latency Spikes During Peak Traffic (SLA Breach)**
    Hosting large HNSW indices on disk or network storage without pre-warming the cache. When traffic surges, initial queries suffer 1–3 second latencies while index graph pages are paged into RAM from SSD storage.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Arbitrary Cosine Similarity Thresholding (Silent Recall Drop)**
    Hard-coding rigid score cutoff filters (e.g., `WHERE score >= 0.85`) across different embedding models. Different model architectures have completely distinct score distributions; a 0.75 in Cohere may represent higher confidence than a 0.85 in OpenAI.
    """)
    st.warning("""
    **Trap 4: Unindexed Payload Attributes in High-Throughput Filtering (High CPU Load)**
    Executing complex metadata filters against payload fields that lack inverted secondary indexes in the vector database. The database is forced to evaluate filters iteratively across millions of graph nodes, driving CPU usage to 100%.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: Why is Cosine Similarity scale-invariant?")
        st.markdown("**Back**: It divides the dot product by the product of vector magnitudes, isolating the directional angle while canceling out length discrepancies.")
    with st.container(border=True):
        st.markdown("**Front**: What is the Two-Tower architecture in recommendations?")
        st.markdown("**Back**: Separate neural encoders for users and items projecting into a shared vector space, enabling sub-5ms candidate generation via vector DB search.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: How does RAG optimize LLM prompt token costs?")
        st.markdown("**Back**: It retrieves only the 2-3 most relevant paragraphs rather than stuffing whole documents into prompts, reducing token overhead by up to 98%.")
    with st.container(border=True):
        st.markdown("**Front**: What is the role of Cross-Encoder Reranking in vector search?")
        st.markdown("**Back**: It scores the top 50 candidates from vector search using joint cross-attention to produce precision rankings for final output.")

st.divider()

# Section 13: Self-Assessment Checklist
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Vector Database Application Topologies before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand why semantic search resolves synonyms and intent that lexical search engines miss.", value=True, key="chk_vdb_imp_1")
    chk2 = st.checkbox("I can explain how vector databases power Two-Tower candidate generation in recommendation systems.", value=True, key="chk_vdb_imp_2")
    chk3 = st.checkbox("I understand how RAG pipelines prevent hallucinations and manage token budgets using vector retrieval.", value=True, key="chk_vdb_imp_3")
with c2:
    chk4 = st.checkbox("I know how multi-modal models like CLIP map text and images into a shared coordinate space.", value=True, key="chk_vdb_imp_4")
    chk5 = st.checkbox("I understand the difference between Cosine Similarity, Dot Product, and Euclidean L2 distance.", value=True, key="chk_vdb_imp_5")
    chk6 = st.checkbox("I can identify production traps such as dimension drift, cold cache latency, and uncalibrated score thresholds.", value=True, key="chk_vdb_imp_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Vector Database Applications Mastery Demonstrated!** You are ready to advance to Lesson 1.3 in the curriculum.")
