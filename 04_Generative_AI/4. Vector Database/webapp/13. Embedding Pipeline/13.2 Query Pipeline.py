import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="13.2 Query Pipeline - Vector DB Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("13.2 Query Pipeline: The 6-Stage Online Retrieval Lifecycle")
st.caption("Chapter 13: Embedding Pipeline | Domain: 4. Vector Database")
st.markdown("""
While the Ingestion Pipeline runs offline or asynchronously in batches, the **Query Pipeline operates synchronously in the hot path of user interaction**.

When a user submits a question to a real-time Generative AI or search application, the system has an aggressive 
latency budget—typically **under 20 milliseconds**—to complete a complex sequence of operations:
1. **Receive & Sanitize User Query**: Strip injection characters and normalize whitespace.
2. **Generate Query Embedding**: Execute a single forward inference pass using the identical embedding model.
3. **Execute HNSW Vector Search**: Traverse the multi-layer skip graph to discover nearest candidate vectors.
4. **Evaluate Metadata Filters**: Apply deterministic boolean predicates (`where={...}`) to prune unauthorized records.
5. **Calibrate Distances & Rank**: Sort candidates by geometric distance and compute confidence scores.
6. **Hydrate & Return Context**: Fetch raw document strings and provenance metadata from SQLite to build the LLM prompt.

In this lesson, we trace each step of the query lifecycle, dissect latency bottlenecks, examine asymmetric embedding model prefixes, 
and test an interactive Query Trace & Latency Breakdown Workbench.
""")

st.divider()

# Section 1: The Six-Stage Online Retrieval Lifecycle
st.header("1. The Six-Stage Online Retrieval Lifecycle", divider="rainbow")

col_life1, col_life2 = col_life1, col_life2 = st.columns([1.1, 0.9])

with col_life1:
    st.markdown("""
    ### The 6 Discrete Retrieval Stages
    
    * **Stage 1: Prompt Ingestion & Preprocessing**:
      * Cleans the user's natural language question. If using models like BGE or E5, prepends the required task instruction prefix (`"query: "`).
    * **Stage 2: Single-Vector Embedding Inference**:
      * Passes the query string to the embedding model to produce a normalized 1D float array (e.g., 1,536 dimensions).
    * **Stage 3: Graph Traversal (HNSW Beam Search)**:
      * Navigates down HNSW layers from `L_max` to `Layer 0` using dynamic queue depth `efSearch`.
    * **Stage 4: Metadata Predicate Evaluation**:
      * Verifies candidate IDs against user tenant claims and clearance levels.
    * **Stage 5: Distance Calculation & Sorting**:
      * Computes exact distance scores for final top-K candidates and sorts in ascending order of distance.
    * **Stage 6: Text Payload Hydration**:
      * Executes a fast indexed primary-key lookup in SQLite to fetch raw text chunks and provenance dictionaries for the top-K IDs.
    """)

with col_life2:
    st.markdown("### Online Query Lifecycle Sequence")
    seq_dot = """
    digraph QueryLifecycle {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        User [label="1. User Prompt Ingestion\\n'How to configure mTLS Envoy'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Embed [label="2. Query Vectorization\\n1 forward pass (~8ms)", fillcolor="#FEF3C7", color="#D97706"];

        HNSW [label="3. HNSW Beam Traversal\\nNavigate Layer Max -> Layer 0 (~2ms)", fillcolor="#D1FAE5", color="#059669"];

        Filter [label="4. Metadata & Security Filter\\nEnforce tenant + clearance (<0.5ms)", fillcolor="#FCE7F3", color="#DB2777"];

        Rank [label="5. Distance Calibration\\nCosine Sort & Confidence (<0.2ms)", fillcolor="#E0E7FF", color="#4338CA"];

        Hydrate [label="6. Hydrate Document Text\\nSQLite PK Fetch (~1ms)", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        ContextOut [label="Final Grounded Context\\nInjected into LLM Prompt", fillcolor="#A7F3D0", style="filled,bold"];

        User -> Embed;
        Embed -> HNSW;
        HNSW -> Filter;
        Filter -> Rank;
        Rank -> Hydrate;
        Hydrate -> ContextOut;
    }
    """
    st.graphviz_chart(seq_dot, use_container_width=False)

st.divider()

# Section 2: The Latency Budget Breakdown in Production
st.header("2. The Latency Budget Breakdown in Production", divider="rainbow")

st.markdown("""
Where does time actually go during a vector query? Understanding the latency profile helps engineers identify 
where to spend optimization effort:
""")

LATENCY_BUDGET = [
    {
        "Lifecycle Stage": "1. Prompt Ingestion & Validation",
        "Typical Duration": "0.1 - 0.3 ms",
        "Primary Bottleneck": "Regex sanitization & string allocation",
        "Optimization Strategy": "Lightweight pure Python checks"
    },
    {
        "Lifecycle Stage": "2. Query Vector Generation",
        "Typical Duration": "5.0 - 25.0 ms",
        "Primary Bottleneck": "Transformer forward pass or network API round-trip",
        "Optimization Strategy": "Semantic caching & local GPU quantization"
    },
    {
        "Lifecycle Stage": "3. HNSW Graph Navigation",
        "Typical Duration": "1.0 - 3.5 ms",
        "Primary Bottleneck": "CPU L3 cache misses & floating-point SIMD ops",
        "Optimization Strategy": "Tune efSearch & enforce AVX-512 vectorization"
    },
    {
        "Lifecycle Stage": "4. Metadata Filtering",
        "Typical Duration": "0.2 - 1.0 ms",
        "Primary Bottleneck": "Single-stage graph edge checking / B-Tree masks",
        "Optimization Strategy": "Pre-index selective metadata fields"
    },
    {
        "Lifecycle Stage": "5. Distance Calibration & Sorting",
        "Typical Duration": "0.1 - 0.4 ms",
        "Primary Bottleneck": "Sorting priority queue of size K",
        "Optimization Strategy": "Keep K small (5 - 20)"
    },
    {
        "Lifecycle Stage": "6. SQLite Text Payload Hydration",
        "Typical Duration": "0.5 - 2.0 ms",
        "Primary Bottleneck": "Disk / Page cache random read of text strings",
        "Optimization Strategy": "Ensure SQLite page cache fits in RAM"
    }
]

st.dataframe(pd.DataFrame(LATENCY_BUDGET), use_container_width=True, hide_index=True)

st.info("""
💡 **Key Architectural Insight**: Over **75% to 85% of total query latency** is spent in **Stage 2 (Generating the query embedding)**! 
The actual vector database search inside HNSW (Stage 3) typically completes in under 2.5 milliseconds.
""")

st.divider()

# Section 3: The Asymmetric Model Prefix Trap
st.header("3. The Asymmetric Model Prefix Trap", divider="rainbow")

col_asym1, col_asym2 = col_asym1, col_asym2 = st.columns([1, 1])

with col_asym1:
    st.markdown("""
    ### Asymmetric Retrieval Models (BGE, E5, INSTRUCTOR)
    Modern state-of-the-art embedding models are trained **asymmetrically**:
    * **Document Chunks** represent authoritative reference passages.
    * **User Queries** represent short, interrogative inquiries.
    
    Because queries and documents serve different linguistic roles, models like **BAAI/bge-large-en-v1.5** and **intfloat/e5-large-v2** require adding a **specific prefix to queries, but NOT to documents**:
    
    * **At Ingestion Time**:
      ```python
      doc_text = "Kubernetes routes ingress via Envoy."
      doc_emb = model.encode(doc_text)  # NO prefix
      ```
    * **At Query Time**:
      ```python
      query_text = "How does Kubernetes route ingress?"
      # REQUIRED PREFIX FOR BGE:
      query_with_prefix = f"Represent this sentence for searching relevant passages: {query_text}"
      query_emb = model.encode(query_with_prefix)
      ```
    """)

with col_asym2:
    st.warning("""
    ### 🚨 The Devastating Impact of Missing Prefixes
    If you forget the instruction prefix at query time:
    1. The query vector lands in the **symmetric sentence similarity subspace** rather than the **retrieval search subspace**.
    2. Cosine similarity between your question and the exact ground-truth passage drops from **0.88 down to 0.42**.
    3. Retrieval recall collapses by up to **40%**, causing the RAG assistant to return completely unrelated chunks.
    
    * **ChromaDB Solution**: Use Chroma's built-in `EmbeddingFunction` wrappers, which automatically handle prefix injection.
    """)

st.divider()

# Section 4: Interactive Workbench: Live 6-Stage Query Trace & Latency Profiler
st.header("4. Interactive Workbench: Live 6-Stage Query Trace & Latency Profiler", divider="rainbow")

st.markdown("""
Execute a real-time query simulation and inspect the precise execution timing, intermediate data representations, 
and output payload at every step of the 6-stage lifecycle.
""")

col_sim_q, col_sim_cfg = st.columns([1.2, 0.8])

with col_sim_q:
    st.subheader("1. Enter Search Query")
    sample_queries = [
        "How do I configure mTLS mutual authentication in Envoy proxy?",
        "What are our database backup retention policies under GDPR?",
        "What is the maximum token limit for OpenAI text-embedding-3?"
    ]
    user_query = st.selectbox("Select or Type Query:", sample_queries)
    custom_query = st.text_input("Or Enter Custom Query:", value="")
    final_query = custom_query.strip() if custom_query.strip() else user_query

with col_sim_cfg:
    st.subheader("2. Query Execution Parameters")
    req_top_k = st.slider("Requested Top-K Results:", min_value=1, max_value=10, value=3)
    sim_ef_search = st.slider("HNSW efSearch Depth:", min_value=16, max_value=200, value=64, step=16)
    enforce_filter = st.checkbox("Enforce Department Filter (`dept == 'Security'`)", value=True)

# Simulation Execution
st.divider()

# Mock latencies
t_clean = round(np.random.uniform(0.12, 0.25), 2)
t_embed = round(np.random.uniform(7.5, 12.0), 2)
t_hnsw = round(np.random.uniform(1.2, 2.4) * (sim_ef_search / 64.0), 2)
t_filter = round(np.random.uniform(0.3, 0.6), 2) if enforce_filter else 0.05
t_rank = round(np.random.uniform(0.1, 0.3), 2)
t_hydrate = round(np.random.uniform(0.6, 1.4), 2)
t_total = round(t_clean + t_embed + t_hnsw + t_filter + t_rank + t_hydrate, 2)

col_out_trace, col_out_results = st.columns([1.1, 0.9])

with col_out_trace:
    st.subheader(f"3. Latency Waterfall Trace (Total: {t_total:.2f} ms)")
    
    trace_data = [
        {"Stage": "1. Sanitization & Preprocessing", "Latency (ms)": t_clean, "Pct": f"{(t_clean/t_total)*100:.1f}%"},
        {"Stage": "2. Query Vector Inference", "Latency (ms)": t_embed, "Pct": f"{(t_embed/t_total)*100:.1f}%"},
        {"Stage": "3. HNSW Beam Traversal", "Latency (ms)": t_hnsw, "Pct": f"{(t_hnsw/t_total)*100:.1f}%"},
        {"Stage": "4. Metadata Filtering", "Latency (ms)": t_filter, "Pct": f"{(t_filter/t_total)*100:.1f}%"},
        {"Stage": "5. Distance Ranking", "Latency (ms)": t_rank, "Pct": f"{(t_rank/t_total)*100:.1f}%"},
        {"Stage": "6. Text Payload Hydration", "Latency (ms)": t_hydrate, "Pct": f"{(t_hydrate/t_total)*100:.1f}%"}
    ]
    df_trace = pd.DataFrame(trace_data)
    
    st.dataframe(
        df_trace.style.background_gradient(subset=["Latency (ms)"], cmap="Oranges"),
        use_container_width=True,
        hide_index=True
    )
    
    st.progress(min(1.0, t_total / 25.0))
    st.caption(f"SLA Target: < 25.0 ms | Achieved: **{t_total} ms** (Status: **PASS**)")

with col_out_results:
    st.subheader("4. Retrieved Context Payload")
    
    mock_retrieved = [
        {
            "id": "SEC-ENV-01",
            "distance": 0.0841,
            "confidence": "91.6%",
            "dept": "Security",
            "doc": "Envoy proxy mTLS configuration requires mounting client certificates in /etc/envoy/certs and enforcing require_client_certificate: true in downstream listener filter chains."
        },
        {
            "id": "SEC-ENV-02",
            "distance": 0.1420,
            "confidence": "85.8%",
            "dept": "Security",
            "doc": "Automated certificate rotation for Envoy listeners uses SPIFFE/SPIRE agents communicating over local UNIX domain sockets."
        },
        {
            "id": "SEC-NET-03",
            "distance": 0.1895,
            "confidence": "81.1%",
            "dept": "Security",
            "doc": "Zero-trust ingress architectures terminate external TLS at edge Cloudflare load balancers before re-encrypting via internal mTLS."
        }
    ][:req_top_k]
    
    for rank_idx, item in enumerate(mock_retrieved):
        with st.expander(f"🏅 Rank {rank_idx + 1}: {item['id']} (Distance: {item['distance']} | {item['confidence']})", expanded=True):
            st.markdown(f"**Content**: *\"{item['doc']}\"*")
            st.caption(f"Metadata: `dept: {item['dept']}` | Grounding Status: **Verified**")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the primary cause of latency in a production vector database query pipeline?"):
    st.markdown("""
    **Answer**:
    In 90% of production RAG architectures, **Query Embedding Generation (Stage 2)** dominates total query latency:
    * Passing the query text through an embedding model (e.g. OpenAI `text-embedding-3` over HTTPS or a local BERT model on CPU) typically takes **8ms to 35ms**.
    * In contrast, the actual vector index search inside HNSW (Stage 3) typically completes in **1ms to 3ms**.
    * Therefore, optimizing vector database index parameters (`M`, `efSearch`) will only shave 1-2ms, whereas implementing **Semantic Caching** on the query embedding can save 20-30ms!
    """)

with st.expander("Q2: What is Semantic Caching and how does it optimize the query pipeline?"):
    st.markdown("""
    **Answer**:
    Traditional HTTP/Redis caches require exact string matches (`cache_key = hash(query_string)`). If a user asks *"How do I reset my password?"* and another asks *"How can I reset my account password?"*, an exact cache misses.
    
    **Semantic Caching (e.g., GPTCache / Redis Vector Cache)**:
    1. Embeds incoming query `Q`.
    2. Searches a dedicated small in-memory cache vector index of previously answered queries.
    3. If a cached query exists with **Cosine Similarity > 0.96**, returns the cached result immediately!
    4. **Impact**: Bypasses the main vector database and LLM generation entirely, dropping response latency from 800ms down to **2ms** and cutting API costs by 40-70%.
    """)

with st.expander("Q3: What is the danger of using different embedding models at query time versus ingestion time?"):
    st.markdown("""
    **Answer**:
    Different embedding models (e.g. `text-embedding-ada-002` vs `text-embedding-3-small`, or BGE vs MiniLM) map text into **entirely different, incompatible mathematical vector spaces**:
    1. Dimensions may not match (1536 vs 384 dims), triggering immediate database exceptions.
    2. Even if dimensions match, the coordinate axes and manifold orientations are completely uncorrelated.
    3. Querying model B's vector against model A's database produces random noise, resulting in near-zero recall.
    **Rule**: The query embedding model must be **100% identical in weights, version, and tokenizer** to the ingestion model.
    """)

with st.expander("Q4: Why does ChromaDB separate vector graph traversal from document text hydration?"):
    st.markdown("""
    **Answer**:
    * **Memory Footprint**: Keeping megabytes of raw text strings resident in high-speed RAM alongside HNSW graph edges would bloat process memory by 10x, causing Out-Of-Memory crashes.
    * **Cache Locality**: HNSW graph traversal only needs vector coordinates and integer IDs to compute distances and navigate pointers.
    * **Lazy Hydration**: Only after the algorithm identifies the final top-K nearest IDs (e.g. top 5 out of 1,000,000) does it query SQLite to fetch the raw text strings for those 5 IDs. This keeps the HNSW graph compact and cache-friendly.
    """)

with st.expander("Q5: How do you handle cold-start latency spikes on the first query after a vector database reboot?"):
    st.markdown("""
    **Answer**:
    After a reboot, operating system memory-mapped pages and database caches are cold; vectors must be read from disk into the page cache, spiking latency from 2ms to 200ms+.
    
    **Production Warm-Up Protocol**:
    1. During container startup (e.g. in Kubernetes `readinessProbe` handler), execute a synthetic warm-up script.
    2. The script runs 50 representative vector queries across the collection with `efSearch=128`.
    3. This forces the OS to fault the HNSW upper graph layers and frequently accessed vectors into physical RAM page cache before opening the container to live user traffic.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Forgetting Asymmetric Query Instruction Prefixes**  
    Models like BGE require `query: ` or task instructions for questions. Omitting this prefix drops retrieval recall by up to 40% because the model treats the query as a reference document.
    """)
    st.warning("""
    **Trap 2: Requesting Unnecessarily Large Top-K**  
    Asking for `n_results = 100` when the downstream RAG prompt only uses 3 chunks increases HNSW traversal time, SQLite hydration disk reads, and network serialization overhead. Set `n_results` to exactly what you need.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Model Version Drift Between Ingestion & Search**  
    Upgrading the embedding model in your backend API without re-indexing existing documents in the database mixes two incompatible vector spaces, resulting in complete search breakdown.
    """)
    st.warning("""
    **Trap 4: Synchronous Blocking on Cloud Embedding APIs**  
    Calling third-party embedding APIs synchronously without a connection timeout can hang user requests for 30+ seconds during cloud outages. Always enforce strict timeouts (e.g. 2.0s) with fallback mechanisms.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What takes the most time in a vector query?"):
        st.markdown("""
        **Bottleneck**:  
        Generating the query embedding (75-85% of total latency); HNSW search itself takes <2.5ms.
        """)
    with st.expander("🗂️ Flashcard: What is Semantic Caching?"):
        st.markdown("""
        **Semantic Cache**:  
        Caches previous query vectors; returns cached results when a new query has Cosine Similarity > 0.96.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is lazy text hydration?"):
        st.markdown("""
        **Lazy Hydration**:  
        HNSW searches using vectors and IDs only; raw text is fetched from SQLite only for the final top-K items.
        """)
    with st.expander("🗂️ Flashcard: Why warm up a vector database?"):
        st.markdown("""
        **Warm-Up**:  
        Populates OS page cache with hot HNSW graph pages to prevent 200ms cold-start page fault spikes.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know all 6 stages of the online vector query lifecycle.", value=True)
chk2 = st.checkbox("I understand the production latency budget and know that query vectorization is the primary bottleneck.", value=True)
chk3 = st.checkbox("I understand asymmetric embedding model prefixes (e.g. BGE/E5) and why missing them breaks recall.", value=True)
chk4 = st.checkbox("I know how Semantic Caching bypasses both the vector database and LLM generation.", value=True)
chk5 = st.checkbox("I understand why lazy document hydration keeps the HNSW graph compact in RAM.", value=True)
chk6 = st.checkbox("I can design and profile an end-to-end vector search pipeline with sub-20ms p95 latency SLAs.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Query Pipeline and are ready for Chapter 13.3: Embedding Optimization!")
