import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="8.2 Query vs Get - ChromaDB Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("8.2 Query vs Get: Comparative Architecture, Routing Decision Framework & Strategies")
st.caption("Chapter 8: Getting and Retrieving Data | Domain: 5. ChromaDB")
st.markdown("""
ChromaDB is a **hybrid dual-engine system**: it contains an embedded relational database (SQLite) 
paired with an embedded approximate nearest-neighbor vector graph (HNSW via C++ `hnswlib`).

Engineers frequently ask:
* *When should my application execute `collection.get()` versus `collection.query()`?*
* *Can we combine both primitives to build multi-stage hierarchical retrieval systems?*
* *What are the computational cost, latency, and memory implications of each approach?*

Choosing the wrong retrieval primitive degrades search latency and increases token costs:
* Running `query()` for an exact document lookup incurs unnecessary vector model inference and graph traversals.
* Running `get()` for conceptual or thematic searches fails completely because keyword matches lack semantic awareness.

In this lesson, we perform a deep architectural comparison of `get()` versus `query()`, 
construct a production Query Routing Decision Framework, implement hierarchical Parent-Child retrieval in Python, 
and test an interactive Live Architecture Profiler and Retrieval Router.
""")

st.divider()

# Section 1: Side-by-Side Architectural Comparison
st.header("1. Side-by-Side Architectural Comparison", divider="rainbow")

col_comp_tbl, col_comp_diag = col_comp_tbl, col_comp_diag = st.columns([1.1, 0.9])

with col_comp_tbl:
    st.markdown("### Engine Mechanics Matrix")
    
    COMPARISON_MATRIX = [
        {
            "Feature / Metric": "Underlying Subsystem",
            "collection.get()": "SQLite Relational Database (Disk/Memory)",
            "collection.query()": "hnswlib C++ Navigable Graph + SQLite"
        },
        {
            "Feature / Metric": "Search Mechanism",
            "collection.get()": "Deterministic B-Tree Primary Key / Column Scan",
            "collection.query()": "Approximate Nearest Neighbor (ANN) Graph Traversal"
        },
        {
            "Feature / Metric": "Mathematical Overhead",
            "collection.get()": "Zero Vector Math (No dot products)",
            "collection.query()": "High (Vector embedding inference + SIMD Cosine)"
        },
        {
            "Feature / Metric": "Typical Latency",
            "collection.get()": "Sub-millisecond (0.1ms – 0.8ms)",
            "collection.query()": "2ms – 15ms (Excluding embedding API latency)"
        },
        {
            "Feature / Metric": "Return Data Structure",
            "collection.get()": "1D Flat Parallel Arrays (`[idx]`)",
            "collection.query()": "2D Nested Arrays (`[query_idx][rank_idx]`)"
        },
        {
            "Feature / Metric": "Returns 'distances'?",
            "collection.get()": "Never (Geometric distances do not exist)",
            "collection.query()": "Always (Sorted by ascending distance)"
        }
    ]
    st.dataframe(pd.DataFrame(COMPARISON_MATRIX), use_container_width=True, hide_index=True)

with col_comp_diag:
    st.markdown("### Dual-Engine Routing Topology")
    dual_dot = """
    digraph DualEngine {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InboundReq [label="Inbound Application Request\\nRoute by Retrieval Intent", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Router [label="Query Intent Router\\nExact ID / Filter vs Semantic Meaning", shape=diamond, fillcolor="#FEF3C7"];

        GetEngine [label="collection.get()\\nSQLite Primary Key & Column Index\\nLatency: ~0.3ms (Zero SIMD)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        QueryEngine [label="collection.query()\\nHNSW Vector Graph Traversal\\nLatency: ~5ms (Full Vector Math)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        InboundReq -> Router;
        Router -> GetEngine [label="Known IDs / Filter Only"];
        Router -> QueryEngine [label="Natural Language Meaning"];
    }
    """
    st.graphviz_chart(dual_dot, use_container_width=False)

st.divider()

# Section 2: Production Decision Framework & Retrieval Strategies
st.header("2. Production Decision Framework & Retrieval Strategies", divider="rainbow")

col_strat1, col_strat2 = col_strat1, col_strat2 = st.columns([1, 1])

with col_strat1:
    st.markdown("""
    ### When to Use `collection.get()`
    * **1. Direct Document Hydration by ID**:
      * You already know the exact IDs from a cache or secondary index.
      * Example: `collection.get(ids=["doc_101", "doc_102"])`.
    * **2. Relational Metadata Auditing**:
      * Listing all documents created by an author or belonging to a tenant.
      * Example: `collection.get(where={"tenant_id": "acme_corp"}, limit=100)`.
    * **3. Full-Collection ETL Backups**:
      * Iterating through the entire dataset with `limit` and `offset`.
    * **4. Parent Document Retrieval (Hierarchical RAG)**:
      * Fetching the full parent article after finding a small child chunk!
    """)

with col_strat2:
    st.markdown("""
    ### When to Use `collection.query()`
    * **1. Natural Language Question Answering**:
      * The user asks a conceptual question where exact keywords may not match.
      * Example: *"How do we troubleshoot high database latency?"*
    * **2. Semantic Recommendation & Deduplication**:
      * Finding documents that discuss similar ideas to a candidate article.
    * **3. Multi-Query Hybrid Expansion**:
      * Running multiple query variants simultaneously to maximize recall.
    * **4. Cross-Lingual & Multimodal Search**:
      * Searching English runbooks using a Spanish query via multilingual vectors.
    """)

st.divider()

# Section 3: Clean Python Implementation of Hierarchical Parent-Child Retrieval
st.header("3. Clean Python Implementation of Hierarchical Parent-Child Retrieval", divider="rainbow")

st.markdown("""
The most powerful enterprise retrieval pattern combines both `query()` and `get()`: 
**Parent-Child Hierarchical Retrieval**. 
We search with small 200-character chunks using `query()` for high semantic precision, 
and then use `get()` to hydrate the entire 2,000-character parent document for comprehensive LLM context!
""")

st.code("""import chromadb
from typing import List, Dict, Any

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("docs_kb")

def parent_child_hierarchical_retrieval(
    collection,
    user_query: str,
    top_k_chunks: int = 3
) -> List[Dict[str, Any]]:
    \"\"\"Combines query() for chunk search and get() for full parent hydration.\"\"\"
    
    # Stage 1: Fine-Grained Semantic Search on Child Chunks via query()
    chunk_results = collection.query(
        query_texts=[user_query],
        n_results=top_k_chunks,
        include=["metadatas", "distances"]
    )
    
    # Stage 2: Extract Parent Document IDs from child chunk metadata
    parent_ids = list(set([
        meta["parent_doc_id"]
        for meta in chunk_results["metadatas"][0]
        if "parent_doc_id" in meta
    ]))
    
    if not parent_ids:
        return []
        
    # Stage 3: Exact Relational Hydration of Full Parent Documents via get()
    parent_results = collection.get(
        ids=parent_ids,
        include=["documents", "metadatas"]
    )
    
    # Stage 4: Package full context for LLM prompt assembly
    hydrated_parents = []
    for p_id, p_text, p_meta in zip(
        parent_results["ids"],
        parent_results["documents"],
        parent_results["metadatas"]
    ):
        hydrated_parents.append({
            "parent_id": p_id,
            "full_document": p_text,
            "metadata": p_meta
        })
        
    return hydrated_parents
""", language="python")

st.divider()

# Section 4: Interactive Workbench: Live Architecture Profiler & Router
st.header("4. Interactive Workbench: Live Architecture Profiler & Router", divider="rainbow")

st.markdown("""
Enter an incoming user request to observe how an intelligent query router classifies the intent, 
directs the operation to `get()` or `query()`, and profiles simulated execution performance.
""")

col_wb_in, col_wb_cfg = col_wb_in, col_wb_cfg = st.columns([1.1, 0.9])

with col_wb_in:
    st.subheader("1. Inbound Request Simulator")
    sample_preset = st.selectbox(
        "Choose Preset Request Scenario:",
        [
            "Scenario A: 'Find document with ID doc_sec_409'",
            "Scenario B: 'How do we resolve Kubernetes CrashLoopBackOff errors?'",
            "Scenario C: 'List all runbooks authored by SRE team in 2024'",
            "Scenario D: 'What is our disaster recovery RTO policy?'"
        ]
    )

with col_wb_cfg:
    st.subheader("2. Routing Logic Inspector")
    is_exact_lookup = "ID" in sample_preset or "authored by" in sample_preset
    
    if is_exact_lookup:
        target_op = "collection.get()"
        engine_target = "SQLite B-Tree Index"
        math_eval = "Zero SIMD Math (No vectorization)"
        sim_lat = "0.25 ms"
        card_color = "#D1FAE5"
    else:
        target_op = "collection.query()"
        engine_target = "hnswlib HNSW Vector Graph"
        math_eval = "Vector Model Inference + Cosine Distance"
        sim_lat = "6.40 ms"
        card_color = "#EDE9FE"

st.divider()

col_route_res1, col_route_res2 = col_route_res1, col_route_res2 = st.columns([1, 1])

with col_route_res1:
    st.subheader("Routing Decision Scorecard")
    st.markdown(f"""
    * **Classified Intent**: `{'Deterministic Exact Lookup' if is_exact_lookup else 'Conceptual Semantic Search'}`
    * **Recommended Method**: **`{target_op}`**
    * **Target Storage Engine**: `{engine_target}`
    * **Vector Math Overhead**: `{math_eval}`
    * **Estimated Execution Latency**: `{sim_lat}`
    """)

with col_route_res2:
    st.subheader("Generated Python Code Snippet")
    if is_exact_lookup:
        st.code("""# High-Speed Deterministic Fetch (<1ms)
results = collection.get(
    ids=["doc_sec_409"],
    include=["documents", "metadatas"]
)
doc_text = results["documents"][0]
""", language="python")
        st.success("🟢 **Bypassed Vector Graph**: Saved 95% latency by routing through SQLite B-Tree!")
    else:
        st.code("""# High-Recall Semantic Vector Search (~6ms)
results = collection.query(
    query_texts=["Kubernetes CrashLoopBackOff errors"],
    n_results=3,
    include=["documents", "distances"]
)
doc_text = results["documents"][0][0]
""", language="python")
        st.info("ℹ️ **Traversed HNSW Graph**: Retrieved conceptually relevant documents without requiring exact keyword matches.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What are the three primary differences between `get()` and `query()` in ChromaDB?"):
    st.markdown("""
    **Answer**:
    1. **Target Subsystem**: `get()` queries SQLite relational B-Tree tables directly; `query()` traverses the HNSW vector graph (`hnswlib`).
    2. **Execution Cost**: `get()` requires zero vector embedding math and executes in ~0.2ms; `query()` requires computing query embeddings and evaluating distance metrics in vector space (~5–10ms).
    3. **Output Structure**: `get()` returns flat 1D lists without distances; `query()` returns nested 2D lists with geometric distances.
    """)

with st.expander("Q2: Why is Parent-Child Retrieval superior to standard naive chunk retrieval?"):
    st.markdown("""
    **Answer**:
    * **The Chunk Sizing Paradox**:
      * Small chunks (200 tokens) are ideal for **vector retrieval** because embeddings maintain sharp, focused semantic meaning.
      * Large documents (2,000 tokens) are ideal for **LLM generation** because models need comprehensive context to reason without hallucinating.
    * **Parent-Child Solution**:
      * Index small child chunks with `query()` to achieve high semantic precision.
      * Store the parent document ID in the child chunk's metadata.
      * When a child chunk matches, use `collection.get(ids=[parent_id])` to retrieve the complete parent document for the LLM prompt.
    """)

with st.expander("Q3: Can `collection.query()` be called with only metadata filters and no search text or embeddings?"):
    st.markdown("""
    **Answer**:
    * **No**. In ChromaDB, `collection.query()` requires either `query_texts` or `query_embeddings`.
    * If you want to retrieve documents purely based on metadata criteria without vector similarity, you must use **`collection.get(where={...})`**.
    """)

with st.expander("Q4: What happens if an application uses `collection.query()` for ID lookups?"):
    st.markdown("""
    **Answer**:
    * ChromaDB does not support searching by ID inside `collection.query()`.
    * Attempting to pass `where={"id": "doc_101"}` in `query()` will fail because the primary document ID is stored in a separate internal column, not within the user metadata dictionary.
    * Use `collection.get(ids=["doc_101"])` for all ID lookups.
    """)

with st.expander("Q5: How do `limit` and `offset` differ between `get()` and `query()`?"):
    st.markdown("""
    **Answer**:
    * `collection.get()` supports both **`limit` and `offset`** for standard SQL-like pagination across the collection.
    * `collection.query()` supports **`n_results`** (which functions like `limit`), but does **NOT support `offset`**. You cannot ask `query()` to return candidates ranked 11 through 20 directly; you must fetch `n_results=20` and slice `[10:20]` client-side.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Using `query()` with Exact Keywords Expecting Exact Match**  
    Passing `"SKU-99214"` into `query_texts` computes vector similarity, which may rank completely unrelated SKUs higher! Use `get()` with exact filters.
    """)
    st.warning("""
    **Trap 2: Attempting to Paginate `query()` with `offset`**  
    Passing `offset` to `collection.query()` is not supported. Fetch `n_results=K` and slice the resulting list in Python.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Slicing `get()` Results with 2D `[0][0]` Coordinates**  
    `get()` returns a 1D flat list! Writing `results['documents'][0][0]` extracts the first letter of the text rather than the document string.
    """)
    st.warning("""
    **Trap 4: Missing Parent Chunk Hydration in Technical Documentation RAG**  
    Feeding tiny isolated 150-word chunks into an LLM lacks surrounding context. Use Parent-Child retrieval via `get()` to provide full runbook context.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: When to use get()?"):
        st.markdown("""
        **Exact Lookups**:  
        When querying by ID, metadata attributes, or paginating full collections.
        """)
    with st.expander("🗂️ Flashcard: When to use query()?"):
        st.markdown("""
        **Semantic Meaning**:  
        When searching natural language concepts or finding similar documents.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Parent-Child Retrieval?"):
        st.markdown("""
        **Two-Stage Search**:  
        Search small chunks with `query()`, then hydrate full parent doc with `get()`.
        """)
    with st.expander("🗂️ Flashcard: Does query() support offset?"):
        st.markdown("""
        **No**:  
        `query()` only supports `n_results`. Slice lists client-side for pagination.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural differences between `get()` (SQLite) and `query()` (HNSW).", value=True)
chk2 = st.checkbox("I know when to route requests to `get()` for sub-millisecond deterministic lookups.", value=True)
chk3 = st.checkbox("I understand why `query()` is required for semantic conceptual questions.", value=True)
chk4 = st.checkbox("I can implement a Parent-Child Hierarchical Retrieval pipeline combining `query()` and `get()`.", value=True)
chk5 = st.checkbox("I understand why `get()` returns 1D lists while `query()` returns 2D nested lists.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 9: Metadata Management in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completely mastered Chapter 8: Getting and Retrieving Data in ChromaDB!")
