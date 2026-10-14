import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="7.2 Query Results - ChromaDB Curriculum",
    page_icon="📑",
    layout="wide"
)

# Title & Overview
st.title("7.2 Query Results: Returned Documents, IDs, Metadata, Distances & 2D Result Unpacking")
st.caption("Chapter 7: Querying ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
When you execute a query in ChromaDB (`collection.query(...)`), the database returns a **`QueryResult` dictionary**. 

Unlike standard SQL databases where results return as a flat list of rows (`List[Dict[str, Any]]`), 
ChromaDB returns a **dictionary of parallel 2-Dimensional lists** organized by query batch index and rank index.

Understanding this 2D structure is essential for writing bug-free retrieval and RAG pipelines:
* **`ids: List[List[str]]`**: Document string identifiers ordered by ascending distance.
* **`documents: List[List[str]]`**: Raw text strings retrieved from SQLite storage.
* **`metadatas: List[List[Dict[str, Any]]]`**: Key-value metadata dictionaries attached to each document.
* **`distances: List[List[float]]`**: Geometric distances between query vector and retrieved documents.
* **`include` Filtering**: Selectively hydrating only the fields needed to minimize serialization overhead and memory footprint.

In this lesson, we dissect the anatomy of the 2D results dictionary, write clean typed unpacking patterns 
in Python, handle missing/excluded fields gracefully, and test an interactive 2D Query Result Unpacker and Visualizer.
""")

st.divider()

# Section 1: Anatomy of the QueryResult Dictionary
st.header("1. Anatomy of the QueryResult Dictionary", divider="rainbow")

col_anat1, col_anat2 = col_anat1, col_anat2 = st.columns([1.1, 0.9])

with col_anat1:
    st.markdown("""
    ### The 2D Parallel Matrix Representation
    
    * In a batch query with **$Q$ queries** and **$K$ results per query (`n_results=K`)**:
      * Every key in the `results` dictionary maps to a nested list of dimensions `(Q, K)`.
      * **Dimension 1 (`i = 0 ... Q-1`)**: Corresponds to which search query in the batch was executed.
      * **Dimension 2 (`j = 0 ... K-1`)**: Corresponds to the ranked search hit for that specific query (Rank 1, Rank 2, ...).
    
    * **Parallel Alignment Invariant**:
      * For any query index `i` and rank index `j`:
        `results["ids"][i][j]` maps directly to:
        `results["documents"][i][j]`,
        `results["metadatas"][i][j]`, and
        `results["distances"][i][j]`.
    """)

with col_anat2:
    st.markdown("### 2D Parallel Payload Matrix")
    matrix_dot = """
    digraph ResultMatrix {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        DictRoot [label="QueryResult Dictionary\\n{'ids': [...], 'documents': [...], ...}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Q0 [label="Query #0 Batch Slices (i=0)\\nTop-K Hits for Query 0", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        Q1 [label="Query #1 Batch Slices (i=1)\\nTop-K Hits for Query 1", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        Hit0_0 [label="Rank 1 (j=0): [id_101]\\nDoc: 'Postgres WAL...'\\nDist: 0.082", fillcolor="#D1FAE5", color="#059669"];

        Hit0_1 [label="Rank 2 (j=1): [id_102]\\nDoc: 'MySQL Binlog...'\\nDist: 0.145", fillcolor="#A7F3D0", color="#059669"];

        DictRoot -> Q0;
        DictRoot -> Q1;
        Q0 -> Hit0_0;
        Q0 -> Hit0_1;
    }
    """
    st.graphviz_chart(matrix_dot, use_container_width=False)

st.divider()

# Section 2: Controlling Payloads with the `include` Parameter
st.header("2. Controlling Payloads with the `include` Parameter", divider="rainbow")

col_inc1, col_inc2 = col_inc1, col_inc2 = st.columns([1, 1])

with col_inc1:
    st.markdown("""
    ### Default vs Explicit Payloads
    * By default, ChromaDB hydrates and returns:
      `include=["documents", "metadatas", "distances"]`.
    * Document text is retrieved from SQLite. If documents are large (e.g. 5 KB each) and $K=20$, hydrating text consumes significant CPU time and JSON serialization memory.
    
    ### Available `include` Tokens:
    * `"documents"`: Hydrate raw text chunk strings.
    * `"metadatas"`: Hydrate dictionary metadata.
    * `"distances"`: Include computed geometric distances.
    * `"embeddings"`: Hydrate stored high-dimensional float arrays (Warning: heavy!).
    * `"uris"`: Hydrate external file URI strings.
    * `"data"`: Hydrate binary data (images/audio).
    """)

with col_inc2:
    st.markdown("### Production Payload Optimization")
    st.code("""# High-Performance Search: Return only IDs and Distances!
# Skips reading raw document strings from SQLite.
lightweight_results = collection.query(
    query_texts=["Search query"],
    n_results=10,
    include=["distances"]  # documents and metadatas will be None!
)

# results['documents'] is None
# results['metadatas'] is None
# results['ids'] is [["doc_1", "doc_2", ...]]
# results['distances'] is [[0.12, 0.25, ...]]
""", language="python")

st.divider()

# Section 3: Clean Python Implementation of Result Unpacker & Pydantic Modeler
st.header("3. Clean Python Implementation of Result Unpacker & Pydantic Modeler", divider="rainbow")

st.code("""import chromadb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SearchHit:
    id: str
    document: Optional[str]
    metadata: Optional[Dict[str, Any]]
    distance: float
    similarity_pct: float

def unpack_chroma_results(results: Dict[str, Any], query_idx: int = 0) -> List[SearchHit]:
    \"\"\"Converts raw 2D ChromaDB query results into a clean list of typed SearchHit objects.\"\"\"
    hits: List[SearchHit] = []
    
    ids_list = results.get("ids", [[]])[query_idx]
    docs_list = results.get("documents", [[]])[query_idx] if results.get("documents") else [None] * len(ids_list)
    metas_list = results.get("metadatas", [[]])[query_idx] if results.get("metadatas") else [None] * len(ids_list)
    dists_list = results.get("distances", [[]])[query_idx] if results.get("distances") else [0.0] * len(ids_list)
    
    for doc_id, text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
        hits.append(SearchHit(
            id=doc_id,
            document=text,
            metadata=meta,
            distance=dist,
            similarity_pct=round(max(0.0, 1.0 - dist) * 100, 2)
        ))
        
    return hits

# Example Usage:
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("docs_kb")
raw_res = collection.query(query_texts=["PostgreSQL"], n_results=2)

clean_hits = unpack_chroma_results(raw_res, query_idx=0)
for h in clean_hits:
    print(f"[{h.id}] Match: {h.similarity_pct}% (Dist: {h.distance:.4f}) -> {h.document}")
""", language="python")

st.divider()

# Section 4: Interactive Workbench: 2D Result Unpacker & Payload Visualizer
st.header("4. Interactive Workbench: 2D Result Unpacker & Payload Visualizer", divider="rainbow")

st.markdown("""
Inspect how a multi-query search result dictionary is structured and unpacked. 
Switch between the raw 2D JSON payload and the clean, flattened tabular representation.
""")

col_wb_ctrl1, col_wb_ctrl2 = col_wb_ctrl1, col_wb_ctrl2 = st.columns(2)

with col_wb_ctrl1:
    st.subheader("1. Inbound Query Batch Selection")
    active_q_idx = st.radio("Select Query to Inspect:", ["Query #0: 'Database recovery'", "Query #1: 'Kubernetes ingress'"], index=0)
    q_index_num = 0 if "Query #0" in active_q_idx else 1

with col_wb_ctrl2:
    st.subheader("2. Payload Hydration Simulation (`include`)")
    include_docs = st.checkbox("Include Documents (`'documents'`)", value=True)
    include_metas = st.checkbox("Include Metadatas (`'metadatas'`)", value=True)
    include_dists = st.checkbox("Include Distances (`'distances'`)", value=True)

# Generate mock 2D results dictionary
MOCK_RAW_RESULTS = {
    "ids": [
        ["doc_db_01", "doc_db_02"],
        ["doc_k8s_01", "doc_k8s_02"]
    ],
    "documents": [
        ["PostgreSQL WAL continuous streaming to encrypted S3.", "MySQL automated snapshot lifecycle dumps."],
        ["Kubernetes Ingress routes traffic to cluster services.", "Traefik proxy acts as dynamic edge router."]
    ] if include_docs else None,
    "metadatas": [
        [{"category": "database", "priority": 1}, {"category": "database", "priority": 2}],
        [{"category": "networking", "priority": 1}, {"category": "networking", "priority": 2}]
    ] if include_metas else None,
    "distances": [
        [0.082, 0.145],
        [0.055, 0.120]
    ] if include_dists else None
}

st.divider()

col_raw_json, col_flat_table = col_raw_json, col_flat_table = st.columns([1, 1.2])

with col_raw_json:
    st.subheader("Raw 2D QueryResult Dictionary")
    st.caption("Notice the 2-dimensional nested parallel lists indexed by `[query_idx][rank_idx]`:")
    st.json(MOCK_RAW_RESULTS)

with col_flat_table:
    st.subheader(f"Flattened Table for Query #{q_index_num}")
    
    ids_slice = MOCK_RAW_RESULTS["ids"][q_index_num]
    docs_slice = MOCK_RAW_RESULTS["documents"][q_index_num] if include_docs else ["(Excluded)"] * len(ids_slice)
    metas_slice = MOCK_RAW_RESULTS["metadatas"][q_index_num] if include_metas else ["(Excluded)"] * len(ids_slice)
    dists_slice = MOCK_RAW_RESULTS["distances"][q_index_num] if include_dists else [0.0] * len(ids_slice)
    
    rows = []
    for rank, (i, d, m, dist) in enumerate(zip(ids_slice, docs_slice, metas_slice, dists_slice), start=1):
        sim = f"{max(0.0, 1.0 - dist) * 100:.1f}%" if include_dists else "N/A"
        rows.append({
            "Rank": f"#{rank}",
            "Document ID": i,
            "Distance": f"{dist:.4f}" if include_dists else "N/A",
            "Similarity": sim,
            "Text Preview": str(d)[:45] + "..." if len(str(d)) > 45 else str(d),
            "Metadata": str(m)
        })
        
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.success(f"🟢 Successfully unpacked Query #{q_index_num} with {len(rows)} ranked hits!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why does ChromaDB return results as a dictionary of parallel lists instead of a list of objects?"):
    st.markdown("""
    **Answer**:
    * **Columnar Memory Layout**: Returning a dictionary of lists (`{"ids": [...], "distances": [...]}`) aligns directly with columnar data processing (NumPy, Arrow, Pandas).
    * **Vectorized Slicing**: Avoids instantiating thousands of small Python dictionary objects inside C++ / Rust bindings, reducing memory allocation overhead during high-throughput queries.
    * **Payload Selectivity**: Allows omitting entire columns (e.g. setting `include=["distances"]` completely eliminates allocating `documents` lists).
    """)

with st.expander("Q2: How do you access the Top-1 document text when executing a single query?"):
    st.markdown("""
    **Answer**:
    * The correct syntax is:
      `top_document = results["documents"][0][0]`.
    * Explanation:
      * The first `[0]` accesses Query Index 0 (the first query in the batch).
      * The second `[0]` accesses Rank Index 0 (the closest nearest-neighbor hit).
    """)

with st.expander("Q3: What happens to `results['documents']` if you omit `'documents'` from the `include` parameter?"):
    st.markdown("""
    **Answer**:
    * If `'documents'` is not listed in `include`, `results['documents']` is returned as `None` (or omitted from the dictionary).
    * Attempting to access `results['documents'][0][0]` will raise a `TypeError: 'NoneType' object is not subscriptable`.
    * **Best Practice**: Always check `if results.get('documents'):` or use a defensive helper function before unpacking.
    """)

with st.expander("Q4: Can you request the raw high-dimensional embeddings of the retrieved documents in the query results?"):
    st.markdown("""
    **Answer**:
    * **Yes**. Pass `"embeddings"` in the `include` parameter:
      `collection.query(query_texts=["..."], include=["documents", "distances", "embeddings"])`.
    * Chroma will return a 3-dimensional list: `results['embeddings'][query_idx][rank_idx] -> List[float]`.
    * **Caution**: This returns hundreds of floating-point numbers per document, significantly increasing network latency and memory usage.
    """)

with st.expander("Q5: How do you convert ChromaDB query results into a Pandas DataFrame?"):
    st.markdown("""
    **Answer**:
    ```python
    import pandas as pd

    def chroma_to_dataframe(results, query_idx=0):
        return pd.DataFrame({
            "id": results["ids"][query_idx],
            "document": results["documents"][query_idx],
            "metadata": results["metadatas"][query_idx],
            "distance": results["distances"][query_idx]
        })
    ```
    This flattens the parallel lists for a given query into a clean, queryable DataFrame.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Accessing `results['documents'][0]` Expecting a Single String**  
    `results['documents'][0]` yields a list of all retrieved documents for Query 0! To get the top hit string, use `[0][0]`.
    """)
    st.warning("""
    **Trap 2: Subscripting `NoneType` When Using Custom `include`**  
    Setting `include=['distances']` causes `results['documents']` to be `None`. Accessing it raises an unhandled `TypeError`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Requesting `'embeddings'` in Production Web APIs**  
    Hydrating 1,536-dimensional float arrays across 20 documents serializes megabytes of JSON for every search query, degrading API response times.
    """)
    st.warning("""
    **Trap 4: Forgetting that Distances Are Ascending (Smaller = Better)**  
    Sorting query results by descending distance mistakenly moves the most irrelevant documents to the top! Order is already sorted by ascending distance.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: How to access Top-1 doc text?"):
        st.markdown("""
        **`results['documents'][0][0]`**:  
        Index 0 for query, Index 0 for rank.
        """)
    with st.expander("🗂️ Flashcard: What does include parameter do?"):
        st.markdown("""
        **Selective Hydration**:  
        Controls whether documents, metadatas, distances, or embeddings are returned.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why parallel 2D lists?"):
        st.markdown("""
        **Columnar Alignment**:  
        Matches NumPy/Arrow layouts and eliminates Python object allocation overhead.
        """)
    with st.expander("🗂️ Flashcard: Are distances sorted?"):
        st.markdown("""
        **Yes, Ascending**:  
        Index 0 is the closest nearest neighbor (smallest distance).
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the 2D parallel matrix structure of the `QueryResult` dictionary.", value=True)
chk2 = st.checkbox("I know why `results['documents'][0][0]` is required to access the top text hit for Query 0.", value=True)
chk3 = st.checkbox("I know how to use `include` to omit unused fields and optimize search performance.", value=True)
chk4 = st.checkbox("I understand how to handle `None` values safely when specific fields are excluded via `include`.", value=True)
chk5 = st.checkbox("I can unpack raw ChromaDB query dictionaries into typed dataclasses or Pandas DataFrames.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 7.3: Query Optimization in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 7.2: Query Results and are ready for Chapter 7.3: Query Optimization!")
