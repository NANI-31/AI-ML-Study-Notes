import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="8.3 Understanding Results - Vector DB Curriculum",
    page_icon="📊",
    layout="wide"
)

# Title & Overview
st.title("8.3 Understanding Results: Structure, Parallel Arrays & Distance Interpretation")
st.caption("Chapter 8: Querying a Vector Database | Domain: 4. Vector Database")
st.markdown("""
Once a vector database executes a similarity query, understanding how to **parse, unpack, and interpret the returned result payload** 
is critical for downstream applications.

Unlike traditional SQL databases that return a list of rows (`[Row(a=1, b=2), ...]`), ChromaDB returns a dictionary of 
**parallel two-dimensional nested lists**:
```python
results = {
    "ids": [["doc_1", "doc_2"]],
    "distances": [[0.0821, 0.1540]],
    "documents": [["Text 1...", "Text 2..."]],
    "metadatas": [[{"dept": "DevOps"}, {"dept": "DevOps"}]]
}
```

In this lesson, we master the data structure of Chroma results, learn why 2D nested lists exist to support batch queries, 
examine how to convert **raw distance values into human-readable confidence scores**, and build an enterprise 
result parser to construct grounded RAG citations.
""")

st.divider()

# Section 1: The Parallel 2D Nested Array Architecture
st.header("1. The Parallel 2D Nested Array Architecture", divider="rainbow")

col_res1, col_res2 = col_res1, col_res2 = st.columns([1, 1])

with col_res1:
    st.markdown("""
    ### Why Chroma Returns 2D Nested Lists
    * Because `collection.query()` supports **batching multiple query strings** simultaneously (`query_texts=["Q1", "Q2"]`), the outer list corresponds to the **Query Index**, and the inner list corresponds to the **Rank Index**:
      * `results["ids"][0]`: List of top-K IDs for Query 1.
      * `results["ids"][1]`: List of top-K IDs for Query 2.
      * `results["ids"][0][2]`: The 3rd ranked document ID for Query 1.
    
    ### The 4 Parallel Columns
    1. **`ids` (`List[List[str]]`)**: Unique string record identifiers.
    2. **`distances` (`List[List[float]]`)**: Geometric distances to the query vector.
    3. **`documents` (`List[List[str]]`)**: Hydrated raw document text chunks.
    4. **`metadatas` (`List[List[Dict[str, Any]]]` | `List[List[None]]`)**: Structured payload attributes.
    """)

with col_res2:
    st.markdown("### Chroma Result Dictionary Anatomy")
    anat_dot = """
    digraph ResultAnatomy {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        QueryOutput [label="results = collection.query()\\nTop-Level Dictionary", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_parallel {
            label="Parallel 2D Arrays (Query 0, Rank 0..K)"; style="dashed"; color="#059669";
            IDs [label="results['ids'][0][i]\\nRecord Identifier", fillcolor="#D1FAE5"];
            Dists [label="results['distances'][0][i]\\nDistance Metric Score", fillcolor="#A7F3D0"];
            Docs [label="results['documents'][0][i]\\nRaw Text Payload", fillcolor="#6EE7B7"];
            Metas [label="results['metadatas'][0][i]\\nProvenance Dictionary", fillcolor="#34D399"];
        }

        UnifiedItem [label="Unified SearchResultItem\\n(Pydantic Object for LLM / UI)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        QueryOutput -> IDs;
        QueryOutput -> Dists;
        QueryOutput -> Docs;
        QueryOutput -> Metas;

        IDs -> UnifiedItem [label="Zip by index i"];
        Dists -> UnifiedItem;
        Docs -> UnifiedItem;
        Metas -> UnifiedItem;
    }
    """
    st.graphviz_chart(anat_dot, use_container_width=False)

st.divider()

# Section 2: Interpreting Distances Across Metric Spaces
st.header("2. Interpreting Distances Across Distance Spaces", divider="rainbow")

st.markdown("""
A common source of confusion in vector search is misinterpreting distance values. 
Because Chroma returns **distances** (where smaller is better), understanding their mathematical bounds 
is essential for thresholding:
""")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("""
    #### 1. Cosine Space (`"cosine"`)
    * **Formula**: `dist = 1.0 - cos_sim`.
    * **Range**: `[0.0, 2.0]`.
    * **Interpretation**:
      * `0.00`: Identical vector (100% match).
      * `0.10`: Extremely high confidence.
      * `0.25`: Good semantic relevance.
      * `1.00`: Completely orthogonal (0% overlap).
      * `2.00`: Diametrically opposite.
    * **Confidence Formula**:
      `Confidence % = (1.0 - dist) * 100`.
    """)

with col_m2:
    st.markdown("""
    #### 2. Squared L2 Space (`"l2"`)
    * **Formula**: `dist = sum((u_i - v_i)^2)`.
    * **Range**: `[0.0, infinity)`.
    * **Unit Sphere Equivalence**:
      * If vectors are unit-normalized:
        `dist = 2.0 - 2.0 * cos_sim`.
      * `cos_sim = 1.0 - (dist / 2.0)`.
    * **Interpretation**:
      * `0.00`: Identical vector.
      * `0.20`: Very close neighbors.
      * `2.00`: Orthogonal vectors.
      * `4.00`: Opposite vectors.
    """)

with col_m3:
    st.markdown("""
    #### 3. Inner Product Space (`"ip"`)
    * **Formula**: `dist = 1.0 - (u . v)`.
    * **Range**: `(-infinity, infinity)`.
    * **Unit Sphere Equivalence**:
      * If vectors are unit-normalized, `u . v == cos_sim`.
      * Distance simplifies to standard Cosine distance `1.0 - cos_sim`.
    * **Warning**:
      * If vectors are unnormalized, distance can be **negative**!
      * Always normalize vectors before using Inner Product.
    """)

st.divider()

# Section 3: Interactive Result Parser & Confidence Interpreter
st.header("3. Interactive Result Parser & Confidence Interpreter", divider="rainbow")

st.markdown("""
Input simulated raw Chroma query distances across different metric spaces to inspect how they translate 
into **Cosine Similarities**, **Confidence Ratings**, and **RAG Prompt Citations**:
""")

col_calc_in, col_calc_out = st.columns([1, 1])

with col_calc_in:
    st.subheader("Raw Chroma Result Input")
    sim_metric = st.selectbox("Collection Distance Metric (`hnsw:space`)", ["cosine", "l2 (Squared Euclidean)", "ip (Inner Product)"])
    raw_dist = st.slider("Raw Chroma Distance Output", 0.0, 2.0, 0.14, 0.01)
    
    st.markdown("---")
    st.markdown("**Sample Record Payload:**")
    samp_id = st.text_input("Record ID", value="kb_security_042")
    samp_doc = st.text_area("Document Content", value="SAML 2.0 Single Sign-On requires uploading your identity provider metadata XML into the Okta admin portal.")
    samp_source = st.text_input("Metadata Source", value="sso_admin_guide.pdf (Page 12)")

with col_calc_out:
    st.subheader("Interpreted Quality & Citation Card")
    
    # Calculate Cosine Similarity & Confidence
    if "cosine" in sim_metric or "ip" in sim_metric:
        cos_sim = max(0.0, min(1.0, 1.0 - raw_dist))
    else: # l2
        cos_sim = max(0.0, min(1.0, 1.0 - (raw_dist / 2.0)))
        
    conf_pct = cos_sim * 100.0
    
    if conf_pct >= 85.0:
        quality_label = "🟢 Excellent High-Confidence Match"
        action = "Safe to inject directly into LLM prompt."
    elif conf_pct >= 70.0:
        quality_label = "🟡 Moderate Relevant Context"
        action = "Useful supporting context; monitor for topic drift."
    elif conf_pct >= 50.0:
        quality_label = "🟠 Weak Marginal Match"
        action = "Risk of hallucinations; consider discarding."
    else:
        quality_label = "🔴 Irrelevant Semantic Noise"
        action = "DISCARD IMMEDIATELY! Do not send to LLM."
        
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw Chroma Distance", f"{raw_dist:.4f}")
        st.metric("Derived Cosine Similarity", f"{cos_sim:.4f}")
    with c2:
        st.metric("Match Confidence Rating", f"{conf_pct:.1f}%")
        st.markdown(f"**Status:** {quality_label}")
        
    st.info(f"**RAG Citation Card Preview:**\n\n> *\"{samp_doc}\"*\n\n**Source:** `{samp_source}` | **Confidence:** `{conf_pct:.1f}%`")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Robust Chroma Result Unpacker & Citation Builder", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates how to unpack Chroma's parallel 2D lists into 
clean, typed **`SearchResultItem`** objects, compute confidence percentages, and format structured RAG prompt citations:
""")

code_parser_demo = '''import chromadb
from chromadb.config import Settings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class SearchResultItem:
    """Clean, typed representation of a retrieved vector database item."""
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    confidence_score: float

    def format_citation(self) -> str:
        source = self.metadata.get("source", "Unknown Source")
        page = self.metadata.get("page", 1)
        return f"[Doc ID: {self.id} | {source} (p. {page}) | Confidence: {self.confidence_score:.1f}%]:\\n{self.document}"

class ChromaResultUnpacker:
    """Unpacks Chroma's parallel 2D lists into clean domain objects."""
    
    @staticmethod
    def unpack_first_query(results: Dict[str, Any], metric: str = "cosine") -> List[SearchResultItem]:
        """Unpacks the top-k results for the first query (query index 0)."""
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        items = []
        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
        documents = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)

        for doc_id, dist, doc_text, meta in zip(ids, distances, documents, metadatas):
            # Compute confidence score based on metric
            if metric == "cosine" or metric == "ip":
                cos_sim = max(0.0, min(1.0, 1.0 - dist))
            elif metric == "l2":
                cos_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            else:
                cos_sim = max(0.0, min(1.0, 1.0 - dist))
                
            conf = cos_sim * 100.0

            items.append(SearchResultItem(
                id=doc_id,
                document=doc_text,
                metadata=meta or {},
                distance=dist,
                confidence_score=conf
            ))

        return items

# Demonstration with ChromaDB
client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
col = client.create_collection("security_kb", metadata={"hnsw:space": "cosine"})

col.add(
    documents=[
        "SAML 2.0 Single Sign-On configuration guide for Okta.",
        "Role-Based Access Control (RBAC) permissions matrix for finance admins."
    ],
    metadatas=[
        {"source": "sso_manual.pdf", "page": 4},
        {"source": "rbac_policy.pdf", "page": 12}
    ],
    ids=["doc_sso", "doc_rbac"]
)

# Execute Query
raw_results = col.query(query_texts=["How to configure SAML SSO?"], n_results=2)

# Unpack
unpacked = ChromaResultUnpacker.unpack_first_query(raw_results, metric="cosine")

print("=== Formatted Grounded RAG Citations ===")
for item in unpacked:
    print(item.format_citation())
    print("-----------------------------------------")
'''

st.code(code_parser_demo, language="python")

if st.button("🚀 Execute Result Unpacker Demo", type="primary"):
    exec_scope = {}
    exec(code_parser_demo, globals(), exec_scope)
    
    unpacked_items = exec_scope["unpacked"]
    
    st.success(f"Successfully unpacked {len(unpacked_items)} structured search items!")
    for itm in unpacked_items:
        st.markdown(f"**ID:** `{itm.id}` | **Distance:** `{itm.distance:.4f}` | **Confidence:** `{itm.confidence_score:.1f}%`")
        st.markdown(f"*Document:* `{itm.document}`")
        st.markdown(f"*Metadata:* `{itm.metadata}`")
        st.markdown("---")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why does ChromaDB return a dictionary of parallel 2D lists rather than a list of dictionary objects?"""
a1 = """* **Memory & Serialization Efficiency**:
  * Representing 1,000 search results as 1,000 independent Python dictionary objects creates massive object allocation overhead and GC pressure.
  * In contrast, parallel lists map directly to underlying columnar memory buffers (C++ vectors) with zero-copy serialization.
* **Batch Query Support**:
  * Because `query_texts` can accept B query strings simultaneously, the 2D array structure allows returning results for all B queries in a single contiguous payload (`results["ids"][b][k]`)."""

q2 = """2. How do you convert Chroma's Squared Euclidean (L2) distance into Cosine Similarity for unit vectors?"""
a2 = """* **Mathematical Equivalence on Unit Hypersphere**:
  * For two unit-normalized vectors (`||u|| = ||v|| = 1.0`), their squared Euclidean distance is:
    `L2_squared = ||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u . v) = 1 + 1 - 2*cos_sim = 2 - 2*cos_sim`.
* **Inverting the Formula**:
  * `2*cos_sim = 2 - L2_squared`.
  * `cos_sim = 1.0 - (L2_squared / 2.0)`.
* *Example*: If Chroma returns an L2 distance of `0.20`:
  `cos_sim = 1.0 - (0.20 / 2.0) = 1.0 - 0.10 = 0.90` (90% similarity)."""

q3 = """3. What happens if a collection contains records where some records have metadata and others have `None`?"""
a3 = """* If a record was inserted with `metadata=None`, Chroma stores it as an empty row in SQLite.
* When querying, Chroma preserves parallel list alignment by placing `None` (or an empty dict `{}`) in the corresponding position:
  `results["metadatas"] = [[{"dept": "HR"}, None, {"dept": "Sales"}]]`.
* **Production Guardrail**: Always check `if meta is not None:` before accessing metadata keys (`meta.get("key")`) to avoid `AttributeError: 'NoneType' object has no attribute 'get'`."""

q4 = """4. How can you handle multi-query results when passing a batch of 5 questions into `query_texts`?"""
a4 = """* The outer list has length equal to the number of input queries: `len(results["ids"]) == len(query_texts)`.
* To process each query's results independently, iterate using `enumerate`:
  ```python
  for q_idx, query_text in enumerate(query_texts):
      q_ids = results["ids"][q_idx]
      q_dists = results["distances"][q_idx]
      q_docs = results["documents"][q_idx]
      print(f"Results for Query '{query_text}': {q_ids}")
  ```"""

q5 = """5. Why does an Inner Product (IP) query sometimes return negative distances in ChromaDB?"""
a5 = """* In Chroma's Inner Product space (`"hnsw:space": "ip"`), distance is defined as:
  `distance = 1.0 - (u . v)`.
* If vectors are unnormalized, their dot product `(u . v)` can exceed `1.0` (e.g. `2.5`).
* Thus: `distance = 1.0 - 2.5 = -1.5`!
* A negative distance simply means the dot product was greater than 1.0. To avoid confusing negative distances, always L2-normalize vectors prior to indexing with Inner Product."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Result Handling Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Single-Level Indexing Bug on Query Results**  
> Writing `for doc in results["documents"]:` assumes that `documents` is a 1D list of strings. 
> Because Chroma returns a 2D list (`List[List[str]]`), `doc` will be the entire list of results for Query 0! 
> Iterating over `doc` treats it as a list, causing subtle type bugs in downstream prompt formatters. 
> Always index `results["documents"][0]` when executing single queries.

> [!WARNING]
> **Trap 2: The `None` Metadata `AttributeError` Crash**  
> If some documents were added without metadata, `results["metadatas"][0][i]` will be `None`. 
> Writing `results["metadatas"][0][i]["source"]` will immediately crash with `'NoneType' object is not subscriptable`. 
> Always guard metadata access: `meta = results["metadatas"][0][i] or {}`.

> [!WARNING]
> **Trap 3: Inverting Cosine Distance vs Similarity in LLM Thresholds**  
> Developers often write `if dist > 0.80:` to keep high-similarity items, forgetting that Chroma returns **distance**. 
> This disastrously discards all the best documents (which have small distances like 0.08) and feeds the LLM 
> only completely irrelevant semantic noise! Always check `if dist <= MAX_ALLOWED_DISTANCE`.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is the Result Structure?"):
        st.markdown("""
        **Parallel 2D Arrays**:  
        Chroma returns `{ids: [[]], distances: [[]], documents: [[]], metadatas: [[]]}` where outer list is query index and inner is rank.
        """)
    with st.expander("🗂️ Flashcard: How to Convert Cosine Distance to %?"):
        st.markdown("""
        **Confidence Formula**:  
        `Confidence % = (1.0 - distance) * 100`. (0.0 distance = 100% confidence).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What if Metadata is `None`?"):
        st.markdown("""
        **Metadata Nullability**:  
        Records added without metadata return `None` in `metadatas`. Always guard with `meta = m or {}`.
        """)
    with st.expander("🗂️ Flashcard: How to Convert Unit L2 to Cosine?"):
        st.markdown("""
        **Unit L2 Conversion**:  
        `cos_sim = 1.0 - (L2_squared / 2.0)`. (L2 distance of 0.20 = 90% cosine similarity).
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why ChromaDB returns parallel 2D nested lists (`[query_idx][rank_idx]`).", value=True)
chk2 = st.checkbox("I know how to zip `ids`, `distances`, `documents`, and `metadatas` into clean domain objects.", value=True)
chk3 = st.checkbox("I can mathematically convert Cosine and Squared L2 distances into confidence percentages.", value=True)
chk4 = st.checkbox("I understand how to safely handle `None` values in the returned `metadatas` array.", value=True)
chk5 = st.checkbox("I know why negative distances can occur in unnormalized Inner Product collections.", value=True)
chk6 = st.checkbox("I can implement an automated Chroma result unpacker and citation builder in Python.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Understanding Results and are ready for Chapter 8.4: Multi-Query Retrieval!")
