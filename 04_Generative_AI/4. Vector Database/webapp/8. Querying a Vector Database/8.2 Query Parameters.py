import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="8.2 Query Parameters - Vector DB Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("8.2 Query Parameters: Sizing, Filtering, Document Substrings & Distance Thresholds")
st.caption("Chapter 8: Querying a Vector Database | Domain: 4. Vector Database")
st.markdown("""
Executing an effective vector search query is an exercise in **multi-parameter constraint satisfaction**. 
In real-world RAG and search applications, you rarely fire an unfiltered vector query into the database. 

Instead, you coordinate six interlocking query parameters:
1. **`n_results`**: Bounding candidate depth for context window budgeting.
2. **`query_texts` vs. `query_embeddings`**: Selecting input modality and batching multiple queries.
3. **`where`**: Enforcing structured business logic (department, tenant, date ranges).
4. **`where_document`**: Requiring exact keyword substrings to eliminate hallucinated near-matches.
5. **`include`**: Specifying which payload fields to hydrate (documents, metadatas, distances, embeddings).
6. **Distance Thresholds (`max_distance`)**: Discarding irrelevant low-confidence candidates to prevent hallucinations.

In this lesson, we dissect all 6 parameters, examine their underlying execution coordination, and build an interactive 
multi-parameter query tuning workbench.
""")

st.divider()

# Section 1: The 6 Core Query Parameters Dissected
st.header("1. The 6 Core Query Parameters Dissected", divider="rainbow")

col_p1, col_p2 = col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("""
    ### 1. Ingress & Sizing Parameters
    * **`n_results` (Integer)**:
      * The maximum number of nearest neighbors to retrieve per query.
      * *Default*: `10`. Typically set to `3 - 5` for LLM prompt context injection.
    * **`query_texts` (`List[str]`)**:
      * A list of natural language query strings.
      * Automatically vectorized in batch by the collection's bound `embedding_function`.
    * **`query_embeddings` (`List[List[float]]`)**:
      * A list of pre-computed dense floating-point vectors.
      * Skips the embedding function entirely; dispatches directly to the C++ HNSWlib index.
    """)

with col_p2:
    st.markdown("""
    ### 2. Filtering & Projection Parameters
    * **`where` (`Dict[str, Any]`)**:
      * Metadata filter expression applying boolean logic (`$eq`, `$gt`, `$in`, `$and`, `$or`) on payload attributes.
    * **`where_document` (`Dict[str, str]`)**:
      * Substring constraint on the raw text content:
        `where_document={"$contains": "Kubernetes"}`.
    * **`include` (`List[str]`)**:
      * Projection list specifying which data fields to return.
      * Supported: `["documents", "metadatas", "distances", "embeddings"]`.
      * *Best Practice*: Omit `"embeddings"` in web APIs to save 90%+ network bandwidth.
    """)

st.markdown("### Master Parameter Reference Matrix")
param_matrix_df = pd.DataFrame({
    "Parameter": ["n_results", "query_texts", "query_embeddings", "where", "where_document", "include"],
    "Type": ["int", "List[str]", "List[List[float]]", "Dict[str, Any]", "Dict[str, str]", "List[str]"],
    "Default Value": ["10", "None", "None", "None", "None", "['documents', 'metadatas', 'distances']"],
    "Engine Action": [
        "Limits Top-K candidate heap capacity",
        "Encodes text via embedding function -> HNSW",
        "Direct SIMD distance scan on HNSW graph",
        "Bitset mask evaluation on metadata index",
        "Lexical substring filtering on SQLite text",
        "Payload projection during result hydration"
    ]
})
st.dataframe(param_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Complete Multi-Parameter Execution Flowchart
st.header("2. Multi-Parameter Query Execution Engine Flowchart", divider="rainbow")

st.markdown("""
How does ChromaDB coordinate all six parameters under the hood? 
The diagram below details the exact query execution order:
""")

exec_flow_dot = """
digraph ParameterExecutionFlow {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    QueryCall [label="collection.query(\\nquery_texts=['...'],\\nwhere={'dept': 'DevOps'},\\nwhere_document={'$contains': 'HPA'},\\nn_results=3)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_coord {
        label="Query Execution Planner"; style="dashed"; color="#059669";
        Vectorize [label="1. Vectorize Query String\\nq = embed(query_texts[0])", fillcolor="#D1FAE5"];
        MetaFilter [label="2. Evaluate Metadata (where)\\nBuild Roaring Bitset Mask", fillcolor="#A7F3D0"];
        DocFilter [label="3. Evaluate Content (where_document)\\nIntersect with Substring Bitset", fillcolor="#6EE7B7"];
        HNSWTraverse [label="4. Traverse HNSW Graph Index\\nCheck allowed bitset during hops", fillcolor="#34D399", color="#059669", style="filled,bold"];
        HeapCollect [label="5. Collect Top-N Nearest\\nBounded Min-Distance Heap", fillcolor="#A7F3D0"];
        ThresholdCheck [label="6. Post-Filter Distance Threshold\\nDrop if dist > max_distance", fillcolor="#FEF3C7"];
    }

    FinalPayload [label="Hydrated Response Output\\n{ids, distances, documents, metadatas}", fillcolor="#FDE68A", color="#D97706", style="filled,bold"];

    QueryCall -> Vectorize -> MetaFilter -> DocFilter -> HNSWTraverse -> HeapCollect -> ThresholdCheck -> FinalPayload;
}
"""
st.graphviz_chart(exec_flow_dot, use_container_width=False)

st.divider()

# Section 3: Calibrating Distance Thresholds to Prevent Hallucinations
st.header("3. Calibrating Distance Thresholds to Prevent Hallucinations", divider="rainbow")

st.markdown("""
Vector databases always return `n_results` candidates—even if the user query is completely nonsensical or unrelated 
to your domain.

Without **Distance Thresholding**, your LLM will be fed irrelevant garbage context, triggering hallucinations!
""")

col_thresh1, col_thresh2 = col_thresh1, col_thresh2 = st.columns([1, 1])

with col_thresh1:
    st.markdown("""
    ### Interpreting Chroma's Distance Metric
    * ChromaDB reports **`distances`**, where **smaller is closer** (`0.0` is identical).
    * In **Cosine Space (`hnsw:space: "cosine"`)**:
      * `distance = 1.0 - cosine_similarity`.
      * `dist <= 0.15`: High-confidence match (Cosine similarity >= 0.85).
      * `0.15 < dist <= 0.30`: Moderate topical match (Cosine similarity 0.70 to 0.85).
      * `dist > 0.35`: Semantic noise (Cosine similarity < 0.65) -> **DISCARD!**
    """)

with col_thresh2:
    st.markdown("""
    ### Implementing Threshold Pruning in Python
    ChromaDB does not have a native `max_distance` query argument, so threshold pruning is applied 
    as an immediate post-retrieval guardrail:
    ```python
    results = collection.query(query_texts=["..."], n_results=5)

    # Prune candidates exceeding maximum allowable distance
    MAX_ALLOWABLE_DISTANCE = 0.30

    filtered_hits = []
    for doc_id, dist, doc_text in zip(results["ids"][0], results["distances"][0], results["documents"][0]):
        if dist <= MAX_ALLOWABLE_DISTANCE:
            filtered_hits.append({"id": doc_id, "dist": dist, "text": doc_text})

    if not filtered_hits:
        print("Zero confident matches found. Abort LLM call!")
    ```
    """)

st.divider()

# Section 4: Interactive Multi-Parameter Query Simulator
st.header("4. Interactive Multi-Parameter Query Simulator", divider="rainbow")

st.markdown("""
Experiment with all 6 query parameters against an active technical knowledge base. 
Observe how combining `n_results`, `where`, `where_document`, and distance thresholds 
filters and refines the returned context:
""")

# Sample Knowledge Base
param_kb = [
    {"id": "doc_k8s_01", "text": "Kubernetes Horizontal Pod Autoscaler deployment with custom CPU thresholds.", "dept": "DevOps", "tier": "Enterprise", "year": 2024},
    {"id": "doc_k8s_02", "text": "Kubernetes cluster security hardening with Calico network policies.", "dept": "DevOps", "tier": "Standard", "year": 2023},
    {"id": "doc_db_01", "text": "PostgreSQL connection pool optimization using PgBouncer transaction pooling.", "dept": "Database", "tier": "Enterprise", "year": 2024},
    {"id": "doc_db_02", "text": "MySQL InnoDB buffer pool sizing and dirty page flush tuning.", "dept": "Database", "tier": "Standard", "year": 2022},
    {"id": "doc_sec_01", "text": "Single Sign-On SAML 2.0 integration with Okta identity provider.", "dept": "Security", "tier": "Enterprise", "year": 2024}
]

col_sim_in, col_sim_out = st.columns([1, 1])

with col_sim_in:
    st.subheader("Query Configuration Controls")
    sim_query_text = st.text_input("Query Text (`query_texts`)", value="How to configure pod autoscaling?")
    sim_k = st.slider("Top-K Limit (`n_results`)", 1, 5, 3)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_dept = st.selectbox("Metadata Filter (`where={'dept': ...}`)", ["All Departments", "DevOps", "Database", "Security"])
    with col_f2:
        f_tier = st.selectbox("Tier Filter (`where={'tier': ...}`)", ["All Tiers", "Enterprise", "Standard"])
        
    doc_substring = st.text_input("Document Substring (`where_document={'$contains': ...}`)", value="")
    max_dist_cutoff = st.slider("Max Allowable Distance Threshold Cutoff", 0.05, 1.0, 0.40, 0.05)

with col_sim_out:
    st.subheader("Simulated Retrieval Execution & Pruning")
    
    # 1. Apply where filters
    allowed_candidates = []
    for d in param_kb:
        if f_dept != "All Departments" and d["dept"] != f_dept:
            continue
        if f_tier != "All Tiers" and d["tier"] != f_tier:
            continue
        if doc_substring and doc_substring.lower() not in d["text"].lower():
            continue
        allowed_candidates.append(d)
        
    # 2. Simulate vector distances
    scored = []
    for d in allowed_candidates:
        if "autoscaling" in sim_query_text.lower() and "Autoscaler" in d["text"]:
            dist = 0.08
        elif "security" in sim_query_text.lower() and "security" in d["text"].lower():
            dist = 0.12
        elif "connection" in sim_query_text.lower() and "connection pool" in d["text"].lower():
            dist = 0.09
        elif "DevOps" in d["dept"]:
            dist = 0.28
        else:
            dist = 0.65
        scored.append((d, dist))
        
    scored.sort(key=lambda x: x[1])
    top_candidates = scored[:sim_k]
    
    # 3. Apply Distance Threshold Pruning
    retained_hits = [c for c in top_candidates if c[1] <= max_dist_cutoff]
    
    st.markdown(f"**Execution Summary:** Evaluated {len(param_kb)} total records -> {len(allowed_candidates)} passed filters -> Top-{sim_k} collected -> **{len(retained_hits)} passed distance threshold**.")
    
    if retained_hits:
        for idx, (doc, dist) in enumerate(retained_hits, 1):
            st.markdown(f"**Hit {idx}:** `{doc['id']}` | **Distance:** `{dist:.4f}` (Passes <= `{max_dist_cutoff:.2f}`)")
            st.markdown(f"*Text:* `{doc['text']}`")
            st.markdown(f"*Metadata:* `{{'dept': '{doc['dept']}', 'tier': '{doc['tier']}', 'year': {doc['year']}}}`")
            st.markdown("---")
    else:
        st.error("Zero candidates met the distance threshold cutoff! (Hallucination prevention guardrail triggered).")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Multi-Parameter Query Demonstration in ChromaDB", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates executing a query coordinating all 6 query parameters: 
`query_texts`, `n_results`, `where`, `where_document`, `include`, and a post-query distance threshold guardrail:
""")

code_params_demo = '''import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
col = client.create_collection("cloud_handbook", metadata={"hnsw:space": "cosine"})

# Ingest documents
col.add(
    documents=[
        "Kubernetes Horizontal Pod Autoscaler configuration for CPU thresholds.",
        "Kubernetes Calico network policy security hardening.",
        "PostgreSQL connection pooling with PgBouncer.",
        "MySQL InnoDB buffer pool sizing guide."
    ],
    metadatas=[
        {"dept": "DevOps", "tier": "Enterprise", "year": 2024},
        {"dept": "DevOps", "tier": "Standard", "year": 2023},
        {"dept": "Database", "tier": "Enterprise", "year": 2024},
        {"dept": "Database", "tier": "Standard", "year": 2022}
    ],
    ids=["k8s_01", "k8s_02", "db_01", "db_02"]
)

# Multi-Parameter Query
print("=== Executing Multi-Parameter Query ===")
results = col.query(
    query_texts=["How to configure autoscaling?"],
    n_results=3,
    where={"dept": "DevOps"},                       # 1. Metadata filter
    where_document={"$contains": "Autoscaler"},       # 2. Document content filter
    include=["documents", "metadatas", "distances"]  # 3. Payload projection
)

# 4. Post-Query Distance Threshold Guardrail
MAX_DISTANCE_CUTOFF = 0.35
filtered_hits = []

for doc_id, dist, doc_text in zip(results["ids"][0], results["distances"][0], results["documents"][0]):
    if dist <= MAX_DISTANCE_CUTOFF:
        filtered_hits.append({"id": doc_id, "dist": dist, "text": doc_text})
    else:
        print(f"Pruned noisy candidate {doc_id} with distance {dist:.4f} > {MAX_DISTANCE_CUTOFF}")

print(f"\\nRetained {len(filtered_hits)} high-confidence match(es):")
for hit in filtered_hits:
    print(f"  * ID: {hit['id']} | Distance: {hit['dist']:.4f} | Text: '{hit['text']}'")
'''

st.code(code_params_demo, language="python")

if st.button("🚀 Execute Multi-Parameter Query Demo", type="primary"):
    exec_scope = {}
    exec(code_params_demo, globals(), exec_scope)
    
    hits_out = exec_scope["filtered_hits"]
    
    st.success("Multi-Parameter Query executed and threshold pruned successfully!")
    for h in hits_out:
        st.markdown(f"`ID: {h['id']}` | **Distance:** `{h['dist']:.4f}`")
        st.markdown(f"*Text:* `{h['text']}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. How do `where` (metadata filter) and `where_document` (content filter) interact during Chroma query execution?"""
a1 = """* `where` filters against structured JSON metadata key-values (e.g. `{"dept": "DevOps"}`).
* `where_document` filters against the raw text string (e.g. `{"$contains": "Autoscaler"}`).
* ChromaDB evaluates both filters as an **implicit logical AND intersection**:
  * A document MUST pass both the metadata condition AND contain the required substring to be included in the candidate list.
* This allows combining high-level governance filtering (RBAC, tenant) with strict lexical keyword enforcement."""

q2 = """2. Why is post-retrieval distance thresholding essential to prevent hallucinations in enterprise RAG?"""
a2 = """* **The Unconditional Retrieval Problem**:
  * If a user asks `"What is the refund policy for employee stock options?"` against a technical software documentation database, Top-K retrieval will STILL return the top 3 closest documents!
  * Those documents might be about Git pull requests, with a dismal distance of `0.85` (cosine similarity `0.15`).
  * If injected into the LLM context, the LLM will attempt to answer using the irrelevant text, leading to severe hallucinations.
* **The Threshold Fix**: By enforcing `distance <= 0.30`, all 3 documents are discarded. The RAG service immediately returns: `'I do not have documentation to answer this question.'`"""

q3 = """3. How does omitting `"embeddings"` from the `include` parameter improve API response times?"""
a3 = """* In high-throughput microservices, network serialization and JSON encoding are significant bottlenecks.
* A 1536-dimensional vector contains 1,536 floating-point numbers. In JSON, serializing 1,536 floats takes ~15 KB of ASCII text per document.
* If returning `n_results=10`, including embeddings adds **150 KB of JSON payload overhead** per search request!
* Omitting `"embeddings"` reduces the response to ~2 KB (text + metadata), saving 98% of payload size and accelerating JSON parsing by 5x-10x."""

q4 = """4. What happens if `n_results` is set larger than the total number of matching records in the collection?"""
a4 = """* ChromaDB gracefully handles this without throwing an error.
* If a collection has only 3 documents matching the metadata filter, but `n_results=10` is requested, Chroma simply returns the 3 available matching documents.
* `len(results["ids"][0])` will be `3` instead of `10`."""

q5 = """5. How can you calibrate the optimal distance threshold value for a production RAG application?"""
a5 = """* **Golden Dataset Calibration**:
  1. Assemble a validation test set of 200 in-domain questions (where relevant documents exist) and 100 out-of-domain distractor questions (where zero relevant documents exist).
  2. Run all queries and plot the histogram of distances for Rank-1 results across both sets.
  3. The distributions will typically show two clear peaks: in-domain queries concentrate at `dist in [0.05, 0.25]`, while out-of-domain queries concentrate at `dist > 0.40`.
  4. Place your cutoff threshold in the valley between the two peaks (e.g. `0.30`) to maximize true positives while rejecting distractors."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Query Parameter Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: Hardcoding Distance Thresholds Across Model Upgrades**  
> If an application migrates from `text-embedding-ada-002` to `text-embedding-3-small`, the geometric density 
> of the vector space changes. A distance threshold of `0.25` that worked perfectly on the old model may 
> filter out valid matches on the new model. Always recalibrate distance thresholds when changing embedding models.

> [!WARNING]
> **Trap 2: The Over-Constrained Filter Zero-Match Silent Failure**  
> Combining multiple overly strict `$and` filters with `where_document={"$contains": "..."}` often matches zero 
> documents. If your application code does not check `if not results['ids'][0]:`, attempting to access `results['ids'][0][0]` 
> will throw an `IndexError: list index out of range`! Always validate list length before indexing.

> [!WARNING]
> **Trap 3: Misinterpreting Chroma Distance as Cosine Similarity**  
> Writing `if dist >= 0.70:` assuming that higher is better will completely invert your retrieval logic, 
> keeping the worst semantic noise and discarding the best matching documents! 
> Remember that Chroma returns **distance** (`0.0` is identical; `1.0` is orthogonal).
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What Does `include` Control?"):
        st.markdown("""
        **`include` Projection**:  
        Specifies payload fields to return (`documents`, `metadatas`, `distances`, `embeddings`). Omit embeddings to save network bandwidth.
        """)
    with st.expander("🗂️ Flashcard: How to Filter Text Substrings?"):
        st.markdown("""
        **`where_document`**:  
        Enforces lexical substring filtering on raw document text (e.g. `{"$contains": "keyword"}`).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Use Distance Thresholds?"):
        st.markdown("""
        **Hallucination Prevention**:  
        Discards low-confidence candidates with high distances, preventing out-of-domain queries from hallucinating LLM answers.
        """)
    with st.expander("🗂️ Flashcard: How Does Chroma Score Work?"):
        st.markdown("""
        **Chroma Distance**:  
        Returns distance (lower is closer). In cosine space, `distance = 1.0 - cosine_similarity`.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can explain all 6 query parameters supported by ChromaDB.", value=True)
chk2 = st.checkbox("I understand how `where` and `where_document` combine as an implicit logical AND.", value=True)
chk3 = st.checkbox("I know why omitting `embeddings` from `include` saves up to 98% network payload size.", value=True)
chk4 = st.checkbox("I understand how distance thresholds prevent LLM hallucinations on out-of-domain queries.", value=True)
chk5 = st.checkbox("I know how to calibrate a distance threshold using a validation test set.", value=True)
chk6 = st.checkbox("I can implement a multi-parameter query pipeline with threshold pruning in Python.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Query Parameters and are ready for Chapter 8.3: Understanding Results!")
