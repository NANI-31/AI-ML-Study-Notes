import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Optional

try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(
    page_title="14.2 Querying ChromaDB - Embeddings Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("14.2 Querying ChromaDB: Similarity Search, Distances, & Metadata Filtering")
st.caption("Chapter 14: Embeddings with ChromaDB | Domain: 3. Vector Embeddings")
st.markdown("""
Once vectors, documents, and payloads are stored inside ChromaDB, the primary operational task is **semantic querying**: finding the most semantically relevant documents for a given prompt or question.

In this lesson, you will master the mechanics of querying ChromaDB:
* **Query Vectors vs. Query Texts**: Letting ChromaDB embed queries automatically vs. providing pre-computed query vectors.
* **Top-K Parameter (`n_results`)**: Retrieving the nearest $K$ neighbors from the HNSW index.
* **Understanding Distance Metrics**: Interpreting the returned `distances` (ChromaDB returns distances where lower means more similar, not raw cosine similarities).
* **Metadata Filtering (`where`)**: Applying Boolean, relational, and set operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`).
* **Document Content Filtering (`where_document`)**: Filtering directly on raw document substring matches (`$contains`, `$not_contains`).
""")

st.divider()

# Section 1: ChromaDB Query Execution Architecture
st.header("1. ChromaDB Query Execution Architecture", divider="rainbow")

col_q1, col_q2 = st.columns([1, 1])

with col_q1:
    st.markdown("""
    ### Query Pipeline Stages
    * **Stage 1: Query Embedding Generation**:
      * If you pass `query_texts=["What is LoRA?"]`, ChromaDB executes the collection's configured `EmbeddingFunction` to produce a 1D query vector.
      * Alternatively, you can pass pre-computed vectors directly to `query_embeddings=[...]`.
    * **Stage 2: Filter Evaluation & Bitmasking**:
      * If a `where` or `where_document` clause is provided, ChromaDB queries SQLite to identify eligible document IDs.
    * **Stage 3: HNSW Nearest-Neighbor Traversal**:
      * The query vector traverses the HNSW graph index to locate the top $K$ nearest vectors according to the collection's metric space.
    * **Stage 4: Distance Scoring & Payload Assembly**:
      * ChromaDB returns structured dictionaries containing `ids`, `documents`, `metadatas`, and `distances`.
    """)

with col_q2:
    st.markdown("### Query Execution Workflow")
    q_dot = """
    digraph ChromaQuery {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        UserQuery [label="User Query String\\n'Explain attention mechanism'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        WhereClause [label="Metadata Filter (where)\\n{department: 'ai', year: {$gte: 2022}}", fillcolor="#FEF3C7", color="#D97706"];
        
        EF [label="Embedding Function\\nGenerates query vector (d=384)", fillcolor="#EDE9FE", color="#8B5CF6"];
        HNSWSearch [label="HNSW Index Search\\nEvaluates candidate nodes\\nFiltered by metadata", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        
        TopK [label="Top-K Ranked Results\\nIDs, Documents, Metadatas, Distances", fillcolor="#D1FAE5", color="#059669"];

        UserQuery -> EF [label="Embed string"];
        EF -> HNSWSearch [label="Query vector"];
        WhereClause -> HNSWSearch [label="Filter predicate"];
        HNSWSearch -> TopK [label="Top n_results nearest"];
    }
    """
    st.graphviz_chart(q_dot, use_container_width=False)

st.markdown("### Interpreting ChromaDB Distance Values")
st.markdown("""
When querying ChromaDB with `metadata={"hnsw:space": "cosine"}`, ChromaDB returns **Cosine Distance**, defined as:
```
Cosine Distance = 1.0 - Cosine Similarity
```
* **Distance = 0.0**: Exact identical semantic meaning (similarity = 1.0).
* **Distance ~ 0.1 to 0.3**: Extremely close semantic match.
* **Distance ~ 0.5 to 0.7**: Moderate relevance.
* **Distance >= 1.0**: Orthogonal or unrelated documents.
* **Distance ~ 2.0**: Diametrically opposite vectors.
""")

st.divider()

# Section 2: Production Code: Advanced Querying with Filters
st.header("2. Production Code: Querying with Logical Metadata & Content Filters", divider="rainbow")

st.markdown("""
The production snippet below demonstrates querying with composite Boolean filters (`$and`, `$in`, `$gte`) and content filters (`$contains`).
""")

st.code(r'''
import chromadb
from typing import Dict, Any

def run_production_query(client: chromadb.Client, collection_name: str):
    collection = client.get_collection(name=collection_name)

    # Example 1: Basic Top-K Query
    basic_results = collection.query(
        query_texts=["How do neural networks optimize loss functions?"],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print("Basic Query Results:", basic_results["ids"][0])

    # Example 2: Composite Metadata Filtering ($and, $gte, $in)
    filtered_results = collection.query(
        query_texts=["Generative models for medical image synthesis"],
        n_results=5,
        where={
            "$and": [
                {"year": {"$gte": 2023}},
                {"category": {"$in": ["radiology", "ophthalmology"]}}
            ]
        },
        include=["documents", "metadatas", "distances"]
    )

    # Example 3: Document Content Filtering (where_document)
    content_filtered = collection.query(
        query_texts=["Distributed computing architectures"],
        n_results=3,
        where_document={"$contains": "Kubernetes"},
        include=["documents", "metadatas", "distances"]
    )

    return filtered_results
''', language="python")

st.divider()

# Section 3: Interactive ChromaDB Querying & Filter Workbench
st.header("3. Interactive Workbench: Live ChromaDB Semantic Search Engine", divider="rainbow")

st.markdown("""
Interact with a live ChromaDB collection loaded with 8 enterprise engineering documents. Test queries, tweak $K$, and apply real-time metadata filters.
""")

# Initialize or cache demo collection
@st.cache_resource
def get_demo_collection():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="enterprise_search_demo",
        metadata={"hnsw:space": "cosine"}
    )
    
    docs = [
        "Transformers rely on multi-head scaled dot-product attention to model sequence context without recurrence.",
        "Convolutional neural networks apply learnable filter kernels across spatial dimensions to detect visual patterns.",
        "Stochastic Gradient Descent and Adam optimize neural network weights by minimizing differentiable loss functions.",
        "Amazon Web Services S3 offers secure object storage with 99.999999999% durability and bucket versioning.",
        "Google Kubernetes Engine orchestrates container clusters with automated horizontal pod autoscaling.",
        "PostgreSQL supports relational ACID transactions, multi-version concurrency control, and JSONB indexing.",
        "Retrieval-Augmented Generation combines dense vector retrieval with LLMs to eliminate hallucinations.",
        "Quantization reduces neural network weights from 16-bit floating point to 4-bit integers for efficient edge inference."
    ]
    metas = [
        {"domain": "ai", "subfield": "nlp", "year": 2023, "department": "research"},
        {"domain": "ai", "subfield": "vision", "year": 2021, "department": "research"},
        {"domain": "ai", "subfield": "optimization", "year": 2022, "department": "research"},
        {"domain": "cloud", "subfield": "storage", "year": 2023, "department": "infrastructure"},
        {"domain": "cloud", "subfield": "devops", "year": 2024, "department": "infrastructure"},
        {"domain": "database", "subfield": "sql", "year": 2022, "department": "data"},
        {"domain": "ai", "subfield": "rag", "year": 2024, "department": "engineering"},
        {"domain": "ai", "subfield": "efficiency", "year": 2024, "department": "engineering"}
    ]
    ids = [f"doc_{i+1:02d}" for i in range(len(docs))]
    
    collection.add(documents=docs, metadatas=metas, ids=ids)
    return collection

demo_collection = get_demo_collection()

col_ctrl, col_res = st.columns([1, 1])

with col_ctrl:
    st.subheader("1. Query Parameters")
    user_query = st.text_input(
        "Enter Natural Language Query:",
        value="How does self-attention work in language models?"
    )

    k_results = st.slider("Top-K Candidates (`n_results`):", min_value=1, max_value=8, value=3)

    st.subheader("2. Metadata Filter (`where`)")
    use_filter = st.checkbox("Enable Metadata Filtering", value=False)
    
    filter_domain = st.selectbox("Filter by Domain:", ["all", "ai", "cloud", "database"], index=1)
    filter_min_year = st.slider("Minimum Year (`$gte`):", min_value=2020, max_value=2024, value=2023)

    where_filter: Optional[Dict[str, Any]] = None
    if use_filter:
        conditions = []
        if filter_domain != "all":
            conditions.append({"domain": {"$eq": filter_domain}})
        conditions.append({"year": {"$gte": filter_min_year}})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        st.json(where_filter)

    st.subheader("3. Document Content Filter (`where_document`)")
    use_doc_filter = st.checkbox("Enable Keyword Match (`where_document`)", value=False)
    keyword_match = st.text_input("Must contain substring:", value="attention")
    where_doc = {"$contains": keyword_match} if use_doc_filter and keyword_match else None

with col_res:
    st.subheader("4. Query Execution & Results")
    
    if st.button("🚀 Execute Semantic Search", type="primary"):
        with st.spinner("Searching HNSW index..."):
            try:
                res = demo_collection.query(
                    query_texts=[user_query],
                    n_results=k_results,
                    where=where_filter,
                    where_document=where_doc,
                    include=["documents", "metadatas", "distances"]
                )

                matched_ids = res["ids"][0]
                matched_docs = res["documents"][0]
                matched_metas = res["metadatas"][0]
                matched_dists = res["distances"][0]

                if len(matched_ids) == 0:
                    st.warning("No documents matched the combination of semantic search and metadata filters!")
                else:
                    st.success(f"Retrieved {len(matched_ids)} nearest documents!")

                    for rank, (doc_id, text, meta, dist) in enumerate(zip(matched_ids, matched_docs, matched_metas, matched_dists), 1):
                        similarity_pct = max(0.0, (1.0 - dist) * 100.0)
                        with st.expander(f"Rank {rank} | ID: {doc_id} | Similarity: {similarity_pct:.1f}% (Dist: {dist:.4f})", expanded=True):
                            st.markdown(f"**Document Text:**\n> {text}")
                            
                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            m_col1.caption(f"Domain: `{meta.get('domain')}`")
                            m_col2.caption(f"Subfield: `{meta.get('subfield')}`")
                            m_col3.caption(f"Year: `{meta.get('year')}`")
                            m_col4.caption(f"Dept: `{meta.get('department')}`")
                            
                            st.progress(float(min(1.0, max(0.0, 1.0 - dist))))

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Query Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the relationship between the distance returned by ChromaDB and Cosine Similarity?**
    * **Answer**: ChromaDB returns distance values rather than similarities. Under cosine space (`hnsw:space: "cosine"`), the formula is `Distance = 1.0 - CosineSimilarity`. A distance of 0.0 indicates identical vectors, while a distance of 1.0 indicates orthogonal (unrelated) vectors. To convert to similarity percentage: `Similarity = (1.0 - Distance) * 100%`.

    **Q2: What is the difference between `where` and `where_document` in ChromaDB?**
    * **Answer**:
      * `where`: Evaluates metadata attributes stored in SQLite using logical and comparison operators (`$eq`, `$in`, `$gte`, etc.).
      * `where_document`: Evaluates the raw document string text itself using string predicates (`$contains` or `$not_contains`).

    **Q3: Can ChromaDB query using multiple query texts in a single call?**
    * **Answer**: Yes. ChromaDB supports batch querying: `collection.query(query_texts=["Query 1", "Query 2"], n_results=5)`. The returned dictionary contains nested lists where index `0` corresponds to Query 1 and index `1` corresponds to Query 2, enabling high-throughput parallel inference.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Requesting More `n_results` Than Total Available Records**:
      * *Issue*: Calling `collection.query(..., n_results=10)` when the collection only contains 4 documents.
      * *Consequence*: In earlier versions of ChromaDB, this raised an unhandled `ValueError`. In modern versions, it returns all available documents but issues a warning. Always cap `n_results <= collection.count()`.

    * **Trap 2: Forgetting That `where` Filters Are Case-Sensitive**:
      * *Issue*: Filtering with `where={"category": "AI"}` when the metadata was stored as `{"category": "ai"}`.
      * *Consequence*: Zero results returned despite semantically identical text.
      * *Fix*: Standardize all string metadata values to lowercase during the ingestion pipeline.

    * **Trap 3: Overly Constrained Metadata Filters Returning 0 Results**:
      * *Issue*: Combining multiple `$and` predicates that eliminate all candidate vectors prior to HNSW retrieval.
      * *Consequence*: The RAG application receives an empty context and hallucinations increase.
      * *Fix*: Implement fallback logic: if a filtered search returns 0 documents, loosen constraints or fall back to an unfiltered semantic search with a relevance score threshold.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `collection.query(n_results=K)`**
      * *Definition*: The method to retrieve the top $K$ nearest neighbor documents and metadata for given query texts or embedding vectors.
    * **Flashcard 2: Cosine Distance in ChromaDB**
      * *Definition*: `1.0 - Cosine Similarity`. Lower values mean higher semantic similarity.
    * **Flashcard 3: `$contains` in `where_document`**
      * *Definition*: A substring search filter that restricts search results to documents containing a specific substring.
    * **Flashcard 4: `include` parameter**
      * *Definition*: A list specifying which fields to return in the query response (`documents`, `metadatas`, `embeddings`, `distances`).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Querying ChromaDB:

- [ ] I understand how ChromaDB embeds query strings and navigates the HNSW graph.
- [ ] I know how to convert ChromaDB cosine distance into cosine similarity.
- [ ] I can construct complex metadata filter queries using `$and`, `$or`, `$gte`, and `$in`.
- [ ] I understand how `where_document` differs from `where`.
- [ ] I know how to interpret and unpack batch query responses.
""")
