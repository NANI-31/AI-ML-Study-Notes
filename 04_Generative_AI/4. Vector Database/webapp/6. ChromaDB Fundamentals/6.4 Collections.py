import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="6.4 Collections - Vector DB Curriculum",
    page_icon="📁",
    layout="wide"
)

# Title & Overview
st.title("6.4 Chroma Collections: Lifecycle Management, `hnsw:space` Metrics & Custom Embedding Functions")
st.caption("Chapter 6: ChromaDB Fundamentals | Domain: 4. Vector Database")
st.markdown("""
In ChromaDB, the **Collection** is the fundamental logical namespace where your documents, embeddings, and metadata reside. 
It is the direct equivalent of a table in a relational database.

Managing collections effectively requires mastering Chroma's specific collection APIs and configuration patterns:
* **Creation & Idempotency**: `create_collection()` vs. `get_or_create_collection()`.
* **Distance Metric Configuration**: Configuring `"hnsw:space"` (`"cosine"`, `"l2"`, `"ip"`) via collection metadata.
* **Embedding Function Binding**: Associating custom embedding models (OpenAI, HuggingFace, local ONNX) with a collection.
* **Inspection & Housekeeping**: Listing collections, checking record counts (`count()`), and cleanly purging collections (`delete_collection()`).

In this lesson, we explore the complete Chroma collection API, analyze the underlying HNSW configuration metadata, 
and learn how to prevent common production pitfalls.
""")

st.divider()

# Section 1: The Chroma Collection API Suite
st.header("1. The Core Chroma Collection API Suite", divider="rainbow")

col_api1, col_api2 = col_api1, col_api2 = st.columns([1, 1])

with col_api1:
    st.markdown("""
    ### 1. Creation & Retrieval Methods
    * **`client.create_collection(name, metadata=..., embedding_function=...)`**:
      * Initializes a new collection.
      * Throws a `ValueError` or `UniqueConstraintError` if a collection with the same name already exists.
    * **`client.get_collection(name, embedding_function=...)`**:
      * Loads an existing collection from database storage.
      * Throws `ValueError` if the collection does not exist.
      * **Critical Rule**: You must pass the exact same `embedding_function` used during creation!
    * **`client.get_or_create_collection(name, ...)`**:
      * The production best practice. Idempotently returns the existing collection or creates a fresh one.
    """)

with col_api2:
    st.markdown("""
    ### 2. Inspection & Deletion Methods
    * **`client.list_collections()`**:
      * Returns a list of all active collection objects in the database catalog.
    * **`client.delete_collection(name)`**:
      * Permanently drops the collection, removes its metadata from SQLite, and deletes its HNSW index segment files on disk.
    * **`collection.count()`**:
      * Returns the total number of records currently stored in the collection.
    * **`collection.modify(name=..., metadata=...)`**:
      * Renames the collection or updates its high-level metadata dictionary.
    """)

st.markdown("### Chroma Collection Method Reference")
methods_df = pd.DataFrame({
    "Method Call": [
        "client.create_collection()",
        "client.get_collection()",
        "client.get_or_create_collection()",
        "client.list_collections()",
        "client.delete_collection()",
        "collection.count()",
        "collection.modify()"
    ],
    "Input Arguments": [
        "name, metadata, embedding_function",
        "name, embedding_function",
        "name, metadata, embedding_function",
        "None",
        "name",
        "None",
        "name (optional), metadata (optional)"
    ],
    "Return Value": [
        "Collection object",
        "Collection object",
        "Collection object",
        "List[Collection]",
        "None",
        "Integer count",
        "None"
    ],
    "Production Role": [
        "Explicit initialization during migration scripts",
        "Fetching known collection in query handlers",
        "Idempotent initialization across multi-pod startups",
        "Monitoring, administrative dashboards & healthchecks",
        "GDPR data deletion & testing cleanup",
        "Verifying ingestion batch counts & monitoring drift",
        "Updating collection metadata or renaming collections"
    ]
})
st.dataframe(methods_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Distance Metric Configuration via `hnsw:space`
st.header("2. Configuring Distance Metrics via `hnsw:space`", divider="rainbow")

st.markdown("""
By default, older versions of Chroma defaulted to Euclidean L2 distance (`"l2"`), while modern Chroma defaults 
to **Cosine Similarity (`"cosine"`)**. 

To guarantee deterministic, predictable distance calculations across your pipeline, you should **always explicitly 
specify the distance metric** when creating a collection via the `metadata` dictionary:
""")

col_met1, col_met2 = col_met1, col_met2 = st.columns([1, 1])

with col_met1:
    st.markdown("""
    ### The 3 Supported Distance Spaces
    * **1. Cosine Distance (`"hnsw:space": "cosine"`)**:
      * *Formula*: `D = 1.0 - cos_sim(u, v)`.
      * *Range*: `[0.0, 2.0]`. Identical vectors have distance `0.0`.
      * *Best For*: General natural language text, OpenAI embeddings, BGE models.
    * **2. Squared L2 Distance (`"hnsw:space": "l2"`)**:
      * *Formula*: `D = sum((u_i - v_i)^2)`.
      * *Range*: `[0.0, infinity)`.
      * *Best For*: Absolute physical coordinate data, image color histograms, geospatial points.
    * **3. Inner Product (`"hnsw:space": "ip"`)**:
      * *Formula*: `D = 1.0 - sum(u_i * v_i)` (Inverted dot product).
      * *Best For*: Unit-normalized embeddings where ultra-fast SIMD hardware throughput is required.
    """)

with col_met2:
    st.markdown("### Code Implementation Pattern")
    st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Explicitly configure Cosine Distance and HNSW hyperparameters
collection = client.create_collection(
    name="enterprise_knowledge_base",
    metadata={
        "hnsw:space": "cosine",        # Distance metric
        "hnsw:construction_ef": 200,   # Exploration depth during build
        "hnsw:M": 32,                  # Max links per node
        "hnsw:search_ef": 64           # Exploration depth during query
    }
)""", language="python")

st.divider()

# Section 3: Custom Embedding Functions in Chroma
st.header("3. Custom Embedding Functions: OpenAI, Cohere & Local Models", divider="rainbow")

st.markdown("""
Chroma provides first-class embedding function wrappers in `chromadb.utils.embedding_functions`:
""")

col_ef1, col_ef2 = col_ef1, col_ef2 = st.columns([1, 1])

with col_ef1:
    st.markdown("""
    ### Supported Commercial & Local Wrappers
    * **OpenAI Embedding Function**:
      ```python
      from chromadb.utils import embedding_functions

      openai_ef = embedding_functions.OpenAIEmbeddingFunction(
          api_key="sk-...",
          model_name="text-embedding-3-small"
      )
      col = client.get_or_create_collection(
          name="openai_docs",
          embedding_function=openai_ef
      )
      ```
    * **Hugging Face / Sentence-Transformers**:
      ```python
      hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
          model_name="BAAI/bge-base-en-v1.5"
      )
      ```
    * **Google Generative AI (Gemini)**:
      ```python
      google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
          api_key="AIza...",
          model_name="models/text-embedding-004"
      )
      ```
    """)

with col_ef2:
    st.markdown("""
    ### Authoring a Custom Embedding Function
    You can implement a custom embedding class by conforming to the `EmbeddingFunction` protocol:
    ```python
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    import numpy as np

    class CustomNormEmbeddingFunction(EmbeddingFunction):
        def __init__(self, dimension: int = 768):
            self.dim = dimension

        def __call__(self, input: Documents) -> Embeddings:
            # Your custom model inference logic here
            embeddings = []
            for doc in input:
                vec = np.random.randn(self.dim).tolist()
                embeddings.append(vec)
            return embeddings
    ```
    """)

st.divider()

# Section 4: Interactive Chroma Collection Manager Workbench
st.header("4. Interactive Chroma Collection Manager Workbench", divider="rainbow")

st.markdown("""
Manage collections interactively in real time. Create collections with customized `hnsw:space` distance metrics, 
inspect collection counts, add sample documents, and test deletion:
""")

# Session State for Collections Workbench
if "chroma_wb_collections" not in st.session_state:
    st.session_state.chroma_wb_collections = {
        "customer_support_kb": {
            "metric": "cosine",
            "M": 16,
            "ef_con": 128,
            "docs": ["Troubleshooting Wi-Fi connection drops on router.", "How to configure guest network access."]
        },
        "legal_contracts_2024": {
            "metric": "ip",
            "M": 32,
            "ef_con": 200,
            "docs": ["Non-disclosure agreement governing confidentiality obligations.", "Vendor master services agreement."]
        }
    }

col_wb_ctrl, col_wb_list = st.columns([1, 1])

with col_wb_ctrl:
    st.subheader("Collection Operations")
    wb_action = st.radio("Select Action", ["Create Collection", "Add Document to Collection", "Delete Collection"])
    
    if wb_action == "Create Collection":
        c_name = st.text_input("Collection Name", value="engineering_wiki_v1")
        c_space = st.selectbox("Distance Space (`hnsw:space`)", ["cosine", "l2", "ip"])
        c_m = st.slider("HNSW M (Links per node)", 8, 64, 16)
        c_ef = st.slider("HNSW construction_ef", 64, 512, 128)
        
        if st.button("➕ Create Collection", type="primary"):
            if c_name in st.session_state.chroma_wb_collections:
                st.error(f"Collection `{c_name}` already exists!")
            elif len(c_name) < 3 or len(c_name) > 63:
                st.error("Collection name must be between 3 and 63 characters long!")
            else:
                st.session_state.chroma_wb_collections[c_name] = {
                    "metric": c_space,
                    "M": c_m,
                    "ef_con": c_ef,
                    "docs": []
                }
                st.success(f"Collection `{c_name}` created with metric `{c_space}`!")
                
    elif wb_action == "Add Document to Collection":
        target_col = st.selectbox("Select Target Collection", list(st.session_state.chroma_wb_collections.keys()))
        new_doc_text = st.text_area("Document Text", value="Architecture guide for deploying high-availability Redis clusters.")
        if st.button("📥 Insert Document", type="primary"):
            st.session_state.chroma_wb_collections[target_col]["docs"].append(new_doc_text)
            st.success(f"Document added to `{target_col}`! New count: {len(st.session_state.chroma_wb_collections[target_col]['docs'])}")
            
    else: # Delete
        del_col_name = st.selectbox("Select Collection to Delete", list(st.session_state.chroma_wb_collections.keys()))
        if st.button("🗑️ Permanently Delete Collection", type="primary"):
            del st.session_state.chroma_wb_collections[del_col_name]
            st.warning(f"Collection `{del_col_name}` dropped and its HNSW files purged!")

with col_wb_list:
    st.subheader("Active Collection Catalog")
    
    cat_rows = []
    for name, data in st.session_state.chroma_wb_collections.items():
        cat_rows.append({
            "Collection Name": name,
            "Metric (`hnsw:space`)": data["metric"].upper(),
            "HNSW M": data["M"],
            "HNSW efConstruction": data["ef_con"],
            "Record Count": len(data["docs"])
        })
        
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
    
    st.markdown("#### Sample Ingested Documents in Selected Collection:")
    disp_sel = st.selectbox("Inspect Documents for", list(st.session_state.chroma_wb_collections.keys()), key="inspect_col_docs")
    if disp_sel:
        docs_list = st.session_state.chroma_wb_collections[disp_sel]["docs"]
        if docs_list:
            for idx, doc in enumerate(docs_list, 1):
                st.markdown(f"**{idx}.** `{doc}`")
        else:
            st.info("Collection is currently empty.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Full Chroma Collection Lifecycle Script", divider="rainbow")

st.markdown("""
The following complete Python program exercises the entire **Chroma Collection Lifecycle**: 
creation with custom `hnsw:space`, idempotency checks, listing collections, inspecting counts, 
and clean deletion:
""")

code_col_lifecycle = '''import chromadb
from chromadb.config import Settings

# Initialize ephemeral in-memory client for testing
client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

# 1. Create Collection with Explicit Metric Configuration
print("=== 1. Creating Collection with Cosine Space ===")
kb_col = client.create_collection(
    name="customer_faq_v1",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100}
)
print(f"Collection Created: '{kb_col.name}' with Metadata: {kb_col.metadata}")

# 2. Ingest Documents
print("\\n=== 2. Ingesting Documents ===")
kb_col.add(
    documents=[
        "How to request a corporate tax exemption certificate.",
        "Updating primary credit card billing details."
    ],
    metadatas=[{"dept": "Tax"}, {"dept": "Billing"}],
    ids=["faq_01", "faq_02"]
)
print(f"Verified Record Count: {kb_col.count()}")

# 3. Idempotent get_or_create_collection
print("\\n=== 3. Idempotent get_or_create_collection ===")
same_col = client.get_or_create_collection(
    name="customer_faq_v1",
    metadata={"hnsw:space": "cosine"}
)
print(f"Idempotent Fetch Succeeded: '{same_col.name}' (Count: {same_col.count()})")

# 4. List Collections
print("\\n=== 4. Listing Collections in Cluster ===")
collections = client.list_collections()
print(f"Active Collections: {[c.name for c in collections]}")

# 5. Clean Deletion
print("\\n=== 5. Deleting Collection ===")
client.delete_collection("customer_faq_v1")
remaining = client.list_collections()
print(f"Remaining Collections After Deletion: {[c.name for c in remaining]}")
'''

st.code(code_col_lifecycle, language="python")

if st.button("🚀 Execute Collection Lifecycle Script", type="primary"):
    exec_scope = {}
    exec(code_col_lifecycle, globals(), exec_scope)
    
    cnt_val = exec_scope["kb_col"].count()
    rem_cols = [c.name for c in exec_scope["remaining"]]
    
    st.success("Collection Lifecycle script executed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown(f"**Ingested Document Count:** `{cnt_val}`")
    with col_out2:
        st.markdown(f"**Remaining Collections Post-Deletion:** `{rem_cols}` (Cleanly purged!)")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. How does ChromaDB configure distance metrics, and what are the valid values for `hnsw:space`?"""
a1 = """* In ChromaDB, distance metrics are configured at collection creation time via the `metadata` dictionary parameter:
  `client.create_collection(name="...", metadata={"hnsw:space": "<metric_name>"})`.
* **Valid Values**:
  * `"cosine"`: Cosine Distance (`1 - cos_sim`). Bounded in `[0.0, 2.0]`. Standard for natural language embeddings.
  * `"l2"`: Squared Euclidean Distance (`sum((u_i - v_i)^2)`). Sensitive to vector magnitude.
  * `"ip"`: Inverted Inner Product (`1 - sum(u_i * v_i)`). Optimized for unit-normalized vectors.
* **Production Rule**: If `"hnsw:space"` is omitted, older Chroma versions defaulted to `"l2"`, while newer versions default to `"cosine"`. Always specify it explicitly to prevent version migration bugs."""

q2 = """2. What happens if you call `client.get_collection("my_col")` without specifying the original `embedding_function`?"""
a2 = """* **The Silent Mismatch Hazard**:
  * Suppose you created `"my_col"` using the OpenAI embedding function (`1536` dimensions).
  * In a separate script or query handler, you fetch the collection with `client.get_collection("my_col")` without passing `embedding_function`.
  * Chroma will fallback to its default embedding function: `all-MiniLM-L6-v2` (`384` dimensions)!
  * When you execute `collection.query(query_texts=["..."])`, Chroma will embed the query into 384 dimensions and attempt to search a 1536-dimensional HNSW index.
  * The query immediately crashes with: `ValueError: Dimension mismatch: Expected 1536, got 384`!
* **Best Practice**: Always define a centralized helper function or wrapper that binds the correct `embedding_function` whenever calling `get_collection`."""

q3 = """3. What are ChromaDB's collection naming constraints, and why do they exist?"""
a3 = """* **Naming Rules**:
  1. Length must be between **3 and 63 characters**.
  2. Must start and end with an alphanumeric character (`[a-zA-Z0-9]`).
  3. Can only contain alphanumerics, dots (`.`), underscores (`_`), or hyphens (`-`).
  4. Cannot be a valid IPv4 address.
* **Why These Constraints Exist**:
  * In Persistent and Client-Server modes, Chroma creates local directory segment folders and SQLite table records using the collection identifier.
  * Strict naming prevents directory traversal attacks (`../../`) and ensures compatibility across Windows, Linux, and cloud object storage naming rules."""

q4 = """4. How does `delete_collection()` clean up disk resources in Persistent mode?"""
a4 = """* When `client.delete_collection("my_col")` is called:
  1. Chroma executes an atomic transaction in `chroma.sqlite3`, deleting the collection row and all associated embedding metadata records.
  2. It unmaps the C++ HNSWlib memory-mapped graph segment files from process memory.
  3. It recursively deletes the `<collection-uuid>/` directory containing `data_level0.bin`, `header.bin`, and other binary segment files on disk.
* Reclaims both RAM and disk space permanently."""

q5 = """5. How can you update the metadata of an existing Chroma collection without re-indexing all vectors?"""
a5 = """* Use the **`collection.modify()`** API:
  `collection.modify(name="new_name", metadata={"hnsw:space": "cosine", "tier": "gold"})`.
* Modifying high-level metadata updates the SQLite catalog record without altering the underlying HNSW vector graph structure.
* Note: You **CANNOT** alter `"hnsw:space"` on an existing populated collection, as changing distance metrics would invalidate the entire graph topology!"""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Collection Management Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Missing Embedding Function Runtime Crash**  
> Creating a collection with a custom OpenAI or BGE embedding function in your ingestion worker, but fetching 
> it with a raw `client.get_collection("name")` in your FastAPI query handler without passing the same embedding 
> function, results in immediate `Dimension mismatch` crashes. Always pass the matching embedding function.

> [!WARNING]
> **Trap 2: Assuming Distance Scores are Normalized Similarities**  
> ChromaDB query results return `distances`, **NOT** similarities! 
> In Cosine space (`hnsw:space: "cosine"`), a distance of `0.0` represents an identical match, while `1.0` represents 
> orthogonal concepts. Developers mistakenly treating distance as similarity often invert their ranking logic or 
> filter out the best matches! Use `similarity = 1.0 - distance` for cosine metrics.

> [!WARNING]
> **Trap 3: Unchecked `delete_collection` in CI/CD Cleanup Scripts**  
> CI/CD test cleanup scripts that execute `client.delete_collection("my_kb")` pointing to a shared staging database 
> will accidentally wipe testing data for other engineers. Always inject a random timestamp or test UUID into 
> collection names used in automated test runs (e.g. `f"test_kb_{uuid.uuid4().hex[:8]}"`).
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: How to Set Cosine Metric in Chroma?"):
        st.markdown("""
        **Set Metric**:  
        Pass `metadata={"hnsw:space": "cosine"}` during `create_collection()`.
        """)
    with st.expander("🗂️ Flashcard: What Does Chroma Query Return?"):
        st.markdown("""
        **Returned Metric**:  
        Chroma returns `distances`, where smaller is closer (0.0 is identical in cosine space).
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Pass `embedding_function` on Get?"):
        st.markdown("""
        **Embedding Function Binding**:  
        Failing to pass the matching `embedding_function` causes Chroma to default to 384-d MiniLM, triggering dimension mismatch errors.
        """)
    with st.expander("🗂️ Flashcard: What is `get_or_create_collection`?"):
        st.markdown("""
        **Idempotent Fetch**:  
        Loads an existing collection if it exists or creates it if missing, preventing startup race conditions.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to create, retrieve, list, and delete Chroma collections using the official API.", value=True)
chk2 = st.checkbox("I understand how to explicitly configure distance metrics using `metadata={'hnsw:space': '...'}`.")
chk3 = st.checkbox("I know why the exact same `embedding_function` must be passed when re-opening existing collections.")
chk4 = st.checkbox("I understand that Chroma query results return `distances` rather than similarities.")
chk5 = st.checkbox("I know how to bind custom embedding functions (OpenAI, HuggingFace) to a collection.")
chk6 = st.checkbox("I can implement a complete collection lifecycle script in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Incredible achievement! You have mastered Chroma Collections and completed Chapter 6 of the Vector Database curriculum!")
