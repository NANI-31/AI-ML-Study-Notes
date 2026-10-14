import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="2.2 Collections - ChromaDB Curriculum",
    page_icon="📁",
    layout="wide"
)

# Title & Overview
st.title("2.2 Collections: Lifecycle Management, Strict Naming Rules & Distance Space Configuration")
st.caption("Chapter 2: ChromaDB Basics | Domain: 5. ChromaDB")
st.markdown("""
In ChromaDB, a **Collection** is the fundamental organizational container for your vector data. 

Analogous to a table in SQL or a collection in MongoDB, a collection groups documents, embeddings, 
and metadata under a unified namespace. 

However, unlike a relational table where rows can have arbitrary columns, a ChromaDB collection represents 
a **Rigid Geometric Metric Space**:
* Every collection is bound to a single embedding dimensionality (e.g., 384 or 1536 floats).
* Every collection enforces an explicit distance metric (`cosine`, `l2`, or `ip`).
* Every collection enforces strict naming rules that are validated before creation.

In this lesson, we master the collection lifecycle (`create`, `get`, `get_or_create`, `list`, `delete`), 
dissect strict naming constraints, configure HNSW index parameters, and test an interactive 
Collection Manager and Naming Validator.
""")

st.divider()

# Section 1: The Collection Lifecycle APIs
st.header("1. The Collection Lifecycle APIs", divider="rainbow")

col_api1, col_api2 = col_api1, col_api2 = st.columns([1.1, 0.9])

with col_api1:
    st.markdown("""
    ### Five Core Collection Lifecycle Methods
    
    * **1. `client.create_collection(name, ...)`**:
      * Creates a new collection.
      * *Exception*: Throws an error if a collection with the same name already exists.
    
    * **2. `client.get_collection(name, ...)`**:
      * Retrieves an existing collection reference.
      * *Exception*: Throws `ValueError: Collection does not exist` if the name is not found.
    
    * **3. `client.get_or_create_collection(name, ...)`**:
      * **Idempotent Production Standard**: Creates the collection if missing; retrieves it if already present. 
      * Eliminates race conditions in concurrent multi-worker environments.
    
    * **4. `client.list_collections()`**:
      * Returns a list of all active collections in the database.
    
    * **5. `client.delete_collection(name)`**:
      * Drops the collection, purging all SQLite metadata records and deleting HNSW binary index segments from disk.
    """)

with col_api2:
    st.markdown("### Collection Lifecycle State Machine")
    col_dot = """
    digraph CollectionLifecycle {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        NonExistent [label="Non-Existent Collection", fillcolor="#FEE2E2", color="#DC2626"];

        ActiveColl [label="Active Chroma Collection\\n(HNSW Graph in RAM + SQLite Table)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        Dropped [label="Permanently Deleted\\n(Disk binaries unlinked)", fillcolor="#F3F4F6", color="#9CA3AF"];

        NonExistent -> ActiveColl [label="create_collection() /\\nget_or_create_collection()", color="#059669", style="bold"];
        ActiveColl -> ActiveColl [label="get_collection() /\\nadd() / query()"];
        ActiveColl -> Dropped [label="delete_collection()", color="#DC2626"];
    }
    """
    st.graphviz_chart(col_dot, use_container_width=False)

st.divider()

# Section 2: Strict Collection Naming Rules & Regex Validation
st.header("2. Strict Collection Naming Rules & Regex Validation", divider="rainbow")

col_name1, col_name2 = col_name1, col_name2 = st.columns([1, 1])

with col_name1:
    st.markdown("""
    ### ChromaDB Collection Naming Rules
    To maintain cross-platform compatibility across SQLite, Docker, and REST URLs, ChromaDB 
    enforces strict lexical constraints:
    
    1. **Length**: Must be between **3 and 63 characters** long.
    2. **Character Set**: May only contain lowercase letters, numbers, hyphens (`-`), underscores (`_`), and single dots (`.`).
    3. **Start and End Characters**: Must **begin and end with an alphanumeric character** (`[a-zA-Z0-9]`).
    4. **No Consecutive Dots**: Cannot contain two consecutive periods (`..`).
    5. **No IP Address Formatting**: Cannot be formatted as a valid IPv4 address (e.g. `192.168.1.1` is prohibited).
    """)

with col_name2:
    st.markdown("### Python Naming Validation Pattern")
    st.code(r"""import re

def validate_chroma_collection_name(name: str) -> bool:
    \"\"\"Validates collection name against Chroma's internal regex.\"\"\"
    if len(name) < 3 or len(name) > 63:
        return False
        
    # Must start and end with alphanumeric; allowed chars: [a-zA-Z0-9_.-]
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9]$"
    if not re.match(pattern, name):
        return False
        
    # Check for consecutive dots
    if ".." in name:
        return False
        
    # Check IPv4 prohibition
    ipv4_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ipv4_pattern, name):
        return False
        
    return True

# Examples:
print(validate_chroma_collection_name("kb_prod_v1"))    # True
print(validate_chroma_collection_name("-invalid-"))    # False (starts with hyphen)
print(validate_chroma_collection_name("my..collection")) # False (consecutive dots)
""", language="python")

st.divider()

# Section 3: Distance Metric & HNSW Configuration
st.header("3. Distance Metric & HNSW Configuration", divider="rainbow")

st.markdown("""
When creating a collection, configure its geometric distance metric and HNSW tuning parameters 
via the `metadata` dictionary:
""")

st.code("""import chromadb

client = chromadb.PersistentClient(path="./chroma_vault")

# Configure collection with explicit metric space and HNSW construction parameters
collection = client.get_or_create_collection(
    name="enterprise_kb",
    metadata={
        "hnsw:space": "cosine",             # Distance metric: "cosine", "l2", or "ip"
        "hnsw:construction_ef": 128,        # Beam width during index building (Default: 100)
        "hnsw:M": 16,                       # Max bidirectional links per node (Default: 16)
        "hnsw:search_ef": 64                # Beam width during runtime queries (Default: 10)
    }
)
""", language="python")

METRIC_CHOICES = [
    {
        "Distance Metric (`hnsw:space`)": "`cosine`",
        "Mathematical Space": "Cosine Distance: 1.0 - Cosine Similarity",
        "Value Range": "[0.0, 2.0] (0 = identical direction)",
        "Best Use Case": "Natural language text embeddings (OpenAI, MiniLM, BGE)."
    },
    {
        "Distance Metric (`hnsw:space`)": "`l2`",
        "Mathematical Space": "Squared Euclidean Distance: sum((x_i - y_i)^2)",
        "Value Range": "[0.0, +infinity) (0 = identical points)",
        "Best Use Case": "Computer vision vectors, unnormalized spatial features."
    },
    {
        "Distance Metric (`hnsw:space`)": "`ip`",
        "Mathematical Space": "Inner Product: 1.0 - Dot Product (for normalized vectors)",
        "Value Range": "(-infinity, +infinity)",
        "Best Use Case": "Pre-normalized unit vectors where dot product equals cosine."
    }
]

st.dataframe(pd.DataFrame(METRIC_CHOICES), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Collection Manager & Name Validator
st.header("4. Interactive Workbench: Collection Manager & Name Validator", divider="rainbow")

st.markdown("""
Test custom collection names against ChromaDB's strict validation rules, configure distance spaces, 
and inspect the generated creation statements.
""")

col_name_input, col_meta_cfg = col_name_input, col_meta_cfg = st.columns(2)

with col_name_input:
    st.subheader("1. Test Collection Name")
    candidate_name = st.text_input("Enter Desired Collection Name:", value="customer_support_kb_v2")
    
    # Live validation
    val_len = 3 <= len(candidate_name) <= 63
    val_regex = bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9]$", candidate_name))
    val_dots = ".." not in candidate_name
    val_ip = not bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", candidate_name))
    
    is_valid = val_len and val_regex and val_dots and val_ip

with col_meta_cfg:
    st.subheader("2. Metric Space & HNSW Tuning")
    chosen_space = st.selectbox("Distance Space (`hnsw:space`):", ["cosine", "l2", "ip"], index=0)
    chosen_ef_c = st.slider("Construction Beam (`hnsw:construction_ef`):", min_value=50, max_value=300, value=128)
    chosen_m = st.slider("Max Links per Node (`hnsw:M`):", min_value=8, max_value=64, value=16)

st.divider()

col_val_res, col_code_preview = col_val_res, col_code_preview = st.columns([1, 1.2])

with col_val_res:
    st.subheader("Validation Scorecard")
    if is_valid:
        st.success(f"🟢 **VALID NAME**: `'{candidate_name}'` meets all ChromaDB naming standards!")
    else:
        st.error(f"🔴 **INVALID NAME**: `'{candidate_name}'` violates ChromaDB naming rules!")
        
    st.markdown(f"""
    * **Length (3–63 chars)**: {'✅ Pass' if val_len else '❌ Fail'} (`{len(candidate_name)} chars`)
    * **Alphanumeric Start & End**: {'✅ Pass' if val_regex else '❌ Fail'}
    * **No Consecutive Dots (`..`)**: {'✅ Pass' if val_dots else '❌ Fail'}
    * **Not an IPv4 Address**: {'✅ Pass' if val_ip else '❌ Fail'}
    """)

with col_code_preview:
    st.subheader("Generated Python Creation Code")
    st.code(f"""import chromadb

client = chromadb.PersistentClient(path="./chroma_data")

# Create validated collection
collection = client.get_or_create_collection(
    name="{candidate_name if is_valid else 'valid_collection_name'}",
    metadata={{
        "hnsw:space": "{chosen_space}",
        "hnsw:construction_ef": {chosen_ef_c},
        "hnsw:M": {chosen_m}
    }}
)
print("Collection created successfully:", collection.name)
""", language="python")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: Why should production applications use `get_or_create_collection` instead of `create_collection`?"):
    st.markdown("""
    **Answer**:
    * **Idempotency**: In production microservices, multiple worker pods (or threads) initialize concurrently at boot time.
    * If Pod A and Pod B both call `create_collection("kb")`, the second caller crashes with `UniqueConstraintError: Collection already exists`.
    * Conversely, `get_collection("kb")` crashes if the collection has not yet been initialized.
    * `get_or_create_collection` provides **thread-safe, idempotent initialization**: if the collection exists, it returns a reference; if it is missing, it atomically creates it.
    """)

with st.expander("Q2: Why does ChromaDB forbid changing a collection's distance metric after creation?"):
    st.markdown("""
    **Answer**:
    * The distance metric (`cosine`, `l2`, `ip`) governs how the **HNSW graph is mathematically constructed**.
    * Every edge in the graph connects neighbors based on the chosen metric.
    * Changing the distance metric would invalidate all existing graph edges, requiring a complete, destructive rebuild of the index.
    * If you need to switch from `l2` to `cosine`, you must create a new collection with the desired metric and re-index the documents.
    """)

with st.expander("Q3: What are the naming constraints for ChromaDB collections, and why do they exist?"):
    st.markdown("""
    **Answer**:
    * **Constraints**: 3–63 characters; lowercase alphanumeric, hyphens, underscores, dots; must start/end with an alphanumeric character; no consecutive dots (`..`); no IPv4 format.
    * **Rationale**: Collection names map directly to SQLite table names, directory paths on disk, and REST URL slugs (`/api/v1/collections/{name}`). Strict naming prevents path traversal attacks, SQL injection, and URL encoding bugs.
    """)

with st.expander("Q4: How do `hnsw:M` and `hnsw:construction_ef` impact collection performance?"):
    st.markdown("""
    **Answer**:
    * **`hnsw:M` (Max Links per Node, default: 16)**:
      * Higher values (e.g. 32–64) improve recall on high-dimensional vectors but increase graph memory consumption.
    * **`hnsw:construction_ef` (Build-Time Beam Width, default: 100)**:
      * Higher values (e.g. 128–200) improve graph quality and search recall, at the cost of slower document ingestion speed.
    """)

with st.expander("Q5: What happens to underlying data on disk when `client.delete_collection()` is called?"):
    st.markdown("""
    **Answer**:
    * SQLite metadata tables associated with the collection are dropped.
    * The collection's HNSW graph in RAM is deallocated.
    * The corresponding index binary files in the persistent storage folder are unlinked and marked for deletion.
    * **Caution**: This operation is instantaneous and completely irreversible.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Using Capital Letters or Spaces in Collection Names**  
    Passing `"My Documents"` or `"Customer Support"` throws immediate validation errors. Use snake_case or kebab-case (`"my_documents"`).
    """)
    st.warning("""
    **Trap 2: Using `create_collection` in Autoscaling Pods**  
    Calling `create_collection` in container startup scripts causes crash-loops when multiple pods start simultaneously. Use `get_or_create_collection`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Omitting Distance Metric Configuration**  
    Relying on default distance space without knowing that L2 vs Cosine fundamentally changes threshold calibration leads to bad similarity filtering.
    """)
    st.warning("""
    **Trap 4: Deleting Collections in Shared Multi-Tenant Instances**  
    Calling `delete_collection` on a shared collection accidentally destroys all tenant documents at once. Use metadata-based document deletion.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is get_or_create_collection?"):
        st.markdown("""
        **Idempotent Gold Standard**:  
        Creates collection if missing, fetches it if existing; prevents race condition crashes.
        """)
    with st.expander("🗂️ Flashcard: What are naming length limits?"):
        st.markdown("""
        **3 to 63 Characters**:  
        Must start and end with alphanumeric; no consecutive dots; no IPv4 formatting.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is hnsw:space?"):
        st.markdown("""
        **Distance Metric**:  
        Configures geometric space: `"cosine"`, `"l2"`, or `"ip"`. Cannot be changed post-creation.
        """)
    with st.expander("🗂️ Flashcard: What does delete_collection do?"):
        st.markdown("""
        **Irreversible Wipe**:  
        Deallocates HNSW RAM graph, drops SQLite tables, and deletes disk binary segments.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the collection lifecycle methods: create, get, get_or_create, list, and delete.", value=True)
chk2 = st.checkbox("I know why `get_or_create_collection` is the idempotent standard for production deployments.", value=True)
chk3 = st.checkbox("I understand ChromaDB's strict collection naming rules (3–63 chars, alphanumeric boundaries).", value=True)
chk4 = st.checkbox("I know how to configure `hnsw:space` for cosine, l2, and inner product distance calculations.", value=True)
chk5 = st.checkbox("I understand how `hnsw:M` and `hnsw:construction_ef` balance index memory against recall.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 2.3: Collection Operations in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 2.2: Collections and are ready for Chapter 2.3: Collection Operations!")
