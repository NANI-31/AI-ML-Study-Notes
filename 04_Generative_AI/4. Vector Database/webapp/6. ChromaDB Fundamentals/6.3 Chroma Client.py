import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.3 Chroma Client - Vector DB Curriculum",
    page_icon="🔌",
    layout="wide"
)

# Title & Overview
st.title("6.3 Chroma Client: Ephemeral, Persistent Disk & Remote HttpClient Architectures")
st.caption("Chapter 6: ChromaDB Fundamentals | Domain: 4. Vector Database")
st.markdown("""
In ChromaDB, the **Client** is the primary gateway to all database operations: creating collections, configuring distance metrics, 
inserting vectors, and executing similarity queries.

A major strength of ChromaDB is its **tri-modal client architecture**, which allows developers to use the exact same 
Python API across three distinct operational environments:
1. **Ephemeral In-Memory Client (`EphemeralClient`)**: Volatile, in-memory execution for rapid prototyping and CI unit tests.
2. **Persistent Disk Client (`PersistentClient`)**: Embedded local persistence storing SQLite tables and HNSW index files on disk.
3. **Remote HTTP Client (`HttpClient`)**: Decoupled microservice architecture connecting to a standalone ChromaDB server over network sockets.

In this lesson, we dissect client initialization, examine the underlying on-disk directory layout, master the `Settings` class, 
and learn how to transition from embedded scripts to production web services.
""")

st.divider()

# Section 1: The 3 Client Modalities Compared
st.header("1. The 3 Chroma Client Modalities Dissected", divider="rainbow")

col_cl1, col_cl2 = col_cl1, col_cl2 = st.columns([1, 1])

with col_cl1:
    st.markdown("""
    ### 1. Ephemeral Client (`chromadb.EphemeralClient()`)
    * **Characteristics**:
      * Data lives 100% in volatile RAM (`:memory:`).
      * When the Python script exits or the notebook kernel restarts, **all data is permanently wiped**.
    * **Best Use Cases**:
      * Automated unit testing in GitHub Actions / CI/CD pipelines.
      * Interactive scratchpad experiments in Jupyter notebooks.
      * Disposable sandbox demonstrations.
    
    ### 2. Persistent Client (`chromadb.PersistentClient()`)
    * **Characteristics**:
      * Embeds the database engine directly inside your Python process.
      * Writes database tables to `chroma.sqlite3` and persists vector graphs inside a designated local directory:
        `client = chromadb.PersistentClient(path="./my_chroma_data")`.
    * **Best Use Cases**:
      * Single-process Python desktop applications, CLI scripts, and local batch data processing.
    """)

with col_cl2:
    st.markdown("""
    ### 3. Remote HTTP Client (`chromadb.HttpClient()`)
    * **Characteristics**:
      * Connects over network HTTP / REST sockets to an independent standalone ChromaDB server process:
        `client = chromadb.HttpClient(host="localhost", port=8000)`.
      * Running the standalone server:
        `chroma run --path ./my_chroma_data --port 8000`.
    * **Why This is Mandatory for Web Applications**:
      * Multi-worker web frameworks (FastAPI, Gunicorn, Django, Celery) run multiple worker processes concurrently.
      * If multiple processes attempt to access the embedded SQLite database on disk simultaneously, SQLite throws **database lock errors**!
      * `HttpClient` decouples web workers from database concurrency, routing all writes through a single managed server daemon.
    """)

st.markdown("### Client Modalities Comparative Matrix")
client_comp_df = pd.DataFrame({
    "Client Type": ["EphemeralClient()", "PersistentClient(path=...)", "HttpClient(host=..., port=...)"],
    "Storage Location": ["Volatile Memory (RAM)", "Local Filesystem Disk Directory", "Remote Server Daemon on Network"],
    "Process Model": ["In-Process (Embedded)", "In-Process (Embedded)", "Client-Server (Decoupled over HTTP)"],
    "Data Lifespan": ["Process lifetime only", "Permanent (Survives restarts)", "Permanent (Managed on server)"],
    "Concurrency Limit": ["Single thread / process", "Single writer (SQLite lock)", "Multi-client concurrent web requests"],
    "Production Readiness": ["Testing / CI only", "CLI scripts / Desktop apps", "Production Web Services & Kubernetes"]
})
st.dataframe(client_comp_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Complete Architectural Client Flowchart
st.header("2. Complete Client Architecture Flowchart", divider="rainbow")

st.markdown("""
The diagram below illustrates how client calls route either to in-process embedded storage or over HTTP sockets:
""")

client_flow_dot = """
digraph ChromaClientArchitectures {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    App [label="Python Application\\nFastAPI / Script / Streamlit", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

    subgraph cluster_embedded {
        label="Embedded In-Process Client Modes"; style="dashed"; color="#059669";
        Ephemeral [label="EphemeralClient()\\nIn-Memory volatile RAM", fillcolor="#D1FAE5"];
        Persistent [label="PersistentClient(path='./data')\\nLocal SQLite + HNSW files", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
    }

    subgraph cluster_server {
        label="Client-Server Remote Mode"; style="dashed"; color="#7C3AED";
        HttpClient [label="HttpClient(host, port)\\nNetwork REST / gRPC", fillcolor="#EDE9FE"];
        ChromaServer [label="Standalone Chroma Daemon\\n(chroma run --path ./data)\\nManages Concurrency & WAL", fillcolor="#DDD6FE", color="#7C3AED", style="filled,bold"];
        DiskStorage [label="Persistent Storage Volume\\nchroma.sqlite3 + segment dirs", fillcolor="#C4B5FD"];
    }

    App -> Ephemeral [label="Testing"];
    App -> Persistent [label="Local CLI"];
    App -> HttpClient [label="Web App"];
    HttpClient -> ChromaServer [label="HTTP Socket"];
    ChromaServer -> DiskStorage;
}
"""
st.graphviz_chart(client_flow_dot, use_container_width=False)

st.divider()

# Section 3: Deep Dive into On-Disk Directory Anatomy
st.header("3. What Happens on Disk: Anatomy of `persist_directory`", divider="rainbow")

st.markdown("""
When you initialize `chromadb.PersistentClient(path="./my_data")`, Chroma creates a structured local folder:
""")

col_disk1, col_disk2 = col_disk1, col_disk2 = st.columns([1, 1])

with col_disk1:
    st.markdown("""
    ### Directory Layout
    ```text
    my_data/
    ├── chroma.sqlite3
    └── <collection-uuid>/
        ├── data_level0.bin
        ├── header.bin
        ├── hex_data.bin
        └── length.bin
    ```
    * **`chroma.sqlite3`**:
      * The central relational database.
      * Contains tables for `collections`, `segments`, `embeddings`, and metadata JSON key-values.
      * Handles transaction durability via SQLite Write-Ahead Log (`chroma.sqlite3-wal`).
    * **`<collection-uuid>/` Directory**:
      * Contains the binary HNSW index segment files serialized directly from C++ HNSWlib.
      * `data_level0.bin`: The dense base layer of the proximity graph storing vector coordinates and nearest-neighbor links.
      * `header.bin`: Graph index configuration parameters (`M`, `efConstruction`, metric type).
    """)

with col_disk2:
    st.markdown("""
    ### Client Configuration with `Settings`
    You can fine-tune client behavior using the `Settings` class:
    ```python
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path="./production_data",
        settings=Settings(
            anonymized_telemetry=False, # Disable phone-home
            allow_reset=True            # Allow client.reset()
        )
    )
    ```
    * **`anonymized_telemetry`**: Set to `False` in enterprise networks to block outbound telemetry pings.
    * **`allow_reset`**: If `True`, enables `client.reset()`, which wipes the entire database (use with extreme caution in production!).
    """)

st.divider()

# Section 4: Interactive Chroma Client Simulator
st.header("4. Interactive Chroma Client Configurator & Persistence Simulator", divider="rainbow")

st.markdown("""
Configure client parameters, inspect the generated Python initialization snippet, and test 
how data persists across simulated process reboots:
""")

col_cfg_in, col_cfg_out = st.columns([1, 1])

with col_cfg_in:
    st.subheader("Client Modality Selection")
    sel_mode = st.radio("Client Type", ["1. Ephemeral Client (In-Memory)", "2. Persistent Client (Local Disk)", "3. Remote HTTP Client (Server Mode)"])
    
    if "Persistent" in sel_mode:
        cfg_path = st.text_input("Persistence Directory Path", value="./chroma_storage")
    elif "HTTP" in sel_mode:
        cfg_host = st.text_input("Server Host", value="localhost")
        cfg_port = st.number_input("Server Port", value=8000)
        cfg_auth = st.checkbox("Enable Token Authentication Header", value=False)
        
    cfg_telemetry = st.checkbox("Disable Anonymous Telemetry", value=True)

with col_cfg_out:
    st.subheader("Generated Python Code & Architecture Review")
    
    if "Ephemeral" in sel_mode:
        code_snip = f"""import chromadb
from chromadb.config import Settings

# Ephemeral In-Memory Client (Zero Disk Footprint)
client = chromadb.EphemeralClient(
    settings=Settings(anonymized_telemetry={not cfg_telemetry})
)

collection = client.create_collection("temporary_notes")
collection.add(documents=["Ephemeral test"], ids=["id_1"])"""
        st.code(code_snip, language="python")
        st.warning("⚠️ **Volatile Memory Warning**: All data in this client will be wiped when the Python script exits.")
        
    elif "Persistent" in sel_mode:
        code_snip = f"""import chromadb
from chromadb.config import Settings

# Persistent Embedded Client (Disk Storage)
client = chromadb.PersistentClient(
    path="{cfg_path}",
    settings=Settings(
        anonymized_telemetry={not cfg_telemetry},
        allow_reset=False
    )
)

collection = client.get_or_create_collection("production_kb")
collection.add(documents=["Persistent contract"], ids=["doc_42"])"""
        st.code(code_snip, language="python")
        st.success(f"✅ **Persistent Storage**: Data permanently written to `{cfg_path}/chroma.sqlite3` and segment folders.")
        
    else: # HTTP
        auth_header = '{"X-Chroma-Token": "secret_token_123"}' if cfg_auth else "None"
        code_snip = f"""import chromadb
from chromadb.config import Settings

# Remote Client-Server Mode (Multi-Process Safe)
client = chromadb.HttpClient(
    host="{cfg_host}",
    port={int(cfg_port)},
    headers={auth_header},
    settings=Settings(anonymized_telemetry={not cfg_telemetry})
)

collection = client.get_or_create_collection("enterprise_kb")"""
        st.code(code_snip, language="python")
        st.info("💡 **Production Web Recommendation**: Required for multi-worker FastAPI / Gunicorn applications.")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Ephemeral vs. Persistent Client Demonstration", divider="rainbow")

st.markdown("""
The following complete Python program demonstrates the contrast between an **Ephemeral Client** 
and a **Persistent Client**, testing data persistence and client reset operations:
""")

code_client_demo = '''import chromadb
from chromadb.config import Settings
import tempfile
import shutil

# 1. Ephemeral Client Demonstration (Volatile)
print("=== 1. Testing EphemeralClient (In-Memory) ===")
ephemeral_client = chromadb.EphemeralClient(
    settings=Settings(anonymized_telemetry=False)
)

col_eph = ephemeral_client.create_collection("scratchpad")
col_eph.add(documents=["This document exists only in volatile RAM."], ids=["eph_1"])
print(f"Ephemeral Collection Record Count: {col_eph.count()}")

# 2. Persistent Client Demonstration (Local Filesystem)
print("\\n=== 2. Testing PersistentClient (Disk Persistence) ===")
temp_dir = tempfile.mkdtemp(prefix="chroma_test_")

try:
    # First Session: Write to disk
    client_sess1 = chromadb.PersistentClient(
        path=temp_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    col_p1 = client_sess1.create_collection("customer_contracts")
    col_p1.add(documents=["Master Services Agreement executed 2024."], ids=["c_01"])
    print(f"Session 1: Wrote record 'c_01' to {temp_dir}")
    del client_sess1 # Simulate process shutdown

    # Second Session: Re-open client on same directory
    client_sess2 = chromadb.PersistentClient(
        path=temp_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    col_p2 = client_sess2.get_collection("customer_contracts")
    print(f"Session 2: Re-opened database! Verified Count: {col_p2.count()}")
    retrieved = col_p2.get(ids=["c_01"])
    print(f"Session 2: Retrieved Document Content: '{retrieved['documents'][0]}'")

finally:
    # Cleanup temporary test directory
    shutil.rmtree(temp_dir)
    print("\\nTemporary test directory cleaned up successfully.")
'''

st.code(code_client_demo, language="python")

if st.button("🚀 Execute Client Demonstration Script", type="primary"):
    exec_scope = {}
    exec(code_client_demo, globals(), exec_scope)
    
    c_eph_cnt = exec_scope["col_eph"].count()
    retrieved_doc = exec_scope["retrieved"]["documents"][0]
    
    st.success("Client demonstration executed and persistence verified across simulated sessions!")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown(f"**Ephemeral Client Count:** `{c_eph_cnt}`")
    with col_res2:
        st.markdown(f"**Persistent Client Re-hydrated Content:** `{retrieved_doc}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. In a multi-worker production web application (FastAPI with 4 Gunicorn workers), why does `PersistentClient` fail with database lock errors?"""
a1 = """* **The Concurrency Problem**:
  * `PersistentClient` embeds the SQLite engine directly into the application process.
  * When Gunicorn forks 4 worker processes, each worker has its own independent connection to the same `chroma.sqlite3` file on disk.
  * SQLite uses file-level locking: when Worker A acquires an exclusive write lock to insert vectors, Workers B, C, and D are blocked.
  * Under concurrent web traffic, workers queue up, exceed the SQLite busy timeout (typically 5 seconds), and crash with `sqlite3.OperationalError: database is locked`.
* **The Solution**:
  * Run ChromaDB as an independent standalone daemon: `chroma run --path /data --port 8000`.
  * Configure all FastAPI workers to connect via `chromadb.HttpClient(host="...", port=8000)`. The Chroma daemon manages concurrency internally."""

q2 = """2. What files are created on disk inside ChromaDB's `persist_directory`, and what is their role?"""
a2 = """* **`chroma.sqlite3`**:
  * Stores the system catalog: collections, UUID mappings, raw text document chunks, and JSON metadata payloads.
  * Ensures ACID durability on disk.
* **`chroma.sqlite3-wal` & `chroma.sqlite3-shm`**:
  * SQLite Write-Ahead Log and shared memory index files supporting concurrent read transactions during active writes.
* **`<collection-uuid>/` Folders**:
  * Contains the serialized C++ HNSWlib index segments.
  * `data_level0.bin`: The base layer of the HNSW graph containing vector coordinates and nearest neighbor edge lists.
  * `header.bin`: Index hyperparameters (`M`, `efConstruction`, distance metric type)."""

q3 = """3. How does `client.get_or_create_collection()` prevent race conditions during service startup?"""
a3 = """* In microservices, multiple container replicas often start up simultaneously and attempt to initialize the same collection.
* If a service uses `create_collection()`, the first replica succeeds, but subsequent replicas crash with `UniqueConstraintError: Collection already exists`.
* `get_or_create_collection()` executes an atomic check: if the collection exists, it loads and returns it; if missing, it creates it.
* Guarantees safe, idempotent startup across distributed application pods."""

q4 = """4. How can you secure a ChromaDB server when running in Client-Server (`HttpClient`) mode?"""
a4 = """* **1. Token-Based Authentication**:
  * Server: Set `CHROMA_SERVER_AUTH_PROVIDER="chromadb.auth.token.TokenAuthServerProvider"` and configure valid API tokens.
  * Client: Pass `headers={"X-Chroma-Token": "secret_api_key"}` in `HttpClient`.
* **2. Network Boundary Security**:
  * Bind ChromaDB strictly to internal private VPC subnets (`10.0.x.x`) or Kubernetes cluster-internal DNS (`chroma.default.svc.cluster.local`) rather than exposing port 8000 to the public internet.
* **3. TLS Encryption**:
  * Put an NGINX or Envoy reverse proxy in front of ChromaDB to terminate HTTPS/TLS traffic."""

q5 = """5. What does `client.reset()` do, and why should it be disabled in production?"""
a5 = """* `client.reset()` completely wipes the database: it drops all collections, deletes all indexed vectors, and purges all underlying SQLite tables.
* If enabled in production, a bug in an administrative script or an unauthorized API call could permanently destroy enterprise knowledge bases.
* By default, modern ChromaDB disables `reset()`. It can only be executed if the client was explicitly instantiated with `Settings(allow_reset=True)`."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Chroma Client Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: Storing Persistent SQLite Files on Network NFS or AWS EFS Shares**  
> Running `chromadb.PersistentClient` with a `path` pointing to a mounted Network File System (NFS, SMB, or AWS EFS) 
> causes frequent SQLite database corruption and severe latency spikes. SQLite requires POSIX byte-range locking, 
> which is notoriously buggy over network file systems. Always store persistent SQLite files on local NVMe or EBS block storage.

> [!WARNING]
> **Trap 2: Hardcoding Relative Paths in Docker Containers**  
> Initializing `PersistentClient(path="./data")` inside a containerized application without an absolute path 
> defaults to the container's working directory. If the container restarts or the process is launched from a different 
> directory, Chroma creates a fresh empty database in the new directory, making existing data appear "lost"! 
> Always use absolute paths (e.g. `path="/var/lib/chroma_data"`).

> [!WARNING]
> **Trap 3: Re-instantiating the Client Inside High-Frequency Query Loops**  
> Calling `client = chromadb.PersistentClient(...)` inside a web request handler opens and closes SQLite file descriptors 
> and re-reads index headers on every single search query. Always instantiate the client as a global singleton or 
> inject it via application dependency injection.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: When to Use `EphemeralClient()`?"):
        st.markdown("""
        **`EphemeralClient()`**:  
        Runs 100% in volatile RAM; data is destroyed on process exit. Ideal for fast CI/CD unit testing and disposable scratchpads.
        """)
    with st.expander("🗂️ Flashcard: When to Use `HttpClient()`?"):
        st.markdown("""
        **`HttpClient()`**:  
        Connects over network HTTP to a standalone Chroma server; mandatory for multi-worker web applications to avoid SQLite locks.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is `chroma.sqlite3`?"):
        st.markdown("""
        **`chroma.sqlite3`**:  
        The persistent SQLite database storing collection metadata, record IDs, and raw text document payloads on disk.
        """)
    with st.expander("🗂️ Flashcard: What is `get_or_create_collection`?"):
        st.markdown("""
        **Idempotent Startup**:  
        Safely loads an existing collection or creates it if missing, preventing race conditions across multi-pod startups.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural differences between Ephemeral, Persistent, and HTTP clients.", value=True)
chk2 = st.checkbox("I know why multi-worker web applications must use `HttpClient` to prevent SQLite lock errors.")
chk3 = st.checkbox("I understand the on-disk directory structure of `persist_directory` (`chroma.sqlite3` and segment folders).")
chk4 = st.checkbox("I know how to configure client settings using the `Settings` class.")
chk5 = st.checkbox("I know why `get_or_create_collection()` is preferred over `create_collection()` in production.")
chk6 = st.checkbox("I understand why persistent Chroma databases must never be hosted on network NFS/EFS shares.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Chroma Client and are ready for Chapter 6.4: Collections!")
