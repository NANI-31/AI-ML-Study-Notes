import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="2.1 Creating a Chroma Client - ChromaDB Curriculum",
    page_icon="🔌",
    layout="wide"
)

# Title & Overview
st.title("2.1 Creating a Chroma Client: In-Memory, Persistent, HTTP & Settings Lifecycle")
st.caption("Chapter 2: ChromaDB Basics | Domain: 5. ChromaDB")
st.markdown("""
The **Client** is the foundational entrypoint to every operation in ChromaDB. 

Whether you are performing quick experimentation in a Jupyter notebook, persisting a local knowledge base 
on an SSD, or querying a multi-node cluster across an enterprise Kubernetes mesh, your architecture begins with 
instantiating the correct Client type.

ChromaDB provides three primary Client modes:
1. **`EphemeralClient`**: 100% in-memory; ultra-fast; automatically destroyed when the Python process terminates. Ideal for unit tests and CI/CD pipelines.
2. **`PersistentClient`**: Embedded file-backed client; writes SQLite tables and HNSW binary graphs directly to disk (`./chroma_data`). Ideal for local RAG and desktop tools.
3. **`HttpClient`**: Connects over network HTTP/gRPC to a remote standalone ChromaDB server container. Essential for multi-process production web architectures.

In this lesson, we explore client topologies, analyze the `Settings` configuration engine, review connection lifecycles, 
and test an interactive Client Configuration Arena.
""")

st.divider()

# Section 1: The Three Client Topologies
st.header("1. The Three Client Topologies", divider="rainbow")

col_top1, col_top2 = col_top1, col_top2 = st.columns([1.1, 0.9])

with col_top1:
    st.markdown("""
    ### Architectural Differences Across Client Modes
    
    * **1. Ephemeral Client (`chromadb.EphemeralClient()`)**:
      * *Where Data Lives*: System RAM only.
      * *Persistence*: Zero disk writes; data vanishes on process exit.
      * *Best For*: Unit tests, temporary scratchpad calculations, CI/CD runners.
    
    * **2. Persistent Client (`chromadb.PersistentClient(path="./vault")`)**:
      * *Where Data Lives*: Local filesystem directory containing `chroma.sqlite3` and graph `.bin` files.
      * *Persistence*: Full ACID durability across application restarts.
      * *Concurrency Limitation*: Single-process only. Multi-process writes cause SQLite database lock conflicts.
      * *Best For*: Single-user desktop apps, CLI tools, offline research notebooks.
    
    * **3. HTTP Client (`chromadb.HttpClient(host="...", port=8000)`)**:
      * *Where Data Lives*: Remote ChromaDB container or cloud cluster.
      * *Persistence*: Managed by the remote server process.
      * *Concurrency*: High concurrency; handles thousands of simultaneous web requests across Uvicorn/Gunicorn workers.
      * *Best For*: Production microservices, web apps, and distributed RAG gateways.
    """)

with col_top2:
    st.markdown("### Client Mode Decision Matrix")
    client_dot = """
    digraph ClientMatrix {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Start [label="Select ChromaDB Client Mode", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Q1 [label="Does data need to survive\\nprocess restart?", shape=diamond, fillcolor="#FEF3C7"];

        Ephemeral [label="chromadb.EphemeralClient()\\n(In-Memory / Fast / Tests)", fillcolor="#D1FAE5", color="#059669"];

        Q2 [label="Are multiple web worker processes\\naccessing the database?", shape=diamond, fillcolor="#EDE9FE"];

        Persistent [label="chromadb.PersistentClient(path)\\n(Local Disk / Single Process)", fillcolor="#D1FAE5", color="#059669"];

        HTTP [label="chromadb.HttpClient(host, port)\\n(Production Microservices / Docker)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        Start -> Q1;
        Q1 -> Ephemeral [label="No (Temp)"];
        Q1 -> Q2 [label="Yes (Persist)"];
        Q2 -> Persistent [label="Single Process"];
        Q2 -> HTTP [label="Multi-Worker"];
    }
    """
    st.graphviz_chart(client_dot, use_container_width=False)

st.divider()

# Section 2: Client Instantiation Code Examples
st.header("2. Client Instantiation Code Examples", divider="rainbow")

col_code_a, col_code_b = col_code_a, col_code_b = st.columns([1, 1])

with col_code_a:
    st.markdown("### 1. Ephemeral & Persistent Clients")
    st.code("""import chromadb
from chromadb.config import Settings

# --- Pattern A: Ephemeral In-Memory Client ---
mem_client = chromadb.EphemeralClient(
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True  # Allows client.reset()
    )
)

# --- Pattern B: Persistent Disk Client ---
disk_client = chromadb.PersistentClient(
    path="./production_vector_store",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=False  # Safety lock against accidental wipes
    )
)

print("Persistent database root:", disk_client.get_version())
""", language="python")

with col_code_b:
    st.markdown("### 2. Remote HTTP Client with Authentication")
    st.code("""import chromadb
from chromadb.config import Settings

# --- Pattern C: Production HTTP Client ---
remote_client = chromadb.HttpClient(
    host="chroma-cluster.internal.net",
    port=8000,
    ssl=True,  # Enforce TLS encryption in transit
    headers={
        "X-Chroma-Token": "sk-production-auth-key-8912"
    },
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials="sk-production-auth-key-8912",
        anonymized_telemetry=False
    )
)

# Verify heartbeat
heartbeat = remote_client.heartbeat()
print("Remote cluster nanosecond heartbeat:", heartbeat)
""", language="python")

st.divider()

# Section 3: Client Configuration Deep-Dive: The `Settings` Engine
st.header("3. Client Configuration Deep-Dive: The `Settings` Engine", divider="rainbow")

st.markdown("""
The `Settings` object controls global behavior, telemetry, authentication, and safety guardrails across ChromaDB:
""")

SETTINGS_MATRIX = [
    {
        "Setting Parameter": "`anonymized_telemetry`",
        "Type": "bool (Default: True)",
        "Description": "Controls whether anonymous product usage data is sent to Chroma. Set to `False` in compliance-regulated enterprise environments."
    },
    {
        "Setting Parameter": "`allow_reset`",
        "Type": "bool (Default: False)",
        "Description": "Controls whether `client.reset()` can be called to wipe all collections and database state. Must remain `False` in production to prevent catastrophic data loss."
    },
    {
        "Setting Parameter": "`chroma_client_auth_provider`",
        "Type": "str (Optional)",
        "Description": "Configures authentication provider for remote clusters (e.g. `chromadb.auth.token.TokenAuthClientProvider`)."
    },
    {
        "Setting Parameter": "`chroma_client_auth_credentials`",
        "Type": "str (Optional)",
        "Description": "Bearer token or API key credentials supplied to the remote server."
    },
    {
        "Setting Parameter": "`is_persistent`",
        "Type": "bool (Read-Only)",
        "Description": "Internal boolean indicating whether the client persists state to disk."
    }
]

st.dataframe(pd.DataFrame(SETTINGS_MATRIX), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Client Configuration & Mode Arena
st.header("4. Interactive Workbench: Client Configuration & Mode Arena", divider="rainbow")

st.markdown("""
Simulate instantiating different ChromaDB client modes with custom configuration settings. 
Inspect the resulting generated connection code, runtime memory behavior, and production suitability.
""")

col_cfg_mode, col_cfg_toggles = col_cfg_mode, col_cfg_toggles = st.columns(2)

with col_cfg_mode:
    st.subheader("1. Client Mode Selection")
    chosen_mode = st.radio(
        "Choose Client Type:",
        [
            "EphemeralClient (In-Memory Testing)",
            "PersistentClient (Embedded Disk Storage)",
            "HttpClient (Remote Standalone Cluster)"
        ]
    )
    
    if "Persistent" in chosen_mode:
        target_path = st.text_input("Local Disk Path:", value="./my_knowledge_base")
    elif "Http" in chosen_mode:
        target_host = st.text_input("Server Host / URL:", value="chroma.internal.vpc")
        target_port = st.number_input("Server Port:", value=8000)

with col_cfg_toggles:
    st.subheader("2. Settings Guardrails")
    toggle_telemetry = st.checkbox("Disable Anonymized Telemetry (`anonymized_telemetry=False`)", value=True)
    toggle_allow_reset = st.checkbox("Allow Database Reset (`allow_reset=True`)", value=False)
    toggle_ssl = st.checkbox("Enforce TLS / HTTPS (`ssl=True`)", value=True)

st.divider()

col_code_gen, col_verdict = col_code_gen, col_verdict = st.columns([1.1, 0.9])

with col_code_gen:
    st.subheader("Generated Python Instantiation Code")
    
    if "Ephemeral" in chosen_mode:
        code_str = f"""import chromadb
from chromadb.config import Settings

client = chromadb.EphemeralClient(
    settings=Settings(
        anonymized_telemetry={not toggle_telemetry},
        allow_reset={toggle_allow_reset}
    )
)"""
    elif "Persistent" in chosen_mode:
        code_str = f"""import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="{target_path}",
    settings=Settings(
        anonymized_telemetry={not toggle_telemetry},
        allow_reset={toggle_allow_reset}
    )
)"""
    else:
        code_str = f"""import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="{target_host}",
    port={target_port},
    ssl={toggle_ssl},
    settings=Settings(
        anonymized_telemetry={not toggle_telemetry},
        allow_reset={toggle_allow_reset}
    )
)"""
    st.code(code_str, language="python")

with col_verdict:
    st.subheader("Production Architectural Assessment")
    
    if toggle_allow_reset:
        st.error("⚠️ **PRODUCTION SAFETY WARNING**: `allow_reset=True` is enabled! A single errant call to `client.reset()` in application code will permanently delete all indexed enterprise data!")
    else:
        st.success("🔒 **Safety Guardrail Active**: `allow_reset=False` protects data from accidental deletion.")
        
    if "Ephemeral" in chosen_mode:
        st.info("💡 **In-Memory Volatility**: Data will vanish as soon as the script terminates. Excellent for automated testing.")
    elif "Persistent" in chosen_mode:
        st.info("💡 **Single-Process Durability**: Data is saved to disk. Remember: do not share this directory across multiple concurrent web worker processes.")
    else:
        st.success("🏆 **Enterprise Grade**: Scalable, multi-process compatible, and ready for containerized Kubernetes workloads.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What happens if two separate Python processes run `PersistentClient` on the same directory?"):
    st.markdown("""
    **Answer**:
    **Concurrency Conflict (`sqlite3.OperationalError: database is locked`)**:
    * ChromaDB's persistent client relies on SQLite for metadata storage.
    * SQLite uses file-level locking during write transactions.
    * When Process A begins inserting documents, it locks `chroma.sqlite3`. If Process B attempts to write simultaneously, SQLite raises a locking exception.
    * **Remedy**: `PersistentClient` must only be used in **single-process architectures**. In multi-worker environments (e.g. FastAPI with 4 Uvicorn workers), run Chroma as a standalone server and connect via `HttpClient`.
    """)

with st.expander("Q2: Why is `allow_reset=False` considered a critical production setting?"):
    st.markdown("""
    **Answer**:
    * Calling `client.reset()` deletes all collections, drops all SQLite tables, and purges all HNSW vector graph segments from disk.
    * If `allow_reset` is left at `True` in production, a malicious user or rogue testing script that calls `/api/v1/reset` will instantaneously wipe the entire company knowledge base!
    * Setting `allow_reset=False` makes `client.reset()` raise an immediate `ValueError: Reset is not allowed by configuration`.
    """)

with st.expander("Q3: How does `client.heartbeat()` work, and what unit of time does it return?"):
    st.markdown("""
    **Answer**:
    * `client.heartbeat()` sends a lightweight ping to verify that the ChromaDB engine is alive and responding.
    * It returns a **64-bit integer timestamp measured in nanoseconds** (e.g. `1728800000000000000`).
    * In Kubernetes and Docker architectures, `/api/v1/heartbeat` is the standard endpoint configured for container **Liveness and Readiness probes**.
    """)

with st.expander("Q4: Is the ChromaDB Client thread-safe within a single Python process?"):
    st.markdown("""
    **Answer**:
    * **`HttpClient`**: Fully thread-safe. Uses standard HTTP connection pooling (via `urllib3` / `requests`) allowing multiple background threads to issue concurrent search queries.
    * **`PersistentClient` / `EphemeralClient`**: Query operations (reads) are generally thread-safe, but concurrent write operations (`add`, `upsert`, `delete`) across background threads can trigger SQLite concurrency contention. Writes should be serialized or routed through an ingestion queue.
    """)

with st.expander("Q5: What is the recommended pattern for sharing a Chroma client in FastAPI?"):
    st.markdown("""
    **Answer**:
    Use FastAPI's **Lifespan Context Manager** or **Dependency Injection Singleton**:
    ```python
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    import chromadb

    db_client = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global db_client
        # Initialize client once at startup
        db_client = chromadb.HttpClient(host="chroma-server", port=8000)
        yield
        # Clean up connections at shutdown

    app = FastAPI(lifespan=lifespan)
    ```
    This prevents creating new client connection overhead on every incoming HTTP request.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Instantiating `PersistentClient` Inside Web Request Handlers**  
    Calling `client = chromadb.PersistentClient()` inside a FastAPI route re-opens SQLite files on every single HTTP request, causing massive latency and file descriptor leaks. Instantiate as a singleton!
    """)
    st.warning("""
    **Trap 2: Leaving `allow_reset=True` in Production**  
    Allowing database resets in production invites catastrophic accidental database wipes from test scripts. Lock `allow_reset=False`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Forgetting Authentication Headers in `HttpClient`**  
    Connecting to a secured remote Chroma server without configuring `headers` or `Settings` auth credentials causes immediate HTTP 401 Unauthorized failures.
    """)
    st.warning("""
    **Trap 4: Missing Trailing Slash or Port in `HttpClient`**  
    Passing `http://chroma:8000` into `host` while leaving `port=8000` causes duplicate port string parsing bugs. Pass hostname to `host` and integer to `port`.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: When to use EphemeralClient?"):
        st.markdown("""
        **In-Memory Testing**:  
        Zero disk writes; vanishes upon process exit. Perfect for fast unit tests.
        """)
    with st.expander("🗂️ Flashcard: When to use HttpClient?"):
        st.markdown("""
        **Multi-Process Production**:  
        Connects remotely to a standalone Chroma container, supporting concurrent web workers.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What does allow_reset do?"):
        st.markdown("""
        **Destructive Reset**:  
        Controls whether `client.reset()` can wipe the entire database. Must be False in prod.
        """)
    with st.expander("🗂️ Flashcard: What is client.heartbeat()?"):
        st.markdown("""
        **Liveness Probe**:  
        Returns nanosecond timestamp confirming database engine health and availability.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural differences between EphemeralClient, PersistentClient, and HttpClient.", value=True)
chk2 = st.checkbox("I know why PersistentClient must not be used across multiple concurrent web worker processes.", value=True)
chk3 = st.checkbox("I understand how the `Settings` object controls telemetry, authentication, and reset permissions.", value=True)
chk4 = st.checkbox("I know how to instantiate an authenticated remote `HttpClient` with custom headers.", value=True)
chk5 = st.checkbox("I can implement singleton client dependency injection in web frameworks like FastAPI.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 2.2: Collections Management in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 2.1: Creating a Chroma Client and are ready for Chapter 2.2: Collections!")
