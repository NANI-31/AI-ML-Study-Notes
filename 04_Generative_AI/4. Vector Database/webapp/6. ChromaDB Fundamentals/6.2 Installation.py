import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import sys
import sqlite3

st.set_page_config(
    page_title="6.2 Installation - Vector DB Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title & Overview
st.title("6.2 ChromaDB Installation: Environments, C++ Wheels & SQLite Overrides")
st.caption("Chapter 6: ChromaDB Fundamentals | Domain: 4. Vector Database")
st.markdown("""
Installing ChromaDB is typically as simple as `pip install chromadb`. However, in enterprise environments—spanning 
Windows development workstations, air-gapped corporate subnets, and legacy Linux production servers—installing 
ChromaDB requires navigating critical low-level binary dependencies:
* **The C++ Compiler Prerequisite**: ChromaDB's core vector index (`hnswlib`) is authored in C++ and requires pre-built binary wheels or a local C++ compiler.
* **The SQLite Version Constraint**: ChromaDB strictly requires **SQLite >= 3.35.0** (for `RETURNING` clause support), which frequently fails on older Linux distributions (CentOS 7, Ubuntu 20.04).
* **Automated Embedding Weights**: Chroma downloads default ONNX weights on first initialization, which can fail without proxy configuration.

In this lesson, we explore robust Python environment setup, resolve the infamous SQLite version trap, verify installation 
integrity, and configure enterprise telemetry settings.
""")

st.divider()

# Section 1: Standard Installation & System Prerequisites
st.header("1. Standard Installation & System Prerequisites", divider="rainbow")

col_inst1, col_inst2 = col_inst1, col_inst2 = st.columns([1, 1])

with col_inst1:
    st.markdown("""
    ### 1. Step-by-Step Installation Guide
    * **Step 1: Create an Isolated Virtual Environment**:
      ```bash
      # Using Python standard venv
      python -m venv .venv
      # Windows PowerShell
      .venv\\Scripts\\Activate.ps1
      # Linux / macOS
      source .venv/bin/activate
      ```
    * **Step 2: Upgrade Pip & Install ChromaDB**:
      ```bash
      pip install --upgrade pip
      pip install chromadb
      ```
    * **Step 3: Verify in Python**:
      ```python
      import chromadb
      print(chromadb.__version__)
      ```
    """)

with col_inst2:
    st.markdown("""
    ### 2. Under-the-Hood System Dependencies
    * **C++ Compiler (for HNSWlib)**:
      * On Windows: Pre-compiled wheels are standard. If compiling from source, **Visual C++ 2019+ Build Tools** is required.
      * On Linux: `gcc` / `g++` and `python3-dev`.
      * On macOS: Xcode Command Line Tools (`xcode-select --install`).
    * **The SQLite Dependency**:
      * SQLite version must be **>= 3.35.0** (released March 2021).
      * If your system SQLite is older, ChromaDB will throw a fatal `RuntimeError` on import.
    """)

st.markdown("### The Dependency Stack Flowchart")
dep_flow_dot = """
digraph ChromaDependencies {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    PythonEnv [label="Python Environment\\nPython 3.9 - 3.12", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
    
    subgraph cluster_c_deps {
        label="Binary C / C++ Libraries"; style="dashed"; color="#DC2626";
        HNSWLibWheel [label="C++ HNSWlib Wheel\\nSIMD AVX-512 Binaries", fillcolor="#FEE2E2"];
        SQLiteLib [label="SQLite >= 3.35.0\\nJSON & RETURNING Clause", fillcolor="#FECACA", color="#DC2626", style="filled,bold"];
    }

    subgraph cluster_python_deps {
        label="Core Python Packages"; style="dashed"; color="#059669";
        ONNXRuntime [label="onnxruntime (CPU)\\nDefault MiniLM Embeddings", fillcolor="#D1FAE5"];
        Pydantic [label="pydantic v2\\nSchema Validation", fillcolor="#A7F3D0"];
        ChromaCore [label="chromadb Core Engine", fillcolor="#6EE7B7", color="#059669", style="filled,bold"];
    }

    PythonEnv -> HNSWLibWheel;
    PythonEnv -> SQLiteLib;
    HNSWLibWheel -> ChromaCore;
    SQLiteLib -> ChromaCore;
    ONNXRuntime -> ChromaCore;
    Pydantic -> ChromaCore;
}
"""
st.graphviz_chart(dep_flow_dot, use_container_width=False)

st.divider()

# Section 2: The Infamous SQLite Version Trap & The `pysqlite3` Fix
st.header("2. The Infamous SQLite Version Trap & The `pysqlite3` Fix", divider="rainbow")

st.markdown("""
One of the most frequent errors encountered by developers deploying ChromaDB on Linux servers (e.g. AWS Linux, Debian, older Ubuntu) is:
```text
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
Please visit https://docs.trychroma.com/troubleshooting#sqlite to learn how to upgrade.
```

Because Python's standard library `sqlite3` module is compiled against the operating system's system-level SQLite library, 
upgrading Python does not automatically upgrade SQLite!
""")

col_fix1, col_fix2 = col_fix1, col_fix2 = st.columns([1, 1])

with col_fix1:
    st.markdown("""
    ### The Root Cause
    * Older Linux distributions package SQLite 3.31 or 3.32.
    * ChromaDB relies on SQLite's `RETURNING` clause (introduced in 3.35.0) for atomic upsert and delete operations.
    * Standard `pip install sqlite3` does not exist because `sqlite3` is a C extension built into the Python runtime.
    """)

with col_fix2:
    st.markdown("""
    ### The Universal Production Fix: `pysqlite3-binary`
    * **Step 1: Install `pysqlite3-binary`**:
      ```bash
      pip install pysqlite3-binary
      ```
    * **Step 2: Add Monkey-Patch at the Very Top of your Entrypoint Script**:
      ```python
      # Must be the FIRST lines before importing chromadb!
      __import__('pysqlite3')
      import sys
      sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

      # Now safe to import chromadb
      import chromadb
      ```
    * This dynamically swaps the system SQLite module with the bundled modern SQLite 3.40+ binary in user space!
    """)

st.divider()

# Section 3: Interactive System Diagnostic & Installation Generator
st.header("3. Interactive Environment Diagnostic & Installer Generator", divider="rainbow")

st.markdown("""
Inspect your current Python environment's SQLite version and generate tailored installation scripts 
for your specific operating system:
""")

col_diag1, col_diag2 = col_diag1, col_diag2 = st.columns([1, 1])

with col_diag1:
    st.subheader("Live System Runtime Audit")
    py_ver = sys.version.split(" ")[0]
    sql_ver = sqlite3.sqlite_version
    
    st.metric("Current Python Version", py_ver)
    
    # Check SQLite version >= 3.35.0
    sql_parts = [int(p) for p in sql_ver.split(".")[:3]]
    is_sql_valid = (sql_parts[0] > 3) or (sql_parts[0] == 3 and sql_parts[1] >= 35)
    
    if is_sql_valid:
        st.metric("System SQLite Version", sql_ver, "✅ Compatible (>= 3.35.0)")
        st.success("Your system has a modern SQLite version! Standard `pip install chromadb` will work cleanly.")
    else:
        st.metric("System SQLite Version", sql_ver, "❌ Incompatible (< 3.35.0)")
        st.error("Your SQLite version is too old for ChromaDB! You must apply the `pysqlite3-binary` override.")

with col_diag2:
    st.subheader("Custom Installation Script Generator")
    target_os = st.selectbox("Target Operating System", ["Windows 10/11 (PowerShell)", "macOS (Apple Silicon M-Series / Intel)", "Linux (Ubuntu / Debian / AWS EC2)", "Docker Container (Dockerfile)"])
    disable_telemetry = st.checkbox("Disable Anonymous Telemetry (Enterprise Policy)", value=True)
    
    if "Windows" in target_os:
        script = f"""# 1. Ensure Python 3.9 - 3.12 is installed
python -m venv .venv
.venv\\Scripts\\Activate.ps1

# 2. Upgrade pip and install ChromaDB
pip install --upgrade pip
pip install chromadb

# 3. Disable telemetry (Optional)
{"$env:ANONYMIZED_TELEMETRY='False'" if disable_telemetry else "# Telemetry enabled"}"""
    elif "macOS" in target_os:
        script = f"""# 1. Install command line tools if needed
xcode-select --install

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install ChromaDB
pip install --upgrade pip
pip install chromadb

# 4. Environment flags
{"export ANONYMIZED_TELEMETRY=False" if disable_telemetry else ""}"""
    elif "Linux" in target_os:
        script = f"""# 1. Update system & install C build tools
sudo apt-get update && sudo apt-get install -y build-essential python3-dev

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install ChromaDB with SQLite fallback override
pip install --upgrade pip
pip install chromadb pysqlite3-binary

# 4. Environment flags
{"export ANONYMIZED_TELEMETRY=False" if disable_telemetry else ""}"""
    else: # Docker
        script = f"""FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install ChromaDB
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir chromadb

{"ENV ANONYMIZED_TELEMETRY=False" if disable_telemetry else ""}

COPY . .
CMD ["python", "app.py"]"""

    st.code(script, language="docker" if "Docker" in target_os else "bash")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Comprehensive ChromaDB Healthcheck Suite", divider="rainbow")

st.markdown("""
The following complete Python diagnostic program verifies your ChromaDB installation, checks SQLite compatibility, 
inspects HNSWlib availability, validates embedding dependencies, and creates a test in-memory collection:
""")

code_healthcheck = '''import sys
import sqlite3
import time

def verify_chromadb_environment() -> dict:
    """Audits system prerequisites for running ChromaDB."""
    report = {}
    
    # 1. Python Version Check
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    report["python_version"] = f"{py_major}.{py_minor}.{sys.version_info.micro}"
    report["python_compatible"] = (py_major == 3 and 9 <= py_minor <= 12)
    
    # 2. SQLite Version Check
    sql_ver = sqlite3.sqlite_version
    report["sqlite_version"] = sql_ver
    sql_parts = [int(p) for p in sql_ver.split(".")[:2]]
    report["sqlite_compatible"] = (sql_parts[0] > 3) or (sql_parts[0] == 3 and sql_parts[1] >= 35)

    # 3. ChromaDB Package Check
    try:
        import chromadb
        report["chroma_installed"] = True
        report["chroma_version"] = chromadb.__version__
        
        # Test basic client creation
        client = chromadb.Client()
        col = client.create_collection(name=f"test_healthcheck_{int(time.time())}")
        col.add(documents=["Healthcheck test document"], ids=["t1"])
        res = col.query(query_texts=["Healthcheck"], n_results=1)
        client.delete_collection(col.name)
        
        report["smoke_test_passed"] = True
        report["smoke_test_result"] = res["ids"][0]
    except ImportError:
        report["chroma_installed"] = False
        report["chroma_version"] = None
        report["smoke_test_passed"] = False
        report["smoke_test_error"] = "chromadb is not installed in the active environment."
    except Exception as e:
        report["chroma_installed"] = True
        report["smoke_test_passed"] = False
        report["smoke_test_error"] = str(e)

    return report

# Run Healthcheck
results = verify_chromadb_environment()
print("ChromaDB Environment Health Report:")
for k, v in results.items():
    print(f"  * {k}: {v}")
'''

st.code(code_healthcheck, language="python")

if st.button("🚀 Execute ChromaDB Environment Healthcheck", type="primary"):
    exec_scope = {}
    exec(code_healthcheck, globals(), exec_scope)
    rep = exec_scope["results"]
    
    st.success("Environment audit executed successfully!")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Python Version:** `{rep['python_version']}` (Compatible: `{rep['python_compatible']}`)")
        st.markdown(f"**SQLite Version:** `{rep['sqlite_version']}` (Compatible: `{rep['sqlite_compatible']}`)")
    with c2:
        st.markdown(f"**ChromaDB Installed:** `{rep['chroma_installed']}` (Version: `{rep['chroma_version']}`)")
        st.markdown(f"**In-Memory Smoke Test Passed:** `{rep['smoke_test_passed']}`")

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why does ChromaDB fail on older Linux distributions with a SQLite version error, and how does `pysqlite3-binary` fix it?"""
a1 = """* **The Root Cause**:
  * Python's standard library `sqlite3` module links dynamically against the host operating system's system-level SQLite shared library (`libsqlite3.so`).
  * Older operating systems (such as CentOS 7 or Ubuntu 20.04) ship with SQLite 3.31 or older.
  * ChromaDB relies heavily on SQLite's `RETURNING` clause (introduced in SQLite 3.35.0) to perform atomic insert/upsert operations and retrieve generated IDs.
* **How `pysqlite3-binary` Solves It**:
  * `pysqlite3-binary` is a standalone, pre-compiled wheel that packages its own modern SQLite 3.40+ binary.
  * By monkey-patching `sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')` before importing `chromadb`, Python is fooled into using the bundled modern SQLite binary without requiring root access or upgrading the host OS."""

q2 = """2. Why does ChromaDB require C++ build tools during installation on some operating systems?"""
a2 = """* ChromaDB uses **HNSWlib** as its underlying vector index engine.
* HNSWlib is written entirely in C++ with hardware-specific SIMD intrinsics (AVX-512, SSE, ARM NEON) for maximum speed.
* If a pre-built binary Python wheel (`.whl`) is available on PyPI for your exact Python version and OS architecture, pip installs it instantly.
* However, on newer Python versions (e.g. Python 3.13 immediately after release) or uncommon architectures (e.g. Alpine Linux, ARMv7), pre-built wheels may not exist.
* In these cases, pip downloads the C++ source code (`.tar.gz`) and attempts to compile it locally, requiring `gcc` / `g++` or Visual C++ Build Tools."""

q3 = """3. How do you deploy ChromaDB in an air-gapped environment with zero internet access?"""
a3 = """* By default, ChromaDB's default embedding function attempts to download the `all-MiniLM-L6-v2` ONNX weights from the Hugging Face hub on first run.
* In an air-gapped corporate subnet, this triggers an HTTP connection timeout!
* **Air-Gapped Setup**:
  1. Pre-download the ONNX model files or Sentence-Transformers weights in a connected environment.
  2. Copy the files to a local directory inside the air-gapped machine.
  3. Initialize Chroma with a custom embedding function pointing to the local folder:
     `embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="/opt/models/all-MiniLM-L6-v2")`."""

q4 = """4. How do you disable anonymous telemetry in ChromaDB for enterprise compliance?"""
a4 = """* By default, ChromaDB reports anonymous usage statistics (OS, Python version, error counts) to PostHog.
* In enterprise and healthcare environments, this data transmission violates strict zero-egress compliance policies.
* **How to Disable**:
  * Via Environment Variable: `export ANONYMIZED_TELEMETRY=False`
  * Via Python Client Settings:
    ```python
    import chromadb
    from chromadb.config import Settings
    client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
    ```"""

q5 = """5. Why should production Docker containers avoid using Alpine Linux with ChromaDB?"""
a5 = """* Alpine Linux uses `musl libc` instead of `glibc`.
* Most pre-compiled Python binary wheels for complex scientific and C++ packages (`numpy`, `onnxruntime`, `hnswlib`) are built for `glibc` (manylinux wheels).
* On Alpine, pip fails to find compatible wheels and attempts to compile everything from C/C++ source code, resulting in excruciatingly slow build times (15+ minutes) or compiler errors.
* **Best Practice**: Use `python:3.11-slim` (Debian-based) instead of `python:3.11-alpine` for predictable binary wheel compatibility."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Installation Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The HuggingFace Cache Disk Exhaustion Trap**  
> Running batch unit tests or ephemeral containers that initialize Chroma without specifying a persistent 
> cache path can repeatedly download 100MB model weights into `~/.cache/chroma/onnx_models`, filling up 
> ephemeral container disk space. Mount a shared cache volume or pass pre-computed vectors.

> [!WARNING]
> **Trap 2: Placing the `pysqlite3` Monkey-Patch Too Late**  
> If you import *any* package that imports `sqlite3` before executing the monkey-patch (e.g. `import langchain; import chromadb`), 
> Python will have already loaded the system's old SQLite library into memory! The monkey-patch MUST be placed on 
> line 1 of your application's entrypoint script.

> [!WARNING]
> **Trap 3: Mixing Python 32-bit and 64-bit on Windows**  
> Installing a 32-bit Python interpreter on 64-bit Windows causes binary wheel resolution failures for `onnxruntime` 
> and `chromadb`, forcing pip to attempt compilation from source and failing with cryptic compiler errors. 
> Always ensure you are running 64-bit Python (`python -c "import platform; print(platform.architecture())"`).
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What SQLite Version is Required?"):
        st.markdown("""
        **SQLite Prerequisite**:  
        ChromaDB strictly requires SQLite >= 3.35.0 for atomic `RETURNING` clause support.
        """)
    with st.expander("🗂️ Flashcard: How to Fix Old SQLite on Linux?"):
        st.markdown("""
        **`pysqlite3-binary` Fix**:  
        Install `pysqlite3-binary` and monkey-patch `sys.modules['sqlite3']` at the top of your script.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How to Disable Telemetry?"):
        st.markdown("""
        **Disable Telemetry**:  
        Set the environment variable `ANONYMIZED_TELEMETRY=False` or pass `Settings(anonymized_telemetry=False)`.
        """)
    with st.expander("🗂️ Flashcard: Which Docker Base Image is Best?"):
        st.markdown("""
        **Docker Base**:  
        Use `python:3.11-slim` (glibc-based) rather than Alpine to avoid building C++ HNSWlib wheels from source.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I know how to install ChromaDB in an isolated Python virtual environment.", value=True)
chk2 = st.checkbox("I understand the SQLite >= 3.35.0 requirement and why older Linux distros fail.")
chk3 = st.checkbox("I can implement the `pysqlite3-binary` monkey-patch to resolve SQLite errors.")
chk4 = st.checkbox("I know how to configure ChromaDB for air-gapped environments without internet access.")
chk5 = st.checkbox("I know how to disable anonymous telemetry for enterprise compliance.")
chk6 = st.checkbox("I can execute an automated environment verification script in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered ChromaDB Installation and are ready for Chapter 6.3: Chroma Client!")
