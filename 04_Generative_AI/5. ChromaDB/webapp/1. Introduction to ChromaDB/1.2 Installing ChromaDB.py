import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="1.2 Installing ChromaDB - ChromaDB Curriculum",
    page_icon="🎨",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulation
class EnvironmentAuditor:
    """Audits system prerequisites for ChromaDB runtime compatibility."""
    def __init__(self):
        self.diagnostics = [
            {"component": "Python Runtime", "required": ">= 3.9, < 3.13", "detected": "3.10.12", "status": "Compatible"},
            {"component": "SQLite Engine", "required": ">= 3.35.0", "detected": "3.40.1", "status": "Compatible"},
            {"component": "C++ Compiler (hnswlib)", "required": "MSVC / GCC / Clang", "detected": "Prebuilt Wheel", "status": "Installed"},
            {"component": "ONNX Runtime", "required": ">= 1.14.0", "detected": "1.16.3", "status": "Ready"},
            {"component": "Telemetry Setting", "required": "ANONYMIZED_TELEMETRY", "detected": "Configurable", "status": "Valid"}
        ]

    def audit(self) -> List[Dict[str, Any]]:
        return self.diagnostics

# Title & Overview
st.title("1.2 Installing ChromaDB: Environment Setup, Dependencies & Verification")
st.caption("Chapter 1: Introduction to ChromaDB | Domain: 5. ChromaDB")
st.markdown("""
Installing ChromaDB is straightforward in modern Python environments, but production deployments require 
understanding its underlying system dependencies—most notably **SQLite version requirements (>= 3.35.0)**, 
**C++ binary extensions (`hnswlib`)**, and **ONNX Runtime bindings**.

Whether deploying locally on macOS/Windows or containerizing across Linux server distributions (Debian, Alpine, RHEL), 
proper virtual environment isolation ensures conflict-free operation across LangChain, LlamaIndex, and OpenAI client libraries.

In this lesson, you will master:
1. **The Installation Lifecycle**: Installing ChromaDB via pip, uv, and poetry across isolated virtual environments.
2. **The SQLite 3.35+ Constraint**: Diagnosing and overcoming legacy Linux SQLite version limits using `pysqlite3-binary`.
3. **Installation Verification**: Writing diagnostic verification scripts to validate in-memory and persistent clients.
4. **Importing & Telemetry Configuration**: Disabling anonymized telemetry and configuring environment settings cleanly.
5. **Production Installation Traps**: Avoiding missing C++ build tools, Alpine Linux musl incompatibilities, and dependency collisions.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: ChromaDB Environment & Runtime Dependencies", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChromaInstallBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_os {
            label="Tier 1: Host Operating System & Native Binaries";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            host_env [label="Operating System Runtime\\nLinux (glibc) / macOS / Windows", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            native_sqlite [label="Native SQLite Dynamic Library\\nMust satisfy version >= 3.35.0", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            cpp_build [label="C++ Build Tools & Header Cache\\nPre-compiled PyPI Wheels or g++ / clang", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            host_env -> native_sqlite;
            host_env -> cpp_build;
        }

        subgraph cluster_venv {
            label="Tier 2: Isolated Python Virtual Environment (venv / conda)";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            pip_installer [label="Package Manager (pip / uv / poetry)\\npip install chromadb", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            hnsw_lib [label="hnswlib C++ Binding\\nHigh-speed HNSW ANN graph engine", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            onnx_run [label="ONNX Runtime Engine\\nRuns all-MiniLM-L6-v2 embedding model", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            pydantic_core [label="Pydantic v2 Core\\nSchema validation and client serialization", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            pip_installer -> hnsw_lib;
            pip_installer -> onnx_run;
            pip_installer -> pydantic_core;
            native_sqlite -> pip_installer;
            cpp_build -> hnsw_lib;
        }

        subgraph cluster_app {
            label="Tier 3: Chroma Client Runtime & Telemetry Layer";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            import_chroma [label="Application Import: import chromadb\\nReads environment variables & settings", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            telemetry [label="Telemetry Filter\\nANONYMIZED_TELEMETRY=False", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            ready_client [label="Initialized Client Ready\\nPersistentClient / EphemeralClient", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            hnsw_lib -> import_chroma;
            onnx_run -> import_chroma;
            import_chroma -> telemetry -> ready_client;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Installing with Pip
st.header("2. Installing with Pip", divider="rainbow")
st.markdown("""
The standard method to install ChromaDB in any modern Python environment is via `pip`:

```bash
# Standard installation
pip install chromadb

# Fast installation using Astral uv
uv pip install chromadb

# Installing with optional full dependencies
pip install "chromadb[all]"
```

* **Core Dependencies Installed Automatically**:
  * `hnswlib`: Fast approximate nearest neighbor search library implemented in C++ with Python bindings.
  * `onnxruntime`: High-performance inference engine running the default local embedding model.
  * `pydantic`: Schema validation and type enforcement for collection configs and query filters.
  * `tokenizers`: Hugging Face Fast Tokenizers for text chunk preprocessing.
""")
st.divider()

# Section 3: Installing in a Virtual Environment
st.header("3. Installing in a Virtual Environment", divider="rainbow")
st.markdown("""
Always isolate ChromaDB within a dedicated virtual environment to prevent dependency collisions across packages 
like `pydantic`, `torch`, or `transformers`:
""")

col_env1, col_env2 = col_env1, col_env2 = st.columns(2)
with col_env1:
    st.markdown("""
    ### Option A: Standard Python `venv`
    ```bash
    # Create virtual environment
    python -m venv .venv

    # Activate on Linux / macOS
    source .venv/bin/activate

    # Activate on Windows (PowerShell)
    .venv\\Scripts\\Activate.ps1

    # Upgrade pip and install ChromaDB
    pip install --upgrade pip
    pip install chromadb
    ```
    """)
with col_env2:
    st.markdown("""
    ### Option B: Conda / Mamba Environment
    ```bash
    # Create conda environment with Python 3.11
    conda create -n rag_env python=3.11 -y

    # Activate environment
    conda activate rag_env

    # Install ChromaDB via conda-forge or pip
    pip install chromadb
    ```
    """)
st.divider()

# Section 4: Verifying Installation
st.header("4. Verifying ChromaDB Installation & SQLite Version Constraints", divider="rainbow")
st.markdown("""
ChromaDB requires **SQLite version 3.35.0 or greater** because its metadata query planner relies on modern 
SQL syntax (such as `RETURNING` clauses and mathematical functions).

Some legacy Linux distributions (such as CentOS 7, Amazon Linux 2, or older Ubuntu LTS releases) ship with 
ancient SQLite versions (e.g., 3.7 or 3.28), triggering the infamous error:
`RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.`
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
To resolve SQLite version errors in legacy environments without recompiling the operating system kernel, 
install `pysqlite3-binary` and monkey-patch Python's standard `sqlite3` module at the very top of your entrypoint script:
""")

st.code('''
# Insert at the very top of your application script before importing chromadb:
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now chromadb uses the modern bundled SQLite engine seamlessly
import chromadb
''', language="python")
st.divider()

# Section 5: Importing ChromaDB
st.header("5. Importing ChromaDB & Telemetry Configuration", divider="rainbow")
st.markdown("""
By default, ChromaDB gathers anonymized usage telemetry to help maintainers track popular features and crash patterns. 
In enterprise environments with strict data privacy guidelines, telemetry should be explicitly disabled:
""")

st.code('''
# Disable anonymized usage telemetry via Settings
from chromadb.config import Settings
import chromadb

# Initialize client with programmatic settings
client = chromadb.PersistentClient(
    path="./enterprise_chroma",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)
''', language="python")
st.divider()

# Section 6: Production Implementation
st.header("6. Production Implementation: Automated Environment Diagnostics & Bootstrap Script", divider="rainbow")
st.markdown("""
Below is an automated Python verification script that audits runtime compatibility, verifies SQLite versions, 
and executes a live smoke test against an in-memory Chroma collection.
""")

st.code('''
import sys
import sqlite3

def run_chroma_diagnostics() -> bool:
    """Verifies that the Python environment satisfies all ChromaDB prerequisites."""
    print("--- ChromaDB Environment Verification ---")
    
    # 1. Check Python version
    py_ver = sys.version_info
    print(f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 8) or py_ver >= (3, 13):
        print("WARNING: ChromaDB is recommended for Python 3.9 through 3.12.")

    # 2. Check SQLite version
    sqlite_ver = sqlite3.sqlite_version
    print(f"Detected SQLite Version: {sqlite_ver}")
    sqlite_tuple = tuple(map(int, sqlite_ver.split(".")[:3]))
    if sqlite_tuple < (3, 35, 0):
        print("ERROR: SQLite version must be >= 3.35.0! Install pysqlite3-binary.")
        return False

    # 3. Import and execute smoke test
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("smoke_test")
        collection.add(
            documents=["Hello ChromaDB verification test"],
            ids=["test_01"]
        )
        res = collection.query(query_texts=["verification"], n_results=1)
        assert len(res["ids"][0]) == 1
        print("SUCCESS: ChromaDB installed and operational!")
        return True
    except Exception as e:
        print(f"FAILURE during ChromaDB execution: {e}")
        return False

if __name__ == "__main__":
    run_chroma_diagnostics()
''', language="python")
st.divider()

# Section 7: Interactive Workbench
st.header("7. Interactive Workbench: Live ChromaDB Environment & Version Auditor", divider="rainbow")
st.markdown("""
Inspect simulated host environment diagnostics to determine whether your deployment target satisfies ChromaDB requirements.
""")

auditor = EnvironmentAuditor()
diag_data = auditor.audit()

st.subheader("Runtime Diagnostic Report")
df_diag = pd.DataFrame(diag_data)
st.dataframe(df_diag, use_container_width=True)

st.success("All primary runtime components satisfy ChromaDB production operational constraints.")
st.divider()

# Section 8: Technical Interview Preparation & Production Traps
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: Why does ChromaDB mandate SQLite version 3.35.0 or higher?", expanded=True):
    st.markdown("""
    **Answer:**
    SQLite introduced significant enhancements in version 3.35.0 (released March 2021), including native mathematical functions 
    (`sqrt`, `pow`, `cos`, `sin`), `RETURNING` syntax on `INSERT/UPDATE/DELETE` operations, and optimized inverted indexing. 
    ChromaDB's metadata persistence engine relies directly on these modern SQL constructs to maintain transactional consistency 
    and fast metadata filtering.
    """)

with st.expander("Q2: How do you fix SQLite version errors on older Linux Docker containers without upgrading the base OS?"):
    st.markdown("""
    **Answer:**
    1. Install `pysqlite3-binary` via pip: `pip install pysqlite3-binary`.
    2. In the Python application entrypoint, override `sys.modules['sqlite3']` with `pysqlite3` prior to importing ChromaDB:
       ```python
       __import__('pysqlite3')
       import sys
       sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
       import chromadb
       ```
    This dynamically binds Python to a modern, self-contained SQLite shared library bundled with the wheel.
    """)

with st.expander("Q3: What role does `hnswlib` play during ChromaDB installation?"):
    st.markdown("""
    **Answer:**
    `hnswlib` is a fast, highly optimized C++ library with Python bindings that implements the Hierarchical Navigable Small World (HNSW) 
    approximate nearest neighbor algorithm. ChromaDB relies on `hnswlib` as its primary vector indexing engine. When wheels are available, 
    it installs pre-compiled binaries; otherwise, it requires a local C++11 compliant compiler (`gcc`, `clang`, or MSVC) to build from source.
    """)

with st.expander("Q4: How do you disable ChromaDB telemetry in air-gapped or HIPAA/GDPR-compliant production environments?"):
    st.markdown("""
    **Answer:**
    Telemetry can be disabled via environment variables or programmatic settings:
    * Environment variable: `export ANONYMIZED_TELEMETRY=False`.
    * Code configuration:
      ```python
      from chromadb.config import Settings
      client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
      ```
    """)

with st.expander("Q5: Why does installing ChromaDB on Alpine Linux Docker images frequently fail?"):
    st.markdown("""
    **Answer:**
    Alpine Linux uses `musl libc` rather than standard `glibc`. PyPI binary wheels for `hnswlib` and `onnxruntime` are compiled 
    against `glibc` (manylinux wheels). Installing on Alpine forces pip to attempt building from source, which frequently fails due 
    to missing C++ build chains, Fortran libraries, or incompatibilities with musl. The production best practice is to use Debian-based 
    slim images (`python:3.11-slim`) instead of Alpine.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Building on Alpine Linux Without glibc (Fatal Wheel Build Error)**
    Using `python:3.11-alpine` as a Docker base image. Pip fails to find pre-compiled wheels for `hnswlib` and `onnxruntime`, attempting a source compilation that aborts with cryptic C++ compiler errors. Always use `python:3.11-slim` for ChromaDB containerization.
    """)
    st.error("""
    **Trap 2: System SQLite Version Incompatibility on Enterprise RHEL / CentOS (Runtime Crash)**
    Deploying on legacy enterprise Linux servers with SQLite < 3.35.0. Application crashes immediately upon `import chromadb`. Resolve using the `pysqlite3-binary` module monkey-patch.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Dependency Version Conflicts with Pydantic v1 vs v2 (Import Errors)**
    ChromaDB versions >= 0.4.x utilize Pydantic v2. Integrating ChromaDB into legacy codebases pinned to Pydantic v1 can trigger conflicting schema validation errors. Use `pydantic.v1` compatibility namespaces where necessary.
    """)
    st.warning("""
    **Trap 4: Missing Disk Write Permissions in Containerized PersistentClient (IO Error)**
    Configuring `chromadb.PersistentClient(path="/data/chroma")` in a Docker container running under a non-root user that lacks write permissions to `/data`. The client fails when attempting to create `chroma.sqlite3`.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What minimum SQLite version does ChromaDB require?")
        st.markdown("**Back**: SQLite 3.35.0 or greater, required for modern mathematical functions and RETURNING SQL syntax.")
    with st.container(border=True):
        st.markdown("**Front**: How do you bypass SQLite version errors on older Linux?")
        st.markdown("**Back**: Install `pysqlite3-binary` and swap `sys.modules['sqlite3']` before importing `chromadb`.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why avoid Alpine Linux for ChromaDB deployments?")
        st.markdown("**Back**: Pre-compiled binary wheels for `hnswlib` and `onnxruntime` require glibc; use `python:slim` instead.")
    with st.container(border=True):
        st.markdown("**Front**: How do you disable usage telemetry in ChromaDB?")
        st.markdown("**Back**: Set the environment variable `ANONYMIZED_TELEMETRY=False` or pass `Settings(anonymized_telemetry=False)`.")

st.divider()

# Section 9: Self-Assessment Checklist
st.header("9. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of ChromaDB Installation and Environment Configuration before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I know how to install ChromaDB using pip, uv, and isolated virtual environments.", value=True, key="chk_inst_1")
    chk2 = st.checkbox("I understand the SQLite 3.35+ requirement and how to resolve legacy version conflicts using pysqlite3-binary.", value=True, key="chk_inst_2")
    chk3 = st.checkbox("I can write a diagnostic script to verify ChromaDB installation and in-memory execution.", value=True, key="chk_inst_3")
with c2:
    chk4 = st.checkbox("I understand why Debian-slim is preferred over Alpine Linux for containerizing ChromaDB.", value=True, key="chk_inst_4")
    chk5 = st.checkbox("I know how to disable telemetry in enterprise and privacy-compliant environments.", value=True, key="chk_inst_5")
    chk6 = st.checkbox("I can diagnose production traps including missing C++ build tools, permissions issues, and Pydantic conflicts.", value=True, key="chk_inst_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **ChromaDB Installation Mastery Demonstrated!** You are ready to advance to Lesson 1.3 in the curriculum.")
