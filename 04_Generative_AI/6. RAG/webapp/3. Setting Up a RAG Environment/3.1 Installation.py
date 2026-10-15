import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="3.1 Installation - RAG Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("3.1 Installation: Python Runtimes, Isolation, & Core RAG Toolchains")
st.caption("Chapter 3: Setting Up a RAG Environment | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Setting up a resilient, conflict-free Python environment is the operational foundation of production RAG engineering. 
Because generative AI frameworks evolve rapidly—with frequent breaking updates across orchestration libraries, 
provider SDKs, and vector database drivers—relying on global system runtimes leads directly to dependency collisions.

Furthermore, enterprise document ingestion pipelines interface with native system libraries (Poppler for PDF rendering, 
Tesseract for optical character recognition, and C++ extensions for accelerated vector indexing). 

In this lesson, you will master:
1. Python runtime selection, C-extension compatibility, and virtual environment isolation (`venv`, `uv`).
2. Dependency locking and preventing cascading version conflicts across generative AI frameworks.
3. Installing and decoupling modular orchestration layers (`langchain-core` vs monolithic packages).
4. Local and cloud vector database drivers (ChromaDB, Qdrant, FAISS).
5. Installing high-performance embedding frameworks and multimodal document parsers.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class PackageDependencyChecker:
    def __init__(self):
        self.package_catalog = {
            "langchain-core": {"category": "Orchestration", "min_ver": "0.3.0", "notes": "Base abstractions, runnables, LCEL"},
            "langchain-community": {"category": "Orchestration", "min_ver": "0.3.0", "notes": "Third-party community integrations"},
            "chromadb": {"category": "Vector DB", "min_ver": "0.5.0", "notes": "Embedded & client-server vector store"},
            "qdrant-client": {"category": "Vector DB", "min_ver": "1.10.0", "notes": "High-performance Rust-backed vector engine"},
            "sentence-transformers": {"category": "Embeddings", "min_ver": "3.0.0", "notes": "Hugging Face local dense embeddings"},
            "pypdf": {"category": "Document Parsing", "min_ver": "4.0.0", "notes": "Pure-Python PDF text extraction"},
            "pymupdf": {"category": "Document Parsing", "min_ver": "1.24.0", "notes": "High-speed C-backed PDF parser (MuPDF)"}
        }

    def simulate_compatibility_audit(self, selected_packages: List[str], python_version: str) -> Dict[str, Any]:
        results = []
        is_compatible = True
        warnings = []

        if python_version == "Python 3.13":
            warnings.append("Python 3.13 is experimental for PyTorch and certain compiled C-extensions (FAISS). Python 3.11 or 3.12 is recommended.")
            is_compatible = False
        elif python_version == "Python 3.9":
            warnings.append("Python 3.9 is nearing end-of-life; modern LangChain releases require Python 3.10+.")
            is_compatible = False

        for pkg in selected_packages:
            meta = self.package_catalog.get(pkg, {})
            results.append({
                "Package": pkg,
                "Category": meta.get("category", "General"),
                "Required Version": f">= {meta.get('min_ver', '1.0.0')}",
                "Status": "COMPATIBLE" if is_compatible else "POTENTIAL_CONFLICT",
                "Description": meta.get("notes", "")
            })

        return {
            "python_version": python_version,
            "overall_status": "READY" if is_compatible and len(warnings) == 0 else "WARNINGS_DETECTED",
            "warnings": warnings,
            "audit_table": results
        }

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The Enterprise RAG Toolchain & Environment Matrix", divider="rainbow")
st.markdown("""
A production RAG environment is organized into four distinct modular layers, separating base runtimes from 
parsers, embedding frameworks, vector indices, and synthesis clients:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_toolchain = """
    digraph RAGToolchain {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_layer1 {
            label="Layer 1: Base Runtime & Virtual Isolation Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            py [label="Python Runtime (3.11 or 3.12 LTS)\\nStrictly 64-bit with C++ Build Tools", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            env [label="Virtual Environment Manager\\nuv / venv / Poetry (Hermetic Project Isolation)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            py -> env;
        }

        subgraph cluster_layer2 {
            label="Layer 2: Ingestion & Parsing Toolchain";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            parsers [label="Document Parsers: PyMuPDF / PyPDF / Unstructured\\nExtracts clean text, structured tables, and page metadata", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            splitters [label="Text Chunkers: langchain-text-splitters\\nRecursive character & semantic boundary splitters", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            env -> parsers -> splitters;
        }

        subgraph cluster_layer3 {
            label="Layer 3: Embeddings & Vector Database Drivers";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            embeds [label="Embedding Engine: sentence-transformers / PyTorch\\nLocal CPU/GPU inference for dense semantic vectors", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            vstores [label="Vector Stores: chromadb / qdrant-client / pinecone\\nHNSW graph index storage with metadata filtering", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            splitters -> embeds -> vstores;
        }

        subgraph cluster_layer4 {
            label="Layer 4: Orchestration & Generative Synthesis";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            orch [label="Modular Orchestrator: langchain-core / llama-index\\nAssembles LCEL runnable pipelines with type safety", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            clients [label="Provider SDKs: openai / anthropic / google-genai\\nCommunicates with frontier LLM synthesis endpoints", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            vstores -> orch -> clients;
        }
    }
    """
    st.graphviz_chart(dot_toolchain, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In early tutorials, developers installed monolithic metapackages like `pip install langchain`, which pulled over 
400 transitive dependencies, bloated container images to 8GB, and caused frequent dependency conflicts. 
Modern production stacks strictly install **decoupled modular packages** (`langchain-core`, `langchain-community`, 
`langchain-openai`) using modern package managers like **`uv`** (written in Rust), which resolves and installs 
isolated environments in seconds rather than minutes.
""")

st.divider()

# ==============================================================================
# 2. Installing Python
# ==============================================================================
st.header("2. Installing Python & Runtime Architecture", divider="rainbow")
st.markdown("""
Choosing the correct Python runtime is the first critical architectural decision in RAG engineering:
* **Recommended Versions**: **Python 3.11** or **Python 3.12**. These provide significant performance optimizations while retaining mature wheel support for compiled C-extensions (FAISS, PyTorch, PyMuPDF).
* **Avoid Python 3.13 (Initial Releases)**: Many scientific computing and vector index libraries lack precompiled binary wheels for early minor versions of Python 3.13, requiring local C++ compilation.
* **Architecture Check**: Ensure a 64-bit installation (`x86_64` or `arm64`) because high-dimensional vector search libraries fail on 32-bit systems.
""")

st.code("""
# Verify Python version and architecture
python --version
python -c "import platform; print(platform.architecture()[0])"
""", language="bash")

st.divider()

# ==============================================================================
# 3. Creating Virtual Environments
# ==============================================================================
st.header("3. Creating Virtual Environments (`venv`, `uv`, & `conda`)", divider="rainbow")
st.markdown("""
A virtual environment creates an isolated directory containing a private Python binary and site-packages folder. 
Never install RAG packages into the global operating system Python environment.
""")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.subheader("Standard Library: `venv`")
    st.markdown("""
    * Built into every Python distribution with zero external tooling.
    * Completely portable and supported by all IDEs.
    """)
    st.code("""
    # Create virtual environment
    python -m venv .venv

    # Activate on Linux/macOS
    source .venv/bin/activate

    # Activate on Windows (PowerShell)
    .venv\\Scripts\\Activate.ps1
    """, language="bash")

with col_v2:
    st.subheader("Modern Standard: `uv` (Blazing Fast)")
    st.markdown("""
    * Written in Rust; up to 10-100x faster package installation than standard pip.
    * Automatically manages Python versions and hermetic lockfiles.
    """)
    st.code("""
    # Install uv standalone
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create virtual environment with specific Python
    uv venv --python 3.11 .venv

    # Install packages with sub-second caching
    uv pip install langchain-core chromadb
    """, language="bash")

st.divider()

# ==============================================================================
# 4. Installing RAG Libraries
# ==============================================================================
st.header("4. Installing Core RAG Libraries & Dependency Locking", divider="rainbow")
st.markdown("""
A production RAG application requires pinning dependencies using a lockfile to prevent upstream breaking changes 
from disrupting automated CI/CD deployments:
""")

df_rag_libs = pd.DataFrame([
    {"Layer": "Orchestration", "Recommended Package": "langchain-core, llama-index-core", "Purpose": "Execution graph management, prompt composition, output parsing."},
    {"Layer": "Vector Stores", "Recommended Package": "chromadb, qdrant-client", "Purpose": "Local and distributed vector storage and HNSW similarity search."},
    {"Layer": "Embeddings", "Recommended Package": "sentence-transformers, torch", "Purpose": "Local dense vector representations and cross-encoder reranking."},
    {"Layer": "Parsers", "Recommended Package": "pypdf, pymupdf, python-docx", "Purpose": "Document extraction from unstructured enterprise file formats."}
])
st.dataframe(df_rag_libs, use_container_width=True)

st.divider()

# ==============================================================================
# 5. Installing LangChain
# ==============================================================================
st.header("5. Installing LangChain Modular Packages", divider="rainbow")
st.markdown("""
LangChain is decoupled into independent, lightweight packages. Never install the monolithic legacy package:
""")

with st.expander("🔍 View LangChain Modular Architecture Diagram", expanded=True):
    dot_lc = """
    digraph LangChainModularity {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        core [label="langchain-core\\n(Zero heavy dependencies; base Runnables & LCEL)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        comm [label="langchain-community\\n(Community retrievers, vector stores, and loaders)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        prov [label="Partner Packages: langchain-openai / langchain-anthropic\\n(Lightweight, officially maintained provider wrappers)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        text [label="langchain-text-splitters\\n(Pure chunking logic: recursive, character, markdown)", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        core -> comm;
        core -> prov;
        core -> text;
    }
    """
    st.graphviz_chart(dot_lc, use_container_width=True)

st.code("""
# Install the modular production suite
pip install langchain-core langchain-community langchain-text-splitters langchain-openai
""", language="bash")

st.divider()

# ==============================================================================
# 6. Installing Vector Databases
# ==============================================================================
st.header("6. Installing Vector Database Drivers", divider="rainbow")
st.markdown("""
Depending on your enterprise architecture, choose between local embedded stores and distributed cloud clients:
* **ChromaDB**: The ideal local embedded store for prototyping and edge devices (`pip install chromadb`).
* **Qdrant**: High-performance vector engine with native payload filtering and client-server scalability (`pip install qdrant-client`).
* **FAISS**: Facebook AI Similarity Search for dense CPU/GPU clustering (`pip install faiss-cpu`).
""")

st.code("""
# Install vector store drivers
pip install chromadb qdrant-client faiss-cpu
""", language="bash")

st.divider()

# ==============================================================================
# 7. Installing Embedding Libraries
# ==============================================================================
st.header("7. Installing Embedding Libraries", divider="rainbow")
st.markdown("""
For generating dense semantic vectors locally without recurring API costs, install Hugging Face's `sentence-transformers` 
and PyTorch:
""")

st.code("""
# Install PyTorch (CPU version) and sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
""", language="bash")

st.divider()

# ==============================================================================
# 8. Installing Document-Processing Libraries
# ==============================================================================
st.header("8. Installing Document-Processing Libraries", divider="rainbow")
st.markdown("""
Ingestion pipelines must parse varied document formats with high speed and fidelity:
* **PyPDF**: Pure Python PDF parser with zero native C-dependencies (`pip install pypdf`).
* **PyMuPDF (fitz)**: Ultra-fast C-backed PDF engine capable of extracting tables, text blocks, and vector coordinates (`pip install pymupdf`).
* **python-docx & openpyxl**: Word and Excel document extractors (`pip install python-docx openpyxl`).
""")

st.code("""
# Install document extraction suite
pip install pypdf pymupdf python-docx openpyxl
""", language="bash")

st.divider()

# ==============================================================================
# 9. Production Implementation
# ==============================================================================
st.header("9. Production Implementation: Automated Environment Validator & Dependency Bootstrapper", divider="rainbow")
st.markdown("""
Below is a runnable Python production script that audits the installed Python runtime, validates import readiness 
for core RAG libraries, and flags compatibility issues:
""")

prod_env_code = """
import sys
import platform
from typing import Dict, List, Any

class RAGEnvironmentAuditor:
    def __init__(self):
        self.required_libraries = [
            "streamlit",
            "pandas",
            "numpy",
            "graphviz"
        ]

    def audit_runtime(self) -> Dict[str, Any]:
        version_info = sys.version_info
        is_supported = (version_info.major == 3 and 10 <= version_info.minor <= 12)
        arch = platform.architecture()[0]
        
        return {
            "python_version": f"{version_info.major}.{version_info.minor}.{version_info.micro}",
            "is_supported_lts": is_supported,
            "architecture": arch,
            "os_platform": platform.system()
        }

    def verify_library_imports(self) -> Dict[str, bool]:
        status = {}
        for lib in self.required_libraries:
            try:
                __import__(lib)
                status[lib] = True
            except ImportError:
                status[lib] = False
        return status

    def run_full_diagnostic(self) -> Dict[str, Any]:
        runtime = self.audit_runtime()
        imports = self.verify_library_imports()
        all_passed = runtime["is_supported_lts"] and all(imports.values())
        return {
            "system_ready_for_rag": all_passed,
            "runtime_info": runtime,
            "import_audit": imports
        }

if __name__ == "__main__":
    auditor = RAGEnvironmentAuditor()
    diagnostic = auditor.run_full_diagnostic()
    print("Environment Diagnostic Result:", diagnostic)
"""
st.code(prod_env_code, language="python")

st.divider()

# ==============================================================================
# 10. Interactive Workbench
# ==============================================================================
st.header("10. Interactive Workbench: Environment Diagnostic & Compatibility Studio", divider="rainbow")
st.markdown("""
Simulate an enterprise environment audit by selecting your target Python runtime and proposed package suite:
""")

checker = PackageDependencyChecker()

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    py_ver_choice = st.selectbox(
        "Target Python Runtime:",
        ["Python 3.11", "Python 3.12", "Python 3.13", "Python 3.9"]
    )
with col_cfg2:
    selected_pkgs = st.multiselect(
        "Select Planned RAG Packages:",
        list(checker.package_catalog.keys()),
        default=["langchain-core", "chromadb", "sentence-transformers", "pypdf"]
    )

if st.button("Run Environment Compatibility Audit", type="primary"):
    audit_res = checker.simulate_compatibility_audit(selected_pkgs, py_ver_choice)
    
    if audit_res["overall_status"] == "READY":
        st.success(f"✅ Runtime Compatibility Verified for {py_ver_choice}! All packages are validated.")
    else:
        st.warning(f"⚠️ Compatibility Warnings Detected for {py_ver_choice}:")
        for w in audit_res["warnings"]:
            st.markdown(f"- ⚠️ {w}")

    st.subheader("Package Architecture Audit Table")
    st.dataframe(pd.DataFrame(audit_res["audit_table"]), use_container_width=True)

st.divider()

# ==============================================================================
# 11. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why should you avoid installing the monolithic 'langchain' package in production?"):
    st.markdown("""
    **Answer**: The monolithic `langchain` package pulls hundreds of unneeded transitive dependencies (Azure, AWS, GCP, dozens of vector stores), 
    inflating container images to 8+ GB and creating high dependency vulnerability (CVE) exposure. Production architectures install only 
    `langchain-core` and the specific partner packages required (`langchain-openai`, `langchain-community`).
    """)

with st.expander("Q2: What is the architectural advantage of using 'uv' over standard 'pip' in enterprise pipelines?"):
    st.markdown("""
    **Answer**: `uv` is written in Rust and utilizes global content-addressable package caching, parallel wheel building, and 
    fast dependency resolution. It reduces CI/CD container build times from 3 minutes down to under 5 seconds while enforcing 
    deterministic universal lockfiles across platforms.
    """)

with st.expander("Q3: Why are precompiled binary wheels critical when deploying Vector DB drivers like FAISS or ChromaDB?"):
    st.markdown("""
    **Answer**: Vector database libraries rely on compiled C++ and SIMD instructions (AVX2, AVX-512) for accelerated distance 
    computations. If precompiled wheels are missing for your Python version, `pip` falls back to compiling from source, requiring 
    a local C++ toolchain and often failing inside lightweight production Docker containers.
    """)

with st.expander("Q4: When should you select PyMuPDF over PyPDF for enterprise document parsing?"):
    st.markdown("""
    **Answer**: While `pypdf` is pure Python and easy to install, `pymupdf` (built on MuPDF in C) is 10-20x faster on massive 
    multi-page documents and provides superior layout parsing, extracting bounding boxes, tables, and images with higher fidelity.
    """)

with st.expander("Q5: How do you prevent dependency drift across development, staging, and production in RAG applications?"):
    st.markdown("""
    **Answer**: Use hermetic dependency locking (`uv.lock`, `poetry.lock`, or `pip-tools` compile) combined with containerization. 
    Never use loose version bounds like `langchain>=0.1` in production requirements files without an accompanying frozen lockfile.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Global Site-Packages Trap)**: Installing generative AI packages directly into the global operating system Python environment, leading to system-wide dependency collisions when different projects require conflicting library versions.")
    st.warning("**Trap 2 (Bleeding-Edge Python Runtime Trap)**: Adopting a brand new Python minor version (e.g. 3.13 on launch week) before major vector search and scientific computing packages have published precompiled wheels.")
with col_trap2:
    st.error("**Trap 3 (Monolithic Image Bloat)**: Installing full unneeded orchestration suites into Docker containers, producing multi-gigabyte container images that slow down autoscaling and cluster deployments.")
    st.warning("**Trap 4 (Missing Native System Dependencies)**: Expecting document parsers like OCR or layout analyzers to work in Docker without installing underlying system shared libraries (e.g., Poppler, libmagic).")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of `langchain-core`?")
        st.caption("Flip for answer:")
        st.info("**Back**: Provides the base abstractions, interfaces, and LangChain Expression Language (LCEL) runnables with zero heavy third-party dependencies.")
    with st.container(border=True):
        st.markdown("**Front**: Why is `uv` preferred in modern Python engineering?")
        st.caption("Flip for answer:")
        st.info("**Back**: A blazing-fast Rust-based package installer and resolver that cuts environment setup times by 10-100x compared to legacy pip.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Which Python versions are recommended for production RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: Python 3.11 or 3.12, providing high execution performance with mature precompiled binary wheels for PyTorch and vector stores.")
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between PyPDF and PyMuPDF?")
        st.caption("Flip for answer:")
        st.info("**Back**: PyPDF is pure Python (simple, zero native C-deps). PyMuPDF is C-based (10-20x faster, superior layout and table extraction).")

st.divider()

# ==============================================================================
# 12. Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I know which Python runtimes (3.11/3.12) offer the best stability for RAG libraries.", value=True)
c2 = st.checkbox("I can isolate project environments using standard `venv` or modern `uv`.", value=True)
c3 = st.checkbox("I understand why installing modular `langchain-core` is superior to monolithic metapackages.", value=True)
c4 = st.checkbox("I can select and install appropriate vector database drivers for local and cloud environments.", value=True)
c5 = st.checkbox("I know how to install embedding frameworks and document processing suites.", value=True)
c6 = st.checkbox("I understand how to build automated environment validation scripts to prevent deployment drift.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 3.1 Complete! You are ready to advance to Lesson 3.2: API Configuration.")
